"""Anchored residual space-time VAE ([ARCH §4.4]).

The other VAEs reconstruct the absolute clip ``x``. Under KL pressure the
latent then collapses onto the clip's *mean pose* — the large, static part of
``x`` — because encoding it buys the biggest reconstruction gain per KL nat,
and the movement (a small residual) washes out. This model removes that failure
mode by construction.

Each clip is split into a deterministic conditioning part and a residual:

    a = mean_t x_t                 (J, D)  the anchor: mean pose over the clip
    s = median_t ||shoulder_mid - hip_mid||   scalar  the size / torso scale
    r_t = (x_t - a) / s            (T, J, D)  the movement, pose & size removed
    c = (vec(a), s)                (D*J + 1,) the conditioning variable

The VAE only models ``r``: the encoder tokenises ``r`` (one token per
(joint, frame)), the decoder emits ``r_hat``, and the clip is reassembled
*outside* the network as ``x_hat = a + s * r_hat``. Because ``a`` and ``s`` are
added back deterministically, the latent ``z`` can only shape the movement — no
KL nat is spent on the mean pose. ``c`` enters only as FiLM modulation
([Perez et al., 2018]): once in the encoder (a single site in the first block)
and at every block in the decoder, so the decoder is strongly conditioned on
where and how big the person is.

The backbone is the factorised space-time attention of
:class:`SpatioTemporalTransformerVAE` (alternating spatial and temporal
blocks). Same public interface as the other VAEs (``encode`` /
``decode_full`` / ``decode_inp`` / ``forward``) so it drops into training,
evaluation, and the analysis unchanged, and generic in the coordinate
dimension ``n_dims``.

Decode conditioning. ``forward`` (training / evaluation) reassembles with the
clip's own ``(a, s)`` — a faithful ``x_hat``. A bare ``decode_full(z)`` with no
conditioning falls back to the *canonical* frame ``a = 0, s = 1`` and returns
``r_hat`` directly: this is what the analysis toolkit calls, so the decoder
Jacobians, the pull-back metric, and latent traversals all live in residual
(movement) space — exactly the quantity ``z`` controls — instead of being
dominated by a mean-pose offset. To reassemble an analysis reconstruction into
image space, pass the clip's ``(a, s)`` (see :meth:`anchor_scale`) as ``c``.
"""

from __future__ import annotations

from .common import (torch, nn, BottleneckHeads, reparameterise,
                     sinusoidal_positional_encoding)


class _FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning embedding.

    Maps a ``(B, d_model)`` conditioning vector to a per-channel scale and
    shift and applies ``(1 + gamma) * h + beta`` to a ``(B, ..., d_model)``
    token tensor. Zero-initialised, so at the start of training it is the
    identity and the model behaves as if unconditioned, then learns to use the
    conditioning.
    """

    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, h, cond):
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)   # (B, d_model)
        # Broadcast over the token axes between batch and channel.
        view = (cond.shape[0],) + (1,) * (h.dim() - 2) + (h.shape[-1],)
        return (1 + gamma.view(view)) * h + beta.view(view)


class _STBlock(nn.Module):
    """One factorised block: spatial self-attention then temporal, with FiLM.

    Operates on ``(B, T, J, d_model)``. Spatial attention mixes the J joints
    within each frame; temporal attention mixes the T frames within each joint.
    When ``film`` is set, a FiLM site precedes *each* sub-attention (the
    decoder's "modulation at every sub-layer"); the encoder passes ``film=False``
    and modulates once, outside the block.
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int,
                 dropout: float, film: bool):
        super().__init__()
        self.spatial = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.temporal = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.film_spatial = _FiLM(d_model, d_model) if film else None
        self.film_temporal = _FiLM(d_model, d_model) if film else None

    def forward(self, x, cond=None):
        B, T, J, d = x.shape
        if self.film_spatial is not None and cond is not None:
            x = self.film_spatial(x, cond)
        x = self.spatial(x.reshape(B * T, J, d)).reshape(B, T, J, d)
        if self.film_temporal is not None and cond is not None:
            x = self.film_temporal(x, cond)
        x = x.permute(0, 2, 1, 3).reshape(B * J, T, d)
        x = self.temporal(x).reshape(B, J, T, d).permute(0, 2, 1, 3)
        return x.contiguous()


class AnchoredSpatioTemporalVAE(nn.Module):
    """Residual VAE that conditions on the clip's mean pose and scale.

    See the module docstring for the decomposition. ``shoulder_joints`` and
    ``hip_joints`` name the four keypoints whose two midpoints define the torso
    length ``s``; when either is ``None`` a generic per-frame bounding-box
    diagonal (median over frames) is used instead, so the model still runs on an
    arbitrary skeleton.
    """

    def __init__(self, T: int, J: int, d_z: int = 32,
                 d_model: int = 96, n_heads: int = 4, n_layers: int = 3,
                 ffn_ratio: int = 4, dropout: float = 0.1,
                 inpainting: bool = False,
                 n_cond: int = 0, cond_dim: int = 8,
                 cond_dropout: float = 0.0, n_dims: int = 3,
                 shoulder_joints: tuple[int, int] | None = None,
                 hip_joints: tuple[int, int] | None = None,
                 scale_eps: float = 1e-3):
        super().__init__()
        if n_cond > 0:
            raise NotImplementedError(
                "AnchoredSpatioTemporalVAE conditions on the anchor/scale via "
                "FiLM; cohort conditioning (n_cond > 0) is not supported. Use "
                "transformer_attention='factorized' for a cohort CVAE.")
        self.T = T
        self.J = J
        self.d_z = d_z
        self.n_dims = n_dims
        self.d_model = d_model
        self.inpainting = inpainting
        self.shoulder_joints = tuple(shoulder_joints) if shoulder_joints else None
        self.hip_joints = tuple(hip_joints) if hip_joints else None
        self.scale_eps = float(scale_eps)

        ff = d_model * ffn_ratio
        cond_in = n_dims * J + 1                      # (vec(a), s)

        # Shared conditioning embedding c -> (B, d_model), then per-site FiLM.
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))

        # ---- Encoder ------------------------------------------------------
        # One token per (joint, frame): the joint's n_dims residual coordinates.
        # A learned mask token replaces hidden positions.
        self.token_embed = nn.Linear(n_dims, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.enc_joint = nn.Parameter(torch.zeros(1, 1, J, d_model))
        self.register_buffer(
            "enc_time", sinusoidal_positional_encoding(T, d_model),
            persistent=False)
        self.enc_film = _FiLM(d_model, d_model)      # the single encoder site
        self.enc_blocks = nn.ModuleList(
            [_STBlock(d_model, n_heads, ff, dropout, film=False)
             for _ in range(n_layers)])
        self.heads = BottleneckHeads(d_model, d_z)

        # ---- Decoder ------------------------------------------------------
        self.query_lift = nn.Linear(d_z, d_model)
        self.dec_joint = nn.Parameter(torch.zeros(1, 1, J, d_model))
        self.register_buffer(
            "dec_time", sinusoidal_positional_encoding(T, d_model),
            persistent=False)
        self.dec_blocks = nn.ModuleList(
            [_STBlock(d_model, n_heads, ff, dropout, film=True)
             for _ in range(n_layers)])
        self.dec_output_full = nn.Linear(d_model, n_dims)
        if inpainting:
            self.dec_output_inp = nn.Linear(d_model + 1, n_dims)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.enc_joint, std=0.02)
        nn.init.normal_(self.dec_joint, std=0.02)

    # ---- Anchor / scale / conditioning -----------------------------------
    def anchor_scale(self, X):
        """Deterministic (anchor, scale) of a clip batch.

        Args:
            X: (B, T, J, D).
        Returns:
            (a, s): a is (B, J, D), the mean pose; s is (B,), the torso length
            (or bounding-box diagonal), clamped to ``scale_eps``.
        """
        a = X.mean(dim=1)                                 # (B, J, D)
        if self.shoulder_joints is not None and self.hip_joints is not None:
            sh = 0.5 * (X[:, :, self.shoulder_joints[0]]
                        + X[:, :, self.shoulder_joints[1]])   # (B, T, D)
            hp = 0.5 * (X[:, :, self.hip_joints[0]]
                        + X[:, :, self.hip_joints[1]])
            s = torch.linalg.vector_norm(sh - hp, dim=-1).median(dim=1).values
        else:
            extent = X.amax(dim=2) - X.amin(dim=2)            # (B, T, D)
            s = torch.linalg.vector_norm(extent, dim=-1).median(dim=1).values
        return a, s.clamp_min(self.scale_eps)

    def _cond_embed(self, a, s):
        """FiLM conditioning embedding h_c from (a, s)."""
        c = torch.cat([a.reshape(a.shape[0], -1), s.unsqueeze(-1)], dim=-1)
        return self.cond_mlp(c)

    def _canonical_cond(self, batch: int, device):
        c = torch.zeros(batch, self.n_dims * self.J + 1, device=device)
        c[:, -1] = 1.0                                    # s = 1, a = 0
        return self.cond_mlp(c)

    # ---- Encoder ---------------------------------------------------------
    def encode(self, X, M, c=None):
        """Map (clip, mask) to (mu, logvar). ``c`` is ignored (FiLM uses a, s)."""
        B, T, J, D = X.shape
        a, s = self.anchor_scale(X)
        r = (X - a.unsqueeze(1)) / s.view(B, 1, 1, 1)     # (B, T, J, D)
        h_c = self._cond_embed(a, s)                      # (B, d_model)

        tok = self.token_embed(r)                         # (B, T, J, d_model)
        vis = M.unsqueeze(-1)
        tok = vis * tok + (1.0 - vis) * self.mask_token   # learned mask token
        tok = tok + self.enc_joint + self.enc_time[None, :, None, :]
        tok = self.enc_film(tok, h_c)                     # single encoder site
        for blk in self.enc_blocks:
            tok = blk(tok)
        h = tok.mean(dim=(1, 2))                          # pool T, J
        return self.heads(h)

    # ---- Decoder ---------------------------------------------------------
    def _decode_trunk(self, z, a=None, s=None):
        B = z.shape[0]
        h_c = (self._cond_embed(a, s) if a is not None
               else self._canonical_cond(B, z.device))
        q = self.query_lift(z)[:, None, None, :].expand(
            B, self.T, self.J, self.d_model)
        q = q + self.dec_joint + self.dec_time[None, :, None, :]
        h = q
        for blk in self.dec_blocks:
            h = blk(h, h_c)                               # FiLM every block
        return h                                          # (B, T, J, d_model)

    def _reassemble(self, r_hat, a, s):
        """x_hat = a + s * r_hat, or r_hat itself in the canonical frame."""
        if a is None:
            return r_hat
        return a.unsqueeze(1) + s.view(-1, 1, 1, 1) * r_hat

    def decode_full(self, z, c=None):
        """Full-clip reconstruction.

        ``c`` is an optional ``(a, s)`` pair (the anchor/scale to reassemble
        with). ``None`` (the analysis default) decodes in the canonical frame
        and returns the residual ``r_hat`` directly.
        """
        a, s = c if c is not None else (None, None)
        r_hat = self.dec_output_full(self._decode_trunk(z, a, s))
        return self._reassemble(r_hat, a, s)

    def decode_inp(self, z, M, c=None):
        """Mask-conditioned inpainting head (Recipe 3 only)."""
        if not self.inpainting:
            raise RuntimeError("Model was built without the inpainting head.")
        a, s = c if c is not None else (None, None)
        h = self._decode_trunk(z, a, s)
        h = torch.cat([h, M.unsqueeze(-1)], dim=-1)       # (B, T, J, d_model + 1)
        r_hat = self.dec_output_inp(h)
        return self._reassemble(r_hat, a, s)

    # ---- Combined --------------------------------------------------------
    def forward(self, X, M, c=None):
        """Encode, sample, decode, reassembling with the clip's own (a, s)."""
        a, s = self.anchor_scale(X)
        mu, logvar = self.encode(X, M)
        z = reparameterise(mu, logvar)
        X_hat = self.decode_full(z, (a, s))
        if self.inpainting:
            return X_hat, self.decode_inp(z, M, (a, s)), mu, logvar
        return X_hat, mu, logvar
