"""Factorised space-time transformer VAE ([ARCH §4.3]).

The frame-token :class:`TransformerVAE` collapses each frame's J joints into a
single token and attends only over time. This variant keeps **one token per
(joint, frame)** and alternates two cheaper attentions:

    spatial   — across the J joints *within* a frame (pose structure), and
    temporal  — across the T frames *within* a joint (motion),

the "divided" / factorised space-time attention of PoseFormer / the
ST-Transformer. It is quadratic in J and in T separately rather than in J*T,
and it separates the two inductive biases: a spatial block reasons about the
pose, a temporal block about how it moves.

Same interface as the other VAEs (``encode`` / ``decode_full`` /
``decode_inp`` / ``forward``), so it drops into training, evaluation, and the
analysis unchanged. Generic in the coordinate dimension ``n_dims``.
"""

from __future__ import annotations

from .common import (torch, nn, BottleneckHeads, ConditioningEmbedding,
                     reparameterise, sinusoidal_positional_encoding)


class _SpatioTemporalBlock(nn.Module):
    """One factorised block: spatial self-attention, then temporal.

    Operates on a ``(B, T, J, d_model)`` tensor. Spatial attention mixes the J
    joints within each frame; temporal attention mixes the T frames within each
    joint. Each sub-attention is a pre-norm ``TransformerEncoderLayer`` (its own
    LayerNorms, attention, and feedforward).
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int,
                 dropout: float):
        super().__init__()
        self.spatial = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.temporal = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)

    def forward(self, x):
        B, T, J, d = x.shape
        # Spatial: attend across the J joints of each frame (B*T sequences of J).
        x = self.spatial(x.reshape(B * T, J, d)).reshape(B, T, J, d)
        # Temporal: attend across the T frames of each joint (B*J sequences of T).
        x = x.permute(0, 2, 1, 3).reshape(B * J, T, d)
        x = self.temporal(x).reshape(B, J, T, d).permute(0, 2, 1, 3)
        return x.contiguous()


class SpatioTemporalTransformerVAE(nn.Module):
    """Frame×joint transformer VAE with factorised spatial/temporal attention.

    Encoder: a per-(joint, frame) token embeds the joint's ``n_dims``
    coordinates plus a mask channel; a learned joint (spatial) embedding and a
    sinusoidal temporal positional encoding are added; ``n_layers`` factorised
    blocks run; the tokens are mean-pooled over time and joints and fed to the
    posterior heads.

    Decoder: the latent is lifted to a query, broadcast to every (joint, frame)
    position with the same spatial + temporal encodings, refined by ``n_layers``
    factorised blocks, and projected per token to ``n_dims`` coordinates. The
    Recipe-3 inpainting head concatenates the per-joint mask bit before the
    projection.

    ``n_cond > 0`` turns it into a CVAE exactly as the other models
    ([CARE-PD §6]); ``n_cond == 0`` (default) is the plain VAE.
    """

    def __init__(self, T: int, J: int, d_z: int = 32,
                 d_model: int = 96, n_heads: int = 4, n_layers: int = 3,
                 ffn_ratio: int = 4, dropout: float = 0.1,
                 inpainting: bool = False,
                 n_cond: int = 0, cond_dim: int = 8,
                 cond_dropout: float = 0.0, n_dims: int = 3):
        super().__init__()
        self.T = T
        self.J = J
        self.d_z = d_z
        self.n_dims = n_dims
        self.d_model = d_model
        self.inpainting = inpainting

        self.n_cond = n_cond
        d_c = cond_dim if n_cond > 0 else 0
        self.cond = (ConditioningEmbedding(n_cond, cond_dim, cond_dropout)
                     if n_cond > 0 else None)

        ff = d_model * ffn_ratio

        # ---- Encoder ------------------------------------------------------
        # Per-(joint, frame) token: n_dims coordinates + one mask channel.
        self.token_embed = nn.Linear(n_dims + 1, d_model)
        # Learned joint (spatial) embedding; sinusoidal temporal PE.
        self.enc_joint = nn.Parameter(torch.zeros(1, 1, J, d_model))
        self.register_buffer(
            "enc_time", sinusoidal_positional_encoding(T, d_model),
            persistent=False)
        self.enc_blocks = nn.ModuleList(
            [_SpatioTemporalBlock(d_model, n_heads, ff, dropout)
             for _ in range(n_layers)])
        self.heads = BottleneckHeads(d_model + d_c, d_z)

        # ---- Decoder ------------------------------------------------------
        self.query_lift = nn.Linear(d_z + d_c, d_model)
        self.dec_joint = nn.Parameter(torch.zeros(1, 1, J, d_model))
        self.register_buffer(
            "dec_time", sinusoidal_positional_encoding(T, d_model),
            persistent=False)
        self.dec_blocks = nn.ModuleList(
            [_SpatioTemporalBlock(d_model, n_heads, ff, dropout)
             for _ in range(n_layers)])
        self.dec_output_full = nn.Linear(d_model, n_dims)
        if inpainting:
            # Mask-conditioned per token: each joint token is told if it is
            # hidden ([MVAE §5.1]).
            self.dec_output_inp = nn.Linear(d_model + 1, n_dims)

        nn.init.normal_(self.enc_joint, std=0.02)
        nn.init.normal_(self.dec_joint, std=0.02)

    # ---- Encoder ---------------------------------------------------------
    def encode(self, X, M, c=None):
        """Map (clip, mask[, cohort]) to (mu, logvar).

        Args:
            X: (B, T, J, D).
            M: (B, T, J), 1 for visible.
            c: optional (B,) conditioning ids; ignored for a plain VAE.
        Returns:
            (mu, logvar), each (B, d_z).
        """
        B = X.shape[0]
        Xm = X * M.unsqueeze(-1)                          # zero the hidden joints
        feat = torch.cat([Xm, M.unsqueeze(-1)], dim=-1)   # (B, T, J, D + 1)
        h = self.token_embed(feat)                        # (B, T, J, d_model)
        h = h + self.enc_joint + self.enc_time[None, :, None, :]
        for blk in self.enc_blocks:
            h = blk(h)
        h = h.mean(dim=(1, 2))                            # pool T, J -> (B, d_model)
        if self.cond is not None:
            e = self.cond.encoder_vector(c, B, h.device)
            h = torch.cat([h, e], dim=1)
        return self.heads(h)

    # ---- Decoder ---------------------------------------------------------
    def _decode_trunk(self, z, c=None):
        B = z.shape[0]
        if self.cond is not None:
            e = self.cond.decoder_vector(c, B, z.device, self.training)
            z = torch.cat([z, e], dim=1)
        q = self.query_lift(z)                            # (B, d_model)
        q = q[:, None, None, :].expand(B, self.T, self.J, self.d_model)
        q = q + self.dec_joint + self.dec_time[None, :, None, :]
        h = q
        for blk in self.dec_blocks:
            h = blk(h)
        return h                                          # (B, T, J, d_model)

    def decode_full(self, z, c=None):
        """Full-clip reconstruction head, ignoring the mask."""
        h = self._decode_trunk(z, c)
        return self.dec_output_full(h)                    # (B, T, J, D)

    def decode_inp(self, z, M, c=None):
        """Mask-conditioned inpainting head (Recipe 3 only)."""
        if not self.inpainting:
            raise RuntimeError("Model was built without the inpainting head.")
        h = self._decode_trunk(z, c)
        h = torch.cat([h, M.unsqueeze(-1)], dim=-1)       # (B, T, J, d_model + 1)
        return self.dec_output_inp(h)                     # (B, T, J, D)

    # ---- Combined --------------------------------------------------------
    def forward(self, X, M, c=None):
        """Encode, sample, decode.

        Returns:
            (X_hat_full, mu, logvar) without the inpainting head, or
            (X_hat_full, X_hat_inp, mu, logvar) for Recipe 3.
        """
        mu, logvar = self.encode(X, M, c)
        z = reparameterise(mu, logvar)
        X_hat_full = self.decode_full(z, c)
        if self.inpainting:
            return X_hat_full, self.decode_inp(z, M, c), mu, logvar
        return X_hat_full, mu, logvar
