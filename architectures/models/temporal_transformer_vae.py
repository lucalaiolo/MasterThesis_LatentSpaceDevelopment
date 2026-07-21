"""Temporal-latent transformer VAE ([ARCH §4.6]).

The transformer counterpart of :class:`TemporalConvVAE`: instead of compressing a
clip to a single latent vector (which collapses the reconstruction to a static
mean pose), it keeps a latent **per time-window**. This is the attention version
of the temporally-downsampled latent sequence used by T2M-GPT / MotionGPT and
PRISM, but continuous (no VQ) so it stays a plain VAE.

Two attention patterns share the same per-window latent (set by ``attention``):

* ``"temporal"`` (default) — **frame tokens**. Each frame is one token (its J
  joints + a mask channel projected to ``d_model``); the tokens attend over the
  T frames. Cheapest; the pose structure is baked into the token embedding.
* ``"factorized"`` — **(joint, frame) tokens**. Each joint of each frame is a
  token; ``n_layers`` factorised blocks alternate spatial attention (across the J
  joints within a frame) and temporal attention (across the T frames within a
  joint), the divided space-time attention of PoseFormer / the ST-Transformer.
  This keeps the "don't sacrifice split spatial/temporal attention" inductive
  bias *inside* the motion-preserving temporal-latent model.

Either way the T frames are down-sampled by ``l`` into ``T/l`` windows and a
posterior head reads a ``d_z`` latent at **each** window, so the posterior is
``(d_z, T/l)`` flattened to ``(B, d_z * T/l)`` — the standard ELBO / KL / beta
machinery and the analysis toolkit (and ``window_latents`` / ``flatten_windows``
/ ``pool_global``) run unchanged for both attention modes.

Because a distinct latent controls each window, the decoded clip moves — it
cannot collapse to one static pose.
"""

from __future__ import annotations

from .common import (torch, nn, pack_encoder_input, reparameterise,
                     sinusoidal_positional_encoding)
from .spatiotemporal_vae import _SpatioTemporalBlock


class TemporalTransformerVAE(nn.Module):
    """Transformer VAE with a per-window (temporal) latent.

    ``downsample`` (``l``) sets the window length: the clip has ``T/l`` latent
    windows, each of width ``d_z``. ``attention`` picks the frame-token
    (``"temporal"``) or factorised space-time (``"factorized"``) trunk. Generic
    in the coordinate dimension.
    """

    def __init__(self, T: int, J: int, d_z: int = 32,
                 d_model: int = 96, n_heads: int = 4, n_layers: int = 3,
                 ffn_ratio: int = 4, dropout: float = 0.1,
                 inpainting: bool = False,
                 n_cond: int = 0, cond_dim: int = 8,
                 cond_dropout: float = 0.0, n_dims: int = 3,
                 downsample: int = 4, attention: str = "temporal"):
        super().__init__()
        if n_cond > 0:
            raise NotImplementedError(
                "TemporalTransformerVAE does not implement cohort conditioning "
                "yet; use n_cond=0.")
        if attention not in ("temporal", "factorized"):
            raise ValueError(
                f"attention must be 'temporal' or 'factorized', got {attention!r}.")
        assert T % downsample == 0, "clip length must divide the downsample factor."
        self.T = T
        self.J = J
        self.d_z = d_z
        self.n_dims = n_dims
        self.d_model = d_model
        self.inpainting = inpainting
        self.l = downsample
        self.n_win = T // downsample
        self.attention = attention

        ff = d_model * ffn_ratio

        # Temporal positional encodings (shared by both trunks, length T).
        self.register_buffer(
            "enc_pos", sinusoidal_positional_encoding(T, d_model),
            persistent=False)
        self.register_buffer(
            "dec_pos", sinusoidal_positional_encoding(T, d_model),
            persistent=False)

        if attention == "factorized":
            # ---- Factorised (joint, frame) trunk --------------------------
            # Per-joint token embedding (n_dims coords + 1 mask channel) and a
            # learned spatial (joint) positional embedding; temporal position is
            # the shared sinusoid above.
            self.token_embed_j = nn.Linear(n_dims + 1, d_model)
            self.joint_pos = nn.Parameter(torch.zeros(J, d_model))
            self.enc_blocks = nn.ModuleList([
                _SpatioTemporalBlock(d_model, n_heads, ff, dropout)
                for _ in range(n_layers)])
            self.enc_norm = nn.LayerNorm(d_model)
            # Decoder: a learned per-joint query, upsampled window latents added.
            self.joint_query = nn.Parameter(torch.zeros(J, d_model))
            self.dec_blocks = nn.ModuleList([
                _SpatioTemporalBlock(d_model, n_heads, ff, dropout)
                for _ in range(n_layers)])
            self.dec_norm = nn.LayerNorm(d_model)
            self.dec_output_full = nn.Linear(d_model, n_dims)       # per (t, j)
            if inpainting:
                self.dec_output_inp = nn.Linear(d_model + 1, n_dims)
        else:
            # ---- Frame-token trunk ---------------------------------------
            self.token_embed = nn.Linear((n_dims + 1) * J, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True)
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
            dec_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True)
            self.decoder = nn.TransformerEncoder(
                dec_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
            self.dec_output_full = nn.Linear(d_model, n_dims * J)   # per frame
            if inpainting:
                self.dec_output_inp = nn.Linear(d_model + J, n_dims * J)

        # ---- Per-window posterior heads and latent lift (shared) ----------
        self.to_mu = nn.Linear(d_model, d_z)          # per window
        self.to_logvar = nn.Linear(d_model, d_z)
        self.from_z = nn.Linear(d_z, d_model)         # per window

    def n_windows(self) -> int:
        """Number of temporal latent windows (``T / downsample``)."""
        return self.n_win

    # ---- Window layout (transformer order: latent is (B, n_win, d_z)) ----
    def window_latents(self, z):
        """Reshape a flattened latent ``(B, d_z*n_win)`` to ``(B, n_win, d_z)``.

        The per-window head lays the latent out window-major — ``mu`` is
        ``(B, n_win, d_z)`` before flattening — so recovering the window
        sequence is a plain reshape (no transpose, unlike the conv model). Holds
        for both attention modes.
        """
        B = z.shape[0]
        return z.reshape(B, self.n_win, self.d_z)

    def flatten_windows(self, w):
        """Inverse of :meth:`window_latents`: ``(B, n_win, d_z) -> (B, d_z*n_win)``."""
        B = w.shape[0]
        return w.reshape(B, -1)

    # ---- Encoder ---------------------------------------------------------
    def encode(self, X, M, c=None):
        """Map (clip, mask) to per-window (mu, logvar), flattened.

        Returns:
            (mu, logvar), each ``(B, d_z * T/l)``. Reshape with
            ``(B, T/l, d_z)`` to recover the windows.
        """
        B = X.shape[0]
        if self.attention == "factorized":
            feat = torch.cat([X, M.unsqueeze(-1)], dim=-1)   # (B, T, J, D+1)
            tok = self.token_embed_j(feat)                   # (B, T, J, d_model)
            tok = (tok + self.enc_pos[None, :, None, :]
                   + self.joint_pos[None, None, :, :])
            h = tok
            for blk in self.enc_blocks:
                h = blk(h)                                   # (B, T, J, d_model)
            h = self.enc_norm(h)
            # Pool each window's l frames AND the J joints -> one token/window.
            h = h.reshape(B, self.n_win, self.l, self.J, self.d_model).mean(dim=(2, 3))
        else:
            x = pack_encoder_input(X, M)                     # (B, T, (D+1)J)
            tok = self.token_embed(x) + self.enc_pos.unsqueeze(0)
            h = self.encoder(tok)                            # (B, T, d_model)
            h = h.reshape(B, self.n_win, self.l, self.d_model).mean(dim=2)
        mu = self.to_mu(h)                                   # (B, n_win, d_z)
        logvar = self.to_logvar(h)
        return mu.reshape(B, -1), logvar.reshape(B, -1)

    # ---- Decoder ---------------------------------------------------------
    def _decode_trunk(self, z):
        """Window latents -> hidden tokens ready for the output head.

        Returns ``(B, T, d_model)`` for the frame-token trunk, or
        ``(B, T, J, d_model)`` for the factorised trunk.
        """
        B = z.shape[0]
        w = z.view(B, self.n_win, self.d_z)           # (B, n_win, d_z)
        w = self.from_z(w)                            # (B, n_win, d_model)
        q = w.repeat_interleave(self.l, dim=1)        # (B, T, d_model) upsample
        if self.attention == "factorized":
            q = q.unsqueeze(2).expand(B, self.T, self.J, self.d_model)
            q = (q + self.joint_query[None, None, :, :]
                 + self.dec_pos[None, :, None, :])
            h = q
            for blk in self.dec_blocks:
                h = blk(h)                            # (B, T, J, d_model)
            return self.dec_norm(h)
        q = q + self.dec_pos.unsqueeze(0)
        return self.decoder(q)                        # (B, T, d_model)

    def decode_full(self, z, c=None):
        """Full-clip reconstruction from the per-window latent."""
        B = z.shape[0]
        h = self._decode_trunk(z)
        if self.attention == "factorized":
            return self.dec_output_full(h)            # (B, T, J, n_dims)
        x_hat = self.dec_output_full(h)               # (B, T, DJ)
        return x_hat.reshape(B, self.T, self.J, self.n_dims)

    def decode_inp(self, z, M, c=None):
        """Mask-conditioned inpainting head (Recipe 3 only)."""
        if not self.inpainting:
            raise RuntimeError("Model was built without the inpainting head.")
        B = z.shape[0]
        h = self._decode_trunk(z)
        if self.attention == "factorized":
            h = torch.cat([h, M.unsqueeze(-1)], dim=-1)   # (B, T, J, d_model+1)
            return self.dec_output_inp(h)                 # (B, T, J, n_dims)
        h = torch.cat([h, M], dim=-1)                     # (B, T, d_model + J)
        x_hat = self.dec_output_inp(h)
        return x_hat.reshape(B, self.T, self.J, self.n_dims)

    # ---- Analysis helper -------------------------------------------------
    def pool_global(self, z):
        """Mean over the windows -> one ``(B, d_z)`` per-clip summary."""
        B = z.shape[0]
        return z.view(B, self.n_win, self.d_z).mean(dim=1)

    # ---- Combined --------------------------------------------------------
    def forward(self, X, M, c=None):
        """Encode, sample, decode."""
        mu, logvar = self.encode(X, M)
        z = reparameterise(mu, logvar)
        X_hat_full = self.decode_full(z)
        if self.inpainting:
            return X_hat_full, self.decode_inp(z, M), mu, logvar
        return X_hat_full, mu, logvar
