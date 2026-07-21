"""Temporal-latent convolutional VAE ([ARCH §4.5]).

Every other model here compresses a whole clip into a **single** latent vector
``z``. For motion that is a known failure mode: the reconstruction collapses to
the temporal mean pose (the single global latent cannot carry the trajectory,
and posterior collapse finishes the job). The motion-generation literature fixes
this by keeping a latent that varies over (down-sampled) **time** — a sequence of
latent tokens, one per short window — rather than one vector per clip. This is
the continuous (non-VQ) analogue of the T2M-GPT / MotionGPT motion tokenizer and
of the temporally-downsampled latents in PRISM:

    T2M-GPT / MotionGPT   conv VQ-VAE, stride-2 temporal downsampling ->
                          Z = [z_1, ..., z_{T/l}]  (Zhang et al. 2023)
    PRISM                 causal spatio-temporal VAE, latent tokens over
                          ~4-frame windows (2026)

Here the convolutional encoder down-samples time by ``l`` (the product of the
conv strides, 4 by default) and emits a posterior **per window** — a latent of
shape ``(d_z, T/l)`` rather than ``(d_z,)``. Because a distinct latent controls
each window, the decoded clip can (and does) move: it *cannot* collapse to a
single static pose.

Interface: ``encode`` returns the per-window latents **flattened** to
``(B, d_z * T/l)`` so the standard ELBO machinery (``reparameterise``,
``kl_gaussian`` summed over the whole latent, the beta schedule) works unchanged,
and ``encode`` / ``decode_full`` stay exact inverses on that latent so the
analysis toolkit (Jacobians, traversals, pull-back metric) runs as-is. The
effective latent width is therefore ``latent_dim * (T / downsample)`` — see
:meth:`n_windows`. For a single per-clip descriptor (e.g. phenotype clustering)
pool the windows with :meth:`pool_global`.
"""

from __future__ import annotations

from .common import (torch, nn, LayerNormChannels, pack_encoder_input,
                     reparameterise)
from .conv_vae import _conv_block, _conv_transpose_block


class TemporalConvVAE(nn.Module):
    """Convolutional VAE with a per-window (temporal) latent.

    Encoder: the same three residual conv blocks as :class:`ConvVAE` take the
    ``(D + 1)J`` input channels down to width ``2C`` and time ``T / l`` (``l`` the
    product of the strides). A 1×1 conv then reads a ``d_z`` posterior at **each**
    of the ``T / l`` positions, giving per-window ``(mu, logvar)``.

    Decoder: a 1×1 conv lifts each window's latent back to ``2C``, two
    transposed-conv blocks up-sample ``T / l -> T``, and a final conv projects to
    the ``DJ`` output channels. Recipe 3 adds the mask-conditioned inpainting
    head, exactly as :class:`ConvVAE`.
    """

    def __init__(self, T: int, J: int, d_z: int = 32, base_channels: int = 64,
                 kernels: tuple[int, int, int] = (5, 3, 3),
                 strides: tuple[int, int, int] = (1, 2, 2),
                 inpainting: bool = False,
                 n_cond: int = 0, cond_dim: int = 8,
                 cond_dropout: float = 0.0, n_dims: int = 3):
        super().__init__()
        if n_cond > 0:
            raise NotImplementedError(
                "TemporalConvVAE does not implement cohort conditioning yet; "
                "use n_cond=0.")
        self.T = T
        self.J = J
        self.d_z = d_z
        self.n_dims = n_dims
        self.inpainting = inpainting

        C = base_channels
        in_channels = (n_dims + 1) * J
        out_channels = n_dims * J
        downsample = strides[0] * strides[1] * strides[2]
        assert T % downsample == 0, "clip length must divide the total stride."
        self.T_bottleneck = T // downsample
        self.bottleneck_channels = 2 * C

        # Encoder ([ARCH §3.1]): identical trunk to ConvVAE.
        self.enc = nn.Sequential(
            _conv_block(in_channels, C,     kernels[0], strides[0]),
            _conv_block(C,           2 * C, kernels[1], strides[1]),
            _conv_block(2 * C,       2 * C, kernels[2], strides[2]),
        )
        # Per-window posterior: a 1x1 conv reads d_z at every T/l position.
        self.to_mu = nn.Conv1d(2 * C, d_z, kernel_size=1)
        self.to_logvar = nn.Conv1d(2 * C, d_z, kernel_size=1)

        # Decoder: 1x1 conv lifts each window latent back to 2C, then upsample.
        self.from_z = nn.Conv1d(d_z, 2 * C, kernel_size=1)
        self.dec_upsample = nn.Sequential(
            _conv_transpose_block(2 * C, 2 * C, kernels[2], strides[2]),
            _conv_transpose_block(2 * C, C,     kernels[1], strides[1]),
        )
        self.dec_output_full = nn.Conv1d(C, out_channels,
                                         kernel_size=kernels[0], stride=1,
                                         padding=(kernels[0] - 1) // 2)
        if inpainting:
            self.dec_output_inp = nn.Conv1d(C + J, out_channels,
                                            kernel_size=kernels[0], stride=1,
                                            padding=(kernels[0] - 1) // 2)

    def n_windows(self) -> int:
        """Number of temporal latent windows (``T / downsample``)."""
        return self.T_bottleneck

    # ---- Window layout (conv order: latent is (B, d_z, n_win)) -----------
    def window_latents(self, z):
        """Reshape a flattened latent ``(B, d_z*n_win)`` to ``(B, n_win, d_z)``.

        The conv head lays the latent out channel-major — ``mu`` is
        ``(B, d_z, n_win)`` before flattening — so recovering the window
        sequence transposes the two axes. Used by the stitcher and by
        per-window (temporal) dynamics.
        """
        B = z.shape[0]
        return z.reshape(B, self.d_z, self.T_bottleneck).transpose(1, 2)

    def flatten_windows(self, w):
        """Inverse of :meth:`window_latents`: ``(B, n_win, d_z) -> (B, d_z*n_win)``.

        Lets callers build a latent from an explicit window sequence (e.g. a
        constant-state block for decoding an HMM state's appearance) in the
        exact order ``decode_full`` expects.
        """
        B = w.shape[0]
        return w.transpose(1, 2).reshape(B, -1)

    # ---- Encoder ---------------------------------------------------------
    def encode(self, X, M, c=None):
        """Map (clip, mask) to per-window (mu, logvar), flattened.

        Returns:
            (mu, logvar), each ``(B, d_z * T/l)`` — the per-window posteriors
            flattened so the standard ELBO / KL machinery treats them as one
            latent. Reshape with ``(B, d_z, T/l)`` to recover the windows.
        """
        x = pack_encoder_input(X, M)                 # (B, T, (D + 1)J)
        h = self.enc(x.transpose(1, 2))              # (B, 2C, T/l)
        mu = self.to_mu(h)                           # (B, d_z, T/l)
        logvar = self.to_logvar(h)                   # (B, d_z, T/l)
        B = X.shape[0]
        return mu.reshape(B, -1), logvar.reshape(B, -1)

    # ---- Decoder ---------------------------------------------------------
    def _decode_trunk(self, z):
        B = z.shape[0]
        g = z.view(B, self.d_z, self.T_bottleneck)   # (B, d_z, T/l)
        g = self.from_z(g)                           # (B, 2C, T/l)
        return self.dec_upsample(g)                  # (B, C, T)

    def decode_full(self, z, c=None):
        """Full-clip reconstruction from the per-window latent."""
        g = self._decode_trunk(z)
        x_hat = self.dec_output_full(g)              # (B, DJ, T)
        B = z.shape[0]
        return x_hat.transpose(1, 2).reshape(B, self.T, self.J, self.n_dims)

    def decode_inp(self, z, M, c=None):
        """Mask-conditioned inpainting head (Recipe 3 only)."""
        if not self.inpainting:
            raise RuntimeError("Model was built without the inpainting head.")
        g = self._decode_trunk(z)
        g = torch.cat([g, M.transpose(1, 2)], dim=1)  # (B, C + J, T)
        x_hat = self.dec_output_inp(g)               # (B, DJ, T)
        B = z.shape[0]
        return x_hat.transpose(1, 2).reshape(B, self.T, self.J, self.n_dims)

    # ---- Analysis helper -------------------------------------------------
    def pool_global(self, z):
        """Mean over the windows -> one ``(B, d_z)`` per-clip summary.

        A convenience for downstream code that wants a single vector per clip
        (e.g. phenotype clustering) rather than the full per-window latent.
        """
        B = z.shape[0]
        return z.view(B, self.d_z, self.T_bottleneck).mean(dim=2)

    # ---- Combined --------------------------------------------------------
    def forward(self, X, M, c=None):
        """Encode, sample, decode.

        Returns:
            (X_hat_full, mu, logvar), or (X_hat_full, X_hat_inp, mu, logvar)
            for Recipe 3.
        """
        mu, logvar = self.encode(X, M)
        z = reparameterise(mu, logvar)
        X_hat_full = self.decode_full(z)
        if self.inpainting:
            return X_hat_full, self.decode_inp(z, M), mu, logvar
        return X_hat_full, mu, logvar
