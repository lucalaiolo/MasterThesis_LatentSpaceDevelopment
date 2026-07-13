"""1D temporal convolutional VAE ([ARCH §3]).

Three residual blocks widen the channel count and halve T twice. The
decoder mirrors the encoder with transposed convolutions. For Recipe 3
the decoder branches into two heads on top of a shared trunk: a full
head that reconstructs the whole clip, and a mask-conditioned inpainting
head scored only on hidden positions ([MVAE §5]).
"""

from __future__ import annotations

from .common import (torch, nn, LayerNormChannels, BottleneckHeads,
                     ConditioningEmbedding, pack_encoder_input, reparameterise)


def _conv_block(c_in: int, c_out: int, kernel: int, stride: int) -> nn.Module:
    """One convolution block: Conv1d, LayerNorm across channels, GELU."""
    padding = (kernel - 1) // 2
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding),
        LayerNormChannels(c_out),
        nn.GELU(),
    )


def _conv_transpose_block(c_in: int, c_out: int, kernel: int, stride: int) -> nn.Module:
    """Transposed convolution block for the decoder.

    With stride s and padding p = (k - 1) // 2, output_padding = s - 1
    doubles the length exactly for stride 2 and keeps it for stride 1.
    """
    padding = (kernel - 1) // 2
    output_padding = stride - 1
    return nn.Sequential(
        nn.ConvTranspose1d(c_in, c_out, kernel_size=kernel, stride=stride,
                           padding=padding, output_padding=output_padding),
        LayerNormChannels(c_out),
        nn.GELU(),
    )


class ConvVAE(nn.Module):
    """The convolutional VAE.

    Encoder ([ARCH §3.1]): Block_1 keeps T at base width C, Block_2 halves
    T at width 2C, Block_3 halves T again at 2C. Flatten, then two linear
    heads produce (mu, logvar).

    Decoder ([ARCH §3.2]): a linear lift reshapes z to (2C, T / 4), two
    transposed-convolution blocks return to (C, T), and a final Conv1d
    projects to the (3J, T) output.

    The `inpainting` flag adds Recipe 3's second decoder head. The
    shared decoder trunk feeds a `dec_output_full` that reconstructs the
    whole clip, and a `dec_output_inp` that concatenates the mask before
    projecting — the inpainting head is told which joints it must fill in.

    The `n_cond` flag turns the model into a CVAE ([CARE-PD §6]): a
    learned embedding e(c) of the conditioning id is concatenated to the
    encoder's flattened bottleneck (so the encoder can stop routing c into
    z) and to the latent at the decoder input (so the decoder reproduces
    c-specific artefacts from c directly). ``n_cond == 0`` leaves the plain
    VAE untouched — no extra modules, identical parameter count.
    """

    def __init__(self, T: int, J: int, d_z: int = 32, base_channels: int = 64,
                 kernels: tuple[int, int, int] = (5, 3, 3),
                 strides: tuple[int, int, int] = (1, 2, 2),
                 inpainting: bool = False,
                 n_cond: int = 0, cond_dim: int = 8,
                 cond_dropout: float = 0.0):
        super().__init__()
        self.T = T
        self.J = J
        self.d_z = d_z
        self.inpainting = inpainting

        C = base_channels
        in_channels = 4 * J
        out_channels = 3 * J
        downsample = strides[0] * strides[1] * strides[2]
        assert T % downsample == 0, "clip length must divide the total stride."

        # Conditioning ([CARE-PD §6]). Built only when requested so the
        # plain-VAE path keeps its exact parameter budget.
        self.n_cond = n_cond
        d_c = cond_dim if n_cond > 0 else 0
        self.cond = (ConditioningEmbedding(n_cond, cond_dim, cond_dropout)
                     if n_cond > 0 else None)

        # Encoder ([ARCH §3.1]).
        self.enc = nn.Sequential(
            _conv_block(in_channels, C,     kernels[0], strides[0]),
            _conv_block(C,           2 * C, kernels[1], strides[1]),
            _conv_block(2 * C,       2 * C, kernels[2], strides[2]),
        )
        self.T_bottleneck = T // downsample
        self.bottleneck_channels = 2 * C
        flat = self.bottleneck_channels * self.T_bottleneck
        # e(c) is concatenated to the pooled encoder features before the
        # posterior heads, and to z before the decoder lift.
        self.heads = BottleneckHeads(flat + d_c, d_z)

        # Decoder trunk ([ARCH §3.2]). The last block returns to width C.
        self.lift = nn.Linear(d_z + d_c, flat)
        self.dec_upsample = nn.Sequential(
            _conv_transpose_block(2 * C, 2 * C, kernels[2], strides[2]),
            _conv_transpose_block(2 * C, C,     kernels[1], strides[1]),
        )
        # Full-clip head. Not mask-conditioned; used by all three recipes.
        self.dec_output_full = nn.Conv1d(C, out_channels,
                                         kernel_size=kernels[0], stride=1,
                                         padding=(kernels[0] - 1) // 2)
        # Recipe 3 inpainting head. Mask-conditioned so the decoder is
        # told which joints it must fill in ([MVAE §5.1]).
        if inpainting:
            self.dec_output_inp = nn.Conv1d(C + J, out_channels,
                                            kernel_size=kernels[0], stride=1,
                                            padding=(kernels[0] - 1) // 2)

    # ---- Encoder ---------------------------------------------------------
    def encode(self, X, M, c=None):
        """Map (clip, mask[, cohort]) to (mu, logvar).

        Args:
            X: (B, T, J, 3).
            M: (B, T, J).
            c: optional (B,) conditioning ids; ignored for a plain VAE.
        Returns:
            (mu, logvar), each (B, d_z).
        """
        x = pack_encoder_input(X, M)                 # (B, T, 4J)
        h = self.enc(x.transpose(1, 2))              # (B, 2C, T / down)
        h = h.flatten(1)
        if self.cond is not None:
            e = self.cond.encoder_vector(c, h.shape[0], h.device)
            h = torch.cat([h, e], dim=1)
        return self.heads(h)

    # ---- Decoder ---------------------------------------------------------
    def _decode_trunk(self, z, c=None):
        B = z.shape[0]
        if self.cond is not None:
            e = self.cond.decoder_vector(c, B, z.device, self.training)
            z = torch.cat([z, e], dim=1)
        g = self.lift(z).view(B, self.bottleneck_channels, self.T_bottleneck)
        return self.dec_upsample(g)                  # (B, C, T)

    def decode_full(self, z, c=None):
        """Full-clip reconstruction head, ignoring the mask."""
        g = self._decode_trunk(z, c)
        x_hat = self.dec_output_full(g)              # (B, 3J, T)
        B = z.shape[0]
        return x_hat.transpose(1, 2).reshape(B, self.T, self.J, 3)

    def decode_inp(self, z, M, c=None):
        """Mask-conditioned inpainting head (Recipe 3 only)."""
        if not self.inpainting:
            raise RuntimeError("Model was built without the inpainting head.")
        g = self._decode_trunk(z, c)
        g = torch.cat([g, M.transpose(1, 2)], dim=1)  # (B, C + J, T)
        x_hat = self.dec_output_inp(g)               # (B, 3J, T)
        B = z.shape[0]
        return x_hat.transpose(1, 2).reshape(B, self.T, self.J, 3)

    # ---- Combined --------------------------------------------------------
    def forward(self, X, M, c=None):
        """Encode, sample, decode.

        Returns:
            (X_hat_full, mu, logvar) when built without the inpainting head.
            (X_hat_full, X_hat_inp, mu, logvar) for Recipe 3.
        """
        mu, logvar = self.encode(X, M, c)
        z = reparameterise(mu, logvar)
        X_hat_full = self.decode_full(z, c)
        if self.inpainting:
            return X_hat_full, self.decode_inp(z, M, c), mu, logvar
        return X_hat_full, mu, logvar
