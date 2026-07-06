"""1D temporal convolutional VAE ([ARCH §3]).

Three residual blocks widen the channel count and halve T twice. The
decoder mirrors the encoder with transposed convolutions. An optional
inpainting head takes the mask as a decoder input, for Recipe 3.
"""

from __future__ import annotations

from .common import (torch, nn, LayerNormChannels, BottleneckHeads,
                     pack_encoder_input, reparameterise)


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

    The `inpainting` flag toggles Recipe 3's decoder head, which
    concatenates the mask before the output projection.
    """

    def __init__(self, T: int, J: int, d_z: int = 32, base_channels: int = 64,
                 kernels: tuple[int, int, int] = (5, 3, 3),
                 strides: tuple[int, int, int] = (1, 2, 2),
                 inpainting: bool = False):
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

        # Encoder ([ARCH §3.1]).
        self.enc = nn.Sequential(
            _conv_block(in_channels, C,     kernels[0], strides[0]),
            _conv_block(C,           2 * C, kernels[1], strides[1]),
            _conv_block(2 * C,       2 * C, kernels[2], strides[2]),
        )
        self.T_bottleneck = T // downsample
        self.bottleneck_channels = 2 * C
        flat = self.bottleneck_channels * self.T_bottleneck
        self.heads = BottleneckHeads(flat, d_z)

        # Decoder ([ARCH §3.2]). The last block returns to width C, then
        # a final Conv1d projects to 3J output channels.
        self.lift = nn.Linear(d_z, flat)
        self.dec_upsample = nn.Sequential(
            _conv_transpose_block(2 * C, 2 * C, kernels[2], strides[2]),
            _conv_transpose_block(2 * C, C,     kernels[1], strides[1]),
        )
        # For Recipe 3 the last layer sees the mask as J extra channels.
        final_in = C + (J if inpainting else 0)
        self.dec_output = nn.Conv1d(final_in, out_channels,
                                    kernel_size=kernels[0], stride=1,
                                    padding=(kernels[0] - 1) // 2)

    # ---- Encoder ---------------------------------------------------------
    def encode(self, X, M):
        """Map (clip, mask) to (mu, logvar).

        Args:
            X: (B, T, J, 3).
            M: (B, T, J).
        Returns:
            (mu, logvar), each (B, d_z).
        """
        x = pack_encoder_input(X, M)                 # (B, T, 4J)
        h = self.enc(x.transpose(1, 2))              # (B, 2C, T / down)
        return self.heads(h.flatten(1))

    # ---- Decoder ---------------------------------------------------------
    def decode(self, z, M=None):
        """Map z to reconstructed clip.

        Args:
            z: (B, d_z).
            M: (B, T, J) if `inpainting`; ignored otherwise.
        Returns:
            X_hat, (B, T, J, 3).
        """
        B = z.shape[0]
        g = self.lift(z).view(B, self.bottleneck_channels, self.T_bottleneck)
        g = self.dec_upsample(g)                     # (B, C, T)
        if self.inpainting:
            if M is None:
                raise ValueError("Inpainting decoder needs the mask.")
            g = torch.cat([g, M.transpose(1, 2)], dim=1)  # (B, C + J, T)
        x_hat = self.dec_output(g)                   # (B, 3J, T)
        return x_hat.transpose(1, 2).reshape(B, self.T, self.J, 3)

    # ---- Combined --------------------------------------------------------
    def forward(self, X, M):
        mu, logvar = self.encode(X, M)
        z = reparameterise(mu, logvar)
        X_hat = self.decode(z, M if self.inpainting else None)
        return X_hat, mu, logvar
