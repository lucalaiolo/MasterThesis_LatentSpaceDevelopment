"""Shared model helpers.

Small pieces used by both the convolutional and the transformer VAE. The
LayerNorm helper normalises across the channel axis of a (B, C, T)
tensor, which the design note [ARCH §3.1] calls for.
"""

from __future__ import annotations

import math


def _import_torch():
    try:
        import torch
        from torch import nn
        return torch, nn
    except ImportError as e:
        raise ImportError("This module needs PyTorch. Install torch to train.") from e


torch, nn = _import_torch()


class LayerNormChannels(nn.Module):
    """LayerNorm across the channel axis of a (B, C, T) tensor.

    PyTorch's `nn.LayerNorm(C)` normalises the last dimension, so for a
    channels-first tensor we transpose, normalise, and transpose back.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def sinusoidal_positional_encoding(length: int, d_model: int):
    """The fixed sinusoidal positional encoding of Vaswani et al. (2017).

    Args:
        length: sequence length. For the encoder this is T + 1 to cover
            the class token; for the decoder it is T.
        d_model: model width.
    Returns:
        Tensor of shape (length, d_model), the same on every forward pass.
    """
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def reparameterise(mu, logvar):
    """The reparameterisation trick: z = mu + sigma * eps, eps ~ N(0, I)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def pack_encoder_input(X, M):
    """Reshape (X, M) to the encoder input tensor.

    Combines the masked pose with an explicit mask channel, following
    [ARCH §2.1, §2.2]. The mask nulls the hidden joints and, on top of
    that, appears as its own channel so the encoder can tell a hidden
    joint apart from a real joint at the origin.

    Args:
        X: (B, T, J, 3).
        M: (B, T, J), 1 for visible.
    Returns:
        (B, T, 4J).
    """
    B, T, J, _ = X.shape
    X_masked = X * M.unsqueeze(-1)                 # (B, T, J, 3)
    X_flat = X_masked.reshape(B, T, J * 3)         # (B, T, 3J)
    return torch.cat([X_flat, M], dim=-1)          # (B, T, 4J)


class BottleneckHeads(nn.Module):
    """Two linear heads mapping a feature vector to mu and log-variance.

    Both models emit a feature `h` at the top of the encoder. Two linear
    layers turn it into the posterior parameters ([ARCH §2.3]).
    """

    def __init__(self, in_dim: int, d_z: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, d_z)
        self.logvar = nn.Linear(in_dim, d_z)

    def forward(self, h):
        return self.mu(h), self.logvar(h)
