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


class ConditioningEmbedding(nn.Module):
    """Learned embedding e(c) of a discrete conditioning variable ([CARE-PD §6]).

    Maps an integer id in ``[0, n_cond)`` to a ``cond_dim`` vector that is
    concatenated into both the encoder's pooled representation and the
    decoder's latent input. The embedding is shared across the two
    injection points so ``c`` means the same thing on both sides.

    Conditioning dropout ([CARE-PD §6, §10]) is applied *only on the
    decoder path* and *only in training*: with probability ``dropout`` the
    whole embedding is zeroed for a sample, forcing the decoder to keep
    reconstructing from ``z`` alone rather than degenerating into one
    sub-decoder per cohort. The encoder always sees the clean embedding so
    it can learn to *stop* routing cohort information into ``z``.
    """

    def __init__(self, n_cond: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n_cond = n_cond
        self.cond_dim = cond_dim
        self.dropout = dropout
        self.embed = nn.Embedding(n_cond, cond_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def _resolve(self, c, batch_size: int, device):
        """Return a valid long tensor of ids, defaulting to zeros.

        ``c`` may be None (analysis / traversal code that decodes without a
        cohort) — then the zero id is used, matching the dropped-embedding
        signal the decoder already learned to tolerate.
        """
        if c is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not torch.is_tensor(c):
            c = torch.as_tensor(c, device=device)
        return c.to(device=device, dtype=torch.long).reshape(batch_size)

    def encoder_vector(self, c, batch_size: int, device):
        """Clean embedding for the encoder path (no dropout)."""
        ids = self._resolve(c, batch_size, device)
        return self.embed(ids)

    def decoder_vector(self, c, batch_size: int, device, training: bool):
        """Embedding for the decoder path, with conditioning dropout."""
        ids = self._resolve(c, batch_size, device)
        e = self.embed(ids)
        if training and self.dropout > 0.0:
            keep = (torch.rand(batch_size, device=device) >= self.dropout)
            e = e * keep.unsqueeze(1).to(e.dtype)
        return e
