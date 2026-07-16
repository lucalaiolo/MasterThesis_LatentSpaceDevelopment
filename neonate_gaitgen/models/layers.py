"""Shared layers for the disentangled RVQ-VAE.

The gradient-reversal layer (GRL, [paper Eq. 6, ref. 18]), a self-attention
temporal pool used by the classifiers ([paper Sec. 3.1.1] "self-attention
block to aggregate channel information"), and a 1D residual conv block for
the encoders / decoder.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


# ---- Gradient reversal ([paper Eq. 6]) ------------------------------------

class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def gradient_reverse(x, lambd: float = 1.0):
    """Identity forward; gradient multiplied by ``-lambd`` on the backward."""
    return _GradReverse.apply(x, lambd)


# ---- Self-attention temporal pool -----------------------------------------

class SelfAttnPool(nn.Module):
    """Aggregate a (B, T', D) latent to (B, D) by learned attention over time.

    A single-head additive-attention pool: scores per time step, softmax,
    weighted sum. Stands in for the paper's "self-attention block" that
    aggregates the quantised latent before the classifier MLP.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                      # x: (B, T', D)
        w = torch.softmax(self.score(x), dim=1)   # (B, T', 1)
        return (w * x).sum(dim=1)                  # (B, D)


# ---- 1D residual conv block ------------------------------------------------

class ConvResBlock(nn.Module):
    """Residual block: two Conv1d + GroupNorm + GELU on a (B, C, T) tensor."""

    def __init__(self, channels: int, kernel: int = 3, groups: int = 8):
        super().__init__()
        pad = kernel // 2
        g = max(1, min(groups, channels))
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad),
            nn.GroupNorm(g, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel, padding=pad),
            nn.GroupNorm(g, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))
