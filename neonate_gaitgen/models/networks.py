"""Encoders, decoder, and classifiers ([paper Sec. 3.1.1]).

Two 1D-conv ResNet encoders (motion / pathology) downsampling time by 4, a
mirror decoder, and two self-attention-pooled MLP classifiers — a pathology
classifier on ``q_p`` and an adversarial one on ``q_m`` behind a gradient
reversal layer.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .layers import ConvResBlock, SelfAttnPool, gradient_reverse


def _n_stride_stages(downsample: int) -> int:
    n = int(round(math.log2(downsample)))
    if 2 ** n != downsample:
        raise ValueError(f"downsample ({downsample}) must be a power of 2.")
    return n


class _ConvStack(nn.Module):
    """Conv stem → `n` stride-2 downsampling stages → projection to `d_out`."""

    def __init__(self, in_dim: int, hidden: int, d_out: int, downsample: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 3, padding=1), nn.GELU())
        stages = []
        for _ in range(_n_stride_stages(downsample)):
            stages += [nn.Conv1d(hidden, hidden, 4, stride=2, padding=1),
                       nn.GELU(), ConvResBlock(hidden)]
        self.down = nn.Sequential(*stages)
        self.out = nn.Conv1d(hidden, d_out, 3, padding=1)

    def forward(self, x):                      # x: (B, C_in, T)
        return self.out(self.down(self.stem(x)))   # (B, d_out, T')


class MotionEncoder(nn.Module):
    """E_m: unconditional, pathology-invariant motion latent z_m ([paper])."""

    def __init__(self, in_dim: int, hidden: int, d_out: int, downsample: int):
        super().__init__()
        self.net = _ConvStack(in_dim, hidden, d_out, downsample)

    def forward(self, x):                      # x: (B, T, C_in)
        z = self.net(x.transpose(1, 2))        # (B, d_out, T')
        return z.transpose(1, 2)               # (B, T', d_out)


class PathologyEncoder(nn.Module):
    """E_p: conditional on c_p, pathology-specific latent z_p ([paper]).

    The class embedding e(c_p) is broadcast across time and concatenated to
    the input channels, so pathology structure separates by class.
    """

    def __init__(self, in_dim: int, n_classes: int, cond_dim: int,
                 hidden: int, d_out: int, downsample: int):
        super().__init__()
        self.embed = nn.Embedding(n_classes, cond_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.net = _ConvStack(in_dim + cond_dim, hidden, d_out, downsample)

    def forward(self, x, c_p):                 # x: (B, T, C_in), c_p: (B,)
        B, T, _ = x.shape
        e = self.embed(c_p).unsqueeze(1).expand(B, T, -1)   # (B, T, cond_dim)
        xc = torch.cat([x, e], dim=-1)                       # (B, T, C_in+cond)
        z = self.net(xc.transpose(1, 2))
        return z.transpose(1, 2)               # (B, T', d_out)


class Decoder(nn.Module):
    """Mirror of the encoders: (B, T', D) -> (B, T, C_in) ([paper])."""

    def __init__(self, out_dim: int, hidden: int, d_in: int, downsample: int):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(d_in, hidden, 3, padding=1),
                                 nn.GELU())
        stages = []
        for _ in range(_n_stride_stages(downsample)):
            stages += [nn.ConvTranspose1d(hidden, hidden, 4, stride=2,
                                          padding=1), nn.GELU(),
                       ConvResBlock(hidden)]
        self.up = nn.Sequential(*stages)
        self.out = nn.Conv1d(hidden, out_dim, 3, padding=1)

    def forward(self, q):                      # q: (B, T', D)
        h = self.up(self.inp(q.transpose(1, 2)))
        return self.out(h).transpose(1, 2)     # (B, T, C_in)


class PathologyClassifier(nn.Module):
    """phi_p: predicts c_p from q_p ([paper Eq. 5]). Drives L_cls."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.pool = SelfAttnPool(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_classes))

    def forward(self, q):                      # q: (B, T', D)
        return self.mlp(self.pool(q))          # (B, n_classes)


class AdversarialClassifier(nn.Module):
    """psi_m: predicts c_p from q_m behind a GRL ([paper Eq. 6]).

    The gradient-reversal is placed at the latent input (DANN-standard): the
    adversary's own pool+MLP train normally to classify, while the reversed
    gradient flows into q_m / the motion encoder, driving pathology
    invariance. Equivalent in intent to the paper's ``psi_m(GRL(f_m))``.
    """

    def __init__(self, dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.pool = SelfAttnPool(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_classes))

    def forward(self, q, lambd: float):        # q: (B, T', D)
        return self.mlp(self.pool(gradient_reverse(q, lambd)))
