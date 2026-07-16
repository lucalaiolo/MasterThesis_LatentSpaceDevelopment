"""Residual Vector Quantization ([paper Sec. 3.1.1, Eqs. 1-2, 8]).

A stack of EMA-updated vector quantizers applied to the residuals between
the input and the running quantized sum, so coarse-to-fine gait structure
is captured across layers. Codebooks update by exponential moving average
with dead-code reset ([paper ref. 81]); quantization dropout truncates the
residual depth during training so the lower layers stay useful.
"""

from __future__ import annotations

import torch
from torch import nn


class VectorQuantizerEMA(nn.Module):
    """One EMA codebook. Returns the (non-straight-through) code lookup.

    The codebook is a buffer (never receives gradient); it is moved toward
    the encoder outputs by EMA. The commitment term ``||z - sg[e]||^2`` (the
    per-layer piece of paper Eq. 8) is returned for the encoder's loss.
    """

    def __init__(self, num_codes: int, dim: int, decay: float = 0.99,
                 eps: float = 1e-5, reset_threshold: float = 1.0):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.reset_threshold = reset_threshold

        codebook = torch.randn(num_codes, dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", codebook.clone())
        self.register_buffer("_initted", torch.tensor(False))

    def _nearest(self, x):
        # x: (M, D) -> indices (M,) of the nearest code.
        d = (x.pow(2).sum(1, keepdim=True)
             - 2 * x @ self.codebook.t()
             + self.codebook.pow(2).sum(1))
        return d.argmin(1)

    @torch.no_grad()
    def _ema_update(self, x, idx):
        onehot = torch.zeros(x.shape[0], self.num_codes, device=x.device)
        onehot.scatter_(1, idx.unsqueeze(1), 1)
        n = onehot.sum(0)                                   # (K,)
        embed_sum = onehot.t() @ x                          # (K, D)

        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        total = self.cluster_size.sum()
        smoothed = ((self.cluster_size + self.eps)
                    / (total + self.num_codes * self.eps) * total)
        self.codebook.copy_(self.embed_avg / smoothed.unsqueeze(1))

        # Dead-code reset: revive codes barely used, to random batch vectors.
        dead = self.cluster_size < self.reset_threshold
        if dead.any() and x.shape[0] > 0:
            pick = torch.randint(0, x.shape[0], (int(dead.sum()),),
                                 device=x.device)
            self.codebook[dead] = x[pick]
            self.embed_avg[dead] = x[pick]
            self.cluster_size[dead] = 1.0

    def forward(self, x):
        """Quantize (M, D) -> (quantized (M, D), indices (M,), commit_loss)."""
        if self.training and not bool(self._initted) and x.shape[0] >= self.num_codes:
            # Data-driven init: seed codes from the first batch's vectors.
            pick = torch.randperm(x.shape[0], device=x.device)[:self.num_codes]
            self.codebook.copy_(x[pick].detach())
            self.embed_avg.copy_(x[pick].detach())
            self._initted.fill_(True)

        idx = self._nearest(x)
        quantized = self.codebook[idx]                      # (M, D), no grad
        commit = (x - quantized.detach()).pow(2).mean()
        if self.training:
            self._ema_update(x.detach(), idx)
        return quantized, idx, commit

    def usage(self) -> float:
        """Fraction of codes currently active (cluster_size above eps)."""
        return float((self.cluster_size > self.eps).float().mean())


class ResidualVQ(nn.Module):
    """A stack of :class:`VectorQuantizerEMA` over successive residuals.

    ``forward(z)`` returns the straight-through quantized sum ``q`` (paper
    ``q_k = sum_n e_k^(n)``), the per-layer code indices, the summed
    commitment loss, and per-layer codebook usage.
    """

    def __init__(self, n_layers: int, num_codes: int, dim: int,
                 decay: float = 0.99, quant_dropout: float = 0.0,
                 reset_threshold: float = 1.0):
        super().__init__()
        self.n_layers = n_layers
        self.quant_dropout = quant_dropout
        self.layers = nn.ModuleList([
            VectorQuantizerEMA(num_codes, dim, decay=decay,
                               reset_threshold=reset_threshold)
            for _ in range(n_layers)])

    def _n_active(self) -> int:
        if self.training and self.quant_dropout > 0 and \
                torch.rand(()).item() < self.quant_dropout:
            # Truncate to a random depth so lower layers stay informative.
            return int(torch.randint(1, self.n_layers + 1, ()).item())
        return self.n_layers

    def forward(self, z):
        """z: (B, T', D). Returns (q, indices (B, T', N), commit, usage)."""
        B, T, D = z.shape
        flat = z.reshape(-1, D)                             # (B*T', D)
        residual = flat
        q_sum = torch.zeros_like(flat)
        commit = z.new_zeros(())
        idx_layers = []
        n_active = self._n_active()

        for n, layer in enumerate(self.layers):
            if n < n_active:
                q, idx, c = layer(residual)
                residual = residual - q
                q_sum = q_sum + q
                commit = commit + c
            else:
                idx = flat.new_zeros(flat.shape[0], dtype=torch.long)
            idx_layers.append(idx)

        # Straight-through: value q_sum, gradient flows to z as identity.
        q_st = flat + (q_sum - flat).detach()
        q_st = q_st.reshape(B, T, D)
        indices = torch.stack(idx_layers, dim=-1).reshape(B, T, self.n_layers)
        usage = [lyr.usage() for lyr in self.layers]
        return q_st, indices, commit, usage

    @torch.no_grad()
    def quantize_from_z(self, z):
        """Deterministic full-depth quantize (no dropout), for analysis."""
        was_training = self.training
        self.eval()
        q, indices, _, _ = self.forward(z)
        self.train(was_training)
        return q, indices
