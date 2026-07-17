"""Kernel independence tests between latents and labels ([plan §6.5]).

HSIC (Hilbert-Schmidt Independence Criterion) with RBF kernels and the
median-distance heuristic, plus a permutation-test p-value. Used for:

    HSIC(q_m, c_p)   — should be small (motion latent free of pathology),
    HSIC(q_p, c_nuis)— should be small (pathology latent free of nuisance),
    HSIC(q_m, q_p)   — should be small (the two latents independent).
"""

from __future__ import annotations

import numpy as np


def _rbf(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    sq = np.sum(x * x, axis=1)
    d2 = np.clip(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0, None)
    if sigma is None:
        iu = np.triu_indices(len(x), k=1)
        med = np.median(np.sqrt(d2[iu])) if len(iu[0]) else 1.0
        sigma = med if med > 0 else 1.0
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _onehot(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    _, inv = np.unique(y.astype(str), return_inverse=True)
    oh = np.zeros((len(inv), inv.max() + 1))
    oh[np.arange(len(inv)), inv] = 1.0
    return oh


def hsic(x: np.ndarray, y: np.ndarray, y_is_categorical: bool = False) -> float:
    """Biased HSIC estimate between ``x`` and ``y`` (RBF kernels)."""
    n = len(x)
    K = _rbf(x)
    L = _rbf(_onehot(y)) if y_is_categorical else _rbf(y)
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def hsic_test(x: np.ndarray, y: np.ndarray, y_is_categorical: bool = False,
              n_perm: int = 200, max_n: int = 1500, seed: int = 0) -> dict:
    """HSIC with a permutation-test p-value ([plan §6.5]).

    Subsamples to ``max_n`` points for tractability. Returns
    ``{"hsic", "p_value", "n"}``.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        x, y = x[idx], np.asarray(y)[idx]
        n = max_n
    stat = hsic(x, y, y_is_categorical)
    count = 0
    yv = np.asarray(y)
    for _ in range(n_perm):
        if hsic(x, yv[rng.permutation(n)], y_is_categorical) >= stat:
            count += 1
    return {"hsic": stat, "p_value": (count + 1) / (n_perm + 1), "n": int(n)}
