"""Information measures on the latent (Part I Section 9, Part II Section 19).

The total-correlation decomposition splits the average divergence into
three readable parts. The active-unit count reads collapse. The
rate-distortion sweep sets the trade weight on principle rather than by
guess.
"""

from __future__ import annotations

import numpy as np


def _log_gaussian(z, mu, logvar):
    """Log density of a diagonal Gaussian, summed over dimensions.

    Broadcasts z (n, 1, d) against mu, logvar (1, m, d) to score every
    sample under every posterior.
    """
    c = -0.5 * np.log(2 * np.pi)
    return np.sum(c - 0.5 * logvar - 0.5 * (z - mu) ** 2 / np.exp(logvar), axis=-1)


def tc_decomposition(latent, batch: int = 1024,
                     rng: np.random.Generator | None = None) -> dict:
    """Total-correlation decomposition of the average divergence (Section 9.1).

    Splits E[ divergence(posterior || prior) ] into mutual information,
    total correlation (dimension coupling), and dimension-wise divergence,
    using the minibatch-weighted estimator of Chen et al. (2018). Total
    correlation (TC) is the part disentanglement work targets.

    Args:
        latent: a LatentSet.
        batch: sample count for the estimate.
    Returns:
        Dict with mutual_information, total_correlation, and
        dimension_wise, all in nats.
    """
    rng = np.random.default_rng() if rng is None else rng
    idx = rng.integers(0, latent.n, size=batch)
    mu, lv = latent.mu[idx], latent.logvar[idx]
    std = np.exp(0.5 * lv)
    z = mu + std * rng.standard_normal(mu.shape)     # (B, d)

    B, d = z.shape
    # log q(z | x_j) for the sampled j: diagonal of the score matrix.
    log_qz_x = _log_gaussian(z, mu, lv)              # (B,)

    # Score every z against every posterior for the aggregate estimate.
    zz = z[:, None, :]                               # (B, 1, d)
    mm = mu[None, :, :]                              # (1, B, d)
    ll = lv[None, :, :]
    # Per-dimension log densities: (B, B, d)
    c = -0.5 * np.log(2 * np.pi)
    logmat = c - 0.5 * ll - 0.5 * (zz - mm) ** 2 / np.exp(ll)

    logsumexp = lambda a, ax: (np.max(a, axis=ax) +
                               np.log(np.mean(np.exp(a - np.max(a, axis=ax, keepdims=True)), axis=ax)))

    # log q(z) marginal over the batch (all dims jointly).
    log_qz = logsumexp(logmat.sum(axis=-1), 1)       # (B,)
    # Sum over dims of log q(z_i) marginals.
    log_prod_qzi = logsumexp(logmat, 1).sum(axis=-1)  # (B,)
    # log prior p(z).
    log_pz = np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * z ** 2, axis=-1)

    mutual_info = np.mean(log_qz_x - log_qz)
    total_corr = np.mean(log_qz - log_prod_qzi)
    dim_wise = np.mean(log_prod_qzi - log_pz)
    return {"mutual_information": float(mutual_info),
            "total_correlation": float(total_corr),
            "dimension_wise": float(dim_wise)}


def active_units(latent, threshold: float = 0.01) -> dict:
    """Count active latent dimensions (Section 9.2).

    A dimension is active when the variance of its posterior mean across
    clips exceeds a small threshold. A low count against the latent width
    argues for a narrower latent.
    """
    var = np.var(latent.mu, axis=0)
    active = var > threshold
    return {"variance": var, "active": active, "n_active": int(active.sum()),
            "d_z": latent.d_z}


def rate_distortion_curve(records: list[dict]) -> dict:
    """Assemble a rate-distortion sweep from trained checkpoints (Section 19).

    You train a short sweep of the trade weight and pass one record per
    run. Each record needs the weight, the average divergence (the rate,
    in nats), and the average reconstruction cost (the distortion). This
    orders them and marks the knee: the point of greatest curvature, the
    efficient operating weight.

    Args:
        records: list of dicts with keys `beta`, `rate`, `distortion`.
    Returns:
        Dict with the sorted arrays and the knee index.
    """
    recs = sorted(records, key=lambda r: r["rate"])
    rate = np.array([r["rate"] for r in recs])
    dist = np.array([r["distortion"] for r in recs])
    beta = np.array([r["beta"] for r in recs])
    rate_bits = rate / np.log(2)

    knee = None
    if len(rate) >= 3:
        # Discrete curvature of the distortion-rate curve.
        d2 = dist[2:] - 2 * dist[1:-1] + dist[:-2]
        knee = int(np.argmax(np.abs(d2)) + 1)
    return {"beta": beta, "rate_nats": rate, "rate_bits": rate_bits,
            "distortion": dist, "knee_index": knee,
            "knee_beta": None if knee is None else float(beta[knee])}
