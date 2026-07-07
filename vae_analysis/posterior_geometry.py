"""Aggregate-posterior geometry (Part I, Section 3).

Three checks on the aggregate posterior q(z), the mixture of per-clip
posteriors. The evidence lower bound (ELBO) pushes each per-clip
posterior toward the prior, not the aggregate, so the aggregate can
drift and leave regions the prior covers but the encoder never visits.
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def _median_bandwidth(A: np.ndarray, B: np.ndarray) -> float:
    """Median pairwise distance over the pooled sample (the median heuristic)."""
    P = np.vstack([A, B])
    # Pairwise squared distances without forming the full N-by-N matrix twice.
    sq = np.sum(P * P, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (P @ P.T)
    d2 = np.clip(d2, 0.0, None)
    iu = np.triu_indices(len(P), k=1)
    med = np.median(np.sqrt(d2[iu]))
    return float(med) if med > 0 else 1.0


def _rbf(A: np.ndarray, B: np.ndarray, h: float) -> np.ndarray:
    sq_a = np.sum(A * A, axis=1)[:, None]
    sq_b = np.sum(B * B, axis=1)[None, :]
    d2 = np.clip(sq_a + sq_b - 2.0 * (A @ B.T), 0.0, None)
    return np.exp(-d2 / (2.0 * h * h))


def mmd2_unbiased(A: np.ndarray, B: np.ndarray, h: float | None = None) -> float:
    """Unbiased squared maximum mean discrepancy with a radial-basis kernel.

    The maximum mean discrepancy (MMD) measures the distance between two
    samples. Zero means the samples look identical to the kernel.

    Args:
        A, B: samples, shapes (M, d) and (M', d).
        h: kernel bandwidth; the median heuristic sets it when None.
    Returns:
        The unbiased estimate of MMD squared.
    """
    if h is None:
        h = _median_bandwidth(A, B)
    Kaa = _rbf(A, A, h)
    Kbb = _rbf(B, B, h)
    Kab = _rbf(A, B, h)
    m, n = len(A), len(B)
    np.fill_diagonal(Kaa, 0.0)
    np.fill_diagonal(Kbb, 0.0)
    term_a = Kaa.sum() / (m * (m - 1))
    term_b = Kbb.sum() / (n * (n - 1))
    term_ab = 2.0 * Kab.mean()
    return float(term_a + term_b - term_ab)


def mmd_prior_test(latent, n_samples: int = 2000, n_perm: int = 500,
                   rng: np.random.Generator | None = None) -> dict:
    """Test the aggregate posterior against the prior (Part I Section 3.1).

    Draws latents from the aggregate posterior and from the standard-normal
    prior, computes the maximum mean discrepancy, and reads a p-value from a
    label-permutation null.

    Args:
        latent: a LatentSet.
        n_samples: sample size drawn from each side.
        n_perm: permutation count for the null.
    Returns:
        Dict with the statistic, the p-value, and the bandwidth.
    """
    rng = np.random.default_rng() if rng is None else rng
    idx = rng.integers(0, latent.n, size=n_samples)
    std = np.exp(0.5 * latent.logvar[idx])
    q = latent.mu[idx] + std * rng.standard_normal((n_samples, latent.d_z))
    p = latent.prior_like(n_samples, rng)

    h = _median_bandwidth(q, p)
    stat = mmd2_unbiased(q, p, h)

    pooled = np.vstack([q, p])
    labels = np.array([0] * n_samples + [1] * n_samples)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        a = pooled[perm == 0]
        b = pooled[perm == 1]
        if mmd2_unbiased(a, b, h) >= stat:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {"mmd2": stat, "p_value": p_value, "bandwidth": h}


def intrinsic_dimension_twonn(points: np.ndarray,
                              discard_fraction: float = 0.1) -> dict:
    """Estimate intrinsic dimension by the two-nearest-neighbour method.

    For each point, the ratio of its second-nearest to nearest-neighbour
    distance follows a Pareto(d) law whose shape is the intrinsic
    dimension (Facco et al., 2017). The MLE is M / sum_i log(mu_i), and
    the asymptotic Fisher information gives standard error d_hat/sqrt(M).

    Args:
        points: sample, shape (M, d).
        discard_fraction: drop this top fraction of the sorted log-ratios
            to blunt cluster-hopping outliers (Denti et al., 2022).
    Returns:
        Dict with `d_hat`, `standard_error`, `n_used`, and the raw
        `mu` ratios (for the scale sweep and diagnostic plots).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3).fit(points)
    dist, _ = nn.kneighbors(points)
    r1 = dist[:, 1]
    r2 = dist[:, 2]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    logmu = np.sort(np.log(mu))
    keep = int(len(logmu) * (1.0 - discard_fraction))
    logmu_kept = logmu[:keep]
    d_hat = float(len(logmu_kept) / logmu_kept.sum())
    return {
        "d_hat": d_hat,
        "standard_error": d_hat / float(np.sqrt(len(logmu_kept))),
        "n_used": int(len(logmu_kept)),
        "mu": mu,
    }


def cluster_structure(latent, k_range=range(2, 13),
                      rng_seed: int = 0) -> dict:
    """Fit a Gaussian mixture on the posterior means and pick K by BIC.

    The Bayesian information criterion (BIC) trades fit against component
    count. Returns the chosen model, the per-video composition, and the
    scores for every K tried.

    Args:
        latent: a LatentSet; uses `mu` and, if present, `video_id`.
        k_range: candidate component counts.
    Returns:
        Dict with the fitted mixture, the chosen K, the BIC curve, and
        (when video labels exist) the composition matrix Pi[v, k].
    """
    bics = {}
    best, best_bic = None, np.inf
    for k in k_range:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=rng_seed).fit(latent.mu)
        b = gm.bic(latent.mu)
        bics[k] = b
        if b < best_bic:
            best, best_bic = gm, b

    out = {"model": best, "k": best.n_components, "bic": bics}
    if latent.video_id is not None:
        labels = best.predict(latent.mu)
        vids = np.unique(latent.video_id)
        Pi = np.zeros((len(vids), best.n_components))
        for i, v in enumerate(vids):
            rows = labels[latent.video_id == v]
            for k in range(best.n_components):
                Pi[i, k] = np.mean(rows == k)
        out["composition"] = Pi
        out["video_ids"] = vids
    return out
