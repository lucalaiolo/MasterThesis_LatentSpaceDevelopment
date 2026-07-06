"""Disentanglement with proxy factors (Part II Section 16).

We have no ground-truth generative factors, so the kinematic features
stand in as proxy factors. Three scores measure whether each factor maps
to one latent dimension or smears across many, and a control guards
against reading structure that is not there.
"""

from __future__ import annotations

import numpy as np


def _mutual_information(z_i: np.ndarray, phi_k: np.ndarray, n_bins: int = 20) -> float:
    """Mutual information between one latent and one factor by binning.

    A histogram estimate; coarse but dependency-free. For a sharper number
    swap in a k-nearest-neighbour estimator.
    """
    hist, _, _ = np.histogram2d(z_i, phi_k, bins=n_bins)
    p = hist / hist.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / (px @ py)[nz])))


def _entropy(phi_k: np.ndarray, n_bins: int = 20) -> float:
    hist, _ = np.histogram(phi_k, bins=n_bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def mig(latent, features: np.ndarray) -> dict:
    """Mutual Information Gap over the proxy factors (Section 16.1).

    For each factor, the gap between the top two latent dimensions by
    mutual information, normalised by the factor entropy. High means one
    dominant dimension per factor.

    Args:
        latent: a LatentSet.
        features: proxy factors, shape (N, F).
    Returns:
        Dict with the per-factor gap and the mean.
    """
    d_z = latent.d_z
    F = features.shape[1]
    gaps = np.zeros(F)
    for k in range(F):
        mis = np.array([_mutual_information(latent.mu[:, i], features[:, k])
                        for i in range(d_z)])
        order = np.sort(mis)[::-1]
        H = _entropy(features[:, k]) + 1e-12
        gaps[k] = (order[0] - order[1]) / H
    return {"per_factor": gaps, "mig": float(gaps.mean())}


def dci(latent, features: np.ndarray, test_fraction: float = 0.2) -> dict:
    """Disentanglement, completeness, and informativeness (Section 16.2).

    Fits a gradient-boosted regressor from the latent to each factor,
    reads the per-dimension importances, and turns them into a
    disentanglement score per dimension and a completeness score per
    factor (Eastwood and Williams, 2018).

    Args:
        latent: a LatentSet.
        features: proxy factors, shape (N, F).
    Returns:
        Dict with mean disentanglement, mean completeness, and the
        held-out error per factor (the informativeness).
    """
    from sklearn.ensemble import GradientBoostingRegressor

    N = latent.n
    cut = int(N * (1 - test_fraction))
    tr, te = slice(0, cut), slice(cut, N)
    d_z, F = latent.d_z, features.shape[1]

    importance = np.zeros((d_z, F))
    errors = np.zeros(F)
    for k in range(F):
        reg = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        reg.fit(latent.mu[tr], features[tr, k])
        importance[:, k] = reg.feature_importances_
        pred = reg.predict(latent.mu[te])
        errors[k] = np.sqrt(np.mean((features[te, k] - pred) ** 2))

    def norm_entropy(p, base):
        p = p / (p.sum() + 1e-12)
        p = p[p > 0]
        return -np.sum(p * np.log(p)) / np.log(base) if base > 1 else 0.0

    D = np.array([1 - norm_entropy(importance[i], F) for i in range(d_z)])
    C = np.array([1 - norm_entropy(importance[:, k], d_z) for k in range(F)])
    weight = importance.sum(axis=1) / (importance.sum() + 1e-12)
    return {"disentanglement": float(np.sum(weight * D)),
            "completeness": float(C.mean()),
            "informativeness_rmse": errors,
            "importance": importance}


def sap(latent, features: np.ndarray, test_fraction: float = 0.2) -> dict:
    """Separated Attribute Predictability (Section 16.3).

    For each factor, the gap in single-dimension predictive score between
    the best and second-best latent dimension. Reported alongside the
    Mutual Information Gap, since the two disagree when the information
    estimate is noisy.
    """
    from sklearn.linear_model import LinearRegression

    N = latent.n
    cut = int(N * (1 - test_fraction))
    tr, te = slice(0, cut), slice(cut, N)
    d_z, F = latent.d_z, features.shape[1]

    gaps = np.zeros(F)
    for k in range(F):
        scores = np.zeros(d_z)
        for i in range(d_z):
            reg = LinearRegression().fit(latent.mu[tr, i:i + 1], features[tr, k])
            pred = reg.predict(latent.mu[te, i:i + 1])
            ss_res = np.sum((features[te, k] - pred) ** 2)
            ss_tot = np.sum((features[te, k] - features[te, k].mean()) ** 2) + 1e-12
            scores[i] = 1 - ss_res / ss_tot
        order = np.sort(scores)[::-1]
        gaps[k] = order[0] - order[1]
    return {"per_factor": gaps, "sap": float(gaps.mean())}


def selectivity(latent, features: np.ndarray, states: np.ndarray,
                score_fn=mig, rng: np.random.Generator | None = None) -> dict:
    """Probe selectivity against a control task (Section 16.4).

    Builds a control by shuffling each factor within behavioural states,
    which kills the true relation but keeps the marginal, then reports the
    real score minus the control score. Only a large selectivity licenses
    a claim that the latent codes the factor.

    Args:
        latent: a LatentSet.
        features: proxy factors, shape (N, F).
        states: a state label per clip, shape (N,), for the within-state
            shuffle. Pass an all-zero array to shuffle globally.
        score_fn: mig or sap.
    Returns:
        Dict with the real score, the control score, and their difference.
    """
    rng = np.random.default_rng() if rng is None else rng
    real = score_fn(latent, features)
    real_val = real.get("mig", real.get("sap"))

    control = features.copy()
    for s in np.unique(states):
        idx = np.where(states == s)[0]
        for k in range(features.shape[1]):
            control[idx, k] = features[rng.permutation(idx), k]
    ctrl = score_fn(latent, control)
    ctrl_val = ctrl.get("mig", ctrl.get("sap"))
    return {"real": real_val, "control": ctrl_val,
            "selectivity": float(real_val - ctrl_val)}
