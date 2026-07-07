"""Generation-quality checks (Part I Section 8).

Decoded prior samples test whether the model learned a usable generative
distribution. These score bone rigidity, the match to real motion in
feature space, and the smoothness of latent interpolation.
"""

from __future__ import annotations

import numpy as np


def bone_length_cv(X: np.ndarray, skeleton) -> np.ndarray:
    """Coefficient of variation of every bone length across a clip (Section 8.1).

    Bones are rigid, so real motion holds each length near constant. The
    coefficient of variation is the per-bone standard deviation over its
    mean. Compare generated to real by the ratio of these.

    Args:
        X: clips, shape (N, T, J, 3).
        skeleton: a Skeleton with bones.
    Returns:
        Array of shape (N, n_bones): the coefficient of variation per bone.
    """
    bones = skeleton.bone_index()
    a = X[:, :, bones[:, 0], :]
    b = X[:, :, bones[:, 1], :]
    length = np.linalg.norm(a - b, axis=-1)      # (N, T, n_bones)
    mean = length.mean(axis=1)
    std = length.std(axis=1)
    return std / (mean + 1e-12)


def bone_plausibility(gen: np.ndarray, real: np.ndarray, skeleton) -> dict:
    """Per-bone plausibility as the ratio of generated to real variation.

    Values near one say the decoder holds bones rigid; large values say it
    stretches them.
    """
    cv_gen = bone_length_cv(gen, skeleton).mean(axis=0)
    cv_real = bone_length_cv(real, skeleton).mean(axis=0)
    return {"cv_generated": cv_gen, "cv_real": cv_real,
            "ratio": cv_gen / (cv_real + 1e-12)}


def frechet_distance(feat_real: np.ndarray, feat_gen: np.ndarray) -> float:
    """Frechet distance between two feature distributions (Section 8.2).

    Treats each set of feature vectors as a Gaussian and returns the
    closed-form distance between them. Compute the features (from the
    kinematic set) on real and on generated clips.

    Args:
        feat_real, feat_gen: shapes (N, F) and (N', F).
    Returns:
        The Frechet distance.
    """
    from scipy.linalg import sqrtm

    mu_r, mu_g = feat_real.mean(0), feat_gen.mean(0)
    cov_r = np.cov(feat_real, rowvar=False)
    cov_g = np.cov(feat_gen, rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))


def interpolation_curvature(model, latent, n_pairs: int = 100, n_steps: int = 32,
                            rng: np.random.Generator | None = None) -> float:
    """Mean curvature of straight-line latent interpolations (Section 8.3).

    Decodes along the line between random posterior-mean pairs and sums
    the norm of the pose sequence's second difference. Lower is smoother.
    The pullback metric predicts where the rough stretches sit.
    """
    rng = np.random.default_rng() if rng is None else rng
    total = 0.0
    for _ in range(n_pairs):
        i, j = rng.integers(0, latent.n, size=2)
        ts = np.linspace(0, 1, n_steps)[:, None]
        path = (1 - ts) * latent.mu[i][None] + ts * latent.mu[j][None]
        dec = model.decode(path)                       # (n_steps, T, J, 3)
        second = dec[2:] - 2 * dec[1:-1] + dec[:-2]
        total += np.sqrt((second ** 2).sum(axis=(1, 2, 3))).sum()
    return float(total / n_pairs)
