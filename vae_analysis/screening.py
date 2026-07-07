"""Screening and attention (Part II Sections 21 and 23).

Density-based screening ranks clips by typicality for a clinician to
read, with heavy caveats on the small sample. Attention analysis
diagnoses whether the transformer's attention buys anything over the
convolutional model.
"""

from __future__ import annotations

import numpy as np


def fit_density(latent, method: str = "gmm", n_components: int = 8,
                seed: int = 0):
    """Fit a density to the posterior means (Section 21).

    Args:
        latent: a LatentSet.
        method: "gmm" for a Gaussian mixture, or "kde" for a kernel
            density estimate. Use a mixture unless the sample is tiny.
        n_components: mixture component count.
    Returns:
        A fitted density with a `score_samples` method.
    """
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(n_components=n_components,
                               covariance_type="full",
                               random_state=seed).fit(latent.mu)
    if method == "kde":
        from sklearn.neighbors import KernelDensity
        bw = np.std(latent.mu) * len(latent.mu) ** (-1.0 / (latent.d_z + 4))
        return KernelDensity(bandwidth=bw).fit(latent.mu)
    raise ValueError("method must be 'gmm' or 'kde'.")


def typicality_score(density, latent) -> np.ndarray:
    """Log-density of every clip under the fitted density (Section 21).

    A low score marks a clip as unlike the training set. This is unlike
    the training set, not abnormal: with a handful of infants the training
    set is not the population, so treat the score as a research signal,
    not a diagnosis. Fix one mask policy for every clip you score.

    Returns:
        Score per clip, shape (N,).
    """
    return density.score_samples(latent.mu)


def screening_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the curve of the typicality score against clinical labels.

    Only meaningful when labels exist. Lower typicality should mark the
    positive class, so the score is negated before scoring.
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, -scores))


def attention_entropy(attention: np.ndarray) -> dict:
    """Entropy of the class-token attention over frames, per head (Section 23).

    Low entropy marks a focused head that reads a few frames; high entropy
    marks a diffuse averaging head. If no head is focused and none is
    distinct, attention buys nothing here.

    Args:
        attention: class-token attention weights, shape (N, H, T), each row
            over frames summing to one.
    Returns:
        Dict with the mean per-head entropy and the per-head redundancy
        (mean pairwise correlation of attention maps).
    """
    a = np.clip(attention, 1e-12, 1.0)
    ent = -np.sum(a * np.log(a), axis=-1)      # (N, H)
    mean_ent = ent.mean(axis=0)                # (H,)

    H = attention.shape[1]
    flat = attention.transpose(1, 0, 2).reshape(H, -1)  # (H, N*T)
    corr = np.corrcoef(flat)
    iu = np.triu_indices(H, k=1)
    redundancy = float(np.mean(corr[iu])) if H > 1 else 0.0
    return {"mean_entropy": mean_ent, "redundancy": redundancy}


def selected_frames(attention: np.ndarray, motion: np.ndarray,
                    entropy_quantile: float = 0.25) -> dict:
    """Check whether focused heads select high-motion frames (Section 23).

    For the most focused heads, correlate the attention weight with a
    per-frame motion signal. A positive correlation says attention does
    event detection.

    Args:
        attention: shape (N, H, T).
        motion: per-frame motion, shape (N, T).
        entropy_quantile: fraction of heads counted as focused.
    Returns:
        Dict with the focused head indices and their attention-motion
        correlation.
    """
    a = np.clip(attention, 1e-12, 1.0)
    ent = (-np.sum(a * np.log(a), axis=-1)).mean(axis=0)  # (H,)
    cut = np.quantile(ent, entropy_quantile)
    focused = np.where(ent <= cut)[0]

    corrs = {}
    m = motion - motion.mean(axis=1, keepdims=True)
    for h in focused:
        ah = attention[:, h, :] - attention[:, h, :].mean(axis=1, keepdims=True)
        num = np.sum(ah * m)
        den = np.sqrt(np.sum(ah ** 2) * np.sum(m ** 2)) + 1e-12
        corrs[int(h)] = float(num / den)
    return {"focused_heads": focused, "attention_motion_corr": corrs}
