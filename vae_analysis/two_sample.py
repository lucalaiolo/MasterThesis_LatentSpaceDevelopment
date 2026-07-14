"""Two-sample tests and topology (Part II Sections 20 and 18).

The classifier two-sample test is a sharper prior-match check than the
maximum mean discrepancy in high dimension, and it names the directions
that differ. Persistent homology finds loops the clustering cannot.
"""

from __future__ import annotations

import numpy as np


def classifier_two_sample(latent, n_samples: int = 4000,
                          rng: np.random.Generator | None = None) -> dict:
    """Classifier Two-Sample Test of aggregate posterior against prior (Section 20).

    Trains a classifier to tell aggregate-posterior samples from prior
    samples and reads its held-out accuracy. Under a true match the
    accuracy sits at one half; the gap above is the statistic, with a
    binomial-tail p-value (Lopez-Paz and Oquab, 2017).

    Args:
        latent: a LatentSet.
        n_samples: draws per side.
    Returns:
        Dict with accuracy, area under the curve, the p-value, and the
        linear direction that separates the two.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from scipy.stats import binomtest

    rng = np.random.default_rng() if rng is None else rng
    idx = rng.integers(0, latent.n, size=n_samples)
    std = np.exp(0.5 * latent.logvar[idx])
    q = latent.mu[idx] + std * rng.standard_normal((n_samples, latent.d_z))
    p = latent.prior_like(n_samples, rng)

    Xd = np.vstack([q, p])
    y = np.array([1] * n_samples + [0] * n_samples)
    Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.5,
                                          random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(np.mean(pred == yte))
    auc = float(roc_auc_score(yte, clf.decision_function(Xte)))
    n_correct = int(np.sum(pred == yte))
    p_value = binomtest(n_correct, len(yte), 0.5, alternative="greater").pvalue
    return {"accuracy": acc, "auc": auc, "p_value": float(p_value),
            "direction": clf.coef_[0]}


def persistent_homology(points: np.ndarray, max_dim: int = 1,
                        n_bootstrap: int = 100,
                        subsample: float = 0.5,
                        rng: np.random.Generator | None = None) -> dict:
    """Persistent homology of the latent point cloud (Section 18).

    Grows balls around the points and tracks connected components (the
    zeroth Betti number) and loops (the first Betti number) across radius.
    A long-lived loop is the signature of a cyclic movement family. A
    bootstrap over subsamples gives a confidence band on the diagrams.

    Needs the ripser library.

    Args:
        points: sample, shape (M, d).
        max_dim: top homology dimension; 1 to see loops.
        n_bootstrap: subsample rounds for the band.
        subsample: fraction kept per round.
    Returns:
        Dict with the persistence diagrams and the bootstrap band width
        (a bottleneck-distance spread) per dimension.
    """
    import warnings

    try:
        from ripser import ripser
        from persim import bottleneck
    except ImportError as e:
        raise ImportError("persistent_homology needs ripser and persim.") from e

    rng = np.random.default_rng() if rng is None else rng
    full = ripser(points, maxdim=max_dim)["dgms"]

    widths = {}
    # persim warns on every bottleneck call whenever a diagram has
    # infinite death times — H0 always does (the top-level connected
    # component never dies), so the warning is expected and noisy.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dgm[12] has points with non-finite death times",
            category=UserWarning,
        )
        for dim in range(max_dim + 1):
            dists = []
            for _ in range(n_bootstrap):
                keep = rng.choice(len(points), int(len(points) * subsample),
                                  replace=False)
                sub = ripser(points[keep], maxdim=max_dim)["dgms"][dim]
                try:
                    dists.append(bottleneck(full[dim], sub))
                except Exception:
                    pass
            widths[dim] = float(np.percentile(dists, 95)) if dists else np.nan
    return {"diagrams": full, "band_width": widths}
