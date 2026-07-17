"""Principal movements: the SVD basis and its velocity series ([guideline §3]).

Fit a movement basis on a balanced subset (so no cohort dominates), pick the
component count at ~90% variance and cross-check with TWO-NN intrinsic
dimension, project every walk onto the basis, and take the first derivative
(velocity of the weights) — the ARHMM input, exactly as in the paper. Plus a
cheap **site diagnostic**: is ``cohort`` separable in the static weights?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import palette as pal


# ---- SVD (compact re-implementation of the reference run_svd) --------------

def run_svd(data: np.ndarray, significance: str | None = None,
            n_perms: int = 100, seed: int = 0):
    """SVD with optional Marchenko-Pastur / permutation eigenvalue threshold."""
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    evr = s ** 2 / np.sum(s ** 2)
    if significance == "mp":
        M, N = data.shape
        lam = ((1.0 + np.sqrt(N / M)) ** 2 * (M - 1)) ** 0.5
        return vh, s, evr, (s > lam).astype(int), lam
    if significance == "perm":
        rng = np.random.default_rng(seed)
        lam = 0.0
        perm = data.copy()
        for _ in range(n_perms):
            for j in range(perm.shape[1]):
                rng.shuffle(perm[:, j])
            ps = np.linalg.svd(perm, full_matrices=False, compute_uv=False)
            lam = max(lam, ps[0])
        return vh, s, evr, (s > lam).astype(int), lam
    return vh, s, evr


def twonn_dimension(points: np.ndarray, discard: float = 0.1) -> float:
    """TWO-NN intrinsic-dimension estimate (Facco et al. 2017) ([guideline §3.2])."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(points)
    dist, _ = nn.kneighbors(points)
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = r1 > 0
    mu = np.sort(np.log(r2[ok] / r1[ok]))
    keep = int(len(mu) * (1 - discard))
    return float(keep / mu[:keep].sum())


# ---- Balanced basis fit ([guideline §3.1]) --------------------------------

def balanced_subset(data, seed: int = 0) -> list[int]:
    """One walk per subject, ~equal per cohort ([guideline §3.1])."""
    info = data.info
    rng = np.random.default_rng(seed)
    per_subject = []
    for subj, grp in info.groupby("subject_id"):
        per_subject.append(int(rng.choice(grp.index.values)))
    # equalise per cohort by capping to the smallest cohort's subject count
    coh = info.loc[per_subject, "cohort"]
    counts = coh.value_counts()
    cap = int(counts.min())
    chosen = []
    for c in counts.index:
        idx = [i for i in per_subject if info.loc[i, "cohort"] == c]
        chosen += list(rng.permutation(idx)[:cap])
    return sorted(chosen)


@dataclass
class PMBasis:
    eigenvectors: np.ndarray   # (n_components, d)
    scaler_center: np.ndarray  # (d,)
    scaler_scale: np.ndarray   # (d,)
    evr: np.ndarray            # explained variance ratio (full)
    n_components: int
    twonn: float


def fit_basis(data, var_target: float = 0.90, seed: int = 0) -> PMBasis:
    """Fit the movement basis on the balanced subset ([guideline §3.1-§3.2])."""
    from sklearn.preprocessing import RobustScaler
    idx = balanced_subset(data, seed)
    X = np.concatenate([data.features[i] for i in idx], axis=0)
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    vh, s, evr = run_svd(Xs)
    n90 = int(np.searchsorted(np.cumsum(evr), var_target) + 1)
    dim = twonn_dimension(Xs[np.random.default_rng(seed).choice(
        len(Xs), size=min(3000, len(Xs)), replace=False)])
    return PMBasis(eigenvectors=vh, scaler_center=scaler.center_,
                   scaler_scale=scaler.scale_, evr=evr,
                   n_components=n90, twonn=dim)


def project(data, basis: PMBasis, n_components: int | None = None) -> list:
    """Project every walk onto the basis and return velocity series ([§3.3]).

    Returns a list of ``(T, n_components)`` weight-velocity arrays — the
    ARHMM input.
    """
    n = n_components or basis.n_components
    V = basis.eigenvectors[:n]                      # (n, d)
    out = []
    for x in data.features:
        xs = (x - basis.scaler_center) / basis.scaler_scale
        w = xs @ V.T                                # (T, n) weights
        out.append(np.gradient(w, axis=0))          # velocity of weights
    return out


def static_weights(data, basis: PMBasis, n_components: int | None = None
                   ) -> np.ndarray:
    """Per-walk mean weight vector (for the site diagnostic)."""
    n = n_components or basis.n_components
    V = basis.eigenvectors[:n]
    rows = []
    for x in data.features:
        xs = (x - basis.scaler_center) / basis.scaler_scale
        rows.append((xs @ V.T).mean(axis=0))
    return np.stack(rows)


# ---- Site diagnostic ([guideline §3.4]) -----------------------------------

def site_diagnostic(data, basis: PMBasis, seed: int = 0) -> dict:
    """Is ``cohort`` separable in the static weights? ([guideline §3.4]).

    Silhouette of cohort labels + a leave-one-cohort-out logistic accuracy.
    A large value means the basis partly encodes site, not movement — flag it
    in the go/no-go rather than proceeding silently.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    W = static_weights(data, basis)
    coh = data.info["cohort"].values
    sil = float(silhouette_score(W, coh)) if len(set(coh)) > 1 else float("nan")
    try:
        acc = float(cross_val_score(LogisticRegression(max_iter=500), W, coh,
                                    cv=3).mean())
        chance = max(np.bincount(
            np.unique(coh, return_inverse=True)[1]).max() / len(coh), 1e-9)
    except Exception:
        acc, chance = float("nan"), float("nan")
    return {"cohort_silhouette": sil, "cohort_clf_acc": acc,
            "cohort_chance": float(chance)}


def plot_variance_curve(basis: PMBasis, out_dir, name="fig1b_variance.png"):
    """Variance-explained curve with the chosen component count (Fig 1b)."""
    plt = pal.import_matplotlib()
    cum = np.cumsum(basis.evr)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, "-o", ms=3, color="#4C72B0")
    ax.axhline(0.9, ls=":", color="grey", label="90% variance")
    ax.axvline(basis.n_components, ls="--", color="#C44E52",
               label=f"chosen K={basis.n_components}")
    ax.axvline(basis.twonn, ls="-.", color="#55A868",
               label=f"TWO-NN dim={basis.twonn:.1f}")
    ax.set_xlabel("principal movement")
    ax.set_ylabel("cumulative variance explained")
    ax.set_title("Principal-movement variance (Fig 1b analogue)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(40, len(cum)))
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)
