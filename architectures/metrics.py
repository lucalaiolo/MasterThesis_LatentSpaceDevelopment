"""Evaluation metrics for the CARE-PD latent-structure claims ([CARE-PD §11]).

The scientific claim is about latent geometry, so it has to be measured,
not asserted. These are the primary outputs of the thesis; building them
before the training grid keeps each run cheap to evaluate ([CARE-PD §12,
step 4]).

Contents map onto the protocol sections:

    site_probe               §11.1  how much cohort survives in the latent
    cluster_label_agreement  §11.2  ARI / NMI of clusters vs clinical labels
    kmeans_labels            §11.2  post-hoc clustering for models without
    hdbscan_labels                  a native q(y|x)
    linear_probe             §11.3  linear read-out of UPDRS / freezer / med
    occupancy                §10    per-component GM occupancy monitor

Everything takes frozen NumPy latents so it is agnostic to which of the
four models produced them. scikit-learn is imported lazily with a clear
message, and HDBSCAN degrades to a no-op if unavailable.
"""

from __future__ import annotations

import numpy as np


def _sklearn():
    try:
        import sklearn  # noqa: F401
        return sklearn
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "The metrics module needs scikit-learn. `pip install scikit-learn`."
        ) from e


def _split_indices(n: int, test_fraction: float, seed: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    return perm[n_test:], perm[:n_test]


# ---- §11.1 Site probe -----------------------------------------------------

def site_probe(latents: np.ndarray, cohort_ids: np.ndarray,
               train_idx: np.ndarray | None = None,
               test_idx: np.ndarray | None = None,
               hidden: tuple[int, ...] = (128, 64),
               test_fraction: float = 0.25, seed: int = 0,
               max_iter: int = 300) -> dict:
    """Train a two-layer MLP on frozen latents to predict cohort ([§11.1]).

    Near-ceiling accuracy means the latent still carries the nuisance
    cohort axis (expected for the plain VAE and GM-VAE); near-chance means
    the conditioning stripped it (the CVAE / GM-CVAE invariance claim).

    Args:
        latents: (N, d_z) frozen posterior means / samples.
        cohort_ids: (N,) integer cohort labels.
        train_idx, test_idx: explicit split; if omitted a random split of
            ``test_fraction`` is drawn with ``seed``.
        hidden: MLP hidden-layer sizes (two layers by default, [§11.1]).
        test_fraction, seed: control the fallback random split.
        max_iter: MLP training iterations.
    Returns:
        Dict with ``top1`` accuracy, ``chance`` (1 / n_classes present in
        the test split), and ``n_test``.
    """
    _sklearn()
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    latents = np.asarray(latents, dtype=np.float64)
    cohort_ids = np.asarray(cohort_ids)
    n = len(latents)
    if train_idx is None or test_idx is None:
        train_idx, test_idx = _split_indices(n, test_fraction, seed)

    scaler = StandardScaler().fit(latents[train_idx])
    Xtr = scaler.transform(latents[train_idx])
    Xte = scaler.transform(latents[test_idx])

    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter,
                        random_state=seed)
    clf.fit(Xtr, cohort_ids[train_idx])
    pred = clf.predict(Xte)
    top1 = float(np.mean(pred == cohort_ids[test_idx]))
    n_classes = len(np.unique(cohort_ids[test_idx]))
    return {"top1": top1, "chance": 1.0 / max(n_classes, 1),
            "n_test": int(len(test_idx))}


# ---- §11.2 Cluster–label agreement ---------------------------------------

def cluster_label_agreement(assignments: np.ndarray, labels: np.ndarray
                            ) -> dict:
    """ARI and NMI between cluster assignments and a clinical label ([§11.2]).

    Rows where the label is missing (``None`` / NaN) are dropped first.

    Args:
        assignments: (N,) integer cluster ids (native argmax q(y|x) or
            post-hoc k-means / HDBSCAN).
        labels: (N,) clinical labels (UPDRS_GAIT, freezer, medication).
    Returns:
        Dict with ``ari``, ``nmi``, and ``n`` (rows scored).
    """
    _sklearn()
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score)

    assignments = np.asarray(assignments)
    labels = np.asarray(labels, dtype=object)
    keep = np.array([l is not None and not _is_nan(l) for l in labels])
    a, l = assignments[keep], labels[keep]
    if len(a) == 0:
        return {"ari": float("nan"), "nmi": float("nan"), "n": 0}
    # Encode arbitrary label values to ints.
    _, l_int = np.unique(l.astype(str), return_inverse=True)
    return {"ari": float(adjusted_rand_score(l_int, a)),
            "nmi": float(normalized_mutual_info_score(l_int, a)),
            "n": int(len(a))}


def kmeans_labels(z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Post-hoc k-means assignment on the marginal latent ([§11.2]).

    The fair comparison for models without a native q(y|x): fit k-means on
    the same frozen z and score its clusters against the labels.
    """
    _sklearn()
    from sklearn.cluster import KMeans
    z = np.asarray(z, dtype=np.float64)
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(z)


def hdbscan_labels(z: np.ndarray, min_cluster_size: int = 25):
    """Post-hoc HDBSCAN assignment, or ``None`` if HDBSCAN is unavailable.

    HDBSCAN needs no k and marks outliers as ``-1``; returning ``None``
    lets callers skip it gracefully rather than crash a run.
    """
    try:
        try:
            from sklearn.cluster import HDBSCAN  # sklearn >= 1.3
        except ImportError:
            from hdbscan import HDBSCAN  # type: ignore
    except ImportError:
        return None
    z = np.asarray(z, dtype=np.float64)
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(z)


# ---- §11.3 Linear probing -------------------------------------------------

def linear_probe(z: np.ndarray, targets: np.ndarray, task: str,
                 train_idx: np.ndarray | None = None,
                 test_idx: np.ndarray | None = None,
                 test_fraction: float = 0.25, seed: int = 0) -> dict:
    """Fit a linear read-out of a phenotype target on frozen z ([§11.3]).

    The CVAE / GM-CVAE should match the plain VAE here despite having the
    cohort axis removed — phenotype signal should not ride on cohort.

    Args:
        z: (N, d_z) frozen latents.
        targets: (N,) target values; rows with missing targets are dropped.
        task: ``"regression"`` (UPDRS_GAIT, reports R^2) or
            ``"classification"`` (freezer / medication, reports balanced
            accuracy).
        train_idx, test_idx: explicit split, else a random one.
        test_fraction, seed: fallback split controls.
    Returns:
        Dict with ``score`` (R^2 or balanced accuracy), ``metric`` name,
        and ``n``.
    """
    _sklearn()
    z = np.asarray(z, dtype=np.float64)
    targets = np.asarray(targets, dtype=object)
    keep = np.array([t is not None and not _is_nan(t) for t in targets])
    z, y = z[keep], targets[keep]
    n = len(z)
    if n < 4:
        return {"score": float("nan"), "metric": "n/a", "n": int(n)}
    if train_idx is None or test_idx is None:
        train_idx, test_idx = _split_indices(n, test_fraction, seed)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(z[train_idx])
    Xtr, Xte = scaler.transform(z[train_idx]), scaler.transform(z[test_idx])

    if task == "regression":
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        yf = y.astype(np.float64)
        model = Ridge().fit(Xtr, yf[train_idx])
        score = float(r2_score(yf[test_idx], model.predict(Xte)))
        return {"score": score, "metric": "r2", "n": int(n)}
    elif task == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        _, yi = np.unique(y.astype(str), return_inverse=True)
        model = LogisticRegression(max_iter=1000).fit(Xtr, yi[train_idx])
        score = float(balanced_accuracy_score(yi[test_idx], model.predict(Xte)))
        return {"score": score, "metric": "balanced_acc", "n": int(n)}
    raise ValueError(f"unknown task {task!r}; use 'regression' or 'classification'.")


# ---- §10 GM occupancy -----------------------------------------------------

def occupancy(responsibilities: np.ndarray) -> np.ndarray:
    """Per-component occupancy rho_k = mean_n q(y_n = k | x_n) ([§10]).

    A near-zero entry is a dead component; watch this from the first epoch
    and prefer reducing K over letting components sit empty.

    Args:
        responsibilities: (N, K) soft assignments.
    Returns:
        (K,) occupancy vector, sums to 1.
    """
    resp = np.asarray(responsibilities, dtype=np.float64)
    return resp.mean(axis=0)


def _is_nan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False
