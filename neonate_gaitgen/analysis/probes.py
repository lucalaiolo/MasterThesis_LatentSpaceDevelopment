"""Subject-split probes on the latents ([plan §6.3, §6.4]).

A two-layer MLP trained on a frozen latent to predict a categorical label,
under a **subject-level** train/test split (never by clip, [plan §5, §9]).
Used for the invariance probe on ``q_m`` (want near chance) and the
specificity probe on ``q_p`` (want near perfect), and for descriptive
labels we did not condition on.
"""

from __future__ import annotations

import numpy as np


def _keep(values):
    return np.array([v is not None and not _is_nan(v) for v in values])


def _is_nan(x):
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def probe(z: np.ndarray, y: np.ndarray, subjects: np.ndarray,
          n_splits: int = 5, hidden=(128, 64), seed: int = 0) -> dict:
    """Subject-grouped MLP probe accuracy for ``z -> y`` ([plan §6.3/§6.4]).

    Returns ``{"acc", "balanced_acc", "chance", "n"}`` where ``acc`` and
    ``balanced_acc`` are pooled over the held-out subject folds.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import balanced_accuracy_score

    z = np.asarray(z, dtype=np.float64)
    keep = _keep(y)
    z, subs = z[keep], np.asarray(subjects)[keep]
    _, y_int = np.unique(np.asarray(y, dtype=object)[keep].astype(str),
                         return_inverse=True)
    n_groups = len(np.unique(subs))
    if n_groups < 2 or len(y_int) < 4 or len(np.unique(y_int)) < 2:
        return {"acc": float("nan"), "balanced_acc": float("nan"),
                "chance": float("nan"), "n": int(len(y_int))}

    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    pred = np.zeros(len(y_int), dtype=int)
    for tr, te in gkf.split(z, y_int, subs):
        sc = StandardScaler().fit(z[tr])
        try:
            clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=300,
                                random_state=seed)
            clf.fit(sc.transform(z[tr]), y_int[tr])
            pred[te] = clf.predict(sc.transform(z[te]))
        except Exception:
            vals, cnts = np.unique(y_int[tr], return_counts=True)
            pred[te] = vals[np.argmax(cnts)]
    return {"acc": float(np.mean(pred == y_int)),
            "balanced_acc": float(balanced_accuracy_score(y_int, pred)),
            "chance": 1.0 / len(np.unique(y_int)), "n": int(len(y_int))}
