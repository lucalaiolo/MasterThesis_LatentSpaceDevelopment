"""Kinematic features and latent regression (Part I Section 5).

Hand-built per-clip descriptors that a clinician can read: speeds per
limb, left-right laterality, upper-lower balance, and joint energy. The
regression from the latent onto these features is a report card of what
the latent carries. The feature set is generic in J; per-limb features
appear only for the limbs the skeleton names.
"""

from __future__ import annotations

import numpy as np


def kinematic_features(X: np.ndarray, skeleton) -> tuple[np.ndarray, list[str]]:
    """Per-clip kinematic descriptors.

    Args:
        X: clips, shape (N, T, J, 3).
        skeleton: a Skeleton; uses `limbs` and `left_right`.
    Returns:
        (features, names): features shape (N, F), and the F names.
    """
    N = X.shape[0]
    vel = np.linalg.norm(np.diff(X, axis=1), axis=-1)  # (N, T-1, J) speeds

    feats, names = [], []

    feats.append(vel.mean(axis=(1, 2)))
    names.append("mean_speed")

    feats.append((vel ** 2).sum(axis=(1, 2)))
    names.append("kinetic_energy")

    limb_speed = {}
    for name, joints in skeleton.limbs.items():
        s = vel[:, :, joints].mean(axis=(1, 2))
        limb_speed[name] = s
        feats.append(s)
        names.append(f"speed_{name}")

    # Laterality across named limb pairs, if the caller used the convention
    # of matching "left_*" and "right_*" limb names.
    lefts = {n[5:]: n for n in limb_speed if n.startswith("left_")}
    rights = {n[6:]: n for n in limb_speed if n.startswith("right_")}
    for key in set(lefts) & set(rights):
        feats.append(limb_speed[lefts[key]] - limb_speed[rights[key]])
        names.append(f"laterality_{key}")

    # Upper-lower balance if the skeleton names arms and legs.
    arms = [limb_speed[n] for n in limb_speed if "arm" in n]
    legs = [limb_speed[n] for n in limb_speed if "leg" in n]
    if arms and legs:
        feats.append(np.sum(arms, axis=0) - np.sum(legs, axis=0))
        names.append("upper_lower_balance")

    return np.stack(feats, axis=1), names


def _temporal_split(latent, test_fraction: float = 0.2):
    """Hold out the last fraction of each video by time, not at random."""
    if latent.time_index is None or latent.video_id is None:
        n = latent.n
        cut = int(n * (1 - test_fraction))
        train = np.zeros(n, bool)
        train[:cut] = True
        return train, ~train
    train = np.zeros(latent.n, bool)
    for v in np.unique(latent.video_id):
        idx = np.where(latent.video_id == v)[0]
        order = idx[np.argsort(latent.time_index[idx])]
        cut = int(len(order) * (1 - test_fraction))
        train[order[:cut]] = True
    return train, ~train


def feature_regression(latent, features: np.ndarray, names: list[str],
                       alpha: float = 1.0, test_fraction: float = 0.2) -> dict:
    """Ridge regression from the posterior mean onto the features (Section 5).

    Splits by time within each video, fits ridge on the training half, and
    reports the held-out coefficient of determination per feature. A low
    score on a feature the clinical work cares about is a blind spot.

    Args:
        latent: a LatentSet.
        features: shape (N, F).
        names: feature names.
        alpha: ridge penalty.
    Returns:
        Dict mapping each name to its held-out R-squared, plus the mean.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    train, test = _temporal_split(latent, test_fraction)
    xs = StandardScaler().fit(latent.mu[train])
    Xtr, Xte = xs.transform(latent.mu[train]), xs.transform(latent.mu[test])

    r2 = {}
    for k, name in enumerate(names):
        ytr, yte = features[train, k], features[test, k]
        model = Ridge(alpha=alpha).fit(Xtr, ytr)
        pred = model.predict(Xte)
        ss_res = np.sum((yte - pred) ** 2)
        ss_tot = np.sum((yte - yte.mean()) ** 2) + 1e-12
        r2[name] = float(1 - ss_res / ss_tot)
    r2["_mean"] = float(np.mean([v for k, v in r2.items() if not k.startswith("_")]))
    return r2


def canonical_correlation(latent, features: np.ndarray, n_components: int = 5) -> dict:
    """Canonical correlation between the latent and the features (Section 5).

    Finds the linear combinations of latent and of features that correlate
    most strongly. The count of strong pairs is the number of shared axes.

    Returns:
        Dict with the per-component correlations and the fitted transform.
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(latent.mu)
    F = StandardScaler().fit_transform(features)
    k = min(n_components, Z.shape[1], F.shape[1])
    cca = CCA(n_components=k).fit(Z, F)
    Zc, Fc = cca.transform(Z, F)
    corrs = [float(np.corrcoef(Zc[:, i], Fc[:, i])[0, 1]) for i in range(k)]
    return {"correlations": corrs, "model": cca}
