"""Temporal analyses on the discrete RVQ token sequences ([plan §6.9]).

A categorical HMM over the base-layer motion tokens (state count by BIC,
per-state occupancy grouped by ``c_p``, transition matrix), and a
dependency-light change-point count on each clip's token stream (mode-shift
segmentation). Both operate natively on the discrete token setting.
"""

from __future__ import annotations

import numpy as np

from . import palette as pal


def categorical_hmm(idx: np.ndarray, c_p: np.ndarray, n_codes: int,
                    layer: int = 0, states_range=range(2, 7),
                    seed: int = 0) -> dict | None:
    """Fit a categorical HMM on base-layer tokens; BIC-select the state count.

    Returns per-state occupancy overall and grouped by ``c_p``, plus the
    transition matrix. ``None`` if hmmlearn is unavailable or data too small.
    """
    try:
        from hmmlearn.hmm import CategoricalHMM
    except ImportError:
        return None
    seqs = idx[:, :, layer].astype(int)               # (N, T')
    T = seqs.shape[1]
    X = seqs.reshape(-1, 1)
    lengths = [T] * len(seqs)
    if len(seqs) < 2 or T < 2:
        return None

    best, best_bic, best_k = None, np.inf, None
    for k in states_range:
        try:
            hmm = CategoricalHMM(n_components=k, n_features=n_codes,
                                 n_iter=50, random_state=seed)
            hmm.fit(X, lengths)
            ll = hmm.score(X, lengths)
        except Exception:
            continue
        n_params = k * k + k * n_codes + k
        bic = -2 * ll + n_params * np.log(len(X))
        if bic < best_bic:
            best, best_bic, best_k = hmm, bic, k
    if best is None:
        return None

    states = best.predict(X, lengths).reshape(len(seqs), T)
    occ = np.array([np.mean(states == s) for s in range(best_k)])
    by_class = {}
    for cls in sorted(set(np.asarray(c_p).tolist())):
        m = np.asarray(c_p) == cls
        sc = states[m]
        by_class[int(cls)] = [float(np.mean(sc == s)) for s in range(best_k)]
    return {"k": best_k, "occupancy": occ.tolist(),
            "transition": best.transmat_.tolist(),
            "occupancy_by_class": by_class}


def token_change_points(idx: np.ndarray, layer: int = 0,
                        window: int = 3) -> np.ndarray:
    """Per-clip change-point count on the token stream (mode-shift) ([§6.9]).

    Counts positions where the dominant token in a trailing window differs
    from that in a leading window — a dependency-free categorical segmenter.
    Returns an (N,) array of change-point counts.
    """
    seqs = idx[:, :, layer].astype(int)
    counts = np.zeros(len(seqs), dtype=int)
    for i, s in enumerate(seqs):
        n_cp = 0
        for t in range(window, len(s) - window):
            left = np.bincount(s[t - window:t]).argmax()
            right = np.bincount(s[t:t + window]).argmax()
            if left != right:
                n_cp += 1
        counts[i] = n_cp
    return counts


def plot_hmm(hmm: dict, out_dir, name: str = "GAITGen"):
    """Transition heatmap + per-class state occupancy ([plan §6.9])."""
    plt = pal.import_matplotlib()
    k = hmm["k"]
    fig, axes = plt.subplots(1, 2, figsize=(5 + 0.8 * k, 4.2))
    T = np.array(hmm["transition"])
    im = axes[0].imshow(T, cmap=pal.HEATMAP_CMAP, vmin=0, vmax=1)
    axes[0].set_title(f"{name} — token HMM transitions (K={k})")
    axes[0].set_xlabel("to state")
    axes[0].set_ylabel("from state")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    classes = sorted(hmm["occupancy_by_class"])
    bottom = np.zeros(len(classes))
    for s in range(k):
        vals = np.array([hmm["occupancy_by_class"][c][s] for c in classes])
        axes[1].bar(range(len(classes)), vals, bottom=bottom,
                    color=pal.class_color(s), label=f"state {s}")
        bottom += vals
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels([f"c_p={c}" for c in classes])
    axes[1].set_ylabel("state occupancy")
    axes[1].set_title("state usage by class")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, f"token_hmm_{name}.png")
