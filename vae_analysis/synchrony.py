"""Limb co-movement (synchrony) analysis for cramped-synchronised GMs.

Cramped-synchronised general movements — the abnormal form most strongly
associated with later cerebral palsy (Prechtl et al. 1997; Ferrari et al. 2002) —
are defined by *simultaneity*: limbs and trunk contract and relax together instead
of moving independently. That is a property of a single instant, so neither the
ordering of the state sequence nor its traversal can express it; it needs an
instantaneous, direction-aware measure.

This module implements exactly that, and it is **independent of the encoder, the
latent space, and the state model** — it uses raw keypoint displacements only.

Construction
------------
Speed discards direction, and direction is what separates genuine co-contraction
from two limbs that merely move at the same instant, so the **signed** displacement
is kept. For the four limbs ``L in {RA, LA, RL, LL}``,

.. math::
    \\mathbf{w}^{(i)}_L(t) = \\frac{1}{3}\\sum_{j\\in L} R_L
        \\bigl(x^{(i)}_{t+1,j} - x^{(i)}_{t,j}\\bigr) \\in \\mathbb{R}^2,

with the reflection ``R_L = diag(-1, 1)`` on the left limbs and ``I`` on the right.
The reflection changes coordinates for each limb separately, **before any pair is
formed**, so the first axis points away from the body midline on that limb's own
side; without it a bilaterally symmetric movement would register as opposition.
The construct lives at frame rate — averaging signed displacements would cancel
the very oscillation it exists to detect.

A recording is partitioned into ``E_i`` consecutive non-overlapping epochs of
``tau = 5`` s (125 frames at 25 fps — long enough that a single co-contraction
does not fill an epoch, short enough to stay within one bout of activity),
discarding any trailing remainder. Within an epoch the frames are stacked into
``W_L in R^{2n}``, with energy ``a_L = ||W_L||^2``, and normalised
``\\widehat W_L = W_L/||W_L||``. The **co-movement matrix** is the cosine
similarity

.. math::
    D^{(i,e)}_{LL'} = \\langle \\widehat W_L, \\widehat W_{L'}\\rangle \\in [-1,1],

a Gram matrix of unit vectors — positive semi-definite with unit diagonal. Its
sign is clinical: **positive** means the two limbs move together and in the same
anatomical sense (the cramped-synchronised direction), **negative** means they
move in opposition (the reciprocal pattern of normal movement).

The six off-diagonal entries are the reported object, and they group anatomically:

======================  ================  =========================================
class                   pairs             reading of a positive entry
======================  ================  =========================================
homologous              RA-LA, RL-LL      both arms / both legs contracting together
                                          — the cramped-synchronised signature
same-side               RA-RL, LA-LL      arm and leg of one side moving as a unit
contralateral           RA-LL, LA-RL      diagonal coupling
======================  ================  =========================================

Inference
---------
Group contrasts by permuting the **diagnostic labels, never the epochs** (epochs
within a recording are not exchangeable), with the per-recording median
``med_e D^{(i,e)}_p`` as the statistic for pair ``p``. The six pairs are tested
together under a **maximum-statistic** (Westfall-Young) correction, and the result
is reported for all six whatever any one shows.
"""

from __future__ import annotations

import numpy as np

# Limb order used throughout; the co-movement matrix is indexed in this order.
LIMB_ORDER: tuple[str, ...] = ("RA", "LA", "RL", "LL")
LIMB_KEYS: dict[str, str] = {"RA": "right_arm", "LA": "left_arm",
                             "RL": "right_leg", "LL": "left_leg"}

# The six off-diagonal pairs, grouped anatomically (the interpretive key).
PAIRS: tuple[tuple[str, str], ...] = (
    ("RA", "LA"), ("RL", "LL"),      # homologous
    ("RA", "RL"), ("LA", "LL"),      # same-side
    ("RA", "LL"), ("LA", "RL"),      # contralateral
)
PAIR_CLASS: dict[tuple[str, str], str] = {
    ("RA", "LA"): "homologous",    ("RL", "LL"): "homologous",
    ("RA", "RL"): "same-side",     ("LA", "LL"): "same-side",
    ("RA", "LL"): "contralateral", ("LA", "RL"): "contralateral",
}
PAIR_NAMES: tuple[str, ...] = tuple(f"{a}-{b}" for a, b in PAIRS)


def _is_mirrored(limb: str, mirror_side: str | None) -> bool:
    if mirror_side is None:
        return False
    return limb.startswith("L") if mirror_side == "left" else limb.startswith("R")


# ---------------------------------------------------------------------------
# 1. Signed, midline-reflected limb displacements
# ---------------------------------------------------------------------------
def limb_velocities(video: np.ndarray, limbs: dict[str, list[int]], *,
                    mirror_side: str | None = "left") -> np.ndarray:
    """Mean signed limb displacement per frame, midline-reflected (Eq. limbvel).

    Args:
        video: one recording ``(F, J, 2)``.
        limbs: skeleton limb map (needs ``right_arm``/``left_arm``/``right_leg``/
            ``left_leg``), e.g. ``bundle.limbs``.
        mirror_side: which side gets ``R_L = diag(-1, 1)``. ``"left"`` (default,
            as specified), ``"right"`` if the image x-convention is inverted, or
            ``None`` to disable the reflection (diagnostic only — without it a
            bilaterally symmetric movement reads as opposition).

    Returns:
        ``(F-1, 4, 2)`` displacements in ``LIMB_ORDER``.
    """
    x = np.asarray(video, float)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"video must be (F, J, 2); got {x.shape}.")
    dx = np.diff(x, axis=0)                                  # (F-1, J, 2)
    out = np.empty((len(dx), len(LIMB_ORDER), 2), float)
    for k, L in enumerate(LIMB_ORDER):
        key = LIMB_KEYS[L]
        if not limbs.get(key):
            raise ValueError(f"limb map is missing {key!r}.")
        v = dx[:, list(limbs[key]), :].mean(axis=1)           # (F-1, 2)
        if _is_mirrored(L, mirror_side):
            v = v * np.array([-1.0, 1.0])                     # R_L
        out[:, k, :] = v
    return out


# ---------------------------------------------------------------------------
# 2. Epoch-wise co-movement (cosine Gram) matrices
# ---------------------------------------------------------------------------
def _bandpass(x, fps, band, order=4):
    """Zero-phase Butterworth band-pass along axis 0.

    Zero phase matters: any group delay would shift one limb relative to another
    and destroy the instantaneous alignment the construct measures.
    """
    from scipy.signal import butter, filtfilt
    lo, hi = band[0] / (fps / 2.0), band[1] / (fps / 2.0)
    if not (0 < lo < hi < 1):
        raise ValueError(f"band {band} is not inside (0, Nyquist={fps/2}) Hz")
    b, a = butter(order, [lo, hi], btype="band")
    if len(x) <= 3 * max(len(a), len(b)):
        return x
    return filtfilt(b, a, x, axis=0)


def comovement_matrices(video: np.ndarray, limbs: dict[str, list[int]], *,
                        fps: float = 25.0, epoch_seconds: float = 5.0,
                        mirror_side: str | None = "left",
                        band: tuple[float, float] | None = None,
                        eps: float = 1e-12
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch co-movement matrices ``D`` and per-limb energies (Eq. comov).

    Frames are the *displacement* frames (``F-1`` of them); they are partitioned
    into non-overlapping epochs of ``round(fps * epoch_seconds)`` frames and any
    trailing remainder is discarded.

    A limb with (near-)zero energy in an epoch has no direction, so every entry
    involving it is ``NaN`` rather than an arbitrary value.

    ``band`` optionally band-limits the limb displacement series (zero-phase)
    before the epochs are formed; ``None`` is the construct exactly as specified.
    Differencing is high-pass, so per-frame keypoint noise dominates the raw
    displacement and attenuates ``|D|`` by roughly
    ``1/(1 + sigma_n^2/sigma_s^2)``. Being independent across limbs the noise
    does not bias the sign, but it can bury a real effect: at 0.02 torso units of
    keypoint noise a perfectly synchronous pair measures ``+0.18`` unfiltered
    against ``+0.96`` band-limited, while independent limbs stay near zero either
    way.

    Returns:
        ``D``: ``(E, 4, 4)`` cosine similarities in ``[-1, 1]``, PSD with unit
        diagonal. ``energy``: ``(E, 4)`` per-limb epoch energies ``||W_L||^2``.
    """
    W = limb_velocities(video, limbs, mirror_side=mirror_side)   # (T, 4, 2)
    if band is not None:
        W = _bandpass(W, fps, band)
    n = int(round(fps * epoch_seconds))
    if n < 2:
        raise ValueError(f"epoch of {epoch_seconds}s at {fps}fps is too short.")
    E = len(W) // n
    if E == 0:
        return np.empty((0, 4, 4)), np.empty((0, 4))
    W = W[:E * n].reshape(E, n, len(LIMB_ORDER), 2)
    # stack an epoch's frames into one vector per limb: (E, 4, 2n)
    V = W.transpose(0, 2, 1, 3).reshape(E, len(LIMB_ORDER), 2 * n)
    energy = (V ** 2).sum(-1)                                    # (E, 4)
    norm = np.sqrt(energy)
    good = norm > eps
    Vh = np.where(good[..., None], V / np.where(good[..., None], norm[..., None], 1.0),
                  np.nan)
    D = np.einsum("ead,ebd->eab", Vh, Vh)
    return D, energy


def pair_values(D: np.ndarray) -> np.ndarray:
    """Extract the six off-diagonal pair entries from ``(E, 4, 4)`` -> ``(E, 6)``."""
    idx = {L: i for i, L in enumerate(LIMB_ORDER)}
    if len(D) == 0:
        return np.empty((0, len(PAIRS)))
    return np.stack([D[:, idx[a], idx[b]] for a, b in PAIRS], axis=1)


def comovement_dataset(videos, limbs, *, fps: float = 25.0,
                       epoch_seconds: float = 5.0,
                       mirror_side: str | None = "left",
                       band: tuple[float, float] | None = None
                       ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Co-movement for every recording.

    Returns:
        ``per_recording``: list of ``(E_i, 6)`` epoch values (pair order
        ``PAIR_NAMES``); ``medians``: ``(n_rec, 6)`` per-recording medians
        (the inference statistic); ``n_epochs``: ``(n_rec,)``.
    """
    per_recording, n_epochs = [], []
    for v in videos:
        D, _ = comovement_matrices(v, limbs, fps=fps, epoch_seconds=epoch_seconds,
                                   mirror_side=mirror_side, band=band)
        vals = pair_values(D)
        per_recording.append(vals)
        n_epochs.append(len(vals))
    medians = np.array([
        np.nanmedian(r, axis=0) if len(r) else np.full(len(PAIRS), np.nan)
        for r in per_recording], float)
    return per_recording, medians, np.asarray(n_epochs, int)


# ---------------------------------------------------------------------------
# 3. Group contrast: label permutation with a max-statistic correction
# ---------------------------------------------------------------------------
def _contrast(M: np.ndarray, y: np.ndarray, statistic: str) -> np.ndarray:
    """Per-pair group contrast (abnormal vs normal), one value per column."""
    pos, neg = M[y == 1], M[y == 0]
    if statistic == "median_diff":
        with np.errstate(invalid="ignore"):
            return np.nanmedian(pos, axis=0) - np.nanmedian(neg, axis=0)
    if statistic == "auc":
        # rank-based; centred so 0 = no separation, same scale for every pair
        out = np.empty(M.shape[1])
        for p in range(M.shape[1]):
            a, b = pos[:, p], neg[:, p]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) == 0 or len(b) == 0:
                out[p] = np.nan; continue
            gt = (a[:, None] > b[None, :]).sum()
            eq = (a[:, None] == b[None, :]).sum()
            out[p] = (gt + 0.5 * eq) / (len(a) * len(b)) - 0.5
        return out
    raise ValueError(f"unknown statistic {statistic!r}.")


def permutation_test_maxstat(medians: np.ndarray, labels, *, n_perm: int = 10000,
                             statistic: str = "median_diff", seed: int = 0) -> dict:
    """Label-permutation test over the six pairs with a max-statistic correction.

    The **labels** are permuted (the 38 diagnostic labels), never the epochs —
    epochs within a recording are not exchangeable, and the per-recording median
    is what enters the contrast, so the exchangeable unit is the recording.

    Family-wise error across the six pairs is controlled by the Westfall-Young
    maximum-statistic procedure: the null distribution is that of
    ``max_p |T_p|`` over permutations, and each pair's corrected p-value is the
    tail of *that* distribution at its observed statistic.

    Returns:
        Dict with ``observed`` (6,), ``p_corrected`` (6,), ``p_uncorrected`` (6,),
        the ``null_max`` draws, and the settings.
    """
    M = np.asarray(medians, float)
    y = np.asarray(labels).astype(int)
    if len(y) != len(M):
        raise ValueError(f"{len(M)} recordings but {len(y)} labels.")
    n, n1 = len(y), int(y.sum())
    if n1 == 0 or n1 == n:
        raise ValueError("labels are degenerate (one group is empty).")

    observed = _contrast(M, y, statistic)
    rng = np.random.default_rng(seed)
    P = M.shape[1]
    null_all = np.empty((n_perm, P))
    for b in range(n_perm):
        yp = np.zeros(n, int)
        yp[rng.choice(n, n1, replace=False)] = 1
        null_all[b] = _contrast(M, yp, statistic)
    null_max = np.nanmax(np.abs(null_all), axis=1)               # (n_perm,)

    absobs = np.abs(observed)
    p_corr = (1 + (null_max[:, None] >= absobs[None, :]).sum(0)) / (1 + n_perm)
    p_unc = (1 + (np.abs(null_all) >= absobs[None, :]).sum(0)) / (1 + n_perm)
    return {"observed": observed, "p_corrected": p_corr, "p_uncorrected": p_unc,
            "null_max": null_max, "statistic": statistic, "n_perm": n_perm,
            "pairs": PAIR_NAMES, "n_pos": n1, "n_neg": n - n1}


# ---------------------------------------------------------------------------
# 4. The figure: six panels, one per pair, a strip of epoch values per recording
# ---------------------------------------------------------------------------
def plot_comovement_strips(per_recording, labels=None, *, medians=None,
                           order_by_label: bool = True, jitter: float = 0.28,
                           label_colors=("0.35", "crimson"),
                           label_names=("normal", "abnormal"),
                           title="Limb co-movement per epoch", save=None):
    """Six panels (one per limb pair); each recording is a strip of its epochs.

    Within-recording spread separates *consistent* from *intermittent*
    co-movement; between-recording differences are read across strips — so
    consistency and magnitude appear in one display. Panels are laid out by
    anatomical class (rows: homologous / same-side / contralateral).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_rec = len(per_recording)
    y = (np.zeros(n_rec, int) if labels is None else np.asarray(labels).astype(int))
    if medians is None:
        medians = np.array([np.nanmedian(r, axis=0) if len(r) else
                            np.full(len(PAIRS), np.nan) for r in per_recording])
    medians = np.asarray(medians, float)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharey=True)
    rng = np.random.default_rng(0)
    for p, (pair, name) in enumerate(zip(PAIRS, PAIR_NAMES)):
        ax = axes[p // 2][p % 2]
        # order recordings: group by label, then by that pair's median
        idx = np.arange(n_rec)
        if order_by_label:
            key = np.lexsort((np.nan_to_num(medians[:, p], nan=-9), y))
            idx = key
        for slot, i in enumerate(idx):
            vals = per_recording[i]
            if len(vals) == 0:
                continue
            v = vals[:, p]; v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            col = label_colors[y[i]] if labels is not None else label_colors[0]
            ax.scatter(slot + rng.uniform(-jitter, jitter, len(v)), v,
                       s=6, color=col, alpha=0.35, edgecolor="none", zorder=2)
            ax.plot([slot - 0.42, slot + 0.42], [np.nanmedian(v)] * 2,
                    color=col, lw=1.8, zorder=3)          # per-recording median
        ax.axhline(0.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.set_title(f"{name}   ({PAIR_CLASS[pair]})", fontsize=10)
        ax.set_xlim(-1, n_rec); ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        if p % 2 == 0:
            ax.set_ylabel("co-movement  $D$")
    if labels is not None:
        handles = [Line2D([], [], marker="o", ls="", color=label_colors[k],
                          label=label_names[k]) for k in (0, 1)]
        axes[0][1].legend(handles=handles, frameon=False, loc="upper right",
                          fontsize=9)
    fig.suptitle(f"{title}   (+ = move together, − = opposition; "
                 f"one strip per recording)", fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. One call
# ---------------------------------------------------------------------------
def run_synchrony_analysis(videos, limbs, *, labels=None, video_names=None,
                           positive_ids=None, fps: float = 25.0,
                           epoch_seconds: float = 5.0,
                           mirror_side: str | None = "left",
                           band: tuple[float, float] | None = None,
                           statistic: str = "median_diff", n_perm: int = 10000,
                           seed: int = 0, save_prefix: str | None = None,
                           show: bool = True) -> dict:
    """Co-movement analysis end to end: epochs -> six pairs -> figure -> test.

    Labels may be given directly (aligned to ``videos``) or derived from
    ``video_names`` + ``positive_ids``. Without labels the descriptive part still
    runs and the test is skipped.

    Returns:
        Dict with ``per_recording``, ``medians``, ``n_epochs``, ``labels``,
        ``test`` (or None) and ``figure``.
    """
    import re
    per_recording, medians, n_epochs = comovement_dataset(
        videos, limbs, fps=fps, epoch_seconds=epoch_seconds,
        mirror_side=mirror_side, band=band)
    print(f"[synchrony] {len(videos)} recordings | epochs/recording: "
          f"min={n_epochs.min()} med={int(np.median(n_epochs))} max={n_epochs.max()} "
          f"(tau={epoch_seconds}s @ {fps}fps"
          + (f", band-limited {band[0]}-{band[1]} Hz)" if band else
             ", unfiltered as specified)"))

    if labels is None and positive_ids is not None and video_names is not None:
        pos = {str(p).zfill(4) for p in positive_ids}
        labels = np.array([int(any(t.zfill(4) in pos
                                   for t in re.findall(r"\d+", str(nm))))
                           for nm in video_names])
        print(f"[synchrony] labels: {int(labels.sum())} positive / "
              f"{int((np.asarray(labels) == 0).sum())} negative")

    test = None
    if labels is not None:
        test = permutation_test_maxstat(medians, labels, n_perm=n_perm,
                                        statistic=statistic, seed=seed)
        print(f"[synchrony] label permutation ({statistic}, B={n_perm}, "
              f"max-statistic corrected over {len(PAIRS)} pairs)")
        for p, name in enumerate(PAIR_NAMES):
            print(f"    {name:7s} ({PAIR_CLASS[PAIRS[p]]:>13s}): "
                  f"T={test['observed'][p]:+.3f}  "
                  f"p_corr={test['p_corrected'][p]:.4f}  "
                  f"p_unc={test['p_uncorrected'][p]:.4f}")

    fig = plot_comovement_strips(
        per_recording, labels, medians=medians,
        save=(f"{save_prefix}_comovement.png" if save_prefix else None))
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return {"per_recording": per_recording, "medians": medians,
            "n_epochs": n_epochs, "labels": labels, "test": test, "figure": fig,
            "pairs": PAIR_NAMES}
