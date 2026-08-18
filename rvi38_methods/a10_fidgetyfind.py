"""FidgetyFind: the literature's fidgety-movement detector, run on this cohort.

Implements the construct of

    Morais, R., Le, V., Morgan, C., Spittle, A., Badawi, N., Valentine, J.,
    Hurrion, E. M., Dawson, P. A., Tran, T., Venkatesh, S. (2023). *Robust and
    Interpretable General Movement Assessment Using Fidgety Movement
    Detection.* IEEE Journal of Biomedical and Health Informatics 27(10),
    5042-5053,

following the reference implementation the authors released
(``github.com/RomeroBarata/fidgetyfind``: ``fidgetyfind/proximal.py``,
``fidgetyfind/distal.py``, ``fidgetyfind/skeleton_smoothing.py``).

Why it belongs here. Every other construct in this package is ours: fluency
(``Phi``) reads the state path of a fitted latent model, mixing (Kemeny) reads
its transition matrix, WCLR-PP reads raw limb velocities. FidgetyFind is the
published detector of the *same* clinical target the GMA label encodes — the
presence of fidgety movements — computed from the keypoints alone. It is
therefore the external yardstick: a construct nobody in this project designed,
against which our own can be contrasted on the identical cohort, and whose
agreement (or disagreement) with ``Phi`` is itself a result.

The idea. Fidgety movements are *small* and *directionally variable*: the limb
wanders, it does not stroke. So take each frame-to-frame displacement of a
joint of interest, keep only those whose amplitude falls in a small-movement
band (measured as a percentage of the parent limb's length, so the measure is
scale-free), record the direction of each kept displacement *relative to the
parent limb's own axis*, and take the Shannon entropy of the direction
histogram inside a short window. Directions spread over the circle give
entropy near 1: fidgety movement is present. Directions piled into one bin, or
too few small movements to judge, give entropy near 0: fidgety movement is
absent -- the abnormal pole, and the one the GMA label marks.

What transfers verbatim from the reference implementation
---------------------------------------------------------
* the per-frame feature triple ``(angle, magnitude, score)``: magnitude
  ``100 |x_c(t+1) - x_c(t)| / |x_c(t) - x_b(t)|`` (percent of the parent limb
  per frame) and angle ``atan2`` of the displacement against the limb axis
  ``x_c(t+1) - x_b(t+1)``, for the moving joint ``c`` and its parent ``b``;
* the rescaling of every per-frame magnitude by ``fps / 30`` before any
  threshold is applied, since the published thresholds are calibrated on
  30 fps recordings and a displacement per frame is not fps-invariant;
* the window gates -- too many low-confidence frames, or too many frames whose
  displacement exceeds ``large_motion``, make a window *unscoreable* (NaN);
* the small-amplitude band ``[minr, maxr]``, and the rule that a window with
  fewer than ``in_range_rate`` of its frames inside that band scores **0.0**
  rather than NaN (it was assessable, and no fidgety movement was found);
* the histogram over ``[-pi, pi]``, its Shannon entropy, and the division by
  ``log(bins)`` that puts every window on ``[0, 1]``;
* the reference's window geometry (``start_frame=100``, ``window=50``,
  ``stride=20``) and its numeric defaults (``minr=4.5``, ``maxr=8.0``,
  ``large_motion=10.0`` and the rate thresholds), and the score-weighted
  5-frame Gaussian temporal smoothing of the keypoints.

What is adapted, and why
------------------------
1. **The distal path.** The reference scores hands and feet from *dense
   optical flow* over segmented hand/foot pixels, because OpenPose has no hand
   or foot keypoint. We have no video for this cohort -- the analysis consumes
   a keypoint table -- so hands and feet are scored by the same skeleton-only
   estimator applied one joint proximally: the wrist moving against the
   forearm, and the ankle moving against the shank. That is the very axis and
   reference length the flow path normalises by, and the reference's own
   proximal path is exactly this estimator at the knee. The distal gates keep
   the reference's intent: a window is dropped when the *parent* joint (elbow,
   knee) is itself transporting the limb, since then the distal motion is
   carriage, not fidget (``parent_motion_hand=1.0`` and
   ``parent_motion_foot=2.5``, in percent of the Neck-MidHip length per frame,
   as in the reference's ``large_parent_motion_threshold``).
2. **Bins.** The reference uses 16 bins for the distal histograms because each
   frame contributes thousands of flow vectors. Ours contributes one
   displacement per frame, so all six chains use the proximal setting of 8
   bins; 16 bins over at most 50 samples would read sparsity as order.
3. **The small-movement band and an in-band rate gate on the distal chains.**
   The reference gates distal flow at 8% of the parent limb per frame with no
   upper edge, and applies no in-band rate gate -- with thousands of flow
   vectors per frame it needs neither. Our distal chains carry the same kind of
   signal as the proximal ones (one smoothed keypoint displacement per frame),
   so they take the same band ``[minr, maxr]`` and the same ``in_range_rate``,
   which is what those settings were calibrated on. ``minr_distal`` /
   ``maxr_distal`` restore the reference's flow thresholds if wanted.
4. **Confidence.** The reference gates on OpenPose's per-joint detection
   score. Our table carries an ``observed`` flag instead (interior gaps are
   linearly interpolated upstream), so ``observed`` is the confidence: 1.0 for
   a real detection, 0.0 for an interpolated one. When the flag is absent
   every frame is treated as observed and the confidence gates never fire.
5. **The camera-motion gate is omitted.** The reference rejects windows in
   which the *background* optical flow is large, which needs the video. These
   are fixed-camera cot recordings; a caller who can compute the mask may pass
   it as ``scoreable``.
6. **The reduction to one number per recording.** The released code stops at
   the per-window entropies; the paper's own reduction is not in it. Ours is
   declared here and is deliberately plain: per chain, the median entropy over
   scoreable windows and the fraction of those windows above ``theta``; per
   recording, the mean of each over the six chains. ``theta = 0.5`` is fixed a
   priori (halfway up the normalised entropy scale: a window whose directions
   fill two of eight bins scores 0.33, four of eight 0.67).

Directionality. High entropy = fidgety movement present = the *normal* pole.
The GMA label marks *absence*, so the abnormal group is expected **lower**, and
the reported AUC is expected below 0.5. Nothing here is one-sided; the tests
are two-sided as everywhere else in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from build_pose import JOINTS

_J = {n: i for i, n in enumerate(JOINTS)}
NECK, MIDHIP = _J["Neck"], _J["MidHip"]

# (parent joint b, moving joint c). The estimator watches c move and measures
# that motion against the limb b->c: its length sets the amplitude scale and
# its direction sets the angular reference.
CHAINS: dict[str, tuple[int, int]] = {
    "R hip":  (_J["RHip"], _J["RKnee"]),      # the reference's proximal path
    "L hip":  (_J["LHip"], _J["LKnee"]),
    "R hand": (_J["RElbow"], _J["RWrist"]),   # skeleton-only stand-in for the
    "L hand": (_J["LElbow"], _J["LWrist"]),   # reference's flow-based hands
    "R foot": (_J["RKnee"], _J["RAnkle"]),    # ... and feet
    "L foot": (_J["LKnee"], _J["LAnkle"]),
}
CHAIN_CLASS: dict[str, str] = {
    "R hip": "proximal", "L hip": "proximal",
    "R hand": "hand", "L hand": "hand",
    "R foot": "foot", "L foot": "foot",
}
CHAIN_ORDER: tuple[str, ...] = tuple(CHAINS)
PROXIMAL_CHAINS = tuple(c for c in CHAIN_ORDER if CHAIN_CLASS[c] == "proximal")
DISTAL_CHAINS = tuple(c for c in CHAIN_ORDER if CHAIN_CLASS[c] != "proximal")

EPS = 1e-7


@dataclass(frozen=True)
class FFParams:
    """Every threshold FidgetyFind uses. Defaults are the published values.

    Magnitudes are percentages *per frame at 30 fps*: a displacement is first
    divided by its reference length, multiplied by 100, then multiplied by
    ``fps / standard_fps`` so that recordings at other frame rates meet the
    same thresholds.
    """

    fps: float = 25.0
    standard_fps: float = 30.0

    # window geometry (frames)
    start_frame: int = 100       # skip the camera-adjustment head of the video
    window: int = 50
    stride: int = 20

    bins: int = 8                # direction-histogram bins over [-pi, pi]

    # --- proximal chains (the reference's own path) ---
    minr: float = 4.5            # small-movement band, % of parent limb / frame
    maxr: float = 8.0
    large_motion: float = 10.0   # a frame this large is not a fidget
    large_motion_rate: float = 0.2   # ... and this many of them void the window
    lowconf_rate: float = 0.1

    # --- distal chains (wrist against forearm, ankle against shank) ---
    # The band defaults to the proximal one. The reference gates its distal
    # path at 8% of the parent limb per frame with no upper edge, but that
    # threshold sizes *dense optical flow*; ours sizes a smoothed keypoint
    # displacement, which is the signal ``minr``/``maxr`` were set for. Set
    # these to depart from that.
    minr_distal: float | None = None
    maxr_distal: float | None = None
    lowconf_rate_distal: float = 0.2
    parent_motion_hand: float = 1.0          # % of Neck-MidHip / frame
    parent_motion_rate_hand: float = 0.3
    parent_motion_foot: float = 2.5
    parent_motion_rate_foot: float = 0.1

    in_range_rate: float = 0.2   # below this, the window scores 0.0, not NaN

    # --- keypoint smoothing (reference: 5-frame Gaussian, sigma 2) ---
    smooth: bool = True
    smooth_window: int = 5
    smooth_sigma: float = 2.0

    theta: float = 0.5           # a window above this counts as fidgety-positive

    chains: tuple[str, ...] = CHAIN_ORDER

    def with_fps(self, fps: float) -> "FFParams":
        return replace(self, fps=float(fps))

    @property
    def rate_scale(self) -> float:
        """``fps / 30``: converts a per-frame displacement to its 30 fps size."""
        return float(self.fps) / float(self.standard_fps)

    def band(self, chain: str) -> tuple[float, float]:
        if CHAIN_CLASS[chain] == "proximal":
            return self.minr, self.maxr
        return (self.minr if self.minr_distal is None else self.minr_distal,
                self.maxr if self.maxr_distal is None else self.maxr_distal)

    def lowconf(self, chain: str) -> float:
        return (self.lowconf_rate if CHAIN_CLASS[chain] == "proximal"
                else self.lowconf_rate_distal)

    def motion_gate(self, chain: str) -> tuple[str, float, float]:
        """``(what is gated, threshold, tolerated rate)`` for one chain.

        Proximal chains gate on the moving joint's own displacement, as the
        reference does. Distal chains gate on the parent joint's displacement:
        a wrist carried by a swinging elbow is transport, not fidget.
        """
        cls = CHAIN_CLASS[chain]
        if cls == "proximal":
            return "moving", self.large_motion, self.large_motion_rate
        if cls == "hand":
            return "parent", self.parent_motion_hand, self.parent_motion_rate_hand
        return "parent", self.parent_motion_foot, self.parent_motion_rate_foot


# ---------------------------------------------------------------------------
# keypoint smoothing
# ---------------------------------------------------------------------------
def gaussian_kernel(window: int, sigma: float) -> np.ndarray:
    """Symmetric Gaussian window, matching ``scipy.signal.windows.gaussian``."""
    n = np.arange(window) - (window - 1) / 2.0
    return np.exp(-0.5 * (n / float(sigma)) ** 2)


def smooth_tracks(pose: np.ndarray, conf: np.ndarray | None = None,
                  window: int = 5, sigma: float = 2.0) -> np.ndarray:
    """Confidence-weighted temporal Gaussian smoothing of ``(F, J, 2)`` poses.

    The reference smooths each joint and channel along time with a 5-frame
    Gaussian whose weights are multiplied by the detection score, so an
    unreliable neighbour cannot drag the estimate. This matters more than it
    looks: FidgetyFind's band starts at a few percent of a limb length per
    frame, which is the scale of keypoint jitter, and unsmoothed jitter is
    directionally uniform -- it would read as fidgety movement everywhere.

    Frames whose entire smoothing neighbourhood is unobserved keep their
    original coordinates (and stay excluded by the confidence gate).
    """
    x = np.asarray(pose, float)
    if window is None or window <= 1:
        return x
    F, J, D = x.shape
    w = (np.ones((F, J), float) if conf is None
         else np.asarray(conf, float).reshape(F, J).clip(0.0, None))
    k = gaussian_kernel(int(window), float(sigma))
    half = int(window) // 2

    xp = np.zeros((F + 2 * half, J, D))
    wp = np.zeros((F + 2 * half, J))
    xp[half:half + F] = x
    wp[half:half + F] = w

    num = np.zeros_like(x)
    den = np.zeros((F, J))
    for u in range(len(k)):
        ww = k[u] * wp[u:u + F]
        num += ww[..., None] * xp[u:u + F]
        den += ww
    good = den > 0
    out = np.where(good[..., None], num / np.where(good, den, 1.0)[..., None], x)
    return out


# ---------------------------------------------------------------------------
# per-frame motion features
# ---------------------------------------------------------------------------
def motion_features(pose: np.ndarray, observed: np.ndarray | None = None,
                    params: FFParams = FFParams()) -> dict:
    """Per-frame ``(angle, magnitude, score, parent magnitude)`` per chain.

    ``pose`` is ``(F, 15, 2)``; ``observed`` is ``(F, 15)`` of 0/1 (or None,
    meaning every keypoint was detected). Returns arrays of shape
    ``(F - 1, C)`` in the order of ``params.chains``.

    * ``angle`` -- direction of the moving joint's displacement measured
      against the parent limb's axis, in ``[-pi, pi]``;
    * ``magnitude`` -- that displacement as a percentage of the parent limb's
      length, rescaled to 30 fps;
    * ``score`` -- the minimum confidence over the two joints and the two
      frames the displacement spans;
    * ``parent`` -- the parent joint's own displacement as a percentage of the
      Neck-MidHip length, rescaled to 30 fps (the distal transport gate).
    """
    x = np.asarray(pose, float)
    if x.ndim != 3 or x.shape[2] != 2:
        raise ValueError(f"pose must be (F, J, 2), got {x.shape}")
    F = x.shape[0]
    c = (np.ones(x.shape[:2], float) if observed is None
         else np.asarray(observed, float).reshape(x.shape[:2]))
    if params.smooth:
        x = smooth_tracks(x, c, params.smooth_window, params.smooth_sigma)

    chains = tuple(params.chains)
    C = len(chains)
    ang = np.zeros((F - 1, C))
    mag = np.zeros((F - 1, C))
    score = np.zeros((F - 1, C))
    parent = np.zeros((F - 1, C))
    if F < 2:
        return {"angle": ang, "magnitude": mag, "score": score,
                "parent": parent, "chains": chains}

    torso = np.linalg.norm(x[1:, NECK] - x[1:, MIDHIP], axis=-1)
    torso = np.where(torso > 0, torso, np.nan)

    for ci, name in enumerate(chains):
        b, cj = CHAINS[name]
        v = x[1:, cj] - x[:-1, cj]                    # displacement of c
        ref = np.linalg.norm(x[:-1, cj] - x[:-1, b], axis=-1)   # parent limb
        axis = x[1:, cj] - x[1:, b]                   # limb axis at t+1
        s = np.minimum(np.minimum(c[:-1, b], c[:-1, cj]),
                       np.minimum(c[1:, b], c[1:, cj]))
        ok = (ref > 0) & (s > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = 100.0 * np.linalg.norm(v, axis=-1) / ref * params.rate_scale
            dot = axis[:, 0] * v[:, 0] + axis[:, 1] * v[:, 1]
            det = axis[:, 0] * v[:, 1] - axis[:, 1] * v[:, 0]
            a = np.arctan2(det, dot)
            p = (100.0 * np.linalg.norm(x[1:, b] - x[:-1, b], axis=-1) / torso
                 * params.rate_scale)
        ang[:, ci] = np.where(ok, a, 0.0)
        mag[:, ci] = np.where(ok, m, 0.0)
        score[:, ci] = np.where(ref > 0, s, 0.0)
        # A parent magnitude that could not be normalised (a degenerate torso)
        # is recorded as infinite, not as zero: the distal gate reads this as
        # "the limb may be being transported and we cannot tell", which voids
        # the window rather than silently scoring it.
        parent[:, ci] = np.where(np.isfinite(p), p, np.inf)
    return {"angle": ang, "magnitude": mag, "score": score, "parent": parent,
            "chains": chains}


def direction_entropy(angles: np.ndarray, bins: int = 8) -> float:
    """Shannon entropy of a direction histogram on ``[-pi, pi]``, in ``[0, 1]``.

    ``0`` means every displacement pointed into one bin, ``1`` means they were
    spread uniformly over the circle. The division by ``log(bins)`` is the
    reference's normalisation and is what makes windows comparable.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    # Wrap into the histogram's own range: ``np.histogram`` silently discards
    # anything outside it, which would turn an angle convention mismatch into a
    # quietly wrong entropy rather than an error.
    a = (a + np.pi) % (2 * np.pi) - np.pi
    counts, _ = np.histogram(a, bins=int(bins), range=(-np.pi, np.pi))
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    return float(-(p * np.log(p + EPS)).sum() / np.log(int(bins)))


# ---------------------------------------------------------------------------
# per-window entropies
# ---------------------------------------------------------------------------
def window_starts(n_frames: int, params: FFParams = FFParams()) -> np.ndarray:
    """First frame of each scored window, on the reference's tiling."""
    n_feat = max(int(n_frames) - 1, 0)
    return np.arange(params.start_frame, max(n_feat - params.window + 1,
                                             params.start_frame), params.stride)


def window_entropies(feat: dict, params: FFParams = FFParams(),
                     scoreable: np.ndarray | None = None) -> dict:
    """Per-window direction entropy for each chain.

    Returns ``E`` of shape ``(n_windows, C)``: a value in ``[0, 1]`` where the
    window could be scored, ``NaN`` where a gate voided it. ``0.0`` is a
    *score*, not a failure -- it says the window was assessable and held no
    small, directionally varied movement.

    ``scoreable`` optionally masks windows externally (the camera-motion gate
    of the reference, which needs the video).
    """
    ang, mag = feat["angle"], feat["magnitude"]
    score, parent = feat["score"], feat["parent"]
    chains = tuple(feat["chains"])
    n_feat = ang.shape[0]
    starts = window_starts(n_feat + 1, params)
    E = np.full((len(starts), len(chains)), np.nan)
    gates = np.zeros((len(starts), len(chains)), int)   # 0 ok, 1 conf, 2 motion

    for wi, s0 in enumerate(starts):
        sl = slice(int(s0), int(s0) + params.window)
        n = ang[sl].shape[0]
        if n == 0:
            continue
        for ci, name in enumerate(chains):
            if scoreable is not None and not bool(np.asarray(scoreable).ravel()[wi]):
                gates[wi, ci] = 3
                continue
            sc = score[sl, ci]
            if np.mean(sc < 0.1) > params.lowconf(name):
                gates[wi, ci] = 1
                continue
            what, thr, rate = params.motion_gate(name)
            gated = mag[sl, ci] if what == "moving" else parent[sl, ci]
            if np.mean(gated > thr) > rate:
                gates[wi, ci] = 2
                continue
            lo, hi = params.band(name)
            m = mag[sl, ci]
            in_band = (sc > 0) & (m >= lo) & (m <= hi)
            if in_band.sum() / n < params.in_range_rate:
                E[wi, ci] = 0.0        # assessable, and no fidgety movement
                continue
            E[wi, ci] = direction_entropy(ang[sl, ci][in_band], params.bins)
    return {"E": E, "starts": starts, "chains": chains, "gates": gates}


# ---------------------------------------------------------------------------
# per-recording reduction
# ---------------------------------------------------------------------------
def _nanmedian(v):
    v = np.asarray(v, float)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")


def aggregate(E: np.ndarray, chains, params: FFParams = FFParams()) -> dict:
    """Reduce ``(n_windows, C)`` window entropies to one recording's numbers.

    Per chain: the median entropy over scoreable windows, the fraction of those
    above ``theta``, and the coverage (how much of the recording was
    assessable at all). Per recording: the mean of each over the chains, plus
    the proximal-only mean, which is the part of the construct that needs no
    adaptation from the published method.
    """
    E = np.asarray(E, float)
    chains = tuple(chains)
    med = np.array([_nanmedian(E[:, i]) for i in range(len(chains))])
    cov = np.array([float(np.isfinite(E[:, i]).mean()) if E.shape[0] else np.nan
                    for i in range(len(chains))])
    pos = np.full(len(chains), np.nan)
    for i in range(len(chains)):
        col = E[:, i]
        ok = np.isfinite(col)
        if ok.any():
            pos[i] = float((col[ok] >= params.theta).mean())
    prox = [i for i, c in enumerate(chains) if CHAIN_CLASS[c] == "proximal"]
    dist = [i for i, c in enumerate(chains) if CHAIN_CLASS[c] != "proximal"]

    def _mean(v, idx):
        v = np.asarray(v, float)[idx]
        return float(np.nanmean(v)) if len(idx) and np.isfinite(v).any() else float("nan")

    return {"median_entropy": med, "positive_rate": pos, "coverage": cov,
            "n_windows": int(E.shape[0]),
            "n_scoreable": np.isfinite(E).sum(axis=0).astype(int),
            "score": _mean(med, list(range(len(chains)))),
            "score_proximal": _mean(med, prox),
            "score_distal": _mean(med, dist),
            "positive_rate_mean": _mean(pos, list(range(len(chains)))),
            "coverage_mean": _mean(cov, list(range(len(chains))))}


def fidgetyfind_recording(pose: np.ndarray, observed: np.ndarray | None = None,
                          params: FFParams = FFParams(),
                          scoreable: np.ndarray | None = None) -> dict:
    """FidgetyFind for one recording: window entropies and their reduction."""
    feat = motion_features(pose, observed, params)
    win = window_entropies(feat, params, scoreable=scoreable)
    out = dict(win)
    out.update(aggregate(win["E"], win["chains"], params))
    out["fps"] = params.fps
    return out


def fidgetyfind_dataset(poses, observeds=None, params: FFParams = FFParams()
                        ) -> dict:
    """FidgetyFind over a cohort. ``poses`` is a list of ``(F, 15, 2)`` arrays.

    Returns the per-recording window entropies (ragged, kept as a list) and the
    per-recording matrices the group tests consume: ``median_entropy``,
    ``positive_rate`` and ``coverage`` are ``(n_recordings, C)``; ``score``,
    ``score_proximal``, ``score_distal``, ``positive_rate_mean`` and
    ``coverage_mean`` are ``(n_recordings,)``.
    """
    poses = list(poses)
    n = len(poses)
    if observeds is None:
        observeds = [None] * n
    observeds = list(observeds)
    if len(observeds) != n:
        raise ValueError(f"{n} recordings but {len(observeds)} observed masks")
    chains = tuple(params.chains)

    per, E_list = [], []
    for i in range(n):
        r = fidgetyfind_recording(poses[i], observeds[i], params)
        per.append(r)
        E_list.append(r["E"])

    def col(key):
        return np.array([r[key] for r in per], float)

    return {"chains": chains,
            "chain_class": tuple(CHAIN_CLASS[c] for c in chains),
            "E": E_list,
            "starts": [r["starts"] for r in per],
            "median_entropy": np.vstack([r["median_entropy"] for r in per]),
            "positive_rate": np.vstack([r["positive_rate"] for r in per]),
            "coverage": np.vstack([r["coverage"] for r in per]),
            "n_windows": np.array([r["n_windows"] for r in per], int),
            "score": col("score"),
            "score_proximal": col("score_proximal"),
            "score_distal": col("score_distal"),
            "positive_rate_mean": col("positive_rate_mean"),
            "coverage_mean": col("coverage_mean"),
            "params": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in vars(params).items()}}
