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

Does it fire? The band is a *calibration*, not a definition, and a cohort whose
per-frame amplitudes sit elsewhere drives every window to the legitimate score
0.0 and every recording to 0.0 -- reported by exactly the numbers a genuine
null result would produce. On ``rvi38_analysis.csv`` that is what happens at the
published band. Call ``diagnose`` after ``fidgetyfind_dataset`` and read its
verdict before any contrast; ``docs/FIDGETYFIND_FIDELITY.md`` section 13 has
the evidence and the causes.

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
    # How much of each window fell inside the small-movement band, recorded for
    # every window whether or not a gate later voided it. This is the number
    # that says whether the *band* fits the cohort at all, independently of the
    # gates, and ``diagnose`` reads it. See the note there on silent failure.
    in_band_rate = np.zeros((len(starts), len(chains)))

    for wi, s0 in enumerate(starts):
        sl = slice(int(s0), int(s0) + params.window)
        n = ang[sl].shape[0]
        if n == 0:
            continue
        for ci, name in enumerate(chains):
            sc = score[sl, ci]
            lo, hi = params.band(name)
            m = mag[sl, ci]
            in_band = (sc > 0) & (m >= lo) & (m <= hi)
            frac = in_band.sum() / n
            in_band_rate[wi, ci] = frac
            if scoreable is not None and not bool(np.asarray(scoreable).ravel()[wi]):
                gates[wi, ci] = 3
                continue
            if np.mean(sc < 0.1) > params.lowconf(name):
                gates[wi, ci] = 1
                continue
            what, thr, rate = params.motion_gate(name)
            gated = mag[sl, ci] if what == "moving" else parent[sl, ci]
            if np.mean(gated > thr) > rate:
                gates[wi, ci] = 2
                continue
            if frac < params.in_range_rate:
                E[wi, ci] = 0.0        # assessable, and no fidgety movement
                continue
            E[wi, ci] = direction_entropy(ang[sl, ci][in_band], params.bins)
    return {"E": E, "starts": starts, "chains": chains, "gates": gates,
            "in_band_rate": in_band_rate}


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

    # The mean over windows, kept beside the median because the median is
    # fragile in exactly the way this cohort is: when most windows are quiet the
    # median window is a rate-rule 0.0 and the median collapses, while the mean
    # still registers the active windows. ``diagnose`` compares the two.
    mean = np.array([float(np.nanmean(E[:, i])) if np.isfinite(E[:, i]).any()
                     else np.nan for i in range(len(chains))])
    return {"median_entropy": med, "mean_entropy": mean,
            "positive_rate": pos, "coverage": cov,
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
            "gates": [r["gates"] for r in per],
            "in_band_rate": [r["in_band_rate"] for r in per],
            "median_entropy": np.vstack([r["median_entropy"] for r in per]),
            "mean_entropy": np.vstack([r["mean_entropy"] for r in per]),
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


# ---------------------------------------------------------------------------
# is the construct measuring anything at all?
# ---------------------------------------------------------------------------
GATE_NAMES = {0: "scored", 1: "low confidence", 2: "large motion",
              3: "external mask"}


def diagnose(ds: dict, params: FFParams = FFParams()) -> dict:
    """Check that FidgetyFind actually fired on this cohort.

    FidgetyFind fails *silently*, and the failure is indistinguishable from a
    real negative result unless you look for it. Its band ``[minr, maxr]`` is a
    calibration: 4.5-8.0 % of the parent limb per frame is what a fidgety
    movement looked like through the authors' detector on the authors' cohort.
    Point the same band at data whose per-frame amplitudes live an order of
    magnitude lower and every window falls short of ``in_range_rate``, every
    window takes the legitimate score ``0.0`` ("assessable, nothing fidgety"),
    and every recording reduces to ``0.0``. The group contrast then reports
    AUC 0.5 and p = 1 -- which reads exactly like "the published detector found
    no difference between the groups", when what happened is that the detector
    never fired. One is a finding; the other is a broken measurement. Nothing
    downstream can tell them apart, so it has to be checked here.

    ``ds`` is the dict returned by ``fidgetyfind_dataset``. Returns per-chain
    diagnostics and a list of ``warnings``; ``degenerate`` is True when the
    per-recording score carries essentially no information.
    """
    E = [np.asarray(e, float) for e in ds["E"]]
    chains = tuple(ds["chains"])
    C = len(chains)
    allE = (np.concatenate(E, axis=0) if len(E)
            else np.zeros((0, C)))
    gates = (np.concatenate([np.asarray(g) for g in ds["gates"]], axis=0)
             if ds.get("gates") else np.zeros((0, C), int))
    ibr = (np.concatenate([np.asarray(r, float) for r in ds["in_band_rate"]],
                          axis=0) if ds.get("in_band_rate")
           else np.zeros((0, C)))

    scoreable = np.isfinite(allE)
    n_win = allE.shape[0]
    out = {
        "chains": chains,
        "n_windows": int(n_win),
        "coverage": scoreable.mean(axis=0) if n_win else np.full(C, np.nan),
        "gate_rate": np.array([[float((gates[:, ci] == g).mean()) if n_win
                                else np.nan for g in (0, 1, 2, 3)]
                               for ci in range(C)]),
        "median_in_band_rate": (np.median(ibr, axis=0) if n_win
                                else np.full(C, np.nan)),
        "zero_rate": np.array([
            float((allE[scoreable[:, ci], ci] == 0.0).mean())
            if scoreable[:, ci].any() else np.nan for ci in range(C)]),
    }

    score = np.asarray(ds["score"], float)
    finite = score[np.isfinite(score)]
    out["score_zero_fraction"] = (float((finite == 0.0).mean())
                                  if finite.size else np.nan)
    out["score_distinct_values"] = int(np.unique(finite).size)

    # One line per *kind* of failure, naming the chains it hit -- the per-chain
    # numbers are already in the table, and a warning block longer than the
    # report it warns about gets skimmed.
    w = []
    lo, hi = params.minr, params.maxr

    def _hit(mask, msg):
        names = [chains[ci] for ci in range(C) if mask[ci]]
        if names:
            w.append(f"{', '.join(names)}: {msg}")

    r = out["median_in_band_rate"]
    thin = np.isfinite(r) & (r < params.in_range_rate)
    if thin.any():
        _hit(thin,
             f"the median window puts {100 * np.nanmedian(r[thin]):.0f}% of its "
             f"frames inside the band [{lo:.2f}, {hi:.2f}], against the "
             f"{100 * params.in_range_rate:.0f}% this construct needs before it "
             f"will compute an entropy at all, so the typical window takes the "
             f"score 0.0 by the rate rule and not by measurement")
    z = out["zero_rate"]
    _hit(np.isfinite(z) & (z > 0.9),
         "over 90% of assessable windows scored exactly 0.0 -- the direction "
         "entropy is essentially never evaluated")
    c = out["coverage"]
    low = np.isfinite(c) & (c < 0.5)
    if low.any():
        g = out["gate_rate"]
        worst = int(np.argmax(np.nanmean(g[low][:, 1:], axis=0))) + 1
        _hit(low, f"fewer than half the windows were assessable; the "
                  f"{GATE_NAMES[worst]} gate is the main cause "
                  f"({100 * np.nanmean(g[low][:, worst]):.0f}% of windows)")

    # The median-over-windows reduction (section 8, ours) collapses to 0.0 as
    # soon as most windows are quiet, even where the band is fine and the active
    # windows carry a perfectly good measurement. That is a defect of the
    # reduction, not of the band, and it needs a different fix -- so it is
    # reported separately rather than folded into the band warning above.
    med = np.asarray(ds["median_entropy"], float)
    mean = np.asarray(ds.get("mean_entropy", med), float)
    with np.errstate(invalid="ignore"):
        med_dead = np.array([float(np.nanmean(med[:, ci] == 0.0))
                             if np.isfinite(med[:, ci]).any() else np.nan
                             for ci in range(C)])
        mean_live = np.array([float(np.nanmean(mean[:, ci] > 0.0))
                              if np.isfinite(mean[:, ci]).any() else np.nan
                              for ci in range(C)])
    out["median_zero_fraction"] = med_dead
    out["mean_nonzero_fraction"] = mean_live
    rescued = np.isfinite(med_dead) & (med_dead > 0.9) & (mean_live > 0.5)
    _hit(rescued,
         "the median window entropy is 0.0 for over 90% of recordings while the "
         "mean is not -- most windows are quiet, so the median reduction of "
         "section 8 is throwing away a measurement the active windows did make. "
         "Use mean_entropy, or report the per-window entropies directly")

    zf = out["score_zero_fraction"]
    out["degenerate"] = bool(
        (np.isfinite(zf) and zf > 0.9) or out["score_distinct_values"] < 3)
    if out["degenerate"]:
        w.insert(0,
                 f"the per-recording score is degenerate: "
                 f"{100 * zf:.0f}% of recordings score exactly 0.0 and the "
                 f"score takes {out['score_distinct_values']} distinct "
                 f"value(s) across the cohort. Any AUC, p-value or correlation "
                 f"computed from it is a statement about the band "
                 f"[{lo:.2f}, {hi:.2f}] not fitting this data, NOT about the "
                 f"infants.")
    out["warnings"] = w
    return out


def format_diagnosis(d: dict) -> str:
    """The diagnosis as a printable block."""
    L = ["  measurement check (does the construct fire on this cohort?):",
         f"     {'chain':8s} {'assessable':>11} {'in-band rate':>13} "
         f"{'windows = 0':>12}  gates (conf / motion)"]
    for ci, nm in enumerate(d["chains"]):
        g = d["gate_rate"][ci]
        L.append(f"     {nm:8s} {100 * d['coverage'][ci]:10.1f}% "
                 f"{100 * d['median_in_band_rate'][ci]:12.1f}% "
                 f"{100 * d['zero_rate'][ci]:11.1f}%  "
                 f"{100 * g[1]:.0f}% / {100 * g[2]:.0f}%")
    if d["warnings"]:
        L.append("  " + "!" * 68)
        for msg in d["warnings"]:
            L.append(f"  !! {msg}")
        L.append("  " + "!" * 68)
    else:
        L.append("     no degeneracy detected: the band fits and the entropy "
                 "is being computed")
    return "\n".join(L)


def calibrate_band(poses, observeds=None, params: FFParams = FFParams(),
                   centre_pct: float = 75.0, min_samples: int = 10,
                   window_grid=(50, 70, 100, 150)) -> FFParams:
    """Slide the published amplitude ladder onto this cohort's own scale.

    Use only when ``diagnose`` says the published band does not fit, and say in
    print that you used it: the result is no longer the published measurement.

    The published numbers ``minr=4.5``, ``maxr=8.0``, ``large_motion=10.0`` are
    a calibration against the authors' detector, cohort and preprocessing. They
    fail to transfer in two independent ways when the amplitude distribution
    differs (see ``docs/FIDGETYFIND_FIDELITY.md`` section 13):

    * **location** -- the whole ladder can sit far from where the data lives;
    * **width** -- ``maxr/minr`` is fixed at 1.78, and a band that narrow cannot
      hold ``in_range_rate`` of a broad amplitude distribution *at any
      location*, so the rate rule fires however the band is placed.

    This fixes both, without touching the construct's internal geometry:

    1. all four amplitude thresholds are multiplied by one factor ``s``, chosen
       so the band's geometric centre ``sqrt(minr*maxr) = 6.0`` lands on the
       ``centre_pct``-th percentile of the cohort's pooled per-frame amplitudes
       over observed frames. The published ratios 4.5 : 8.0 : 10.0 survive;
    2. ``in_range_rate`` is replaced by the *sample count* it encodes upstream
       (the reference's 0.2 x 50 frames = 10 directions in the histogram), and
       the shortest window in ``window_grid`` that supplies 10 samples at the
       cohort's realised in-band share is chosen.

    ``centre_pct`` is a free choice and there is no principled value for it --
    that is the honest cost of the published band not transferring. **Fix it
    before you look at the labels.** On this cohort the group contrast is
    stable for ``centre_pct`` in roughly 65-85, but stability is not
    significance: corrected for the whole search, the effect does not reach it.
    """
    poses = list(poses)
    observeds = [None] * len(poses) if observeds is None else list(observeds)
    pool = []
    for x, o in zip(poses, observeds):
        f = motion_features(x, o, params)
        pool.append(f["magnitude"][f["score"] > 0])
    pool = np.concatenate(pool) if pool else np.zeros(0)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        raise ValueError("no observed frames to calibrate against")

    centre = float(np.percentile(pool, float(centre_pct)))
    s = centre / float(np.sqrt(params.minr * params.maxr))
    out = replace(params,
                  minr=params.minr * s, maxr=params.maxr * s,
                  large_motion=params.large_motion * s,
                  parent_motion_hand=params.parent_motion_hand * s,
                  parent_motion_foot=params.parent_motion_foot * s)
    if params.minr_distal is not None:
        out = replace(out, minr_distal=params.minr_distal * s)
    if params.maxr_distal is not None:
        out = replace(out, maxr_distal=params.maxr_distal * s)

    # Pick the window from the realised per-window in-band *count*, not from the
    # pooled per-frame share: window activity is heavy-tailed, so the median
    # window holds noticeably less than the cohort average and sizing on the
    # average leaves the median window still failing the rate rule.
    need = float(min_samples)
    feats = [motion_features(x, o, out) for x, o in zip(poses, observeds)]
    window = max(window_grid)
    for w in sorted(window_grid):
        counts = []
        probe = replace(out, window=w, stride=max(w // 2, 1))
        for f in feats:
            mag, sc = f["magnitude"], f["score"]
            for s0 in window_starts(mag.shape[0] + 1, probe):
                sl = slice(int(s0), int(s0) + w)
                m, c = mag[sl], sc[sl]
                counts.append(((c > 0) & (m >= out.minr) & (m <= out.maxr)).sum(axis=0))
        if counts and float(np.median(np.concatenate(counts))) >= need:
            window = w
            break
    return replace(out, window=window, stride=max(window // 2, 1),
                   in_range_rate=need / window)
