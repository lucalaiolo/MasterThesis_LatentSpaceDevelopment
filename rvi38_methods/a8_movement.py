"""Raw-kinematic constructs: the fidgety band ratio and per-state velocity.

Everything in this module reads **raw keypoint displacements only**. It never
touches the encoder, the latent space, or the state model, so it is a genuinely
independent estimator alongside the state-based quantities of A1 and A7. Only
the per-state grouping of :func:`state_velocity_profile` consults the fitted
model, and then only for the state labels.

The inter-limb coordination construct that once lived here (the cosine-Gram
co-movement matrix) has been superseded by the vector-valued WCLR-PP measure in
:mod:`a9_wclrpp`, which separates genuine coupling from shared autocorrelation
and keeps lead-lag direction. What remains here:

- **Fidgety band ratio** (FBR): the fraction of raw-velocity spectral power in
  the 0.5-2 Hz fidgety band, the direct spectral counterpart of the dwell-time
  frequency map.
- **Per-state velocity profile**: the high-velocity fraction per body group and
  state, the only construct that consults the model (for the state labels only).
"""

from __future__ import annotations

import numpy as np

from build_pose import JOINTS

# ---------------------------------------------------------------------------
# limb definitions on the BODY-15 layout (used by the velocity-profile groups)
# ---------------------------------------------------------------------------
LIMBS: dict[str, list[int]] = {
    "RA": [JOINTS.index(n) for n in ("RShoulder", "RElbow", "RWrist")],
    "LA": [JOINTS.index(n) for n in ("LShoulder", "LElbow", "LWrist")],
    "RL": [JOINTS.index(n) for n in ("RHip", "RKnee", "RAnkle")],
    "LL": [JOINTS.index(n) for n in ("LHip", "LKnee", "LAnkle")],
}


# ---------------------------------------------------------------------------
# fidgety band ratio on the raw velocities
# ---------------------------------------------------------------------------
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def band_power_ratio(signal: np.ndarray, fs: float,
                     band: tuple[float, float] = (0.5, 2.0),
                     nperseg: int | None = None) -> float:
    """Fraction of Welch spectral power inside ``band`` (channels averaged)."""
    from scipy.signal import welch
    x = np.asarray(signal, float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 8:
        return np.nan
    f, P = welch(x, fs=fs, axis=0, nperseg=min(len(x), nperseg or 256))
    P = P.mean(axis=1)
    total = _trapz(P, f)
    if not np.isfinite(total) or total <= 0:
        return np.nan
    m = (f >= band[0]) & (f <= band[1])
    return float(_trapz(P[m], f[m]) / total)


def raw_velocity_fbr(video: np.ndarray, fps: float = 25.0,
                     band: tuple[float, float] = (0.5, 2.0),
                     joints=None) -> float:
    """FBR of the raw keypoint velocities of one recording.

    Velocities are frame differences of the pose, continuous across the whole
    recording with no encoder seams, so this figure cannot be corrupted by any
    latent-stitching artefact. Constant joints contribute nothing and are
    excluded by default via ``build_pose.FREE``.
    """
    from build_pose import FREE
    x = np.asarray(video, float)[:, (FREE if joints is None else joints), :]
    v = np.diff(x, axis=0).reshape(len(x) - 1, -1)
    return band_power_ratio(v, fps, band)


def fbr_dataset(vids, fps: float = 25.0,
                band: tuple[float, float] = (0.5, 2.0)) -> np.ndarray:
    """Per-recording raw-velocity FBR."""
    return np.array([raw_velocity_fbr(v, fps, band) for v in vids], float)


# ---------------------------------------------------------------------------
# 5. per-state velocity profile (the only construct that reads the model)
# ---------------------------------------------------------------------------
def _window_frame_spans(n_frames: int, geom) -> list[tuple[int, int]]:
    """Frame span of every kept window, in the stitched trajectory's order."""
    spans = []
    if n_frames < geom.clip:
        return spans
    for s in range(0, n_frames - geom.clip + 1, geom.stride):
        for w in range(geom.lo, geom.lo + geom.step_win):
            spans.append((s + w * geom.l, s + (w + 1) * geom.l))
    return spans


def state_velocity_profile(st, vid, vids, geom, groups: dict[str, list[int]],
                           top_frac: float = 0.10, K: int | None = None) -> dict:
    """Percentage of a state's frames that are high-velocity, per body group.

    For each recording and joint the frames in that joint's top ``top_frac`` by
    speed are "high velocity"; for each ``(state, group)`` the statistic is the
    percentage of that state's frames that are high velocity, averaged over the
    group's joints — **one value per recording**, so the display carries the
    between-subject spread rather than a single pooled number.

    A quiet state reads low in every group, a whole-body state high everywhere,
    and a limb- or side-specific state shows one group elevated. Velocities are
    raw, so only the state *labels* come from the model.

    ``groups`` maps a name to joint indices — e.g. head/arms/legs, or the
    lateralised left/right limbs for an asymmetry read.
    """
    st = np.asarray(st)
    vid = np.asarray(vid)
    K = int(st.max()) + 1 if K is None else K
    out = {s: {g: [] for g in groups} for s in range(K)}
    for i, video in enumerate(vids):
        seq = st[vid == i]
        spans = _window_frame_spans(len(video), geom)
        if not spans or len(seq) == 0:
            continue
        # The delta stream has one state fewer than windows; map each state to
        # the "from" window of its transition.
        spans = spans[:len(seq)]
        seq = seq[:len(spans)]
        x = np.asarray(video, float)
        speed = np.linalg.norm(np.diff(x, axis=0), axis=-1)
        speed = np.vstack([speed, speed[-1:]])            # pad to F frames
        thr = np.quantile(speed, 1.0 - top_frac, axis=0)
        hv = speed >= thr[None, :]
        F = len(video)
        for s in range(K):
            fr = [t for (a, b), ws in zip(spans, seq) if ws == s
                  for t in range(a, min(b, F))]
            for g, joints in groups.items():
                out[s][g].append(100.0 * hv[np.ix_(np.asarray(fr), joints)].mean()
                                 if fr else np.nan)
    for s in range(K):
        for g in groups:
            out[s][g] = np.asarray(out[s][g], float)
    out["k"] = K
    out["groups"] = list(groups)
    return out


def region_groups() -> dict[str, list[int]]:
    """Head / arms / legs. Constant joints are excluded by construction."""
    return {"head": [JOINTS.index("Nose")],
            "arms": LIMBS["RA"] + LIMBS["LA"],
            "legs": LIMBS["RL"] + LIMBS["LL"]}


def lateral_groups() -> dict[str, list[int]]:
    """The four limbs kept separate, for a left-versus-right asymmetry read."""
    return {"left_arm": LIMBS["LA"], "right_arm": LIMBS["RA"],
            "left_leg": LIMBS["LL"], "right_leg": LIMBS["RL"]}
