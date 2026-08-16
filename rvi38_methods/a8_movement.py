"""Raw-kinematic construct: the per-state velocity profile.

Everything in this module reads **raw keypoint displacements only**. It never
touches the encoder or the latent space, so it is a genuinely independent
estimator alongside the state-based quantities of A1 and A7. Only the per-state
grouping of :func:`state_velocity_profile` consults the fitted model, and then
only for the state labels.

Two constructs that once lived here are gone:

- the inter-limb coordination measure (the cosine-Gram co-movement matrix),
  superseded by the vector-valued WCLR-PP measure in :mod:`a9_wclrpp`, which
  separates genuine coupling from shared autocorrelation and keeps lead-lag
  direction;
- the band-power ratio of raw velocities, which scored a recording by the share
  of its spectral power inside a fixed 0.5-2 Hz window. That window had no
  empirical basis in this data, so the ratio measured nothing, and it has been
  dropped rather than re-tuned.

What remains is the **per-state velocity profile**: the high-velocity fraction
per body group and state, the only construct that consults the model (for the
state labels only).
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
# per-state velocity profile (the only construct that reads the model)
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
