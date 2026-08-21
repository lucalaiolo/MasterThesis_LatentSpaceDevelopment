"""FidgetyFind: the literature's fidgety-movement detector, adapted to keypoints.

Implements the construct of

    Morais, R., Le, V., Morgan, C., Spittle, A., Badawi, N., Valentine, J.,
    Hurrion, E. M., Dawson, P. A., Tran, T., Venkatesh, S. (2023). *Robust and
    Interpretable General Movement Assessment Using Fidgety Movement
    Detection.* IEEE Journal of Biomedical and Health Informatics 27(10),
    5042-5053,

in the adapted form specified by METHODS §FidgetyFind, which is what this
module follows. The adaptation exists because this cohort has no RGB data: the
published hand and foot paths read dense optical flow over segmented hand/foot
pixels, and here every chain is scored from the keypoints alone.

The question the construct asks
------------------------------
FidgetyFind asks one question of each limb: are its small movements varied in
direction, or do they all point the same way? Varied direction is fidgety
movement; a single direction is its absence.

For a moving joint ``c`` and its parent ``b`` (knee against hip, wrist against
elbow, ankle against knee), each pair of frames gives the displacement
``v_c(t) = x_c(t+1) - x_c(t)``. Its direction is taken against the limb axis
``u(t) = x_c(t+1) - x_b(t+1)`` and its size as a fraction of the limb length
``l = ||x_c - x_b||``, held constant by the rigid skeleton::

    r(t)     = 100 ||v_c(t)|| / l                       (percent of the limb)
    alpha(t) = angle(u(t), v_c(t))  in (-pi, pi]        (signed)

Dividing by the limb length removes body size and camera distance; measuring
the angle against the limb removes the infant's pose. The parent joint carries
its own displacement, taken as a fraction of the trunk::

    q_b(t) = 100 ||x_b(t+1) - x_b(t)|| / ||x_Neck - x_MidHip||

which separates a limb being transported by its parent from a limb fidgeting on
a still one.

Two readings of the published paper are corrected here, as METHODS records:
the published thresholds are read as percentages (hence the explicit factor
100), and the published arccosine angle, whose range is ``[0, pi]``, is
replaced by the signed angle, which is what the eight bins over ``(-pi, pi]``
are consistent with. The published distal gate is defined on the distal joint's
own displacement over the trunk; here it is the *parent* joint's displacement,
so ``tau_hand`` and ``tau_foot`` are the published values applied to a
different quantity. The paper prints the distal gate inequality in the opposite
direction to the proximal one, which would keep only windows dominated by large
movements; that is treated as a misprint and the proximal direction is applied
to every chain.

The window score
----------------
Over a window ``W_i`` of ``L`` frames -- which carries ``L - 1`` displacements,
and ``L - 1`` is the denominator of every rate below -- the in-band frames are
those carrying a small movement::

    B_i = { t in W_i : r_min <= r(t) <= r_max }

and the score of chain ``c`` in window ``i`` is

* ``NaN`` when ``|{t : g_c(t) <= tau_m1}| / (L-1) < tau_m`` -- too few frames
  were small in amplitude for the window to be assessable at all, fidgety
  movement being small in scale;
* ``0`` when ``|B_i| / (L-1) < nu`` -- the window was assessable and too few of
  its frames fell in the band. This is a *score*, not a failure;
* otherwise the normalised entropy ``sum(-p log p) / log(B)`` of the directions
  ``{alpha(t) : t in B_i}`` over ``B = 8`` equal bins of ``(-pi, pi]``: near one
  when they pointed everywhere, near zero when they pointed one way.

The amplitude gate ``(g_c, tau_m1, tau_m)`` is ``(r, tau_hip, 0.2)`` on the hip
chains, ``(q_b, tau_hand, 0.3)`` on the hand chains and ``(q_b, tau_foot, 0.1)``
on the foot chains.

The reduction to one number per recording
-----------------------------------------
Three levels, mirroring how a rater judges a recording: fidgety movement counts
if any joint on a side shows it, if it recurs through the recording rather than
appearing only briefly, and if it is present on both sides. Group the six
chains by body side, ``C_R = {R hip, R hand, R foot}`` and ``C_L`` likewise, and

1. within a window ``i`` and side ``sigma``, take the largest entropy over the
   chains assessable there, ``s_sigma(i) = max{E_c(i)}``. The window is
   *scorable* for that side when at least one of its chains is assessable;
2. summarise each side by the ninetieth percentile of its scorable-window
   scores, ``S_sigma = Q90({s_sigma(i)})``. The ninetieth percentile encodes the
   clinical benchmark of roughly fifteen seconds of fidgety movement in a
   three-minute recording;
3. take the smaller of the two sides, ``FF = min(S_L, S_R)``.

A recording is scored only when at least a quarter of its windows on each side
are scorable; below that FidgetyFind declines to score it, and the recording is
held as ``NaN`` rather than forced to a value. Restricting the same three levels
to the two hip chains gives ``FF_hip``, and to the four limb chains ``FF_dist``.

Directionality. High ``FF`` means fidgety movement is present, the normal pole,
so the at-risk group is expected below the normal one and the reported AUC below
0.5. Nothing here is one-sided; the tests are two-sided as everywhere else.

Calibration
-----------
The published band ``[4.5, 8.0]`` percent of the parent limb per frame is tuned
to raw OpenPose output. These keypoints are smoothed and rigidified upstream, so
per-frame amplitudes are smaller and almost nothing lands in the published band.
:func:`calibrate` rescales every threshold by the single factor

    varsigma = Q75(r) / sqrt(r_min r_max)

with ``Q75`` the 75th percentile of ``r(t)`` pooled over all recordings and
chains, and picks the shortest window length whose median window collects ten
in-band frames. See that function.
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
    "R hip":  (_J["RHip"], _J["RKnee"]),      # the published proximal path
    "L hip":  (_J["LHip"], _J["LKnee"]),
    "R hand": (_J["RElbow"], _J["RWrist"]),   # keypoint-only stand-ins for the
    "L hand": (_J["LElbow"], _J["LWrist"]),   # published flow-based hands
    "R foot": (_J["RKnee"], _J["RAnkle"]),    # ... and feet
    "L foot": (_J["LKnee"], _J["LAnkle"]),
}
CHAIN_CLASS: dict[str, str] = {
    "R hip": "hip", "L hip": "hip",
    "R hand": "hand", "L hand": "hand",
    "R foot": "foot", "L foot": "foot",
}
CHAIN_ORDER: tuple[str, ...] = tuple(CHAINS)

# Body sides, the first level of the reduction.
SIDES: dict[str, tuple[str, ...]] = {
    "R": ("R hip", "R hand", "R foot"),
    "L": ("L hip", "L hand", "L foot"),
}

# The three recording-level endpoints, each the same three-level reduction
# restricted to a set of chains.
GROUPS: dict[str, tuple[str, ...]] = {
    "FF": CHAIN_ORDER,
    "FF_hip": ("R hip", "L hip"),
    "FF_dist": ("R hand", "L hand", "R foot", "L foot"),
}

# tau_m^c: the share of a window's frames that must be small in amplitude for
# the chain to be assessable there. Fixed by the published method per chain
# class, so it is not a tunable of FFParams.
TAU_M: dict[str, float] = {"hip": 0.2, "hand": 0.3, "foot": 0.1}

# The reduction's two fixed constants: the percentile that summarises a side,
# and the share of windows that must be scorable on each side before the
# recording is scored at all.
PERCENTILE = 90.0
MIN_SCORABLE = 0.25


@dataclass(frozen=True)
class FFParams:
    """Every threshold FidgetyFind uses. Defaults are the published values.

    ``r_min``, ``r_max`` and ``tau_hip`` are percentages of the parent limb per
    frame; ``tau_hand`` and ``tau_foot`` are percentages of the trunk per frame.
    ``scale`` records the factor :func:`calibrate` applied to all five (1.0 =
    the published ladder, untouched).
    """

    # small-movement band, percent of the parent limb per frame
    r_min: float = 4.5
    r_max: float = 8.0

    # tau_m1 per chain class: the amplitude below which a frame counts as small
    tau_hip: float = 10.0          # on r,   percent of the parent limb
    tau_hand: float = 1.0          # on q_b, percent of the trunk
    tau_foot: float = 2.5          # on q_b, percent of the trunk

    window: int = 50               # L, frames per window
    stride: int = 20               # frames between window starts
    nu: float = 0.2                # in-band floor; below it the window scores 0
    bins: int = 8                  # B, direction bins over (-pi, pi]

    scale: float = 1.0             # varsigma, recorded for the report

    def gate(self, chain: str) -> tuple[str, float, float]:
        """``(signal, tau_m1, tau_m)`` of the amplitude gate for one chain.

        The signal is ``"r"`` on the hip chains -- the moving joint's own
        displacement over the parent limb -- and ``"q"`` on the hand and foot
        chains, the *parent* joint's displacement over the trunk: a wrist
        carried by a swinging elbow is transport, not fidget.
        """
        cls = CHAIN_CLASS[chain]
        tau_m1 = {"hip": self.tau_hip, "hand": self.tau_hand,
                  "foot": self.tau_foot}[cls]
        return ("r" if cls == "hip" else "q"), tau_m1, TAU_M[cls]


PUBLISHED = FFParams()


# ---------------------------------------------------------------------------
# per-frame motion features
# ---------------------------------------------------------------------------
def motion_features(pose: np.ndarray, chains=CHAIN_ORDER) -> dict:
    """Per-frame ``(r, alpha, q)`` for every chain of a ``(F, 15, 2)`` recording.

    Returns arrays of shape ``(F - 1, C)`` in the order of ``chains``:

    * ``r`` -- the moving joint's displacement as a percentage of the parent
      limb's length;
    * ``alpha`` -- the signed angle from the limb axis ``u(t)`` to that
      displacement, in ``(-pi, pi]``;
    * ``q`` -- the parent joint's own displacement as a percentage of the
      trunk length.

    The limb length is read as ``||u(t)||``, the length of the very axis the
    angle is measured against. The skeleton behind this cohort is rigid, so that
    is the constant ``l`` of the specification and no frame is privileged; a
    non-rigid input would make the choice matter, and it is stated here rather
    than hidden. Frames whose limb or trunk length is zero, or whose
    coordinates are not finite, yield ``NaN``, and a ``NaN`` satisfies no
    inequality downstream: it is neither small in amplitude nor in band.
    """
    x = np.asarray(pose, float)
    if x.ndim != 3 or x.shape[2] != 2:
        raise ValueError(f"pose must be (F, J, 2), got {x.shape}")
    chains = tuple(chains)
    F, C = x.shape[0], len(chains)
    r = np.full((max(F - 1, 0), C), np.nan)
    alpha = np.full((max(F - 1, 0), C), np.nan)
    q = np.full((max(F - 1, 0), C), np.nan)
    if F < 2:
        return {"r": r, "alpha": alpha, "q": q, "chains": chains}

    trunk = np.linalg.norm(x[1:, NECK] - x[1:, MIDHIP], axis=-1)
    trunk = np.where(trunk > 0, trunk, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        for ci, name in enumerate(chains):
            b, c = CHAINS[name]
            v = x[1:, c] - x[:-1, c]              # displacement of the joint
            u = x[1:, c] - x[1:, b]              # limb axis at t+1
            ell = np.linalg.norm(u, axis=-1)
            ell = np.where(ell > 0, ell, np.nan)
            r[:, ci] = 100.0 * np.linalg.norm(v, axis=-1) / ell
            alpha[:, ci] = np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0],
                                      u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1])
            q[:, ci] = (100.0
                        * np.linalg.norm(x[1:, b] - x[:-1, b], axis=-1) / trunk)
    return {"r": r, "alpha": alpha, "q": q, "chains": chains}


def direction_entropy(angles, bins: int = 8) -> float:
    """Normalised Shannon entropy of a direction histogram on ``(-pi, pi]``.

    ``0`` means every displacement pointed into one bin, ``1`` means they were
    spread uniformly over the circle. The division by ``log(bins)`` is what puts
    every window on the same scale. Empty bins contribute nothing (``0 log 0 =
    0``); no epsilon is added, so a one-bin histogram returns exactly ``0``.

    Angles are binned by ``ceil((a + pi) / w) - 1`` modulo ``bins``, which
    realises the half-open convention ``(-pi, pi]`` of the specification and
    folds any angle outside that range onto its equivalent inside it, rather
    than discarding it.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    B = int(bins)
    if a.size == 0 or B < 2:
        return float("nan")
    w = 2.0 * np.pi / B
    idx = (np.ceil((a + np.pi) / w).astype(np.int64) - 1) % B
    p = np.bincount(idx, minlength=B) / a.size
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(B))


# ---------------------------------------------------------------------------
# per-window entropies
# ---------------------------------------------------------------------------
def window_starts(n_frames: int, params: FFParams = PUBLISHED) -> np.ndarray:
    """First frame of every window of ``L`` frames, at the given stride.

    A window of ``L`` frames spans feature indices ``[s0, s0 + L - 1)``, so a
    recording of ``F`` frames admits starts up to ``(F - 1) - (L - 1)``. Partial
    windows at the tail are not emitted: the specification's rates all divide by
    ``L - 1``, which only a full window supplies.
    """
    n_feat = max(int(n_frames) - 1, 0)
    span = max(int(params.window) - 1, 1)
    if n_feat < span:
        return np.zeros(0, int)
    return np.arange(0, n_feat - span + 1, max(int(params.stride), 1))


def window_entropies(feat: dict, params: FFParams = PUBLISHED) -> dict:
    """Per-window direction entropy for each chain: the three-branch score.

    Returns ``E`` of shape ``(n_windows, C)``: ``NaN`` where the amplitude gate
    voided the chain, ``0.0`` where it was assessable but too little of it fell
    in the band, and the normalised direction entropy otherwise.
    """
    r, alpha, q = feat["r"], feat["alpha"], feat["q"]
    chains = tuple(feat["chains"])
    starts = window_starts(r.shape[0] + 1, params)
    span = max(int(params.window) - 1, 1)          # L - 1
    E = np.full((len(starts), len(chains)), np.nan)

    with np.errstate(invalid="ignore"):
        for wi, s0 in enumerate(starts):
            sl = slice(int(s0), int(s0) + span)
            for ci, name in enumerate(chains):
                rr, aa = r[sl, ci], alpha[sl, ci]
                signal, tau_m1, tau_m = params.gate(name)
                g = rr if signal == "r" else q[sl, ci]
                small = np.count_nonzero(np.isfinite(g) & (g <= tau_m1)) / span
                if small < tau_m:
                    continue                        # unassessable: stays NaN
                in_band = (np.isfinite(rr) & (rr >= params.r_min)
                           & (rr <= params.r_max))
                if np.count_nonzero(in_band) / span < params.nu:
                    E[wi, ci] = 0.0                 # assessable, nothing fidgety
                    continue
                E[wi, ci] = direction_entropy(aa[in_band], params.bins)
    return {"E": E, "starts": starts, "chains": chains}


# ---------------------------------------------------------------------------
# per-recording reduction
# ---------------------------------------------------------------------------
def side_scores(E: np.ndarray, chains, subset) -> np.ndarray:
    """``s_sigma(i)``: the largest entropy over the chains assessable in window ``i``.

    ``NaN`` where no chain of ``subset`` was assessable, which is what makes the
    window unscorable for that side.
    """
    E = np.asarray(E, float)
    idx = [i for i, c in enumerate(tuple(chains)) if c in set(subset)]
    out = np.full(E.shape[0], np.nan)
    if not idx or E.shape[0] == 0:
        return out
    sub = E[:, idx]
    ok = np.isfinite(sub).any(axis=1)
    if ok.any():
        out[ok] = np.nanmax(sub[ok], axis=1)
    return out


def reduce_group(E: np.ndarray, chains, group: str) -> dict:
    """The three-level reduction of one chain group to one number.

    ``group`` names a key of :data:`GROUPS`. Returns the endpoint ``value``
    (``NaN`` when the quarter-of-windows rule declines to score the recording),
    the two side summaries ``S_L`` / ``S_R``, and the share of windows scorable
    on each side.
    """
    E = np.asarray(E, float)
    wanted = set(GROUPS[group])
    n = E.shape[0]
    out: dict = {"group": group, "n_windows": int(n)}
    S, cov = {}, {}
    for sd in ("L", "R"):
        s = side_scores(E, chains, [c for c in SIDES[sd] if c in wanted])
        ok = np.isfinite(s)
        cov[sd] = float(ok.mean()) if n else 0.0
        S[sd] = float(np.percentile(s[ok], PERCENTILE)) if ok.any() else np.nan
        out[f"s_{sd}"] = s
    scored = bool(n and all(cov[sd] >= MIN_SCORABLE and np.isfinite(S[sd])
                            for sd in ("L", "R")))
    out.update({"S_L": S["L"], "S_R": S["R"],
                "scorable_L": cov["L"], "scorable_R": cov["R"],
                "scored": scored,
                "value": float(min(S["L"], S["R"])) if scored else float("nan")})
    return out


def aggregate(E: np.ndarray, chains, params: FFParams = PUBLISHED) -> dict:
    """Every endpoint of one recording, plus the per-chain assessable fraction.

    ``FF``, ``FF_hip`` and ``FF_dist`` are the reported numbers; ``coverage`` is
    descriptive, the share of windows each chain could be scored in.
    """
    E = np.asarray(E, float)
    chains = tuple(chains)
    out = {g: reduce_group(E, chains, g) for g in GROUPS}
    cov = np.array([float(np.isfinite(E[:, i]).mean()) if E.shape[0] else np.nan
                    for i in range(len(chains))])
    flat = {"chains": chains, "coverage": cov, "n_windows": int(E.shape[0]),
            "reduction": out}
    for g, r in out.items():
        flat[g] = r["value"]
        flat[f"{g}_S_L"], flat[f"{g}_S_R"] = r["S_L"], r["S_R"]
        flat[f"{g}_scorable_L"] = r["scorable_L"]
        flat[f"{g}_scorable_R"] = r["scorable_R"]
        flat[f"{g}_scored"] = r["scored"]
    return flat


def fidgetyfind_recording(pose: np.ndarray,
                          params: FFParams = PUBLISHED) -> dict:
    """FidgetyFind for one recording: window entropies and their reduction."""
    feat = motion_features(pose)
    win = window_entropies(feat, params)
    out = dict(win)
    out.update(aggregate(win["E"], win["chains"], params))
    return out


def fidgetyfind_dataset(poses, params: FFParams = PUBLISHED) -> dict:
    """FidgetyFind over a cohort. ``poses`` is a list of ``(F, 15, 2)`` arrays.

    ``FF``, ``FF_hip`` and ``FF_dist`` are ``(n_recordings,)`` and are the three
    endpoints; ``coverage`` is ``(n_recordings, 6)``; ``E`` and ``starts`` are
    ragged and kept as lists.
    """
    poses = list(poses)
    chains = CHAIN_ORDER
    per = [fidgetyfind_recording(p, params) for p in poses]

    def col(key, dtype=float):
        return np.array([r[key] for r in per], dtype)

    out = {"chains": chains,
           "chain_class": tuple(CHAIN_CLASS[c] for c in chains),
           "groups": tuple(GROUPS),
           "E": [r["E"] for r in per],
           "starts": [r["starts"] for r in per],
           "coverage": (np.vstack([r["coverage"] for r in per]) if per
                        else np.zeros((0, len(chains)))),
           "n_windows": col("n_windows", int),
           "params": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in vars(params).items()}}
    for g in GROUPS:
        out[g] = col(g)
        for suffix in ("S_L", "S_R", "scorable_L", "scorable_R"):
            out[f"{g}_{suffix}"] = col(f"{g}_{suffix}")
        out[f"{g}_scored"] = col(f"{g}_scored", bool)
    return out


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def calibrate(poses, params: FFParams = PUBLISHED, pct: float = 75.0,
              min_samples: int = 10, window_grid=(50, 70, 100, 150),
              overlap: float = 20.0 / 50.0) -> dict:
    """Slide the published ladder onto this cohort's own amplitude scale.

    The published band is tuned to raw OpenPose output. These keypoints are
    smoothed and rigidified, so per-frame amplitudes are smaller: the published
    floor 4.5 sits near this cohort's 85th percentile and almost nothing lands
    in the band. Every threshold is rescaled by one factor. Pool ``r(t)`` over
    all recordings and chains, let ``Q75`` be its 75th percentile, and set::

        varsigma = Q75 / sqrt(r_min r_max)
        (r_min, r_max, tau_hip, tau_hand, tau_foot)
            <- varsigma (4.5, 8.0, 10.0, 1.0, 2.5)

    The first three values are fractions of the parent limb and the last two are
    fractions of the trunk, so a single factor across all five assumes the two
    scales shifted together.

    Smaller amplitudes then need a longer window to collect the same number of
    in-band frames. The published window is 50 frames with an in-band floor of
    0.2, so a window needs at least 10 in-band frames; the shortest ``L`` whose
    *median* window over every length-``L`` position in every recording reaches
    ten is chosen, and ``nu = 10 / L``::

        L = min{ L' in window_grid : median_i |B_i(L')| >= 10 },   nu = 10 / L

    The stride is set to ``overlap * L``, the published ratio ``20/50``, so the
    window overlap is preserved.

    Returns ``{"params": FFParams, "scale": varsigma, "q75": ..., "window": L,
    "median_in_band": {L': median count}, "window_reached": bool}``. When no
    ``L'`` in the grid reaches ``min_samples`` the largest is used and
    ``window_reached`` is False.
    """
    poses = list(poses)
    feats = [motion_features(p) for p in poses]
    pool = np.concatenate([f["r"].ravel() for f in feats]) if feats \
        else np.zeros(0)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        raise ValueError("no finite per-frame amplitudes to calibrate against")

    q75 = float(np.percentile(pool, float(pct)))
    scale = q75 / float(np.sqrt(params.r_min * params.r_max))
    scaled = replace(params,
                     r_min=params.r_min * scale, r_max=params.r_max * scale,
                     tau_hip=params.tau_hip * scale,
                     tau_hand=params.tau_hand * scale,
                     tau_foot=params.tau_foot * scale,
                     scale=params.scale * scale)

    # In-band counts of every length-L' window, at every start position: the
    # specification's I(L') is *all* length-L' windows, so the counts are taken
    # at stride 1 by a cumulative sum rather than at the reporting stride.
    in_band = [(np.isfinite(f["r"]) & (f["r"] >= scaled.r_min)
                & (f["r"] <= scaled.r_max)).astype(np.int64) for f in feats]
    medians: dict[int, float] = {}
    chosen = max(window_grid)
    reached = False
    for L in sorted(int(w) for w in window_grid):
        span = max(L - 1, 1)
        counts = []
        for m in in_band:
            n_feat = m.shape[0]
            if n_feat < span:
                continue
            cs = np.vstack([np.zeros((1, m.shape[1]), np.int64),
                            np.cumsum(m, axis=0)])
            counts.append(cs[span:] - cs[:n_feat - span + 1])
        if not counts:
            medians[L] = float("nan")
            continue
        medians[L] = float(np.median(np.concatenate(counts).ravel()))
        if not reached and medians[L] >= float(min_samples):
            chosen, reached = L, True
            break

    out = replace(scaled, window=chosen,
                  stride=max(int(round(overlap * chosen)), 1),
                  nu=float(min_samples) / float(chosen))
    return {"params": out, "scale": scale, "q75": q75, "window": chosen,
            "median_in_band": medians, "window_reached": reached,
            "min_samples": int(min_samples)}
