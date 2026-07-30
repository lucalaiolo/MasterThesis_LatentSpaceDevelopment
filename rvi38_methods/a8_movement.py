"""Raw-kinematic constructs: limb co-movement and the fidgety band ratio.

Everything in this module reads **raw keypoint displacements only**. It never
touches the encoder, the latent space, or the state model, so it is a genuinely
independent estimator alongside the state-based quantities of A1 and A7. Only
the per-state grouping of :func:`state_velocity_profile` consults the fitted
model, and then only for the state labels.

Two constructs:

1. **Limb co-movement** (cramped-synchronised general movements). Cramped-
   synchronised GMs are the abnormal form most strongly associated with later
   cerebral palsy (Prechtl et al. 1997; Ferrari et al. 2002), and their defining
   feature is *simultaneity*: limbs and trunk contract and relax together
   instead of moving independently. That is a property of a single instant, so
   neither fluency (§7) nor mixing (§9) can express it. Clinical scoring further
   separates consistent from intermittent cramped-synchronised movement and only
   the consistent form carries the high risk, so the construct reports a
   *distribution* over each recording rather than one average.

2. **Fidgety band ratio** (FBR): the fraction of raw-velocity spectral power in
   the 0.5-2 Hz fidgety band, the direct spectral counterpart of the
   dwell-time frequency map.
"""

from __future__ import annotations

import numpy as np

from build_pose import JOINTS

# ---------------------------------------------------------------------------
# limb definitions on the BODY-15 layout
# ---------------------------------------------------------------------------
LIMBS: dict[str, list[int]] = {
    "RA": [JOINTS.index(n) for n in ("RShoulder", "RElbow", "RWrist")],
    "LA": [JOINTS.index(n) for n in ("LShoulder", "LElbow", "LWrist")],
    "RL": [JOINTS.index(n) for n in ("RHip", "RKnee", "RAnkle")],
    "LL": [JOINTS.index(n) for n in ("LHip", "LKnee", "LAnkle")],
}
LIMB_ORDER: tuple[str, ...] = ("RA", "LA", "RL", "LL")

# The six off-diagonal pairs, grouped anatomically. The grouping is the
# interpretive key: a positive homologous entry is the cramped-synchronised
# signature, a positive same-side entry is an arm and leg moving as a unit, and
# a positive contralateral entry is diagonal coupling.
PAIRS: tuple[tuple[str, str], ...] = (
    ("RA", "LA"), ("RL", "LL"),      # homologous
    ("RA", "RL"), ("LA", "LL"),      # same-side
    ("RA", "LL"), ("LA", "RL"),      # contralateral
)
PAIR_NAMES: tuple[str, ...] = tuple(f"{a}-{b}" for a, b in PAIRS)
PAIR_CLASS: dict[tuple[str, str], str] = {
    ("RA", "LA"): "homologous",    ("RL", "LL"): "homologous",
    ("RA", "RL"): "same-side",     ("LA", "LL"): "same-side",
    ("RA", "LL"): "contralateral", ("LA", "RL"): "contralateral",
}


# ---------------------------------------------------------------------------
# 1. signed, midline-reflected limb displacement
# ---------------------------------------------------------------------------
def limb_velocities(video: np.ndarray, mirror_side: str | None = "left"
                    ) -> np.ndarray:
    """Mean signed limb displacement per frame, midline-reflected.

    .. math::
        \\mathbf{w}_L(t) = \\frac{1}{3}\\sum_{j\\in L}
            R_L\\bigl(x_{t+1,j} - x_{t,j}\\bigr),
        \\qquad R_L = \\operatorname{diag}(-1,1) \\text{ on the left limbs.}

    Speed discards direction, and direction is what separates genuine
    co-contraction from two limbs that merely move at the same instant, so the
    **signed** displacement is kept. The reflection changes coordinates for each
    limb separately, *before any pair is formed*, so the first axis points away
    from the body midline on that limb's own side; without it a bilaterally
    symmetric movement would register as opposition. In the torso-normalised
    frame used here anatomical right sits at negative ``x`` (RShoulder ~ -0.22)
    and left at positive ``x``, which is what makes ``mirror_side="left"``
    correct.

    Returns:
        ``(F-1, 4, 2)`` in ``LIMB_ORDER``.
    """
    x = np.asarray(video, float)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"video must be (F, J, 2); got {x.shape}")
    dx = np.diff(x, axis=0)
    out = np.empty((len(dx), len(LIMB_ORDER), 2), float)
    for k, L in enumerate(LIMB_ORDER):
        v = dx[:, LIMBS[L], :].mean(axis=1)
        flip = (mirror_side == "left" and L.startswith("L")) or \
               (mirror_side == "right" and L.startswith("R"))
        out[:, k, :] = v * np.array([-1.0, 1.0]) if flip else v
    return out


# ---------------------------------------------------------------------------
# 2. epoch-wise co-movement (cosine Gram) matrices
# ---------------------------------------------------------------------------
def comovement_matrices(video: np.ndarray, fps: float = 25.0,
                        epoch_seconds: float = 5.0,
                        mirror_side: str | None = "left",
                        eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch co-movement matrices ``D`` and per-limb energies.

    A recording is partitioned into consecutive **non-overlapping** epochs of
    ``epoch_seconds`` (125 frames at 25 fps, spanning 2.5-10 cycles of the
    fidgety band), discarding any trailing remainder. Within an epoch the frames
    are stacked into ``W_L`` of length ``2n``, with energy ``a_L = ||W_L||^2``,
    and normalised. The co-movement matrix is the cosine similarity

    .. math::
        D_{LL'} = \\langle \\widehat W_L, \\widehat W_{L'} \\rangle \\in [-1,1],

    a Gram matrix of unit vectors, hence positive semi-definite with unit
    diagonal. Its **sign is clinical**: positive means the two limbs move
    together and in the same anatomical sense (the cramped-synchronised
    direction), negative means they move in opposition (the reciprocal pattern
    of normal movement).

    The construct lives at frame rate: averaging signed displacements over an
    epoch would cancel the oscillation it exists to detect, so the frames are
    stacked, not averaged.

    A limb with (near-)zero energy in an epoch has no direction, so every entry
    involving it is ``NaN`` rather than an arbitrary value.
    """
    W = limb_velocities(video, mirror_side)
    n = int(round(fps * epoch_seconds))
    if n < 2:
        raise ValueError(f"epoch of {epoch_seconds}s at {fps}fps is too short")
    E = len(W) // n
    if E == 0:
        return np.empty((0, 4, 4)), np.empty((0, 4))
    W = W[:E * n].reshape(E, n, len(LIMB_ORDER), 2)
    V = W.transpose(0, 2, 1, 3).reshape(E, len(LIMB_ORDER), 2 * n)
    energy = (V ** 2).sum(-1)
    norm = np.sqrt(energy)
    good = norm > eps
    Vh = np.where(good[..., None],
                  V / np.where(good[..., None], norm[..., None], 1.0), np.nan)
    return np.einsum("ead,ebd->eab", Vh, Vh), energy


def pair_values(D: np.ndarray) -> np.ndarray:
    """The six off-diagonal entries: ``(E, 4, 4) -> (E, 6)`` in ``PAIR_NAMES``."""
    idx = {L: i for i, L in enumerate(LIMB_ORDER)}
    if len(D) == 0:
        return np.empty((0, len(PAIRS)))
    return np.stack([D[:, idx[a], idx[b]] for a, b in PAIRS], axis=1)


def comovement_dataset(vids, fps: float = 25.0, epoch_seconds: float = 5.0,
                       mirror_side: str | None = "left"):
    """Co-movement for every recording.

    Returns ``(per_recording, medians, n_epochs)``: a list of ``(E_i, 6)`` epoch
    values, the ``(n, 6)`` per-recording medians that carry the inference, and
    the per-recording epoch counts.
    """
    per, n_ep = [], []
    for v in vids:
        D, _ = comovement_matrices(v, fps, epoch_seconds, mirror_side)
        vals = pair_values(D)
        per.append(vals)
        n_ep.append(len(vals))
    med = np.array([np.nanmedian(r, axis=0) if len(r) else
                    np.full(len(PAIRS), np.nan) for r in per], float)
    return per, med, np.asarray(n_ep, int)


# ---------------------------------------------------------------------------
# 3. group contrast: label permutation with a maximum-statistic correction
# ---------------------------------------------------------------------------
def _contrast(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Median difference (abnormal - normal), one value per pair."""
    with np.errstate(invalid="ignore"):
        return np.nanmedian(M[y == 1], axis=0) - np.nanmedian(M[y == 0], axis=0)


def comovement_test(medians: np.ndarray, labels, n_perm: int = 10000,
                    seed: int = 0) -> dict:
    """Label-permutation test over the six pairs, max-statistic corrected.

    The **labels** are permuted, never the epochs: epochs within a recording are
    not exchangeable, and the per-recording median is what enters the contrast,
    so the exchangeable unit is the recording. Family-wise error across the six
    pairs is controlled by the Westfall-Young maximum-statistic procedure — the
    null is that of ``max_p |T_p|`` — and **all six pairs are reported whatever
    any one shows**. Because every ``D`` lives on the same ``[-1,1]`` scale the
    maximum is taken without standardisation.
    """
    M = np.asarray(medians, float)
    y = np.asarray(labels).astype(int)
    if len(y) != len(M):
        raise ValueError(f"{len(M)} recordings but {len(y)} labels")
    n, n1 = len(y), int(y.sum())
    if n1 == 0 or n1 == n:
        raise ValueError("labels are degenerate (one group is empty)")
    observed = _contrast(M, y)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, M.shape[1]))
    for b in range(n_perm):
        yp = np.zeros(n, int)
        yp[rng.choice(n, n1, replace=False)] = 1
        null[b] = _contrast(M, yp)
    null_max = np.nanmax(np.abs(null), axis=1)
    a = np.abs(observed)
    return {"observed": observed,
            "p_corrected": (1 + (null_max[:, None] >= a[None, :]).sum(0)) / (1 + n_perm),
            "p_uncorrected": (1 + (np.abs(null) >= a[None, :]).sum(0)) / (1 + n_perm),
            "null_max": null_max, "pairs": PAIR_NAMES, "n_perm": n_perm,
            "n_pos": n1, "n_neg": n - n1}


# ---------------------------------------------------------------------------
# 4. fidgety band ratio on the raw velocities
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
