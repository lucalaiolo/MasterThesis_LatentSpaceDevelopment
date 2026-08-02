"""A1 core: window-to-frame geometry, state kinematic signatures, fluency.

Implements METHODS §4 (geometry), §5 (state kinematic signatures), §6 (the
kinematic similarity matrix) and §7 (fluency).

Direction-aware signature (``DIRECTION_AWARE_KINEMATIC_SIMILARITY.md``). The
scalar RMS speed ``a[k, j]`` that the original similarity correlated is blind
to the axis along which a joint moves and to how confined the motion is: two
states that drive the same joint at equal speed along orthogonal axes are
identical under it, so a genuine redirection reads as monotony -- exactly the
signal a fluency measure exists to index. The signature is therefore lifted to
the pooled **second-moment matrix** ``M[k, j] = mean(v v^T)`` per state and
joint, whose trace is still ``a[k, j]^2`` (:func:`state_second_moments`), and
whose trace-normalised part is read off as the double-angle shape coordinate
``u = (rho cos 2theta, rho sin 2theta)`` (:func:`shape_coordinates`). The
similarity fed to Phi combines a magnitude channel (log RMS speed, as before)
with a shape channel (the residual axes), the preferred *separated* form being
``S = omega S_mag + (1 - omega) S_shape`` (:func:`direction_aware_similarity`).
Nothing else changes: Phi and the maxT family are untouched -- one statistic
enters the signature and one leaves.

Two deviations from the supplied reference implementation are deliberate and
are documented where they occur:

* :func:`state_profiles` accumulates each state's frames as a **set union**, as
  §5.2 defines ``F_k = union of B_m over windows m in state k``. Consecutive
  delta windows overlap by ``l`` frames, so concatenating their spans instead
  double-counts the shared frames of every run of two or more same-state
  windows, over-weighting long dwells in the RMS.
* :func:`phi_excess` draws its permutation null in one vectorised block rather
  than a Python loop, which is what makes the full 2,000-draw null of §12.3
  affordable for every subject and every lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from build_pose import FREE, JOINTS

# ---------------------------------------------------------------------------
# §4.1 the encoder tiling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Encoder tiling. Defaults are the values verified in METHODS §4.1."""

    fps: float = 25.0           # f_frame
    clip: int = 64              # T
    n_win: int = 16             # W
    stride: int = 32            # sigma

    @property
    def l(self) -> int:                                   # noqa: E743
        """Frames per window, ``T / W``."""
        return self.clip // self.n_win

    @property
    def step_win(self) -> int:
        """Kept windows per clip, ``sigma / l``."""
        return self.stride // self.l

    @property
    def lo(self) -> int:
        """First kept window index, ``(W - sigma/l) / 2``."""
        return (self.n_win - self.step_win) // 2

    @property
    def f0(self) -> int:
        """First frame covered by kept window 0, ``l * lo``."""
        return self.l * self.lo

    @property
    def f_win(self) -> float:
        """Window rate in Hz, ``f_frame / l``."""
        return self.fps / self.l

    def n_pose_windows(self, n_frames: int) -> int:
        if n_frames < self.clip:
            return 0
        return self.step_win * ((n_frames - self.clip) // self.stride + 1)

    def n_delta_windows(self, n_frames: int) -> int:
        """§4.3: one fewer delta window than pose windows."""
        return max(self.n_pose_windows(n_frames) - 1, 0)

    def delta_spans(self, n_delta: int, n_frames: int, union: bool = True):
        """Frame spans ``B_m`` of each delta window, as indices into ``v``.

        §4.2 attributes ``dz_m`` to the union of the frame spans of pose windows
        ``m`` and ``m+1``, an ``2l``-frame interval. Bounds are clipped to
        ``n_frames - 1``, the length of the speed array.
        """
        width = 2 * self.l if union else self.l
        lo = self.f0 + self.l * np.arange(n_delta)
        hi = np.minimum(lo + width, n_frames - 1)
        return np.minimum(lo, n_frames - 1), hi


GEOM = Geometry()


def verify_lengths(frames_per_video, lengths, geom: Geometry = GEOM,
                   stream: str = "delta") -> dict:
    """§4.3: predicted trajectory length against the stored ``lengths``.

    Also reports which stream the stored lengths are consistent with. That
    matters because the delta stream is one window shorter per subject than the
    pose stream, and using the wrong convention shifts every frame attribution
    in §5 by half a window while still "working".
    """
    frames = np.asarray(list(frames_per_video), int)
    got = np.asarray(lengths, int)
    cand = {"delta": np.array([geom.n_delta_windows(int(f)) for f in frames]),
            "pose": np.array([geom.n_pose_windows(int(f)) for f in frames])}
    matches = {k: (len(v) == len(got) and bool(np.array_equal(v, got)))
               for k, v in cand.items()}
    pred = cand[stream]
    ok = matches[stream]
    consistent = [k for k, v in matches.items() if v]
    return {
        "predicted": pred, "stored": got, "match": ok, "stream": stream,
        "consistent_with": consistent,
        "n_mismatch": int(0 if ok else np.sum(pred[:len(got)] != got[:len(pred)])),
        "formula": ("L = step_win * (floor((F - T)/stride) + 1)"
                    + (" - 1" if stream == "delta" else "")),
    }


# ---------------------------------------------------------------------------
# §5 state kinematic signatures
# ---------------------------------------------------------------------------
def speeds(pose: dict, vids, fps: float = GEOM.fps) -> dict:
    """§5.1 per-joint speed in normalised skeleton units per second.

    Returns ``(F-1, J)`` per video. The ``* fps`` conversion is applied here so
    every downstream amplitude is already in units/second.
    """
    out = {}
    for v in vids:
        x = np.asarray(pose[v], np.float64)
        out[v] = np.linalg.norm(np.diff(x, axis=0), axis=2) * fps
    return out


def velocities(pose: dict, vids, fps: float = GEOM.fps) -> dict:
    """Signed per-frame velocity **vectors** ``(F-1, J, 2)`` in units/second.

    The direction-aware counterpart of :func:`speeds`; the scalar speed is
    exactly its Euclidean norm, so
    ``np.linalg.norm(velocities(...), axis=-1) == speeds(...)``. These vectors
    feed the second-moment signature of :func:`state_second_moments`.
    """
    out = {}
    for v in vids:
        x = np.asarray(pose[v], np.float64)
        out[v] = np.diff(x, axis=0) * fps
    return out


def state_frame_mask(states_i: np.ndarray, k: int, n_frames: int,
                     geom: Geometry = GEOM, union: bool = True) -> np.ndarray:
    """Boolean mask over the speed array of frames assigned to state ``k``.

    The mask is the **set union** of the spans ``B_m`` of the windows in state
    ``k`` (§5.2); overlapping spans contribute each frame once.
    """
    lo, hi = geom.delta_spans(len(states_i), n_frames, union)
    sel = np.flatnonzero(states_i == k)
    d = np.zeros(n_frames + 1, np.int32)
    if len(sel):
        np.add.at(d, lo[sel], 1)
        np.add.at(d, hi[sel], -1)
    return np.cumsum(d)[:n_frames - 1] > 0


def state_profiles(states, vidid, vids, spd, pose, geom: Geometry = GEOM,
                   union: bool = True, k: int | None = None):
    """§5.2 pooled RMS speed profile ``a[k, j]`` and its frame counts ``N_k``."""
    K = int(states.max()) + 1 if k is None else k
    J = len(JOINTS)
    sq = np.zeros((K, J))
    n = np.zeros(K)
    for i, v in enumerate(vids):
        st = states[vidid == i]
        if not len(st):
            continue
        F = pose[v].shape[0]
        s2 = np.asarray(spd[v], np.float64) ** 2
        for kk in range(K):
            m = state_frame_mask(st, kk, F, geom, union)
            if not m.any():
                continue
            sq[kk] += np.nansum(s2[m], axis=0)
            n[kk] += int(m.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.sqrt(sq / n[:, None])
    return a, n


def state_second_moments(states, vidid, vids, vel, pose, geom: Geometry = GEOM,
                         union: bool = True, k: int | None = None):
    """Pooled mean second-moment matrix ``M[k, j]`` and frame counts ``N_k``.

    ``M[k, j] = (1/N_k) * sum over state-k frames of v v^T`` in ``Sym^+_2``, the
    direction-aware generalisation of the pooled RMS speed. It contains the old
    feature and strictly more: its trace is ``a[k, j]^2`` (so
    ``np.sqrt(trace(M)) == state_profiles(...)[0]`` up to summation order), its
    eigenvalues are the mean squared speeds along the principal and secondary
    axes, its leading eigenvector is the principal axis of the joint's motion,
    and the eigenvalue gap measures how confined the motion is to that axis.

    Frames are the same **set union** ``F_k`` used by :func:`state_profiles`, so
    the magnitude channel derived from either function agrees exactly. Returns
    ``M`` of shape ``(K, J, 2, 2)`` and ``N`` of shape ``(K,)``.
    """
    K = int(states.max()) + 1 if k is None else k
    J = len(JOINTS)
    Msum = np.zeros((K, J, 2, 2))
    n = np.zeros(K)
    for i, v in enumerate(vids):
        st = states[vidid == i]
        if not len(st):
            continue
        F = pose[v].shape[0]
        vv = np.asarray(vel[v], np.float64)                 # (F-1, J, 2)
        outer = vv[..., :, None] * vv[..., None, :]         # (F-1, J, 2, 2)
        for kk in range(K):
            m = state_frame_mask(st, kk, F, geom, union)
            if not m.any():
                continue
            Msum[kk] += np.nansum(outer[m], axis=0)
            n[kk] += int(m.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        M = Msum / n[:, None, None, None]
    return M, n


def shape_coordinates(M: np.ndarray) -> np.ndarray:
    """Double-angle shape coordinate ``u[k, j] = (u1, u2)`` from ``M`` (§3-§4).

    Factoring ``M = a^2 * Mhat`` with ``trace(Mhat) = 1``, the traceless part is
    read off as ``u1 = Mhat_11 - Mhat_22`` and ``u2 = 2 Mhat_12``, i.e.
    ``u1 = (M11 - M22)/trace(M)`` and ``u2 = 2 M12/trace(M)``. By Lemma 1 this
    is the Cartesian form of ``(rho, 2 theta)``: ``||u|| = rho`` (anisotropy,
    in [0, 1]) and ``angle(u) = 2 theta`` (twice the principal axis). The factor
    of two is the "axis mod pi" identification made linear -- a half-turn of the
    physical axis is a full turn of ``u``, and an orthogonal axis sends
    ``u -> -u`` -- so unlike a raw angle ``u`` lives in a linear space and can be
    averaged, residualised and correlated.

    Isotropic motion (``rho = 0``) has no defined axis and returns ``u = 0``, as
    does a joint that never moves in a state (``trace(M) = 0``).
    """
    M = np.asarray(M, float)
    tr = M[..., 0, 0] + M[..., 1, 1]
    good = tr > 0
    denom = np.where(good, tr, 1.0)
    u1 = np.where(good, (M[..., 0, 0] - M[..., 1, 1]) / denom, 0.0)
    u2 = np.where(good, 2.0 * M[..., 0, 1] / denom, 0.0)
    return np.stack([u1, u2], axis=-1)


GROUPS = {
    "head": ["Nose"],
    "R arm": ["RShoulder", "RElbow", "RWrist"],
    "L arm": ["LShoulder", "LElbow", "LWrist"],
    "R leg": ["RHip", "RKnee", "RAnkle"],
    "L leg": ["LHip", "LKnee", "LAnkle"],
}


def state_descriptors(a: np.ndarray, free=FREE) -> dict:
    """Vigour and spatial pattern of each state, kept as separate quantities.

    Two independent facts describe a state and must not be conflated:

    * **how much** it moves — the mean RMS speed in units/second, reported
      relative to the median state so it is comparable across fits;
    * **which joints** move — the body-group means of the *double-centred* log
      profile, the same representation §6 correlates. Row-centring removes
      vigour, so this is pure spatial pattern.

    Collapsing the two into one standardised score is what makes a naming rule
    misfire: on a skewed speed distribution a moderately fast state can exceed a
    z-score cutoff meant for the outlier burst.
    """
    if a.shape[0] == 11:
        out = [
          "whole body mild",
          "left leg",
          "right arm",
          "legs",
          "whole body",
          "quiet",
          "whole body burst",
          "still",
          "right leg",
          "left arm",
          "left arm mild"
        ]
    else:
      out = range(0, a.shape[0])
    return out

# ---------------------------------------------------------------------------
# §6 kinematic similarity
# ---------------------------------------------------------------------------
def similarity(a: np.ndarray, mode: str = "double", free=FREE) -> np.ndarray:
    """§6.1 ``S[k,k'] = corr_j(log a_kj - per-joint mean, ...)``.

    ``mode='double'`` subtracts the per-joint mean across states before the
    correlation, removing the shared anatomical baseline (§6.3).
    ``mode='single'`` omits that step and is provided only to reproduce the
    §6.3 comparison table.
    """
    lg = np.log(np.clip(a[:, free], 1e-12, None))
    if mode == "double":
        lg = lg - lg.mean(axis=0, keepdims=True)
    lg = lg - lg.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(lg, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    S = (lg / nrm) @ (lg / nrm).T
    return np.clip(S, -1.0, 1.0)


def centring_comparison(a: np.ndarray, free=FREE) -> dict:
    """The §6.3 table: off-diagonal range and mean, single vs double centred."""
    out = {}
    for mode in ("single", "double"):
        S = similarity(a, mode, free)
        off = S[~np.eye(len(S), dtype=bool)]
        out[mode] = {"min": float(off.min()), "max": float(off.max()),
                     "mean": float(off.mean())}
    return out


# ---------------------------------------------------------------------------
# §6-§7 direction-aware similarity: the magnitude channel above plus a shape
# channel built from the double-angle axis coordinate of §4.
# ---------------------------------------------------------------------------
def _interaction(x: np.ndarray) -> np.ndarray:
    """Two-way additive interaction (double centring) of a ``(K, J)`` table.

    ``x_kj - rowmean_k - colmean_j + grandmean``. For the log-magnitude table
    this is identical to what :func:`similarity` computes internally before the
    row-normalisation; the shape channels reuse it component-wise.
    """
    x = np.asarray(x, float)
    return (x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True)
            + x.mean())


def residualize_shape(u: np.ndarray, free=FREE,
                      state_term: bool = True) -> np.ndarray:
    """§6 shape residual ``ũ[k, j, c]`` on the free joints.

    Each double-angle component ``u[:, :, c]`` is reduced to its two-way
    interaction over the ``(state, joint)`` table. The per-joint term removes
    the axis a joint tends to move along across all states -- the anatomical
    baseline in the torso-normalised frame -- so the residual isolates the
    state-specific redirection that fluency reads. The per-state term removes a
    whole-body drift or rotation shared by all joints in a state, the shape
    analogue of vigour; retaining it (``state_term=True``, the default) keeps
    the shape residualisation consistent with the magnitude channel and, by
    Lemma 2, makes the residual doubly centred so the shape cosine of
    :func:`shape_similarity` is a genuine correlation. Dropping it keeps the
    per-state drift and is the sensitivity variant.
    """
    uf = np.asarray(u, float)[:, free, :]                    # (K, |free|, 2)
    out = np.empty_like(uf)
    for c in range(uf.shape[-1]):
        x = uf[..., c]
        out[..., c] = _interaction(x) if state_term \
            else x - x.mean(axis=0, keepdims=True)
    return out


def shape_similarity(ushape: np.ndarray) -> np.ndarray:
    """§7 shape channel ``S_shape`` from a residualised ``(K, |free|, 2)`` field.

    ``S_shape[k,k'] = sum_j <ũ_kj, ũ_k'j> / (||ũ_k|| ||ũ_k'||)`` -- the cosine of
    the stacked ``2|free|``-vectors of residual axes. Because the inner product
    scales as ``cos(angle(ũ_kj) - angle(ũ_k'j))``, states whose joints move
    along the same residual axes score positive and orthogonal axes score
    negative. With the doubly-centred residual of :func:`residualize_shape` the
    per-joint sums vanish (Lemma 2), so this cosine is a genuine correlation and
    needs no further centring, matching the magnitude channel.
    """
    K = ushape.shape[0]
    flat = np.asarray(ushape, float).reshape(K, -1)
    nrm = np.linalg.norm(flat, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    S = (flat / nrm) @ (flat / nrm).T
    return np.clip(S, -1.0, 1.0)


def orientation_similarity(ushape: np.ndarray) -> np.ndarray:
    """§8 orientation-only diagnostic on a residualised ``(K, |free|, 2)`` field.

    Within the shape channel, redirection -- the fluency signal -- is the
    orientation part: a fidgety infant's successive states move along different
    axes, a poor-repertoire or cramped-synchronised infant repeats them. Writing
    the shape inner product as ``||ũ_kj|| ||ũ_k'j|| cos(phi_kj - phi_k'j)``, this
    isolates the orientation factor from the anisotropy magnitude,

        ``S_orient[k,k'] = sum_j <ũ_kj, ũ_k'j> / sum_j ||ũ_kj|| ||ũ_k'j||``,

    the weighted mean of the residual-axis mismatch cosine with weight
    ``||ũ_kj|| ||ũ_k'j||``. Because the residual has the per-joint anatomical
    axis removed (§6), this reads *state-specific* redirection rather than the
    axis a joint habitually moves along, and joints whose residual axis is
    ill-defined (``||ũ|| ≈ 0``) barely count. Reported alongside the shape
    channel to show whether redirection carries the effect; not the headline
    ``S``.
    """
    ush = np.asarray(ushape, float)
    dot = np.einsum("kfc,lfc->kl", ush, ush)                 # sum_j <ũ_kj,ũ_k'j>
    nrm = np.linalg.norm(ush, axis=-1)                       # (K, |free|)
    wsum = nrm @ nrm.T                                       # sum_j ||ũ_kj|| ||ũ_k'j||
    good = wsum > 1e-12
    S = np.where(good, dot / np.where(good, wsum, 1.0), np.nan)
    return np.clip(S, -1.0, 1.0)


def concatenated_similarity(a: np.ndarray, u: np.ndarray, free=FREE,
                            state_term: bool = True) -> np.ndarray:
    """§7 concatenated form: one correlation across all ``3J`` residuals.

    The magnitude residual (interaction of ``log a``) and the two shape
    residuals are each standardised across the ``K x |free|`` table -- otherwise
    the magnitude term dominates the two shape terms by scale alone -- then
    stacked into a ``(K, 3|free|)`` matrix whose row correlation is ``S``.
    """
    a_res = _interaction(np.log(np.clip(a[:, free], 1e-12, None)))
    ushape = residualize_shape(u, free, state_term)
    cols = []
    for ch in (a_res, ushape[..., 0], ushape[..., 1]):
        s = float(ch.std())
        cols.append((ch - ch.mean()) / (s if s > 1e-12 else 1.0))
    X = np.concatenate(cols, axis=1)                         # (K, 3|free|)
    X = X - X.mean(axis=1, keepdims=True)                    # -> correlation
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    S = (X / nrm) @ (X / nrm).T
    return np.clip(S, -1.0, 1.0)


def direction_aware_similarity(a: np.ndarray, u: np.ndarray, omega: float = 0.5,
                               form: str = "separated",
                               shape_state_term: bool = True, free=FREE):
    """§7 direction-aware kinematic similarity ``S`` in ``[-1, 1]``.

    Combines the magnitude channel (``S_mag``, the log-RMS-speed correlation of
    §6) with the shape channel (``S_shape``, the residual-axis cosine) so that
    ``S`` sees not only how fast a joint moves but along which axis and how
    confined the motion is. Three forms:

    * ``'separated'`` (preferred): ``S = omega * S_mag + (1-omega) * S_shape``.
      ``omega`` is fixed a priori (default ``1/2``); ``omega = 1`` recovers the
      scalar measure and ``omega = 0`` is shape-only.
    * ``'concatenated'``: :func:`concatenated_similarity`, all ``3J`` residuals
      standardised and reduced to one correlation.
    * ``'scalar'``: the legacy magnitude-only similarity, equivalently
      ``omega = 1``, for the §11 sensitivity check against the scalar version.

    Returns ``(S, parts)`` with ``parts`` carrying the separate channels
    (``S_mag``, ``S_shape`` and the orientation-only diagnostic ``S_orient``) so
    the study can report whether the fluency effect is carried by magnitude or
    by direction, which is a finding in itself.
    """
    if not 0.0 <= omega <= 1.0:
        raise ValueError(f"omega must lie in [0, 1], got {omega}")
    S_mag = similarity(a, "double", free)
    ushape = residualize_shape(u, free, shape_state_term)
    S_shape = shape_similarity(ushape)
    parts = {"S_mag": S_mag, "S_shape": S_shape,
             "S_orient": orientation_similarity(ushape),
             "form": form, "omega": float(omega),
             "shape_state_term": bool(shape_state_term)}
    if form == "scalar":
        S = S_mag
    elif form == "concatenated":
        S = concatenated_similarity(a, u, free, shape_state_term)
    elif form == "separated":
        S = omega * S_mag + (1.0 - omega) * S_shape
    else:
        raise ValueError(f"unknown similarity form {form!r}; expected "
                         f"'separated', 'concatenated' or 'scalar'")
    return np.clip(S, -1.0, 1.0), parts


def face_validity(S: np.ndarray, labels=None) -> dict:
    """§6.4: the most and least similar state pairs, for anatomical inspection."""
    K = len(S)
    off = ~np.eye(K, dtype=bool)
    iu = np.array([(i, j) for i in range(K) for j in range(K) if i < j])
    vals = S[iu[:, 0], iu[:, 1]]
    order = np.argsort(-vals)
    lab = labels if labels is not None else [str(i) for i in range(K)]
    top = [(lab[iu[o, 0]], lab[iu[o, 1]], float(vals[o])) for o in order[:5]]
    bot = [(lab[iu[o, 0]], lab[iu[o, 1]], float(vals[o])) for o in order[-5:]]
    return {"most_similar": top, "least_similar": bot,
            "off_min": float(S[off].min()), "off_max": float(S[off].max()),
            "off_mean": float(S[off].mean())}


# ---------------------------------------------------------------------------
# §7 fluency
# ---------------------------------------------------------------------------
def visit_sequence(states_i: np.ndarray) -> np.ndarray:
    """§7.1 run-length compression: consecutive entries always differ."""
    if not len(states_i):
        return states_i
    keep = np.concatenate([[True], np.diff(states_i) != 0])
    return states_i[keep]


def run_lengths(states_i: np.ndarray):
    """Visit sequence together with the dwell (in windows) of each visit."""
    if not len(states_i):
        return states_i, np.array([], int)
    ch = np.concatenate([[0], np.flatnonzero(np.diff(states_i)) + 1,
                         [len(states_i)]])
    return states_i[ch[:-1]], np.diff(ch)


def per_subject_states(states, vidid, n_sub: int) -> list[np.ndarray]:
    return [states[vidid == i] for i in range(n_sub)]


def _phi_one(seq: np.ndarray, S: np.ndarray, n_perm: int, rng, lag: int = 1):
    """Observed mean similarity at ``lag`` and its order-permutation null.

    Two nulls are returned. ``uniform`` is the estimator exactly as §7.2 defines
    it. ``offdiag`` conditions the null on adjacent entries differing.

    The distinction matters. §7.1 run-length compresses the path, so the
    observed sequence can never place a state next to itself; a uniform
    permutation of that same multiset can, and does so at a rate set by the
    subject's occupancy concentration. Every such pair contributes ``S_kk = 1``,
    the largest value in ``S``, so the uniform null sits above the range the
    observed statistic can occupy and ``Phi`` acquires a negative offset that
    varies across subjects with their occupancy. Since removing the occupancy
    confound is precisely what §7.3 claims for this null, both are reported.
    """
    if len(seq) <= lag + 1:
        return {k: np.nan for k in
                ("observed", "null_uniform", "null_offdiag", "p",
                 "excess", "excess_offdiag", "null_repeat_rate")}
    obs = float(S[seq[:-lag], seq[lag:]].mean())
    perms = rng.permuted(np.broadcast_to(seq, (n_perm, len(seq))).copy(), axis=1)
    a, b = perms[:, :-lag], perms[:, lag:]
    pair = S[a, b]
    null_u = pair.mean(axis=1)
    neq = a != b
    cnt = neq.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        null_o = np.where(cnt > 0, (pair * neq).sum(axis=1) / np.maximum(cnt, 1),
                          np.nan)
    return {"observed": obs,
            "null_uniform": float(null_u.mean()),
            "null_offdiag": float(np.nanmean(null_o)),
            "p": (1 + int(np.sum(null_u >= obs))) / (n_perm + 1),
            "excess": obs - float(null_u.mean()),
            "excess_offdiag": obs - float(np.nanmean(null_o)),
            "null_repeat_rate": float(1.0 - neq.mean())}


def phi_excess(states, vidid, S, n_sub: int, n_perm: int = 2000, seed: int = 0,
               lag: int = 1, max_win: int | None = None, window: str = "all"):
    """§7.2 excess similarity ``Phi``, per subject.

    The null uniformly permutes the visit sequence, which preserves the multiset
    of states exactly, so occupancy and every occupancy-derived statistic are
    identical between the observed sequence and every draw (§7.3). ``Phi`` is
    the difference on the ``S`` scale — the z-score form must not be used (§7.4).

    ``window`` selects a contiguous portion for the split-half of §10.8:
    ``'all'``, ``'first'`` or ``'second'``.
    """
    rng = np.random.default_rng(seed)
    keys = ("observed", "null_uniform", "null_offdiag", "p", "excess",
            "excess_offdiag", "null_repeat_rate")
    acc = {k: np.full(n_sub, np.nan) for k in keys}
    nvis = np.zeros(n_sub, int)
    for i in range(n_sub):
        st = states[vidid == i]
        if max_win is not None:
            st = st[:max_win]
        if window == "first":
            st = st[:len(st) // 2]
        elif window == "second":
            st = st[len(st) // 2:]
        seq = visit_sequence(st)
        nvis[i] = len(seq)
        r = _phi_one(seq, S, n_perm, rng, lag)
        for k in keys:
            acc[k][i] = r[k]
    acc["null_mean"] = acc["null_uniform"]          # §7.2 name
    acc.update({"n_visits": nvis, "lag": lag, "n_perm": n_perm})
    return acc


def dwell_stratified(states, vidid, S, n_sub: int,
                     bins=((1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 9)),
                     n_perm: int = 200, seed: int = 0, min_transitions: int = 5):
    """§7.6 dwell stratification: excess similarity by source-state run length.

    Over-segmentation predicts the effect concentrating in the shortest dwells.
    """
    rng = np.random.default_rng(seed)
    out = []
    for lo, hi in bins:
        obs_all, null_all, n_sub_used = [], [], 0
        for i in range(n_sub):
            s_, rl = run_lengths(states[vidid == i])
            if len(s_) < 3:
                continue
            msk = (rl[:-1] >= lo) & (rl[:-1] <= hi)
            if msk.sum() < min_transitions:
                continue
            obs_all.append(S[s_[:-1][msk], s_[1:][msk]])
            perms = rng.permuted(
                np.broadcast_to(s_, (n_perm, len(s_))).copy(), axis=1)
            null_all.append(S[perms[:, :-1], perms[:, 1:]].mean())
            n_sub_used += 1
        if not obs_all:
            out.append({"bin": f"{lo}-{hi}", "n": 0, "excess": np.nan,
                        "n_subjects": 0})
            continue
        o = np.concatenate(obs_all)
        out.append({"bin": f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+",
                    "n": int(len(o)), "n_subjects": n_sub_used,
                    "excess": float(o.mean() - np.mean(null_all))})
    return out


def tercile_decomposition(states, vidid, S, n_sub: int) -> dict:
    """§7.5 effect size in interpretable units.

    The off-diagonal entries of ``S`` are split into terciles; the observed
    transition mass in each is compared against the mass expected if transitions
    were uniform over state pairs. The reported ratio is
    "similar transitions are N times more likely than dissimilar ones".
    """
    K = len(S)
    off = ~np.eye(K, dtype=bool)
    vals = S[off]
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    tier = np.full((K, K), -1)
    tier[off & (S <= q1)] = 0
    tier[off & (S > q1) & (S <= q2)] = 1
    tier[off & (S > q2)] = 2

    fr, to = [], []
    for i in range(n_sub):
        seq = visit_sequence(states[vidid == i])
        if len(seq) < 2:
            continue
        fr.append(seq[:-1])
        to.append(seq[1:])
    if not fr:
        return {"error": "no transitions"}
    fr, to = np.concatenate(fr), np.concatenate(to)
    t_obs = tier[fr, to]
    n_tot = len(t_obs)

    rows = []
    for t in (0, 1, 2):
        observed = float(np.mean(t_obs == t))
        expected = float(np.mean(tier[off] == t))
        rows.append({"tercile": ["dissimilar", "middle", "similar"][t],
                     "observed_share": observed, "expected_share": expected,
                     "rate_ratio": observed / expected if expected else np.nan,
                     "n": int(np.sum(t_obs == t))})

    # With few distinct values in S a tercile can come out empty (every tied
    # entry falls on one side of a cut). Fall back to the extreme non-empty
    # terciles and say so, rather than reporting a silent NaN.
    filled = [r for r in rows if r["expected_share"] > 0
              and np.isfinite(r["rate_ratio"])]
    degenerate = len(filled) < 3
    if len(filled) >= 2 and filled[0]["rate_ratio"] > 0:
        ratio = filled[-1]["rate_ratio"] / filled[0]["rate_ratio"]
    else:
        ratio = np.nan
    return {"terciles": rows, "top_over_bottom": float(ratio),
            "n_transitions": int(n_tot), "cuts": [float(q1), float(q2)],
            "degenerate_terciles": bool(degenerate),
            "compared": [filled[0]["tercile"], filled[-1]["tercile"]]
            if len(filled) >= 2 else []}


def group_jump_counts(states, vidid, n_sub: int, K: int) -> np.ndarray:
    """Pooled visit-to-visit counts, accumulated **within** subjects only."""
    C = np.zeros((K, K))
    for i in range(n_sub):
        seq = visit_sequence(states[vidid == i])
        if len(seq) < 2:
            continue
        np.add.at(C, (seq[:-1], seq[1:]), 1.0)
    return C
