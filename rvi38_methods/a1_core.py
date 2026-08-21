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
Nothing else changes: Phi and the endpoint contrast it feeds are untouched --
one statistic enters the signature and one leaves.

Two deviations from the supplied reference implementation are deliberate and
are documented where they occur:

* :func:`state_profiles` accumulates each state's frames as a **set union**, as
  §5.2 defines ``F_k = union of B_m over windows m in state k``. Consecutive
  delta windows overlap by ``l`` frames, so concatenating their spans instead
  double-counts the shared frames of every run of two or more same-state
  windows, over-weighting long dwells in the RMS.
* :func:`phi_excess` draws its null in one vectorised block rather than a
  Python loop, which is what makes the full 2,000-draw null of §12.3 affordable
  for every subject and every lag. That null reorders the visits under the
  constraint the visit sequence itself carries -- no state twice in a row --
  rather than over all ``n!`` permutations; see :func:`smirnov_sis` for why the
  unconstrained version is not occupancy-neutral despite preserving occupancy.
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


def _swap_value(X, rows, i, j, p):
    """Value at position ``p`` of each row after swapping positions ``i``, ``j``.

    Written out rather than applied, so a proposal can be tested before it is
    accepted. ``i``, ``j`` and ``p`` are per-row and ``p`` must already be in
    range; the two ``where`` calls cover the case ``|i - j| = 1``, where one of
    the affected edges has both its endpoints moved.
    """
    v = X[rows, p]
    v = np.where(p == i, X[rows, j], v)
    v = np.where(p == j, X[rows, i], v)
    return v


def smirnov_shuffle(seq: np.ndarray, n_draw: int, rng, n_steps: int | None = None,
                    rot_every: int | None = None) -> np.ndarray:
    """Draw ``n_draw`` reorderings of ``seq`` that never repeat a state (§7.2).

    The null of §7.2 reorders a subject's visits, and §7.1 run-length
    compresses the path, so a visit sequence *cannot* place a state next to
    itself. The orderings the subject could have produced are therefore the
    arrangements of its visit multiset with no two adjacent entries equal
    (Smirnov words), and those, not all ``n!`` permutations, are what the null
    must draw from. Drawing uniform permutations instead admits adjacent-equal
    pairs, each contributing ``S_kk = 1`` -- the largest value in ``S`` -- at a
    rate set by the subject's occupancy concentration, which pushes the null
    above the range the observed statistic can occupy by an amount that varies
    from subject to subject with exactly the quantity the null exists to hold
    fixed.

    Sampling is by Metropolis, because the two exact routes are both closed.
    Rejection sampling from uniform permutations accepts with probability of
    order ``exp(-n sum_k o_k^2)`` -- around ``1e-8`` at ``n = 200`` over eleven
    states -- and sequential sampling needs the number of valid completions of
    a partial word, whose exact evaluation costs ``O(K n^3)`` in big integers
    per draw. Both are hopeless at this scale, and neither buys anything a
    converged chain does not.

    The chain starts from ``seq`` itself, which is valid by construction, and
    proposes two symmetric moves, so accepting whenever the result is still
    valid leaves the uniform distribution over valid arrangements stationary:

    * a transposition of two positions, rejected when it would put two equal
      states side by side (only the at most four affected edges are examined);
    * a cyclic rotation by one place in either direction, which preserves every
      edge except the join it creates and so is valid exactly when the first and
      last entries differ. Each chain tosses its own coin for this, since a
      rotation applied to every chain on the same step is a fixed parity: on a
      rigid multiset such as ``0101...``, where rotation is the only move that
      is ever accepted, that would leave every draw on the same word.

    ``n_steps`` defaults to ``10 n``; the null estimate is stable from about
    ``2 n`` and independent of the starting word from there on.
    """
    seq = np.asarray(seq)
    n = len(seq)
    X = np.broadcast_to(seq, (int(n_draw), n)).copy()
    if n < 3:
        return X
    n_steps = max(2000, 10 * n) if n_steps is None else int(n_steps)
    rot_every = max(n // 2, 1) if rot_every is None else int(rot_every)
    rows = np.arange(int(n_draw))
    ar = np.arange(n)
    for step in range(n_steps):
        i = rng.integers(0, n, int(n_draw))
        j = rng.integers(0, n, int(n_draw))
        ok = i != j
        for p0 in (i - 1, i, j - 1, j):
            inrange = (p0 >= 0) & (p0 < n - 1)
            p = np.clip(p0, 0, n - 2)
            ok &= ~(inrange & (_swap_value(X, rows, i, j, p)
                               == _swap_value(X, rows, i, j, p + 1)))
        sel = np.flatnonzero(ok)
        if sel.size:
            si, sj = i[sel], j[sel]
            tmp = X[sel, si].copy()
            X[sel, si] = X[sel, sj]
            X[sel, sj] = tmp
        if (step + 1) % rot_every == 0:
            can = (X[:, 0] != X[:, -1]) & (rng.random(int(n_draw)) < 0.5)
            if can.any():
                d = rng.integers(0, 2, int(n_draw)) * 2 - 1
                X[can] = np.take_along_axis(
                    X, (ar[None, :] - d[:, None]) % n, axis=1)[can]
    return X


def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    m = float(x.max())
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.exp(x - m).sum()))


def smirnov_count(counts) -> int:
    """Exact number of arrangements of a multiset with no two adjacent equal.

    Block inclusion-exclusion: merging the ``n_k`` copies of state ``k`` into
    ``j_k`` ordered blocks can be done ``C(n_k - 1, j_k - 1)`` ways with sign
    ``(-1)^(n_k - j_k)``, and ``J = sum j_k`` blocks arrange freely in
    ``J! / prod(j_k!)`` ways; summed with those signs, the arrangements that
    repeat a state cancel. Writing each state's sum as the polynomial
    ``p_n(t) = sum_j (-1)^(n-j) C(n-1, j-1) t^j / j!`` turns the ``K``-fold sum
    into one polynomial product, and the count is ``sum_J c_J J!`` over the
    product's coefficients.

    Exact in ``Fraction`` and ``O(n^2)`` -- under a second up to a few hundred
    visits, a few seconds at a thousand. Its use is certification: the
    importance sampler of :func:`smirnov_sis` estimates this same number from
    its weights, so the two together check each other, which a sampler alone
    cannot do.
    """
    from fractions import Fraction
    from math import comb, factorial

    counts = [int(c) for c in counts if int(c) > 0]
    if not counts:
        return 1
    poly = [Fraction(1)]
    for n in counts:
        pk = [Fraction(0)] * (n + 1)
        for j in range(1, n + 1):
            pk[j] = Fraction((-1) ** (n - j) * comb(n - 1, j - 1), factorial(j))
        out = [Fraction(0)] * (len(poly) + n)
        for i, x in enumerate(poly):
            if x:
                for j, y in enumerate(pk):
                    if y:
                        out[i + j] += x * y
        poly = out
    total = sum(c * factorial(J) for J, c in enumerate(poly) if c)
    assert total.denominator == 1
    return int(total.numerator)


def smirnov_sis(counts, n_draw: int, rng):
    """Independent draws of no-repeat arrangements, with their log weights.

    Sequential importance sampling. The word is built left to right; at each
    step the next state is drawn from those that remain, differ from the one
    just placed, and would not strand the remainder, with probability
    proportional to how many copies are left. Every draw is therefore valid,
    but the proposal is not uniform, so each carries the log weight
    ``-log q(w)``; a weighted mean under those weights estimates any expectation
    under the uniform law over valid arrangements.

    The feasibility test is what makes the proposal dead-end-free. After placing
    ``c`` the remainder has ``r - 1`` slots and ``c`` is barred from the first
    of them, so at most ``(r-1)//2`` copies of ``c`` still fit, while any other
    state has ``r//2``. Both bounds are checked before ``c`` is offered.

    All draws walk in step, so ``r`` is shared and each step is one vectorised
    comparison over a ``(n_draw, K)`` table. Returns ``(words, log_weights)``.

    Independence is the point: there is no chain to converge, and the mean of
    ``exp(log_weights)`` estimates :func:`smirnov_count`, which is computable
    exactly. What the method does not promise is a useful *effective* sample
    size on every input -- weights degenerate as the sequence grows long and
    the occupancy concentrates -- so :func:`smirnov_null` measures the effective
    size rather than assuming it.
    """
    m0 = np.asarray(counts, int)
    K = len(m0)
    n = int(m0.sum())
    B = int(n_draw)
    if n == 0 or B == 0:
        return np.zeros((B, n), int), np.zeros(B)
    m = np.repeat(m0[None, :], B, axis=0)
    word = np.empty((B, n), int)
    logq = np.zeros(B)
    last = np.full(B, -1)
    br = np.arange(B)
    kk = np.arange(K)[None, :]
    for t in range(n):
        r = n - t
        if K > 1:
            srt = np.sort(m, axis=1)
            top1, top2 = srt[:, -1], srt[:, -2]
        else:
            top1 = top2 = m[:, 0]
        am = np.argmax(m, axis=1)
        max_excl = np.where(kk == am[:, None], top2[:, None], top1[:, None])
        ok = m > 0
        if t:
            ok = ok & (kk != last[:, None])
        ok = ok & (m - 1 <= (r - 1) // 2) & (max_excl <= r // 2)
        w = np.where(ok, m, 0).astype(float)
        tot = w.sum(axis=1)
        if not np.all(tot > 0):
            raise ValueError(
                "no valid arrangement of this multiset: some state occupies "
                "more than half the sequence, so it cannot avoid itself")
        p = w / tot[:, None]
        u = (rng.random(B)[:, None] < np.cumsum(p, axis=1)).argmax(axis=1)
        logq += np.log(p[br, u])
        word[:, t] = u
        m[br, u] -= 1
        last = u
    return word, -logq


def smirnov_null(seq: np.ndarray, S: np.ndarray, n_draw: int = 2000, rng=None,
                 lag: int = 1, max_batches: int = 16, certify: int = 400) -> dict:
    """Null mean of the adjacent similarity over uniform no-repeat reorderings.

    The estimator behind ``Phi``'s null term (§7.2). Draws come from
    :func:`smirnov_sis`, which is independent sampling rather than a chain, so
    what has to be watched is not convergence but weight degeneracy: the
    *effective* sample size
    ``ESS = (sum w)^2 / sum w^2`` falls as the visit sequence lengthens and its
    occupancy concentrates. Batches are drawn until ``ESS`` reaches ``n_draw``,
    and if ``max_batches`` is not enough the estimate is taken from
    :func:`smirnov_shuffle` instead, whose draws are unweighted and so cannot
    degenerate. Which route was used is returned in ``method`` and reported.

    Words are discarded batch by batch and only their statistic is kept, so the
    memory cost does not grow with the number of batches.

    When the sequence is short enough to certify (``certify`` visits or fewer),
    the sampler's weight-implied count of valid arrangements is compared against
    the exact :func:`smirnov_count`; ``log_count_sis`` and ``log_count_exact``
    carry both, and their agreement is a direct check that the draws are what
    they claim to be. Set ``certify=0`` to skip it.
    """
    import math

    seq = np.asarray(seq)
    n = len(seq)
    rng = np.random.default_rng() if rng is None else rng
    out = {"null": np.nan, "tail": np.nan, "ess": np.nan, "n_drawn": 0,
           "method": "none", "log_count_sis": np.nan,
           "log_count_exact": np.nan}
    if n <= lag + 1:
        return out
    obs = float(S[seq[:-lag], seq[lag:]].mean())
    counts = np.bincount(seq, minlength=int(S.shape[0]))

    lw_all: list[np.ndarray] = []
    stat_all: list[np.ndarray] = []
    ess = 0.0
    for _ in range(max(int(max_batches), 1)):
        words, lw = smirnov_sis(counts, n_draw, rng)
        stat_all.append(S[words[:, :-lag], words[:, lag:]].mean(axis=1))
        lw_all.append(lw)
        lw_cat = np.concatenate(lw_all)
        ess = float(np.exp(2 * _logsumexp(lw_cat) - _logsumexp(2 * lw_cat)))
        if ess >= n_draw:
            break

    if ess >= n_draw:
        lw_cat = np.concatenate(lw_all)
        stat = np.concatenate(stat_all)
        w = np.exp(lw_cat - lw_cat.max())
        tot = w.sum()
        out.update({"null": float((w * stat).sum() / tot),
                    "tail": float(w[stat >= obs].sum() / tot),
                    "ess": ess, "n_drawn": int(len(stat)), "method": "sis",
                    "log_count_sis": float(_logsumexp(lw_cat)
                                           - math.log(len(lw_cat)))})
        if certify and n <= int(certify):
            out["log_count_exact"] = math.log(smirnov_count(counts))
    else:
        # Weights degenerated: a long sequence whose occupancy concentrates on
        # a few states. The chain's draws are unweighted, so it is unaffected.
        draws = smirnov_shuffle(seq, n_draw, rng)
        stat = S[draws[:, :-lag], draws[:, lag:]].mean(axis=1)
        out.update({"null": float(stat.mean()),
                    "tail": float((1 + int(np.sum(stat >= obs)))
                                  / (n_draw + 1)),
                    "ess": ess, "n_drawn": int(n_draw), "method": "chain"})
    return out


def _phi_one(seq: np.ndarray, S: np.ndarray, n_perm: int, rng, lag: int = 1):
    """Observed mean similarity at ``lag`` and its order-permutation null.

    The reported null reorders the visits under the one constraint the visit
    sequence itself carries -- no state twice in a row -- drawing uniformly
    from the arrangements of the multiset that satisfy it
    (:func:`smirnov_null`). It preserves occupancy exactly, as §7.3 requires,
    and every draw lies in the same space as the observation, so ``Phi`` is the
    excess over orderings the subject could actually have produced.

    ``null_uniform`` is the unconstrained permutation null, reported alongside
    for comparison only. It admits adjacent-equal pairs at rate
    ``null_repeat_rate``, each worth ``S_kk = 1``, so it sits above the reported
    null by an offset that grows with the subject's occupancy concentration --
    the confound the constrained null removes.
    """
    keys = ("observed", "null_mean", "null_uniform", "p", "excess",
            "excess_uniform", "null_repeat_rate", "null_ess", "null_method")
    if len(seq) <= lag + 1:
        return {k: np.nan for k in keys}
    obs = float(S[seq[:-lag], seq[lag:]].mean())

    nul = smirnov_null(seq, S, n_perm, rng, lag)

    perms = rng.permuted(np.broadcast_to(seq, (n_perm, len(seq))).copy(), axis=1)
    a, b = perms[:, :-lag], perms[:, lag:]
    null_u = S[a, b].mean(axis=1)

    return {"observed": obs,
            "null_mean": nul["null"],
            "null_uniform": float(null_u.mean()),
            "p": nul["tail"],
            "excess": obs - nul["null"],
            "excess_uniform": obs - float(null_u.mean()),
            "null_repeat_rate": float(1.0 - (a != b).mean()),
            "null_ess": nul["ess"],
            "null_method": nul["method"]}


def phi_excess(states, vidid, S, n_sub: int, n_perm: int = 2000, seed: int = 0,
               lag: int = 1, max_win: int | None = None, window: str = "all"):
    """§7.2 excess similarity ``Phi``, per subject.

    The null reorders the visit sequence uniformly over the arrangements that
    never repeat a state (:func:`smirnov_null`), which preserves the multiset
    of states exactly, so occupancy and every occupancy-derived statistic are
    identical between the observed sequence and every draw (§7.3), and which
    keeps every draw inside the space the observation lives in. ``Phi`` is the
    difference on the ``S`` scale — the z-score form must not be used (§7.4).
    ``excess_uniform`` carries the unconstrained-permutation version for
    comparison; it is not the reported statistic. ``null_ess`` and
    ``null_method`` say how each subject's null was obtained -- independent
    importance draws, or the chain where their weights degenerated.

    ``window`` selects a contiguous portion for the split-half of §10.8:
    ``'all'``, ``'first'`` or ``'second'``.
    """
    rng = np.random.default_rng(seed)
    keys = ("observed", "null_mean", "null_uniform", "p", "excess",
            "excess_uniform", "null_repeat_rate", "null_ess")
    acc = {k: np.full(n_sub, np.nan) for k in keys}
    acc["null_method"] = np.array(["none"] * n_sub, dtype=object)
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
        acc["null_method"][i] = r["null_method"]
    acc.update({"n_visits": nvis, "lag": lag, "n_perm": n_perm})
    return acc


def dwell_stratified(states, vidid, S, n_sub: int,
                     bins=((1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 9)),
                     n_perm: int = 200, seed: int = 0, min_transitions: int = 5):
    """§7.6 dwell stratification: excess similarity by source-state run length.

    Over-segmentation predicts the effect concentrating in the shortest dwells.
    The null is the same no-repeat reordering :func:`phi_excess` uses, so the
    stratified excesses are on the same scale as ``Phi``.
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
            null_all.append(smirnov_null(s_, S, n_perm, rng)["null"])
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
