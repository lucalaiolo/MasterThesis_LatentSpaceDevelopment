"""A1 core: window-to-frame geometry, state kinematic signatures, fluency.

Implements METHODS §4 (geometry), §5 (state kinematic signatures), §6 (the
kinematic similarity matrix) and §7 (fluency).

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
    speed = a[:, free].mean(1)
    rel = speed / max(float(np.median(speed)), 1e-12)
    lg = np.log(np.clip(a[:, free], 1e-12, None))
    dc = lg - lg.mean(0, keepdims=True)          # remove per-joint baseline
    dc = dc - dc.mean(1, keepdims=True)          # remove vigour -> pure pattern
    cols = {JOINTS[j]: i for i, j in enumerate(free)}
    gz = {g: dc[:, [cols[c] for c in cs if c in cols]].mean(1)
          for g, cs in GROUPS.items()}
    return {"speed": speed, "relative_speed": rel, "group_pattern": gz,
            "double_centred": dc}


def state_labels(a: np.ndarray, free=FREE, pattern_tol: float = 0.35,
                 bilateral_tol: float = 0.20) -> list[str]:
    """Readable state names, as ``"<k> <vigour> (<pattern>)"``.

    A naming aid for figures only — no inference depends on these strings — but
    the vigour tier is taken from **absolute** speed relative to the median
    state, not from a z-score, so exactly the states that are genuinely slow or
    genuinely explosive get called so.
    """
    d = state_descriptors(a, free)
    rel, gz = d["relative_speed"], d["group_pattern"]
    tiers = ((0.5, "quiet"), (0.8, "low"), (1.3, "moderate"), (2.5, "active"))
    out = []
    for k in range(a.shape[0]):
        vig = next((nm for thr, nm in tiers if rel[k] < thr), "burst")
        sc = {g: gz[g][k] for g in GROUPS}
        if max(sc.values()) - min(sc.values()) < pattern_tol:
            pat = "diffuse"
        else:
            arms = (sc["R arm"], sc["L arm"])
            legs = (sc["R leg"], sc["L leg"])
            if min(arms) > 0.15 and abs(arms[0] - arms[1]) < bilateral_tol:
                pat = "arms bilateral"
            elif min(legs) > 0.15 and abs(legs[0] - legs[1]) < bilateral_tol:
                pat = "legs bilateral"
            else:
                rank = sorted(sc, key=lambda g: -sc[g])
                pat = rank[0] + (f" + {rank[1]}" if sc[rank[1]] > 0.25 else "")
        out.append(f"{k} {vig} ({pat})")
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
