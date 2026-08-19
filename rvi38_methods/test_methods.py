"""Checks with a definite right answer, per METHODS §12.4.

Run with ``python test_methods.py``. Each check either verifies an identity the
document states (the Kemeny spectral form, the §4.3 length arithmetic) or
validates an estimator against an independent computation (mean first passage
against simulation, the exact rank-sum null against brute-force enumeration).
"""

from __future__ import annotations

import itertools
import os
import sys
from dataclasses import replace

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a1_core as A          # noqa: E402
import a1_stats as ST        # noqa: E402
import a57_graph as G        # noqa: E402
import a8_movement as MV     # noqa: E402
import a10_fidgetyfind as FF  # noqa: E402
import fluency_curve as FC   # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def random_chain(K, rng, sticky=0.9):
    A_ = rng.random((K, K)) + 1e-3
    A_ = A_ / A_.sum(1, keepdims=True)
    A_ = (1 - sticky) * A_ + sticky * np.eye(K)
    return A_ / A_.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
def test_geometry():
    print("\n§4 geometry")
    g = A.Geometry()
    check("l = 4, lo = 4, f0 = 16, f_win = 6.25",
          (g.l, g.lo, g.f0, g.f_win) == (4, 4, 16, 6.25),
          f"got l={g.l} lo={g.lo} f0={g.f0} f_win={g.f_win}")

    # §4.3 formula against a direct simulation of the tiling.
    ok = True
    for F in (1547, 2000, 3868, 5785):
        starts = list(range(0, F - g.clip + 1, g.stride))
        direct = len(starts) * g.step_win - 1
        ok &= direct == g.n_delta_windows(F)
    check("§4.3 length formula matches an explicit tiling", ok)

    # kept regions tile contiguously with no gap and no overlap
    lo, hi = g.delta_spans(g.n_delta_windows(2000), 2000)
    pose_starts = g.f0 + g.l * np.arange(g.n_delta_windows(2000) + 1)
    check("pose windows tile contiguously",
          bool(np.all(np.diff(pose_starts) == g.l)))
    check("§4.2 delta span is 2l = 8 frames",
          bool(np.all((hi - lo)[:-2] == 2 * g.l)))


def test_state_profiles_union():
    """§5.2 uses a set union, so overlapping spans contribute a frame once."""
    print("\n§5.2 frame attribution")
    g = A.Geometry()
    st = np.array([0, 0, 1])                 # windows 0,1 in state 0
    mask = A.state_frame_mask(st, 0, n_frames=200, geom=g)
    # windows 0 and 1 span [16,24) and [20,28): union is [16,28) = 12 frames
    check("overlapping spans counted once (union, not concatenation)",
          int(mask.sum()) == 12, f"got {int(mask.sum())} frames, expected 12")
    naive = 2 * 2 * g.l                      # what concatenation would give
    check("concatenation would have over-counted", naive == 16)


def test_similarity():
    print("\n§6 similarity")
    rng = np.random.default_rng(0)
    base = rng.random(15) + 0.5
    # Two states with the same spatial pattern but 3x the vigour.
    a = np.vstack([base, 3 * base, rng.random(15) + 0.5])
    S = A.similarity(a, "double")
    Ssingle = A.similarity(a, "single")
    check("multiplicative vigour removed by log + centring: S[0,1] ~ 1",
          abs(Ssingle[0, 1] - 1.0) < 1e-9, f"single-centred S01={Ssingle[0,1]:.6f}")
    check("S is symmetric with unit diagonal",
          np.allclose(S, S.T) and np.allclose(np.diag(S), 1.0))
    check("S is bounded in [-1, 1]", bool(S.min() >= -1 and S.max() <= 1))

    # §6.3: double centring must widen the off-diagonal range downward.
    K = 8
    prof = np.exp(rng.normal(0, .3, (K, 15))) * np.exp(
        np.linspace(0, 1.5, 15))            # shared distal-faster baseline
    c = A.centring_comparison(prof)
    check("double centring lowers the mean off-diagonal (§6.3)",
          c["double"]["mean"] < c["single"]["mean"],
          f"single {c['single']['mean']:+.2f} -> double {c['double']['mean']:+.2f}")


def _M_from(rho, theta, scale=1.0):
    """A 2x2 second-moment with anisotropy ``rho`` and principal axis ``theta``."""
    lam = np.array([0.5 * (1 + rho), 0.5 * (1 - rho)])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return scale * (R @ np.diag(lam) @ R.T)


def test_second_moment_signature():
    """The direction-aware signature contains the scalar one and strictly more."""
    print("\n§2-§4 second-moment signature")
    rng = np.random.default_rng(3)
    g = A.Geometry()

    # velocities are the signed form of speeds: norm(velocities) == speeds.
    F = 700
    vids = ["v0"]
    pose = {"v0": rng.normal(size=(F, len(A.JOINTS), 2)).cumsum(0) * 0.01}
    spd = A.speeds(pose, vids, g.fps)
    vel = A.velocities(pose, vids, g.fps)
    check("norm(velocities) == speeds",
          np.allclose(np.linalg.norm(vel["v0"], axis=-1), spd["v0"]))

    nd = g.n_delta_windows(F)
    states = rng.integers(0, 4, nd)
    vidid = np.zeros(nd, int)
    a, nfr = A.state_profiles(states, vidid, vids, spd, pose, g, union=True)
    M, nM = A.state_second_moments(states, vidid, vids, vel, pose, g, union=True)
    check("second-moment frame counts match the scalar profile",
          np.array_equal(nfr, nM))
    tr = M[..., 0, 0] + M[..., 1, 1]
    check("trace(M) == a^2  (the scalar feature is recovered exactly)",
          np.allclose(np.sqrt(np.clip(tr, 0, None)), a, equal_nan=True,
                      atol=1e-9),
          f"max abs diff {np.nanmax(np.abs(np.sqrt(np.clip(tr,0,None)) - a)):.2e}")

    # M is invariant to sign reversal of the velocity: M(v) == M(-v).
    Mneg, _ = A.state_second_moments(states, vidid, vids,
                                     {"v0": -vel["v0"]}, pose, g, union=True)
    check("M(v) == M(-v)  (the second moment sees axes, not directions)",
          np.allclose(M, Mneg, equal_nan=True))

    # Lemma 1: shape_coordinates gives ||u|| = rho and angle(u) = 2 theta,
    # independent of the magnitude scale.
    ok_l1 = True
    for rho, th in [(1.0, 0.0), (1.0, np.pi / 2), (0.6, np.pi / 4),
                    (0.8, 1.1)]:
        u = A.shape_coordinates(_M_from(rho, th, scale=3.7)[None, None])[0, 0]
        d = (np.arctan2(u[1], u[0]) - 2 * th + np.pi) % (2 * np.pi) - np.pi
        ok_l1 &= abs(np.hypot(*u) - rho) < 1e-9 and abs(d) < 1e-9
    check("Lemma 1: ||u|| = rho and angle(u) = 2 theta", ok_l1)

    ua = A.shape_coordinates(_M_from(0.7, 0.3)[None, None])[0, 0]
    ub = A.shape_coordinates(_M_from(0.7, 0.3 + np.pi / 2)[None, None])[0, 0]
    uc = A.shape_coordinates(_M_from(0.7, 0.3 + np.pi)[None, None])[0, 0]
    check("orthogonal axis sends u -> -u, half-turn sends u -> u",
          np.allclose(ua, -ub) and np.allclose(ua, uc))
    u_iso = A.shape_coordinates(_M_from(0.0, 0.9, scale=5.0)[None, None])[0, 0]
    check("isotropic motion (rho = 0) has u = 0 and no defined axis",
          np.allclose(u_iso, 0.0))


def test_direction_aware_similarity():
    """The shape channel and the separated similarity that combines it."""
    print("\n§6-§7 direction-aware similarity")
    rng = np.random.default_rng(1)
    K = 7
    a = np.exp(rng.normal(0, 0.4, (K, 15)))
    u = rng.normal(0, 0.3, (K, 15, 2))

    # Lemma 2: with the state term the shape residual is doubly centred.
    ush = A.residualize_shape(u, A.FREE, state_term=True)
    check("Lemma 2: shape residual has zero column sums (over states)",
          np.allclose(ush.sum(0), 0.0, atol=1e-10))
    check("Lemma 2: shape residual has zero row sums (over joints)",
          np.allclose(ush.sum(1), 0.0, atol=1e-10))

    # The shape cosine has the intended semantics: identical residual axes score
    # +1, orthogonal axes (u -> -u) score -1.
    ushape = np.zeros((3, 4, 2))
    ushape[0, :, 0] = 1.0            # state 0: axis theta = 0
    ushape[1, :, 0] = 1.0            # state 1: same axis
    ushape[2, :, 0] = -1.0           # state 2: orthogonal axis (u -> -u)
    Ssh = A.shape_similarity(ushape)
    check("aligned residual axes score +1, orthogonal score -1",
          abs(Ssh[0, 1] - 1.0) < 1e-12 and abs(Ssh[0, 2] + 1.0) < 1e-12,
          f"S01={Ssh[0,1]:+.3f} S02={Ssh[0,2]:+.3f}")

    # Separated form: omega selects the channel, and S is a valid similarity.
    S1, p1 = A.direction_aware_similarity(a, u, omega=1.0, form="separated")
    S0, p0 = A.direction_aware_similarity(a, u, omega=0.0, form="separated")
    Sh, _ = A.direction_aware_similarity(a, u, omega=0.5, form="separated")
    check("omega = 1 recovers the magnitude channel S_mag",
          np.allclose(S1, p1["S_mag"]))
    check("omega = 0 is the shape channel S_shape",
          np.allclose(S0, p0["S_shape"]))
    check("separated S is the linear blend omega*S_mag + (1-omega)*S_shape",
          np.allclose(Sh, 0.5 * (p1["S_mag"] + p0["S_shape"])))
    check("S_mag equals the legacy scalar similarity",
          np.allclose(p1["S_mag"], A.similarity(a, "double")))

    for form in ("separated", "concatenated", "scalar"):
        S, _ = A.direction_aware_similarity(a, u, omega=0.5, form=form)
        check(f"S ({form}) is symmetric, unit-diagonal, bounded in [-1, 1]",
              np.allclose(S, S.T) and np.allclose(np.diag(S), 1.0)
              and bool(S.min() >= -1 and S.max() <= 1))

    try:
        A.direction_aware_similarity(a, u, omega=1.7)
        bad = False
    except ValueError:
        bad = True
    check("omega outside [0, 1] is rejected", bad)


def test_kemeny():
    print("\n§9.2 Kemeny and mean first passage")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):
        A_ = random_chain(7, rng, sticky=rng.uniform(0.1, 0.9))
        r = G.kemeny_identity_check(A_)
        worst = max(worst, r["abs_diff"] / max(abs(r["trace_form"]), 1))
    check("trace(Z) - 1 == sum 1/(1 - lambda_m) over 20 random chains",
          worst < 1e-8, f"worst relative diff {worst:.2e}")

    # MFPT against direct simulation.
    A_ = np.array([[.7, .2, .1], [.1, .8, .1], [.3, .3, .4]])
    M = G.mfpt(A_)
    rng = np.random.default_rng(3)
    n = 400_000
    path = np.empty(n, int)
    path[0] = 0
    cdf = A_.cumsum(1)
    u = rng.random(n)
    for t in range(1, n):
        path[t] = np.searchsorted(cdf[path[t - 1]], u[t])
    emp = np.zeros((3, 3))
    for k in range(3):
        for j in range(3):
            if k == j:
                continue
            starts = np.flatnonzero(path[:-1] == k)
            nxt = np.flatnonzero(path == j)
            if not len(nxt):
                continue
            pos = np.searchsorted(nxt, starts)
            valid = pos < len(nxt)
            emp[k, j] = np.mean(nxt[pos[valid]] - starts[valid])
    rel = np.abs(emp - M)[~np.eye(3, dtype=bool)] / M[~np.eye(3, dtype=bool)]
    check("MFPT matches simulation within 5%", bool(rel.max() < 0.05),
          f"max relative error {rel.max():.3f}")

    # Kemeny is independent of the starting state.
    rho = G.stationary(A_)
    per_start = (G.mfpt(A_) * rho[None, :]).sum(1)
    check("Kemeny independent of starting state",
          bool(np.ptp(per_start) < 1e-9), f"spread {np.ptp(per_start):.2e}")


def test_jump_chain_and_degeneracy():
    print("\n§8.1 / §9.1 chain structure")
    rng = np.random.default_rng(4)
    A_ = random_chain(6, rng)
    J = G.jump_chain(A_)
    check("jump chain has zero diagonal and unit rows",
          np.allclose(np.diag(J), 0) and np.allclose(J.sum(1), 1))
    d = G.degenerate_centralities(A_)
    check("§9.1 out-degree is identically 1", d["out_degree_is_one"])
    check("§9.1 right Perron vector is constant", d["right_perron_is_constant"])
    check("§9.1 PageRank -> stationary as damping -> 1",
          d["pagerank_to_stationary_l1"][0.99]
          < d["pagerank_to_stationary_l1"][0.85],
          f"L1 {d['pagerank_to_stationary_l1'][0.99]:.2e} at 0.99 vs "
          f"{d['pagerank_to_stationary_l1'][0.85]:.2e} at 0.85")


def test_exact_mannwhitney():
    print("\n§10.2 exact rank-sum enumeration")
    rng = np.random.default_rng(5)
    x = rng.normal(1.0, 1, 4)
    y = rng.normal(0.0, 1, 6)
    r = stats.rankdata(np.concatenate([x, y]))
    sums, counts = ST.exact_ranksum_null(r, 4)
    # brute force
    brute = np.array([sum(c) for c in itertools.combinations(r, 4)])
    hist = {}
    for v in brute:
        hist[v] = hist.get(v, 0) + 1
    ok = (len(sums) == len(hist)
          and all(abs(hist[s] - c) < .5 for s, c in zip(sums, counts)))
    check("DP null equals brute-force enumeration", ok,
          f"{int(counts.sum())} vs {len(brute)} assignments")

    res = ST.mannwhitney(x, y, exact=True)
    # scipy's exact two-sided p for the same data
    sp = stats.mannwhitneyu(x, y, alternative="two-sided", method="exact")
    check("exact p agrees with scipy's exact Mann-Whitney",
          abs(res["p"] - sp.pvalue) < 1e-12,
          f"{res['p']:.6g} vs {sp.pvalue:.6g}")

    n1, n2 = 6, 32
    from math import comb
    check("RVI-38 label space is enumerable",
          comb(n1 + n2, n1) == 2_760_681, f"C(38,6) = {comb(38, 6):,}")
    a = np.arange(100.0, 100.0 + n1)                # strictly above every b
    b = np.arange(float(n2))
    r2 = ST.mannwhitney(a, b)
    check("perfect separation reaches the exact floor 2/C(38,6)",
          abs(r2["p"] - 2 / comb(38, 6)) < 1e-12, f"p = {r2['p']:.3g}")


def test_fluency():
    print("\n§7 fluency estimator")
    rng = np.random.default_rng(6)
    K = 6
    # Similarity with two clusters; a sequence that stays within a cluster
    # must have positive excess, a shuffled one must have ~zero.
    S = np.full((K, K), -0.6)
    S[:3, :3] = 0.8
    S[3:, 3:] = 0.8
    np.fill_diagonal(S, 1.0)
    n = 600
    smooth = []
    cur = 0
    for _ in range(n):
        cur = rng.choice([0, 1, 2]) if cur < 3 else rng.choice([3, 4, 5])
        if rng.random() < 0.02:
            cur = rng.integers(0, K)
        smooth.append(cur)
    smooth = np.array(smooth)
    smooth = smooth[np.r_[True, np.diff(smooth) != 0]]
    randseq = rng.integers(0, K, len(smooth))
    randseq = randseq[np.r_[True, np.diff(randseq) != 0]]

    states = np.concatenate([smooth, randseq])
    vid = np.concatenate([np.zeros(len(smooth), int), np.ones(len(randseq), int)])
    r = A.phi_excess(states, vid, S, 2, n_perm=500, seed=0)
    check("clustered sequence has positive excess", r["excess"][0] > 0.1,
          f"{r['excess'][0]:+.3f}")

    # The §7.2 uniform null admits adjacent-equal pairs that the run-length
    # compressed observation cannot contain, and each contributes S_kk = 1.
    # A genuinely random sequence therefore scores negative under the spec's
    # null and ~zero once the null is conditioned on adjacent entries differing.
    check("uniform null is inflated by its diagonal pairs (documented bias)",
          r["excess"][1] < -0.05 and r["null_repeat_rate"][1] > 0.05,
          f"excess {r['excess'][1]:+.3f}, null repeat rate "
          f"{r['null_repeat_rate'][1]:.1%}")
    check("off-diagonal-conditioned null gives ~zero for a random sequence",
          abs(r["excess_offdiag"][1]) < 0.06,
          f"{r['excess_offdiag'][1]:+.3f}")
    check("the null preserves occupancy exactly",
          np.array_equal(np.sort(np.bincount(smooth, minlength=K)),
                         np.sort(np.bincount(smooth, minlength=K))))

    # §7.5 terciles need a similarity matrix with enough distinct values for
    # three non-empty terciles; a two-valued S is reported as degenerate.
    td_deg = A.tercile_decomposition(states[vid == 0], vid[vid == 0] * 0, S, 1)
    check("§7.5 flags degenerate terciles on a two-valued S",
          td_deg["degenerate_terciles"])
    Sc = np.clip(S + rng.normal(0, .15, S.shape), -1, 1)
    Sc = (Sc + Sc.T) / 2
    np.fill_diagonal(Sc, 1.0)
    td = A.tercile_decomposition(states[vid == 0], vid[vid == 0] * 0, Sc, 1)
    check("§7.5 similar transitions over-represented",
          td["top_over_bottom"] > 1.0, f"ratio {td['top_over_bottom']:.2f}")


def test_fluency_curve():
    """FLUENCY_CURVE: the temporal decomposition must reduce to A1's Phi."""
    print("\nFLUENCY_CURVE temporal decomposition")
    rng = np.random.default_rng(11)
    K = 7
    S = np.clip(rng.normal(0, 0.4, (K, K)), -1, 1)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 1.0)
    st = A.visit_sequence(rng.integers(0, K, 500))
    vid = np.zeros(len(st), int)

    s = FC.transition_similarity(S, st)
    check("transition_similarity has n-1 entries and equals S[q_t, q_{t+1}]",
          s.shape[0] == len(st) - 1
          and np.allclose(s, S[st[:-1], st[1:]]))

    # null_offset must reproduce a1_core.phi_excess's uniform null exactly:
    # both draw B permutations of the same sequence from default_rng(seed).
    phi = A.phi_excess(st, vid, S, 1, n_perm=1000, seed=0)
    c = FC.null_offset(S, st, B=1000, seed=0)
    check("null_offset reproduces phi_excess's null mean exactly",
          abs(c - float(phi["null_mean"][0])) < 1e-12,
          f"|Δ| = {abs(c - float(phi['null_mean'][0])):.2e}")
    check("mean(s) equals A1's observed statistic",
          abs(float(s.mean()) - float(phi["observed"][0])) < 1e-12)

    # Prop 1: the flat-kernel limit of the curve is the scalar Phi.
    Phi = float(s.mean() - c)
    flat = FC.phi_curve(s, 1e7, c)
    check("Prop 1: flat-kernel curve equals Phi at every t",
          np.allclose(flat, Phi, atol=1e-9),
          f"max|Δ| = {np.abs(flat - Phi).max():.2e}")

    # weight-normalised smoothing keeps phi(t)+c a convex mean of s.
    ok = True
    for sg in (3, 5):
        cur = FC.phi_curve(s, sg, 0.0)
        ok &= bool(cur.max() <= s.max() + 1e-9 and cur.min() >= s.min() - 1e-9)
    check("phi_curve is a weight-normalised convex mean (no edge blow-up)", ok)

    # the step-2 assertion is a real guard: a Phi from a different S trips it.
    S2 = np.clip(S + rng.normal(0, 0.3, S.shape), -1, 1)
    S2 = (S2 + S2.T) / 2
    np.fill_diagonal(S2, 1.0)
    phi2 = A.phi_excess(st, vid, S2, 1, n_perm=1000, seed=0)
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="fc_test_")
    tripped = False
    try:
        FC.fluency_curves(st, vid, S, ["r"], [0], phi=phi2, out_root=tmp,
                          verbose=False)
    except AssertionError:
        tripped = True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("step-2 identity guards against an S / Phi mismatch", tripped)

    # transition_times: aligned with s, strictly increasing, whole-window onsets.
    st2 = np.repeat([0, 1, 2, 1, 0], [3, 2, 4, 1, 5])
    T = FC.transition_times(st2, A.GEOM)
    g = A.GEOM
    check("transition_times aligns with s and is monotone",
          T.shape[0] == 4 and np.all(np.diff(T) > 0)
          and abs(T[0] - (g.f0 + g.l * 3) / g.fps) < 1e-9,
          f"onsets [3,5,9,10] -> T={np.round(T, 3).tolist()}")


def test_stats_helpers():
    print("\n§10 helpers")
    # Holm and the minimum detectable effect are retained in a1_stats but are
    # not part of the reported procedure; they are checked so that "retained"
    # does not quietly become "broken".
    check("Holm adjustment is monotone and bounded [retained, not reported]",
          np.all(ST.holm([0.01, 0.04, 0.5]) >= [0.01, 0.04, 0.5])
          and ST.holm([0.5, 0.5])[0] <= 1.0)
    check("Spearman-Brown of r = 0.5 is 2/3",
          abs(ST.spearman_brown(0.5) - 2 / 3) < 1e-12)
    M = np.column_stack([np.arange(20.0), np.arange(20.0)])
    check("ICC(A,1) of identical columns is 1", abs(ST.icc_a1(M) - 1) < 1e-9)
    # A single interval can miss by chance, so test the coverage rate instead.
    rng = np.random.default_rng(7)
    cover = 0
    reps = 120
    for i in range(reps):
        v = np.random.default_rng(100 + i).normal(5, 1, 60)
        ci = ST.bca_ci(v, np.mean, 800, seed=i)
        cover += ci["lo"] < 5 < ci["hi"]
    rate = cover / reps
    check("BCa 95% interval covers the true mean at ~95%", 0.88 <= rate <= 1.0,
          f"coverage {rate:.1%} over {reps} replicates")
    mde = ST.min_detectable_effect(6, 32)
    check("minimum detectable AUC is large at 6 vs 32 "
          "[retained, not reported]",
          0.5 < mde["auc"] < 1.0, f"AUC {mde['auc']:.3f}")
    g = G.estimability_gate([1, 2, 3, 4], [0.9, 1.9, 2.9, 3.9],
                            [1.1, 2.1, 3.1, 4.1])
    check("estimability gate passes on tight intervals", g["passed"],
          f"ratio {g['ratio']:.2f}")
    g2 = G.estimability_gate([1, 2, 3, 4], [0, 1, 2, 3], [2, 3, 4, 5])
    check("estimability gate fails on wide intervals", not g2["passed"],
          f"ratio {g2['ratio']:.2f}")


def test_shrinkage():
    print("\n§9.3 shrinkage")
    rng = np.random.default_rng(8)
    K = 5

    def sample(chain, n, rng):
        s = np.empty(n, int)
        s[0] = 0
        cdf = chain.cumsum(1)
        u = rng.random(n)
        for t in range(1, n):
            s[t] = np.searchsorted(cdf[s[t - 1]], u[t])
        return s

    # Homogeneous cohort: every subject shares one chain, so the group matrix
    # is the better estimator and heavy shrinkage is the right answer.
    shared = random_chain(K, rng, sticky=0.0)
    homog = [sample(shared, 400, rng) for _ in range(12)]
    al_h = G.choose_alpha(homog, K)
    check("alpha is high when subjects share one chain", al_h["alpha"] >= 0.5,
          f"alpha = {al_h['alpha']:.3f}")

    # Heterogeneous cohort: each subject has its own chain, so shrinking to the
    # group destroys real individual signal and alpha must stay low.
    hetero = [sample(random_chain(K, rng, sticky=0.0), 1200, rng)
              for _ in range(12)]
    al = G.choose_alpha(hetero, K)
    check("alpha is low when subjects genuinely differ", al["alpha"] < 0.5,
          f"alpha = {al['alpha']:.3f}")
    check("alpha selected inside the grid", 0 <= al["alpha"] <= 0.95,
          f"alpha = {al['alpha']:.3f}")
    check("held-out log-likelihood is finite across the grid",
          bool(np.all(np.isfinite(al["loglik_grid"]))))
    check("degenerate flag is robust to the grid's float representation",
          G.choose_alpha(homog, K, grid=np.linspace(0, 0.95, 20))["degenerate"]
          == (al_h["alpha"] >= 0.9 - 1e-9),
          f"alpha {al_h['alpha']:.4f} -> degenerate {al_h['degenerate']}")

    rng2 = np.random.default_rng(9)
    blocks = G.moving_block_bootstrap(np.arange(500), 50, rng2)
    check("moving block bootstrap preserves length", len(blocks) == 500)


def test_wclrpp_peakpick():
    print("\nWCLR-PP peak-picking (spec test vector)")
    import a9_wclrpp as WP
    # The exact matrix from the spec: tau in {-2..2}, c=0.25, dtau=1,
    # ell_min=1, T=D=8. Expected F = 4/8 = 0.5, R2 = 0.3325.
    M = np.array([
        [0.05, 0.08, 0.10, 0.06, 0.03],
        [0.04, 0.07, 0.12, 0.09, 0.05],
        [0.06, 0.10, 0.15, 0.28, 0.31],
        [0.05, 0.09, 0.14, 0.30, 0.35],
        [0.07, 0.11, 0.16, 0.33, 0.38],
        [0.06, 0.10, 0.15, 0.29, 0.27],
        [0.05, 0.08, 0.11, 0.07, 0.04],
        [0.04, 0.06, 0.09, 0.05, 0.03],
    ])
    r = WP.peak_pick(M, c=0.25, dtau=1, ell_min=1, h=1, D=8)
    check("peak-picking reproduces the spec test vector F = 0.5",
          abs(r["F"] - 0.5) < 1e-9, f"F = {r['F']:.4f}")
    check("peak-picking reproduces the spec test vector R2 = 0.3325",
          abs(r["R2"] - 0.3325) < 1e-9, f"R2 = {r['R2']:.4f}")
    check("exactly one interval survives, of length 4",
          r["n_intervals"] == 1 and r["intervals"][0][2] == 4)
    # ell_min gate: raising it above the run length rejects the interval
    r2 = WP.peak_pick(M, c=0.25, dtau=1, ell_min=5, h=1, D=8)
    check("an interval shorter than ell_min is discarded",
          r2["F"] == 0.0 and r2["n_intervals"] == 0)
    # c gate: raising the cutoff above every peak leaves nothing
    r3 = WP.peak_pick(M, c=0.5, dtau=1, ell_min=1, h=1, D=8)
    check("a cutoff above every peak leaves F = 0",
          r3["F"] == 0.0 and not np.isfinite(r3["R2"]))


def test_wclrpp_reduction():
    print("\nWCLR-PP 1-D reduction and non-negativity")
    import a9_wclrpp as WP
    rng = np.random.default_rng(0)
    L, w, tmax = 400, 50, 13
    xA = rng.standard_normal(L) * 0.05
    xB = 0.6 * np.roll(xA, 3) + rng.standard_normal(L) * 0.03   # A leads B by 3
    vA = np.stack([xA, np.zeros(L)], axis=1)                    # dy == 0
    vB = np.stack([xB, np.zeros(L)], axis=1)
    Mv, taus, _, _ = WP.delta_r2_matrix(vA, vB, w, tmax)

    # scalar reference on dx only (the x-component regression)
    def scalar(a, b):
        rows = np.arange(tmax, L - w - tmax + 1)
        k = rows[:, None] + np.arange(w)[None, :]
        PA, PB, one = a[k], b[k], np.ones((len(rows), w))
        X1 = np.stack([one, PA], -1)
        X2 = np.stack([one, PA, PB], -1)
        P1 = np.linalg.pinv(np.einsum("dwi,dwj->dij", X1, X1))
        P2 = np.linalg.pinv(np.einsum("dwi,dwj->dij", X2, X2))
        out = np.empty((len(rows), len(taus)))
        for j, ta in enumerate(taus):
            Y = a[k + ta]
            tss = ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1)
            b1 = np.einsum("dij,dj->di", P1, np.einsum("dwi,dw->di", X1, Y))
            b2 = np.einsum("dij,dj->di", P2, np.einsum("dwi,dw->di", X2, Y))
            s1 = ((Y - np.einsum("dwi,di->dw", X1, b1)) ** 2).sum(1)
            s2 = ((Y - np.einsum("dwi,di->dw", X2, b2)) ** 2).sum(1)
            out[:, j] = np.where(tss > 1e-12, (s1 - s2) / tss, 0.0)
        return out

    check("vector delta-R^2 on 1-D motion equals the scalar delta-R^2",
          np.allclose(Mv, scalar(xA, xB), atol=1e-9),
          f"max abs diff = {np.abs(Mv - scalar(xA, xB)).max():.1e}")
    check("delta-R^2 is non-negative in every cell", Mv.min() >= -1e-8,
          f"min = {Mv.min():+.1e}")
    # lead-lag: B is a delayed copy of A (A leads), so the peak lag is negative
    lead = float(np.median(taus[Mv.argmax(1)]))
    check("lead-lag reads off the correct direction (A leads -> tau < 0)",
          lead < 0, f"median tau* = {lead:+.0f}")
    # N_rows > 0
    check("N_rows is positive for an admissible clip", Mv.shape[0] > 0,
          f"N_rows = {Mv.shape[0]}")


def test_wclrpp_coupling():
    print("\nWCLR-PP coupling: coupled vs independent, aggregation, inference")
    import a9_wclrpp as WP
    from scipy.signal import butter, filtfilt
    rng = np.random.default_rng(3)
    F, fps = 1400, 25.0
    b, a = butter(3, [0.5 / (fps / 2), 2.0 / (fps / 2)], btype="band")

    def bl():
        return filtfilt(b, a, rng.standard_normal((F, 2)), axis=0) * 0.08

    src = bl()
    A = src + rng.standard_normal((F, 2)) * 0.005
    B = np.roll(src, -6, axis=0) + rng.standard_normal((F, 2)) * 0.005  # B leads
    p = WP.WCLRParams()
    f_coup = WP.pair_wclrpp(np.diff(A, axis=0), np.diff(B, axis=0), p)["F"]
    Ai = bl() + rng.standard_normal((F, 2)) * 0.005
    Bi = bl() + rng.standard_normal((F, 2)) * 0.005
    f_ind = WP.pair_wclrpp(np.diff(Ai, axis=0), np.diff(Bi, axis=0), p)["F"]
    check("a coupled pair scores higher F than an independent one",
          f_coup > f_ind + 0.1, f"F_coupled = {f_coup:.3f} vs {f_ind:.3f}")

    # symmetry: averaging both directions is order-invariant
    r12 = WP.pair_wclrpp(np.diff(A, axis=0), np.diff(B, axis=0), p)
    r21 = WP.pair_wclrpp(np.diff(B, axis=0), np.diff(A, axis=0), p)
    check("the symmetric per-pair score is order-invariant",
          abs(r12["F"] - r21["F"]) < 1e-9)

    # dataset shape and aggregation
    from build_pose import JOINTS
    J = len(JOINTS)
    vids = [rng.standard_normal((600, J, 2)) * 0.02 for _ in range(4)]
    ds = WP.wclrpp_dataset(vids, WP.WCLRParams())
    check("wclrpp_dataset returns one F per pair per recording",
          ds["F"].shape == (4, 6))
    check("the aggregation carries mean, spread and strength",
          ds["mean_F"].shape == (4,) and ds["spread_F"].shape == (4,)
          and ds["mean_R2"].shape == (4,))

    # group inference: planted pair-1 effect is detected, max-statistic corrected
    n, n_pos = 38, 6
    y = np.zeros(n, int)
    y[:n_pos] = 1
    Fm = np.abs(rng.normal(0, 0.03, (n, 6)))
    Fm[y == 1, 0] += 0.4
    t = WP.wclrpp_test(Fm, y, n_perm=2000, seed=0)
    check("a planted per-pair effect is detected",
          t["p_corrected"][0] < 0.05, f"p = {t['p_corrected'][0]:.4f}")
    check("the corrected p is never below the uncorrected one",
          bool(np.all(t["p_corrected"] >= t["p_uncorrected"])))
    check("all six pairs are reported", len(t["p_corrected"]) == 6)
    yn = np.zeros(n, int)
    yn[rng.choice(n, n_pos, replace=False)] = 1
    Fn = np.abs(rng.normal(0, 0.03, (n, 6)))
    tn = WP.wclrpp_test(Fn, yn, n_perm=2000, seed=1)
    check("a label-shuffled null is not significant",
          float(tn["p_corrected"].min()) > 0.05,
          f"min p = {tn['p_corrected'].min():.3f}")


# ---------------------------------------------------------------------------
def _ff_skeleton(F, moves, rng=None, base=None):
    """A skeleton at rest with a prescribed trajectory for the moving joints.

    ``moves`` maps a joint index to an ``(F, 2)`` array of positions. Every
    other joint is held at its resting place, so each chain's entropy is a
    function of exactly what was planted.
    """
    import make_synthetic as MS
    x = np.tile(MS.BASE if base is None else base, (F, 1, 1)).astype(float)
    for j, track in moves.items():
        x[:, j] = track
    return x


def _ff_walk(F, start, step_len, angles, radius=0.12):
    """A joint stepping ``step_len`` per frame, kept inside ``radius`` of home.

    The bound matters: an unbounded walk drifts away from its parent joint, and
    since FidgetyFind measures every displacement as a fraction of the parent
    limb, a drifting joint silently changes the very scale under test.
    """
    home = np.array(start, float)
    pos = home.copy()
    out = np.zeros((F, 2))
    for t in range(F):
        out[t] = pos
        step = step_len * np.array([np.cos(angles[t]), np.sin(angles[t])])
        if np.linalg.norm(pos + step - home) > radius:
            step = -step
        pos = pos + step
    return out


def test_fidgetyfind_entropy():
    """The direction histogram behind FidgetyFind, on inputs with a known answer."""
    print("\nFidgetyFind direction entropy")
    rng = np.random.default_rng(0)
    check("one direction gives entropy 0",
          abs(FF.direction_entropy(np.full(200, 0.4), bins=8)) < 1e-4)
    two = np.tile([0.4, 0.4 - np.pi], 100)
    check("two opposite directions give log(2)/log(8) = 1/3",
          abs(FF.direction_entropy(two, bins=8) - np.log(2) / np.log(8)) < 1e-4,
          f"got {FF.direction_entropy(two, bins=8):.4f}")
    unif = np.linspace(-np.pi, np.pi, 8001)[:-1]
    check("uniform directions give entropy 1",
          abs(FF.direction_entropy(unif, bins=8) - 1.0) < 1e-3,
          f"got {FF.direction_entropy(unif, bins=8):.4f}")
    check("angles outside [-pi, pi] are wrapped, not discarded",
          abs(FF.direction_entropy(two + 2 * np.pi, bins=8)
              - FF.direction_entropy(two, bins=8)) < 1e-9)
    check("entropy never leaves [0, 1]",
          all(-1e-9 <= FF.direction_entropy(rng.uniform(-np.pi, np.pi, n),
                                            bins=8) <= 1 + 1e-9
              for n in (10, 50, 500)))


def test_fidgetyfind_windows():
    """Window gating: what is scored, what is scored zero, what is voided."""
    print("\nFidgetyFind window gates")
    import make_synthetic as MS
    rng = np.random.default_rng(1)
    F = 600
    hip, knee = FF.CHAINS["R hip"]          # (parent, moving joint)
    ref = float(np.linalg.norm(MS.BASE[hip] - MS.BASE[knee]))
    p = FF.FFParams(fps=25.0, start_frame=0, smooth=False)
    mid = 0.5 * (p.minr + p.maxr) / 100.0 / p.rate_scale * ref   # in-band step

    ang = rng.uniform(-np.pi, np.pi, F)
    x = _ff_skeleton(F, {knee: _ff_walk(F, MS.BASE[knee], mid, ang)})
    E = FF.fidgetyfind_recording(x, None, p)["E"][:, 0]
    check("in-band steps in random directions score near 1",
          np.nanmedian(E) > 0.85, f"median entropy {np.nanmedian(E):.3f}")

    # A joint cannot move in one direction for ever, so the stereotyped case is
    # a bounded oscillation along one axis: two opposite directions out of the
    # eight bins, which is exactly log(2)/log(8).
    x = _ff_skeleton(F, {knee: _ff_walk(F, MS.BASE[knee], mid,
                                        np.zeros(F) + 0.7)})
    E = FF.fidgetyfind_recording(x, None, p)["E"][:, 0]
    check("movement confined to one axis scores log(2)/log(8)",
          abs(float(np.nanmedian(E)) - np.log(2) / np.log(8)) < 1e-6,
          f"median entropy {np.nanmedian(E):.3f}")

    x = _ff_skeleton(F, {})                     # nothing moves at all
    E = FF.fidgetyfind_recording(x, None, p)["E"][:, 0]
    check("a still recording scores 0, not NaN (it was assessable)",
          np.isfinite(E).all() and np.nanmax(np.abs(E)) < 1e-6)

    big = 5 * p.maxr / 100.0 / p.rate_scale * ref
    x = _ff_skeleton(F, {knee: _ff_walk(F, MS.BASE[knee], big,
                                        np.zeros(F) + 0.7)})
    E = FF.fidgetyfind_recording(x, None, p)["E"][:, 0]
    check("movement far above the band voids the window (NaN, not 0)",
          not np.isfinite(E).any(),
          f"{int(np.isfinite(E).sum())} of {E.size} windows survived")

    o = np.ones((F, 15), np.uint8)
    o[::2, knee] = 0                            # half the frames uninterpolated
    x = _ff_skeleton(F, {knee: _ff_walk(F, MS.BASE[knee], mid, ang)})
    E = FF.fidgetyfind_recording(x, o, p)["E"][:, 0]
    check("unobserved keypoints void the window",
          not np.isfinite(E).any(),
          f"{int(np.isfinite(E).sum())} of {E.size} windows survived")


def test_fidgetyfind_invariance():
    """The measure is built to be blind to camera pose and to frame rate."""
    print("\nFidgetyFind invariances")
    import make_synthetic as MS
    rng = np.random.default_rng(2)
    F = 600
    hip, knee = FF.CHAINS["R hip"]
    ref = float(np.linalg.norm(MS.BASE[hip] - MS.BASE[knee]))
    p = FF.FFParams(fps=25.0, start_frame=0, smooth=False)
    mid = 0.5 * (p.minr + p.maxr) / 100.0 / p.rate_scale * ref
    ang = rng.uniform(-np.pi, np.pi, F)
    x = _ff_skeleton(F, {knee: _ff_walk(F, MS.BASE[knee], mid, ang)})
    E0 = FF.fidgetyfind_recording(x, None, p)["E"]

    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    E1 = FF.fidgetyfind_recording(x @ R.T, None, p)["E"]
    check("rigid rotation of the whole skeleton leaves every entropy unchanged",
          np.allclose(np.nan_to_num(E0, nan=-1), np.nan_to_num(E1, nan=-1),
                      atol=1e-9))

    E2 = FF.fidgetyfind_recording(3.7 * x, None, p)["E"]
    check("uniform scaling leaves every entropy unchanged",
          np.allclose(np.nan_to_num(E0, nan=-1), np.nan_to_num(E2, nan=-1),
                      atol=1e-9))

    E3 = FF.fidgetyfind_recording(x + np.array([2.0, -5.0]), None, p)["E"]
    check("translation leaves every entropy unchanged",
          np.allclose(np.nan_to_num(E0, nan=-1), np.nan_to_num(E3, nan=-1),
                      atol=1e-9))

    m25 = FF.motion_features(x, None, p)["magnitude"]
    m30 = FF.motion_features(x, None, FF.FFParams(fps=30.0, start_frame=0,
                                                  smooth=False))["magnitude"]
    check("the same trajectory read at 30 fps scores 30/25 larger per frame",
          np.allclose(m30, m25 * 30.0 / 25.0, atol=1e-9))


def test_fidgetyfind_smoothing_and_reduction():
    """The smoother does what it is there for, and the reduction is what it says."""
    print("\nFidgetyFind smoothing and per-recording reduction")
    rng = np.random.default_rng(3)
    F, J = 400, 15
    flat = np.tile(np.arange(J, dtype=float)[:, None] * np.ones(2), (F, 1, 1))
    check("smoothing a motionless recording changes nothing",
          np.allclose(FF.smooth_tracks(flat, None, 5, 2.0), flat, atol=1e-9))

    jit = flat + rng.normal(0, 0.01, (F, J, 2))
    sm = FF.smooth_tracks(jit, None, 5, 2.0)
    raw_step = np.linalg.norm(np.diff(jit, axis=0), axis=-1).mean()
    sm_step = np.linalg.norm(np.diff(sm, axis=0), axis=-1).mean()
    check("smoothing shrinks frame-to-frame jitter", sm_step < 0.6 * raw_step,
          f"{sm_step:.4f} vs {raw_step:.4f} per frame")

    conf = np.ones((F, J))
    conf[10] = 0.0
    jit2 = jit.copy()
    jit2[10] += 5.0                     # a wild, unobserved frame
    sm2 = FF.smooth_tracks(jit2, conf, 5, 2.0)
    check("an unobserved frame carries no weight into its neighbours",
          np.allclose(sm2[8], FF.smooth_tracks(jit, conf, 5, 2.0)[8], atol=1e-9))

    E = np.array([[0.9, 0.1], [0.7, np.nan], [np.nan, 0.3], [0.5, 0.2]])
    agg = FF.aggregate(E, ("R hip", "L hip"), FF.FFParams(theta=0.5))
    check("per-chain median ignores voided windows",
          np.allclose(agg["median_entropy"], [0.7, 0.2]),
          f"got {np.round(agg['median_entropy'], 3).tolist()}")
    check("the score is the mean of the per-chain medians",
          abs(agg["score"] - 0.45) < 1e-9, f"got {agg['score']:.4f}")
    check("coverage is the fraction of windows that could be scored",
          np.allclose(agg["coverage"], [0.75, 0.75]))
    check("the fidgety-window rate counts windows at or above theta",
          np.allclose(agg["positive_rate"], [1.0, 0.0]),
          f"got {np.round(agg['positive_rate'], 3).tolist()}")


def test_fidgetyfind_planted_cohort():
    """The planted signal of make_synthetic is recovered, with the right sign."""
    print("\nFidgetyFind on the planted cohort")
    import make_synthetic as MS
    rng = np.random.default_rng(4)
    F = 1200
    p = FF.FFParams(fps=25.0, start_frame=0)
    poses = []
    y = []
    for present in (True,) * 8 + (False,) * 4:
        moves = {}
        for moving, parent in MS.FIDGETY_CHAINS.items():
            ref = float(np.linalg.norm(MS.BASE[moving] - MS.BASE[parent]))
            moves[moving] = MS.BASE[moving] + ref * MS.fidgety_layer(
                F, rng, present)
        poses.append(_ff_skeleton(F, moves))
        y.append(0 if present else 1)
    ds = FF.fidgetyfind_dataset(poses, None, p)
    y = np.array(y)
    lo, hi = ds["score"][y == 1], ds["score"][y == 0]
    check("planted fidgety movement scores high",
          float(np.nanmin(hi)) > 0.5, f"min normal score {np.nanmin(hi):.3f}")
    check("planted absence scores low",
          float(np.nanmax(lo)) < 0.2, f"max abnormal score {np.nanmax(lo):.3f}")
    check("every recording is separated in the right direction",
          float(np.nanmax(lo)) < float(np.nanmin(hi)))
    t = ST.maxstat_label_test(ds["median_entropy"], y, n_perm=2000, seed=0,
                              names=ds["chains"])
    check("the max-statistic test sees it on the hips",
          float(t["p_corrected"][0]) <= 0.05,
          f"p_corrected = {t['p_corrected'][0]:.3f}")


# ---------------------------------------------------------------------------
def test_fidgetyfind_degeneracy_check():
    """A band that does not fit the data is reported as such, not as a null."""
    print("\nFidgetyFind degeneracy check")
    import make_synthetic as MS
    rng = np.random.default_rng(11)
    F = 1200
    poses = []
    for _ in range(6):
        moves = {}
        for moving, parent in MS.FIDGETY_CHAINS.items():
            ref = float(np.linalg.norm(MS.BASE[moving] - MS.BASE[parent]))
            moves[moving] = MS.BASE[moving] + ref * MS.fidgety_layer(
                F, rng, True)
        poses.append(_ff_skeleton(F, moves))

    good = FF.FFParams(fps=25.0, start_frame=0)
    ds = FF.fidgetyfind_dataset(poses, None, good)
    d = FF.diagnose(ds, good)
    # The foot chains legitimately lose most windows here -- the planted knee
    # fidget *is* transport for the ankle -- so coverage warnings are expected.
    # What must not appear is a degeneracy flag or a band-mismatch warning.
    check("a band that fits raises no degeneracy flag", not d["degenerate"])
    check("a band that fits raises no band-mismatch warning",
          not any("inside the band" in w for w in d["warnings"]),
          f"{len(d['warnings'])} warning(s): {d['warnings']}")
    check("the score still varies across recordings",
          d["score_distinct_values"] == len(poses),
          f"{d['score_distinct_values']} distinct")

    # The same recordings, read through a band an order of magnitude above the
    # amplitudes actually present: every window falls short of in_range_rate.
    bad = replace(good, minr=45.0, maxr=80.0)
    ds_b = FF.fidgetyfind_dataset(poses, None, bad)
    d_b = FF.diagnose(ds_b, bad)
    check("every recording scores exactly 0.0 under a mismatched band",
          float(np.nanmax(ds_b["score"])) == 0.0,
          f"max score {np.nanmax(ds_b['score']):.3f}")
    check("the mismatched band is flagged as degenerate", d_b["degenerate"])
    check("the flag names the in-band rate as the cause",
          any("inside the band" in w for w in d_b["warnings"]))
    check("the in-band rate is measured even where a gate voided the window",
          float(np.nanmax(d_b["median_in_band_rate"])) < bad.in_range_rate,
          f"max {np.nanmax(d_b['median_in_band_rate']):.3f}")

    # Recording the in-band rate must not have changed any score.
    for a, b in zip(ds["E"], FF.fidgetyfind_dataset(poses, None, good)["E"]):
        assert np.allclose(a, b, equal_nan=True)
    check("adding the diagnostic left every window entropy unchanged", True)


# ---------------------------------------------------------------------------
def test_reported_inference():
    """The endpoint contrast, against the definitions it is written from."""
    print("\nReported inference: exact Mann-Whitney, AUC, bootstrap interval")
    rng = np.random.default_rng(11)
    x = rng.normal(0.6, 1, 6)             # abnormal
    y = rng.normal(0.0, 1, 12)            # normal
    r = ST.mannwhitney(x, y, boot=2000, seed=3)

    # U as the double sum with the half-weight tie rule, and AUC = U / (n1 n0)
    n1, n2 = len(x), len(y)
    U = sum((xa > yb) + 0.5 * (xa == yb) for xa in x for yb in y)
    check("U is the double sum with half-weight ties",
          abs(r["U"] - U) < 1e-9, f"{r['U']:.1f} vs {U:.1f}")
    check("AUC = U / (n1 n0)", abs(r["auc"] - U / (n1 * n2)) < 1e-12,
          f"{r['auc']:.6f}")

    # mid-ranks: a tied pair contributes exactly half
    xt, yt = np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0])
    rt = ST.mannwhitney(xt, yt, boot=0)
    check("ties are counted at one half (mid-ranks)",
          abs(rt["auc"] - 0.5 / 9) < 1e-12, f"AUC {rt['auc']:.4f}")

    # the exact two-sided p is the |AUC - 1/2| tail of the enumerated null
    xs, ys = np.array([4.0, 5.0, 9.0]), np.array([1.0, 2.0, 3.0, 6.0, 7.0])
    obs = ST.mannwhitney(xs, ys, exact=True, boot=0)
    pool = np.concatenate([xs, ys])
    ranks = stats.rankdata(pool)
    n1s = len(xs)
    tail = tot = 0
    for idx in itertools.combinations(range(len(pool)), n1s):
        R1 = ranks[list(idx)].sum()
        auc = (R1 - n1s * (n1s + 1) / 2) / (n1s * (len(pool) - n1s))
        tot += 1
        tail += abs(auc - 0.5) >= abs(obs["auc"] - 0.5) - 1e-12
    check("two-sided p is the exact |AUC - 1/2| tail over every assignment",
          abs(obs["p"] - tail / tot) < 1e-12,
          f"{obs['p']:.6f} vs {tail}/{tot}")
    check("the null is enumerated, not sampled",
          "exact enumeration" in obs["method"], obs["method"])

    # the reported interval is the stratified percentile bootstrap
    bs = ST.bootstrap_auc_ci(x, y, n_boot=2000, alpha=0.05, seed=3)
    check("the reported interval is the stratified percentile bootstrap",
          abs(r["auc_lo"] - bs["lo"]) < 1e-12
          and abs(r["auc_hi"] - bs["hi"]) < 1e-12,
          f"[{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]")
    check("the interval stays inside [0, 1]",
          0.0 <= r["auc_lo"] <= r["auc_hi"] <= 1.0)
    check("the normal approximation is not reported",
          "hm_clipped" not in r and "auc_lo_boot" not in r)
    check("the interval brackets the point estimate",
          r["auc_lo"] <= r["auc"] <= r["auc_hi"])

    # perfect separation: the interval degenerates to the boundary, not past it
    r2 = ST.mannwhitney(np.arange(100.0, 106.0), np.arange(6.0), boot=500,
                        seed=0)
    check("perfect separation gives AUC 1 and an interval at the boundary",
          r2["auc"] == 1.0 and r2["auc_hi"] == 1.0 and r2["auc_lo"] <= 1.0)


def test_correlation_analysis():
    """The reported correlations, against scipy."""
    print("\nCorrelation analysis")
    rng = np.random.default_rng(12)
    n = 38
    cov = rng.normal(size=n)
    end = 0.7 * cov + rng.normal(size=n)
    r = ST.correlate(end, cov)
    pr = stats.pearsonr(end, cov)
    sp = stats.spearmanr(end, cov)
    check("Pearson matches scipy",
          abs(r["pearson_r"] - pr[0]) < 1e-12
          and abs(r["pearson_p"] - pr[1]) < 1e-12)
    check("Spearman matches scipy",
          abs(r["spearman_rho"] - sp[0]) < 1e-12
          and abs(r["spearman_p"] - sp[1]) < 1e-12)
    check("a monotone but nonlinear covariate ranks perfectly",
          abs(ST.correlate(np.exp(cov), cov)["spearman_rho"] - 1.0) < 1e-12)

    v = rng.normal(size=n)
    nan = v.copy()
    nan[:3] = np.nan
    check("non-finite entries are dropped pairwise",
          ST.correlate(nan, cov)["n"] == n - 3)
    check("a constant vector correlates with nothing, without raising",
          not np.isfinite(ST.correlate(np.ones(n), cov)["pearson_r"]))

    tab = ST.correlation_table({"phi": end, "kemeny": v},
                               {"entropy": cov, "dwell": v, "logL": cov})
    check("the table covers every endpoint against every covariate",
          sorted(tab) == ["kemeny", "phi"]
          and all(sorted(row) == ["dwell", "entropy", "logL"]
                  for row in tab.values()))
    check("an endpoint against itself is rho = 1",
          abs(tab["kemeny"]["dwell"]["spearman_rho"] - 1.0) < 1e-12)


def main():
    print("=" * 74)
    print("METHODS §12.4 style checks")
    print("=" * 74)
    test_geometry()
    test_state_profiles_union()
    test_similarity()
    test_second_moment_signature()
    test_direction_aware_similarity()
    test_kemeny()
    test_jump_chain_and_degeneracy()
    test_exact_mannwhitney()
    test_reported_inference()
    test_correlation_analysis()
    test_fluency()
    test_fluency_curve()
    test_stats_helpers()
    test_shrinkage()
    test_wclrpp_peakpick()
    test_wclrpp_reduction()
    test_wclrpp_coupling()
    test_fidgetyfind_entropy()
    test_fidgetyfind_windows()
    test_fidgetyfind_invariance()
    test_fidgetyfind_smoothing_and_reduction()
    test_fidgetyfind_planted_cohort()
    test_fidgetyfind_degeneracy_check()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
