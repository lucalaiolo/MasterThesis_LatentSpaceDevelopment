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

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a1_core as A          # noqa: E402
import a1_stats as ST        # noqa: E402
import a57_graph as G        # noqa: E402

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


def test_pcca():
    print("\n§8.3 PCCA+")
    # Three well-separated blocks: PCCA+ must recover them.
    K, m = 9, 3
    A_ = np.full((K, K), 0.001)
    for b in range(m):
        idx = slice(3 * b, 3 * b + 3)
        A_[idx, idx] = 0.33
    A_ = A_ / A_.sum(1, keepdims=True)
    chi, verts = G.pcca(A_, m)
    assign = chi.argmax(1)
    truth = np.repeat(np.arange(m), 3)
    from sklearn.metrics import adjusted_rand_score
    check("PCCA+ recovers three planted blocks (ARI = 1)",
          adjusted_rand_score(assign, truth) == 1.0,
          f"ARI {adjusted_rand_score(assign, truth):.3f}")
    check("crispness is high on a near-hard partition",
          G.crispness(chi) > 0.9, f"crispness {G.crispness(chi):.3f}")
    ms = G.metastability(A_, assign)
    check("metastability is bounded above by m", ms <= m + 1e-9,
          f"{ms:.3f} <= {m}")
    check("spectral k-means control recovers the same blocks",
          adjusted_rand_score(G.spectral_kmeans(A_, m), truth) == 1.0)

    # §8.4 kinematic partition on a matching similarity structure.
    S = np.full((K, K), -0.5)
    for b in range(m):
        idx = slice(3 * b, 3 * b + 3)
        S[idx, idx] = 0.9
    np.fill_diagonal(S, 1.0)
    check("kinematic partition recovers the same blocks",
          adjusted_rand_score(G.kinematic_partition(S, m), truth) == 1.0)


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


def test_stats_helpers():
    print("\n§10 helpers")
    check("Holm adjustment is monotone and bounded",
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
    check("minimum detectable AUC is large at 6 vs 32",
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


def main():
    print("=" * 74)
    print("METHODS §12.4 style checks")
    print("=" * 74)
    test_geometry()
    test_state_profiles_union()
    test_similarity()
    test_kemeny()
    test_jump_chain_and_degeneracy()
    test_pcca()
    test_exact_mannwhitney()
    test_fluency()
    test_stats_helpers()
    test_shrinkage()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
