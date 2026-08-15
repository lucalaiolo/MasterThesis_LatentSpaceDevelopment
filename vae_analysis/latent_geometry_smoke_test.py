"""Checks for the latent-space-geometry diagnostics.

Each estimator is tested against a case whose answer is known in advance,
rather than only against itself:

    1. TwoNN     recovers the dimension of an exact Pareto sample, and the
                 dimension of a plane / a 3-space embedded in R^128.
    2. TwoNN     the origin-constrained least-squares fit agrees with the
                 maximum-likelihood estimate on clean data.
    3. MMD       the fast permutation statistic equals the literal block sums
                 of eq:mmd_estimator, and the observed statistic equals
                 posterior_geometry.mmd2_unbiased at the same bandwidth.
    4. MMD       does not reject prior-vs-prior; does reject a shifted sample.
    5. KL        eq:klclosed is zero exactly at mu = 0, sigma^2 = 1, and
                 matches a hand-computed value elsewhere.
    6. Bones     a rigid synthetic skeleton reports zero variation and drift.
    7c. Sweep     the pair sweep ranks by rho, keeps pairs cross-video, and
                  its "far" strategy really does pick more distant pairs.
    7. End-to-end run_latent_geometry on a real (random-weight)
       TemporalConvVAE and synthetic videos: files land, rho <= 1, the
       section's placeholders are all filled.

Run:  python -m vae_analysis.latent_geometry_smoke_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from vae_analysis.latent_geometry import (
    bone_length_report, fill_placeholders, mmd_permutation_test,
    neighbour_ratios, per_dimension_kl, placeholder_values, prior_fit,
    run_latent_geometry, select_clip_pair, twonn_least_squares,
)


# ---- 1-2. TwoNN -----------------------------------------------------------

def test_twonn_exact_pareto():
    """1. On ratios drawn from eq:pareto the slope recovers the exponent."""
    rng = np.random.default_rng(0)
    for d in (3, 8, 20):
        # F(mu) = 1 - mu^-d  =>  mu = (1 - U)^(-1/d) is an exact draw.
        mu = (1.0 - rng.random(20000)) ** (-1.0 / d)
        fit = twonn_least_squares(mu=mu, n_bootstrap=0)
        err = abs(fit["d_hat"] - d) / d
        assert err < 0.05, f"d = {d}: d_hat = {fit['d_hat']:.3f}"
        assert fit["r_squared"] > 0.99, fit["r_squared"]
    print("  [1] TwoNN on exact Pareto ratios OK")


def test_twonn_embedded_manifold():
    """1b. A d-plane embedded in R^128 reads back as d, not 128."""
    rng = np.random.default_rng(1)
    ambient = 128
    for d in (2, 3):
        latent = rng.random((3000, d))
        basis = rng.standard_normal((d, ambient))
        points = latent @ basis                      # a linear d-manifold
        fit = twonn_least_squares(points, n_bootstrap=0)
        assert abs(fit["d_hat"] - d) < 0.35, f"d = {d}: {fit['d_hat']:.3f}"
        assert fit["n_used"] == int(np.floor(3000 * 0.9))
    print("  [1b] TwoNN on an embedded manifold OK")


def test_twonn_matches_mle_and_reports_error():
    """2. Least squares through the origin ~ the full-sample MLE; errors sane.

    The MLE cross-check must be computed on the *untrimmed* ratios: the closed
    form is the MLE of the untruncated Pareto law, so running it on a sample
    whose top decile has been cut inflates it by ~34%.
    """
    rng = np.random.default_rng(2)
    mu = (1.0 - rng.random(5000)) ** (-1.0 / 6.0)
    fit = twonn_least_squares(mu=mu, n_bootstrap=64, rng=rng)
    assert abs(fit["d_hat"] - fit["d_hat_mle"]) / fit["d_hat"] < 0.05
    assert abs(fit["d_hat_mle"] - len(mu) / np.log(mu).sum()) < 1e-9
    assert 0 < fit["standard_error"] < 0.5 * fit["d_hat"]
    assert 0 < fit["bootstrap_se"] < 0.5 * fit["d_hat"]
    # The dropped tail is exactly the discard, and nothing infinite survives.
    assert fit["n_used"] == int(np.floor(5000 * 0.9))
    assert np.isfinite(fit["y"]).all() and np.isfinite(fit["log_mu"]).all()
    print("  [2] least squares vs MLE, standard errors OK")


def test_neighbour_ratios_drops_duplicates():
    """2b. Exact duplicates have an undefined ratio and are reported."""
    rng = np.random.default_rng(3)
    pts = rng.standard_normal((200, 5))
    pts = np.vstack([pts, pts[:10]])                 # 10 exact duplicates
    r = neighbour_ratios(pts)
    assert r["n_dropped_duplicates"] == 20, r["n_dropped_duplicates"]
    assert np.isfinite(r["mu"]).all() and (r["mu"] >= 1.0).all()
    print("  [2b] duplicate handling OK")


# ---- 3-4. The kernel two-sample test --------------------------------------

def _mmd2_literal(X, Y, h):
    """eq:mmd_estimator written out with explicit sums, as the reference."""
    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * h * h))
    m, n = len(X), len(Y)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1))
            - 2.0 * Kxy.sum() / (m * n))


def test_mmd_statistic_matches_reference():
    """3. The fast form equals the literal sums, and the repo's estimator."""
    from vae_analysis.posterior_geometry import mmd2_unbiased

    rng = np.random.default_rng(4)
    X = rng.standard_normal((60, 4))
    Y = rng.standard_normal((60, 4)) + 0.4
    out = mmd_permutation_test(X, Y, n_perm=8, rng=rng)
    h = out["bandwidth"]

    literal = _mmd2_literal(X, Y, h)
    assert abs(out["mmd2"] - literal) < 1e-10, (out["mmd2"], literal)
    repo = mmd2_unbiased(X, Y, h)
    assert abs(out["mmd2"] - repo) < 1e-10, (out["mmd2"], repo)

    # The permuted statistics must also be genuine MMD values: rebuild one
    # partition by hand and compare against the literal estimator.
    perm_rng = np.random.default_rng(99)
    pooled = np.vstack([X, Y])
    idx = perm_rng.permutation(len(pooled))
    a, b = pooled[idx[:60]], pooled[idx[60:]]
    direct = mmd_permutation_test(a, b, n_perm=1, bandwidth=h, rng=perm_rng)
    assert abs(direct["mmd2"] - _mmd2_literal(a, b, h)) < 1e-10
    print("  [3] MMD statistic matches the literal estimator OK")


def test_mmd_rejects_only_when_it_should():
    """4. Prior vs prior: no rejection. Shifted sample: rejection."""
    rng = np.random.default_rng(5)
    same = mmd_permutation_test(rng.standard_normal((300, 8)),
                                rng.standard_normal((300, 8)),
                                n_perm=200, rng=rng)
    assert same["p_value_add_one"] > 0.05, same["p_value_add_one"]

    shifted = mmd_permutation_test(rng.standard_normal((300, 8)) + 0.8,
                                   rng.standard_normal((300, 8)),
                                   n_perm=200, rng=rng)
    assert shifted["p_value_add_one"] < 0.01, shifted["p_value_add_one"]
    assert shifted["mmd2"] > same["mmd2"]

    # prior_fit caps the sample and says so.
    capped = prior_fit(rng.standard_normal((500, 6)), n_perm=50,
                       max_points=100, rng=rng)
    assert capped["n_used"] == 100 and capped["subsampled"]
    assert capped["m"] == capped["n"] == 100
    print("  [4] MMD null / alternative behaviour OK")


# ---- 5. The closed-form divergence ----------------------------------------

def test_per_dimension_kl():
    """5. eq:klclosed: zero at the prior, and a hand-checked value elsewhere."""
    n, d = 64, 10
    at_prior = per_dimension_kl(np.zeros((n, d)), np.zeros((n, d)))
    assert np.allclose(at_prior["kl_per_dim"], 0.0)
    assert at_prior["n_active"] == 0 and at_prior["kl_total"] == 0.0

    mu = np.full((n, d), 2.0)
    logvar = np.full((n, d), np.log(0.5))
    expected = 0.5 * (4.0 + 0.5 - np.log(0.5) - 1.0)
    kl = per_dimension_kl(mu, logvar)
    assert np.allclose(kl["kl_per_dim"], expected), kl["kl_per_dim"][:3]
    assert abs(kl["kl_total"] - d * expected) < 1e-9
    assert kl["n_active"] == d

    # The fold is the model's, so a window-major and a channel-major layout
    # must both come back with each dimension in its true window.
    d_z, n_win = 2, 3
    folded = per_dimension_kl(np.zeros((4, d_z * n_win)),
                              np.zeros((4, d_z * n_win)),
                              fold=lambda v: np.asarray(v).reshape(n_win, d_z))
    assert folded["kl_by_window"].shape == (n_win, d_z)
    assert folded["window_of_dim"].tolist() == [0, 0, 1, 1, 2, 2]

    chan_major = per_dimension_kl(
        np.zeros((4, d_z * n_win)), np.zeros((4, d_z * n_win)),
        fold=lambda v: np.asarray(v).reshape(d_z, n_win).T)
    assert chan_major["window_of_dim"].tolist() == [0, 1, 2, 0, 1, 2]
    print("  [5] closed-form per-dimension KL OK")


# ---- 6. Bone lengths ------------------------------------------------------

def test_bone_length_report():
    """6. A rigid skeleton translating in place has no variation and no drift."""
    S, T = 5, 20
    base = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    shift = np.linspace(0, 2, S * T).reshape(S, T, 1, 1)
    clips = base[None, None] + shift                 # rigid, only translating
    bones = [(0, 1), (1, 2), (2, 3)]
    rep = bone_length_report(clips, bones)
    assert rep["cv_within_max"] < 1e-9, rep["cv_within_max"]
    assert rep["drift_max"] < 1e-9, rep["drift_max"]

    stretched = clips.copy()
    stretched[S // 2, :, 1, 1] += 0.5                # tear one bone mid-morph
    rep2 = bone_length_report(stretched, bones)
    assert rep2["drift_max"] > 0.1 and rep2["drift_max_step"] == S // 2
    print("  [6] bone-length report OK")


# ---- 7. End to end --------------------------------------------------------

def _tiny_model_and_videos(seed=0):
    """A real (random-weight) TemporalConvVAE plus synthetic moving videos."""
    import torch
    from architectures.config import TrainingConfig
    from architectures.models import build_model

    torch.manual_seed(seed)
    cfg = TrainingConfig(
        architecture="temporal_conv", clip_length=16, n_joints=15, n_dims=2,
        latent_dim=4, conv_base_channels=8, conv_strides=(1, 2, 2),
        recipe=1, mask_policy="none", device="cpu", batch_size=8,
    )
    cfg.validate()
    model = build_model(cfg)

    rng = np.random.default_rng(seed)
    videos = []
    for v in range(6):                               # >= 2 so the split is video-wise
        t = np.arange(160)[:, None, None]
        phase = 0.4 * v
        base = rng.standard_normal((1, cfg.n_joints, cfg.n_dims)) * 0.2
        wave = 0.3 * np.sin(0.15 * t + phase) * np.ones((1, cfg.n_joints,
                                                         cfg.n_dims))
        videos.append((base + wave + 0.02 * rng.standard_normal(
            (160, cfg.n_joints, cfg.n_dims))).astype(np.float32))
    return model, cfg, videos


def test_end_to_end(tmp_dir: Path | None = None):
    """7. The driver runs, writes every artefact, and reports a legal rho."""
    import torch

    model, cfg, videos = _tiny_model_and_videos()
    tmp = Path(tmp_dir or tempfile.mkdtemp())
    ckpt = tmp / "best.pt"
    torch.save({"model": model.state_dict(),
                "config": cfg.__dict__, "epoch": 1}, ckpt)

    out = run_latent_geometry(
        ckpt, videos, out_dir=tmp / "geometry", split="val",
        n_perm=40, max_mmd_points=150, n_bootstrap=16,
        n_interp_steps=5, n_interp_frames=2, n_segments=6, geodesic_iters=40,
        seed=0, verbose=False,
    )

    res = out["results"]
    for name in ("fig_twonn.pdf", "fig_interp.pdf", "fig_interp_geodesic.pdf",
                 "fig_kl_per_dim.pdf", "results.json", "values.tex",
                 "summary.md"):
        assert (tmp / "geometry" / name).exists(), f"missing {name}"

    # results.json round-trips, so nothing NumPy leaked into it.
    reloaded = json.loads((tmp / "geometry" / "results.json").read_text())
    assert reloaded["meta"]["n_clips"] == res["meta"]["n_clips"]

    d_latent = res["meta"]["d_latent"]
    assert d_latent == cfg.latent_dim * (cfg.clip_length // 4)
    assert res["meta"]["n_windows"] == cfg.clip_length // 4
    assert 0 < res["twonn"]["d_hat"] <= d_latent
    assert res["twonn"]["n_used"] > 0
    assert 0.0 <= res["prior_fit"]["p_value"] <= 1.0
    assert res["kl"]["n_dims"] == d_latent
    assert len(res["kl"]["kl_per_dim"]) == d_latent

    ip = res["interpolation"]
    assert ip["curvature_ratio"] <= 1.0 + 1e-4, ip["curvature_ratio"]
    assert ip["arclength_geodesic"] <= ip["arclength_straight"] * (1 + 1e-4)
    assert ip["energy_geodesic"] <= ip["energy_straight"] * (1 + 1e-4)
    assert ip["index_a"] != ip["index_b"]

    # Every slot the section leaves open comes back filled.
    slots = placeholder_values(res)
    assert set(slots) == {"[d\\_hat]", "[se]", "[n\\_used]", "[mmd2]", "[p]",
                          "[rho]"}
    assert all(v and "[" not in v for v in slots.values()), slots
    source = (r"$\hat d = \texttt{[d\_hat]} \pm \texttt{[se]}$ over "
              r"$\texttt{[n\_used]}$ points, $\texttt{[mmd2]}$, "
              r"$\texttt{[p]}$, $\rho = \texttt{[rho]}$.")
    filled = fill_placeholders(source, res)
    assert "[d\\_hat]" not in filled and r"\texttt" not in filled, filled

    # A small MMD^2 formats as math; it must not land inside \texttt, where
    # \times and ^ are LaTeX errors.
    tiny = json.loads(json.dumps(res))
    tiny["prior_fit"]["mmd2"] = 1.234e-3
    assert r"\times 10^{-3}" in placeholder_values(tiny)["[mmd2]"]
    assert r"\texttt" not in fill_placeholders(source, tiny)

    # A p-value no permutation reached is a bound: "= < 0.002" would be
    # nonsense, so the '=' is swallowed and it reads "p < 0.002".
    bound = json.loads(json.dumps(res))
    bound["prior_fit"].update({"n_exceed": 0, "n_perm": 500})
    filled_bound = fill_placeholders(r"$\hat p = \texttt{[p]}$", bound)
    assert filled_bound == r"$\hat p < 0.002$", filled_bound
    print("  [7] end-to-end driver OK "
          f"(d_hat = {res['twonn']['d_hat']:.2f}, "
          f"rho = {ip['curvature_ratio']:.4f})")


def test_sweep_curvature_ratio():
    """7c. The pair sweep ranks by rho, respects the strategies, stays <= 1."""
    from vae_analysis.geodesic_interpolation import make_interpolator
    from vae_analysis.latent_geometry import sweep_curvature_ratio

    model, cfg, videos = _tiny_model_and_videos(seed=1)
    interp = make_interpolator(model, config=cfg, device="cpu")

    rng = np.random.default_rng(0)
    X = np.stack([v[s:s + cfg.clip_length]
                  for v in videos for s in (0, 40, 80)]).astype(np.float32)
    vid = np.repeat(np.arange(len(videos)), 3)
    mu = np.stack([interp.encode(x).numpy() for x in X])

    sweep = sweep_curvature_ratio(interp, X, vid, mu=mu, n_pairs=6,
                                  strategy="spread", n_candidates=120,
                                  n_segments=4, iters=20, rng=rng,
                                  verbose=False)
    rho = sweep["rho"]
    assert sweep["n_pairs"] == len(sweep["records"]) <= 6
    assert (np.diff(rho) >= -1e-12).all(), "records are not sorted by rho"
    assert (rho <= 1.0 + 1e-4).all(), rho.max()
    for r in sweep["records"]:
        assert r["index_a"] != r["index_b"]
        assert r["video_a"] != r["video_b"]        # pairs are cross-video
        assert r["arclength_geodesic"] <= r["arclength_straight"] * (1 + 1e-4)

    # "far" must take pairs at least as distant as the spread selection.
    far = sweep_curvature_ratio(interp, X, vid, mu=mu, n_pairs=4,
                                strategy="far", n_candidates=120,
                                n_segments=4, iters=20, rng=rng, verbose=False)
    far_d = np.mean([r["latent_distance"] for r in far["records"]])
    spread_d = np.mean([r["latent_distance"] for r in sweep["records"]])
    assert far_d >= spread_d, (far_d, spread_d)

    # Explicit pairs bypass the sampling entirely.
    given = sweep_curvature_ratio(interp, X, vid, pairs=[(0, 5), (1, 9)],
                                  n_segments=4, iters=20, verbose=False)
    assert {(r["index_a"], r["index_b"]) for r in given["records"]} == {(0, 5), (1, 9)}
    print("  [7c] curvature-ratio sweep OK "
          f"(rho {rho.min():.4f}-{rho.max():.4f})")


def test_pair_selection():
    """7b. Pair choice avoids same-video pairs and honours the mode."""
    rng = np.random.default_rng(7)
    mu = rng.standard_normal((200, 5))
    vid = np.repeat([0, 1, 2, 3], 50)
    for mode in ("random", "median", "farthest"):
        i, j = select_clip_pair(mu, vid, mode=mode, n_candidates=200, rng=rng)
        assert i != j and vid[i] != vid[j], (mode, i, j)

    # With one video only, it must still return a usable pair.
    i, j = select_clip_pair(mu, np.zeros(200, dtype=int), mode="random", rng=rng)
    assert i != j
    print("  [7b] endpoint-pair selection OK")


def main():
    print("Latent-space geometry — estimator and driver checks")
    test_twonn_exact_pareto()
    test_twonn_embedded_manifold()
    test_twonn_matches_mle_and_reports_error()
    test_neighbour_ratios_drops_duplicates()
    test_mmd_statistic_matches_reference()
    test_mmd_rejects_only_when_it_should()
    test_per_dimension_kl()
    test_bone_length_report()
    test_pair_selection()
    test_sweep_curvature_ratio()
    test_end_to_end()
    print("All latent-space-geometry checks passed.")


if __name__ == "__main__":
    main()
