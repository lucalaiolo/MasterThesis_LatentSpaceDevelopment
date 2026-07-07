"""Driver: run a curated set of latent-space analyses on one checkpoint.

`run_all_analyses(checkpoint_path, videos, skeleton, out_dir)` loads a
trained model with `architectures.analyze.load_checkpoint`, encodes the
validation split with `encode_dataset`, runs the core analyses (the
ones that need only NumPy, SciPy, and scikit-learn), writes a
`results.json` of scalar summaries, and renders a small set of
diagnostic plots to PNG.

The Jacobian tools (encoder/decoder geometry) and the optional-package
paths (persistent homology, hidden Markov model, PELT) are not touched
here — call them yourself from a notebook once the core report tells
you the model is worth the deeper look.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .interfaces import Skeleton, encode_dataset


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def _to_python(obj):
    """Recursively convert numpy scalars/arrays to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_all_analyses(checkpoint_path: str | Path,
                     videos: list[np.ndarray],
                     skeleton: Skeleton | None = None,
                     out_dir: str | Path | None = None,
                     device: str = "cpu",
                     stride: int | None = None,
                     mask_policy_override: str | None = None,
                     limbs: dict[str, list[int]] | None = None,
                     n_perm: int = 200,
                     rng_seed: int = 0,
                     include_decoder_geometry: bool = True,
                     n_anchors: int = 16,
                     traversal_steps: tuple[float, ...] = (-3, -2, -1, 0, 1, 2, 3),
                     ) -> dict:
    """Run the core latent-space analyses on one checkpoint.

    Args:
        checkpoint_path: `.pt` file written by `architectures.train`.
        videos: same list of videos used at training.
        skeleton: a `Skeleton`. If None, a bare-bones one with
            `n_joints` inferred from the checkpoint is used — the
            skeleton-dependent analyses (features, symmetry, masking
            recovery) are skipped.
        out_dir: where PNGs and `results.json` are written. Defaults to
            `<checkpoint dir>/vae_analysis/`.
        device: `"cpu"` or `"cuda"`.
        stride: clip stride for the val loader; defaults to
            `clip_length // 2` (same as training).
        mask_policy_override: force a specific mask policy for encoding
            (e.g. "none" for a pure-clean-pose latent). Defaults to the
            policy the checkpoint was trained with.
        limbs: joint-index lists per limb name, needed only when the
            model was trained with `mask_policy="limb"`.
        n_perm: permutation count for `mmd_prior_test`.
        rng_seed: seed for stochastic analyses.

    Returns:
        Dict with `results` (all scalars), `written` (list of PNG
        paths), and `latent` (the LatentSet).
    """
    # Imports are local so this module loads without heavy deps.
    import dataclasses
    from architectures.analyze import load_checkpoint
    from architectures.data import build_clips, train_val_split
    from architectures.mask_policies import build_policy
    from .architectures_adapter import ArchitecturesAdapter
    from . import (posterior_geometry as pg, features as ft, masking as mk,
                   information as inf, symmetry as sym, disentanglement as dis,
                   two_sample as ts, screening as scr, honesty as hon,
                   generation as gen, decoder_geometry as dg)

    plt = _import_matplotlib()
    rng = np.random.default_rng(rng_seed)

    model, config = load_checkpoint(checkpoint_path, device=device)
    if mask_policy_override is not None:
        config = dataclasses.replace(config, mask_policy=mask_policy_override)
    if stride is None:
        stride = config.clip_length // 2

    if skeleton is None:
        skeleton = Skeleton(n_joints=config.n_joints)

    clips, video_id, time_index = build_clips(
        videos, config.clip_length, stride,
    )
    _, val_mask = train_val_split(clips, video_id)
    X_val = clips[val_mask].astype(np.float32)
    vid_val = video_id[val_mask]
    t_val = time_index[val_mask]

    policy = build_policy(config, limbs=limbs)
    # Draw one mask per clip. Speed-based policies need the clip.
    M_val = np.stack([policy.sample(config.clip_length, config.n_joints,
                                    rng, X=X_val[b])
                      for b in range(len(X_val))]).astype(np.float32)

    adapter = ArchitecturesAdapter(model, device=device)
    latent = encode_dataset(adapter, X_val, M_val,
                            video_id=vid_val, time_index=t_val)
    latent.sample(rng)
    print(f"[analysis] encoded {latent.n} val clips, d_z = {latent.d_z}")

    if out_dir is None:
        out_dir = Path(checkpoint_path).parent / "vae_analysis"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(fig, name: str):
        p = out_dir / name
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    results: dict = {}

    # ---- 1. Posterior geometry (§3) ----
    print("[analysis] posterior_geometry ...")
    mmd = pg.mmd_prior_test(latent, n_perm=n_perm, rng=rng)
    intr = pg.intrinsic_dimension_twonn(latent.mu)
    clust = pg.cluster_structure(latent, k_range=range(2, 13))
    results["posterior_geometry"] = {
        "mmd2": mmd["mmd2"], "mmd_p_value": mmd["p_value"],
        "mmd_bandwidth": mmd["bandwidth"],
        "intrinsic_dim": intr["d_hat"],
        "intrinsic_dim_se": intr["standard_error"],
        "cluster_k": clust["k"], "bic_curve": clust["bic"],
    }
    _save(_plot_mmd_summary(mmd, plt), "mmd_prior.png")
    _save(_plot_bic_curve(clust, plt), "cluster_bic.png")
    if "composition" in clust:
        results["posterior_geometry"]["composition"] = clust["composition"]
        _save(_plot_cluster_composition(clust, plt),
              "cluster_composition.png")

    # ---- 2. Information (§9 / II §19) ----
    print("[analysis] information ...")
    au = inf.active_units(latent)
    tc = inf.tc_decomposition(latent, batch=256, rng=rng)
    results["information"] = {
        "n_active": au["n_active"], "d_z": au["d_z"],
        "variance_per_dim": au["variance"],
        "total_correlation": tc,
    }
    _save(_plot_active_units(au, plt), "active_units.png")

    # ---- 3. Skeleton-dependent: features + regression (§5) ----
    skipped: list[str] = []
    if skeleton.limbs:
        print("[analysis] features + regression + CCA ...")
        feats, fnames = ft.kinematic_features(X_val, skeleton)
        r2 = ft.feature_regression(latent, feats, fnames)
        cca = ft.canonical_correlation(latent, feats, n_components=5)
        results["features"] = {
            "r2_per_feature": {k: v for k, v in r2.items() if not k.startswith("_")},
            "r2_mean": r2["_mean"],
            "cca_correlations": cca["correlations"],
        }
        _save(_plot_feature_r2(r2, plt), "feature_r2.png")
        _save(_plot_cca(cca["correlations"], plt), "cca_correlations.png")

        # ---- 4. Disentanglement (II §16) — needs the features. ----
        print("[analysis] disentanglement ...")
        mig = dis.mig(latent, feats)
        sap = dis.sap(latent, feats)
        try:
            dci = dis.dci(latent, feats)
            results["disentanglement"] = {
                "mig": mig["mig"], "sap": sap["sap"],
                "disentanglement": dci["disentanglement"],
                "completeness": dci["completeness"],
                "informativeness_rmse": dci["informativeness_rmse"],
            }
            _save(_plot_dci_importance(dci["importance"], fnames, plt),
                  "dci_importance.png")
        except Exception as e:  # gradient-boosting is heavy; fail-soft.
            print(f"[analysis]   DCI skipped: {e}")
            results["disentanglement"] = {"mig": mig["mig"], "sap": sap["sap"]}
        _save(_plot_disentanglement_scores(results["disentanglement"], plt),
              "disentanglement_scores.png")
    else:
        skipped.append("features / disentanglement (Skeleton has no limbs)")

    # ---- 5. Masking robustness (§6) ----
    print("[analysis] masking robustness ...")
    if skeleton.limbs:
        recovery = mk.latent_recovery(adapter, X_val[:min(200, len(X_val))],
                                      skeleton, rng=rng)
        results["masking"] = {"latent_recovery": recovery}
        _save(_plot_latent_recovery(recovery, plt), "latent_recovery.png")
    else:
        skipped.append("latent_recovery limb legs (Skeleton has no limbs)")

    X_hat = adapter.decode(latent.z)
    split = mk.split_mpjpe(X_val, X_hat, M_val)
    results.setdefault("masking", {})["split_mpjpe"] = split

    # ---- 6. Symmetry (II §15) — needs left_right pairs. ----
    if skeleton.left_right:
        print("[analysis] symmetry ...")
        try:
            eq = sym.fit_equivariance(adapter, X_val, M_val, skeleton)
            sub = sym.laterality_subspace(eq["A"])
            asym = sym.asymmetry_score(latent, sub["projector"])
            results["symmetry"] = {
                "variance_explained": eq["variance_explained"],
                "antisymmetric_dim": sub["antisymmetric_dim"],
                "asymmetry_score_mean": float(asym.mean()),
            }
            _save(_plot_asymmetry_hist(asym, latent.video_id, plt),
                  "asymmetry_scores.png")
        except Exception as e:
            print(f"[analysis]   symmetry skipped: {e}")
    else:
        skipped.append("symmetry (Skeleton has no left_right pairs)")

    # ---- 7. Generation (§8) ----
    print("[analysis] generation ...")
    prior_z = latent.prior_like(min(200, latent.n), rng)
    gen_clips = adapter.decode(prior_z)
    if skeleton.bones:
        bone = gen.bone_plausibility(gen_clips, X_val, skeleton)
        results["generation"] = {
            "bone_plausibility_ratio_mean": float(bone["ratio"].mean()),
        }
    else:
        skipped.append("bone_plausibility (Skeleton has no bones)")
    if skeleton.limbs:
        gf, _ = ft.kinematic_features(gen_clips, skeleton)
        rf, _ = ft.kinematic_features(X_val, skeleton)
        results.setdefault("generation", {})["frechet_distance"] = \
            gen.frechet_distance(rf, gf)
    interp = gen.interpolation_curvature(adapter, latent, n_pairs=30, rng=rng)
    results.setdefault("generation", {})["interpolation_curvature"] = interp

    # ---- 8. Two-sample q(z) vs p(z) (II §20) ----
    print("[analysis] classifier two-sample ...")
    c2st = ts.classifier_two_sample(latent, rng=rng)
    results["two_sample"] = {"c2st_accuracy": c2st["accuracy"]}

    # ---- 9. Screening / typicality (II §21) ----
    print("[analysis] screening ...")
    try:
        dens = scr.fit_density(latent, method="gmm", n_components=4)
        scores = scr.typicality_score(dens, latent)
        results["screening"] = {
            "typicality_mean": float(scores.mean()),
            "typicality_std":  float(scores.std()),
        }
        if latent.video_id is not None:
            _save(_plot_typicality(scores, latent.video_id, plt),
                  "typicality_scores.png")
    except Exception as e:
        print(f"[analysis]   screening skipped: {e}")

    # ---- 10. Honesty (§12) ----
    print("[analysis] honesty (block bootstrap) ...")
    try:
        blocks = hon.time_blocks(latent, block_seconds=5.0, fps=float(config.fps))
        if skeleton.left_right:
            eq = sym.fit_equivariance(adapter, X_val, M_val, skeleton)
            sub = sym.laterality_subspace(eq["A"])
            stat = sym.asymmetry_score(latent, sub["projector"])
            boot = hon.block_bootstrap(stat, blocks, n_boot=200, rng=rng)
            results["honesty"] = {"asymmetry_ci": boot}
        else:
            # No left-right pairs — bootstrap the per-clip L2 norm of mu,
            # a rough proxy for "activity of the posterior".
            per_clip = np.linalg.norm(latent.mu, axis=1)
            boot = hon.block_bootstrap(per_clip, blocks, n_boot=200, rng=rng)
            results["honesty"] = {"mu_norm_ci": boot}
    except Exception as e:
        print(f"[analysis]   honesty skipped: {e}")

    # ---- 11. Decoder geometry (§4) ----
    if include_decoder_geometry:
        try:
            print(f"[analysis] decoder_geometry (Jacobian on {n_anchors} anchors) ...")
            # Anchor latents: pick a subset of posterior means. Sampling
            # a subset instead of the full set keeps the Jacobian cost
            # bounded — one jacrev call per anchor.
            rng_anchor = np.random.default_rng(rng_seed + 7)
            idx = rng_anchor.choice(latent.n,
                                    size=min(n_anchors, latent.n),
                                    replace=False)
            anchors = latent.mu[idx]

            # §4.1 Sensitivity maps.
            sens = dg.sensitivity_maps(adapter, anchors)
            results["decoder_geometry"] = {
                "sensitivity_joint_latent_mean":
                    float(sens["joint_latent"].mean()),
                "sensitivity_time_latent_mean":
                    float(sens["time_latent"].mean()),
            }
            _save(_plot_sensitivity_joint_latent(sens["joint_latent"], plt),
                  "sensitivity_joint_latent.png")
            _save(_plot_sensitivity_time_latent(sens["time_latent"], plt),
                  "sensitivity_time_latent.png")

            # §4.2 Measured traversal on the median anchor.
            if skeleton.bones and skeleton.left_right:
                z_star = np.median(anchors, axis=0)
                trav = dg.measured_traversal(adapter, z_star, skeleton,
                                             steps=traversal_steps)
                _save(_plot_traversal_displacement(trav, plt),
                      "traversal_displacement.png")
                _save(_plot_traversal_laterality(trav, plt),
                      "traversal_laterality.png")
                _save(_plot_traversal_bone_stretch(trav, plt),
                      "traversal_bone_stretch.png")
                results["decoder_geometry"]["traversal_max_abs_stretch"] = \
                    float(np.abs(trav["bone_stretch"]).max())
            else:
                skipped.append("measured_traversal (skeleton needs bones + left_right)")

            # §4.3 Pullback metric.
            spec = dg.metric_spectrum(adapter, anchors)
            results["decoder_geometry"]["mean_condition"] = spec["mean_condition"]
            _save(_plot_metric_spectrum(spec, plt), "metric_spectrum.png")
            _save(_plot_condition_hist(spec["condition"], plt),
                  "metric_condition.png")
        except Exception as e:
            print(f"[analysis]   decoder_geometry skipped: {e}")

    # ---- Write scalar summary ----
    if skipped:
        results["_skipped"] = skipped
    with open(out_dir / "results.json", "w") as f:
        json.dump(_to_python(results), f, indent=2)
    written.append(out_dir / "results.json")

    print(f"\n[analysis] wrote {len(written)} file(s) to {out_dir}")
    return {"results": results, "written": written, "latent": latent}


# ============================================================================
# Plots. Each takes the raw analysis output and returns a matplotlib figure.
# ============================================================================

def _plot_active_units(au, plt):
    v = au["variance"]
    order = np.argsort(v)[::-1]
    fig, ax = plt.subplots(figsize=(max(6, 0.28 * len(v)), 3.6))
    colors = ["seagreen" if au["active"][k] else "lightgray" for k in order]
    ax.bar(np.arange(len(v)), v[order], color=colors)
    ax.axhline(0.01, ls=":", color="firebrick", label="threshold")
    ax.set_yscale("log")
    ax.set_xlabel("latent dimension (sorted)")
    ax.set_ylabel(r"$\mathrm{Var}(\mu_d)$")
    ax.set_title(f"Active units — {au['n_active']} / {au['d_z']}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    return fig


def _plot_feature_r2(r2, plt):
    items = [(k, v) for k, v in r2.items() if not k.startswith("_")]
    items.sort(key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(labels)), 3.6))
    ax.bar(np.arange(len(labels)), vals, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(r"held-out $R^2$")
    ax.set_title(f"Feature regression — mean $R^2$ = {r2['_mean']:.3f}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def _plot_cca(corrs, plt):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(np.arange(len(corrs)), corrs, color="steelblue")
    ax.set_xlabel("canonical component")
    ax.set_ylabel("correlation")
    ax.set_ylim(0, 1)
    ax.set_title("Canonical correlations")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def _plot_dci_importance(importance, fnames, plt):
    fig, ax = plt.subplots(figsize=(max(5, 0.4 * len(fnames)),
                                    max(4, 0.2 * importance.shape[0])))
    im = ax.imshow(importance, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(fnames)))
    ax.set_xticklabels(fnames, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("latent dim")
    ax.set_title("DCI importance — latent dim × factor")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _plot_disentanglement_scores(scores, plt):
    keys = [k for k in ("mig", "sap", "disentanglement", "completeness")
            if k in scores]
    vals = [scores[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(keys, vals, color="steelblue")
    ax.set_ylim(0, max(1.0, max(vals) * 1.05))
    ax.set_title("Disentanglement scores")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def _plot_latent_recovery(recovery, plt):
    unif = [(float(k.split("_")[1]), v) for k, v in recovery.items()
            if k.startswith("uniform_")]
    limb = [(k[5:], v) for k, v in recovery.items() if k.startswith("limb_")]
    unif.sort()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if unif:
        xs, ys = zip(*unif)
        ax.plot(xs, ys, "o-", color="steelblue", label="uniform ρ")
    for i, (name, v) in enumerate(limb):
        ax.axhline(v, ls="--", color=f"C{i+1}", label=f"limb {name}")
    ax.set_xlabel("hidden fraction ρ")
    ax.set_ylabel("mean $\\|\\mu - \\mu_{full}\\|$")
    ax.set_title("Latent drift under masking")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_cluster_composition(clust, plt):
    Pi = clust["composition"]
    fig, ax = plt.subplots(figsize=(max(5, 0.4 * Pi.shape[1]),
                                    max(3, 0.35 * Pi.shape[0])))
    im = ax.imshow(Pi, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("cluster index")
    ax.set_ylabel("video id")
    ax.set_title(f"Cluster composition — K = {clust['k']}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _plot_asymmetry_hist(asym, video_id, plt):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if video_id is not None:
        for v in np.unique(video_id):
            ax.hist(asym[video_id == v], bins=30, alpha=0.55,
                    label=f"video {v}")
        ax.legend(fontsize=8)
    else:
        ax.hist(asym, bins=30, color="steelblue")
    ax.set_xlabel("asymmetry score")
    ax.set_ylabel("count")
    ax.set_title("Laterality asymmetry")
    fig.tight_layout()
    return fig


def _plot_typicality(scores, video_id, plt):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for v in np.unique(video_id):
        ax.hist(scores[video_id == v], bins=30, alpha=0.55,
                label=f"video {v}")
    ax.set_xlabel("typicality score (log-density)")
    ax.set_ylabel("count")
    ax.set_title("Typicality by video")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---- §3.1 / §3.3 additions -------------------------------------------------

def _plot_mmd_summary(mmd, plt):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(["MMD²(q, p)"], [mmd["mmd2"]], color="steelblue")
    ax.set_title(f"MMD² = {mmd['mmd2']:.4g}   p = {mmd['p_value']:.3g}\n"
                 f"bandwidth h = {mmd['bandwidth']:.3g}")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def _plot_bic_curve(clust, plt):
    ks = sorted(clust["bic"])
    bics = [clust["bic"][k] for k in ks]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ks, bics, "o-", color="steelblue")
    ax.axvline(clust["k"], ls=":", color="firebrick",
               label=f"chosen K = {clust['k']}")
    ax.set_xlabel("K")
    ax.set_ylabel("BIC")
    ax.set_title("GMM model-selection curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---- §4.1 decoder Jacobian sensitivity maps --------------------------------

def _plot_sensitivity_joint_latent(M, plt):
    """Heatmap of averaged joint × latent sensitivity."""
    fig, ax = plt.subplots(figsize=(max(5, 0.28 * M.shape[1]),
                                    max(4, 0.14 * M.shape[0])))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("joint index")
    ax.set_title(r"Sensitivity $S_{j,i}$ — which latents move which joints")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _plot_sensitivity_time_latent(M, plt):
    """Heatmap of averaged time × latent sensitivity."""
    fig, ax = plt.subplots(figsize=(max(5, 0.28 * M.shape[1]), 3.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("frame")
    ax.set_title(r"Sensitivity $R_{t,i}$ — when in the clip each latent acts")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# ---- §4.2 measured traversal ----------------------------------------------

def _plot_traversal_displacement(trav, plt):
    """Heatmap of dim × step displacement, aggregated over joints."""
    # displacement (d_z, n_steps, J) -> take the L2 norm across joints
    # so each cell is the total per-joint drift under that traversal.
    disp = np.linalg.norm(trav["displacement"], axis=2)  # (d_z, n_steps)
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * disp.shape[1]),
                                    max(4, 0.14 * disp.shape[0])))
    im = ax.imshow(disp, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(trav["steps"])))
    ax.set_xticklabels([f"{s:+g}" for s in trav["steps"]])
    ax.set_xlabel(r"traversal offset $\alpha$")
    ax.set_ylabel("latent dimension")
    ax.set_title(r"Traversal displacement $\|\Delta p_{j,i}(\alpha)\|_2$")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _plot_traversal_laterality(trav, plt):
    """Heatmap of dim × step laterality, aggregated over left-right pairs."""
    # laterality (d_z, n_steps, n_pairs) -> mean over pairs.
    lat = trav["laterality"].mean(axis=2)             # (d_z, n_steps)
    m = float(np.abs(lat).max()) or 1.0
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * lat.shape[1]),
                                    max(4, 0.14 * lat.shape[0])))
    im = ax.imshow(lat, aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m)
    ax.set_xticks(np.arange(len(trav["steps"])))
    ax.set_xticklabels([f"{s:+g}" for s in trav["steps"]])
    ax.set_xlabel(r"traversal offset $\alpha$")
    ax.set_ylabel("latent dimension")
    ax.set_title(r"Traversal laterality $\ell_i(\alpha)$ (left − right)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _plot_traversal_bone_stretch(trav, plt):
    """Heatmap of dim × step bone stretch — max over bones."""
    # bone_stretch (d_z, n_steps, n_bones) -> max abs over bones.
    stretch = np.abs(trav["bone_stretch"]).max(axis=2)  # (d_z, n_steps)
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * stretch.shape[1]),
                                    max(4, 0.14 * stretch.shape[0])))
    im = ax.imshow(stretch, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(trav["steps"])))
    ax.set_xticklabels([f"{s:+g}" for s in trav["steps"]])
    ax.set_xlabel(r"traversal offset $\alpha$")
    ax.set_ylabel("latent dimension")
    ax.set_title(r"Max bone stretch $|\Delta b_{jk,i}(\alpha)|$ — anatomy check")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# ---- §4.3 pullback metric --------------------------------------------------

def _plot_metric_spectrum(spec, plt):
    """Sorted eigenvalue spectrum of G(z), averaged over anchors."""
    eig = spec["eigenvalues"]                            # (A, d_z)
    order = np.argsort(-eig.mean(axis=0))
    sorted_mean = np.sort(eig, axis=1)[:, ::-1]
    mean_curve = sorted_mean.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(np.arange(1, len(mean_curve) + 1), mean_curve,
            "o-", color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel(r"$\lambda(G(z))$")
    ax.set_title(r"Pullback-metric spectrum "
                 f"(mean cond = {spec['mean_condition']:.2g})")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def _plot_condition_hist(conds, plt):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(conds[np.isfinite(conds)], bins=15, color="steelblue")
    ax.axvline(10, ls=":", color="firebrick",
               label="threshold = 10")
    ax.set_xlabel(r"condition number of $G(z)$")
    ax.set_ylabel("anchor count")
    ax.set_title("Local anisotropy across anchors")
    ax.legend()
    fig.tight_layout()
    return fig
