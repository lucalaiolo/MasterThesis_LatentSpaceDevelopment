"""Latent-space analysis for the YouTube 2D-keypoint VAEs.

The whole analysis battery already exists in ``vae_analysis`` — posterior
geometry, decoder / encoder **Jacobians**, dynamics (**HMM**, change points,
Ornstein-Uhlenbeck), symmetry, disentanglement, information, and so on — and
is generic in the joint count, clip length, and coordinate dimension. This
module wires those analyses onto a trained ``youtube_motion`` checkpoint:

  * builds a COCO-18 :class:`~vae_analysis.interfaces.Skeleton` (bones, limbs,
    left-right pairs) so the skeleton-dependent analyses run,
  * calls :func:`vae_analysis.driver.run_all_analyses` on the checkpoint and
    the bundle's videos, and
  * adds a **UMAP** embedding plot of the posterior means (the one view the
    core driver leaves out — it defaults to PCA).

CLI:

    # analyse one checkpoint
    python -m youtube_motion.analysis --csv keypoints.csv \
        --checkpoint checkpoints/youtube_motion/conv/recipe1_uniform/best.pt

    # or analyse the best model from a finished sweep
    python -m youtube_motion.analysis --csv keypoints.csv \
        --sweep-out checkpoints/youtube_motion --metric mpjpe_all

Import:

    from youtube_motion.data import load_youtube_csv
    from youtube_motion.analysis import analyze_checkpoint, analyze_best

    bundle = load_youtube_csv("keypoints.csv")
    analyze_checkpoint("…/best.pt", bundle, out_dir="…/analysis", device="cuda")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import YoutubeMotionBundle
from .skeleton import COCO18_BONES, COCO18_LEFT_RIGHT, N_JOINTS, coco18_limbs


def coco18_skeleton():
    """A :class:`vae_analysis.interfaces.Skeleton` for the COCO-18 layout.

    Carries the bones, the four limbs + head, and the eight left-right pairs,
    so the symmetry / laterality / kinematic-feature analyses all have what
    they need. ``lateral_axis=0`` (x) is the coordinate a left-right mirror
    negates in the image plane.
    """
    from vae_analysis.interfaces import Skeleton
    return Skeleton(
        n_joints=N_JOINTS,
        bones=list(COCO18_BONES),
        left_right=list(COCO18_LEFT_RIGHT),
        lateral_axis=0,
        limbs=coco18_limbs(),
    )


def _embed_2d(mu: np.ndarray, seed: int = 0):
    """2D embedding of the latents — UMAP if available, else linear PCA.

    Returns ``(coords (N, 2), name)``. UMAP separates modes more clearly when
    they exist; PCA is the honest linear fallback when umap-learn is absent.
    """
    mu = np.asarray(mu, dtype=np.float64)
    try:
        import umap  # type: ignore
        if len(mu) >= 5:
            reducer = umap.UMAP(n_components=2,
                                n_neighbors=min(30, len(mu) - 1),
                                random_state=seed)
            return reducer.fit_transform(mu), "UMAP"
    except ImportError:
        pass
    X = mu - mu.mean(axis=0, keepdims=True)
    Vt = np.linalg.svd(X, full_matrices=False)[2]
    return X @ Vt[:2].T, "PCA (umap-learn not installed)"


def _plot_umap(latent, seed, plt):
    """Two-panel embedding scatter: coloured by video, and by clip time."""
    mu = latent.mu
    coords, name = _embed_2d(mu, seed)
    vid = (latent.video_id if latent.video_id is not None
           else np.zeros(len(mu), dtype=int))
    t = (latent.time_index if latent.time_index is not None
         else np.arange(len(mu)))
    uniq = np.unique(vid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Panel 1 — by video id (categorical).
    for k, v in enumerate(uniq):
        m = vid == v
        axes[0].scatter(coords[m, 0], coords[m, 1], s=9, alpha=0.6,
                        color=plt.cm.tab20(k % 20),
                        label=(f"v{v}" if len(uniq) <= 20 else None))
    axes[0].set_title(f"{name} of posterior means — by video "
                      f"({len(uniq)} videos)")
    if len(uniq) <= 20:
        axes[0].legend(fontsize=7, markerscale=2, ncol=2, loc="best")
    # Panel 2 — by clip start frame (continuous), a proxy for time-in-video.
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], s=9, alpha=0.75,
                         c=t, cmap="viridis")
    fig.colorbar(sc, ax=axes[1], label="clip start frame")
    axes[1].set_title(f"{name} — by clip time")
    for ax in axes:
        ax.set_xlabel(f"{name.split()[0]}-1")
        ax.set_ylabel(f"{name.split()[0]}-2")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def analyze_checkpoint(checkpoint_path: str,
                       bundle: YoutubeMotionBundle,
                       out_dir: str | None = None,
                       device: str = "cpu",
                       add_umap: bool = True,
                       umap_seed: int = 0,
                       **kwargs) -> dict:
    """Run the full latent-space battery on one 2D checkpoint.

    Args:
        checkpoint_path: a ``best.pt`` written by ``architectures.train`` (via
            the ``youtube_motion`` sweep). Its config carries ``n_dims=2``.
        bundle: the **same** dataset the model trained on — the analysis
            rebuilds the identical time-based validation split to encode.
        out_dir: where PNGs and ``results.json`` are written. Defaults to
            ``<checkpoint dir>/vae_analysis``.
        device: ``"cpu"`` or ``"cuda"``.
        add_umap: also write ``umap_embeddings.png`` (UMAP of posterior
            means, coloured by video and by time). Falls back to PCA if
            umap-learn is not installed.
        umap_seed: seed for the UMAP layout.
        **kwargs: forwarded to :func:`vae_analysis.driver.run_all_analyses`
            (e.g. ``include_dynamics``, ``n_anchors``, ``n_jacobian_clips``,
            ``include_persistent_homology``, ``mask_policy_override``).
    Returns:
        The ``run_all_analyses`` result dict (``results`` / ``written`` /
        ``latent``), with the UMAP figure appended to ``written``.
    """
    from vae_analysis.driver import run_all_analyses, _import_matplotlib

    if out_dir is None:
        out_dir = str(Path(checkpoint_path).parent / "vae_analysis")
    out = Path(out_dir)

    skel = coco18_skeleton()
    print(f"[analysis] checkpoint: {checkpoint_path}")
    print(f"[analysis] COCO-18 skeleton: {skel.n_joints} joints, "
          f"{len(skel.bones)} bones, {len(skel.left_right)} L/R pairs, "
          f"{len(skel.limbs)} limbs")

    result = run_all_analyses(
        checkpoint_path, bundle.videos, skeleton=skel,
        limbs=bundle.limbs, out_dir=out, device=device, **kwargs)

    if add_umap:
        plt = _import_matplotlib()
        fig = _plot_umap(result["latent"], umap_seed, plt)
        p = out / "umap_embeddings.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        result["written"].append(p)
        print(f"[analysis] wrote {p}")

    # Plain-language per-model reading of the numbers.
    try:
        meta = {"label": out.name, "checkpoint": str(checkpoint_path)}
        p = write_model_summary(result["results"], out / "summary.md", meta=meta)
        result["written"].append(p)
        print(f"[analysis] wrote {p}")
    except Exception as e:  # noqa: BLE001 - summary is non-essential
        print(f"[analysis]   summary.md skipped: {e}")

    return result


def analyze_best(sweep_out_dir: str,
                 bundle: YoutubeMotionBundle,
                 metric: str = "mpjpe_all",
                 out_dir: str | None = None,
                 **kwargs) -> dict:
    """Analyse the best model from a finished :func:`youtube_motion.driver.run_sweep`.

    Reads ``<sweep_out_dir>/results.json``, picks the run with the lowest
    ``metric`` that has a checkpoint, and runs :func:`analyze_checkpoint` on it.
    """
    results_path = Path(sweep_out_dir) / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"no results.json in {sweep_out_dir!r}; run the sweep first.")
    payload = json.loads(results_path.read_text())
    ok = [r for r in payload.get("records", [])
          if r.get("status") == "ok" and r.get("checkpoint")]
    if not ok:
        raise ValueError(
            f"no successful runs with checkpoints in {results_path}.")
    best = min(ok, key=lambda r: r.get(metric, float("inf")))
    print(f"[analysis] best by {metric}: {best['architecture']} / "
          f"recipe {best['recipe']} / {best['mask_policy']} "
          f"({metric}={best.get(metric)})")
    if out_dir is None:
        out_dir = str(Path(sweep_out_dir) / "analysis_best")
    return analyze_checkpoint(best["checkpoint"], bundle, out_dir=out_dir,
                              **kwargs)


def _dig(d, *path, default=None):
    """Safely walk a nested dict, returning ``default`` on any missing key."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and (x != x):     # NaN
            return "n/a"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def headline_metrics(results: dict) -> dict:
    """Flatten the analysis result into the handful of numbers worth comparing.

    Missing sections (an analysis that skipped) come back as ``None`` so the
    cross-model table still lines up.
    """
    d_z = _dig(results, "information", "d_z") or _dig(results, "encoder_geometry", "d_z")
    n_active = _dig(results, "information", "n_active")
    tc = _dig(results, "information", "total_correlation")
    tc = _dig(tc, "total_correlation") if isinstance(tc, dict) else tc
    return {
        "d_z": d_z,
        "n_active": n_active,
        "active_frac": (n_active / d_z if n_active is not None and d_z else None),
        "total_correlation": tc,
        "intrinsic_dim": _dig(results, "posterior_geometry", "intrinsic_dim"),
        "cluster_k": _dig(results, "posterior_geometry", "cluster_k"),
        "mmd_p_value": _dig(results, "posterior_geometry", "mmd_p_value"),
        "c2st_auc": _dig(results, "two_sample", "c2st_auc"),
        "feature_r2_mean": _dig(results, "features", "r2_mean"),
        "mig": _dig(results, "disentanglement", "mig"),
        "sap": _dig(results, "disentanglement", "sap"),
        "dci_disentanglement": _dig(results, "disentanglement", "disentanglement"),
        "decoder_mean_condition": _dig(results, "decoder_geometry", "mean_condition"),
        "encoder_n_live": _dig(results, "encoder_geometry", "n_live_units"),
        "asymmetry_mean": _dig(results, "symmetry", "asymmetry_score_mean"),
        "bone_plausibility": _dig(results, "generation", "bone_plausibility_ratio_mean"),
        "interp_curvature": _dig(results, "generation", "interpolation_curvature"),
        "mask_jitter_ratio": _dig(results, "masking", "mask_jitter_ratio"),
        "split_mpjpe_visible": _dig(results, "masking", "split_mpjpe", "mpjpe_visible"),
        "split_mpjpe_inpainted": _dig(results, "masking", "split_mpjpe", "mpjpe_inpainted"),
        "hmm_k": _dig(results, "dynamics", "hmm_k"),
        "n_segments": _dig(results, "dynamics", "n_segments"),
    }


def write_model_summary(results: dict, out_path, meta: dict | None = None):
    """Write a plain-language ``summary.md`` for one model's analysis.

    Each section states its numbers and a verdict computed from them — a
    reading of the latent space, not a fixed expectation ([post-hoc §7] style).
    """
    meta = meta or {}
    m = headline_metrics(results)
    L: list[str] = []
    L.append(f"# Latent-space analysis — {meta.get('label', 'model')}\n")
    if meta:
        bits = [f"{k} = {v}" for k, v in meta.items() if k != "label"]
        if bits:
            L.append(", ".join(bits) + "\n")

    # 1. Latent usage.
    L.append("## Latent usage — is the code collapsed?\n")
    af = m["active_frac"]
    L.append(f"- Active units: **{m['n_active']} / {m['d_z']}** "
             f"({_fmt((af or 0) * 100, 0)}%).")
    L.append(f"- Total correlation: {_fmt(m['total_correlation'])} "
             "(higher = more coupling between dims).")
    if af is not None:
        v = ("most of the latent is used — no posterior collapse." if af > 0.5
             else "a healthy fraction of dims carry signal." if af > 0.25
             else "most dims sit at the prior: posterior collapse — lower "
             "beta_max or add free_bits.")
        L.append(f"\n_{v}_\n")

    # 2. Aggregate posterior vs prior.
    L.append("## Aggregate posterior vs prior\n")
    L.append(f"- MMD test p-value: {_fmt(m['mmd_p_value'])} "
             "(small = q(z) differs from N(0, I)).")
    L.append(f"- Classifier two-sample AUC: {_fmt(m['c2st_auc'])} "
             "(0.5 = indistinguishable from the prior, 1.0 = fully separable).")
    auc = m["c2st_auc"]
    if auc is not None:
        v = ("q(z) is close to the prior — the latent is well-regularised." if auc < 0.6
             else "q(z) is moderately structured beyond the prior." if auc < 0.8
             else "q(z) departs strongly from the prior: rich structure, but "
             "generation from N(0, I) may land off-manifold.")
        L.append(f"\n_{v}_\n")

    # 3. Latent geometry.
    L.append("## Latent geometry\n")
    L.append(f"- Intrinsic dimension (TwoNN): {_fmt(m['intrinsic_dim'], 2)}")
    L.append(f"- Preferred cluster count K (BIC): {m['cluster_k']}")
    L.append(f"- Decoder pullback-metric mean condition number: "
             f"{_fmt(m['decoder_mean_condition'], 2)} (1 = isotropic; "
             ">10 = strongly anisotropic).")
    L.append(f"- Encoder live units (precision > threshold): {m['encoder_n_live']}")
    cond = m["decoder_mean_condition"]
    if cond is not None:
        v = ("the decoder maps latent directions to pose fairly isotropically."
             if cond < 10 else
             "the decoder is strongly anisotropic — a few latent directions "
             "dominate pose change, so traversals will feel uneven.")
        L.append(f"\n_{v}_\n")

    # 4. Kinematic content.
    if _dig(results, "features") is not None:
        L.append("## Kinematic content of the latent\n")
        L.append(f"- Mean held-out R² predicting kinematic features from z: "
                 f"**{_fmt(m['feature_r2_mean'])}**")
        r2 = m["feature_r2_mean"]
        if r2 is not None:
            v = ("the latent linearly encodes most kinematic features." if r2 > 0.5
                 else "the latent captures some kinematics linearly." if r2 > 0.2
                 else "little kinematic content is linearly decodable from z.")
            L.append(f"\n_{v}_\n")

    # 5. Disentanglement.
    if _dig(results, "disentanglement") is not None:
        L.append("## Disentanglement\n")
        L.append(f"- MIG: {_fmt(m['mig'])}   SAP: {_fmt(m['sap'])}   "
                 f"DCI-disentanglement: {_fmt(m['dci_disentanglement'])}")
        L.append("\n_Higher is cleaner factor-to-dim alignment; these are "
                 "relative scores, best read across models below._\n")

    # 6. Symmetry.
    if m["asymmetry_mean"] is not None:
        L.append("## Laterality\n")
        L.append(f"- Mean asymmetry score: {_fmt(m['asymmetry_mean'])} "
                 "(projection onto the antisymmetric left-right subspace).")
        L.append("")

    # 7. Dynamics.
    L.append("## Temporal dynamics\n")
    if m["hmm_k"] is not None:
        dwell = _dig(results, "dynamics", "hmm_dwell_seconds")
        ou = _dig(results, "dynamics", "ou_timescales_seconds")
        L.append(f"- HMM regimes: **{m['hmm_k']} states**; change-point "
                 f"segments: {m['n_segments']}.")
        if dwell:
            L.append(f"- Mean dwell per state (s): "
                     f"{[round(float(x), 2) for x in dwell]}")
        if ou:
            L.append(f"- Ornstein-Uhlenbeck return timescales (s): "
                     f"{[round(float(x), 2) for x in ou][:6]}")
        L.append(f"\n_A {m['hmm_k']}-state HMM segments the outer-loop "
                 "trajectory into behavioural regimes._\n")
    else:
        L.append("_Dynamics/HMM skipped (video too short, or hmmlearn missing)._\n")

    # 8. Generation & robustness.
    L.append("## Generation & masking robustness\n")
    L.append(f"- Bone-length plausibility ratio: {_fmt(m['bone_plausibility'])} "
             "(closer to 1 = generated bones match real bone lengths).")
    L.append(f"- Interpolation curvature: {_fmt(m['interp_curvature'])}")
    L.append(f"- Mask-jitter ratio: {_fmt(m['mask_jitter_ratio'])} "
             "(< 0.1 = pose, not mask, sets the latent).")
    L.append(f"- Split MPJPE — visible {_fmt(m['split_mpjpe_visible'])}, "
             f"inpainted {_fmt(m['split_mpjpe_inpainted'])}.")
    jit = m["mask_jitter_ratio"]
    if jit is not None:
        v = ("the latent is mask-robust — repeated mask draws barely move it."
             if jit < 0.1 else
             "the mask still perturbs the latent noticeably; the encoder leans "
             "partly on which joints are hidden.")
        L.append(f"\n_{v}_\n")

    from pathlib import Path as _P
    p = _P(out_path)
    p.write_text("\n".join(L) + "\n")
    return p


def compare_models(labeled_results: dict[str, dict], out_dir) -> dict:
    """Cross-model comparison table from ``{label: results}`` → comparison.{md,json}.

    ``labeled_results`` maps a model label (e.g. ``"conv/recipe1/uniform"``) to
    that model's ``run_all_analyses`` ``results`` dict. Writes a ranked-by-label
    markdown table of the headline metrics plus the machine-readable json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {label: headline_metrics(res) for label, res in labeled_results.items()}

    cols = [
        ("n_active", "active", 0), ("active_frac", "act.frac", 2),
        ("total_correlation", "TC", 2), ("intrinsic_dim", "intr.dim", 1),
        ("cluster_k", "K", 0), ("c2st_auc", "c2st AUC", 2),
        ("feature_r2_mean", "feat.R²", 3), ("mig", "MIG", 3),
        ("dci_disentanglement", "DCI", 3),
        ("decoder_mean_condition", "dec.cond", 1),
        ("encoder_n_live", "enc.live", 0),
        ("asymmetry_mean", "asym", 3), ("hmm_k", "HMM K", 0),
        ("n_segments", "segs", 0),
        ("split_mpjpe_inpainted", "MPJPE inp", 3),
        ("mask_jitter_ratio", "jitter", 3),
    ]
    header = "| model | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(cols)) + "|"
    lines = ["# Cross-model latent-space comparison", "",
             f"{len(rows)} models. Headline metrics from each model's full "
             "analysis (see each model's `summary.md` for the reading).", "",
             header, sep]
    for label in sorted(rows):
        m = rows[label]
        cells = [_fmt(m.get(k), nd) for k, _, nd in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "**How to read:** `act.frac` high = no collapse; "
              "`c2st AUC` near 0.5 = latent close to the prior; higher "
              "`MIG`/`DCI` = cleaner disentanglement; `dec.cond` near 1 = "
              "isotropic decoder; `jitter` < 0.1 = mask-robust latent."]

    md = out_dir / "comparison.md"
    md.write_text("\n".join(lines) + "\n")
    js = out_dir / "comparison.json"
    js.write_text(json.dumps({"metrics": rows}, indent=2, default=float))
    print(f"[analysis] wrote {md}")
    print(f"[analysis] wrote {js}")
    return {"metrics": rows, "comparison_md": str(md), "comparison_json": str(js)}


def analyze_sweep(sweep_out_dir: str,
                  bundle: YoutubeMotionBundle,
                  which: str = "best_per_arch",
                  metric: str = "mpjpe_all",
                  out_dir: str | None = None,
                  **kwargs) -> dict:
    """Analyse several models from a sweep and write a cross-model comparison.

    Args:
        sweep_out_dir: a finished :func:`youtube_motion.driver.run_sweep` dir.
        bundle: the dataset the models trained on.
        which: ``"best_per_arch"`` (default — the best run of each backbone),
            ``"best"`` (single best), or ``"all"`` (every successful run —
            slow: the full battery per model).
        metric: ranking metric for the selection.
        out_dir: where to write ``comparison.{md,json}``; defaults to
            ``<sweep_out_dir>/analysis_comparison``. Each model's own outputs
            go to ``<that>/<label>/``.
        **kwargs: forwarded to :func:`analyze_checkpoint` (e.g. ``device``,
            ``include_persistent_homology=False`` to speed things up).
    Returns:
        ``{"labeled_results", "comparison"}``.
    """
    payload = json.loads((Path(sweep_out_dir) / "results.json").read_text())
    ok = [r for r in payload.get("records", [])
          if r.get("status") == "ok" and r.get("checkpoint")]
    if not ok:
        raise ValueError(f"no successful runs with checkpoints in {sweep_out_dir}.")

    if which == "best":
        chosen = [min(ok, key=lambda r: r.get(metric, float("inf")))]
    elif which == "best_per_arch":
        by_arch: dict[str, dict] = {}
        for r in ok:
            a = r["architecture"]
            if a not in by_arch or r.get(metric, 1e9) < by_arch[a].get(metric, 1e9):
                by_arch[a] = r
        chosen = list(by_arch.values())
    elif which == "all":
        chosen = ok
    else:
        raise ValueError(f"which must be best|best_per_arch|all, got {which!r}")

    if out_dir is None:
        out_dir = str(Path(sweep_out_dir) / "analysis_comparison")
    out_root = Path(out_dir)
    preview = [f"{r['architecture']}/r{r['recipe']}/{r['mask_policy']}"
               for r in chosen]
    print(f"[analysis] comparing {len(chosen)} model(s): {preview}")

    labeled: dict[str, dict] = {}
    for r in chosen:
        label = f"{r['architecture']}_recipe{r['recipe']}_{r['mask_policy']}"
        sub = out_root / label
        res = analyze_checkpoint(r["checkpoint"], bundle, out_dir=str(sub), **kwargs)
        labeled[label] = res["results"]

    comp = compare_models(labeled, out_root)
    return {"labeled_results": labeled, "comparison": comp}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Latent-space analysis (UMAP, Jacobians, HMM, geometry) "
                    "for a YouTube 2D-keypoint VAE checkpoint.")
    ap.add_argument("--csv", required=True, help="the keypoint CSV (same data "
                    "the model trained on)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="path to a best.pt to analyse")
    src.add_argument("--sweep-out", help="a sweep output dir; analyse its best "
                     "model by --metric")
    ap.add_argument("--out", default=None, help="output dir for plots + json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--metric", default="mpjpe_all",
                    choices=["mpjpe_all", "mpjpe_inpainted"])
    ap.add_argument("--which", default="best",
                    choices=["best", "best_per_arch", "all"],
                    help="with --sweep-out: analyse the single best model, the "
                    "best of each backbone, or every model, and write a "
                    "cross-model comparison.md")
    ap.add_argument("--preprocess", default="none",
                    choices=["none", "center", "center_scale"])
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--no-umap", action="store_true",
                    help="skip the UMAP embedding plot")
    ap.add_argument("--no-dynamics", action="store_true",
                    help="skip the dynamics/HMM section")
    ap.add_argument("--no-persistent-homology", action="store_true",
                    help="skip persistent homology (needs ripser)")
    args = ap.parse_args(argv)

    from .data import load_youtube_csv
    bundle = load_youtube_csv(args.csv, preprocess=args.preprocess,
                              max_videos=args.max_videos)
    print(f"[data] {bundle.summary()}")

    kwargs = dict(
        device=args.device,
        add_umap=not args.no_umap,
        include_dynamics=not args.no_dynamics,
        include_persistent_homology=not args.no_persistent_homology,
    )
    if args.checkpoint:
        analyze_checkpoint(args.checkpoint, bundle, out_dir=args.out, **kwargs)
    elif args.which == "best":
        analyze_best(args.sweep_out, bundle, metric=args.metric,
                     out_dir=args.out, **kwargs)
    else:
        analyze_sweep(args.sweep_out, bundle, which=args.which,
                      metric=args.metric, out_dir=args.out, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
