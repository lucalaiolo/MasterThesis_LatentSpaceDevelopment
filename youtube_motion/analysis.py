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
    else:
        analyze_best(args.sweep_out, bundle, metric=args.metric,
                     out_dir=args.out, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
