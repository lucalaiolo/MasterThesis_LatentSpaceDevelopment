"""Post-training latent-space analysis.

`train` writes checkpoints with the model state, the training config,
and the epoch number. `load_checkpoint` rebuilds the model from a
checkpoint file, and `run_latent_analysis` drives the four latent
diagnostics already in `visualize.py` (per-dim KL, active units, PCA
scatter, latent traversal) on a fresh validation loader and writes the
resulting PNGs to disk.

Typical use after a sweep:

    from architectures.analyze import run_latent_analysis
    run_latent_analysis(
        "/content/runs/plain_vae_freebits/recipe1_uniform/best.pt",
        videos,
        pca_color="video_id",
        traversal_dims="top_active",
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import TrainingConfig
from .data import build_clips, make_loader, train_val_split
from .mask_policies import build_policy
from .models import build_model


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Load a training checkpoint and rebuild its model.

    The training loop dumps `{"model": state_dict, "config": dict,
    "epoch": int}`. We reconstruct the `TrainingConfig`, rebuild the
    architecture, and load the weights.

    Returns:
        (model, config): the model on `device` in eval mode, and the
        TrainingConfig it was trained with.
    """
    import torch
    ckpt = torch.load(path, map_location=device)
    config = TrainingConfig(**ckpt["config"])
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def _pick_traversal_dims(stats: dict, spec, k: int = 8) -> list[int]:
    """Resolve `traversal_dims` argument to a concrete list of indices."""
    d_z = stats["mus"].shape[1]
    if spec is None or spec == "top_active":
        # Rank by KL contribution — highest = most information carried.
        order = np.argsort(stats["kl_per_dim"])[::-1]
        return order[:min(k, d_z)].tolist()
    if spec == "all":
        return list(range(d_z))
    return list(spec)


def run_latent_analysis(checkpoint_path: str | Path,
                        videos: list[np.ndarray],
                        limbs: dict[str, list[int]] | None = None,
                        out_dir: str | Path | None = None,
                        device: str = "cpu",
                        stride: int | None = None,
                        max_batches: int | None = 32,
                        pca_color: str | None = "video_id",
                        traversal_dims: str | list[int] | None = "top_active",
                        traversal_n_dims: int = 8,
                        traversal_alphas=(-2.0, -1.0, 0.0, 1.0, 2.0),
                        traversal_joint: int = 0,
                        include_traversal: bool = True,
                        ) -> dict:
    """Load a checkpoint and write the standard latent-analysis plots.

    Args:
        checkpoint_path: `.pt` file written by `train`.
        videos: same list of videos used at training.
        limbs: joint-index lists per limb name; only used if the model
            was trained with the `"limb"` policy.
        out_dir: where PNGs are written. Defaults to
            `<checkpoint dir>/latent_analysis/`.
        device: `"cpu"` or `"cuda"`.
        stride: clip stride for the val loader; defaults to
            `clip_length // 2` (same as training).
        max_batches: cap on val batches used to gather posterior
            statistics. `None` = all.
        pca_color: `"video_id"` to colour the PCA scatter by source
            video, `"time_index"` to colour by clip start frame, `None`
            for uniform colour.
        traversal_dims: which latent dims to sweep. `"top_active"`
            (default) picks the highest-KL dims, `"all"` sweeps every
            dim, or pass a list of indices.
        traversal_n_dims: how many top-active dims to show when
            `traversal_dims="top_active"`.
        traversal_alphas: offsets to add to the picked latent dim.
        traversal_joint: index of the joint whose xyz trajectory is
            plotted for each traversal cell.
        include_traversal: turn off if you only want the fast plots.

    Returns:
        Dict with `stats` (from `collect_latent_stats`), `written`
        (list of PNG paths), `config`, and `model`.
    """
    # Import at call-time so the module loads without matplotlib/torch.
    from .visualize import (collect_latent_stats,
                            plot_latent_kl_per_dim,
                            plot_active_units,
                            plot_latent_pca,
                            plot_latent_traversal)
    import matplotlib.pyplot as plt

    model, config = load_checkpoint(checkpoint_path, device=device)
    if stride is None:
        stride = config.clip_length // 2

    # Rebuild the val loader exactly as `train` did so the analysis
    # numbers are comparable to `history.json`.
    clips, video_id, time_index = build_clips(
        videos, config.clip_length, stride,
    )
    _, val_mask = train_val_split(clips, video_id)
    val_clips = clips[val_mask]
    val_video_id = video_id[val_mask]
    val_time_idx = time_index[val_mask]

    policy = build_policy(config, limbs=limbs)
    val_loader = make_loader(val_clips, policy, config.batch_size,
                             shuffle=False, seed=config.seed + 1)

    if out_dir is None:
        out_dir = Path(checkpoint_path).parent / "latent_analysis"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    def _save(fig, name: str):
        p = out_dir / name
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    stats = collect_latent_stats(model, val_loader, device=device,
                                 max_batches=max_batches)
    n_active = int(stats["active_units"].sum())
    d_z = int(stats["mus"].shape[1])
    print(f"[analysis] {stats['mus'].shape[0]} clips encoded, "
          f"{n_active}/{d_z} active latent units")

    _save(plot_latent_kl_per_dim(stats), "latent_kl_per_dim.png")
    _save(plot_active_units(stats), "active_units.png")

    if stats["mus"].shape[0] >= 2 and d_z >= 2:
        # `collect_latent_stats` only reads `max_batches` batches, so
        # the colour vector needs to be trimmed to the same prefix.
        n_used = stats["mus"].shape[0]
        colors, label = None, ""
        if pca_color == "video_id":
            colors, label = val_video_id[:n_used].astype(np.float64), "video id"
        elif pca_color == "time_index":
            colors, label = val_time_idx[:n_used].astype(np.float64), "clip start frame"
        _save(plot_latent_pca(stats, colors=colors, label=label),
              "latent_pca.png")

    if include_traversal:
        dims = _pick_traversal_dims(stats, traversal_dims,
                                    k=traversal_n_dims)
        # Reference clip: first val clip, mask all-ones so traversal
        # shows the effect of z_d alone, uncoupled from any masking.
        ref_clip = val_clips[0]
        ref_mask = np.ones(ref_clip.shape[:2], dtype=np.float32)
        _save(plot_latent_traversal(model, ref_clip, ref_mask,
                                    dims=dims, alphas=traversal_alphas,
                                    device=device,
                                    joint_idx=traversal_joint),
              "latent_traversal.png")

    print(f"[analysis] wrote {len(written)} figure(s) to {out_dir}")
    return {"stats": stats, "written": written,
            "config": config, "model": model}
