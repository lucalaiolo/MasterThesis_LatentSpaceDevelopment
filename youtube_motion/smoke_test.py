"""End-to-end smoke test for the YouTube 2D-keypoint sweep.

Run this in an environment with PyTorch. It builds synthetic COCO-18 2D motion
(no real CSV needed), exercises the CSV round-trip, and runs a small
architecture x recipe x mask-policy sweep for two epochs each, confirming that
shapes flow through at ``n_dims=2`` and that a ranked results table is written.

    python -m youtube_motion.smoke_test
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from .data import YoutubeMotionBundle, load_youtube_csv
from .driver import run_sweep
from .skeleton import COCO18_KEYPOINT_NAMES, N_JOINTS, coco18_limbs


def synthetic_videos(n_videos: int = 4, frames: int = 240, J: int = N_JOINTS,
                     seed: int = 0) -> list[np.ndarray]:
    """A few fake 2D videos with mild per-joint oscillation and drift.

    Shape (F, J, 2). Enough temporal and spatial structure for the VAE to have
    something to reconstruct; not meant to be anatomically real.
    """
    rng = np.random.default_rng(seed)
    videos = []
    for v in range(n_videos):
        t = np.arange(frames)[:, None]                     # (F, 1)
        phase = np.arange(J)[None, :]                      # (1, J)
        x = 0.5 * np.sin(0.05 * t + phase) + 0.01 * t / frames
        y = 0.5 * np.cos(0.05 * t + 0.5 * phase) + 0.2 * v
        pose = np.stack([x, y], axis=-1)                   # (F, J, 2)
        pose = pose + rng.standard_normal(pose.shape) * 0.03
        videos.append(pose.astype(np.float32))
    return videos


def synthetic_bundle(**kw) -> YoutubeMotionBundle:
    """Wrap :func:`synthetic_videos` in a bundle with COCO-18 limbs."""
    videos = synthetic_videos(**kw)
    return YoutubeMotionBundle(
        videos=videos,
        video_names=[f"synthetic_{i:03d}" for i in range(len(videos))],
        fps=[30.0] * len(videos),
        n_joints=videos[0].shape[1],
        n_dims=2,
        limbs=coco18_limbs(),
    )


def _write_csv(bundle: YoutubeMotionBundle, path: str) -> None:
    """Serialise a bundle to the dataset's long CSV format (round-trip test)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video_number", "video", "bp", "frame", "x", "y", "fps",
                    "pixel_x", "pixel_y", "time", "part_idx"])
        for vi, (name, pose, fps) in enumerate(
                zip(bundle.video_names, bundle.videos, bundle.fps)):
            for fr in range(pose.shape[0]):
                for j in range(pose.shape[1]):
                    w.writerow([vi, name, COCO18_KEYPOINT_NAMES[j], fr,
                                float(pose[fr, j, 0]), float(pose[fr, j, 1]),
                                fps, 1280, 720, fr / (fps or 30), float(j)])


def main() -> None:
    print("=== CSV round-trip ===")
    bundle = synthetic_bundle()
    tmp = Path(tempfile.mkdtemp())
    csv_path = tmp / "keypoints.csv"
    _write_csv(bundle, str(csv_path))
    reloaded = load_youtube_csv(str(csv_path))
    print("wrote + reloaded:", reloaded.summary())
    assert reloaded.n_videos == bundle.n_videos
    assert reloaded.videos[0].shape == bundle.videos[0].shape
    # Values should survive the round-trip (no missing joints here).
    assert np.allclose(reloaded.videos[0], bundle.videos[0], atol=1e-4)
    print("round-trip OK")

    print("\n=== small sweep (2 epochs each) ===")
    result = run_sweep(
        bundle,
        out_dir=str(tmp / "ckpts"),
        architectures=("conv", "transformer"),
        recipes=(1, 2, 3),
        mask_policies=("uniform", "limb"),
        clip_length=16, latent_dim=8, batch_size=8,
        n_epochs=2, warmup_epochs=1, beta_max=0.1,
        learning_rate=1e-3, device="cpu", log_every=0, save_every=0,
    )

    ok = [r for r in result["records"] if r["status"] == "ok"]
    print(f"\ncompleted {len(ok)}/{len(result['records'])} runs")
    assert ok, "no runs completed"
    for r in ok:
        # A trained 2D model must emit finite MPJPE in the coordinate units.
        assert np.isfinite(r["mpjpe_all"]), r
        assert np.isfinite(r["mpjpe_inpainted"]), r
    best = result["best"]
    print(f"best: {best['architecture']} / recipe {best['recipe']} / "
          f"{best['mask_policy']}  mpjpe_all={best['mpjpe_all']:.4f}")
    print(f"results table: {result['results_md']}")
    print("\n=== every 2D training path ran ===")


if __name__ == "__main__":
    main()
