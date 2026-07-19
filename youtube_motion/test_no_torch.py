"""Structural checks that run without PyTorch (numpy + stdlib only).

Covers the parts of the YouTube pipeline that don't need a GPU or torch: the
COCO-18 skeleton, the CSV adapter (pivot, missing-joint interpolation, non-zero
start frames, custom column names), the preprocessing modes, and the analytical
2D parameter counts. The torch-dependent training path is covered by
``smoke_test.py``.

    python -m youtube_motion.test_no_torch
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from architectures import TrainingConfig
from architectures.param_counts import summarise

from .data import (interpolate_missing, load_youtube_csv, preprocess_video,
                   root_center, torso_scale)
from .skeleton import (COCO18_BONES, COCO18_KEYPOINT_NAMES, COCO18_LEFT_RIGHT,
                       COCO18_LIMBS, N_DIMS, N_JOINTS, ROOT_JOINT)


def _write_toy_csv(path: str, videos: dict[str, tuple[int, int]],
                   drop: set[tuple[str, int, int]] | None = None,
                   header=("video_number", "video", "bp", "frame", "x", "y",
                           "fps", "pixel_x", "pixel_y", "time", "part_idx")):
    """Write a toy long CSV. `videos` maps name -> (start_frame, n_frames)."""
    drop = drop or set()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for vn, (name, (f0, F)) in enumerate(videos.items()):
            for fr in range(f0, f0 + F):
                for pi, bp in enumerate(COCO18_KEYPOINT_NAMES):
                    if (name, fr, pi) in drop:
                        continue
                    x = np.sin(0.1 * fr + pi)
                    y = np.cos(0.1 * fr + pi)
                    w.writerow([vn, name, bp, fr, x, y, 29.97, 1280, 720,
                                fr / 29.97, float(pi)])


def test_skeleton():
    assert N_JOINTS == 18 and N_DIMS == 2
    assert len(COCO18_KEYPOINT_NAMES) == 18
    assert COCO18_KEYPOINT_NAMES[ROOT_JOINT] == "Neck"
    # Every limb / bone index is a valid joint.
    for name, idx in COCO18_LIMBS.items():
        assert all(0 <= i < N_JOINTS for i in idx), name
    for a, b in COCO18_BONES:
        assert 0 <= a < N_JOINTS and 0 <= b < N_JOINTS, (a, b)
    # Limbs are disjoint (no joint in two limbs) — a clean masking partition.
    seen: set[int] = set()
    for idx in COCO18_LIMBS.values():
        assert seen.isdisjoint(idx)
        seen.update(idx)
    # Left-right pairs: valid, distinct, and never a midline joint.
    midline = {0, 1}  # Nose, Neck
    for l, r in COCO18_LEFT_RIGHT:
        assert 0 <= l < N_JOINTS and 0 <= r < N_JOINTS, (l, r)
        assert l != r and l not in midline and r not in midline, (l, r)
        assert COCO18_KEYPOINT_NAMES[l].startswith("L")
        assert COCO18_KEYPOINT_NAMES[r].startswith("R")
    print(f"skeleton: 18 COCO joints, {len(COCO18_BONES)} bones, "
          f"{len(COCO18_LIMBS)} limbs, {len(COCO18_LEFT_RIGHT)} L/R pairs, "
          f"all indices valid")


def test_analysis_skeleton():
    # coco18_skeleton() builds a valid vae_analysis Skeleton (numpy-only path).
    from .analysis import coco18_skeleton
    skel = coco18_skeleton()
    assert skel.n_joints == N_JOINTS
    assert len(skel.bones) == len(COCO18_BONES)
    assert len(skel.left_right) == len(COCO18_LEFT_RIGHT)
    assert skel.limbs and skel.lateral_axis == 0
    # The flip permutation must be a genuine involution (swap twice = identity).
    P = skel.flip_permutation()
    assert np.allclose(P @ P, np.eye(N_JOINTS))
    print(f"analysis skeleton: Skeleton(J={skel.n_joints}, "
          f"{len(skel.bones)} bones, {len(skel.left_right)} L/R), "
          f"flip is an involution  OK")


def test_interpolate_missing():
    F, J = 5, 18
    pose = np.full((F, J, 2), np.nan)
    # Joint 0 seen only at t=0 and t=4 -> interior linearly filled.
    pose[0, 0] = [0.0, 10.0]
    pose[4, 0] = [4.0, 14.0]
    # Joint 1 never seen -> filled with 0.
    filled = interpolate_missing(pose)
    assert np.isfinite(filled).all()
    assert np.allclose(filled[:, 0, 0], [0, 1, 2, 3, 4]), filled[:, 0, 0]
    assert np.allclose(filled[:, 0, 1], [10, 11, 12, 13, 14])
    assert np.allclose(filled[:, 1], 0.0)
    print("interpolate_missing: interior linear fill + all-missing -> 0  OK")


def test_csv_basic():
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "toy.csv"
    # video A: frames 0..3; video B: frames 5..7 (non-zero start).
    _write_toy_csv(str(p), {"video_000000": (0, 4), "video_000001": (5, 3)})
    b = load_youtube_csv(str(p))
    assert b.n_videos == 2
    assert b.videos[0].shape == (4, 18, 2), b.videos[0].shape
    assert b.videos[1].shape == (3, 18, 2), b.videos[1].shape   # start shifted to 0
    assert np.isfinite(b.videos[0]).all()
    assert abs(b.fps[0] - 29.97) < 1e-2
    assert b.limbs and "left_arm" in b.limbs
    print(f"csv_basic: {b.summary()}")


def test_csv_missing_interpolated():
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "toy.csv"
    # Drop RWrist (idx 4) at frame 1 of video A -> must be interpolated.
    _write_toy_csv(str(p), {"vidA": (0, 4)}, drop={("vidA", 1, 4)})
    b = load_youtube_csv(str(p))
    got = b.videos[0][1, 4, 0]
    expect = 0.5 * (np.sin(0.1 * 0 + 4) + np.sin(0.1 * 2 + 4))
    assert np.isfinite(got) and abs(got - expect) < 1e-5, (got, expect)
    print(f"csv_missing: RWrist@f1 interpolated to {got:.5f} (~{expect:.5f})  OK")


def test_csv_custom_columns():
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "custom.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "t", "px", "py", "joint"])
        for fr in range(3):
            for j in range(18):
                w.writerow(["c0", fr, 0.1 * j, 0.2 * fr, j])
    b = load_youtube_csv(str(p), columns={"video": "clip", "frame": "t",
                                          "x": "px", "y": "py",
                                          "part_idx": "joint"})
    assert b.n_videos == 1 and b.videos[0].shape == (3, 18, 2)
    print("csv_custom_columns: remapped names load correctly  OK")


def test_preprocess_modes():
    rng = np.random.default_rng(0)
    pose = rng.standard_normal((10, 18, 2)).astype(np.float32) + 5.0
    same = preprocess_video(pose, "none")
    assert np.allclose(same, pose)
    centred = preprocess_video(pose, "center")
    assert np.allclose(centred[:, ROOT_JOINT], 0.0, atol=1e-5)
    scaled = preprocess_video(pose, "center_scale")
    assert np.allclose(scaled[:, ROOT_JOINT], 0.0, atol=1e-5)
    # After scaling, the median torso length is ~1.
    assert abs(torso_scale(scaled) - 1.0) < 0.5
    # root_center with centroid (root_joint<0) zeroes the per-frame mean.
    cen = root_center(pose, root_joint=-1)
    assert np.allclose(cen.mean(axis=1), 0.0, atol=1e-5)
    print("preprocess: none / center / center_scale behave as specified  OK")


def test_param_counts_2d():
    # 2D counts must scale from 3D by the coordinate-dim change only.
    for arch in ("conv", "transformer"):
        c2 = TrainingConfig(architecture=arch, clip_length=32, n_joints=18,
                            n_dims=2, latent_dim=32)
        c3 = TrainingConfig(architecture=arch, clip_length=32, n_joints=18,
                            n_dims=3, latent_dim=32)
        t2, t3 = summarise(c2)["total"], summarise(c3)["total"]
        assert t2 < t3, (arch, t2, t3)         # fewer coords -> fewer params
        print(f"param_counts {arch}: 2D total = {t2:,}  (3D = {t3:,})")
    # The input head shrinks by exactly J params per removed coord channel,
    # output head by J*(kernel or 1). Just assert the conv encoder_block_1
    # input width is (D+1)J.
    conv2 = summarise(TrainingConfig(architecture="conv", clip_length=32,
                                     n_joints=18, n_dims=2))
    # encoder_block_1 = k*(D+1)J*C + C = 5*3*18*64 + 64
    assert conv2["encoder_block_1"] == 5 * (3 * 18) * 64 + 64, conv2["encoder_block_1"]
    print("param_counts: 2D conv encoder input width = (D+1)J confirmed  OK")


def main():
    test_skeleton()
    test_analysis_skeleton()
    test_interpolate_missing()
    test_csv_basic()
    test_csv_missing_interpolated()
    test_csv_custom_columns()
    test_preprocess_modes()
    test_param_counts_2d()
    print("\n=== all no-torch checks passed ===")


if __name__ == "__main__":
    main()
