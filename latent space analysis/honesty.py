"""Statistical honesty for small video counts (Part I Section 12).

With a handful of videos, between-video claims need care. Frame-level and
clip-level statistics stand without hedging; between-video differences
need a block bootstrap for the interval and a permutation test for the
p-value, both blocked in time to respect the window overlap.
"""

from __future__ import annotations

import numpy as np


def _time_blocks(time_index: np.ndarray, block_seconds: float, fps: float):
    """Assign each clip to a time block of the given length."""
    block_frames = block_seconds * fps
    return (time_index // block_frames).astype(int)


def block_bootstrap(values: np.ndarray, blocks: np.ndarray,
                    statistic=np.mean, n_boot: int = 500,
                    ci: float = 0.95,
                    rng: np.random.Generator | None = None) -> dict:
    """Block bootstrap of a per-clip statistic (Section 12).

    Resamples whole time blocks with replacement, so neighbouring
    overlapping clips move together and do not fake precision.

    Args:
        values: per-clip values, shape (N,).
        blocks: block label per clip, shape (N,); see `time_blocks`.
        statistic: the summary to bootstrap.
        n_boot: bootstrap rounds.
        ci: interval mass.
    Returns:
        Dict with the point estimate and the interval bounds.
    """
    rng = np.random.default_rng() if rng is None else rng
    uniq = np.unique(blocks)
    point = statistic(values)
    boot = np.zeros(n_boot)
    for r in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(blocks == b)[0] for b in pick])
        boot[r] = statistic(values[idx])
    lo = np.percentile(boot, 100 * (1 - ci) / 2)
    hi = np.percentile(boot, 100 * (1 + ci) / 2)
    return {"point": float(point), "low": float(lo), "high": float(hi)}


def permutation_between_videos(values: np.ndarray, video_id: np.ndarray,
                               blocks: np.ndarray, n_perm: int = 500,
                               rng: np.random.Generator | None = None) -> dict:
    """Permutation test of a between-video difference (Section 12).

    Shuffles the video label within time blocks, so the null keeps the
    within-video correlation while breaking the video assignment. Works
    for exactly two videos; the statistic is the difference in means.

    Args:
        values: per-clip values, shape (N,).
        video_id: video label per clip, shape (N,).
        blocks: block label per clip, shape (N,).
        n_perm: permutation rounds.
    Returns:
        Dict with the observed difference and the two-sided p-value.
    """
    rng = np.random.default_rng() if rng is None else rng
    vids = np.unique(video_id)
    if len(vids) != 2:
        raise ValueError("This test compares exactly two videos.")

    def diff(labels):
        return values[labels == vids[0]].mean() - values[labels == vids[1]].mean()

    observed = diff(video_id)
    count = 0
    for _ in range(n_perm):
        permuted = video_id.copy()
        for b in np.unique(blocks):
            idx = np.where(blocks == b)[0]
            permuted[idx] = rng.permutation(video_id[idx])
        if abs(diff(permuted)) >= abs(observed):
            count += 1
    return {"observed": float(observed), "p_value": (count + 1) / (n_perm + 1)}


def time_blocks(latent, block_seconds: float = 5.0, fps: float = 25.0) -> np.ndarray:
    """Build time-block labels from a LatentSet's frame index."""
    if latent.time_index is None:
        raise ValueError("LatentSet has no time_index; cannot block in time.")
    return _time_blocks(latent.time_index, block_seconds, fps)
