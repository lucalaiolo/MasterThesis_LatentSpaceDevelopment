"""Clip dataset for the neonate motion VAE.

Videos come in as arrays of shape (F, J, 3): a sequence of F frames of J
joints in 3D. The dataset slices each video into overlapping windows of
length T at stride s, drawing masks fresh per epoch under whichever
policy the training loop passes in.
"""

from __future__ import annotations

import numpy as np


def slice_video(video: np.ndarray, T: int, stride: int) -> np.ndarray:
    """Cut one video into overlapping clips of length T at the given stride.

    Args:
        video: shape (F, J, 3).
        T: clip length.
        stride: hop between clip starts.
    Returns:
        Clips of shape (K, T, J, 3), where K depends on F and stride.
    """
    F = video.shape[0]
    starts = list(range(0, max(F - T + 1, 0), stride))
    if not starts:
        return video[None, :T].copy() if F >= T else np.empty((0, T) + video.shape[1:])
    return np.stack([video[s:s + T] for s in starts])


def build_clips(videos: list[np.ndarray], T: int, stride: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a set of videos and record which video each clip came from.

    Args:
        videos: list of video arrays, each shape (F_v, J, 3), same J.
        T: clip length.
        stride: hop between clip starts.
    Returns:
        (clips, video_id, time_index):
            clips shape (N, T, J, 3),
            video_id shape (N,) with the video number per clip,
            time_index shape (N,) with the start frame per clip.
    """
    parts_X, parts_v, parts_t = [], [], []
    for i, v in enumerate(videos):
        F = v.shape[0]
        starts = list(range(0, max(F - T + 1, 0), stride))
        if not starts:
            continue
        parts_X.append(np.stack([v[s:s + T] for s in starts]))
        parts_v.append(np.full(len(starts), i, dtype=np.int64))
        parts_t.append(np.asarray(starts, dtype=np.int64))
    if not parts_X:
        raise ValueError("No clips built. Are the videos shorter than T?")
    return (np.concatenate(parts_X), np.concatenate(parts_v),
            np.concatenate(parts_t))


class ClipDataset:
    """A torch Dataset that yields (clip, mask) pairs.

    The mask is redrawn every access, so a training epoch sees fresh
    masks even on the same clip.
    """

    def __init__(self, clips: np.ndarray, mask_policy, seed: int = 0):
        """
        Args:
            clips: shape (N, T, J, 3).
            mask_policy: any object with a `sample(T, J, rng)` method.
            seed: seeds the per-clip mask draws.
        """
        # Import torch lazily so the module loads without torch installed.
        import torch
        self.torch = torch
        from torch.utils.data import Dataset  # noqa: F401
        self.clips = clips.astype(np.float32)
        self.policy = mask_policy
        self.rng = np.random.default_rng(seed)
        self.T = clips.shape[1]
        self.J = clips.shape[2]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        X = self.clips[i]
        # Pass the clip to the policy so speed-based policies ([MVAE
        # §2.3–2.5]) can compute their scores. Score-free policies
        # ignore it.
        M = self.policy.sample(self.T, self.J, self.rng, X=X)
        return (self.torch.from_numpy(X),
                self.torch.from_numpy(M))


def make_loader(clips: np.ndarray, mask_policy, batch_size: int,
                shuffle: bool = True, seed: int = 0):
    """Build a torch DataLoader over the clips."""
    from torch.utils.data import DataLoader
    ds = ClipDataset(clips, mask_policy, seed=seed)
    # We keep num_workers at 0 by default. Neonate sets are small and the
    # per-item work is light; workers rarely help and add mask-seeding
    # bookkeeping we do not want.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train_val_split(clips: np.ndarray, video_id: np.ndarray,
                    val_fraction: float = 0.15, seed: int = 0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Split by holding out the last fraction of each video by time.

    Splitting at random would leak content between neighbouring clips.
    Splitting by clip index within each video keeps the two halves apart.

    Returns:
        (train_mask, val_mask): boolean arrays over the clip index.
    """
    train = np.zeros(len(clips), dtype=bool)
    val = np.zeros(len(clips), dtype=bool)
    for v in np.unique(video_id):
        idx = np.where(video_id == v)[0]
        cut = int(len(idx) * (1 - val_fraction))
        train[idx[:cut]] = True
        val[idx[cut:]] = True
    return train, val
