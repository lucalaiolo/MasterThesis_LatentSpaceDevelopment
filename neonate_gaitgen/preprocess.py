"""2D keypoint preprocessing + dataset ([plan §2]).

Pure-NumPy preprocessing (root-centre, torso-scale normalise, window) plus a
torch ``Dataset`` that yields ``(x, c_p, c_nuis, subject)``. Everything is
generic in the keypoint count ``J``; nothing is resampled (windows are at
the native frame rate, [plan §2.4]).

Asymmetry is signal here, so **no left-right mirroring** is applied
([plan §2.3]) — neither as preprocessing nor as augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---- Preprocessing (NumPy, testable without torch) ------------------------

def root_center(pose: np.ndarray, root_joint: int = 0) -> np.ndarray:
    """Subtract the root keypoint from every joint, per frame ([plan §2.1]).

    Args:
        pose: (F, J, 2).
        root_joint: pelvis / torso-midpoint index; if < 0, the per-frame mean
            of all joints is used instead.
    Returns:
        (F, J, 2) root-centred.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if root_joint is not None and root_joint >= 0:
        centre = pose[:, root_joint:root_joint + 1, :]
    else:
        centre = pose.mean(axis=1, keepdims=True)
    return pose - centre


def torso_length(pose: np.ndarray, torso_joints: tuple[int, int] = (0, 1),
                 eps: float = 1e-6) -> float:
    """Median torso length over a clip (shoulder-mid to hip-mid distance).

    Falls back to the clip's mean joint spread when ``torso_joints`` are
    invalid, so scale normalisation still has a sensible denominator.
    """
    pose = np.asarray(pose, dtype=np.float64)
    a, b = torso_joints
    if a is not None and b is not None and a >= 0 and b >= 0:
        d = np.linalg.norm(pose[:, a, :] - pose[:, b, :], axis=-1)  # (F,)
        length = float(np.median(d))
    else:
        # Mean per-frame RMS distance of joints from their centroid.
        centred = pose - pose.mean(axis=1, keepdims=True)
        length = float(np.median(np.linalg.norm(centred, axis=-1)))
    return max(length, eps)


def scale_normalize(pose: np.ndarray, torso_joints: tuple[int, int] = (0, 1)
                    ) -> np.ndarray:
    """Divide the whole clip by its torso length so coords are unitless ([§2.2])."""
    length = torso_length(pose, torso_joints)
    return np.asarray(pose, dtype=np.float64) / length


def preprocess_clip(pose: np.ndarray, root_joint: int = 0,
                    torso_joints: tuple[int, int] = (0, 1)) -> np.ndarray:
    """Root-centre then scale-normalise one sequence ([plan §2.1-§2.2]).

    No mirroring, no resampling. Returns float32 (F, J, 2).
    """
    pose = root_center(pose, root_joint=root_joint)
    pose = scale_normalize(pose, torso_joints=torso_joints)
    return pose.astype(np.float32)


def window_sequence(pose: np.ndarray, clip_length: int = 60,
                    stride: int = 30) -> np.ndarray:
    """Cut a preprocessed sequence into fixed windows ([plan §2.4]).

    Sequences shorter than one window yield an empty array (dropped, never
    padded — matches the neonate/CARE-PD convention).

    Args:
        pose: (F, J, 2).
        clip_length: T (60).
        stride: hop between window starts (30 = 50% overlap).
    Returns:
        (K, T, J, 2).
    """
    F = pose.shape[0]
    starts = list(range(0, max(F - clip_length + 1, 0), stride))
    if not starts:
        return np.empty((0, clip_length) + pose.shape[1:], dtype=pose.dtype)
    return np.stack([pose[s:s + clip_length] for s in starts])


def flatten_spatial(clips: np.ndarray) -> np.ndarray:
    """(N, T, J, 2) -> (N, T, J*2) for the encoder input ([plan §2.5])."""
    clips = np.asarray(clips)
    n, t = clips.shape[0], clips.shape[1]
    return clips.reshape(n, t, -1)


# ---- Assembling a windowed dataset ----------------------------------------

@dataclass
class Sequence:
    """One raw sequence with its labels ([plan §2]).

    Attributes:
        pose: (F, J, 2) raw 2D keypoints.
        c_p: primary discrete pathology label (int in [0, n_classes)).
        subject: subject id, for subject-level splits ([plan §5]).
        c_nuis: optional nuisance categorical (recording session, camera …);
            ``None`` if unused.
    """
    pose: np.ndarray
    c_p: int
    subject: str
    c_nuis: int | None = None


@dataclass
class WindowedData:
    """Windowed clips + per-clip labels, aligned by index.

    ``x`` is (N, T, J*2) float32. ``c_p`` / ``c_nuis`` are (N,) int arrays
    (c_nuis is all -1 when unused). ``subject`` is an (N,) object array so
    splits can group by subject.
    """
    x: np.ndarray
    c_p: np.ndarray
    c_nuis: np.ndarray
    subject: np.ndarray
    n_joints: int

    @property
    def n(self) -> int:
        return len(self.x)


def build_windowed_data(sequences: list[Sequence], config) -> WindowedData:
    """Preprocess + window a list of :class:`Sequence` into :class:`WindowedData`.

    Each sequence is root-centred, scale-normalised, cut into 60-frame
    windows, and flattened to ``(T, J*2)``; its scalar labels are broadcast
    to every window it produced.
    """
    xs, cps, nuis, subs = [], [], [], []
    for seq in sequences:
        pose = preprocess_clip(seq.pose, config.root_joint, config.torso_joints)
        windows = window_sequence(pose, config.clip_length, config.stride)
        if len(windows) == 0:
            continue
        flat = flatten_spatial(windows)               # (K, T, J*2)
        k = len(flat)
        xs.append(flat.astype(np.float32))
        cps.append(np.full(k, int(seq.c_p), dtype=np.int64))
        nuis.append(np.full(k, -1 if seq.c_nuis is None else int(seq.c_nuis),
                            dtype=np.int64))
        subs.append(np.array([str(seq.subject)] * k, dtype=object))
    if not xs:
        raise ValueError("No clips built — are all sequences shorter than "
                         f"clip_length ({config.clip_length})?")
    return WindowedData(
        x=np.concatenate(xs), c_p=np.concatenate(cps),
        c_nuis=np.concatenate(nuis), subject=np.concatenate(subs),
        n_joints=config.n_joints)


# ---- Torch dataset ---------------------------------------------------------

class NeonateKeypointDataset:
    """A torch ``Dataset`` over windowed clips ([plan §2]).

    ``__getitem__`` returns ``(x, c_p, c_nuis, subject)``:
        x        float32 tensor (T, J*2)
        c_p      int64 tensor scalar
        c_nuis   int64 tensor scalar (-1 when unused)
        subject  str (kept as a Python str; default collate lists them)
    """

    def __init__(self, data: WindowedData):
        import torch
        self.torch = torch
        self.data = data

    def __len__(self) -> int:
        return self.data.n

    def __getitem__(self, i: int):
        t = self.torch
        return (t.from_numpy(self.data.x[i]),
                t.tensor(int(self.data.c_p[i]), dtype=t.long),
                t.tensor(int(self.data.c_nuis[i]), dtype=t.long),
                str(self.data.subject[i]))


def subject_split(data: WindowedData, val_fraction: float = 0.2,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split clip indices by **subject**, never by clip ([plan §5, §9]).

    Holds out whole subjects (stratified is not attempted here; the caller
    can stratify by passing pre-chosen subjects). Returns boolean
    ``(train_mask, val_mask)`` over clips.
    """
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(set(data.subject.tolist())), dtype=object)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    val_subj = set(subjects[:n_val].tolist())
    val_mask = np.array([s in val_subj for s in data.subject])
    return ~val_mask, val_mask


def make_loader(data: WindowedData, indices: np.ndarray | None,
                batch_size: int, shuffle: bool, seed: int = 0):
    """Build a torch ``DataLoader`` over a subset of clips."""
    import torch
    from torch.utils.data import DataLoader, Subset
    ds = NeonateKeypointDataset(data)
    if indices is not None:
        ds = Subset(ds, np.asarray(indices).tolist())
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=g)
