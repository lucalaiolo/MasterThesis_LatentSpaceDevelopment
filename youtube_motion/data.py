"""Load the YouTube 2D-keypoint CSV into the clip iterator the VAEs expect.

The dataset is stored **long** — one row per joint per frame:

    video_number,video,bp,frame,x,y,fps,pixel_x,pixel_y,time,part_idx
    0,video_000000,RShoulder,0,-0.4135,1.3078,29.97,1280,720,0.0,2.0
    0,video_000000,RElbow,0,-0.2862,0.7706,29.97,1280,720,0.0,3.0
    ...

``load_youtube_csv`` pivots this into the same shape the ``architectures``
training loop consumes elsewhere: a list of per-video arrays, each
``(F_v, J, 2)`` — F_v frames of J joints in the image plane. The ``x``/``y``
columns are already unitless (torso-normalised), so no preprocessing is
applied by default; pass ``preprocess=...`` to root-centre / scale if your
export holds raw pixels instead.

Only numpy and the standard library are needed — no pandas. The parser makes
a single streaming pass over the file, so a multi-hundred-MB export loads
without blowing up memory beyond the assembled arrays.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from .skeleton import (COCO18_LIMBS, N_DIMS, N_JOINTS, ROOT_JOINT,
                       TORSO_JOINTS, coco18_limbs, skeleton_for)


# ---- Preprocessing (pure NumPy, generic in J) -----------------------------

def root_center(pose: np.ndarray, root_joint: int = ROOT_JOINT) -> np.ndarray:
    """Subtract one joint from every joint, per frame.

    Args:
        pose: (F, J, 2).
        root_joint: joint index to place at the origin; if < 0, the per-frame
            centroid of all joints is used instead.
    Returns:
        (F, J, 2) centred.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if root_joint is not None and root_joint >= 0:
        centre = pose[:, root_joint:root_joint + 1, :]
    else:
        centre = np.nanmean(pose, axis=1, keepdims=True)
    return pose - centre


def torso_scale(pose: np.ndarray, torso_joints: tuple[int, int] = TORSO_JOINTS,
                eps: float = 1e-6) -> float:
    """Median torso length over a clip, a stable per-clip scale.

    Falls back to the median joint spread when ``torso_joints`` are invalid,
    so scale normalisation always has a sensible, non-zero denominator.
    """
    pose = np.asarray(pose, dtype=np.float64)
    a, b = torso_joints
    if a is not None and b is not None and a >= 0 and b >= 0:
        d = np.linalg.norm(pose[:, a, :] - pose[:, b, :], axis=-1)   # (F,)
        length = float(np.nanmedian(d))
    else:
        centred = pose - np.nanmean(pose, axis=1, keepdims=True)
        length = float(np.nanmedian(np.linalg.norm(centred, axis=-1)))
    if not np.isfinite(length) or length <= 0:
        return eps
    return max(length, eps)


def preprocess_video(pose: np.ndarray, mode: str = "none",
                     root_joint: int = ROOT_JOINT,
                     torso_joints: tuple[int, int] = TORSO_JOINTS
                     ) -> np.ndarray:
    """Apply the requested normalisation to one (F, J, 2) sequence.

    Modes:
        "none"          use the coordinates as stored (default — the CSV is
                        already torso-normalised).
        "center"        root-centre on ``root_joint`` each frame.
        "center_scale"  root-centre, then divide the whole clip by its median
                        torso length so coordinates are unitless. Use this if
                        the export holds raw pixel coordinates.
    """
    if mode == "none":
        return np.asarray(pose, dtype=np.float32)
    if mode == "center":
        return root_center(pose, root_joint).astype(np.float32)
    if mode == "center_scale":
        centred = root_center(pose, root_joint)
        return (centred / torso_scale(centred, torso_joints)).astype(np.float32)
    raise ValueError(
        f"unknown preprocess mode {mode!r}; use 'none', 'center', "
        f"or 'center_scale'."
    )


# ---- Missing-detection handling -------------------------------------------

def interpolate_missing(pose: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along time, per joint and coordinate.

    2D pose detectors drop joints on occlusion; those land as NaN rows here.
    Each (joint, coord) series is linearly interpolated over interior gaps and
    held constant past the first/last seen value. A joint never seen in the
    whole clip is filled with 0 (the origin — a neutral, masked-like value).

    Args:
        pose: (F, J, 2), NaN for missing.
    Returns:
        (F, J, 2) with no NaNs.
    """
    pose = np.array(pose, dtype=np.float64, copy=True)
    F, J, D = pose.shape
    t = np.arange(F)
    for j in range(J):
        for d in range(D):
            col = pose[:, j, d]
            good = np.isfinite(col)
            n_good = int(good.sum())
            if n_good == F:
                continue
            if n_good == 0:
                pose[:, j, d] = 0.0
                continue
            # np.interp holds endpoints flat beyond the observed range.
            pose[:, j, d] = np.interp(t, t[good], col[good])
    return pose.astype(np.float32)


# ---- The bundle -----------------------------------------------------------

@dataclass
class YoutubeMotionBundle:
    """Everything the training driver needs, assembled from the CSV.

    Attributes:
        videos: list of per-video arrays, each (F_v, J, 2) float32.
        video_names: the ``video`` id string for each entry in ``videos``.
        fps: per-video recording rate (median of the frame times), or None.
        n_joints: J (15 for BODY-15, 18 for COCO-18).
        n_dims: coordinate dimension (2).
        limbs: limb-name -> joint-index map for the ``limb`` mask policy.
        bones: (a, b) joint-index pairs for stick-figure visualisation.
        left_right: bilateral (left, right) joint pairs for symmetry analyses.
    """
    videos: list[np.ndarray]
    video_names: list[str]
    fps: list[float]
    n_joints: int = N_JOINTS
    n_dims: int = N_DIMS
    limbs: dict[str, list[int]] = field(default_factory=coco18_limbs)
    bones: list[tuple[int, int]] = field(default_factory=list)
    left_right: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_videos(self) -> int:
        return len(self.videos)

    @property
    def total_frames(self) -> int:
        return int(sum(v.shape[0] for v in self.videos))

    def summary(self) -> str:
        """One-line human summary of the loaded set."""
        if not self.videos:
            return "empty bundle"
        lengths = [v.shape[0] for v in self.videos]
        fps_vals = [f for f in self.fps if f]
        fps_txt = (f"{np.median(fps_vals):.2f}" if fps_vals else "n/a")
        return (f"{self.n_videos} videos, {self.total_frames} frames total, "
                f"J={self.n_joints}, D={self.n_dims}, "
                f"len[min/med/max]={min(lengths)}/{int(np.median(lengths))}/"
                f"{max(lengths)}, fps~{fps_txt}")


# ---- CSV -> bundle --------------------------------------------------------

# Default column names, matching the export shown in the dataset image.
DEFAULT_COLUMNS = {
    "video": "video",
    "frame": "frame",
    "x": "x",
    "y": "y",
    "part_idx": "part_idx",
    "fps": "fps",
}


def load_youtube_csv(path: str,
                     n_joints: int | None = None,
                     preprocess: str = "none",
                     min_frames: int = 1,
                     max_videos: int | None = None,
                     columns: dict[str, str] | None = None,
                     limbs: dict[str, list[int]] | None = None,
                     ) -> YoutubeMotionBundle:
    """Read the long-format keypoint CSV into a :class:`YoutubeMotionBundle`.

    One streaming pass groups rows by ``video``; each group is pivoted into a
    dense ``(F, J, 2)`` array indexed by ``frame`` and ``part_idx``. Cells with
    no row (dropped detections, or a wholly missing frame) start as NaN and are
    filled by :func:`interpolate_missing`. Frame indices are shifted so the
    earliest frame of each video maps to 0, preserving gaps (and thus timing).

    The joint count is **auto-detected** by default (``n_joints=None``): J is
    set to ``max(part_idx) + 1`` seen in the file, then the matching skeleton
    (BODY-15 or COCO-18) fixes the limbs, bones, left/right pairs, and the
    root / torso joints used by ``center`` / ``center_scale``. Pass an explicit
    ``n_joints`` to force a layout (rows with ``part_idx >= n_joints`` are then
    dropped).

    Args:
        path: path to the CSV (with a header row).
        n_joints: J, or None to auto-detect from the data (recommended).
        preprocess: "none" (default), "center", or "center_scale"
            (:func:`preprocess_video`). Root / torso joints come from the
            resolved skeleton.
        min_frames: drop videos shorter than this many frames (too short to
            window). Must be >= 1.
        max_videos: keep at most this many videos (handy for a quick look);
            None keeps all.
        columns: override the CSV column names; keys are the logical fields
            ``video, frame, x, y, part_idx, fps`` (see ``DEFAULT_COLUMNS``).
        limbs: limb map for the bundle; defaults to the resolved skeleton's.
    Returns:
        A populated :class:`YoutubeMotionBundle`.
    """
    cols = {**DEFAULT_COLUMNS, **(columns or {})}

    # video id -> {"rows": [(frame, part_idx, x, y)], "fps": [..], order: int}
    grouped: dict[str, dict] = {}
    order: list[str] = []
    max_pi = -1                                    # for auto-detecting J

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in (cols["video"], cols["frame"], cols["x"],
                               cols["y"], cols["part_idx"]) if c not in
                   (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV {path!r} is missing expected column(s) {missing}; found "
                f"{reader.fieldnames}. Pass `columns=` to map custom names."
            )
        have_fps = cols["fps"] in (reader.fieldnames or [])

        for row in reader:
            vid = row[cols["video"]]
            if vid not in grouped:
                if max_videos is not None and len(order) >= max_videos:
                    continue                       # already at the cap
                grouped[vid] = {"rows": [], "fps": []}
                order.append(vid)
            g = grouped[vid]
            try:
                pi = int(float(row[cols["part_idx"]]))
                fr = int(float(row[cols["frame"]]))
            except (TypeError, ValueError):
                continue                           # unparseable index -> skip
            if pi < 0:
                continue
            # When J is fixed, drop out-of-range joints here; when auto-
            # detecting, keep every row and learn J from the observed maximum.
            if n_joints is not None and pi >= n_joints:
                continue
            max_pi = max(max_pi, pi)
            x = _to_float(row[cols["x"]])
            y = _to_float(row[cols["y"]])
            g["rows"].append((fr, pi, x, y))
            if have_fps:
                f = _to_float(row[cols["fps"]])
                if np.isfinite(f):
                    g["fps"].append(f)

    if max_pi < 0:
        raise ValueError(
            f"no usable keypoint rows parsed from {path!r}. Check the column "
            f"mapping ({cols}).")
    J = n_joints if n_joints is not None else max_pi + 1
    skel = skeleton_for(J)                          # raises on an unknown J

    videos: list[np.ndarray] = []
    names: list[str] = []
    fps_out: list[float] = []
    for vid in order:
        rows = grouped[vid]["rows"]
        if not rows:
            continue
        frames = np.fromiter((r[0] for r in rows), dtype=np.int64)
        f_min, f_max = int(frames.min()), int(frames.max())
        F = f_max - f_min + 1
        if F < min_frames:
            continue
        pose = np.full((F, J, N_DIMS), np.nan, dtype=np.float64)
        for fr, pi, x, y in rows:
            if pi >= J:
                continue                            # explicit J smaller than data
            pose[fr - f_min, pi, 0] = x
            pose[fr - f_min, pi, 1] = y
        pose = interpolate_missing(pose)
        pose = preprocess_video(pose, mode=preprocess,
                                root_joint=skel["root_joint"],
                                torso_joints=skel["torso_joints"])
        videos.append(pose)
        names.append(vid)
        fv = grouped[vid]["fps"]
        fps_out.append(float(np.median(fv)) if fv else 0.0)

    if not videos:
        raise ValueError(
            f"no usable videos parsed from {path!r} (min_frames={min_frames}). "
            f"Check the column mapping and that part_idx is in [0, {J})."
        )

    return YoutubeMotionBundle(
        videos=videos, video_names=names, fps=fps_out,
        n_joints=J, n_dims=N_DIMS,
        limbs=(limbs if limbs is not None else skel["limbs"]),
        bones=skel["bones"], left_right=skel["left_right"],
    )


def _to_float(s) -> float:
    """Parse a CSV cell to float; empty / 'nan' / bad values become NaN."""
    if s is None:
        return float("nan")
    s = str(s).strip()
    if not s or s.lower() in ("nan", "na", "none", "null"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")
