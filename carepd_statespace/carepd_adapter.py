"""CARE-PD Tier-1 adapter for the state-space pipeline ([guideline §2]).

Turns the three Tier-1 cohort ``.pkl`` files (BMCLab, KUL-DT-T, E-LC) into
the contract the reference pipeline consumes: a list of variable-length
per-walk feature arrays ``(T_i, d)`` plus an aligned ``info`` dataframe.
This is where most of the work — and most of the harm if done wrong — lives.

Key adaptations from the infant original ([guideline appendix]):
- 3D joints (H36M 17), not 2D keypoints;
- **egocentric normalisation**: root-centre + *per-frame* heading alignment
  (per-frame, because KUL-DT-T contains turns — a per-walk rotation would
  inject the turn as huge spurious variance);
- **keep global motion as channels** (Set B): root planar velocity, root
  angular velocity, root height — the HumanML3D decomposition — so gait
  speed / stride / arm-swing (central PD signals) are not thrown away;
- gentle band-pass, 30 Hz -> **15 Hz** so the AR term spans a real fraction
  of a step; do not high-pass away the ~1-2 Hz cadence;
- walks stay **variable-length** (a list), never padded or concatenated.

The joint extraction assumes the CARE-PD H36M 17-joint 3D asset (or joints
regressed from SMPL with ``beta=0``). Point ``load_cohort_pkls`` at a
``pose_to_joints`` callable if your ``.pkl`` stores raw SMPL parameters.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---- H36M 17-joint conventions ([guideline §2.1]) -------------------------

H36M_PELVIS = 0
H36M_L_HIP = 4
H36M_R_HIP = 1
UP_AXIS = 1                 # y-up (matches the CARE-PD floorXZZplus release)
GROUND_AXES = (0, 2)       # the two horizontal axes

# Body-region groups for the Figure-3b breakdown ([guideline §7.2]).
H36M_REGIONS = {
    "legs": [1, 2, 3, 4, 5, 6],       # hips, knees, ankles/feet
    "trunk": [0, 7, 8],               # pelvis, spine, thorax
    "head": [9, 10],                  # neck, head
    "arms": [11, 12, 13, 14, 15, 16],  # shoulders, elbows, wrists
}


@dataclass
class Walk:
    """One walk's 3D joints + global trajectory + labels ([guideline §2.5])."""
    joints: np.ndarray          # (F, 17, 3) 3D joint positions
    cohort: str                 # BMCLab | KUL-DT-T | E-LC
    subject_id: str
    walk_id: str
    fps: float
    fog: int | float = np.nan            # 0/1
    medication: object = np.nan          # "on"/"off"/NaN
    updrs_gait: float = np.nan           # 0-3/NaN
    sex: object = np.nan
    trans: np.ndarray | None = None      # (F, 3) global translation, if present


# ---- Egocentric normalisation ([guideline §2.2]) --------------------------

def _heading_angle(joints: np.ndarray) -> np.ndarray:
    """Per-frame walking-heading angle (radians) from the hip vector.

    Forward is the ground-plane direction perpendicular to the L->R hip
    vector. Returns (F,) angle of the forward axis w.r.t. ground axis 0.
    """
    a0, a1 = GROUND_AXES
    hip = joints[:, H36M_R_HIP] - joints[:, H36M_L_HIP]     # (F, 3)
    # Forward = hip rotated 90 deg in the ground plane.
    fwd0, fwd1 = -hip[:, a1], hip[:, a0]
    return np.arctan2(fwd1, fwd0)


def egocentric(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Root-centre + per-frame heading align ([guideline §2.2]).

    Returns ``(ego_joints (F,17,3), heading (F,))``. Each frame is rotated
    about the vertical axis so the walking direction is a canonical ground
    axis, so global heading (and KUL-DT-T turns) leave the articulated
    features — retained separately as root angular velocity in Set B.
    """
    joints = np.asarray(joints, dtype=np.float64)
    a0, a1 = GROUND_AXES
    centred = joints - joints[:, H36M_PELVIS:H36M_PELVIS + 1, :]
    heading = _heading_angle(joints)
    cos, sin = np.cos(-heading), np.sin(-heading)          # rotate by -heading
    out = centred.copy()
    x, z = centred[..., a0], centred[..., a1]
    out[..., a0] = cos[:, None] * x - sin[:, None] * z
    out[..., a1] = sin[:, None] * x + cos[:, None] * z
    return out, heading


def root_channels(joints: np.ndarray, heading: np.ndarray, fps: float,
                  trans: np.ndarray | None = None) -> np.ndarray:
    """Set-B global channels: root planar vel (2), angular vel (1), height (1).

    The HumanML3D decomposition ([guideline §2.3]). Planar velocity is the
    pelvis ground-plane displacement rotated into the current heading frame;
    angular velocity is the heading change rate; height is the pelvis
    vertical coordinate. Uses ``trans`` for the global path when present,
    else the pelvis joint.
    """
    a0, a1 = GROUND_AXES
    root = trans if trans is not None else joints[:, H36M_PELVIS, :]
    root = np.asarray(root, dtype=np.float64)
    F = len(root)
    dpos = np.zeros((F, 2))
    dpos[1:, 0] = np.diff(root[:, a0])
    dpos[1:, 1] = np.diff(root[:, a1])
    # Rotate displacement into the heading frame so "forward" is consistent.
    cos, sin = np.cos(-heading), np.sin(-heading)
    lin = np.stack([cos * dpos[:, 0] - sin * dpos[:, 1],
                    sin * dpos[:, 0] + cos * dpos[:, 1]], axis=1) * fps
    ang = np.zeros(F)
    ang[1:] = np.unwrap(np.diff(heading)) * fps
    height = root[:, UP_AXIS]
    return np.concatenate([lin, ang[:, None], height[:, None]], axis=1)  # (F,4)


def walk_features(walk: Walk, feature_set: str = "B") -> np.ndarray:
    """Build the per-frame feature array for one walk ([guideline §2.3]).

    Set A: egocentric joints only (17*3 = 51). Set B (default): Set A plus
    the four root channels (55).
    """
    ego, heading = egocentric(walk.joints)
    A = ego.reshape(len(ego), -1)                          # (F, 51)
    if feature_set.upper() == "A":
        return A
    root = root_channels(walk.joints, heading, walk.fps, walk.trans)
    return np.concatenate([A, root], axis=1)               # (F, 55)


# ---- Filter + resample ([guideline §2.4], adapts reference process_data) ---

def process_walk(x: np.ndarray, src_fps: float, dst_fps: float = 15.0,
                 hp: float = 0.01, do_filter: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass + MAD-outlier + resample one walk to ``dst_fps``.

    Mirrors the reference ``process_data`` (4th-order zero-phase Butterworth,
    MAD flagging at 3*MAD*1.4826, outlier-aware interpolation) but per-walk
    and variable-length. The low-pass sits below the new Nyquist and the
    high-pass (0.01 Hz) only kills drift — the cadence fundamental is kept
    ([guideline §2.4]).
    """
    from scipy.signal import butter, sosfiltfilt
    from scipy.interpolate import interp1d
    x = np.asarray(x, dtype=np.float64)
    F, d = x.shape
    if do_filter and F > 15:
        lp = min(dst_fps / 2 - 1e-6, src_fps / 2 - 1e-6)
        sos = butter(4, (hp, lp), "bandpass", fs=src_fps, output="sos")
        x = sosfiltfilt(sos, x, axis=0)
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    outliers = np.abs(x - med) > (3 * mad * 1.4826 + 1e-9)
    if abs(src_fps - dst_fps) < 1e-9:
        return np.nan_to_num(x), outliers
    new_F = max(int(round(F * dst_fps / src_fps)), 2)
    out = np.zeros((new_F, d))
    grid_old = np.arange(F)
    grid_new = np.linspace(0, F - 1, new_F)
    for f in range(d):
        keep = ~outliers[:, f]
        if keep.sum() < 2:
            keep = np.ones(F, bool)
        fi = interp1d(grid_old[keep], x[keep, f], kind="slinear",
                      bounds_error=False, fill_value=(x[keep, f][0], x[keep, f][-1]))
        out[:, f] = fi(grid_new)
    return np.nan_to_num(out), outliers


# ---- Assembling the dataset ([guideline §2.5-§2.7]) -----------------------

@dataclass
class StateSpaceData:
    """The pipeline contract: per-walk feature list + aligned info dataframe."""
    features: list                      # list of (T_i, d) arrays
    info: pd.DataFrame                  # one row per walk ([guideline §2.5])
    feature_set: str = "B"
    fps: float = 15.0
    meta: dict = field(default_factory=dict)

    @property
    def n_walks(self) -> int:
        return len(self.features)

    @property
    def d(self) -> int:
        return self.features[0].shape[1] if self.features else 0


def build_dataset(walks: list[Walk], feature_set: str = "B",
                  dst_fps: float = 15.0, do_filter: bool = True
                  ) -> StateSpaceData:
    """Turn a list of :class:`Walk` into :class:`StateSpaceData` ([guideline §2])."""
    feats, rows = [], []
    for w in walks:
        raw = walk_features(w, feature_set)
        proc, _ = process_walk(raw, w.fps, dst_fps, do_filter=do_filter)
        if len(proc) < 4:
            continue
        feats.append(proc.astype(np.float64))
        rows.append({
            "recording": f"{w.cohort}:{w.subject_id}:{w.walk_id}",
            "subject_id": f"{w.cohort}:{w.subject_id}", "cohort": w.cohort,
            "fog": w.fog, "medication": w.medication,
            "updrs_gait": w.updrs_gait, "sex": w.sex, "n_frames": len(proc)})
    info = pd.DataFrame(rows).reset_index(drop=True)
    return StateSpaceData(features=feats, info=info, feature_set=feature_set,
                          fps=dst_fps)


# ---- Loading the CARE-PD .pkl files ([guideline §1, §2.1]) ----------------

_MED_MAP = {1: "on", 0: "off", "on": "on", "off": "off",
            "ON": "on", "OFF": "off"}


def load_cohort_pkls(paths: dict[str, str], pose_to_joints=None,
                     joints_key: str = "joints3d") -> list[Walk]:
    """Read the Tier-1 cohort ``.pkl`` files into :class:`Walk` records.

    ``paths`` maps cohort name -> ``.pkl`` path. Each ``.pkl`` is keyed
    ``subject_id -> walk_id -> {pose, trans, beta, fps, UPDRS_GAIT,
    medication, other}`` ([guideline §1]). Joint positions are read from
    ``joints_key`` if present; otherwise ``pose_to_joints(record)`` must
    return ``(F, 17, 3)`` (wire in the SMPL->H36M regressor with beta=0).
    ``other`` carries the FoG label.
    """
    walks: list[Walk] = []
    for cohort, path in paths.items():
        with open(path, "rb") as f:
            blob = pickle.load(f)
        for subject_id, walk_map in blob.items():
            for walk_id, rec in walk_map.items():
                joints = _extract_joints(rec, pose_to_joints, joints_key)
                if joints is None:
                    continue
                walks.append(Walk(
                    joints=joints, cohort=cohort, subject_id=str(subject_id),
                    walk_id=str(walk_id), fps=float(rec.get("fps", 30.0)),
                    fog=_extract_fog(rec), medication=_extract_med(rec),
                    updrs_gait=_num(rec.get("UPDRS_GAIT")),
                    sex=rec.get("sex", rec.get("other", {}).get("sex", np.nan)
                        if isinstance(rec.get("other"), dict) else np.nan),
                    trans=np.asarray(rec["trans"], float)
                    if "trans" in rec else None))
    return walks


def _extract_joints(rec, pose_to_joints, joints_key):
    if isinstance(rec, dict) and joints_key in rec:
        j = np.asarray(rec[joints_key], dtype=np.float64)
    elif pose_to_joints is not None:
        j = np.asarray(pose_to_joints(rec), dtype=np.float64)
    else:
        return None
    if j.ndim == 2 and j.shape[1] % 3 == 0:
        j = j.reshape(j.shape[0], -1, 3)
    return j if j.ndim == 3 and j.shape[-1] == 3 else None


def _extract_fog(rec):
    other = rec.get("other") if isinstance(rec, dict) else None
    for src in (rec, other if isinstance(other, dict) else {}):
        for k in ("fog", "FoG", "freezing", "freezer"):
            if k in src and src[k] is not None:
                return int(bool(_truthy(src[k])))
    return np.nan


def _extract_med(rec):
    v = rec.get("medication") if isinstance(rec, dict) else None
    return _MED_MAP.get(v, np.nan if v is None else v)


def _truthy(v):
    if isinstance(v, str):
        return v.strip().lower() in ("1", "yes", "true", "fog", "freezer", "on")
    try:
        return float(v) > 0.5
    except (TypeError, ValueError):
        return False


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan
