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

Loading the real CARE-PD h36m release ([guideline §1]): joints and labels
travel in **two separate files** (see ``architectures/care_pd.py`` for the
full account). :func:`load_h36m_cohorts` is the loader for it — it reads the
3D joints from the per-cohort ``h36m_3d_world_*.npz`` and joins the clinical
labels from the source SMPL ``.pkl`` by walk id::

    walks = load_h36m_cohorts("/content/drive/MyDrive/CARE-PD_h36m",
                              source_dir="/content/assets/datasets/TWIKMK")

The older :func:`load_cohort_pkls` is kept for hand-built pickles that
*already* carry joints under ``joints_key`` (or a ``pose_to_joints`` SMPL->
H36M regressor); it is **not** the path for the h36m release, whose ``.pkl``
holds only SMPL parameters and labels — the 3D joints are in the ``.npz``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

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


# ---- Loading the CARE-PD data ([guideline §1, §2.1]) ----------------------

_MED_MAP = {1: "on", 0: "off", "on": "on", "off": "off",
            "ON": "on", "OFF": "off"}

# The three Tier-1 mocap cohorts, in their canonical order.
TIER1_COHORTS: tuple[str, ...] = ("BMCLab", "KUL-DT-T", "E-LC")

# World-coordinate 3D variant inside each cohort's h36m directory. The
# ``world2cam*`` / ``world2cam2img*`` siblings are view-projected (a nuisance
# view axis, or dropped depth) and must be skipped — only the floor-aligned
# world file is loaded.
WORLD_NPZ_GLOB = "h36m_3d_world_*.npz"


def _import_care_pd_reader():
    """Import the sibling h36m-release reader, with a pointed error if absent.

    ``architectures.care_pd`` is the tested source of truth for the on-disk
    CARE-PD h36m layout (flat / nested / object-array ``.npz``, world-variant
    globbing, double-keyed label join). It is NumPy-only — no torch — so this
    import stays light.
    """
    try:
        from architectures import care_pd
    except Exception as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "load_h36m_cohorts needs the sibling `architectures.care_pd` "
            "module (the CARE-PD h36m release reader). Run from the "
            "repository root so `architectures/` is importable "
            f"(original import error: {exc!r})."
        ) from exc
    return care_pd


def _resolve_h36m_npz(h36m_root, cohort: str, world_glob: str):
    """Find one cohort's world-3D ``.npz`` under an ``h36m/`` release tree.

    Handles the documented ``<root>/<cohort>/<variant>.npz`` layout and a
    flat ``<root>/<cohort>.npz`` fallback; prefers the world variant, then
    any ``.npz`` in the cohort directory.
    """
    root = Path(h36m_root)
    cohort_dir = root / cohort
    if cohort_dir.is_dir():
        for pattern in (world_glob, "*.npz"):
            matches = sorted(cohort_dir.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"cohort {cohort!r}: no {world_glob!r} (nor any .npz) in "
            f"{cohort_dir}.")
    for ext in (".npz", ".pkl"):
        flat = root / f"{cohort}{ext}"
        if flat.exists():
            return flat
    raise FileNotFoundError(
        f"cohort {cohort!r}: neither {cohort_dir}/ nor "
        f"{root / (cohort + '.npz')} exists under {root}.")


def load_h36m_cohorts(h36m_root=None, cohorts: tuple[str, ...] = TIER1_COHORTS,
                      source_dir=None, *, npz_paths: dict | None = None,
                      pkl_paths: dict | None = None, fps: float = 30.0,
                      world_glob: str = WORLD_NPZ_GLOB) -> list["Walk"]:
    """Load the CARE-PD h36m release: 3D joints (``.npz``) + labels (``.pkl``).

    This is the loader for the **real** CARE-PD h36m release, where joints and
    labels live in separate files ([guideline §1]). For the layout the Colab
    notebook produces::

        <h36m_root>/BMCLab/h36m_3d_world_floorXZZplus_30f_or_longer.npz   # joints
        <source_dir>/BMCLab.pkl                                          # labels

    it is simply::

        walks = load_h36m_cohorts(
            "/content/drive/MyDrive/CARE-PD_h36m",
            source_dir="/content/assets/datasets/TWIKMK")
        data  = build_dataset(walks, feature_set="B")

    Args:
        h36m_root: directory holding ``<cohort>/h36m_3d_world_*.npz`` (or a
            flat ``<cohort>.npz``). May be ``None`` if ``npz_paths`` names
            every cohort's joints file explicitly.
        cohorts: cohort names to load (default: the three Tier-1 mocap
            cohorts, :data:`TIER1_COHORTS`).
        source_dir: directory holding the per-cohort source ``.pkl``
            (``<source_dir>/<cohort>.pkl``) that carries the labels. Optional
            — without it the walks have no clinical labels, which is fine for
            the principal-movement basis and ARHMM fit (only the §6 clinical
            analysis needs them). A missing per-cohort ``.pkl`` is skipped.
        npz_paths: optional ``{cohort: npz_path}`` overriding the
            ``h36m_root`` lookup for those cohorts.
        pkl_paths: optional ``{cohort: pkl_path}`` overriding the
            ``source_dir`` lookup for those cohorts.
        fps: frame rate of the h36m joints. The ``floorXZZplus`` release is
            standardised to 30 fps, so the default is 30; override only for a
            non-standard export. (The SMPL ``.pkl``'s own fps is deliberately
            ignored — it is the *joints*, not the pkl SMPL poses, that
            :func:`build_dataset` resamples downstream.)
        world_glob: filename pattern selecting the world-coordinate 3D file
            inside each cohort directory.

    Returns:
        A list of :class:`Walk` with raw ``(F, 17, 3)`` joints (no
        preprocessing — :func:`build_dataset` runs the egocentric norm, root
        channels, band-pass and resample) and labels attached. ``trans`` is
        left ``None`` on purpose: the ``floorXZZplus`` pelvis joint already
        carries each walk's global path in the joint coordinate frame, whereas
        the SMPL ``.pkl`` translation lives in a *different* frame — mixing
        them would corrupt the Set-B root-velocity channels.
    """
    care_pd = _import_care_pd_reader()
    npz_paths = dict(npz_paths or {})
    pkl_paths = dict(pkl_paths or {})
    source_dir = Path(source_dir) if source_dir is not None else None

    walks: list[Walk] = []
    for cohort in cohorts:
        if cohort in npz_paths:
            npz = Path(npz_paths[cohort])
        elif h36m_root is not None:
            npz = _resolve_h36m_npz(h36m_root, cohort, world_glob)
        else:
            raise ValueError(
                f"cohort {cohort!r}: pass h36m_root or npz_paths[{cohort!r}].")
        pkl = None
        if cohort in pkl_paths:
            pkl = Path(pkl_paths[cohort])
        elif source_dir is not None:
            candidate = source_dir / f"{cohort}.pkl"
            if candidate.exists():
                pkl = candidate
        # preprocess=False: keep raw joints; our egocentric norm (per-frame
        # heading, retained root channels) differs from care_pd's per-walk
        # align, so we do NOT want its preprocessing here.
        arch_walks = care_pd.load_cohort(npz, cohort, source_pkl=pkl,
                                         preprocess=False)
        walks.extend(_walk_from_care_pd(aw, fps) for aw in arch_walks)
    return walks


def _walk_from_care_pd(aw, fps: float) -> "Walk":
    """Convert an ``architectures.care_pd.Walk`` into our :class:`Walk`.

    Maps the flat ``labels`` dict onto our typed clinical fields (FoG,
    medication, UPDRS-gait, sex) using the same tolerant key matching as the
    ``.pkl`` path, and drops ``trans`` (see :func:`load_h36m_cohorts`).
    """
    lab = aw.labels or {}
    updrs = lab.get("UPDRS_GAIT")
    if updrs is None:
        updrs = lab.get("updrs_gait")
    return Walk(
        joints=np.asarray(aw.pose, dtype=np.float64),
        cohort=aw.cohort, subject_id=str(aw.subject), walk_id=str(aw.walk_id),
        fps=float(fps), fog=_extract_fog(lab), medication=_extract_med(lab),
        updrs_gait=_num(updrs), sex=_extract_sex(lab), trans=None)


def load_cohort_pkls(paths: dict[str, str], pose_to_joints=None,
                     joints_key: str = "joints3d") -> list[Walk]:
    """Read the Tier-1 cohort ``.pkl`` files into :class:`Walk` records.

    Use this **only** for a hand-built / legacy pickle that already carries
    joints. The real CARE-PD h36m release does not: its 3D joints live in a
    separate ``.npz`` and the ``.pkl`` holds only SMPL parameters + labels —
    load that with :func:`load_h36m_cohorts` instead.

    ``paths`` maps cohort name -> ``.pkl`` path. Each ``.pkl`` is keyed
    ``subject_id -> walk_id -> {pose, trans, beta, fps, UPDRS_GAIT,
    medication, other}`` ([guideline §1]). Joint positions are read from
    ``joints_key`` if present; otherwise ``pose_to_joints(record)`` must
    return ``(F, 17, 3)`` (wire in the SMPL->H36M regressor with beta=0).
    ``other`` carries the FoG label.
    """
    walks: list[Walk] = []
    n_records = 0
    for cohort, path in paths.items():
        with open(path, "rb") as f:
            blob = pickle.load(f)
        for subject_id, walk_map in blob.items():
            for walk_id, rec in walk_map.items():
                n_records += 1
                joints = _extract_joints(rec, pose_to_joints, joints_key)
                if joints is None:
                    continue
                walks.append(Walk(
                    joints=joints, cohort=cohort, subject_id=str(subject_id),
                    walk_id=str(walk_id), fps=float(rec.get("fps", 30.0)),
                    fog=_extract_fog(rec), medication=_extract_med(rec),
                    updrs_gait=_num(rec.get("UPDRS_GAIT")),
                    sex=_extract_sex(rec),
                    trans=np.asarray(rec["trans"], float)
                    if isinstance(rec, dict) and "trans" in rec else None))
    if not walks:
        hint = (f" — all {n_records} records lacked a {joints_key!r} field and "
                "no pose_to_joints was given" if n_records else " (no records)")
        raise ValueError(
            f"load_cohort_pkls loaded 0 walks from {list(paths)}{hint}. The "
            "CARE-PD h36m release stores 3D joints in a separate .npz (the "
            ".pkl carries only SMPL params + labels) — use "
            "load_h36m_cohorts(h36m_root, source_dir=...) instead, or pass "
            "pose_to_joints=<SMPL->H36M regressor> if this .pkl really holds "
            "SMPL parameters.")
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
    if v is None and isinstance(rec, dict):
        v = rec.get("med")                      # care_pd standardises to "med"
    return _MED_MAP.get(v, np.nan if v is None else v)


def _extract_sex(rec):
    other = rec.get("other") if isinstance(rec, dict) else None
    for src in (rec if isinstance(rec, dict) else {},
                other if isinstance(other, dict) else {}):
        for k in ("sex", "gender"):
            v = src.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
    return np.nan


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
