"""CARE-PD adapter — map the release into the clip iterator ([CARE-PD §8]).

CARE-PD (https://github.com/TaatiTeam/CARE-PD, HuggingFace
``vida-adl/CARE-PD``) aggregates nine Parkinsonian-gait cohorts from eight
clinical sites, harmonised through SMPL fitting and re-exported in several
skeleton formats. This module targets the ``h36m/`` release, whose on-disk
layout is one subdirectory per cohort holding ``.npz`` archives, e.g.

    h36m/BMCLab/h36m_3d_world_floorXZZplus_30f_or_longer.npz

Several coordinate variants ship side by side; the world-coordinate 3D
file (``h36m_3d_world_*.npz``) is the one we want — global joint positions
in the H36M 22-joint skeleton, floor on the X-Z plane so the vertical
(up) axis is Y. The sibling ``h36m_3d_world2cam2img_*.npz`` is the
camera-projected variant for 2D visualisation and is skipped.

Each archive unpickles (``np.load(..., allow_pickle=True)``) to the
nested dict documented on the dataset card::

    { subject_id: { walk_id: {
        "pose": (F, 22, 3) float,   # world joint positions (h36m release)
        "trans": array,             # global translation (unused for h36m)
        "beta": array,              # SMPL shape, zeroed for privacy
        "fps": int,                 # standardised frame rate
        "UPDRS_GAIT": int | None,   # 0-3 gait sub-score
        "medication": str | None,   # ON / OFF
        "other": str | None,        # extra labels (e.g. FoG / freezer)
    } } }

Note the **two** levels of nesting: the *outer* key is the subject id
(so leave-one-subject-out splits read it straight off), the *inner* key
is the walk id.

The module has two independently useful halves:

1. **Preprocessing** ([CARE-PD §8]) — root-centre, walking-direction
   alignment, resampling to a common 30 fps. Pure NumPy, no dependency on
   the release layout, and covered by ``test_care_pd_no_torch.py``. This
   is the part that has to be right; windowing is delegated to the
   existing ``build_clips`` in the training loop. (The h36m release is
   already standardised to 30 fps, so the resample is usually a no-op, but
   it is kept so out-of-spec walks still land on the common grid.)

2. **Loading** — turn the per-cohort ``.npz`` archives into a list of
   ``Walk`` records with cohort id, subject id, fps, and a dict of
   clinical labels for evaluation. The mapping from raw record fields to
   ``Walk`` fields is isolated in :class:`RecordSchema` and the
   :data:`DEFAULT_SCHEMA`; if the release keys differ from the defaults,
   override the schema rather than editing the loader.

The tiering ([CARE-PD §3]) decides which cohorts enter which experiment.
Tier 1 (MoCap, richly labelled) is where the phenotype claim is made;
Tier 2 (RGB, UPDRS-labelled) adds severity coverage and the modality gap;
Tier 3 is unlabelled pre-training / OOD.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---- Cohort tiers ([CARE-PD §3]) -----------------------------------------

TIER1_COHORTS: tuple[str, ...] = ("BMCLab", "KUL-DT-T", "E-LC")
TIER2_COHORTS: tuple[str, ...] = ("PD-GaM", "3DGait", "T-SDU-PD")
TIER3_COHORTS: tuple[str, ...] = ("T-SDU", "T-LTC", "DNE")

ALL_COHORTS: tuple[str, ...] = TIER1_COHORTS + TIER2_COHORTS + TIER3_COHORTS

# Cohorts carrying an ordinal UPDRS_GAIT label ([CARE-PD §4.2]).
UPDRS_COHORTS: tuple[str, ...] = ("BMCLab", "PD-GaM", "3DGait", "T-SDU-PD")

# H36M 22-joint conventions. The pelvis/root is joint 0; the SMPL/H36M
# export used by CARE-PD is y-up, so the horizontal plane is x-z.
H36M_ROOT = 0
H36M_N_JOINTS = 22
DEFAULT_UP_AXIS = 1


def cohort_index(cohorts: tuple[str, ...] | list[str]) -> dict[str, int]:
    """Deterministic ``cohort name -> contiguous id`` map ([CARE-PD §6]).

    The id is the conditioning value ``c`` fed to a CVAE / GM-CVAE, so it
    must be stable and dense in ``[0, len(cohorts))``. Order follows the
    supplied sequence.
    """
    return {name: i for i, name in enumerate(cohorts)}


def tier_cohorts(*tiers: int) -> tuple[str, ...]:
    """Cohort names for one or more tiers, e.g. ``tier_cohorts(1, 2)``."""
    table = {1: TIER1_COHORTS, 2: TIER2_COHORTS, 3: TIER3_COHORTS}
    out: list[str] = []
    for t in tiers:
        if t not in table:
            raise ValueError(f"unknown tier {t!r}; expected 1, 2, or 3.")
        out.extend(table[t])
    return tuple(out)


# ---- Preprocessing ([CARE-PD §8]) ----------------------------------------

def root_center(pose: np.ndarray, root: int = H36M_ROOT) -> np.ndarray:
    """Subtract the root (pelvis) position from every joint, per frame.

    Args:
        pose: (F, J, 3).
        root: index of the pelvis joint.
    Returns:
        (F, J, 3) with the root at the origin in every frame.
    """
    pose = np.asarray(pose, dtype=np.float64)
    return pose - pose[:, root:root + 1, :]


def _horizontal_axes(up_axis: int) -> tuple[int, int]:
    """The two non-vertical coordinate indices, in ascending order."""
    return tuple(a for a in (0, 1, 2) if a != up_axis)  # type: ignore[return-value]


def align_direction(pose: np.ndarray, up_axis: int = DEFAULT_UP_AXIS,
                    root: int = H36M_ROOT, eps: float = 1e-6) -> np.ndarray:
    """Rotate a clip about the vertical axis so mean travel points +x.

    Otherwise a large slice of the latent would encode the (arbitrary)
    walking heading and swamp phenotype information ([CARE-PD §8]). The
    heading is estimated from the net horizontal displacement of the root
    across the sequence; the whole clip is rotated by its negation. Walks
    with negligible net travel (in-place turning, standing) are returned
    unchanged.

    Args:
        pose: (F, J, 3).
        up_axis: index of the vertical axis (default 1, y-up).
        root: pelvis joint index.
        eps: minimum horizontal displacement to bother aligning.
    Returns:
        (F, J, 3) rotated so the mean walking direction is +first
        horizontal axis.
    """
    pose = np.asarray(pose, dtype=np.float64)
    a, b = _horizontal_axes(up_axis)
    root_traj = pose[:, root, :]
    disp = root_traj[-1] - root_traj[0]
    da, db = disp[a], disp[b]
    if np.hypot(da, db) < eps:
        return pose.copy()

    phi = np.arctan2(db, da)
    cos, sin = np.cos(phi), np.sin(phi)
    # Rotate by -phi in the (a, b) plane: (da, db) -> (|disp|, 0).
    out = pose.copy()
    xa = pose[..., a]
    xb = pose[..., b]
    out[..., a] = cos * xa + sin * xb
    out[..., b] = -sin * xa + cos * xb
    return out


def resample_fps(pose: np.ndarray, src_fps: float,
                 dst_fps: float = 30.0) -> np.ndarray:
    """Linearly resample a sequence from ``src_fps`` to ``dst_fps``.

    CARE-PD's original rates span 25–150 fps ([CARE-PD §8]); everything is
    brought to a common 30 fps before windowing so a fixed clip length
    means a fixed duration.

    Args:
        pose: (F, J, 3).
        src_fps: source frame rate.
        dst_fps: target frame rate (default 30).
    Returns:
        (F', J, 3) with F' = round((F - 1) * dst / src) + 1.
    """
    pose = np.asarray(pose, dtype=np.float64)
    F = pose.shape[0]
    if F < 2 or abs(src_fps - dst_fps) < 1e-9:
        return pose.copy()
    duration = (F - 1) / float(src_fps)
    F_new = int(round(duration * dst_fps)) + 1
    F_new = max(F_new, 2)
    t_old = np.linspace(0.0, 1.0, F, dtype=np.float64)
    t_new = np.linspace(0.0, 1.0, F_new, dtype=np.float64)
    J, C = pose.shape[1], pose.shape[2]
    flat = pose.reshape(F, J * C)
    out = np.empty((F_new, J * C), dtype=np.float64)
    for k in range(J * C):
        out[:, k] = np.interp(t_new, t_old, flat[:, k])
    return out.reshape(F_new, J, C)


def preprocess_walk(pose: np.ndarray, src_fps: float, dst_fps: float = 30.0,
                    up_axis: int = DEFAULT_UP_AXIS, root: int = H36M_ROOT,
                    ) -> np.ndarray:
    """Full per-walk pipeline: resample, root-centre, direction-align.

    Resampling first so the direction estimate and the downstream windows
    both live at the common frame rate. Returns a continuous sequence;
    windowing is left to ``build_clips`` in the training loop so the same
    clip machinery serves neonate and CARE-PD data ([CARE-PD §2]).

    Args:
        pose: (F, J, 3) raw walk.
        src_fps: source frame rate of this walk.
        dst_fps: common target rate.
        up_axis: vertical-axis index.
        root: pelvis joint index.
    Returns:
        (F', J, 3) preprocessed walk, float32.
    """
    pose = resample_fps(pose, src_fps, dst_fps)
    pose = root_center(pose, root=root)
    pose = align_direction(pose, up_axis=up_axis, root=root)
    return pose.astype(np.float32)


def make_windows(pose: np.ndarray, clip_length: int = 60,
                 stride: int = 30) -> np.ndarray:
    """Cut a preprocessed walk into overlapping windows.

    Mirrors ``data.slice_video`` but exposed here for tests and for
    callers that want the windows without going through the loader. Walks
    shorter than one window yield an empty array.

    Args:
        pose: (F, J, 3).
        clip_length: T, default 60 (two seconds at 30 fps, [CARE-PD §8]).
        stride: hop between window starts, default 30 (50% overlap).
    Returns:
        (K, T, J, 3).
    """
    F = pose.shape[0]
    starts = list(range(0, max(F - clip_length + 1, 0), stride))
    if not starts:
        return np.empty((0, clip_length) + pose.shape[1:], dtype=pose.dtype)
    return np.stack([pose[s:s + clip_length] for s in starts])


# ---- Walk records and pickle loading -------------------------------------

@dataclass
class Walk:
    """One preprocessed walk with the metadata evaluation needs.

    Attributes:
        pose: (F, 22, 3) preprocessed H36M sequence at the common fps.
        cohort: cohort name (one of :data:`ALL_COHORTS`).
        subject: subject id, for leave-one-subject-out splits.
        fps: the common frame rate after preprocessing.
        labels: clinical annotations, evaluation-only ([CARE-PD §8]).
            Standardised keys where available: ``updrs_gait`` (int 0–3),
            ``freezer`` (bool / 0–1), ``med`` ("ON"/"OFF"), plus any raw
            fields passed straight through.
    """
    pose: np.ndarray
    cohort: str
    subject: str
    fps: float
    labels: dict = field(default_factory=dict)


@dataclass
class RecordSchema:
    """How to read one raw walk record out of a CARE-PD ``.npz`` archive.

    The field names below match the dataset card, but the ``*_keys``
    lists carry a few historical / alternative spellings tried in order so
    a minor release change is absorbed without a code edit. ``pose_keys``
    is required; the rest are best-effort and simply yield missing labels
    when absent. Every non-pose scalar field is also copied into
    ``Walk.labels`` under its raw key, so nothing is silently dropped.

    ``pose_kind`` records the reading of the pose array: for the h36m
    world release it is (F, 22, 3) joint positions. A flat (F, 66) array
    is reshaped to (F, 22, 3) automatically.
    """
    pose_keys: tuple[str, ...] = ("pose", "poses", "joints3d", "keypoints3d",
                                  "h36m", "kpts")
    fps_keys: tuple[str, ...] = ("fps", "frame_rate", "framerate")
    updrs_keys: tuple[str, ...] = ("UPDRS_GAIT", "updrs_gait", "updrs",
                                   "gait_score")
    med_keys: tuple[str, ...] = ("medication", "med", "med_state", "on_off")
    # Freezer / FoG status is not a dedicated field in the card; for
    # KUL-DT-T and E-LC it is carried in ``other``. Tried explicitly here,
    # and ``other`` is passed through raw regardless.
    freezer_keys: tuple[str, ...] = ("freezer", "FoG", "fog", "is_freezer")
    other_keys: tuple[str, ...] = ("other",)
    # Fields never treated as labels.
    non_label_keys: tuple[str, ...] = ("pose", "poses", "joints3d",
                                       "keypoints3d", "h36m", "kpts",
                                       "trans", "beta")
    default_fps: float = 30.0


# Back-compat alias; the old name referred to a .pkl-only loader.
PklSchema = RecordSchema
DEFAULT_SCHEMA = RecordSchema()

# Default filename glob for the world-coordinate 3D variant inside a
# cohort subdirectory. Excludes ``h36m_3d_world2cam2img_*`` because that
# starts "world2", not "world_".
WORLD_NPZ_GLOB = "h36m_3d_world_*.npz"


def _maybe_item(v):
    """Unwrap a 0-d object array to the Python object it holds.

    ``np.load`` on an ``.npz`` returns arrays; a dict saved inside one
    comes back as a 0-d ``object`` array, and ``.item()`` recovers it.
    Anything else is returned unchanged.
    """
    if isinstance(v, np.ndarray) and v.dtype == object and v.ndim == 0:
        return v.item()
    return v


def _first(record, keys: tuple[str, ...], default=None):
    if not isinstance(record, dict):
        return default
    for k in keys:
        if k in record:
            return record[k]
    return default


def _is_record(obj, schema: RecordSchema) -> bool:
    """True if ``obj`` is a single walk record rather than a walk-map.

    A record is either a bare pose array or a dict that carries a known
    field (pose / fps / a label). A walk-map is a dict whose *values* are
    records, so it carries none of those field names at the top level.
    """
    if isinstance(obj, np.ndarray):
        return True
    if not isinstance(obj, dict):
        return False
    known = (set(schema.pose_keys) | set(schema.fps_keys)
             | set(schema.updrs_keys) | set(schema.med_keys)
             | set(schema.freezer_keys) | set(schema.other_keys)
             | {"trans", "beta"})
    return any(k in known for k in obj.keys())


def _iter_walks(blob, schema: RecordSchema):
    """Yield ``(subject_id, walk_id, record)`` from an unpickled archive.

    Canonical CARE-PD layout is the nested ``{subject: {walk: record}}``;
    the subject id therefore comes off the *outer* key. Falls back to a
    flat ``{walk: record}`` dict and to a list of records so older dumps
    still load (subject id then defaults to the walk id).
    """
    if isinstance(blob, dict):
        for outer, inner in blob.items():
            inner = _maybe_item(inner)
            if isinstance(inner, dict) and not _is_record(inner, schema):
                # Nested: outer is the subject, inner is the walk-map.
                for walk_id, rec in inner.items():
                    yield str(outer), str(walk_id), _maybe_item(rec)
            else:
                # Flat: outer is both subject and walk.
                yield str(outer), str(outer), inner
    elif isinstance(blob, (list, tuple)):
        for i, rec in enumerate(blob):
            yield str(i), str(i), _maybe_item(rec)
    else:
        raise TypeError(
            f"unexpected archive top-level type {type(blob)!r}; expected a "
            "nested/flat dict of walks or a list of walk records."
        )


def _extract_pose(rec, schema: RecordSchema, walk_id: str) -> np.ndarray:
    """Pull the pose array out of a raw record and sanity-check its shape.

    Accepts (F, J, 3) directly, or a flat (F, J*3) that is reshaped.
    """
    if isinstance(rec, np.ndarray):
        pose = rec
    elif isinstance(rec, dict):
        pose = _first(rec, schema.pose_keys)
        if pose is None:
            raise KeyError(
                f"walk {walk_id!r}: no pose field found among "
                f"{schema.pose_keys}; keys present: {list(rec)[:12]}."
            )
    else:
        raise TypeError(f"walk {walk_id!r}: unsupported record type {type(rec)!r}.")
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 2 and pose.shape[1] % 3 == 0:
        pose = pose.reshape(pose.shape[0], pose.shape[1] // 3, 3)
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise ValueError(
            f"walk {walk_id!r}: expected pose shape (F, J, 3) or (F, J*3), "
            f"got {pose.shape}. If this is the 6D_SMPL or HumanML3D release "
            f"rather than h36m, point the loader at the h36m folder."
        )
    return pose


def _load_archive(path: Path):
    """Unpickle a cohort archive (``.npz`` primary, ``.pkl`` fallback).

    An ``.npz`` that stores the nested dict as a single object array is
    unwrapped with ``.item()``; a multi-key ``.npz`` is returned as a
    plain ``{key: array}`` dict.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            keys = list(npz.keys())
            if len(keys) == 1:
                return _maybe_item(npz[keys[0]])
            return {k: _maybe_item(npz[k]) for k in keys}
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_cohort_path(root_dir: Path, cohort: str,
                         variant_glob: str) -> Path:
    """Find the archive for one cohort under the ``h36m/`` release tree.

    Handles both the documented ``h36m/<cohort>/<variant>.npz`` layout and
    a flat ``h36m/<cohort>.npz`` / ``.pkl`` fallback.
    """
    root_dir = Path(root_dir)
    cohort_dir = root_dir / cohort
    if cohort_dir.is_dir():
        matches = sorted(cohort_dir.glob(variant_glob))
        if not matches:
            matches = sorted(cohort_dir.glob("*.npz"))
        if not matches:
            raise FileNotFoundError(
                f"no archive matching {variant_glob!r} (or any .npz) in "
                f"{cohort_dir}."
            )
        return matches[0]
    for ext in (".npz", ".pkl"):
        flat = root_dir / f"{cohort}{ext}"
        if flat.exists():
            return flat
    raise FileNotFoundError(
        f"cohort {cohort!r}: neither {cohort_dir}/ nor "
        f"{root_dir / (cohort + '.npz')} exists."
    )


def load_cohort(path: str | Path, cohort: str,
                schema: RecordSchema = DEFAULT_SCHEMA,
                dst_fps: float = 30.0,
                up_axis: int = DEFAULT_UP_AXIS,
                root: int = H36M_ROOT,
                preprocess: bool = True) -> list[Walk]:
    """Load and (optionally) preprocess one cohort archive into walks.

    Args:
        path: path to the cohort's ``.npz`` (or ``.pkl``) archive.
        cohort: cohort name recorded on every walk.
        schema: field-name mapping (see :class:`RecordSchema`).
        dst_fps: common frame rate to resample to.
        up_axis, root: skeleton conventions for preprocessing.
        preprocess: run the resample / root-centre / align pipeline. Set
            False to inspect raw sequences.
    Returns:
        List of :class:`Walk`, with ``subject`` taken from the outer key.
    """
    blob = _load_archive(Path(path))

    walks: list[Walk] = []
    for subject_id, walk_id, rec in _iter_walks(blob, schema):
        pose = _extract_pose(rec, schema, walk_id)
        meta = rec if isinstance(rec, dict) else {}
        src_fps = float(_first(meta, schema.fps_keys, schema.default_fps))

        # Standardised label aliases, plus a raw passthrough of every
        # non-pose scalar so nothing is lost (FoG often lives in "other").
        labels: dict = {}
        for k, v in meta.items():
            if k in schema.non_label_keys or k in schema.fps_keys:
                continue
            labels[k] = v
        updrs = _first(meta, schema.updrs_keys)
        if updrs is not None:
            labels["updrs_gait"] = updrs
        med = _first(meta, schema.med_keys)
        if med is not None:
            labels["med"] = med
        freezer = _first(meta, schema.freezer_keys)
        if freezer is not None:
            labels["freezer"] = freezer

        if preprocess:
            pose = preprocess_walk(pose, src_fps, dst_fps, up_axis, root)
            fps = dst_fps
        else:
            pose = pose.astype(np.float32)
            fps = src_fps
        walks.append(Walk(pose=pose, cohort=cohort, subject=str(subject_id),
                          fps=fps, labels=labels))
    return walks


# Back-compat alias for the earlier .pkl-only name.
load_cohort_pkl = load_cohort


def load_cohorts(root_dir: str | Path, cohorts: tuple[str, ...] | list[str],
                 schema: RecordSchema = DEFAULT_SCHEMA,
                 variant_glob: str = WORLD_NPZ_GLOB, **kwargs) -> list[Walk]:
    """Load several cohorts from an ``h36m/`` release tree, concatenating.

    Args:
        root_dir: the ``h36m/`` directory holding ``<cohort>/<variant>.npz``.
        cohorts: cohort names to load (e.g. ``TIER1_COHORTS``).
        schema: field-name mapping.
        variant_glob: filename pattern selecting the coordinate variant
            inside each cohort subdirectory. Defaults to the
            world-coordinate 3D file, :data:`WORLD_NPZ_GLOB`.
        **kwargs: forwarded to :func:`load_cohort`.
    Returns:
        Flat list of :class:`Walk` across all requested cohorts.
    """
    root_dir = Path(root_dir)
    walks: list[Walk] = []
    for name in cohorts:
        path = _resolve_cohort_path(root_dir, name, variant_glob)
        walks.extend(load_cohort(path, name, schema=schema, **kwargs))
    return walks


# ---- Bridging walks into the training loop -------------------------------

@dataclass
class CarePDBundle:
    """Everything a training / evaluation run needs, aligned by walk index.

    ``videos`` plugs straight into ``train(config, videos,
    cohort_per_video=cohort_ids)``; ``build_clips`` inside the loop does
    the windowing. ``subjects`` and ``labels`` are held aside for the
    split helpers and the metrics module.
    """
    videos: list[np.ndarray]
    cohort_ids: np.ndarray          # (n_walks,) int, index into `cohorts`
    cohort_names: list[str]         # (n_walks,) str
    subjects: list[str]             # (n_walks,)
    labels: list[dict]              # (n_walks,)
    cohorts: tuple[str, ...]        # ordered cohort vocabulary
    index: dict[str, int]           # cohort name -> id

    @property
    def n_cond(self) -> int:
        return len(self.cohorts)


def build_bundle(walks: list[Walk],
                 cohorts: tuple[str, ...] | list[str] | None = None
                 ) -> CarePDBundle:
    """Assemble a :class:`CarePDBundle` from a list of preprocessed walks.

    Args:
        walks: output of :func:`load_cohorts` / :func:`load_cohort_pkl`.
        cohorts: the conditioning vocabulary and its order. Defaults to the
            distinct cohorts present, ordered by :data:`ALL_COHORTS`.
    Returns:
        A bundle whose fields are aligned by walk index.
    """
    if cohorts is None:
        present = {w.cohort for w in walks}
        cohorts = tuple(c for c in ALL_COHORTS if c in present)
    cohorts = tuple(cohorts)
    idx = cohort_index(cohorts)

    videos, cids, cnames, subjects, labels = [], [], [], [], []
    for w in walks:
        if w.cohort not in idx:
            continue
        videos.append(w.pose)
        cids.append(idx[w.cohort])
        cnames.append(w.cohort)
        subjects.append(w.subject)
        labels.append(w.labels)
    return CarePDBundle(
        videos=videos,
        cohort_ids=np.asarray(cids, dtype=np.int64),
        cohort_names=cnames,
        subjects=subjects,
        labels=labels,
        cohorts=cohorts,
        index=idx,
    )


# ---- Split regimes ([CARE-PD §8]) ----------------------------------------

def leave_one_subject_out(bundle: CarePDBundle, cohort: str | None = None):
    """Yield ``(name, train_idx, test_idx)`` holding out one subject each.

    Within-cohort LOSO ([CARE-PD §8]) tests generalisation across patients.
    Restrict to a single cohort with ``cohort=`` (BMCLab is the plan's
    default); otherwise subjects are held out across the whole bundle.

    Args:
        bundle: a :class:`CarePDBundle`.
        cohort: optional cohort to restrict the split to.
    Yields:
        ``(subject_id, train_indices, test_indices)`` over walk indices.
    """
    subjects = np.asarray(bundle.subjects, dtype=object)
    names = np.asarray(bundle.cohort_names, dtype=object)
    pool = np.arange(len(subjects))
    if cohort is not None:
        pool = pool[names[pool] == cohort]
    for subj in sorted({subjects[i] for i in pool}):
        test = np.array([i for i in pool if subjects[i] == subj], dtype=np.int64)
        train = np.array([i for i in pool if subjects[i] != subj], dtype=np.int64)
        yield str(subj), train, test


def leave_one_cohort_out(bundle: CarePDBundle):
    """Yield ``(cohort, train_idx, test_idx)`` holding out one cohort each.

    Cross-cohort LODO ([CARE-PD §8]) stress-tests the cohort-invariance
    claim: the model never sees the held-out cohort in training.

    Yields:
        ``(cohort_name, train_indices, test_indices)`` over walk indices.
    """
    names = np.asarray(bundle.cohort_names, dtype=object)
    for cohort in bundle.cohorts:
        test = np.where(names == cohort)[0].astype(np.int64)
        if test.size == 0:
            continue
        train = np.where(names != cohort)[0].astype(np.int64)
        yield cohort, train, test


def subset(bundle: CarePDBundle, idx: np.ndarray) -> CarePDBundle:
    """Restrict a bundle to the given walk indices, preserving the vocab."""
    idx = np.asarray(idx, dtype=np.int64)
    return CarePDBundle(
        videos=[bundle.videos[i] for i in idx],
        cohort_ids=bundle.cohort_ids[idx],
        cohort_names=[bundle.cohort_names[i] for i in idx],
        subjects=[bundle.subjects[i] for i in idx],
        labels=[bundle.labels[i] for i in idx],
        cohorts=bundle.cohorts,
        index=bundle.index,
    )
