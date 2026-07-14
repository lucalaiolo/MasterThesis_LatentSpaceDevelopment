"""CARE-PD adapter — map the release into the clip iterator ([CARE-PD §8]).

CARE-PD (https://github.com/TaatiTeam/CARE-PD, HuggingFace
``vida-adl/CARE-PD``) aggregates nine Parkinsonian-gait cohorts from eight
clinical sites, harmonised through SMPL fitting and re-exported in several
skeleton formats. This module targets the ``h36m/`` release **as produced
by ``bash scripts/preprocess_smpl2h36m.sh``**, whose on-disk layout is one
subdirectory per cohort holding ``.npz`` archives:

    h36m/BMCLab/h36m_3d_world_floorXZZplus_30f_or_longer.npz
    h36m/BMCLab/h36m_3d_world2cam_sideright_...npz          (camera 3D)
    h36m/BMCLab/h36m_3d_world2cam_backright_...npz          (camera 3D)
    h36m/BMCLab/h36m_3d_world2cam2img_sideright_...npz      (image 2D)
    h36m/BMCLab/h36m_3d_world2cam2img_backright_...npz      (image 2D)

Only the *world* 3D file is used here — the rest are camera-projected
variants that would inject a nuisance axis of view angle (``world2cam``)
or drop depth entirely (``world2cam2img``). The suffix decodes as: the
regressor produces the H36M **17-joint** skeleton (not 22 — that was
speculation before I read the script; SMPL vertices go through
``J_regressor_h36m_correct.npy`` which is the standard 17-joint one), and
the ``floorXZZplus`` transform sets Y=0 at the floor, moves the root's
frame-0 XZ to origin, and rotates about vertical so each walk faces +Z.

The archive is a **flat** dict — one top-level key per walk, value is the
raw pose array::

    { "subject_id__walk_id": (F, 17, 3) float32, ... }

Subject id and walk id are joined with a double-underscore in the key;
:func:`_iter_walks` splits them back. **Labels do not travel with the
h36m release** (only pose arrays are exported by ``smpl2h36m.py``); UPDRS,
medication, freezer, and cohort-specific ``other`` fields live in the
source SMPL ``.pkl``. :func:`load_labels_pkl` reads them from there and
:func:`load_cohort` merges them in when the optional ``source_pkl`` is
given. Training does not need labels — only analysis does — so a run
without ``source_pkl`` is fully functional.

The nested ``{subject: {walk: record}}`` schema handled by the earlier
version is still recognised and unchanged, so a hand-built pickle or an
older CARE-PD dump loads through the same code path.

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

# H36M skeleton conventions. ``smpl2h36m.py`` maps SMPL vertices through
# ``J_regressor_h36m_correct.npy`` to the standard 17-joint H36M skeleton,
# with the pelvis at joint 0. Its ``floorXZZplus`` transform is Y-up (floor
# on the X-Z plane), so the horizontal axes are 0 and 2.
H36M_ROOT = 0
H36M_N_JOINTS = 17
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
        pose: (F, 17, 3) preprocessed H36M sequence at the common fps.
        cohort: cohort name (one of :data:`ALL_COHORTS`).
        subject: subject id, for leave-one-subject-out splits.
        walk_id: within-subject walk id, for label matching against the
            source SMPL ``.pkl``.
        fps: the common frame rate after preprocessing.
        labels: clinical annotations, evaluation-only ([CARE-PD §8]).
            Empty on a bare h36m load — labels only appear once a
            ``source_pkl`` is supplied to :func:`load_cohort`. Standardised
            keys where available: ``updrs_gait`` (int 0–3), ``freezer``
            (bool / 0–1), ``med`` ("ON"/"OFF"), plus any raw fields (e.g.
            ``other``) passed straight through.
    """
    pose: np.ndarray
    cohort: str
    subject: str
    walk_id: str = ""
    fps: float = 30.0
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

    Three layouts are recognised:

    1. **h36m release** — a flat ``{ "subject__walkid": (F, 17, 3) }``
       dict, exactly what ``smpl2h36m.py`` emits. The double-underscore
       separator is defined by that script's
       ``walk_name = str(subject_id) + '__' + str(walk_id)`` line; splitting
       it here recovers the subject id, which LOSO needs.
    2. **Nested source pickle** — ``{subject: {walk: record}}`` (used by
       some of the raw SMPL cohort ``.pkl`` files). Subject comes off the
       outer key, walk id off the inner.
    3. **List of records** — walk id defaults to the list index.
    """
    if isinstance(blob, dict):
        for outer, inner in blob.items():
            inner = _maybe_item(inner)
            if isinstance(inner, dict) and not _is_record(inner, schema):
                # Nested: outer is the subject, inner is the walk-map.
                for walk_id, rec in inner.items():
                    yield str(outer), str(walk_id), _maybe_item(rec)
            else:
                # Flat: either bare "subject__walkid" (h36m npz) or a raw
                # record whose outer key already holds both ids joined by
                # "__". Fall back to using the whole key as both when the
                # separator is missing.
                outer_s = str(outer)
                if "__" in outer_s:
                    subject_id, walk_id = outer_s.split("__", 1)
                else:
                    subject_id = walk_id = outer_s
                yield subject_id, walk_id, inner
    elif isinstance(blob, (list, tuple)):
        for i, rec in enumerate(blob):
            yield str(i), str(i), _maybe_item(rec)
    else:
        raise TypeError(
            f"unexpected archive top-level type {type(blob)!r}; expected a "
            "flat / nested dict of walks or a list of walk records."
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


def _extract_labels(rec, schema: RecordSchema) -> dict:
    """Pull the label dict out of one raw record, applying label aliases."""
    if not isinstance(rec, dict):
        return {}
    labels: dict = {}
    for k, v in rec.items():
        if k in schema.non_label_keys or k in schema.fps_keys:
            continue
        labels[k] = v
    updrs = _first(rec, schema.updrs_keys)
    if updrs is not None:
        labels["updrs_gait"] = updrs
    med = _first(rec, schema.med_keys)
    if med is not None:
        labels["med"] = med
    freezer = _first(rec, schema.freezer_keys)
    if freezer is not None:
        labels["freezer"] = freezer
    return labels


def load_labels_pkl(path: str | Path,
                    schema: RecordSchema = DEFAULT_SCHEMA
                    ) -> dict[str, dict]:
    """Read per-walk labels out of a source SMPL ``.pkl`` (chumpy-free).

    Returned dict is double-keyed so it matches whatever walk id the h36m
    npz uses: entries appear under both ``"subject__walkid"`` (the
    concatenated key ``smpl2h36m.py`` emits) and just ``"walkid"`` (in case
    a cohort's raw pickle used walk-only keys). Only Python objects — no
    chumpy — are read out; the raw ``.pkl`` fields ``pose``/``beta``/
    ``trans`` are ignored.

    Args:
        path: path to a cohort's raw SMPL ``.pkl``.
        schema: field-name mapping.
    Returns:
        Dict mapping walk key -> labels dict.
    """
    blob = _load_archive(Path(path))
    out: dict[str, dict] = {}
    for subject_id, walk_id, rec in _iter_walks(blob, schema):
        lbl = _extract_labels(rec, schema)
        if not lbl:
            continue
        out[f"{subject_id}__{walk_id}"] = lbl
        out.setdefault(walk_id, lbl)
    return out


def load_cohort(path: str | Path, cohort: str,
                schema: RecordSchema = DEFAULT_SCHEMA,
                dst_fps: float = 30.0,
                up_axis: int = DEFAULT_UP_AXIS,
                root: int = H36M_ROOT,
                preprocess: bool = True,
                source_pkl: str | Path | None = None) -> list[Walk]:
    """Load and (optionally) preprocess one cohort archive into walks.

    Args:
        path: path to the cohort's h36m ``.npz`` (or a nested ``.pkl``).
        cohort: cohort name recorded on every walk.
        schema: field-name mapping (see :class:`RecordSchema`).
        dst_fps: common frame rate to resample to (30 by default; the h36m
            release is already at 30, so :func:`resample_fps` is a no-op).
        up_axis, root: skeleton conventions for preprocessing.
        preprocess: run the resample / root-centre / align pipeline. On
            the already-canonical h36m data only per-frame root-centring
            has visible effect; resample and align both early-return.
        source_pkl: optional path to the raw SMPL ``.pkl`` for this cohort
            to attach labels (UPDRS, medication, freezer, …) by walk id.
            Not needed for training; only for the §11 analysis.
    Returns:
        List of :class:`Walk`, with ``subject`` and ``walk_id`` recovered
        from the archive keys.
    """
    blob = _load_archive(Path(path))
    label_lookup = load_labels_pkl(source_pkl, schema) if source_pkl else {}

    walks: list[Walk] = []
    for subject_id, walk_id, rec in _iter_walks(blob, schema):
        pose = _extract_pose(rec, schema, walk_id)
        meta = rec if isinstance(rec, dict) else {}
        src_fps = float(_first(meta, schema.fps_keys, schema.default_fps))

        # Labels present in the archive itself (nested-pickle case), then
        # anything the source ``.pkl`` adds under either key form. The
        # h36m ``.npz`` supplies neither, so both are commonly empty.
        labels = _extract_labels(rec, schema)
        if label_lookup:
            extra = (label_lookup.get(f"{subject_id}__{walk_id}")
                     or label_lookup.get(walk_id) or {})
            labels = {**labels, **extra}

        if preprocess:
            pose = preprocess_walk(pose, src_fps, dst_fps, up_axis, root)
            fps = dst_fps
        else:
            pose = pose.astype(np.float32)
            fps = src_fps
        walks.append(Walk(pose=pose, cohort=cohort,
                          subject=str(subject_id), walk_id=str(walk_id),
                          fps=fps, labels=labels))
    return walks


# Back-compat alias for the earlier .pkl-only name.
load_cohort_pkl = load_cohort


def load_cohorts(root_dir: str | Path, cohorts: tuple[str, ...] | list[str],
                 schema: RecordSchema = DEFAULT_SCHEMA,
                 variant_glob: str = WORLD_NPZ_GLOB,
                 source_dir: str | Path | None = None,
                 **kwargs) -> list[Walk]:
    """Load several cohorts from an ``h36m/`` release tree, concatenating.

    Args:
        root_dir: the ``h36m/`` directory holding ``<cohort>/<variant>.npz``.
        cohorts: cohort names to load (e.g. ``TIER1_COHORTS``).
        schema: field-name mapping.
        variant_glob: filename pattern selecting the coordinate variant
            inside each cohort subdirectory. Defaults to the
            world-coordinate 3D file, :data:`WORLD_NPZ_GLOB`, which excludes
            the ``world2cam*`` and ``world2cam2img*`` view-projected files.
        source_dir: optional directory holding the raw SMPL ``.pkl`` for
            each cohort (typically the sibling of ``h36m/``, e.g.
            ``assets/datasets/``). When given, ``<source_dir>/<cohort>.pkl``
            is used to attach labels; a missing per-cohort ``.pkl`` is
            silently skipped so partial label coverage still works.
        **kwargs: forwarded to :func:`load_cohort` (``source_pkl`` will be
            filled in from ``source_dir`` when not passed explicitly).
    Returns:
        Flat list of :class:`Walk` across all requested cohorts.
    """
    root_dir = Path(root_dir)
    source_dir = Path(source_dir) if source_dir is not None else None
    walks: list[Walk] = []
    for name in cohorts:
        path = _resolve_cohort_path(root_dir, name, variant_glob)
        per_kwargs = dict(kwargs)
        if source_dir is not None and "source_pkl" not in per_kwargs:
            candidate = source_dir / f"{name}.pkl"
            if candidate.exists():
                per_kwargs["source_pkl"] = candidate
        walks.extend(load_cohort(path, name, schema=schema, **per_kwargs))
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
