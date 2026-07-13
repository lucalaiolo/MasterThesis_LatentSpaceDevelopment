"""CARE-PD adapter — map the release into the clip iterator ([CARE-PD §8]).

CARE-PD (https://github.com/TaatiTeam/CARE-PD, HuggingFace
``vida-adl/CARE-PD``) aggregates nine Parkinsonian-gait cohorts from eight
clinical sites, harmonised through SMPL fitting and re-exported in several
skeleton formats. This module targets the ``h36m/`` release: one ``.pkl``
per cohort (``BMCLab.pkl``, ``KUL-DT-T.pkl``, …), each a collection of
walks stored as H36M 22-joint 3D sequences.

The module has two independently useful halves:

1. **Preprocessing** ([CARE-PD §8]) — root-centre, walking-direction
   alignment, resampling to a common 30 fps. Pure NumPy, no dependency on
   the release layout, and covered by ``test_care_pd_no_torch.py``. This
   is the part that has to be right; windowing is delegated to the
   existing ``build_clips`` in the training loop.

2. **Loading** — turn the per-cohort ``.pkl`` files into a list of
   ``Walk`` records with cohort id, subject id, fps, and a dict of
   clinical labels for evaluation. The mapping from raw pickle fields to
   ``Walk`` fields is isolated in :class:`PklSchema` and the
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
class PklSchema:
    """How to read one raw walk record out of a cohort ``.pkl``.

    The CARE-PD ``h36m/*.pkl`` files are collections of walks; the exact
    container (dict keyed by walk id, or a list of records) and the field
    names vary a little across the release history. Rather than bake one
    layout into the loader, the mapping lives here so a mismatch is a
    one-line override, not a code edit.

    Each ``*_keys`` entry is a list of candidate field names tried in
    order; the first present wins. ``pose_keys`` is required; the rest are
    best-effort and simply produce missing labels when absent.
    """
    pose_keys: tuple[str, ...] = ("pose", "poses", "joints3d", "keypoints3d",
                                  "h36m", "kpts")
    subject_keys: tuple[str, ...] = ("subject", "subject_id", "subj", "pid",
                                     "participant")
    fps_keys: tuple[str, ...] = ("fps", "frame_rate", "framerate")
    updrs_keys: tuple[str, ...] = ("updrs_gait", "UPDRS_GAIT", "updrs",
                                   "gait_score")
    freezer_keys: tuple[str, ...] = ("freezer", "FoG", "fog", "is_freezer")
    med_keys: tuple[str, ...] = ("med", "medication", "med_state", "on_off")
    default_fps: float = 30.0


DEFAULT_SCHEMA = PklSchema()


def _first(record: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in record:
            return record[k]
    return default


def _iter_raw_records(blob):
    """Yield ``(walk_id, record_dict)`` from an unpickled cohort blob.

    Handles the two shapes seen in the release: a dict keyed by walk id
    whose values are per-walk dicts, and a flat list of per-walk dicts.
    """
    if isinstance(blob, dict):
        for wid, rec in blob.items():
            yield str(wid), rec
    elif isinstance(blob, (list, tuple)):
        for i, rec in enumerate(blob):
            yield str(i), rec
    else:
        raise TypeError(
            f"unexpected pickle top-level type {type(blob)!r}; expected a "
            "dict of walks or a list of walk records."
        )


def _extract_pose(rec, schema: PklSchema, walk_id: str) -> np.ndarray:
    """Pull the pose array out of a raw record and sanity-check its shape."""
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
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise ValueError(
            f"walk {walk_id!r}: expected pose shape (F, J, 3), got {pose.shape}."
        )
    return pose


def load_cohort_pkl(path: str | Path, cohort: str,
                    schema: PklSchema = DEFAULT_SCHEMA,
                    dst_fps: float = 30.0,
                    up_axis: int = DEFAULT_UP_AXIS,
                    root: int = H36M_ROOT,
                    preprocess: bool = True) -> list[Walk]:
    """Load and (optionally) preprocess one cohort's ``.pkl`` into walks.

    Args:
        path: path to ``<cohort>.pkl`` in the ``h36m/`` release folder.
        cohort: cohort name recorded on every walk.
        schema: field-name mapping (see :class:`PklSchema`).
        dst_fps: common frame rate to resample to.
        up_axis, root: skeleton conventions for preprocessing.
        preprocess: run the resample / root-centre / align pipeline. Set
            False to inspect raw sequences.
    Returns:
        List of :class:`Walk`.
    """
    with open(path, "rb") as f:
        blob = pickle.load(f)

    walks: list[Walk] = []
    for wid, rec in _iter_raw_records(blob):
        pose = _extract_pose(rec, schema, wid)
        meta = rec if isinstance(rec, dict) else {}
        src_fps = float(_first(meta, schema.fps_keys, schema.default_fps))
        subject = str(_first(meta, schema.subject_keys, wid))

        labels: dict = {}
        updrs = _first(meta, schema.updrs_keys)
        if updrs is not None:
            labels["updrs_gait"] = updrs
        freezer = _first(meta, schema.freezer_keys)
        if freezer is not None:
            labels["freezer"] = freezer
        med = _first(meta, schema.med_keys)
        if med is not None:
            labels["med"] = med

        if preprocess:
            pose = preprocess_walk(pose, src_fps, dst_fps, up_axis, root)
            fps = dst_fps
        else:
            pose = pose.astype(np.float32)
            fps = src_fps
        walks.append(Walk(pose=pose, cohort=cohort, subject=subject,
                          fps=fps, labels=labels))
    return walks


def load_cohorts(root_dir: str | Path, cohorts: tuple[str, ...] | list[str],
                 schema: PklSchema = DEFAULT_SCHEMA, **kwargs) -> list[Walk]:
    """Load several cohorts from an ``h36m/`` folder, concatenating walks.

    Args:
        root_dir: the ``h36m/`` directory holding ``<cohort>.pkl`` files.
        cohorts: cohort names to load (e.g. ``TIER1_COHORTS``).
        schema: field-name mapping.
        **kwargs: forwarded to :func:`load_cohort_pkl`.
    Returns:
        Flat list of :class:`Walk` across all requested cohorts.
    """
    root_dir = Path(root_dir)
    walks: list[Walk] = []
    for name in cohorts:
        walks.extend(load_cohort_pkl(root_dir / f"{name}.pkl", name,
                                     schema=schema, **kwargs))
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
