"""Post-hoc data layer: encode clips, align metadata, build trajectories.

Everything the post-hoc analysis ([post-hoc plan §1-§5]) reads flows out
of this module. It turns a :class:`architectures.care_pd.CarePDBundle`
(preprocessed walks + cohort ids + subjects + clinical labels) and one or
more trained encoders into a single :class:`PosthocData` container that
carries, per clip:

    - the posterior mean ``mu`` for every model (plain VAE, CVAE, ...),
    - the clip's cohort id / cohort name / subject / source-walk index,
    - the clinical labels (UPDRS_GAIT, freezer, medication) and the cohort
      itself as a control label,

plus, per walk, the **outer-loop trajectory** ([post-hoc plan §4.1]): the
ordered sequence of window latents used by the HMM and PELT analyses.

The claim is representational and we *do not retrain*: encoders are frozen.
The plain VAE ignores the conditioning id; the CVAE is fed its real cohort
so its latent is the cohort-conditioned one the plan asks for. Clips are
encoded with an all-visible mask by default so the latent reflects the
pose, not a particular mask draw.

Nothing here imports torch; a torch encoder is wrapped through
:class:`TorchCohortEncoder` (see :func:`load_encoder`), and a plain NumPy
callable works just as well, which is what the smoke test uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from ..interfaces import LatentSet


# ---- Label normalisation ---------------------------------------------------

# The label keys the CARE-PD adapter standardises to ([CARE-PD §8],
# ``architectures.care_pd``). ``cohort`` is appended as the §2.2 control.
CATEGORICAL_LABELS: tuple[str, ...] = ("updrs_gait", "freezer", "med", "cohort")


def norm_updrs(v) -> int | None:
    """Coerce a UPDRS_GAIT value to an integer ordinal level, or None.

    The gait item is already an ordinal 0-4 ([CARE-PD §4.2]); "binning the
    ordinal into its levels" ([post-hoc plan §2.2]) is just an integer
    cast. Non-numeric / missing values return None so they drop out of the
    agreement scoring.
    """
    if v is None:
        return None
    try:
        if isinstance(v, str):
            v = v.strip()
            if v == "" or v.lower() in ("nan", "none", "na"):
                return None
        f = float(v)
        if np.isnan(f):
            return None
        return int(round(f))
    except (TypeError, ValueError):
        return None


def norm_freezer(v) -> str | None:
    """Normalise a freezer flag to ``"freezer"`` / ``"non-freezer"`` or None.

    Accepts bools, 0/1, and the usual string spellings. The
    freezer/non-freezer split is a phenotype axis ([CARE-PD §2.3]).
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "nan", "none", "na"):
            return None
        if s in ("freezer", "fog", "yes", "y", "true", "1", "1.0"):
            return "freezer"
        if s in ("non-freezer", "nonfreezer", "non_freezer", "no", "n",
                 "false", "0", "0.0"):
            return "non-freezer"
        return s
    try:
        f = float(v)
        if np.isnan(f):
            return None
        return "freezer" if f > 0.5 else "non-freezer"
    except (TypeError, ValueError):
        return None


def norm_med(v) -> str | None:
    """Normalise a medication-state value to ``"ON"`` / ``"OFF"`` or None.

    Medication state (ON/OFF) is a within-subject phenotype axis recorded
    for BMCLab and E-LC ([CARE-PD §2.4]).
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "nan", "none", "na"):
            return None
        if s in ("on", "1", "1.0", "true"):
            return "ON"
        if s in ("off", "0", "0.0", "false"):
            return "OFF"
        return v.strip().upper()
    try:
        f = float(v)
        if np.isnan(f):
            return None
        return "ON" if f > 0.5 else "OFF"
    except (TypeError, ValueError):
        return None


_LABEL_NORMALISERS: dict[str, Callable] = {
    "updrs_gait": norm_updrs,
    "freezer": norm_freezer,
    "med": norm_med,
}


def normalise_walk_label(label_key: str, walk_labels: dict):
    """Read one standardised label off a walk's raw label dict."""
    raw = walk_labels.get(label_key)
    fn = _LABEL_NORMALISERS.get(label_key)
    return fn(raw) if fn is not None else raw


# ---- FoG interval extraction (E-LC, [post-hoc plan §4.3]) ------------------

def extract_fog_intervals(walk_labels: dict, fps: float, n_frames: int,
                          keys: tuple[str, ...] = (
                              "fog_intervals", "FoG_intervals",
                              "freezing_intervals", "fog", "FoG", "freezing"),
                          units: str = "auto") -> list[tuple[float, float]]:
    """Best-effort parse of annotated FoG intervals into ``(start_s, end_s)``.

    E-LC carries timestamped freezing-of-gait events ([CARE-PD §2.3]); the
    change-point validation ([post-hoc plan §4.3]) needs them as second-
    valued intervals aligned to the walk. The on-disk representation is not
    fixed by the dataset card, so this tries the common shapes and can be
    swapped out through the driver's ``fog_extractor`` hook:

    - a list / array of ``[start, end]`` pairs (the primary case);
    - a flat even-length list ``[s0, e0, s1, e1, ...]``;
    - a dict with ``start``/``onset`` and ``end``/``offset`` lists;
    - the same nested under a raw ``other`` field.

    Args:
        walk_labels: the walk's label dict (``Walk.labels``).
        fps: frames per second of the (preprocessed) walk.
        n_frames: walk length, used to decide the ``"auto"`` unit.
        keys: label keys to search, in order.
        units: ``"seconds"``, ``"frames"``, or ``"auto"`` (guess from the
            magnitude relative to the walk duration).
    Returns:
        List of ``(start_s, end_s)`` in seconds, clipped to the walk and
        sorted; empty when no annotation is found.
    """
    src = None
    search = dict(walk_labels)
    other = walk_labels.get("other")
    if isinstance(other, dict):
        search = {**search, **other}
    for k in keys:
        if k in search and search[k] is not None:
            src = search[k]
            break
    if src is None:
        return []

    pairs: list[tuple[float, float]] = []
    try:
        arr = src
        if isinstance(arr, dict):
            starts = arr.get("start", arr.get("onset"))
            ends = arr.get("end", arr.get("offset"))
            if starts is None or ends is None:
                return []
            pairs = [(float(s), float(e)) for s, e in zip(starts, ends)]
        else:
            a = np.asarray(arr, dtype=float)
            if a.ndim == 2 and a.shape[1] == 2:
                pairs = [(float(s), float(e)) for s, e in a]
            elif a.ndim == 1 and a.size % 2 == 0:
                a = a.reshape(-1, 2)
                pairs = [(float(s), float(e)) for s, e in a]
            else:
                return []
    except (TypeError, ValueError):
        return []

    if not pairs:
        return []

    duration_s = max(n_frames / float(fps), 1e-6)
    use_frames = units == "frames"
    if units == "auto":
        hi = max(e for _, e in pairs)
        # If the largest offset overshoots the walk in seconds but fits in
        # frames, the annotation is in frames.
        use_frames = hi > duration_s * 1.5 and hi <= n_frames * 1.5
    scale = (1.0 / float(fps)) if use_frames else 1.0

    out = []
    for s, e in pairs:
        s_s, e_s = s * scale, e * scale
        if e_s < s_s:
            s_s, e_s = e_s, s_s
        s_s = max(0.0, min(s_s, duration_s))
        e_s = max(0.0, min(e_s, duration_s))
        if e_s > s_s:
            out.append((s_s, e_s))
    out.sort()
    return out


# ---- Encoder interface -----------------------------------------------------

@runtime_checkable
class CohortEncoder(Protocol):
    """A frozen encoder that maps a batch of clips to posterior means.

    Implemented by :class:`TorchCohortEncoder` for the real models and by a
    plain NumPy stand-in in the smoke test. ``conditioned`` says whether the
    model actually consumes ``c`` (CVAE) or ignores it (plain VAE); the
    caller always passes the true cohort id and lets the model decide.
    """

    conditioned: bool

    def encode_mu(self, X: np.ndarray, M: np.ndarray,
                  c: np.ndarray | None) -> np.ndarray:
        """Return posterior means ``mu`` of shape ``(B, d_z)``."""
        ...


class TorchCohortEncoder:
    """Wrap an ``architectures`` VAE / CVAE as a :class:`CohortEncoder`.

    Passes the conditioning id straight into the model's
    ``encode(X, M, c)`` ([CARE-PD §6]); a plain VAE simply ignores it. The
    posterior mean is taken (no reparameterisation) so the latent is
    deterministic, matching "encode every clip to its posterior mean
    μ_φ(x)" ([post-hoc plan §1]).
    """

    def __init__(self, net, config, device: str = "cpu"):
        import torch
        self.torch = torch
        self.net = net.to(device).eval()
        self.config = config
        self.device = device
        self.conditioned = getattr(config, "n_cond", 0) > 0
        self.clip_length = config.clip_length
        self.n_joints = config.n_joints

    def encode_mu(self, X, M, c):
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(X), dtype=torch.float32,
                                device=self.device)
            m = torch.as_tensor(np.asarray(M), dtype=torch.float32,
                                device=self.device)
            cc = None
            if self.conditioned and c is not None:
                cc = torch.as_tensor(np.asarray(c), dtype=torch.long,
                                     device=self.device)
            mu, _ = self.net.encode(x, m, cc)
        return np.asarray(mu.cpu())


def load_encoder(checkpoint_path: str | Path, device: str = "cpu"
                 ) -> tuple[TorchCohortEncoder, str]:
    """Load a checkpoint and wrap it as a named :class:`CohortEncoder`.

    Returns ``(encoder, name)`` where ``name`` is ``"VAE"`` or ``"CVAE"``
    read off the config's conditioning switch — the two core models the
    post-hoc analysis runs on ([post-hoc plan §0]).
    """
    from architectures.analyze import load_checkpoint
    model, config = load_checkpoint(checkpoint_path, device=device)
    name = "CVAE" if getattr(config, "n_cond", 0) > 0 else "VAE"
    if getattr(config, "n_components", 0) > 0:
        raise DeprecationWarning(
            f"{checkpoint_path} is a GM-VAE / GM-CVAE checkpoint, which is "
            "off the active path ([post-hoc plan §0]); the post-hoc analysis "
            "runs on the plain VAE and CVAE only."
        )
    return TorchCohortEncoder(model, config, device=device), name


# ---- Per-walk trajectory metadata ------------------------------------------

@dataclass
class WalkMeta:
    """Metadata for one walk's outer-loop trajectory ([post-hoc plan §4.1])."""
    walk_index: int
    subject: str
    cohort_id: int
    cohort_name: str
    fps: float
    labels: dict = field(default_factory=dict)
    window_starts: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    fog_intervals: list = field(default_factory=list)  # [(start_s, end_s)]

    @property
    def window_times(self) -> np.ndarray:
        """Window centre times in seconds, one per trajectory step."""
        # Centre of a 60-frame window at each start.
        return (self.window_starts + 0.0) / self.fps


# ---- The container everything reads ---------------------------------------

@dataclass
class PosthocData:
    """Encoded clips + metadata + trajectories for the post-hoc battery.

    Attributes:
        models: model names in progression order, e.g. ``["VAE", "CVAE"]``.
        primary: the target model whose latent carries the phenotype claim
            (the CVAE when present, [post-hoc plan §2]).
        clip_mu: ``name -> (N, d_z)`` posterior means, one row per clip.
        clip_id: ``(N,)`` stable clip ids ``0..N-1`` for joins.
        walk_index: ``(N,)`` source-walk index per clip.
        subject / cohort_id / cohort_name: ``(N,)`` clip metadata.
        labels: ``label -> (N,)`` object arrays (``None`` where missing);
            keys are :data:`CATEGORICAL_LABELS`.
        cohorts: ordered cohort vocabulary.
        d_z: latent width.
        traj: ``name -> list`` of ``(T_w, d_z)`` per-walk trajectories.
        traj_meta: per-walk :class:`WalkMeta`, index-aligned with ``traj``.
    """
    models: list[str]
    primary: str
    clip_mu: dict[str, np.ndarray]
    clip_id: np.ndarray
    walk_index: np.ndarray
    subject: np.ndarray
    cohort_id: np.ndarray
    cohort_name: np.ndarray
    labels: dict[str, np.ndarray]
    cohorts: tuple[str, ...]
    d_z: int
    traj: dict[str, list[np.ndarray]] = field(default_factory=dict)
    traj_meta: list[WalkMeta] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.clip_id)

    def latent_set(self, model: str) -> LatentSet:
        """A :class:`LatentSet` view of one model's clip latents.

        ``video_id`` is set to the source-walk index so the toolkit's
        per-video utilities treat walks as videos. ``logvar`` is zero (we
        analyse the deterministic posterior mean).
        """
        mu = self.clip_mu[model]
        return LatentSet(mu=mu, logvar=np.zeros_like(mu),
                         z=mu.copy(), video_id=self.walk_index.copy())

    def label(self, key: str) -> np.ndarray:
        return self.labels[key]

    def has_label(self, key: str) -> bool:
        col = self.labels.get(key)
        if col is None:
            return False
        return any(v is not None for v in col)


# ---- Building the container -------------------------------------------------

def _all_ones_mask(n: int, T: int, J: int) -> np.ndarray:
    return np.ones((n, T, J), dtype=np.float32)


def build_posthoc_data(encoders: dict[str, CohortEncoder],
                       bundle,
                       clip_length: int = 60,
                       stride: int = 30,
                       primary: str | None = None,
                       batch: int = 256,
                       build_trajectories: bool = True,
                       fog_cohort: str = "E-LC",
                       fog_extractor: Callable | None = None,
                       fog_units: str = "auto") -> PosthocData:
    """Encode a bundle into a :class:`PosthocData` for the post-hoc battery.

    Clips are cut once (shared across models) and encoded per model with an
    all-visible mask. The CVAE receives each clip's real cohort id; the
    plain VAE ignores it. Outer-loop trajectories ([post-hoc plan §4.1]) are
    built per walk at the same window/stride.

    Args:
        encoders: ``name -> CohortEncoder`` (e.g. ``{"VAE": ..., "CVAE":
            ...}``). Iteration order sets the progression order.
        bundle: a :class:`architectures.care_pd.CarePDBundle`.
        clip_length: window length T (60 = 2 s at 30 fps, [CARE-PD §8]).
        stride: hop between windows (30 = 50% overlap).
        primary: target model name; defaults to ``"CVAE"`` if present else
            the last encoder.
        batch: encode batch size.
        build_trajectories: also build the per-walk outer-loop latents.
        fog_cohort: cohort whose walks get FoG intervals parsed (E-LC).
        fog_extractor: optional ``(labels, fps, n_frames) -> [(s, e)]``
            override for the FoG annotation format.
        fog_units: unit hint forwarded to :func:`extract_fog_intervals`.
    Returns:
        A populated :class:`PosthocData`.
    """
    from architectures.data import build_clips

    names = list(encoders.keys())
    if primary is None:
        primary = "CVAE" if "CVAE" in names else names[-1]

    videos = bundle.videos
    if not videos:
        raise ValueError("bundle has no videos to encode.")
    J = videos[0].shape[1]

    clips, video_id, _time_index = build_clips(videos, clip_length, stride)
    N = len(clips)
    M = _all_ones_mask(N, clip_length, J)

    subjects_arr = np.asarray(bundle.subjects, dtype=object)
    cohort_names_arr = np.asarray(bundle.cohort_names, dtype=object)
    cohort_ids_arr = np.asarray(bundle.cohort_ids, dtype=np.int64)

    clip_cohort = cohort_ids_arr[video_id]
    clip_cohort_name = cohort_names_arr[video_id]
    clip_subject = subjects_arr[video_id]

    # ---- Encode every clip, per model ----
    clip_mu: dict[str, np.ndarray] = {}
    d_z = None
    for name, enc in encoders.items():
        mus = []
        for i in range(0, N, batch):
            xb = clips[i:i + batch].astype(np.float32)
            mb = M[i:i + batch]
            cb = clip_cohort[i:i + batch]
            mus.append(np.asarray(enc.encode_mu(xb, mb, cb)))
        mu = np.concatenate(mus, axis=0) if mus else np.zeros((0, 0))
        clip_mu[name] = mu
        d_z = mu.shape[1]

    # ---- Per-clip labels (+ cohort control) ----
    labels: dict[str, np.ndarray] = {}
    for key in CATEGORICAL_LABELS:
        if key == "cohort":
            labels[key] = clip_cohort_name.copy()
            continue
        per_walk = np.array(
            [normalise_walk_label(key, lbl) for lbl in bundle.labels],
            dtype=object)
        labels[key] = per_walk[video_id]

    data = PosthocData(
        models=names,
        primary=primary,
        clip_mu=clip_mu,
        clip_id=np.arange(N, dtype=np.int64),
        walk_index=video_id.astype(np.int64),
        subject=clip_subject,
        cohort_id=clip_cohort,
        cohort_name=clip_cohort_name,
        labels=labels,
        cohorts=tuple(bundle.cohorts),
        d_z=int(d_z or 0),
    )

    if build_trajectories:
        _build_trajectories(data, encoders, bundle, clip_length, stride,
                            fog_cohort, fog_extractor, fog_units, batch)
    return data


def _build_trajectories(data: PosthocData, encoders, bundle, clip_length,
                        stride, fog_cohort, fog_extractor, fog_units, batch):
    """Fill ``data.traj`` / ``data.traj_meta`` with per-walk outer loops."""
    from architectures.care_pd import make_windows

    traj: dict[str, list[np.ndarray]] = {name: [] for name in encoders}
    meta: list[WalkMeta] = []

    for wi, pose in enumerate(bundle.videos):
        windows = make_windows(pose, clip_length=clip_length, stride=stride)
        F = pose.shape[0]
        starts = np.arange(0, max(F - clip_length + 1, 0), stride,
                           dtype=np.int64)
        cohort_id = int(bundle.cohort_ids[wi])
        cohort_name = bundle.cohort_names[wi]
        walk_labels = bundle.labels[wi]
        fps = 30.0

        if len(windows) == 0:
            for name in encoders:
                traj[name].append(np.zeros((0, data.d_z), dtype=np.float32))
        else:
            J = windows.shape[2]
            Mw = _all_ones_mask(len(windows), clip_length, J)
            cvec = np.full(len(windows), cohort_id, dtype=np.int64)
            for name, enc in encoders.items():
                mus = []
                for i in range(0, len(windows), batch):
                    mus.append(np.asarray(enc.encode_mu(
                        windows[i:i + batch].astype(np.float32),
                        Mw[i:i + batch], cvec[i:i + batch])))
                traj[name].append(np.concatenate(mus, axis=0))

        fog: list = []
        if cohort_name == fog_cohort:
            if fog_extractor is not None:
                fog = fog_extractor(walk_labels, fps, F)
            else:
                fog = extract_fog_intervals(walk_labels, fps, F,
                                            units=fog_units)
        meta.append(WalkMeta(
            walk_index=wi, subject=str(bundle.subjects[wi]),
            cohort_id=cohort_id, cohort_name=cohort_name, fps=fps,
            labels=walk_labels, window_starts=starts, fog_intervals=fog))

    data.traj = traj
    data.traj_meta = meta
