"""Visualise a CARE-PD h36m walk as a 3D skeleton animation.

Pick a cohort and an index, get an animation of that walk. Built for the
17-joint H36M skeleton the CARE-PD ``smpl2h36m.py`` pipeline produces
([CARE-PD §8]); the bone connectivity :data:`H36M17_EDGES` is the standard
VideoPose3D / Human3.6M ordering.

Typical Colab use::

    from architectures.care_pd import load_cohorts, build_bundle, TIER1_COHORTS
    from architectures import care_pd_viz

    bundle = build_bundle(load_cohorts("assets/datasets/h36m", TIER1_COHORTS))
    care_pd_viz.show_walk(bundle, "BMCLab", 0)          # displays inline

``show_walk`` returns the ``matplotlib`` animation *and* renders it inline
when IPython is available, so a bare call in a notebook cell just works.
Outside a notebook, grab ``anim`` and ``anim.save("walk.mp4", fps=30)``.
"""

from __future__ import annotations

import numpy as np


# H36M 17-joint skeleton, VideoPose3D ordering:
#   0 pelvis  1 r-hip 2 r-knee 3 r-foot   4 l-hip 5 l-knee 6 l-foot
#   7 spine   8 thorax 9 neck/nose 10 head
#   11 l-shoulder 12 l-elbow 13 l-wrist   14 r-shoulder 15 r-elbow 16 r-wrist
H36M17_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3),            # right leg
    (0, 4), (4, 5), (5, 6),            # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),   # spine -> head
    (8, 11), (11, 12), (12, 13),       # left arm
    (8, 14), (14, 15), (15, 16),       # right arm
)

# Joints on the subject's right / left, for a two-tone skeleton that makes
# gait asymmetry (a core PD phenotype axis) readable at a glance.
_RIGHT_JOINTS = frozenset({1, 2, 3, 14, 15, 16})


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 3D projection
        return plt
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Visualisation needs matplotlib. `pip install matplotlib`."
        ) from e


def _slice_frames(pose: np.ndarray, start: int, length: int | None,
                  stride: int) -> np.ndarray:
    """Select ``length`` frames from ``start`` at the given ``stride``."""
    F = pose.shape[0]
    start = max(0, min(start, F - 1))
    end = F if length is None else min(F, start + length * stride)
    return pose[start:end:max(1, stride)]


def _edge_color(a: int, b: int, right: str, left: str, mid: str) -> str:
    if a in _RIGHT_JOINTS or b in _RIGHT_JOINTS:
        return right
    if a != 0 and b != 0:  # both non-root and not on the right -> left side
        return left
    return mid


def animate_walk(pose: np.ndarray,
                 edges: tuple[tuple[int, int], ...] = H36M17_EDGES,
                 fps: int = 30, start: int = 0, length: int | None = None,
                 stride: int = 1, up_axis: int = 1,
                 elev: float = 12.0, azim: float = -70.0,
                 figsize: tuple[float, float] = (5.0, 5.5),
                 title: str = ""):
    """Animate one walk's 3D skeleton.

    Args:
        pose: (F, J, 3) joint positions for a single walk.
        edges: bone connectivity; defaults to the 17-joint H36M skeleton.
        fps: playback rate (sets the frame interval).
        start: first frame to show.
        length: number of frames to show (``None`` = to the end).
        stride: take every ``stride``-th frame (temporal downsample).
        up_axis: which data axis is anatomical "up" (CARE-PD h36m is Y-up,
            axis 1). It is mapped to the plot's vertical so the figure
            stands upright.
        elev, azim: 3D camera angles.
        figsize: figure size in inches.
        title: figure title.
    Returns:
        ``(fig, anim)``. Close ``fig`` after building ``anim`` to suppress
        the static preview a notebook would otherwise draw.
    """
    plt = _import_matplotlib()
    from matplotlib.animation import FuncAnimation

    pose = np.asarray(_slice_frames(np.asarray(pose, dtype=float),
                                    start, length, stride))
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise ValueError(f"expected (F, J, 3) pose, got {pose.shape}.")
    T = pose.shape[0]

    # Reorder axes so the anatomical up-axis becomes the plot's vertical
    # (matplotlib draws its z-axis upward). Keep a right-handed order.
    horiz = [a for a in (0, 1, 2) if a != up_axis]
    order = [horiz[0], horiz[1], up_axis]
    P = pose[:, :, order]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    mins, maxs = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
    center = 0.5 * (mins + maxs)
    half = 0.55 * (maxs - mins).max()
    for i, setl in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setl(center[i] - half, center[i] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # pragma: no cover
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    scat = ax.scatter(P[0, :, 0], P[0, :, 1], P[0, :, 2], s=14, color="0.3")
    lines = []
    for a, b in edges:
        col = _edge_color(a, b, right="tomato", left="steelblue", mid="0.4")
        (ln,) = ax.plot([P[0, a, 0], P[0, b, 0]],
                        [P[0, a, 1], P[0, b, 1]],
                        [P[0, a, 2], P[0, b, 2]], color=col, linewidth=2.0)
        lines.append(ln)
    caption = ax.set_title(f"{title}\nframe 0 / {T - 1}".strip(), fontsize=10)

    def update(t):
        scat._offsets3d = (P[t, :, 0], P[t, :, 1], P[t, :, 2])
        for (a, b), ln in zip(edges, lines):
            ln.set_data([P[t, a, 0], P[t, b, 0]], [P[t, a, 1], P[t, b, 1]])
            ln.set_3d_properties([P[t, a, 2], P[t, b, 2]])
        caption.set_text(f"{title}\nframe {t} / {T - 1}".strip())
        return [scat, *lines, caption]

    anim = FuncAnimation(fig, update, frames=T,
                         interval=1000.0 / max(fps, 1), blit=False)
    return fig, anim


def _pick_walk(source, cohort: str, index: int):
    """Return ``(pose, meta)`` for the ``index``-th walk of ``cohort``.

    ``source`` may be a :class:`~architectures.care_pd.CarePDBundle` or a
    plain list of :class:`~architectures.care_pd.Walk`.
    """
    # Bundle: videos + parallel cohort_names / subjects.
    if hasattr(source, "videos") and hasattr(source, "cohort_names"):
        idxs = [i for i, c in enumerate(source.cohort_names) if c == cohort]
        if not idxs:
            raise ValueError(
                f"cohort {cohort!r} not in bundle; have "
                f"{sorted(set(source.cohort_names))}.")
        if not -len(idxs) <= index < len(idxs):
            raise IndexError(
                f"index {index} out of range for {cohort!r} "
                f"({len(idxs)} walks).")
        gi = idxs[index]
        subj = source.subjects[gi] if gi < len(source.subjects) else "?"
        return source.videos[gi], {"subject": subj, "n": len(idxs)}

    # List of Walk objects.
    walks = [w for w in source if getattr(w, "cohort", None) == cohort]
    if not walks:
        raise ValueError(f"cohort {cohort!r} not found among the walks.")
    if not -len(walks) <= index < len(walks):
        raise IndexError(
            f"index {index} out of range for {cohort!r} ({len(walks)} walks).")
    w = walks[index]
    return w.pose, {"subject": getattr(w, "subject", "?"),
                    "walk_id": getattr(w, "walk_id", ""), "n": len(walks)}


def show_walk(source, cohort: str, index: int = 0, *, display: bool = True,
              **kwargs):
    """Animate the ``index``-th walk of ``cohort`` and render it inline.

    Args:
        source: a ``CarePDBundle`` or a list of ``Walk`` objects.
        cohort: cohort name to pick from (e.g. ``"BMCLab"``).
        index: which walk of that cohort (0-based; negatives allowed).
        display: when True and IPython is available, show the animation
            inline (returns the ``IPython.display.HTML`` too). When False,
            just return ``(fig, anim)``.
        **kwargs: forwarded to :func:`animate_walk` (``start``, ``length``,
            ``stride``, ``fps``, ``elev``, ``azim``, …).
    Returns:
        ``(fig, anim)``, or ``(fig, anim, html)`` when displayed inline.
    """
    pose, meta = _pick_walk(source, cohort, index)
    subj = meta.get("subject", "?")
    wid = meta.get("walk_id", "")
    tag = f"{cohort} [{index}]  subj={subj}" + (f"  {wid}" if wid else "")
    tag += f"  ({pose.shape[0]} frames)"
    fig, anim = animate_walk(pose, title=tag, **kwargs)

    if display:
        try:
            from IPython.display import HTML, display as _disp
            plt = _import_matplotlib()
            html = HTML(anim.to_jshtml())
            plt.close(fig)          # avoid the duplicate static PNG
            _disp(html)
            return fig, anim, html
        except ImportError:
            pass
    return fig, anim
