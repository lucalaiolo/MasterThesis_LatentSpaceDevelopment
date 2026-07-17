"""Shared plotting conventions for the state-space figures ([guideline §8])."""

from __future__ import annotations

from pathlib import Path

COHORT_COLORS = {"BMCLab": "#4C72B0", "KUL-DT-T": "#DD8452", "E-LC": "#55A868"}
GROUP_COLORS = {"FoG": "#C44E52", "non-FoG": "#8DA0CB",
                "on": "#55A868", "off": "#C44E52"}
HEATMAP_CMAP = "magma"


def _cmap(name):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        import matplotlib.pyplot as plt
        return plt.get_cmap(name)


def state_color(k: int):
    return _cmap("tab10")(int(k) % 10)


def import_matplotlib():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def save_fig(fig, out_dir, name: str, dpi: int = 150) -> Path:
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=max(dpi, 150), bbox_inches="tight")
    plt.close(fig)
    return p
