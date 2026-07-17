"""Shared colour + plotting conventions ([plan §6]).

One fixed colour per class label and per subject, imported everywhere so a
class/subject reads the same across every figure. Colourblind-safe maps for
sequential/heatmap use, and a ``save_fig`` helper (>=150 dpi).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HEATMAP_CMAP = "magma"
SEQUENTIAL_CMAP = "viridis"

# Fixed colour per latent, reused in every q_p-vs-q_m comparison.
LATENT_COLORS = {"q_p": "#C44E52", "q_m": "#4C72B0"}
# One colour per model, reused across baseline comparisons.
MODEL_COLORS = {
    "GAITGen": "#C44E52", "GAITGen_wo_dis": "#DD8452",
    "PlainVAE": "#4C72B0", "PCA": "#8172B3",
}


def _cmap(name):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        import matplotlib.pyplot as plt
        return plt.get_cmap(name)


def class_color(k: int):
    """Stable colour for class id ``k`` (tab10)."""
    return _cmap("tab10")(int(k) % 10)


def cluster_color(k: int):
    return "#B0B0B0" if k < 0 else _cmap("tab20")(int(k) % 20)


def class_colors(values):
    return {v: class_color(i) for i, v in enumerate(sorted(set(values)))}


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
