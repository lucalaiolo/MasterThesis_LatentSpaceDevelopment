"""Shared colour and plotting conventions ([post-hoc plan §6]).

One place fixes every colour so the same category reads the same across
figures: a cohort keeps its hue, an HMM state keeps its hue, and the two
models keep one colour each (VAE vs CVAE) in every comparison. Categorical
variables get a well-separated qualitative palette; UPDRS (ordinal) gets a
sequential colourbar; heatmaps get a sequential map; signed quantities get
a diverging map centred at zero. All choices are colourblind-safe
(``viridis`` / ``cividis`` / ``Set2`` / ``tab10``), and every saved figure
goes out at ≥150 dpi.

Import the maps from here everywhere rather than re-picking colours::

    from vae_analysis.posthoc import palette as pal
    ax.scatter(x, y, color=pal.cohort_color("E-LC"))
    pal.save_fig(fig, out_dir, "bic_vs_k.png")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Cohort vocabulary in a fixed order, so a cohort's colour never depends on
# which subset a given figure happens to include ([post-hoc plan §6]).
from architectures.care_pd import ALL_COHORTS


# ---- Colour maps (matplotlib imported lazily) ------------------------------

def _cmap(name: str):
    # matplotlib.cm.get_cmap was removed in 3.9; colormaps registry is the
    # modern accessor (3.6+), with a pyplot fallback for older versions.
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        import matplotlib.pyplot as plt
        return plt.get_cmap(name)


# Sequential map for heatmaps / consensus / transition matrices.
HEATMAP_CMAP = "magma"
# Sequential map for ordinal UPDRS severity.
SEQUENTIAL_CMAP = "viridis"
# Diverging map for signed quantities centred at zero.
DIVERGING_CMAP = "coolwarm"

# One colour per model, reused in every model comparison.
MODEL_COLORS: dict[str, str] = {
    "VAE": "#4C72B0",    # muted blue  — plain baseline
    "CVAE": "#DD8452",   # muted orange — cohort-conditioned
    "AVAE": "#55A868",   # muted green  — adversarial (gradient-reversal)
}

# Freezer / medication get fixed, well-separated hues.
FREEZER_COLORS: dict[str, str] = {
    "freezer": "#C44E52",       # red-ish
    "non-freezer": "#55A868",   # green-ish
}
MED_COLORS: dict[str, str] = {
    "ON": "#4C72B0",
    "OFF": "#8172B3",
}

# Points HDBSCAN flags as noise.
NOISE_COLOR = "#B0B0B0"


def _tab10(i: int) -> tuple:
    return _cmap("tab10")(i % 10)


# Stable cohort -> colour, keyed off the fixed ALL_COHORTS order. Built
# lazily on first use so importing this module does not pull in matplotlib.
_COHORT_COLOR: dict = {}


def _cohort_color_map() -> dict:
    if not _COHORT_COLOR:
        _COHORT_COLOR.update({name: _tab10(i)
                              for i, name in enumerate(ALL_COHORTS)})
    return _COHORT_COLOR


def cohort_color(name: str):
    """Stable colour for a cohort, the same in every figure."""
    cmap = _cohort_color_map()
    if name in cmap:
        return cmap[name]
    # Unknown cohort — hash into the tab10 wheel deterministically.
    return _tab10(abs(hash(name)) % 10)


def state_color(k: int):
    """Stable colour for HMM state ``k`` (or any small integer category)."""
    return _tab10(int(k))


def cluster_color(k: int):
    """Colour for cluster index ``k``; ``-1`` (HDBSCAN noise) is grey."""
    if k < 0:
        return NOISE_COLOR
    # tab20 gives more distinct hues when there are many clusters.
    return _cmap("tab20")(int(k) % 20)


def categorical_colors(categories) -> dict:
    """Map an ordered list of category values to well-separated colours.

    Routes known label domains to their fixed maps (cohort, freezer, med)
    and falls back to ``tab10`` / ``tab20`` otherwise, so a category keeps a
    stable colour across panels.
    """
    cats = list(categories)
    cohort_map = _cohort_color_map()
    out: dict = {}
    for i, c in enumerate(cats):
        if c in FREEZER_COLORS:
            out[c] = FREEZER_COLORS[c]
        elif c in MED_COLORS:
            out[c] = MED_COLORS[c]
        elif c in cohort_map:
            out[c] = cohort_map[c]
        else:
            out[c] = _cmap("tab20")(i % 20) if len(cats) > 10 else _tab10(i)
    return out


def label_colors(label_key: str, values) -> tuple[dict | None, bool]:
    """Colouring for one label column.

    Returns ``(color_map, is_sequential)``. UPDRS is ordinal → the caller
    should use a sequential colourbar (``color_map`` is ``None``, second
    element ``True``). Every other label is categorical → ``color_map``
    maps each present value to a fixed colour.
    """
    if label_key == "updrs_gait":
        return None, True
    present = _ordered_unique(values)
    return categorical_colors(present), False


def _ordered_unique(values) -> list:
    """Distinct non-null values, cohorts/UPDRS in natural order else sorted."""
    vals = [v for v in values if v is not None and not _is_nan(v)]
    uniq = list(dict.fromkeys(vals))
    try:
        return sorted(uniq)
    except TypeError:
        return uniq


def _is_nan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


# ---- Figure saving ---------------------------------------------------------

def save_fig(fig, out_dir, name: str, dpi: int = 150) -> Path:
    """Save ``fig`` to ``out_dir/name`` at ≥150 dpi and close it.

    Returns the path written ([post-hoc plan §6]: every figure saved at
    ≥150 dpi to ``outputs/posthoc/``).
    """
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=max(dpi, 150), bbox_inches="tight")
    plt.close(fig)
    return p


def import_matplotlib():
    """Return ``pyplot`` with a non-interactive backend selected."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


# ---- PCA helper (shared axis labelling) -----------------------------------

def pca_project(mu: np.ndarray, n_components: int = 2):
    """Project ``mu`` onto its top principal components.

    Returns ``(coords, variance_explained)`` where ``variance_explained``
    is the per-PC fraction, used to label axes ("PC1 (23%)") as the plan
    requires for every PCA panel ([post-hoc plan §6]).
    """
    mu = np.asarray(mu, dtype=np.float64)
    if len(mu) == 0:
        return np.zeros((0, n_components)), np.zeros(n_components)
    centered = mu - mu.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    coords = centered @ Vt[:k].T
    total = float((S ** 2).sum()) or 1.0
    var = (S[:k] ** 2) / total
    if coords.shape[1] < n_components:  # pad if d_z < n_components
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])
        var = np.concatenate([var, np.zeros(n_components - len(var))])
    return coords, var


def umap_project(mu: np.ndarray, seed: int = 0, n_neighbors: int = 30):
    """UMAP embedding of ``mu``, or ``None`` if umap-learn is unavailable.

    UMAP separates modes more clearly than PCA when they exist; PCA stays
    as the honest linear view ([post-hoc plan §2.2]).
    """
    try:
        import umap  # type: ignore
    except ImportError:
        return None
    mu = np.asarray(mu, dtype=np.float64)
    if len(mu) < max(n_neighbors, 5):
        return None
    reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, len(mu) - 1),
                        random_state=seed)
    return reducer.fit_transform(mu)
