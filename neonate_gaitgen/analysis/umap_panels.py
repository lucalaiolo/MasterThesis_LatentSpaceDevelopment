"""Paired UMAP of q_p and q_m ([plan §6.2], extends paper Fig. S14).

Panel A (``q_p``) should show clean class separation; Panel B (``q_m``)
should show class overlap — the invariance claim visualised. Both are
mean-pooled over time before UMAP (stated here per the plan) and coloured by
the same class map. Falls back to a 2-component PCA when umap-learn is
absent, labelling the axes accordingly.
"""

from __future__ import annotations

import numpy as np

from . import palette as pal


def _project(z, seed=0):
    """UMAP (or PCA fallback) to 2D; returns (coords, method_name)."""
    z = np.asarray(z, dtype=np.float64)
    try:
        import umap  # type: ignore
        if len(z) >= 10:
            nn = min(30, len(z) - 1)
            r = umap.UMAP(n_components=2, n_neighbors=nn, random_state=seed)
            return r.fit_transform(z), "UMAP"
    except ImportError:
        pass
    c = z - z.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(c, full_matrices=False)
    return c @ Vt[:2].T, "PCA"


def _scatter(ax, coords, c_p, title, method):
    cmap = pal.class_colors(list(c_p))
    for v in sorted(set(c_p)):
        m = np.asarray(c_p) == v
        ax.scatter(coords[m, 0], coords[m, 1], s=10, color=cmap[v],
                   alpha=0.7, linewidths=0, label=f"class {v}")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.set_title(title)
    ax.legend(fontsize=7, title="c_p")


def paired_umap(latents, out_dir, name: str = "GAITGen", seed: int = 0):
    """Two-panel UMAP: q_p (expect separation) and q_m (expect overlap)."""
    plt = pal.import_matplotlib()
    cp = latents.c_p
    coords_p, mp = _project(latents.q_p, seed)
    coords_m, mm = _project(latents.q_m, seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _scatter(axes[0], coords_p, cp,
             f"(A) q_p — expect class separation [{mp}]", mp)
    _scatter(axes[1], coords_m, cp,
             f"(B) q_m — expect class overlap [{mm}]", mm)
    fig.suptitle(f"{name}: pathology (q_p) vs motion (q_m), mean-pooled over "
                 "time, coloured by c_p", y=1.03)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, f"umap_paired_{name}.png")
