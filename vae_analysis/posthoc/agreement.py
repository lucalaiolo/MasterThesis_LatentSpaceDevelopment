"""Cluster–label agreement, subject composition, within-severity structure.

Covers [post-hoc plan §2.2] (do the post-hoc clusters agree with the
clinical labels, and — the key contrast — with cohort?), §2.3 (are the
clusters phenotypes or just individuals?), and §3 (the headline result:
more than one stable phenotype at a *fixed* severity grade).

The cluster–label scoring reuses ``architectures.metrics`` so the numbers
match the training-side evaluators.
"""

from __future__ import annotations

import numpy as np

from . import palette as pal
from . import clustering as clust
from .data import CATEGORICAL_LABELS

# Human-readable column headers for the agreement table / panels.
LABEL_TITLES = {
    "updrs_gait": "UPDRS_GAIT",
    "freezer": "freezer",
    "med": "medication",
    "cohort": "cohort (control)",
}


# ============================================================================
# §2.2  Cluster–label agreement
# ============================================================================

def agreement_table(data, model: str, result: clust.ClusteringResult) -> dict:
    """ARI and NMI of each clustering against each label ([post-hoc plan §2.2]).

    The key contrast is CVAE-latent clusters agreeing with phenotype labels
    (UPDRS / freezer / medication) while **not** agreeing with cohort — on
    the plain VAE latent cohort agreement may instead be high.

    Returns a dict ``{method: {label: {ari, nmi, n}}}`` plus a rendered
    markdown table under ``"markdown"``.
    """
    from architectures.metrics import cluster_label_agreement

    methods = list(result.labels.keys())
    labels_present = [k for k in CATEGORICAL_LABELS if data.has_label(k)]

    scores: dict = {}
    for method in methods:
        assign = result.labels[method]
        scores[method] = {}
        for key in labels_present:
            # HDBSCAN noise (-1) should not be scored as a real cluster.
            a = assign
            lab = data.label(key)
            if method == "hdbscan":
                keep = assign >= 0
                a = assign[keep]
                lab = lab[keep]
            scores[method][key] = cluster_label_agreement(a, lab)

    scores["markdown"] = _agreement_markdown(scores, methods, labels_present)
    scores["_labels"] = labels_present
    return scores


def _agreement_markdown(scores, methods, labels_present) -> str:
    head = "| method | " + " | ".join(LABEL_TITLES.get(k, k)
                                       for k in labels_present) + " |"
    sep = "|" + "---|" * (len(labels_present) + 1)
    rows = [head, sep]
    for m in methods:
        cells = [m]
        for key in labels_present:
            s = scores[m][key]
            ari, nmi = s.get("ari", float("nan")), s.get("nmi", float("nan"))
            cells.append(f"{ari:.2f} ({nmi:.2f})")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _null(v) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _scatter_categorical(ax, coords, values, color_fn, title, fig,
                         legend_title):
    """Scatter coloured by a categorical column, with a legend."""
    vals = np.asarray(values, dtype=object)
    present = [v for v in pal._ordered_unique(vals)]
    # Draw nulls first, faint grey, so real categories sit on top.
    null_mask = np.array([_null(v) for v in vals])
    if null_mask.any():
        ax.scatter(coords[null_mask, 0], coords[null_mask, 1], s=6,
                   color="#DDDDDD", alpha=0.5, linewidths=0)
    for v in present:
        m = np.array([x == v for x in vals]) & ~null_mask
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=8, color=color_fn(v),
                   alpha=0.75, linewidths=0, label=str(v))
    ax.set_title(title)
    ax.legend(title=legend_title, fontsize=7, markerscale=1.5,
              loc="best", framealpha=0.7)


def _scatter_sequential(ax, coords, values, title, fig, cbar_label):
    """Scatter coloured by an ordinal column, with a colourbar (UPDRS)."""
    vals = np.array([np.nan if _null(v) else float(v) for v in values])
    ok = ~np.isnan(vals)
    if (~ok).any():
        ax.scatter(coords[~ok, 0], coords[~ok, 1], s=6, color="#DDDDDD",
                   alpha=0.5, linewidths=0)
    sc = ax.scatter(coords[ok, 0], coords[ok, 1], s=9, c=vals[ok],
                    cmap=pal.SEQUENTIAL_CMAP, alpha=0.85, linewidths=0)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)


def plot_latent_panels(data, model: str, result: clust.ClusteringResult,
                       out_dir, kind: str = "pca", seed: int = 0):
    """One scatter panel per colouring: cluster, UPDRS, freezer, med, cohort.

    ``kind`` is ``"pca"`` (the honest linear view, axes labelled with
    variance explained) or ``"umap"`` (separates modes more clearly when
    they exist). Same layout as the existing PCA diagnostics
    ([post-hoc plan §2.2]).
    """
    plt = pal.import_matplotlib()
    mu = data.clip_mu[model]

    if kind == "umap":
        coords = pal.umap_project(mu, seed=seed)
        if coords is None:
            return None
        ax_lab = ("UMAP-1", "UMAP-2")
        proj_name = "UMAP"
    else:
        coords, var = pal.pca_project(mu, 2)
        ax_lab = (f"PC1 ({var[0] * 100:.0f}%)", f"PC2 ({var[1] * 100:.0f}%)")
        proj_name = "PCA"

    panels = [("cluster", "cluster")]
    for key in ("updrs_gait", "freezer", "med", "cohort"):
        if data.has_label(key):
            panels.append((key, key))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), squeeze=False)
    for ax, (key, _) in zip(axes[0], panels):
        ax.set_xlabel(ax_lab[0])
        ax.set_ylabel(ax_lab[1])
        if key == "cluster":
            labels = result.labels[data_primary_method(result)]
            _scatter_categorical(
                ax, coords, labels, pal.cluster_color,
                "coloured by cluster", fig, "cluster")
        elif key == "updrs_gait":
            _scatter_sequential(ax, coords, data.label(key),
                                "coloured by UPDRS", fig, "UPDRS_GAIT")
        elif key == "cohort":
            _scatter_categorical(ax, coords, data.label(key),
                                 pal.cohort_color, "coloured by cohort",
                                 fig, "cohort")
        else:
            cmap, _ = pal.label_colors(key, data.label(key))
            _scatter_categorical(
                ax, coords, data.label(key),
                lambda v, _m=cmap: _m.get(v, "#333333"),
                f"coloured by {LABEL_TITLES.get(key, key)}", fig, key)
    fig.suptitle(f"{model} latent — {proj_name} of the aggregate posterior",
                 y=1.03)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, f"{kind}_panels_{model}.png")


def data_primary_method(result: clust.ClusteringResult) -> str:
    """Which clustering drives the 'coloured by cluster' panel: GMM if present."""
    return "gmm" if "gmm" in result.labels else next(iter(result.labels))


# ============================================================================
# §2.3  Per-subject composition (identity check)
# ============================================================================

def subject_composition(data, model: str, cluster_labels: np.ndarray,
                        pure_threshold: float = 0.8) -> dict:
    """Build Π[v, k] = fraction of subject v's clips in cluster k ([§2.3]).

    A near-diagonal Π (each subject dominated by one cluster) means we
    recovered individuals, not phenotypes — which would gut the claim.

    Returns the matrix, the subject order (by dominant cluster), the cluster
    ids, and the fraction of subjects that are "pure" (> threshold in one
    cluster).
    """
    labels = np.asarray(cluster_labels)
    subjects = data.subject
    # Ignore HDBSCAN noise when composing.
    valid = labels >= 0 if (labels < 0).any() else np.ones(len(labels), bool)
    ks = np.array(sorted(set(labels[valid].tolist())))
    subj_ids = list(dict.fromkeys(subjects.tolist()))

    Pi = np.zeros((len(subj_ids), len(ks)), dtype=np.float64)
    for i, s in enumerate(subj_ids):
        m = (subjects == s) & valid
        total = int(m.sum())
        if total == 0:
            continue
        for j, k in enumerate(ks):
            Pi[i, j] = np.mean(labels[m] == k)

    dominant = Pi.argmax(axis=1) if Pi.size else np.zeros(len(subj_ids), int)
    order = np.argsort(dominant)
    pure = float(np.mean(Pi.max(axis=1) > pure_threshold)) if Pi.size else \
        float("nan")

    return {"Pi": Pi[order], "subjects": [subj_ids[i] for i in order],
            "clusters": ks, "pure_fraction": pure,
            "pure_threshold": pure_threshold}


def plot_subject_composition(comp: dict, model: str, out_dir,
                             name: str | None = None):
    """Subject × cluster composition heatmap ([post-hoc plan §2.3])."""
    plt = pal.import_matplotlib()
    Pi = comp["Pi"]
    fig, ax = plt.subplots(figsize=(max(4.5, 0.5 * Pi.shape[1] + 2),
                                    max(4.0, 0.16 * Pi.shape[0] + 1)))
    im = ax.imshow(Pi, cmap=pal.HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("cluster")
    ax.set_ylabel("subject (ordered by dominant cluster)")
    ax.set_xticks(np.arange(len(comp["clusters"])))
    ax.set_xticklabels([str(k) for k in comp["clusters"]])
    pure_pct = comp["pure_fraction"] * 100
    ax.set_title(f"{model} — subject composition Π[v,k]\n"
                 f"{pure_pct:.0f}% of subjects are 'pure' "
                 f"(>{comp['pure_threshold']*100:.0f}% in one cluster)")
    fig.colorbar(im, ax=ax, label="fraction of subject's clips")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name or f"subject_composition_heatmap_{model}.png")


# ============================================================================
# §3  Within-severity substructure
# ============================================================================

def within_severity(data, model: str, n_seeds: int = 20,
                    min_level_clips: int = 40, seed: int = 0) -> dict:
    """Cluster the latent within each fixed UPDRS_GAIT level ([post-hoc plan §3]).

    "More than one stable phenotype within a single severity grade is the
    headline finding." For every severity level with enough labelled clips,
    the latent is re-clustered post hoc (BIC-picked K) and its stability
    measured, so a *stable* split at fixed severity is the claim, not just a
    numeric one.

    Returns ``{level: {...}}`` with per-level coords, cluster labels, K,
    between-run ARI, and clip count.
    """
    updrs = data.label("updrs_gait")
    if updrs is None:
        return {}
    levels = sorted({int(v) for v in updrs if not _null(v)})
    mu_all = data.clip_mu[model]

    out: dict = {}
    for lvl in levels:
        idx = np.array([i for i, v in enumerate(updrs)
                        if not _null(v) and int(v) == lvl])
        if len(idx) < min_level_clips:
            out[lvl] = {"n": int(len(idx)), "skipped": True}
            continue
        mu = mu_all[idx]
        # BIC-pick K within this level (2..6, capped by sample size).
        from sklearn.mixture import GaussianMixture
        best_k, best_bic = 1, np.inf
        for k in range(1, min(7, len(idx))):
            gm = GaussianMixture(n_components=k, covariance_type="full",
                                 random_state=seed, reg_covar=1e-5).fit(mu)
            b = gm.bic(mu)
            if b < best_bic:
                best_k, best_bic = k, b
        k = max(best_k, 2)  # we ask "is there >1 sub-cluster", so cluster >=2
        res = clust.cluster_latent(mu, k, n_seeds=n_seeds, model=model,
                                   seed=seed)
        st = clust.cluster_stability(mu, res, n_subsample=5, n_bootstrap=5,
                                     consensus_max=min(200, len(idx)),
                                     seed=seed)
        coords, var = pal.pca_project(mu, 2)
        out[lvl] = {
            "n": int(len(idx)),
            "k_bic": int(best_k),
            "labels": res.labels["gmm"],
            "coords": coords, "var": var,
            "between_run_ari_kmeans": float(np.mean(st.between_run_ari["kmeans"]))
            if len(st.between_run_ari["kmeans"]) else float("nan"),
            "n_stable_subclusters": int(best_k) if best_k >= 2 and
            np.mean(st.between_run_ari["kmeans"]) > 0.5 else 1,
            "skipped": False,
        }
    return out


def plot_within_severity(ws: dict, model: str, out_dir,
                         name: str = "within_severity.png"):
    """Small multiple: one PCA panel per severity, coloured by sub-cluster."""
    plt = pal.import_matplotlib()
    levels = [lvl for lvl, r in ws.items() if not r.get("skipped")]
    if not levels:
        return None
    n = len(levels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for ax, lvl in zip(axes[0], levels):
        r = ws[lvl]
        coords, labels = r["coords"], r["labels"]
        for k in sorted(set(labels.tolist())):
            m = labels == k
            ax.scatter(coords[m, 0], coords[m, 1], s=10,
                       color=pal.cluster_color(k), alpha=0.75,
                       linewidths=0, label=f"sub {k}")
        var = r["var"]
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
        ax.set_title(f"UPDRS = {lvl}  (n={r['n']})\n"
                     f"K_BIC={r['k_bic']}, ARI={r['between_run_ari_kmeans']:.2f}")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{model} — within-severity substructure "
                 "(phenotypes at fixed UPDRS)", y=1.03)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)
