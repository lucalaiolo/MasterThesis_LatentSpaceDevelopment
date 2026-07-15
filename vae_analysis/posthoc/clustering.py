"""BIC sanity check, post-hoc clustering, and cluster stability.

Covers [post-hoc plan §1] (is the latent worth clustering at all?),
§2 (cluster it three ways so agreement across methods is evidence the
structure is real), and §2.1 (the stability battery that is now the
headline trustworthiness diagnostic, replacing native component
occupancy).

Everything runs on frozen posterior means ``mu`` — we read the latent, we
do not reshape it ([post-hoc plan §8]).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import palette as pal


# ============================================================================
# §1  BIC sanity check
# ============================================================================

def bic_vs_k(data, k_range=range(1, 13), seed: int = 0) -> dict:
    """Fit a full-covariance GMM for each K and each model; return BICs.

    "Before building the full post-hoc stack, confirm the latent is worth
    clustering at all" ([post-hoc plan §1]). A BIC that bottoms out at
    K = 1 or 2 says the latent is essentially unimodal and the phenotype
    claim then rests on the continuous probes (§5); a clear elbow at larger
    K justifies the clustering stack.

    Args:
        data: a :class:`PosthocData`.
        k_range: candidate component counts (1..12 per the plan).
        seed: GMM init seed.
    Returns:
        Dict ``{model: {"bic": {k: value}, "k": argmin_k, "modal": bool}}``.
    """
    from sklearn.mixture import GaussianMixture

    out: dict = {}
    for model in data.models:
        mu = np.asarray(data.clip_mu[model], dtype=np.float64)
        bic: dict[int, float] = {}
        for k in k_range:
            if k < 1 or k > len(mu):
                continue
            gm = GaussianMixture(n_components=k, covariance_type="full",
                                 random_state=seed, reg_covar=1e-5).fit(mu)
            bic[k] = float(gm.bic(mu))
        best_k = min(bic, key=bic.get) if bic else 1
        out[model] = {"bic": bic, "k": int(best_k), "modal": best_k >= 3}
    return out


def plot_bic_vs_k(bic_results: dict, out_dir, name: str = "bic_vs_k.png"):
    """Two BIC-vs-K curves (VAE, CVAE), a marker on each minimum ([§1])."""
    plt = pal.import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for model, res in bic_results.items():
        bic = res["bic"]
        ks = sorted(bic)
        vals = [bic[k] for k in ks]
        color = pal.MODEL_COLORS.get(model, None)
        ax.plot(ks, vals, "o-", color=color, label=model, linewidth=2)
        kbest = res["k"]
        if kbest in bic:
            ax.scatter([kbest], [bic[kbest]], color=color, s=140,
                       marker="*", zorder=5, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("number of mixture components K")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("Is the latent worth clustering?  BIC vs K\n"
                 "(★ = BIC-preferred K per model)")
    ax.legend(title="model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)


# ============================================================================
# §2  Post-hoc clustering (k-means, GMM, HDBSCAN)
# ============================================================================

@dataclass
class ClusteringResult:
    """Post-hoc clustering of one model's latent ([post-hoc plan §2]).

    ``labels`` holds the hard assignment per method, aligned to clip order
    so downstream steps can join on the clip id. ``kmeans_runs`` /
    ``gmm_runs`` are the 20 reseeded labelings kept for the stability
    battery and the consensus matrix.
    """
    model: str
    k: int
    labels: dict[str, np.ndarray]                  # method -> (N,)
    gmm_responsibilities: np.ndarray               # (N, k) soft
    kmeans_runs: list[np.ndarray] = field(default_factory=list)
    gmm_runs: list[np.ndarray] = field(default_factory=list)
    hdbscan_min_cluster_size: int | None = None
    hdbscan_persistence: float = float("nan")


def _kmeans(mu, k, seed):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(mu)


def _gmm_fit(mu, k, seed):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         random_state=seed, reg_covar=1e-5).fit(mu)
    return gm.predict(mu), gm.predict_proba(mu)


def hdbscan_sweep(mu, grid=(15, 25, 40, 60), seed: int = 0):
    """HDBSCAN over a ``min_cluster_size`` grid, picked by persistence.

    HDBSCAN needs no K and flags points as noise (``-1``), which is useful
    for spotting the detached-island clips from the PCA diagnostics
    ([post-hoc plan §2]). The grid entry maximising total cluster
    persistence (and yielding ≥ 2 clusters) is chosen.

    Returns ``(labels, best_min_cluster_size, best_persistence)`` or
    ``(None, None, nan)`` when HDBSCAN is unavailable.
    """
    try:
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            from hdbscan import HDBSCAN  # type: ignore
    except ImportError:
        return None, None, float("nan")

    mu = np.asarray(mu, dtype=np.float64)
    best = (None, None, -np.inf)
    for mcs in grid:
        if mcs >= len(mu):
            continue
        try:
            hb = HDBSCAN(min_cluster_size=int(mcs))
            lab = hb.fit_predict(mu)
        except Exception:
            continue
        n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
        if n_clusters < 2:
            continue
        pers = getattr(hb, "cluster_persistence_", None)
        score = float(np.sum(pers)) if pers is not None else float(n_clusters)
        if score > best[2]:
            best = (lab, int(mcs), score)
    if best[0] is None:
        return None, None, float("nan")
    return best


def cluster_latent(mu: np.ndarray, k: int, n_seeds: int = 20,
                   hdbscan_grid=(15, 25, 40, 60), model: str = "",
                   seed: int = 0) -> ClusteringResult:
    """Cluster ``mu`` with k-means, GMM, and HDBSCAN ([post-hoc plan §2]).

    k-means and GMM are fit at the BIC-preferred ``k``; the reseeded runs
    (``n_seeds`` of each) are retained for §2.1. Soft GMM responsibilities
    are stored, not just hard labels.
    """
    mu = np.asarray(mu, dtype=np.float64)
    rng = np.random.default_rng(seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]

    kmeans_runs = [_kmeans(mu, k, s) for s in seeds]
    gmm_runs, gmm_probs = [], None
    for i, s in enumerate(seeds):
        lab, prob = _gmm_fit(mu, k, s)
        gmm_runs.append(lab)
        if i == 0:
            gmm_probs = prob

    labels = {"kmeans": kmeans_runs[0], "gmm": gmm_runs[0]}
    hb_labels, hb_mcs, hb_pers = hdbscan_sweep(mu, hdbscan_grid, seed=seed)
    if hb_labels is not None:
        labels["hdbscan"] = hb_labels

    return ClusteringResult(
        model=model, k=k, labels=labels,
        gmm_responsibilities=gmm_probs,
        kmeans_runs=kmeans_runs, gmm_runs=gmm_runs,
        hdbscan_min_cluster_size=hb_mcs, hdbscan_persistence=hb_pers)


def cluster_k_robustness(mu: np.ndarray, k: int, seed: int = 0) -> dict:
    """k-means labels at K-1, K, K+1 ([post-hoc plan §2], robustness)."""
    out = {}
    for kk in (k - 1, k, k + 1):
        if kk >= 2 and kk <= len(mu):
            out[kk] = _kmeans(np.asarray(mu, np.float64), kk, seed)
    return out


# ============================================================================
# §2.1  Cluster stability — the headline trustworthiness diagnostic
# ============================================================================

@dataclass
class StabilityResult:
    """Stability battery for one model's clustering ([post-hoc plan §2.1])."""
    model: str
    between_run_ari: dict[str, np.ndarray]   # method -> pairwise ARIs
    subsample_ari: dict[str, np.ndarray]     # method -> ARIs vs full labels
    bootstrap_ari: dict[str, np.ndarray]     # method -> ARIs vs full labels
    cross_method_ari: dict[str, float]       # "kmeans_vs_gmm" etc.
    consensus: np.ndarray                    # (n_sub, n_sub) co-assignment
    consensus_labels: np.ndarray             # final labels of the subsample
    n_consensus: int

    def summary(self) -> dict:
        """Scalar digest for the report and results.json."""
        def _mean(d):
            return {m: (float(np.mean(v)) if len(v) else float("nan"))
                    for m, v in d.items()}
        return {
            "between_run_ari_mean": _mean(self.between_run_ari),
            "between_run_ari_std": {m: (float(np.std(v)) if len(v) else
                                        float("nan"))
                                    for m, v in self.between_run_ari.items()},
            "subsample_ari_mean": _mean(self.subsample_ari),
            "bootstrap_ari_mean": _mean(self.bootstrap_ari),
            "cross_method_ari": self.cross_method_ari,
        }


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def _pairwise_ari(runs: list[np.ndarray]) -> np.ndarray:
    out = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            out.append(_ari(runs[i], runs[j]))
    return np.asarray(out, dtype=np.float64)


def cluster_stability(mu: np.ndarray, result: ClusteringResult,
                      n_subsample: int = 10, subsample_frac: float = 0.8,
                      n_bootstrap: int = 10, consensus_max: int = 400,
                      seed: int = 0) -> StabilityResult:
    """Reseed / subsample / bootstrap / cross-method stability ([§2.1]).

    A cluster solution with high between-run and cross-method ARI carries
    the phenotype claim; low stability is itself a finding — the structure
    is continuous, not modal — and is reported as such, not hidden.

    Args:
        mu: ``(N, d_z)`` latent the clustering was fit on.
        result: the :class:`ClusteringResult` (its reseeded runs are reused).
        n_subsample: number of 80% subsamples.
        subsample_frac: fraction kept per subsample.
        n_bootstrap: number of with-replacement resamples.
        consensus_max: cap on points entering the N×N consensus matrix.
        seed: RNG seed.
    """
    mu = np.asarray(mu, dtype=np.float64)
    N, k = len(mu), result.k
    rng = np.random.default_rng(seed + 101)

    # --- Reseed: pairwise ARI between the 20 runs of each method. ---
    between = {
        "kmeans": _pairwise_ari(result.kmeans_runs),
        "gmm": _pairwise_ari(result.gmm_runs),
    }

    # --- Subsample: cluster 80% subsamples, ARI vs full labels on shared. ---
    def _subsample_scores(full_labels, cluster_fn):
        scores = []
        for _ in range(n_subsample):
            m = max(k + 1, int(round(N * subsample_frac)))
            idx = rng.choice(N, size=min(m, N), replace=False)
            lab = cluster_fn(mu[idx])
            scores.append(_ari(full_labels[idx], lab))
        return np.asarray(scores)

    def _bootstrap_scores(full_labels, cluster_fn):
        scores = []
        for _ in range(n_bootstrap):
            idx = rng.choice(N, size=N, replace=True)
            uniq = np.unique(idx)
            lab = cluster_fn(mu[uniq])
            scores.append(_ari(full_labels[uniq], lab))
        return np.asarray(scores)

    km_fn = lambda x: _kmeans(x, k, seed)
    gm_fn = lambda x: _gmm_fit(x, k, seed)[0]
    subsample = {
        "kmeans": _subsample_scores(result.labels["kmeans"], km_fn),
        "gmm": _subsample_scores(result.labels["gmm"], gm_fn),
    }
    bootstrap = {
        "kmeans": _bootstrap_scores(result.labels["kmeans"], km_fn),
        "gmm": _bootstrap_scores(result.labels["gmm"], gm_fn),
    }

    # --- Cross-method ARI on the full data. ---
    cross = {}
    methods = list(result.labels.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            cross[f"{a}_vs_{b}"] = _ari(result.labels[a], result.labels[b])

    # --- Consensus (co-assignment) matrix from the reseeded k-means runs. ---
    n_sub = min(consensus_max, N)
    sub_idx = rng.choice(N, size=n_sub, replace=False)
    runs = result.kmeans_runs
    co = np.zeros((n_sub, n_sub), dtype=np.float64)
    for lab in runs:
        s = lab[sub_idx]
        co += (s[:, None] == s[None, :]).astype(np.float64)
    co /= max(len(runs), 1)
    order = np.argsort(result.labels["kmeans"][sub_idx])
    co = co[np.ix_(order, order)]
    consensus_labels = result.labels["kmeans"][sub_idx][order]

    return StabilityResult(
        model=result.model, between_run_ari=between,
        subsample_ari=subsample, bootstrap_ari=bootstrap,
        cross_method_ari=cross, consensus=co,
        consensus_labels=consensus_labels, n_consensus=n_sub)


def plot_stability_ari(stabilities: dict, out_dir,
                       name: str = "stability_ari_distributions.png"):
    """Box/violin of between-run ARI for k-means & GMM, per model ([§2.1]).

    ``stabilities`` maps model name -> :class:`StabilityResult`. Panels are
    coloured by method; models sit side by side so the CVAE-vs-VAE contrast
    is visible.
    """
    plt = pal.import_matplotlib()
    models = list(stabilities.keys())
    method_colors = {"kmeans": "#4C72B0", "gmm": "#DD8452"}
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.2),
                             squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models):
        st = stabilities[model]
        data = [st.between_run_ari["kmeans"], st.between_run_ari["gmm"]]
        parts = ax.violinplot(data, showmeans=True, showextrema=True)
        for pc, m in zip(parts["bodies"], ("kmeans", "gmm")):
            pc.set_facecolor(method_colors[m])
            pc.set_alpha(0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["k-means", "GMM"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("between-run ARI")
        ax.set_title(f"{model} latent\ncluster reproducibility")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Cluster stability — between-run ARI (higher = more "
                 "trustworthy clusters)", y=1.02)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)


def plot_consensus(stability: StabilityResult, out_dir, name: str):
    """Consensus (co-assignment) heatmap ordered by cluster ([§2.1]).

    A clean block-diagonal (sequential ``magma``) means stable clusters; a
    smeared matrix means they are not.
    """
    plt = pal.import_matplotlib()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(stability.consensus, cmap=pal.HEATMAP_CMAP,
                   vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xlabel("clip (ordered by cluster)")
    ax.set_ylabel("clip (ordered by cluster)")
    ax.set_title(f"{stability.model} consensus matrix "
                 f"(n={stability.n_consensus})\n"
                 "fraction of reseeds co-assigning each pair")
    fig.colorbar(im, ax=ax, label="co-assignment fraction")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)
