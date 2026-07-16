"""Post-hoc clustering on the latents ([plan §6.6, §6.7]).

k-means, GMM (BIC-selected K), and HDBSCAN on a mean-pooled latent, with
between-run and between-method ARI (stability), a subject-composition check
(rule out clusters = individuals), and ARI/NMI against ``c_p``.

Reading direction differs by latent ([plan §6.6/§6.7]): on ``q_m`` we want
label agreement **low** (invariance), on ``q_p`` we want it **high**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import palette as pal


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def bic_k(z: np.ndarray, k_range=range(1, 11), seed: int = 0) -> int:
    from sklearn.mixture import GaussianMixture
    z = np.asarray(z, dtype=np.float64)
    best_k, best = 1, np.inf
    for k in k_range:
        if k > len(z):
            break
        b = GaussianMixture(k, covariance_type="full", random_state=seed,
                            reg_covar=1e-5).fit(z).bic(z)
        if b < best:
            best, best_k = b, k
    return best_k


@dataclass
class ClusterResult:
    name: str
    k: int
    labels: dict = field(default_factory=dict)
    kmeans_runs: list = field(default_factory=list)
    gmm_runs: list = field(default_factory=list)


def cluster(z: np.ndarray, k: int, n_seeds: int = 10, seed: int = 0,
            name: str = "") -> ClusterResult:
    """k-means, GMM, HDBSCAN on ``z`` ([plan §6.6/§6.7])."""
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    z = np.asarray(z, dtype=np.float64)
    rng = np.random.default_rng(seed)
    seeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n_seeds)]
    km = [KMeans(k, random_state=s, n_init=10).fit_predict(z) for s in seeds]
    gm = [GaussianMixture(k, covariance_type="full", random_state=s,
                          reg_covar=1e-5).fit(z).predict(z) for s in seeds]
    labels = {"kmeans": km[0], "gmm": gm[0]}
    hb = _hdbscan(z)
    if hb is not None:
        labels["hdbscan"] = hb
    return ClusterResult(name=name, k=k, labels=labels,
                         kmeans_runs=km, gmm_runs=gm)


def _hdbscan(z):
    try:
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            from hdbscan import HDBSCAN  # type: ignore
    except ImportError:
        return None
    mcs = max(15, len(z) // 50)
    if mcs >= len(z):
        return None
    return HDBSCAN(min_cluster_size=int(mcs)).fit_predict(np.asarray(z, float))


def stability(result: ClusterResult) -> dict:
    def _pairwise(runs):
        out = [_ari(runs[i], runs[j]) for i in range(len(runs))
               for j in range(i + 1, len(runs))]
        return float(np.mean(out)) if out else float("nan")
    cross = {}
    methods = list(result.labels)
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            m_a = result.labels[a][result.labels[a] >= 0] if a == "hdbscan" else result.labels[a]
            # cross-method ARI on full labels (noise kept as its own group).
            cross[f"{a}_vs_{b}"] = _ari(result.labels[a], result.labels[b])
    return {"between_run_kmeans": _pairwise(result.kmeans_runs),
            "between_run_gmm": _pairwise(result.gmm_runs),
            "cross_method": cross}


def label_agreement(labels: np.ndarray, c_p: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    keep = labels >= 0 if (labels < 0).any() else np.ones(len(labels), bool)
    a, y = labels[keep], np.asarray(c_p)[keep]
    return {"ari": float(adjusted_rand_score(y, a)),
            "nmi": float(normalized_mutual_info_score(y, a)), "n": int(keep.sum())}


def subject_composition(labels: np.ndarray, subjects: np.ndarray,
                        pure_threshold: float = 0.8) -> dict:
    labels = np.asarray(labels)
    valid = labels >= 0 if (labels < 0).any() else np.ones(len(labels), bool)
    ks = np.array(sorted(set(labels[valid].tolist())))
    subj_ids = list(dict.fromkeys(np.asarray(subjects).tolist()))
    Pi = np.zeros((len(subj_ids), len(ks)))
    for i, s in enumerate(subj_ids):
        m = (np.asarray(subjects) == s) & valid
        if m.sum():
            for j, k in enumerate(ks):
                Pi[i, j] = np.mean(labels[m] == k)
    pure = float(np.mean(Pi.max(1) > pure_threshold)) if Pi.size else float("nan")
    order = np.argsort(Pi.argmax(1)) if Pi.size else np.arange(len(subj_ids))
    return {"Pi": Pi[order], "pure_fraction": pure, "clusters": ks,
            "subjects": [subj_ids[i] for i in order]}


def analyze(z: np.ndarray, c_p: np.ndarray, subjects: np.ndarray,
            name: str, seed: int = 0) -> dict:
    """Full clustering analysis of one latent: k, stability, agreement, comp."""
    k = max(bic_k(z, seed=seed), 2)
    res = cluster(z, k, name=name, seed=seed)
    prim = "gmm" if "gmm" in res.labels else "kmeans"
    return {"k": k, "stability": stability(res),
            "agreement": {m: label_agreement(res.labels[m], c_p)
                          for m in res.labels},
            "composition": subject_composition(res.labels[prim], subjects),
            "_result": res}


def plot_subject_composition(comp: dict, name: str, out_dir):
    plt = pal.import_matplotlib()
    Pi = comp["Pi"]
    fig, ax = plt.subplots(figsize=(max(4.5, 0.5 * Pi.shape[1] + 2),
                                    max(4, 0.16 * Pi.shape[0] + 1)))
    im = ax.imshow(Pi, cmap=pal.HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("cluster")
    ax.set_ylabel("subject (ordered by dominant cluster)")
    ax.set_title(f"{name} — subject composition\n"
                 f"{comp['pure_fraction']*100:.0f}% of subjects >80% one cluster")
    fig.colorbar(im, ax=ax, label="fraction of subject's clips")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, f"subject_composition_{name}.png")
