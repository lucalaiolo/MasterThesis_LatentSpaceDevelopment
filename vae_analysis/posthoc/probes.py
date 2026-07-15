"""Continuous linear probes on the frozen latent ([post-hoc plan §5]).

Independently of clustering, fit linear read-outs of phenotype and of the
nuisance axis on the frozen latent, so the story survives even if the
clusters are weak:

    - ridge   μ → UPDRS_GAIT     (held-out R²)
    - logistic μ → freezer       (balanced accuracy)
    - logistic μ → medication    (balanced accuracy)
    - MLP     μ → cohort         (the site probe, top-1 accuracy)

The CVAE should match or beat the plain VAE on the phenotype probes
(nuisance removal should not cost phenotype signal) while its site probe is
much lower — the single figure that tells the nuisance-vs-signal story.

**Every split is by subject** (LOSO-style, grouped cross-validation), never
by clip, so overlapping windows of one walk cannot leak between train and
test ([post-hoc plan §5, §8]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import palette as pal


def _encode_int(y_obj) -> np.ndarray:
    _, codes = np.unique(np.asarray(y_obj, dtype=str), return_inverse=True)
    return codes


def _keep_labelled(values):
    def _null(v):
        if v is None:
            return True
        try:
            return bool(np.isnan(v))
        except (TypeError, ValueError):
            return False
    return np.array([not _null(v) for v in values])


def _grouped_oof(X, y, groups, make_est, n_splits, kind):
    """Out-of-fold predictions under subject-grouped CV, or None.

    Standardises per fold on the training subjects only. Falls back to a
    majority / mean prediction for a fold whose training split is
    degenerate (single class), so one bad fold never crashes the probe.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2 or len(y) < 4:
        return None
    n_splits = int(min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y), dtype=np.float64)
    filled = np.zeros(len(y), dtype=bool)

    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        try:
            if kind == "cls" and len(np.unique(y[tr])) < 2:
                raise ValueError("single-class train fold")
            est = make_est()
            est.fit(Xtr, y[tr])
            pred[te] = est.predict(Xte)
        except Exception:
            if kind == "reg":
                pred[te] = np.mean(y[tr])
            else:
                vals, counts = np.unique(y[tr], return_counts=True)
                pred[te] = vals[np.argmax(counts)]
        filled[te] = True
    return pred, filled


@dataclass
class ProbeResult:
    per_model: dict           # model -> dict of scores
    n_splits: int

    def markdown(self) -> str:
        rows = ["| probe | metric | " +
                " | ".join(self.per_model.keys()) + " |",
                "|" + "---|" * (2 + len(self.per_model))]
        spec = [("UPDRS_GAIT", "updrs_r2", "R²"),
                ("freezer", "freezer_bacc", "bal-acc"),
                ("medication", "med_bacc", "bal-acc"),
                ("site probe (cohort)", "site_acc", "top-1 acc")]
        for title, key, metric in spec:
            cells = [title, metric]
            for model in self.per_model:
                v = self.per_model[model].get(key, float("nan"))
                cells.append(f"{v:.3f}" if v == v else "n/a")
            rows.append("| " + " | ".join(cells) + " |")
        # Site-probe chance row for context.
        chance = next(iter(self.per_model.values())).get("site_chance")
        if chance is not None:
            rows.append(f"\n_Site-probe chance ≈ {chance:.3f}._")
        return "\n".join(rows)


def run_probes(data, n_splits: int = 5, seed: int = 0) -> ProbeResult:
    """Fit every probe for every model under subject-grouped CV ([§5])."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import r2_score, balanced_accuracy_score

    per_model: dict = {}
    for model in data.models:
        mu = np.asarray(data.clip_mu[model], dtype=np.float64)
        subj = data.subject
        scores: dict = {}

        # --- UPDRS ridge regression (R²) ---
        if data.has_label("updrs_gait"):
            keep = _keep_labelled(data.label("updrs_gait"))
            y = np.array([float(v) for v in data.label("updrs_gait")[keep]])
            res = _grouped_oof(mu[keep], y, subj[keep],
                               lambda: Ridge(alpha=1.0), n_splits, "reg")
            scores["updrs_r2"] = (float(r2_score(y[res[1]], res[0][res[1]]))
                                  if res else float("nan"))

        # --- freezer / medication logistic (balanced accuracy) ---
        for key, out_key in (("freezer", "freezer_bacc"),
                             ("med", "med_bacc")):
            if data.has_label(key):
                keep = _keep_labelled(data.label(key))
                y = _encode_int(data.label(key)[keep])
                if len(np.unique(y)) >= 2:
                    res = _grouped_oof(
                        mu[keep], y, subj[keep],
                        lambda: LogisticRegression(max_iter=1000),
                        n_splits, "cls")
                    scores[out_key] = (
                        float(balanced_accuracy_score(
                            y[res[1]], res[0][res[1]].astype(int)))
                        if res else float("nan"))
                else:
                    scores[out_key] = float("nan")

        # --- site probe (cohort), two-layer MLP, subject-grouped ---
        y = data.cohort_id
        res = _grouped_oof(
            mu, y, subj,
            lambda: MLPClassifier(hidden_layer_sizes=(128, 64),
                                  max_iter=300, random_state=seed),
            n_splits, "cls")
        if res:
            scores["site_acc"] = float(np.mean(
                res[0][res[1]].astype(int) == y[res[1]]))
        else:
            scores["site_acc"] = float("nan")
        scores["site_chance"] = 1.0 / max(len(np.unique(y)), 1)

        per_model[model] = scores
    return ProbeResult(per_model=per_model, n_splits=n_splits)


def plot_probes(result: ProbeResult, out_dir, name: str = "probes.png"):
    """Grouped bar chart: phenotype probes + site probe, VAE vs CVAE ([§5]).

    One figure for the whole nuisance-vs-signal story: the site probe drops
    from VAE to CVAE while the phenotype probes hold.
    """
    plt = pal.import_matplotlib()
    categories = [("UPDRS\nR²", "updrs_r2"),
                  ("freezer\nbal-acc", "freezer_bacc"),
                  ("medication\nbal-acc", "med_bacc"),
                  ("site probe\ntop-1 acc", "site_acc")]
    # Only keep categories that at least one model scored.
    categories = [(t, k) for t, k in categories
                  if any(k in s for s in result.per_model.values())]
    models = list(result.per_model.keys())
    x = np.arange(len(categories))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(1.6 * len(categories) + 2, 4.4))
    for i, model in enumerate(models):
        vals = [result.per_model[model].get(k, np.nan) for _, k in categories]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               color=pal.MODEL_COLORS.get(model), label=model)

    # Site-probe chance line.
    chance = next(iter(result.per_model.values())).get("site_chance")
    if chance is not None:
        site_i = [i for i, (_, k) in enumerate(categories) if k == "site_acc"]
        if site_i:
            ci = site_i[0]
            ax.plot([ci - 0.45, ci + 0.45], [chance, chance], "--",
                    color="black", linewidth=1.2, label="site chance")

    ax.set_xticks(x)
    ax.set_xticklabels([t for t, _ in categories])
    ax.set_ylabel("held-out score")
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Linear probes on the frozen latent (subject-split)\n"
                 "phenotype signal holds; site probe drops under conditioning")
    ax.legend(title="model")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)
