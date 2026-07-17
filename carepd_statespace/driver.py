"""End-to-end CARE-PD state-space pipeline ([guideline §3-§8]).

Fits the principal-movement basis, selects and fits the ARHMM by
subject-level CV, decodes states, runs the occupancy/dwell/transition
analysis and the clinical mixed-effects models, draws the figures, and
writes the go/no-go verdict to ``RESULTS.md``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import principal_movements as pm, statespace as ss, analysis as an
from . import palette as pal
from .backends import build_arhmm, available_backends


def _subject_folds(subjects, n_folds, seed=0):
    from sklearn.model_selection import GroupKFold
    idx = np.arange(len(subjects))
    gkf = GroupKFold(n_splits=min(n_folds, len(set(subjects))))
    return list(gkf.split(idx, groups=subjects))


def select_model(series, subjects, K_grid, L_grid, n_folds=5, seed=0,
                 backend="numpy", fit_iters=30, verbose=True) -> dict:
    """Subject-CV model selection over (K, L) + family comparison ([§4.3])."""
    folds = _subject_folds(np.asarray(subjects), n_folds, seed)
    rows = []
    for K in K_grid:
        for L in L_grid:
            te_ll = []
            for tr, te in folds:
                m = build_arhmm(K, L, backend, seed).fit(
                    [series[i] for i in tr], n_iter=fit_iters)
                te_ll.append(m.log_likelihood([series[i] for i in te]))
            rows.append({"model": "ar" if L > 0 else "gaussian", "K": K,
                         "L": L, "test_ll": float(np.mean(te_ll))})
            if verbose:
                print(f"  [cv] K={K} L={L} test_LL={np.mean(te_ll):.3f}")
    # GMM family (no progression), best K.
    for K in K_grid:
        te_ll = []
        for tr, te in folds:
            gm = ss.fit_gmm([series[i] for i in tr], K, seed)
            te_ll.append(ss.gmm_loglik_per_frame(gm, [series[i] for i in te]))
        rows.append({"model": "full", "K": K, "L": 0, "test_ll": float(np.mean(te_ll))})
    table = pd.DataFrame(rows)
    ar = table[table.model == "ar"]
    best = ar.loc[ar.test_ll.idxmax()] if len(ar) else table.loc[table.test_ll.idxmax()]
    return {"table": table, "best_K": int(best.K), "best_L": int(best.L)}


def fit_stable(series, K, L, n_restarts=25, seed=0, n_iter=50, backend="numpy"):
    """Refit ``n_restarts`` times, align states, average params ([§4.4]).

    Parameter averaging is only well-defined for the NumPy backend (whose
    state parameters are directly accessible); for ``ssm`` / ``dynamax`` the
    best-log-likelihood restart is returned instead (still aligned-reported).
    """
    models = [build_arhmm(K, L, backend, seed + r).fit(series, n_iter=n_iter)
              for r in range(n_restarts)]
    best = max(models, key=lambda m: m.final_loglik)
    if n_restarts == 1 or getattr(best, "backend", "numpy") != "numpy":
        return best
    Ws, Qs, mu0s, pis, Ps = [], [], [], [], []
    for m in models:
        perm = ss.align_states(best, m)
        p = m.params
        Ws.append(p.W[perm]); Qs.append(np.linalg.inv(p.Q_inv)[perm])
        mu0s.append(p.mu0[perm]); pis.append(p.pi[perm])
        Ps.append(np.exp(p.logP)[np.ix_(perm, perm)])
    avg = ss.ARHMM(K, L, seed=seed)
    avg.params = ss._pack(np.mean(pis, 0), np.log(np.mean(Ps, 0) + 1e-12),
                          np.mean(Ws, 0), np.mean(Qs, 0), np.mean(mu0s, 0),
                          K, L, best.params.d)
    avg.final_loglik = best.final_loglik
    return avg


def generative_check(model, series, out_dir, seed=0):
    """Compare synthetic vs empirical feature distributions (Fig 1e) ([§4.5])."""
    plt = pal.import_matplotlib()
    rng = np.random.default_rng(seed)
    emp = np.concatenate(series, 0)
    lengths = [len(s) for s in series]
    syn = np.concatenate([model.sample(int(rng.choice(lengths)), rng)
                          for _ in range(min(30, len(series)))], 0)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].hist(emp[:, 0], bins=40, density=True, alpha=0.6, label="empirical")
    ax[0].hist(syn[:, 0], bins=40, density=True, alpha=0.6, label="synthetic")
    ax[0].set_title("PM-1 velocity distribution")
    ax[0].legend(fontsize=8)
    ac_e = [np.corrcoef(emp[:-k, 0], emp[k:, 0])[0, 1] for k in range(1, 15)]
    ac_s = [np.corrcoef(syn[:-k, 0], syn[k:, 0])[0, 1] for k in range(1, 15)]
    ax[1].plot(ac_e, label="empirical")
    ax[1].plot(ac_s, label="synthetic")
    ax[1].set_title("autocorrelation (PM-1)")
    ax[1].set_xlabel("lag")
    ax[1].legend(fontsize=8)
    fig.suptitle("Generative validation (Fig 1e)")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, "fig1e_generative.png")


def run_pipeline(data, out_dir="carepd_statespace/outputs",
                 K_grid=(2, 5, 8, 10, 15), L_grid=(1, 2, 3, 5, 8),
                 n_restarts=25, n_folds=5, seed=0, backend="numpy",
                 cv_iters=30, fit_iters=50, verbose=True) -> dict:
    """Run the whole pipeline and write outputs + the go/no-go verdict.

    ``backend`` selects the ARHMM implementation ([guideline §4]):
    ``"numpy"`` (default, always available), ``"ssm"`` (the paper's library),
    ``"dynamax"`` (JAX), or ``"auto"``. The backend actually used is recorded
    in the results and ``RESULTS.md``.
    """
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(parents=True, exist_ok=True)
    written = []

    def log(m):
        if verbose:
            print(f"[statespace] {m}")

    # §3 principal movements
    log("§3 fitting principal-movement basis ...")
    basis = pm.fit_basis(data, seed=seed)
    series = pm.project(data, basis)
    site = pm.site_diagnostic(data, basis, seed=seed)
    written.append(pm.plot_variance_curve(basis, out / "figures"))
    log(f"  {basis.n_components} PMs @90% var; TWO-NN={basis.twonn:.1f}; "
        f"cohort clf acc={site['cohort_clf_acc']:.2f} "
        f"(chance {site['cohort_chance']:.2f})")

    # §4 model selection + stable fit
    if backend == "auto":
        backend = ("ssm" if "ssm" in available_backends()
                   else "dynamax" if "dynamax" in available_backends() else "numpy")
    log(f"§4 CV model selection (backend={backend}) ...")
    sel = select_model(series, data.info.subject_id.values, K_grid, L_grid,
                       n_folds, seed, backend=backend, fit_iters=cv_iters,
                       verbose=verbose)
    K, L = sel["best_K"], sel["best_L"]
    sel["table"].to_csv(out / "tables" / "cv_scores.csv", index=False)
    log(f"  selected K={K}, L={L}; refitting x{n_restarts} ...")
    model = fit_stable(series, K, L, n_restarts=n_restarts, seed=seed,
                       n_iter=fit_iters, backend=backend)
    written.append(generative_check(model, series, out / "figures", seed))

    # §5 decode + metrics
    log("§5 decoding states + occupancy/dwell/transitions ...")
    states = an.decode_all(model, series)
    table = an.build_metric_table(data, states, K)
    table.to_csv(out / "tables" / "walk_metrics.csv", index=False)
    written.append(an.plot_state_sequences(states, data.info, K, out / "figures"))
    if "fog" in table:
        written.append(an.plot_group_occupancy(table, K, "fog", out / "figures"))

    # §6 clinical LME
    log("§6 clinical mixed-effects models ...")
    clinical = an.clinical_analysis(table, K)
    with open(out / "tables" / "clinical_lme.json", "w") as f:
        json.dump(_to_python(clinical), f, indent=2)

    # §7 state characterisation
    log("§7 state characterisation ...")
    sjv = an.state_joint_velocity(data, states, K)
    regions = an.state_region_breakdown(sjv)
    written.append(an.plot_state_characterisation(regions, K, out / "figures"))

    # persist model + basis + states
    with open(out / "model" / "fitted.pkl", "wb") as f:
        pickle.dump({"params": model.params, "K": K, "L": L, "basis": basis,
                     "states": states}, f)

    # §8 go/no-go gate
    verdict = _go_no_go(clinical, site)
    results = {"n_walks": data.n_walks, "n_subjects": int(data.info.subject_id.nunique()),
               "d": data.d, "feature_set": data.feature_set,
               "n_pm": basis.n_components, "twonn": basis.twonn,
               "K": K, "L": L, "backend": getattr(model, "backend", "numpy"),
               "site": site, "clinical": clinical, "verdict": verdict}
    with open(out / "tables" / "results.json", "w") as f:
        json.dump(_to_python(results), f, indent=2)
    written.append(_write_results_md(results, out))
    log(f"VERDICT: {verdict['gate']} — {verdict['reason']}")
    return {"results": results, "written": written, "model": model,
            "states": states, "table": table}


def _go_no_go(clinical, site) -> dict:
    """Any state metric separating FoG/med/UPDRS after Bonferroni? ([§8])."""
    hits = []
    for fam, byvar in clinical.items():
        if fam == "transition_scalars":
            byvar = {"fog": byvar["fog"]}
        for var, recs in byvar.items():
            for r in recs:
                if r.get("sig"):
                    hits.append(f"{fam}/{var}/{r['metric']} "
                                f"(coef={r['coef']:.3g}, p_bonf={r['p_bonf']:.3g})")
    gate = "GO" if hits else "NO-GO"
    reason = ("; ".join(hits[:6]) if hits else
              "no state statistic separated FoG/medication/UPDRS beyond "
              "cohort after Bonferroni — states may be gait-phase-only "
              "(retry longer lag, Set A vs B, or a warped ARHMM)")
    return {"gate": gate, "significant": hits, "reason": reason,
            "cohort_confound": site["cohort_clf_acc"]}


def _write_results_md(results, out) -> Path:
    v = results["verdict"]
    L = [f"# CARE-PD state-space — go/no-go verdict\n",
         f"**{v['gate']}** — {v['reason']}\n",
         f"- Walks: {results['n_walks']}, subjects: {results['n_subjects']}, "
         f"feature set {results['feature_set']} (d={results['d']}).",
         f"- Principal movements: {results['n_pm']} @90% variance "
         f"(TWO-NN intrinsic dim {results['twonn']:.1f}).",
         f"- Selected model: ARHMM K={results['K']}, L={results['L']} "
         f"(backend: **{results.get('backend', 'numpy')}**).",
         f"- **Site confound:** cohort classifiable from PM weights at "
         f"{results['site']['cohort_clf_acc']:.2f} "
         f"(chance {results['site']['cohort_chance']:.2f}) — "
         + ("watch this, states may partly encode site."
            if results['site']['cohort_clf_acc'] > results['site']['cohort_chance'] + 0.2
            else "acceptable.") + "\n",
         "## Significant state statistics (Bonferroni, beyond cohort)\n"]
    if v["significant"]:
        for s in v["significant"]:
            L.append(f"- {s}")
    else:
        L.append("- none")
    L.append("\n_The gate: a GO means the unsupervised states carry clinical "
             "signal beyond cohort; proceed to Tier 2 and ablations. A NO-GO "
             "means retry with the longer lag, Set A vs B, or a warped ARHMM "
             "before abandoning the approach._")
    p = Path(out) / "RESULTS.md"
    p.write_text("\n".join(L))
    return p


def _to_python(o):
    if isinstance(o, dict):
        return {str(k): _to_python(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_python(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o
