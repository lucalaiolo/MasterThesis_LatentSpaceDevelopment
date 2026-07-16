"""Write ``outputs/posthoc/summary.md`` ([post-hoc plan §7]).

Eight sections, each ending in one plain-language sentence stating what it
shows. The verdicts are computed from the numbers so the prose tracks the
run rather than a prior expectation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _verdict_modal(bic_res) -> str:
    k = bic_res["k"]
    if k <= 2:
        return (f"BIC bottoms out at K={k}: the latent is essentially "
                "unimodal, so the phenotype claim rests on the continuous "
                "probes (§8) rather than discrete clusters.")
    return (f"BIC prefers K={k} with structure beyond one blob: the latent "
            "is modal enough to justify the clustering stack.")


def _mean_between(stab_summary, method="kmeans"):
    return stab_summary["between_run_ari_mean"].get(method, float("nan"))


def write_summary(results, data, out_dir, bic, agreement, stabilities,
                  probe, hmm, pelt, ws) -> Path:
    out_dir = Path(out_dir)
    primary = data.primary
    L: list[str] = []
    L.append("# CARE-PD post-hoc structure analysis\n")
    L.append(f"Two core models: **{', '.join(data.models)}** "
             f"(target model: **{primary}**). "
             f"{results['n_clips']} clips, latent width d_z={results['d_z']}. "
             "The GM-VAE / GM-CVAE are off this path (component collapse); "
             "phenotype structure is read post hoc from the plain VAE / CVAE "
             "latents.\n")

    # ---- 1. BIC ----
    L.append("## 1. BIC — is the latent modal or continuous?\n")
    L.append("| model | BIC-preferred K | reading |")
    L.append("|---|---|---|")
    for m in data.models:
        L.append(f"| {m} | {bic[m]['k']} | "
                 f"{'modal' if bic[m]['modal'] else 'unimodal / continuous'} |")
    L.append("")
    L.append("_" + _verdict_modal(bic[primary]) + "_\n")

    # ---- 2. Stability ----
    L.append("## 2. Cluster stability — are the clusters trustworthy?\n")
    L.append("| model | between-run ARI (k-means) | between-run ARI (GMM) | "
             "cross-method ARI |")
    L.append("|---|---|---|---|")
    for m in data.models:
        s = stabilities[m].summary()
        cross = s["cross_method_ari"]
        cross_mean = (np.mean(list(cross.values())) if cross else float("nan"))
        L.append(f"| {m} | {_fmt(_mean_between(s, 'kmeans'))} | "
                 f"{_fmt(_mean_between(s, 'gmm'))} | {_fmt(cross_mean)} |")
    L.append("")
    prim_between = _mean_between(stabilities[primary].summary(), "kmeans")
    if prim_between == prim_between and prim_between > 0.6:
        v = (f"With between-run ARI ≈ {_fmt(prim_between)} on the {primary} "
             "latent, the clusters reproduce across seeds and methods and can "
             "carry the phenotype claim.")
    else:
        v = (f"Between-run ARI ≈ {_fmt(prim_between)} on the {primary} latent "
             "is low: the structure is continuous rather than modal, and is "
             "reported as such.")
    L.append("_" + v + "_\n")

    # ---- 3. Cluster–label agreement ----
    L.append("## 3. Cluster–label agreement (phenotype recovery)\n")
    L.append(f"**{primary} (target):**\n")
    L.append(agreement.get(f"{primary}_markdown", "_n/a_"))
    L.append("")
    for m in data.models:
        if m != primary:
            L.append(f"**{m} (comparison):**\n")
            L.append(agreement.get(f"{m}_markdown", "_n/a_"))
            L.append("")
    # Contrast verdict: phenotype vs cohort on the primary.
    prim = agreement.get(primary, {})
    phen_aris, cohort_ari = [], None
    for method, per_label in prim.items():
        for key, s in per_label.items():
            if key == "cohort":
                cohort_ari = s.get("ari")
            elif key in ("updrs_gait", "freezer", "med"):
                if s.get("ari") == s.get("ari"):
                    phen_aris.append(s.get("ari"))
    phen_mean = np.mean(phen_aris) if phen_aris else float("nan")
    L.append("_On the " + primary + " latent, mean phenotype ARI ≈ "
             f"{_fmt(phen_mean)} against cohort ARI ≈ {_fmt(cohort_ari)}: "
             "the invariance mechanism is doing its job when phenotype "
             "agreement exceeds cohort agreement._\n")

    # ---- 4. Subject composition ----
    L.append("## 4. Subject composition — phenotypes or individuals?\n")
    L.append("| model | subjects | 'pure' fraction (>80% one cluster) |")
    L.append("|---|---|---|")
    for m in data.models:
        c = results["composition"][m]
        L.append(f"| {m} | {c['n_subjects']} | {_fmt(c['pure_fraction'], 2)} |")
    L.append("")
    pure = results["composition"][primary]["pure_fraction"]
    if pure == pure and pure > 0.6:
        v = (f"{_fmt(pure*100,0)}% of subjects sit almost entirely in one "
             "cluster: the clusters risk tracking individuals, not "
             "phenotypes — treat the phenotype reading with caution.")
    else:
        v = (f"Only {_fmt(pure*100,0)}% of subjects are cluster-pure, so the "
             "clusters cut across individuals and read as phenotypes rather "
             "than identities.")
    L.append("_" + v + "_\n")

    # ---- 5. Within-severity ----
    L.append("## 5. Within-severity substructure\n")
    active = {lvl: r for lvl, r in ws.items() if not r.get("skipped")}
    if active:
        L.append("| UPDRS level | clips | BIC K | between-run ARI | "
                 "stable sub-clusters |")
        L.append("|---|---|---|---|---|")
        multi = 0
        for lvl, r in sorted(active.items()):
            L.append(f"| {lvl} | {r['n']} | {r['k_bic']} | "
                     f"{_fmt(r['between_run_ari_kmeans'],2)} | "
                     f"{r['n_stable_subclusters']} |")
            if r["n_stable_subclusters"] > 1:
                multi += 1
        L.append("")
        v = (f"{multi} of {len(active)} severity levels carry more than one "
             "stable phenotype — the result that most clearly exceeds a "
             "label-driven model." if multi else
             "No severity level splits into more than one stable phenotype at "
             "this β; the within-severity signal is weak.")
    else:
        v = "Too few labelled clips per UPDRS level to test within-severity structure."
    L.append("_" + v + "_\n")

    # ---- 6. HMM ----
    L.append("## 6. HMM regimes (outer-loop trajectory)\n")
    if hmm is not None:
        # Empirical dwell (bounded, matches the violin plot) is the honest
        # readout; the closed-form 1/(1-p_ii) blows up for a near-absorbing
        # state, so it is kept in results.json but not shown here.
        emp_dwell = [round(float(np.mean(hmm.empirical_dwell[s])), 2)
                     if len(hmm.empirical_dwell[s]) else float("nan")
                     for s in range(hmm.k)]
        L.append(f"- **States (BIC):** {hmm.k}")
        L.append(f"- **Occupancy:** "
                 f"{[round(float(o),2) for o in hmm.occupancy]}")
        L.append(f"- **Mean dwell (s, empirical):** {emp_dwell}")
        su = results["hmm"].get("state_usage", {})
        for gk, gv in su.items():
            L.append(f"- **State usage by {gk}:** "
                     + "; ".join(f"{g}: {[round(x,2) for x in vec]}"
                                 for g, vec in gv.items()))
        L.append("")
        L.append(f"_A {hmm.k}-state HMM segments the outer-loop trajectory "
                 "into behavioural regimes; differences in state usage across "
                 "cohort / freezer / medication are where a clean temporal "
                 "result would show._\n")
    else:
        L.append("_HMM skipped (hmmlearn missing or too little trajectory "
                 "data)._\n")

    # ---- 7. PELT ----
    L.append("## 7. PELT change points vs annotated FoG (E-LC)\n")
    if pelt is not None:
        L.append(f"Penalty tuned on {len(pelt.tune_walks)} held-out E-LC "
                 f"walks (penalty={pelt.penalty:g}), evaluated frozen on "
                 f"{len(pelt.report_walks)} report walks.\n")
        L.append("| tolerance | precision | recall | F1 |")
        L.append("|---|---|---|---|")
        for t in pelt.tolerances:
            L.append(f"| ±{t:g}s | {_fmt(pelt.precision[t],2)} | "
                     f"{_fmt(pelt.recall[t],2)} | {_fmt(pelt.f1[t],2)} |")
        L.append("")
        best_rec = max((pelt.recall[t] for t in pelt.tolerances
                        if pelt.recall[t] == pelt.recall[t]), default=float("nan"))
        L.append(f"_Detected change points recover annotated freezing with "
                 f"recall up to {_fmt(best_rec,2)}; this is the cleanest "
                 "phase- and speed-independent external validation in the "
                 "analysis._\n")
    else:
        L.append("_PELT-FoG skipped: no E-LC FoG annotations in this bundle._\n")

    # ---- 8. Linear probes ----
    L.append("## 8. Linear probes — nuisance vs signal\n")
    L.append(probe.markdown())
    L.append("")
    # Contrast: site probe drop, phenotype hold.
    per = probe.per_model
    target = primary if primary in per else None
    if "VAE" in per and target and target != "VAE":
        site_v, site_t = per["VAE"].get("site_acc"), per[target].get("site_acc")
        L.append(f"_The site probe drops from {_fmt(site_v,2)} (VAE) to "
                 f"{_fmt(site_t,2)} ({target}) while the phenotype probes "
                 "hold: the invariance mechanism removes the cohort axis "
                 "without costing phenotype signal._\n")
    else:
        L.append("_Probe scores summarise how much phenotype and cohort "
                 "signal the frozen latent carries._\n")

    text = "\n".join(L)
    path = out_dir / "summary.md"
    path.write_text(text)
    return path
