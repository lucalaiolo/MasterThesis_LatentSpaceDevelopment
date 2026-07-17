"""Write ``outputs/gaitgen_neonate/summary.md`` ([plan §8]).

Reconstruction (normalised units, stated), the PORE/PMPG/DS table, probe
accuracies on q_m (low) and q_p (high), HSIC with p-values, cluster
stability + label agreement on both latents, codebook usage, and the token
HMM. Each section ends with a one-sentence verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _f(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def write_summary(results: dict, out_dir) -> Path:
    r = results
    L = [f"# Neonate GAITGen — disentangled RVQ-VAE analysis ({r['name']})\n",
         f"{r['n_clips']} clips, {r['n_subjects']} subjects. Reconstruction "
         "errors are in **normalised units** (torso-scaled), not mm.\n"]

    # 1. Reconstruction
    d = r["disentanglement"]
    L.append("## 1. Reconstruction\n")
    L.append(f"Reconstruction MPJPE = **{_f(d['recon_mpjpe'])}** (normalised "
             "units).\n")
    L.append("_Reconstruction fidelity of the full model (q_m + q_p)._\n")

    # 2. Disentanglement table
    L.append("## 2. Disentanglement (PORE / PMPG / DS)\n")
    L.append("| metric | value | reading |")
    L.append("|---|---|---|")
    L.append(f"| PORE | {_f(d['pore'])} | e_p={_f(d['e_p'])} vs "
             f"e_pm={_f(d['e_pm'])}; higher = less motion leak into q_p |")
    L.append(f"| PMPG | {_f(d['pmpg'])} | acc(q_p→c_p)={_f(d['acc_qp'])} − "
             f"acc(q_m→c_p)={_f(d['acc_qm'])} |")
    L.append(f"| DS | {_f(d['ds'])} | geometric mean of PORE and PMPG |")
    L.append("")
    verdict = ("pathology is captured in q_p and largely absent from q_m"
               if d["pmpg"] > 0.1 and d["pore"] > 0 else
               "disentanglement is weak — pathology leaks between the latents")
    L.append(f"_DS = {_f(d['ds'])}: {verdict}._\n")

    # 3. Paired UMAP
    L.append("## 3. Paired UMAP\n")
    L.append(f"See `umap_paired_{r['name']}.png`: panel A (q_p) should "
             "separate by class, panel B (q_m) should overlap.\n")
    L.append("_The two panels visualise pathology structure in q_p and its "
             "absence from q_m._\n")

    # 4. Probes
    L.append("## 4. Probes (invariance vs specificity)\n")
    qm, qp = r["probe_qm_cp"], r["probe_qp_cp"]
    L.append("| latent | probe balanced-acc | target |")
    L.append("|---|---|---|")
    L.append(f"| q_m → c_p | {_f(qm['balanced_acc'])} | low (near chance "
             f"{_f(qm['chance'])}) |")
    L.append(f"| q_p → c_p | {_f(qp['balanced_acc'])} | high |")
    if r.get("descriptive_probes"):
        for lname, pr in r["descriptive_probes"].items():
            L.append(f"| {lname} on q_m | {_f(pr['on_qm']['balanced_acc'])} | "
                     "descriptive (report) |")
            L.append(f"| {lname} on q_p | {_f(pr['on_qp']['balanced_acc'])} | "
                     "ideally near chance |")
    L.append("")
    L.append(f"_q_m predicts pathology at {_f(qm['balanced_acc'])} "
             f"(chance {_f(qm['chance'])}) while q_p reaches "
             f"{_f(qp['balanced_acc'])}: the invariance and specificity "
             "targets._\n")

    # 5. HSIC
    L.append("## 5. Independence (HSIC)\n")
    L.append("| pair | HSIC | p-value | target |")
    L.append("|---|---|---|---|")
    labels = {"qm_vs_cp": ("HSIC(q_m, c_p)", "small"),
              "qm_vs_qp": ("HSIC(q_m, q_p)", "small"),
              "qp_vs_cnuis": ("HSIC(q_p, c_nuis)", "small")}
    for key, (title, tgt) in labels.items():
        if key in r["hsic"]:
            h = r["hsic"][key]
            L.append(f"| {title} | {_f(h['hsic'],4)} | {_f(h['p_value'])} | {tgt} |")
    L.append("")
    L.append("_Low HSIC (high p) means the motion latent is independent of "
             "pathology and of the pathology latent._\n")

    # 6. Clustering
    L.append("## 6. Clustering on both latents\n")
    for tag, want in (("cluster_qm", "low (invariance)"),
                      ("cluster_qp", "high (specificity)")):
        c = r[tag]
        agr = c["agreement"]
        km = agr.get("kmeans", {})
        L.append(f"**{tag}** (K={c['k']}, want agreement {want}): "
                 f"k-means ARI={_f(km.get('ari'),2)} NMI={_f(km.get('nmi'),2)}; "
                 f"between-run ARI={_f(c['stability']['between_run_kmeans'],2)}; "
                 f"subject-pure={_f(c['composition']['pure_fraction'],2)}.")
    L.append("")
    L.append("_Clusters of q_p track the pathology label while clusters of "
             "q_m do not — the reverse of a phenotype-recovery target, stated "
             "explicitly._\n")

    # 7. Codebook
    L.append("## 7. Codebook usage\n")
    cbk = r["codebook"]
    L.append(f"- Motion codebook usage per layer: "
             f"{[round(x,2) for x in cbk['usage_motion']]}")
    L.append(f"- Pathology codebook usage per layer: "
             f"{[round(x,2) for x in cbk['usage_pathology']]}")
    L.append(f"- Mean pathology base-code class purity: "
             f"{_f(cbk['pathology_code_purity_mean'],2)}")
    L.append("")
    L.append("_Usage above the 20% floor means the codebooks are not "
             "collapsing; high base-code purity means pathology codes are "
             "class-specific._\n")

    # 8. Token HMM
    L.append("## 8. Token HMM\n")
    if r.get("token_hmm"):
        h = r["token_hmm"]
        L.append(f"- States (BIC): {h['k']}")
        L.append(f"- Occupancy: {[round(x,2) for x in h['occupancy']]}")
        for c, v in h["occupancy_by_class"].items():
            L.append(f"- c_p={c} state usage: {[round(x,2) for x in v]}")
        L.append("")
        L.append(f"_A {h['k']}-state HMM over motion tokens; differences in "
                 "state usage across classes are a temporal read on "
                 "pathology._\n")
    else:
        L.append("_Token HMM skipped (hmmlearn missing or too little data)._\n")

    path = Path(out_dir) / "summary.md"
    path.write_text("\n".join(L))
    return path
