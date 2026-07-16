"""Run the neonate-GAITGen analysis battery ([plan §6]) -> outputs/gaitgen_neonate/.

Encodes the disentangled latents once, then runs disentanglement metrics,
paired UMAP, invariance/specificity probes, HSIC independence, clustering on
both latents, codebook usage, and the token HMM — writing every figure,
``results.json``, and ``summary.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import (encode as enc, metrics as met, probes as prb, umap_panels as um,
               independence as ind, clustering as clu, codebook as cb,
               temporal as tmp, report as rep)


def _to_python(o):
    if isinstance(o, dict):
        return {str(k): _to_python(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_python(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run_analysis(model, data, config, out_dir="outputs/gaitgen_neonate",
                 descriptive_labels: dict | None = None,
                 name: str = "GAITGen", device: str | None = None,
                 verbose: bool = True) -> dict:
    """Run every §6 analysis on a trained model + windowed data.

    ``descriptive_labels`` optionally maps a label name to a per-clip array
    of categoricals we did **not** condition on (gestational-age band, sex …)
    for the extra probes of [plan §6.3/§6.4].
    """
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = str(next(model.parameters()).device)
    written, results = [], {"name": name}

    def log(m):
        if verbose:
            print(f"[gaitgen-analysis] {m}")

    log("encoding disentangled latents ...")
    lat = enc.encode_latents(model, data, device=device)
    results["n_clips"] = lat.n
    results["n_subjects"] = int(len(set(lat.subject)))

    # §6.1 disentanglement metrics
    log("§6.1 PORE / PMPG / DS ...")
    results["disentanglement"] = met.disentanglement_table(model, lat, data, device)

    # §6.2 paired UMAP
    log("§6.2 paired UMAP ...")
    written.append(um.paired_umap(lat, out_dir, name=name))

    # §6.3 invariance probes on q_m (want low), §6.4 specificity on q_p (high)
    log("§6.3/§6.4 probes ...")
    results["probe_qm_cp"] = prb.probe(lat.q_m, lat.c_p, lat.subject)
    results["probe_qp_cp"] = prb.probe(lat.q_p, lat.c_p, lat.subject)
    if descriptive_labels:
        results["descriptive_probes"] = {}
        for lname, y in descriptive_labels.items():
            results["descriptive_probes"][lname] = {
                "on_qm": prb.probe(lat.q_m, y, lat.subject),
                "on_qp": prb.probe(lat.q_p, y, lat.subject)}

    # §6.5 HSIC independence
    log("§6.5 HSIC independence ...")
    hsic = {"qm_vs_cp": ind.hsic_test(lat.q_m, lat.c_p, y_is_categorical=True),
            "qm_vs_qp": ind.hsic_test(lat.q_m, lat.q_p)}
    if lat.has_nuisance():
        m = lat.c_nuis >= 0
        hsic["qp_vs_cnuis"] = ind.hsic_test(lat.q_p[m], lat.c_nuis[m],
                                            y_is_categorical=True)
    results["hsic"] = hsic

    # §6.6 clustering on q_m (want low agreement), §6.7 on q_p (want high)
    log("§6.6/§6.7 clustering ...")
    cl_m = clu.analyze(lat.q_m, lat.c_p, lat.subject, name=f"{name}_qm")
    cl_p = clu.analyze(lat.q_p, lat.c_p, lat.subject, name=f"{name}_qp")
    results["cluster_qm"] = {k: v for k, v in cl_m.items() if k != "_result"}
    results["cluster_qp"] = {k: v for k, v in cl_p.items() if k != "_result"}
    written.append(clu.plot_subject_composition(cl_m["composition"],
                                                f"{name}_qm", out_dir))
    written.append(clu.plot_subject_composition(cl_p["composition"],
                                                f"{name}_qp", out_dir))

    # §6.8 codebook usage + co-occurrence
    log("§6.8 codebook analysis ...")
    written.append(cb.plot_usage(lat.idx_m, lat.idx_p, config.codebook_motion,
                                 config.codebook_pathology, out_dir, name=name))
    cooc = cb.code_class_cooccurrence(lat.idx_p, lat.c_p, layer=0)
    results["codebook"] = {
        "usage_motion": cb.usage_per_layer(lat.idx_m, config.codebook_motion).tolist(),
        "usage_pathology": cb.usage_per_layer(lat.idx_p, config.codebook_pathology).tolist(),
        "pathology_code_purity_mean": float(np.average(cooc["purity"], weights=cooc["weight"])),
    }

    # §6.9 token HMM + change points
    log("§6.9 token HMM + change points ...")
    hmm = tmp.categorical_hmm(lat.idx_m, lat.c_p, config.codebook_motion)
    if hmm is not None:
        results["token_hmm"] = hmm
        written.append(tmp.plot_hmm(hmm, out_dir, name=name))
    else:
        results["token_hmm"] = None
        log("  token HMM skipped (hmmlearn missing / too little data)")
    results["change_points_mean"] = float(
        tmp.token_change_points(lat.idx_m).mean())

    # write results + report
    with open(out_dir / "results.json", "w") as f:
        json.dump(_to_python(results), f, indent=2)
    written.append(out_dir / "results.json")
    written.append(rep.write_summary(results, out_dir))

    written = [w for w in written if w is not None]
    log(f"wrote {len(written)} file(s) to {out_dir}")
    return {"results": results, "written": written, "latents": lat}
