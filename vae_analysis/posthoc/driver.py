"""Post-hoc analysis driver ([post-hoc plan]): run the whole battery.

``run_posthoc`` takes the two core checkpoints (plain VAE + CVAE) and a
CARE-PD bundle, encodes every clip and outer-loop trajectory, and runs the
post-hoc structure analysis end to end, writing every figure, the per-clip
cluster labels, ``results.json``, and ``summary.md`` to ``outputs/posthoc/``.

    from architectures.care_pd import load_cohorts, build_bundle, TIER1_COHORTS
    from vae_analysis.posthoc import run_posthoc

    walks = load_cohorts("data/h36m", TIER1_COHORTS, source_dir="data/smpl")
    bundle = build_bundle(walks)
    out = run_posthoc(vae_checkpoint="runs/vae/best.pt",
                      cvae_checkpoint="runs/cvae/best.pt",
                      bundle=bundle)

The CVAE is the target model and is foregrounded; the plain VAE is carried
as the comparison ([post-hoc plan §2]). GM-VAE / GM-CVAE are gone from this
path ([post-hoc plan §0]).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import clustering as clust
from . import agreement as agr
from . import temporal as tmp
from . import probes as prb
from . import report as rep
from .data import build_posthoc_data, load_encoder, PosthocData


def _to_python(obj):
    """Recursively convert numpy / dataclass-ish values to JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _save_cluster_labels(data: PosthocData, model: str,
                         result: clust.ClusteringResult, out_dir: Path):
    """Write per-clip hard labels aligned to clip id ([post-hoc plan §2])."""
    path = out_dir / f"cluster_labels_{model}.csv"
    methods = list(result.labels.keys())
    label_keys = [k for k in ("updrs_gait", "freezer", "med")
                  if data.has_label(k)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "walk_index", "subject", "cohort"]
                   + [f"cluster_{m}" for m in methods] + label_keys)
        for i in range(data.n):
            row = [int(data.clip_id[i]), int(data.walk_index[i]),
                   str(data.subject[i]), str(data.cohort_name[i])]
            row += [int(result.labels[m][i]) for m in methods]
            row += ["" if data.label(k)[i] is None else data.label(k)[i]
                    for k in label_keys]
            w.writerow(row)
    return path


def run_posthoc(data: PosthocData | None = None,
                vae_checkpoint=None, cvae_checkpoint=None,
                bundle=None, out_dir: str | Path = "outputs/posthoc",
                clip_length: int = 60, stride: int = 30,
                n_seeds: int = 20, device: str = "cpu", seed: int = 0,
                fog_extractor=None, fog_units: str = "auto",
                make_umap: bool = True, verbose: bool = True) -> dict:
    """Run the post-hoc structure analysis end to end.

    Either pass a pre-built ``data`` (:class:`PosthocData`) — the smoke-test
    path — or the two checkpoints plus a ``bundle`` and let the driver
    encode. Writes all outputs under ``out_dir``.

    Returns a dict with ``results`` (all scalars, also on disk as
    ``results.json``) and ``written`` (paths of every file produced).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def log(msg):
        if verbose:
            print(f"[posthoc] {msg}")

    # ---- 0. Encode (unless a PosthocData was handed in) ----
    if data is None:
        if bundle is None or cvae_checkpoint is None:
            raise ValueError("Pass either `data`, or `bundle` + at least "
                             "`cvae_checkpoint` (VAE optional).")
        encoders: dict = {}
        if vae_checkpoint is not None:
            enc, name = load_encoder(vae_checkpoint, device=device)
            encoders[name] = enc
        enc, name = load_encoder(cvae_checkpoint, device=device)
        encoders[name] = enc
        log(f"encoding clips + trajectories for {list(encoders)} ...")
        data = build_posthoc_data(encoders, bundle, clip_length=clip_length,
                                  stride=stride, fog_extractor=fog_extractor,
                                  fog_units=fog_units)
    log(f"{data.n} clips, d_z={data.d_z}, models={data.models}, "
        f"primary={data.primary}")

    results: dict = {"n_clips": data.n, "d_z": data.d_z,
                     "models": data.models, "primary": data.primary}

    # ---- 1. BIC sanity check ----
    log("§1 BIC sanity check ...")
    bic = clust.bic_vs_k(data)
    results["bic"] = {m: {"k": bic[m]["k"], "modal": bic[m]["modal"],
                          "bic": bic[m]["bic"]} for m in data.models}
    written.append(clust.plot_bic_vs_k(bic, out_dir))

    # ---- 2 / 2.1 / 2.2 / 2.3  clustering, stability, agreement, composition ----
    cluster_results: dict = {}
    stabilities: dict = {}
    results["clustering"] = {}
    results["stability"] = {}
    results["agreement"] = {}
    results["composition"] = {}

    # Foreground the primary (CVAE) first in logs, but process all models.
    ordered = ([data.primary] +
               [m for m in data.models if m != data.primary])
    for model in ordered:
        mu = data.clip_mu[model]
        k = max(bic[model]["k"], 2)   # cluster ≥2 even if BIC prefers 1
        log(f"§2 clustering {model} at K={k} (BIC K={bic[model]['k']}) ...")
        cres = clust.cluster_latent(mu, k, n_seeds=n_seeds, model=model,
                                    seed=seed)
        cluster_results[model] = cres
        log(f"§2.1 stability {model} ...")
        stab = clust.cluster_stability(mu, cres, seed=seed)
        stabilities[model] = stab

        # §2 robustness to the exact K: ARI of the K clustering against K-1 / K+1.
        from sklearn.metrics import adjusted_rand_score
        krob = clust.cluster_k_robustness(mu, k, seed=seed)
        k_robust_ari = {int(kk): float(adjusted_rand_score(cres.labels["kmeans"],
                                                           lab))
                        for kk, lab in krob.items() if kk != k}
        results["clustering"][model] = {
            "k": k, "bic_k": bic[model]["k"],
            "methods": list(cres.labels.keys()),
            "hdbscan_min_cluster_size": cres.hdbscan_min_cluster_size,
            "k_robustness_ari": k_robust_ari,
        }
        results["stability"][model] = stab.summary()

        # §2.2 agreement + panels
        ag = agr.agreement_table(data, model, cres)
        results["agreement"][model] = {m: ag[m] for m in cres.labels}
        results["agreement"][f"{model}_markdown"] = ag["markdown"]
        written.append(agr.plot_latent_panels(data, model, cres, out_dir,
                                               kind="pca", seed=seed))
        if make_umap:
            u = agr.plot_latent_panels(data, model, cres, out_dir,
                                       kind="umap", seed=seed)
            if u is not None:
                written.append(u)

        # §2.3 composition
        prim_method = agr.data_primary_method(cres)
        comp = agr.subject_composition(data, model, cres.labels[prim_method])
        results["composition"][model] = {
            "pure_fraction": comp["pure_fraction"],
            "n_subjects": len(comp["subjects"]),
        }
        written.append(agr.plot_subject_composition(comp, model, out_dir))

        # §2.1 consensus + persist labels / responsibilities
        written.append(clust.plot_consensus(stab, out_dir,
                                            f"consensus_matrix_{model}.png"))
        written.append(_save_cluster_labels(data, model, cres, out_dir))
        if cres.gmm_responsibilities is not None:
            rp = out_dir / f"gmm_responsibilities_{model}.npy"
            np.save(rp, cres.gmm_responsibilities)
            written.append(rp)

    written.append(clust.plot_stability_ari(stabilities, out_dir))

    # ---- 3. Within-severity substructure (primary model) ----
    log("§3 within-severity substructure (primary) ...")
    ws = agr.within_severity(data, data.primary, n_seeds=n_seeds, seed=seed)
    results["within_severity"] = {
        int(lvl): {kk: vv for kk, vv in r.items()
                   if kk not in ("coords", "labels", "var")}
        for lvl, r in ws.items()}
    ws_plot = agr.plot_within_severity(ws, data.primary, out_dir)
    if ws_plot is not None:
        written.append(ws_plot)

    # ---- 4.2 HMM regimes (primary model) ----
    log("§4.2 Gaussian HMM regimes (primary) ...")
    stride_seconds = stride / 30.0
    hmm = tmp.fit_hmm(data, data.primary, stride_seconds=stride_seconds,
                      seed=seed)
    if hmm is not None:
        results["hmm"] = {
            "k": hmm.k, "bic_curve": hmm.bic_curve,
            "occupancy": hmm.occupancy.tolist(),
            "dwell_mean_seconds": hmm.dwell_mean_seconds.tolist(),
            "empirical_dwell_mean_seconds": [
                float(np.mean(hmm.empirical_dwell[s]))
                if len(hmm.empirical_dwell[s]) else float("nan")
                for s in range(hmm.k)],
            "state_usage": {gk: {str(g): v.tolist()
                                 for g, v in tmp.hmm_state_usage(
                                     hmm, data, gk).items()}
                            for gk in ("cohort", "freezer", "med")
                            if tmp._group_present(hmm, data, gk)},
        }
        for fn, nm in ((tmp.plot_hmm_transition, None),
                       (tmp.plot_hmm_dwell, None)):
            written.append(fn(hmm, out_dir, data.primary))
        written.append(tmp.plot_hmm_trajectories(hmm, data, out_dir,
                                                 data.primary))
        occ = tmp.plot_hmm_occupancy(hmm, data, out_dir, data.primary)
        if occ is not None:
            written.append(occ)
    else:
        results["hmm"] = None
        log("  HMM skipped (hmmlearn missing or too little trajectory data)")

    # ---- 4.3 PELT change points vs E-LC FoG ----
    log("§4.3 PELT change points vs annotated FoG (E-LC) ...")
    pelt = tmp.pelt_fog_validation(data, data.primary, seed=seed)
    if pelt is not None:
        results["pelt"] = {
            "penalty": pelt.penalty,
            "precision": {str(t): pelt.precision[t] for t in pelt.tolerances},
            "recall": {str(t): pelt.recall[t] for t in pelt.tolerances},
            "f1": {str(t): pelt.f1[t] for t in pelt.tolerances},
            "n_report_walks": len(pelt.report_walks),
            "n_tune_walks": len(pelt.tune_walks),
            "mean_per_walk_hit_rate": float(np.nanmean(
                list(pelt.per_walk_hit_rate.values())))
            if pelt.per_walk_hit_rate else float("nan"),
        }
        for fn in (tmp.plot_pelt_timelines,):
            p = fn(pelt, data, data.primary, out_dir)
            if p is not None:
                written.append(p)
        written.append(tmp.plot_pelt_pr(pelt, out_dir, data.primary))
        sw = tmp.plot_pelt_single_walk(pelt, data, data.primary, out_dir)
        if sw is not None:
            written.append(sw)
    else:
        results["pelt"] = None
        log("  PELT-FoG skipped (no E-LC FoG annotations in this bundle)")

    # ---- 5. Continuous probes ----
    log("§5 continuous linear probes (subject-split) ...")
    probe = prb.run_probes(data, seed=seed)
    results["probes"] = {m: {k: v for k, v in s.items()}
                         for m, s in probe.per_model.items()}
    written.append(prb.plot_probes(probe, out_dir))

    # ---- Write results.json ----
    with open(out_dir / "results.json", "w") as f:
        json.dump(_to_python(results), f, indent=2)
    written.append(out_dir / "results.json")

    # ---- 7. Summary report ----
    log("§7 writing summary.md ...")
    summary_path = rep.write_summary(
        results, data, out_dir,
        bic=bic, agreement=results["agreement"],
        stabilities=stabilities, probe=probe, hmm=hmm, pelt=pelt, ws=ws)
    written.append(summary_path)

    written = [p for p in written if p is not None]
    log(f"wrote {len(written)} file(s) to {out_dir}")
    return {"results": results, "written": written, "data": data}
