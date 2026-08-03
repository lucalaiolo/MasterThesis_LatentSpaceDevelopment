#!/usr/bin/env python3
"""End-to-end runner for analyses A1 and A7 plus the raw kinematics (METHODS §1-§12).

    python run_analysis.py --csv rvi38_analysis.csv \
        --arhmm arhmm_rvi38_stream_delta.pkl \
        --hmm hmm_rvi38_stream_delta.pkl --outdir rvi38_out

The AR-HMM is the primary model and the Gaussian HMM is the independent
replication of §7.6. Either may be omitted: with only one model the
replication columns are reported as unavailable rather than silently skipped.

Every number in the output is computed here; nothing is hard-coded. Results land
in ``<outdir>/results.json`` alongside per-subject CSVs and the figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a1_core as A          # noqa: E402
import a1_stats as ST        # noqa: E402
import a57_graph as G        # noqa: E402
import a8_movement as MV     # noqa: E402
import a9_wclrpp as WP       # noqa: E402
import build_pose            # noqa: E402
import load_models as L      # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Tee:
    """Echo stdout into the run log so the console and the file agree."""

    def __init__(self, path):
        self.f = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if o is None or isinstance(o, str):
        return o
    return str(o)


def resolve_state_names(amp, k: int, path: str | None = None):
    """Names for the states, in order of preference.

    1. ``--state-names FILE`` — a JSON list or one name per line. Keeping the
       names in a data file means they survive any edit to the analysis code.
    2. ``a1_core.state_labels(amp)``, if that function exists.
    3. ``a1_core.state_descriptors(amp)``, if it returns a sequence of names
       rather than the descriptor dict.
    4. plain state numbers.

    Names are only ever figure text; no statistic reads them.
    """
    if path:
        with open(path) as fh:
            raw = fh.read().strip()
        names = (json.loads(raw) if raw.startswith("[")
                 else [ln.strip() for ln in raw.splitlines() if ln.strip()])
        if len(names) != k:
            raise ValueError(f"{path} has {len(names)} names but the model has "
                             f"{k} states")
        return [str(n) for n in names], os.path.basename(path)
    for fn, src in ((getattr(A, "state_labels", None), "a1_core.state_labels"),
                    (getattr(A, "state_descriptors", None),
                     "a1_core.state_descriptors")):
        if fn is None:
            continue
        try:
            out = fn(amp)
        except Exception:                                   # noqa: BLE001
            continue
        if isinstance(out, dict):                # the descriptor dict, not names
            continue
        out = [str(v) for v in out]
        if len(out) == k:
            return out, src
    return [str(i) for i in range(k)], "state numbers"


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# The §3.4 autoregressive lag-block check is deliberately not implemented.
# It resolves how `ssm` serialises A^(1) and A^(2) inside `ar_As`, but no
# quantity in A1 or A7 reads those matrices — every analysis here consumes
# only `states` and `transition`. The check therefore verifies a convention
# nothing downstream depends on.


# ---------------------------------------------------------------------------
# per-model pipeline
# ---------------------------------------------------------------------------
def analyse_model(model, vids, pose, spd, vel, labels, geom, cfg, tag,
                  stream="delta"):
    """§5-§9 for one fitted model. Returns everything the figures need."""
    st, vid = model["states"], model["vidid"]
    K, n_sub = model["k"], model["n_subjects"]
    out = {"tag": tag, "k": K, "n_subjects": n_sub}

    # ---- §5 state kinematic signatures ----
    amp, nframe = A.state_profiles(st, vid, vids, spd, pose, geom,
                                  union=(stream == "delta"))
    labels_txt, name_src = resolve_state_names(amp, K, cfg.get("state_names"))
    out.update({"amplitude": amp, "state_frames": nframe,
                "state_labels": labels_txt, "state_name_source": name_src})
    print(f"  state names from {name_src}")
    print(f"  §5 profiles: frames/state {int(nframe.min()):,}..."
          f"{int(nframe.max()):,}  (>= {int(nframe.min()) * len(build_pose.FREE):,} "
          f"joint-frames each)")

    # ---- direction-aware signature: the second-moment matrix M[k,j], its
    # trace is the same a[k,j]^2 as above, and its trace-normalised part gives
    # the double-angle axis coordinate u[k,j] the shape channel correlates.
    M, _ = A.state_second_moments(st, vid, vids, vel, pose, geom,
                                  union=(stream == "delta"), k=amp.shape[0])
    u = A.shape_coordinates(M)                        # (K, J, 2)
    out["shape_coord"] = u
    rho = np.linalg.norm(u[:, build_pose.FREE, :], axis=-1)   # anisotropy on FREE
    out["anisotropy"] = rho
    print(f"  §4 shape: anisotropy rho over free joints "
          f"[{np.nanmin(rho):.2f},{np.nanmax(rho):.2f}] median "
          f"{np.nanmedian(rho):.2f} (0=isotropic, 1=line)")

    # ---- §6-§7 similarity ----
    S, Sparts = A.direction_aware_similarity(
        amp, u, omega=cfg["fluency_omega"], form=cfg["fluency_similarity"],
        shape_state_term=cfg["fluency_shape_state_term"])
    out["S"] = S
    out["S_mag"], out["S_shape"] = Sparts["S_mag"], Sparts["S_shape"]
    out["S_orient"] = Sparts["S_orient"]
    out["similarity_form"] = {"form": Sparts["form"], "omega": Sparts["omega"],
                              "shape_state_term": Sparts["shape_state_term"]}
    out["centring"] = A.centring_comparison(amp)
    out["face_validity"] = A.face_validity(S, labels_txt)
    off = ~np.eye(K, dtype=bool)
    if cfg["fluency_similarity"] == "separated":
        print(f"  §7 similarity form: separated, omega={cfg['fluency_omega']:.2f}"
              f"  (S = omega*S_mag + (1-omega)*S_shape), shape state-term "
              f"{'kept' if cfg['fluency_shape_state_term'] else 'dropped'}")
    else:
        print(f"  §7 similarity form: {cfg['fluency_similarity']}, shape "
              f"state-term "
              f"{'kept' if cfg['fluency_shape_state_term'] else 'dropped'}")
    print(f"     off-diagonal S: combined mean {S[off].mean():+.2f}  |  "
          f"magnitude {Sparts['S_mag'][off].mean():+.2f}  shape "
          f"{Sparts['S_shape'][off].mean():+.2f}  orientation "
          f"{np.nanmean(Sparts['S_orient'][off]):+.2f}")
    c = out["centring"]
    print(f"  §6 magnitude off-diagonal: single-centred "
          f"[{c['single']['min']:+.2f},{c['single']['max']:+.2f}] mean "
          f"{c['single']['mean']:+.2f}  ->  double-centred "
          f"[{c['double']['min']:+.2f},{c['double']['max']:+.2f}] mean "
          f"{c['double']['mean']:+.2f}")
    fv = out["face_validity"]
    print(f"     most similar {fv['most_similar'][0][0]} ~ "
          f"{fv['most_similar'][0][1]} = {fv['most_similar'][0][2]:+.2f}; "
          f"least {fv['least_similar'][0][0]} ~ {fv['least_similar'][0][1]} = "
          f"{fv['least_similar'][0][2]:+.2f}")

    # ---- §7 fluency ----
    phi = A.phi_excess(st, vid, S, n_sub, n_perm=cfg["n_phi"], seed=0)
    out["phi"] = phi
    exc = phi["excess"]
    out["phi_wilcoxon"] = ST.wilcoxon_signed(exc, "greater")
    w = out["phi_wilcoxon"]
    print(f"  §7 fluency: Phi > 0 in {w['n_positive']}/{w['n']}, "
          f"median {w.get('median', float('nan')):+.4f}, "
          f"Wilcoxon p = {w['p']:.3g}")

    out["tercile"] = A.tercile_decomposition(st, vid, S, n_sub)
    if "top_over_bottom" in out["tercile"]:
        print(f"     §7.5 similar transitions are "
              f"{out['tercile']['top_over_bottom']:.2f}x more likely than "
              f"dissimilar ones")

    # Dwell stratification is the over-segmentation control: it asks whether the
    # effect is confined to the shortest dwells, which is what fragmenting one
    # movement into several similar states would produce.
    out["dwell_strat"] = (A.dwell_stratified(st, vid, S, n_sub,
                                             n_perm=cfg["n_dwell"])
                          if cfg["controls"] == "full" else [])

    # ---- chain matrices (consumed by the mixing analysis below) ----
    Afull = model["transition"]
    Ajump = G.jump_chain(Afull)
    out["A_full"], out["A_jump"] = Afull, Ajump
    out["group_jump_empirical"] = G.row_normalise(
        A.group_jump_counts(st, vid, n_sub, K))

    # ---- §9 mixing structure ----
    out["degenerate"] = G.degenerate_centralities(Afull)
    d = out["degenerate"]
    print(f"  §9.1 degeneracy check: out-degree==1 "
          f"{d['out_degree_is_one']}, right Perron constant "
          f"{d['right_perron_is_constant']}, |PageRank(0.99) - stationary|_1 = "
          f"{d['pagerank_to_stationary_l1'][0.99]:.2e}")
    out["kemeny_identity"] = G.kemeny_identity_check(Afull)
    out["companions_full"] = G.graph_companions(Afull, geom.f_win)
    out["companions_jump"] = G.graph_companions(Ajump, geom.f_win)
    ki = out["kemeny_identity"]
    print(f"  §9.2 Kemeny (full chain) = {out['companions_full']['kemeny']:.2f} "
          f"windows = {out['companions_full']['kemeny_seconds']:.2f} s; "
          f"identity check |diff| = {ki['abs_diff']:.2e} "
          f"({'OK' if ki['match'] else 'FAILED'})")
    print(f"       Kemeny (jump chain) = {out['companions_jump']['kemeny']:.2f} "
          f"jumps")

    # ---- §9.3 per-subject shrinkage estimation ----
    seqs = [A.visit_sequence(st[vid == i]) for i in range(n_sub)]
    out["visit_sequences"] = seqs
    al = G.choose_alpha(seqs, K)
    out["alpha"] = al
    print(f"  §9.3 shrinkage alpha = {al['alpha']:.3f} "
          f"{'(DEGENERATE: per-subject matrices carry little information)' if al['degenerate'] else ''}")

    Abar_jump = G.row_normalise(
        np.where(np.eye(K, dtype=bool), 0.0,
                 G.row_normalise(sum(G.counts_from_path(s, K) for s in seqs))))
    kem = np.array([G.subject_kemeny(st[vid == i], K, Abar_jump, al["alpha"])
                    for i in range(n_sub)])
    out["kemeny_per_subject"] = kem

    lo = np.full(n_sub, np.nan)
    hi = np.full(n_sub, np.nan)
    for i in range(n_sub):
        b = G.block_bootstrap_kemeny(st[vid == i], K, Abar_jump, al["alpha"],
                                     block=cfg["block"], n_boot=cfg["n_block"],
                                     seed=i)
        b = b[np.isfinite(b)]
        if len(b) > 10:
            lo[i], hi[i] = np.percentile(b, [2.5, 97.5])
    out["kemeny_lo"], out["kemeny_hi"] = lo, hi
    out["estimability"] = G.estimability_gate(kem, lo, hi)
    e = out["estimability"]
    print(f"  §9.3 estimability: median CI width {e['median_width']:.2f} vs "
          f"between-subject range {e['between_subject_range']:.2f} -> ratio "
          f"{e['ratio']:.2f} ({'PASS' if e['passed'] else 'FAIL'})")
    return out


# ---------------------------------------------------------------------------
# clinical layer (§10, §11) — labels enter only here
# ---------------------------------------------------------------------------
def clinical(res, labels, lengths, cfg, geom, st, vid, S, n_sub):
    pos = labels == 1
    out = {}
    phi = res["phi"]["excess"]
    kem = res["kemeny_per_subject"]
    logL = np.log(np.asarray(lengths, float))

    # §10.8 split-half reliability of Phi (contiguous halves).
    h1 = A.phi_excess(st, vid, S, n_sub, n_perm=cfg["n_phi"], seed=1,
                      window="first")["excess"]
    h2 = A.phi_excess(st, vid, S, n_sub, n_perm=cfg["n_phi"], seed=2,
                      window="second")["excess"]
    out["phi_split_half"] = ST.split_half(h1, h2)
    out["phi_halves"] = (h1, h2)
    sh = out["phi_split_half"]
    print(f"  §10.8 Phi split-half r = {sh['r_half']:+.3f} -> Spearman-Brown "
          f"r_SB = {sh['r_sb']:+.3f} [{sh['r_sb_lo']:+.3f}, {sh['r_sb_hi']:+.3f}]"
          f", ICC(A,1) = {sh['icc_a1']:+.3f}")

    # §2.3 duration controls.
    trunc = A.phi_excess(st, vid, S, n_sub, n_perm=cfg["n_phi"], seed=3,
                         max_win=cfg["truncate"])["excess"]
    out["duration"] = {
        "phi_vs_logL": ST.duration_control(phi, logL),
        "kemeny_vs_logL": ST.duration_control(kem, logL),
        "phi_truncation": ST.ordering_stability(phi, trunc),
        "truncate_to": cfg["truncate"]}
    du = out["duration"]
    print(f"  §2.3 duration: rho(Phi, logL) = {du['phi_vs_logL']['rho']:+.3f} "
          f"(p {du['phi_vs_logL']['p']:.3f}); rho(Kemeny, logL) = "
          f"{du['kemeny_vs_logL']['rho']:+.3f} "
          f"(p {du['kemeny_vs_logL']['p']:.3f})")
    print(f"  §2.3 truncation to {cfg['truncate']} windows: ordering rho = "
          f"{du['phi_truncation']['rho']:+.3f}")

    # §11 redundancy: occupancy entropy and mean dwell.
    ent = np.zeros(n_sub)
    dwl = np.zeros(n_sub)
    for i in range(n_sub):
        s_i = st[vid == i]
        p = np.bincount(s_i, minlength=res["k"]) / max(len(s_i), 1)
        p = p[p > 0]
        ent[i] = -(p * np.log(p)).sum()
        _, rl = A.run_lengths(s_i)
        dwl[i] = rl.mean() if len(rl) else np.nan
    out["occupancy_entropy"], out["mean_dwell"] = ent, dwl
    out["redundancy"] = {
        "phi_vs_entropy": ST.duration_control(phi, ent),
        "phi_vs_dwell": ST.duration_control(phi, dwl),
        "kemeny_vs_entropy": ST.duration_control(kem, ent),
        "kemeny_vs_dwell": ST.duration_control(kem, dwl)}
    rd = out["redundancy"]
    print(f"  §11 redundancy: rho(Phi, entropy) = "
          f"{rd['phi_vs_entropy']['rho']:+.3f}, rho(Phi, dwell) = "
          f"{rd['phi_vs_dwell']['rho']:+.3f}, rho(K, entropy) = "
          f"{rd['kemeny_vs_entropy']['rho']:+.3f}, rho(K, dwell) = "
          f"{rd['kemeny_vs_dwell']['rho']:+.3f}")

    # §10.2 group contrasts, exact where enumeration is feasible.
    out["phi_test"] = ST.mannwhitney(phi[pos], phi[~pos])
    out["kemeny_test"] = ST.mannwhitney(kem[pos], kem[~pos])
    for nm, r in (("Phi", out["phi_test"]), ("Kemeny", out["kemeny_test"])):
        print(f"  §10.2 {nm:7s} AUC = {r['auc']:.3f} "
              f"[{r['auc_lo']:.3f}, {r['auc_hi']:.3f}], rank-biserial "
              f"{r['rank_biserial']:+.3f}, p = {r['p']:.4g}  [{r['method']}]")

    # §7 magnitude-vs-direction split: is the fluency signal carried by how much
    # a joint moves or by the axis along which it moves? Recompute Phi under
    # each channel of the direction-aware similarity and contrast the groups.
    # Descriptive only -- the confirmatory family stays {Phi, Kemeny}.
    out["channel_split"] = {"combined": out["phi_test"]}
    for nm, s_key in (("magnitude", "S_mag"), ("shape", "S_shape")):
        if s_key in res:
            phi_c = A.phi_excess(st, vid, np.asarray(res[s_key]), n_sub,
                                 n_perm=cfg["n_phi"], seed=0)["excess"]
            out["channel_split"][nm] = ST.mannwhitney(phi_c[pos], phi_c[~pos])
    cs = out["channel_split"]
    print("  §7 fluency channel split (Phi group contrast per similarity "
          "channel):")
    for nm in ("combined", "magnitude", "shape"):
        if nm in cs:
            r = cs[nm]
            print(f"     {nm:9s}: AUC = {r['auc']:.3f}, rank-biserial "
                  f"{r['rank_biserial']:+.3f}, p = {r['p']:.4g}")

    # §10.7 confirmatory family: maxT over {Phi, Kemeny}.
    out["maxt"] = ST.maxt({"phi": phi, "kemeny": kem}, pos, n_perm=cfg["n_maxt"])
    out["holm"] = dict(zip(["phi", "kemeny"],
                           ST.holm([out["phi_test"]["p"],
                                    out["kemeny_test"]["p"]]).tolist()))
    mt = out["maxt"]
    print(f"  §10.7 maxT: Phi p = {mt['phi']['p_maxT']:.4g}, Kemeny p = "
          f"{mt['kemeny']['p_maxT']:.4g}   |   Holm: Phi "
          f"{out['holm']['phi']:.4g}, Kemeny {out['holm']['kemeny']:.4g}")

    # §10.3 nuisance adjustment where duration intrudes.
    out["adjusted"] = {}
    for nm, v in (("phi", phi), ("kemeny", kem)):
        rho = out["duration"][f"{nm}_vs_logL"]["rho"]
        if np.isfinite(rho) and abs(rho) >= cfg["duration_tol"]:
            out["adjusted"][nm] = ST.freedman_lane(v, pos, logL,
                                                   n_perm=cfg["n_maxt"])
            print(f"  §10.3 {nm} correlates with logL (rho {rho:+.3f}); "
                  f"Freedman-Lane adjusted AUC = "
                  f"{out['adjusted'][nm]['auc']:.3f}, p = "
                  f"{out['adjusted'][nm]['p']:.4g}")

    # §10.9 group-level BCa intervals, and §10.12 power.
    if cfg["controls"] == "full":
        out["bca"] = {"phi_median": ST.bca_ci(phi, np.median, cfg["n_bca"]),
                      "kemeny_median": ST.bca_ci(kem, np.median, cfg["n_bca"])}
    out["mde"] = ST.min_detectable_effect(int(pos.sum()), int((~pos).sum()))
    print(f"  §10.12 minimum detectable effect at n = {int(pos.sum())} vs "
          f"{int((~pos).sum())}: AUC {out['mde']['auc']:.3f} "
          f"(rank-biserial {out['mde']['rank_biserial']:+.3f}) for 80% power")
    if cfg["controls"] == "full":
        out["loo"] = {"phi": ST.loo_auc(phi[pos], phi[~pos]),
                      "kemeny": ST.loo_auc(kem[pos], kem[~pos])}
        for nm in ("phi", "kemeny"):
            ps = [r["p"] for r in out["loo"][nm]]
            print(f"  LOO {nm:7s}: p ranges {min(ps):.4g} to {max(ps):.4g}")

    # §11 gates.
    gates = {
        "reliability (r_SB >= 0.6) [Phi]": bool(
            np.isfinite(sh["r_sb"]) and sh["r_sb"] >= 0.6),
        "estimability (CI/range < 0.5) [Kemeny]": bool(
            res["estimability"]["passed"]),
        "redundancy (|rho| < 0.8 vs entropy/dwell) [Phi]": bool(
            max(abs(rd["phi_vs_entropy"]["rho"]),
                abs(rd["phi_vs_dwell"]["rho"])) < 0.8),
        "redundancy (|rho| < 0.8 vs entropy/dwell) [Kemeny]": bool(
            max(abs(rd["kemeny_vs_entropy"]["rho"]),
                abs(rd["kemeny_vs_dwell"]["rho"])) < 0.8),
        "duration (ordering stable under truncation) [Phi]": bool(
            np.isfinite(du["phi_truncation"]["rho"])
            and du["phi_truncation"]["rho"] >= 0.5),
        "shrinkage not degenerate (alpha < 0.9) [Kemeny]": bool(
            not res["alpha"]["degenerate"]),
    }
    out["gates"] = gates
    print("\n  §11 pre-specified gates")
    for k, v in gates.items():
        print(f"     {'PASS' if v else 'FAIL'}  {k}")
    return out


# ---------------------------------------------------------------------------
# raw-kinematic constructs (§8): inter-limb coordination and the fidgety band.
# These read raw keypoint displacements only, so they are independent of the
# encoder, the latent and the state model -- a third estimator alongside
# fluency and mixing, and the slow part of a run. Split out so
# ``--skip-raw-kinematics`` can omit them for a fluency-only run.
# ---------------------------------------------------------------------------
def raw_kinematics(results, models, primary, pose, vids, labels, geom, cfg):
    """WCLR-PP inter-limb coordination, the fidgety band ratio, and the
    per-state velocity profiles. Populates ``results['wclrpp']``,
    ``results['fbr']`` and the primary model's ``velocity_profile_*`` entries.
    """
    section("Raw kinematics: inter-limb coordination (WCLR-PP) and the "
            "fidgety band")
    vid_arrays = [pose[v] for v in vids]

    # WCLR-PP: the variance in one limb's future 2D velocity that the other
    # explains beyond that limb's own past. High coupling is the pathological
    # (cramped-synchronised) pole. Each of the six pairs yields one F (fraction
    # of assessable time coupled) and one R2 (strength when coupled), averaged
    # over both regression directions.
    wp = WP.WCLRParams(w=cfg["wclr_w"], tau_max=cfg["wclr_tau_max"],
                       ell_min=cfg["wclr_ell_min"], c=cfg["wclr_c"],
                       dtau=cfg["wclr_dtau"], fps=geom.fps,
                       limb_signal=cfg["wclr_limb_signal"])

    wc = WP.wclrpp_dataset(vid_arrays, wp)
    results["wclrpp"] = {
        "F": wc["F"], "R2": wc["R2"], "pairs": wc["pairs"],
        "pair_class": wc["pair_class"], "mean_F": wc["mean_F"],
        "spread_F": wc["spread_F"], "mean_R2": wc["mean_R2"],
        "params": {"w": wp.w, "tau_max": wp.tau_max, "ell_min": wp.ell_min,
                   "c": wp.c, "dtau": wp.dtau, "fps": wp.fps,
                   "limb_signal": wp.limb_signal}}
    print(f"  WCLR-PP: {len(vid_arrays)} recordings, six limb pairs, "
          f"limb signal '{wp.limb_signal}' "
          f"({'+'.join(str(j) for j in WP.LIMB_SIGNALS[wp.limb_signal]['RA'])}"
          f" for the right arm), "
          f"w={wp.w} frames ({wp.w / geom.fps:.1f}s), "
          f"tau_max={wp.tau_max} (+/-{wp.tau_max / geom.fps:.2f}s), "
          f"c={wp.c}, ell_min={wp.ell_min} ({wp.ell_min / geom.fps:.2f}s), "
          f"dtau={wp.dtau}")
    for p, nm in enumerate(wc["pairs"]):
        print(f"     {nm:8s} ({wc['pair_class'][p]:>13s}): "
              f"median F over subjects {np.nanmedian(wc['F'][:, p]):.3f}  "
              f"(R2 {np.nanmedian(wc['R2'][:, p]):.3f})")

    wt = WP.wclrpp_test(wc["F"], labels, n_perm=cfg["n_wclr"], seed=0,
                        pairs=wc["pairs"])
    results["wclrpp"]["test"] = wt
    print(f"  §10 group contrast on per-pair F, labels permuted "
          f"({wt['n_perm']:,} draws), max-statistic corrected over six pairs "
          f"[all six reported regardless]:")
    for p, nm in enumerate(wc["pairs"]):
        print(f"     {nm:8s}: dF (abnormal-normal) = {wt['observed'][p]:+.3f}  "
              f"p_corrected = {wt['p_corrected'][p]:.4f}  "
              f"(uncorrected {wt['p_uncorrected'][p]:.4f})")

    # whole-body aggregation: mean F over pairs (a whole-body coupling pattern
    # scores high everywhere) and its across-pair spread.
    agg = ST.mannwhitney(wc["mean_F"][labels == 1], wc["mean_F"][labels == 0])
    results["wclrpp"]["mean_F_test"] = agg
    print(f"  whole-body coupling (mean F over pairs): abnormal median "
          f"{np.nanmedian(wc['mean_F'][labels == 1]):.3f} vs normal "
          f"{np.nanmedian(wc['mean_F'][labels == 0]):.3f}; AUC "
          f"{agg['auc']:.3f}, p = {agg['p']:.4f}  [{agg['method']}]")

    fbr = MV.fbr_dataset(vid_arrays, fps=geom.fps, band=cfg["band"])
    fb = ST.mannwhitney(fbr[labels == 1], fbr[labels == 0])
    lo, hi = ST.hanley_mcneil_ci(fb["auc"], fb["n1"], fb["n2"])
    fb["ci"] = (lo, hi)
    results["fbr"] = {"values": fbr, "band": cfg["band"], "test": fb}
    print(f"  fidgety band ratio {cfg['band'][0]}-{cfg['band'][1]} Hz on raw "
          f"velocities: abnormal median {np.nanmedian(fbr[labels == 1]):.4f} vs "
          f"normal {np.nanmedian(fbr[labels == 0]):.4f}")
    print(f"     AUC = {fb['auc']:.3f} [{lo:.3f}, {hi:.3f}], rank-biserial "
          f"{fb['rank_biserial']:+.3f}, p = {fb['p']:.4f}  [{fb['method']}]")

    # per-state velocity profile: regions and the lateralised limbs
    r_res = results[primary]
    for name, groups in (("regions", MV.region_groups()),
                         ("lateral", MV.lateral_groups())):
        prof = MV.state_velocity_profile(
            models[primary]["states"], models[primary]["vidid"], vid_arrays,
            geom, groups, top_frac=cfg["top_frac"], K=r_res["k"])
        r_res[f"velocity_profile_{name}"] = prof
    vp = r_res["velocity_profile_regions"]
    print(f"  per-state high-velocity fraction (top {cfg['top_frac']:.0%} of "
          f"each joint's frames), median over subjects:")
    for s in range(vp["k"]):
        cells = "  ".join(f"{g}={np.nanmedian(vp[s][g]):4.1f}%"
                          for g in vp["groups"])
        print(f"     state {s:2d}: {cells}")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="rvi38_analysis.csv")
    ap.add_argument("--arhmm", default="arhmm_rvi38_stream_delta.pkl")
    ap.add_argument("--hmm", default="hmm_rvi38_stream_delta.pkl")
    ap.add_argument("--labels", default="RVI_38_labels.mat")
    ap.add_argument("--outdir", default="rvi38_out")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--clip", type=int, default=64)
    ap.add_argument("--nwin", type=int, default=16)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--state-names", default=None,
                    help="file of state names (JSON list or one per line); "
                         "overrides whatever a1_core provides")
    ap.add_argument("--models", choices=("arhmm", "hmm", "both"),
                    default="both",
                    help="which fitted model(s) to analyse; 'arhmm' skips the "
                         "Gaussian HMM entirely (and with it the §7.6/§8.5 "
                         "cross-model replication)")
    ap.add_argument("--controls", choices=("core", "full"), default="full",
                    help="'core' runs only what the §10.13 table and the §11 "
                         "gates require; 'full' adds the §7.6 controls and the "
                         "corroborative tests")
    ap.add_argument("--stream", choices=("delta", "pose", "auto"),
                    default="auto",
                    help="which latent stream the model was fitted on; 'auto' infers it from the stored lengths (§4.3)")
    ap.add_argument("--fast", action="store_true",
                    help="cut resampling counts ~20x for a smoke run")
    ap.add_argument("--band", default="0.5,2.0",
                    help="fidgety band in Hz as LOW,HIGH (default 0.5,2.0). "
                         "Used for the raw-velocity fidgety band ratio.")
    ap.add_argument("--fluency-similarity",
                    choices=("separated", "concatenated", "scalar"),
                    default="separated",
                    help="direction-aware kinematic similarity S feeding the "
                         "fluency measure Phi (§7). 'separated' (preferred, "
                         "default) keeps the magnitude and shape channels apart "
                         "and combines them as omega*S_mag + (1-omega)*S_shape; "
                         "'concatenated' standardises and stacks all 3J "
                         "residuals into one correlation; 'scalar' is the legacy "
                         "log-RMS-speed similarity (magnitude only), i.e. "
                         "omega=1, for the §11 sensitivity check against the "
                         "direction-blind version.")
    ap.add_argument("--fluency-omega", type=float, default=0.5,
                    help="weight of the magnitude channel in the 'separated' "
                         "similarity (§7); omega=1 recovers the scalar measure "
                         "and omega=0 is shape-only. Fixed a priori "
                         "(default 0.5); must lie in [0, 1].")
    ap.add_argument("--fluency-drop-state-term", action="store_true",
                    help="drop the per-state term from the shape residualisation "
                         "(§6): keep only the per-joint anatomical axis removal, "
                         "leaving each state's whole-body drift in. Off by "
                         "default (the state term is kept, matching the "
                         "magnitude channel); pass this for the sensitivity "
                         "variant.")
    ap.add_argument("--skip-raw-kinematics", action="store_true",
                    help="run the fluency (and mixing) analysis only: skip the "
                         "raw-kinematic block -- WCLR-PP inter-limb coordination "
                         "(the synchrony construct, and the slow part of a run), "
                         "the fidgety band ratio, and the per-state velocity "
                         "profiles. The §5-§11 fluency and clinical layer are "
                         "unaffected.")
    ap.add_argument("--wclr-w", type=int, default=50,
                    help="WCLR-PP window width in frames (default 50 = 2 s at "
                         "25 fps). Lean to 62-75 if the autocorrelation length "
                         "exceeds 30 frames.")
    ap.add_argument("--wclr-tau-max", type=int, default=13,
                    help="WCLR-PP maximum lag in frames (default 13 ~ 0.52 s).")
    ap.add_argument("--wclr-c", type=float, default=0.25,
                    help="WCLR-PP delta-R^2 cutoff (default 0.25, a heuristic "
                         "gate; check robustness over {0.2,0.25,0.3}).")
    ap.add_argument("--wclr-ell-min", type=int, default=19,
                    help="WCLR-PP minimum coupled-run length in frames "
                         "(default 19 = 0.76 s).")
    ap.add_argument("--wclr-dtau", type=int, default=1,
                    help="WCLR-PP peak-lag continuity tolerance in frames "
                         "(default 1, the spec value): consecutive windows "
                         "chain into one coupled run only while their peak "
                         "lags stay within this many frames. Raising it "
                         "tolerates more lag wander, so runs survive more "
                         "often and F rises.")
    ap.add_argument("--wclr-limb-signal", choices=sorted(WP.LIMB_SIGNALS),
                    default=WP.DEFAULT_LIMB_SIGNAL,
                    help="which joints define a limb's velocity: "
                         "'end_effector' (wrist/ankle only, the specified "
                         "construct), 'distal' (elbow+wrist, knee+ankle) or "
                         "'limb' (whole chain). 'limb' folds in shoulder and "
                         "hip, whose torso-normalised velocities are close to "
                         "negations of each other across the midline and so "
                         "manufacture coupling in the homologous pairs.")
    ap.add_argument("--fluency-curve", action="store_true",
                    help="also write the exploratory FLUENCY_CURVE temporal "
                         "decomposition of Phi (one panel per recording under "
                         "figures/fluency_curve/, plus results/fluency_curve.csv"
                         "). Reuses the primary model's S(omega) and the cached "
                         "A1 null so the flat-kernel curve equals Phi exactly; "
                         "the reported scalar stays the transition-averaged Phi.")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.outdir, "run.log"))
    t0 = time.time()

    f = args.fast
    cfg = {
        "n_phi": 200 if f else 2000,          # §12.3 occupancy-matched null
        "n_corr": 100 if f else 400,
        "n_dwell": 50 if f else 200,
        "n_mantel": 2_000 if f else 20_000,   # §12.3 Mantel
        "n_block": 100 if f else 400,         # §12.3 block bootstrap
        "n_maxt": 20_000 if f else 200_000,   # §12.3 headline contrasts
        "n_bca": 2_000 if f else 10_000,
        "block": 50, "truncate": 387, "m_max": 6, "duration_tol": 0.3,
        "controls": args.controls, "state_names": args.state_names,
        "n_wclr": 2_000 if f else 20_000,    # WCLR-PP label permutations
        "wclr_w": args.wclr_w, "wclr_tau_max": args.wclr_tau_max,
        "wclr_c": args.wclr_c, "wclr_ell_min": args.wclr_ell_min,
        "wclr_dtau": args.wclr_dtau,
        "wclr_limb_signal": args.wclr_limb_signal,
        "band": tuple(float(x) for x in args.band.split(",")),
        "top_frac": 0.10,                    # high-velocity frame fraction
        "fluency_similarity": args.fluency_similarity,
        "fluency_omega": args.fluency_omega,
        "fluency_shape_state_term": not args.fluency_drop_state_term,
        "skip_raw_kinematics": args.skip_raw_kinematics,
    }
    if not 0.0 <= cfg["fluency_omega"] <= 1.0:
        raise SystemExit(f"--fluency-omega must lie in [0, 1], got "
                         f"{cfg['fluency_omega']}")
    geom = A.Geometry(args.fps, args.clip, args.nwin, args.stride)

    section("§2  Data and verification")
    print(f"  geometry: T={geom.clip} W={geom.n_win} l={geom.l} "
          f"stride={geom.stride} lo={geom.lo} f0={geom.f0} "
          f"f_win={geom.f_win:.2f} Hz")
    vids, pose, obs, rep = build_pose.build(
        args.csv, out=os.path.join(args.outdir, "pose.npz"))
    labels, lab_src = build_pose.load_labels(args.labels, len(vids))
    print(f"  labels from {lab_src}: {int(labels.sum())} positive, "
          f"{int((1 - labels).sum())} negative "
          f"(1-based positives {np.flatnonzero(labels) + 1})")

    frames = [pose[v].shape[0] for v in vids]
    sec = np.array(frames) / geom.fps
    print(f"  §2.3 duration: {min(frames)}-{max(frames)} frames "
          f"({sec.min():.0f}-{sec.max():.0f} s), "
          f"{max(frames) / min(frames):.1f}-fold range")

    wanted = {"arhmm": ("AR-HMM",), "hmm": ("Gaussian HMM",),
              "both": ("AR-HMM", "Gaussian HMM")}[args.models]
    models = {}
    for path, tag in ((args.arhmm, "AR-HMM"), (args.hmm, "Gaussian HMM")):
        if tag not in wanted:
            print(f"  {tag}: skipped (--models {args.models})")
            continue
        if path and os.path.exists(path):
            try:
                models[tag] = L.load_normalised(path, tag)
                m = models[tag]
                print(f"  loaded {tag}: K={m['k']}, {len(m['states']):,} "
                      f"windows, {m['n_subjects']} subjects")
            except Exception as exc:                      # noqa: BLE001
                print(f"  WARNING: {tag} at {path} failed to load: {exc}")
        else:
            print(f"  {tag}: not found at {path} (skipped)")
    if not models:
        raise SystemExit(
            "No model could be loaded. Pass --arhmm/--hmm with valid paths.")

    primary = "AR-HMM" if "AR-HMM" in models else "Gaussian HMM"

    # Which stream was the model fitted on? The delta trajectory is exactly one
    # window per subject shorter than the pose trajectory, so the stored
    # `lengths` identify it (§4.3). Guessing wrong shifts every §5 frame
    # attribution by half a window, which no later check would catch.
    stream = args.stream
    if stream == "auto":
        probe = A.verify_lengths(frames, models[primary]["lengths"], geom) \
            if "lengths" in models[primary] else {"consistent_with": []}
        cons = probe.get("consistent_with", [])
        stream = cons[0] if len(cons) == 1 else "delta"
        print(f"  §4.3 stream inferred from stored lengths: {stream}"
              + ("" if len(cons) == 1 else
                 f"  (WARNING: lengths match {cons or 'neither'} convention; "
                 f"defaulting to delta — pass --stream to override)"))
    else:
        print(f"  stream set explicitly: {stream}")

    results = {"config": cfg, "geometry": vars(geom) | {
        "l": geom.l, "lo": geom.lo, "f0": geom.f0, "f_win": geom.f_win},
        "data_report": rep, "labels": labels, "video_names": vids,
        "frames": frames, "primary": primary, "models_loaded": list(models),
        "stream": stream}

    # §12.4 checks with a definite right answer.
    checks = {
        "frame contiguity (§2.1)": rep["contiguous"],
        "constant joints (§2.2)": rep["constant_joints_as_documented"],
        "label alignment (§2.1)": bool(int(labels.sum()) == 6),
    }
    for tag, m in models.items():
        if "lengths" in m:
            v = A.verify_lengths(frames, m["lengths"], geom, stream)
            checks[f"window-count arithmetic (§4.3) [{tag}]"] = v["match"]
            results[f"lengths_{tag}"] = v
            if not v["match"]:
                print(f"  WARNING §4.3 [{tag}]: predicted vs stored lengths "
                      f"differ in {v['n_mismatch']} subjects. Subject order or "
                      f"geometry is wrong; downstream frame attribution is "
                      f"unsafe.")
    results["checks"] = checks
    print("\n  §12.4 verification")
    for k, v in checks.items():
        print(f"     {'OK  ' if v else 'FAIL'}  {k}")

    spd = A.speeds(pose, vids, geom.fps)
    vel = A.velocities(pose, vids, geom.fps)          # signed vectors for §4 shape

    for tag, m in models.items():
        section(f"§5-§9  {tag}")
        if m["n_subjects"] != len(vids):
            print(f"  WARNING: model has {m['n_subjects']} subjects but the CSV "
                  f"has {len(vids)} videos; alignment is assumed by sort order.")
        results[tag] = analyse_model(m, vids, pose, spd, vel, labels, geom, cfg,
                                     tag, stream)

    section(f"§10-§11  Clinical layer  ({primary})")
    m = models[primary]
    res = results[primary]
    results["clinical"] = clinical(
        res, labels, m["lengths"] if "lengths" in m
        else np.bincount(m["vidid"]), cfg, geom, m["states"], m["vidid"],
        res["S"], m["n_subjects"])
    results["_halves"] = results["clinical"].pop("phi_halves", None)

    # §7.6 replication on the independent model.
    if len(models) > 1:
        other = [t for t in models if t != primary][0]
        section(f"§7.6  Replication on the independent model ({other})")
        r2 = results[other]
        rho, p = stats.spearmanr(res["phi"]["excess"], r2["phi"]["excess"])
        results["replication"] = {"phi_spearman": {"rho": float(rho),
                                                   "p": float(p)}}
        print(f"  Phi agreement across models: Spearman rho = {rho:+.3f} "
              f"(p = {p:.3g})")

    # ---- raw-kinematic constructs (§8): WCLR-PP inter-limb coordination (the
    # synchrony construct and the slow part of a run), the fidgety band ratio,
    # and the per-state velocity profiles. A third estimator alongside fluency
    # and mixing; omitted for a fluency-only run. See raw_kinematics().
    if cfg["skip_raw_kinematics"]:
        section("Raw kinematics: skipped (--skip-raw-kinematics)")
        print("  WCLR-PP synchrony, fidgety band ratio and per-state velocity "
              "profiles omitted; the fluency and clinical layers above are "
              "unaffected.")
    else:
        raw_kinematics(results, models, primary, pose, vids, labels, geom, cfg)

    # ---- plain-language summary of every test that was run ----
    section("Statistical tests performed")
    cl = results["clinical"]
    r_res = results[primary]
    w = r_res["phi_wilcoxon"]
    sh = cl["phi_split_half"]
    rows = [
        ("Is fluency above zero within each infant?",
         "Wilcoxon signed-rank on Phi, one-sided", w["p"],
         f"{w['n_positive']}/{w['n']} infants positive"),
        ("Does fluency reproduce across the recording?",
         "Spearman on contiguous halves, Spearman-Brown corrected",
         sh.get("p_half"), f"r_SB = {sh['r_sb']:.3f}"),
        ("Do abnormal infants differ in fluency?",
         "Mann-Whitney, exact enumeration of all label assignments",
         cl["phi_test"]["p"], f"AUC = {cl['phi_test']['auc']:.3f}"),
        ("Do abnormal infants differ in mixing time?",
         "Mann-Whitney, exact enumeration of all label assignments",
         cl["kemeny_test"]["p"], f"AUC = {cl['kemeny_test']['auc']:.3f}"),
        ("Fluency, after correcting for testing two endpoints",
         "maxT over the pair, resampled jointly",
         cl["maxt"]["phi"]["p_maxT"], f"Holm {cl['holm']['phi']:.3g}"),
        ("Mixing time, after correcting for testing two endpoints",
         "maxT over the pair, resampled jointly",
         cl["maxt"]["kemeny"]["p_maxT"], f"Holm {cl['holm']['kemeny']:.3g}"),
    ]
    for nm, meth in (("phi", "Fluency"), ("kemeny", "Mixing time")):
        if nm in cl.get("adjusted", {}):
            rows.append((f"{meth}, with recording length partialled out",
                         "Freedman-Lane residual permutation",
                         cl["adjusted"][nm]["p"],
                         f"AUC = {cl['adjusted'][nm]['auc']:.3f}"))
    for q, meth, pv, extra in rows:
        pstr = "n/a" if pv is None or not np.isfinite(pv) else f"{pv:.4g}"
        print(f"  {q}\n      {meth}\n      p = {pstr}   ({extra})")
    print(f"\n  Effect sizes are AUC (= probability a random abnormal infant "
          f"exceeds a random normal one).")
    print(f"  AUC intervals are reported two ways: the Hanley-McNeil normal "
          f"approximation, and\n  a percentile bootstrap resampling each group "
          f"separately ({cl['phi_test'].get('n1', 0)} and "
          f"{cl['phi_test'].get('n2', 0)} infants). The normal approximation "
          f"is\n  unreliable at this sample size and is clipped at 0 and 1; "
          f"prefer the bootstrap.")
    for nm, key in (("Fluency", "phi_test"), ("Kemeny", "kemeny_test")):
        r = cl[key]
        print(f"     {nm:8s} AUC {r['auc']:.3f}  normal-approx "
              f"[{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]"
              f"{'  (CLIPPED)' if r.get('hm_clipped') else ''}"
              f"   bootstrap [{r.get('auc_lo_boot', float('nan')):.3f}, "
              f"{r.get('auc_hi_boot', float('nan')):.3f}]")
    print(f"  Smallest AUC this study could detect at 80% power: "
          f"{cl['mde']['auc']:.3f}")

    # ---- outputs ----
    section("Outputs")
    import pandas as pd
    cl = results["clinical"]
    per = pd.DataFrame({
        "subject": np.arange(1, len(vids) + 1), "video": vids,
        "label": labels, "n_windows": res["phi"]["n_visits"] * 0
        + (m["lengths"] if "lengths" in m else np.bincount(m["vidid"])),
        "n_visits": res["phi"]["n_visits"],
        "phi_excess": res["phi"]["excess"], "phi_observed": res["phi"]["observed"],
        "kemeny_jumps": res["kemeny_per_subject"],
        "kemeny_lo": res["kemeny_lo"], "kemeny_hi": res["kemeny_hi"],
        "occupancy_entropy": cl["occupancy_entropy"],
        "mean_dwell_windows": cl["mean_dwell"]})
    per.to_csv(os.path.join(args.outdir, "per_subject.csv"), index=False)
    np.savetxt(os.path.join(args.outdir, "similarity_matrix.csv"),
               res["S"], delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(args.outdir, "state_amplitude_profile.csv"),
               res["amplitude"], delimiter=",", fmt="%.6f")
    for nm, key in (("magnitude", "S_mag"), ("shape", "S_shape")):
        if key in res:
            np.savetxt(os.path.join(args.outdir,
                                    f"similarity_matrix_{nm}.csv"),
                       np.asarray(res[key]), delimiter=",", fmt="%.6f")

    # direction-aware shape signature (§4), long format on the free joints:
    # anisotropy rho = ||u|| and principal-axis angle theta = angle(u)/2.
    u_arr = np.asarray(res["shape_coord"])
    shp_rows = []
    for k in range(res["k"]):
        for j in build_pose.FREE:
            u1, u2 = float(u_arr[k, j, 0]), float(u_arr[k, j, 1])
            shp_rows.append({
                "state": k, "state_label": res["state_labels"][k],
                "joint": build_pose.JOINTS[j], "rho": float(np.hypot(u1, u2)),
                "theta_deg": float(0.5 * np.degrees(np.arctan2(u2, u1))),
                "u1": u1, "u2": u2})
    pd.DataFrame(shp_rows).to_csv(
        os.path.join(args.outdir, "state_shape_profile.csv"), index=False)

    with open(os.path.join(args.outdir, "results.json"), "w") as fh:
        json.dump(_json_safe(results), fh, indent=1)
    print(f"  wrote per_subject.csv, similarity_matrix.csv, "
          f"similarity_matrix_{{magnitude,shape}}.csv, "
          f"state_amplitude_profile.csv, state_shape_profile.csv, results.json")

    # FLUENCY_CURVE: exploratory temporal decomposition of Phi. Reuses the
    # primary model's S(omega), its run-length-compressed visit sequences and
    # the cached §7.2 null so the flat-kernel curve equals Phi (Prop 1). The
    # reported scalar is unchanged; this only unrolls it along the transition
    # index. Opt-in because it writes one figure per recording.
    if args.fluency_curve:
        section("FLUENCY_CURVE  temporal decomposition of Phi (exploratory)")
        import fluency_curve as FCV
        FCV.fluency_curves(
            m["states"], m["vidid"], np.asarray(res["S"]), vids, labels,
            phi=res["phi"], geom=geom, sigmas=(3, 5), out_root=args.outdir,
            label_names=("normal", "abnormal"))

    if not args.no_figures:
        import figures
        made = figures.make_all(results, args.outdir)
        for p in made:
            print(f"  figure: {p}")

    print(f"\ndone in {time.time() - t0:.1f} s -> {args.outdir}/")
    return results


if __name__ == "__main__":
    main()
