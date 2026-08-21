"""One call that runs the whole RVI-38 analysis and returns every result.

    from report import run_report
    out = run_report(csv="rvi38_analysis.csv",
                     arhmm="arhmm_rvi38_stream_delta.pkl",
                     hmm="hmm_rvi38_stream_delta.pkl",
                     labels="RVI_38_labels.mat", outdir="rvi38_out")

Five constructs, in the order they are reported:

1. **Fluency** ``Phi`` -- how kinematically alike consecutive movements are,
   above what the infant's own state occupancy would give by chance.
2. **The fluency curve** -- that same average unrolled along the recording, one
   panel per infant, so a scalar becomes a time course.
3. **Mixing** -- the Kemeny constant of the state chain, the expected time to
   reach a randomly drawn state: how long the movement repertoire takes to get
   anywhere.
4. **Synchrony** -- WCLR-PP inter-limb coordination, the fraction of time each
   limb pair moves as one, which is the cramped-synchronised pole.
5. **FidgetyFind** -- the published detector (Morais et al., 2023) of the
   fidgety movements the GMA label is about, computed from the keypoints alone.
   It is the external yardstick the four constructs above are read against.

Everything is computed by :mod:`run_analysis`; this module chooses the settings
that produce the full picture (the fluency curve and the per-recording
FidgetyFind panels are on), collects the headline numbers of each block into
one summary, writes ``summary.json`` and ``summary.md`` next to the run, and --
in a notebook -- displays the figures.

``run_report`` returns a dict:

* ``results`` -- the full results object of the run (also on disk as
  ``results.json``);
* ``summary`` -- the headline numbers per construct, the same content as
  ``summary.md``;
* ``figures`` -- every figure written, grouped by construct;
* ``outdir``, ``log`` -- where it all landed.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_analysis                                        # noqa: E402

# Which figures belong to which construct, in reading order. Entries ending in
# ``/`` are directories of per-recording panels.
FIGURE_GROUPS: dict[str, tuple[str, ...]] = {
    "fluency": ("state_signature", "similarity_matrix", "similarity_channels",
                "fluency_per_subject", "fluency_by_dwell"),
    "fluency_curve": ("fluency_curve/",),
    "kemeny": ("degenerate_statistics", "mfpt_matrix", "graph_companions",
               "kemeny_per_subject", "shrinkage_selection"),
    "synchrony": ("wclrpp_pairs", "wclrpp_summary"),
    "fidgetyfind": ("fidgetyfind_subject", "fidgetyfind_chains",
                    "fidgetyfind_windows", "fidgetyfind/"),
    "clinical": ("auc_effect_sizes", "correlations",
                 "state_velocity_regions", "state_velocity_lateral"),
}
PER_RECORDING = ("fluency_curve/", "fidgetyfind/")


def _f(v, default=float("nan")):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def _auc_row(t: dict | None) -> dict:
    """AUC, its bootstrap interval and its exact p, from :func:`mannwhitney`."""
    if not t:
        return {}
    return {"auc": _f(t.get("auc")), "p": _f(t.get("p")),
            "auc_ci": [_f(t.get("auc_lo")), _f(t.get("auc_hi"))],
            "n_pos": int(t.get("n1", 0)), "n_neg": int(t.get("n2", 0)),
            "method": t.get("method", ""),
            "ci_method": t.get("ci_method", "")}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarise(results: dict, outdir: str) -> dict:
    """The headline numbers of every construct, in one dict."""
    primary = results.get("primary")
    res = results.get(primary, {})
    cl = results.get("clinical", {})
    loaded = list(results.get("models_loaded", []))
    out: dict = {"cohort": {
        "n_recordings": len(results.get("video_names", [])),
        "n_abnormal": int(np.sum(np.asarray(results.get("labels", [])))),
        "primary_model": primary, "k_states": res.get("k"),
        "models": [{"name": nm, "k": results.get(nm, {}).get("k"),
                    "path": results.get("model_paths", {}).get(nm),
                    "primary": nm == primary} for nm in loaded],
        "replication": {nm: {"phi_rho": _f(v.get("phi_spearman", {})
                                           .get("rho")),
                             "phi_p": _f(v.get("phi_spearman", {}).get("p")),
                             "kemeny_rho": _f(v.get("kemeny_spearman", {})
                                              .get("rho")),
                             "kemeny_p": _f(v.get("kemeny_spearman", {})
                                            .get("p"))}
                        for nm, v in (results.get("replication") or {}).items()},
        "stream": results.get("stream"),
        "fps": results.get("geometry", {}).get("fps")}}

    # 1. fluency -------------------------------------------------------------
    phi = np.asarray(res.get("phi", {}).get("excess", []), float)
    y = np.asarray(results.get("labels", []), int)
    fl = {"phi_median": _f(np.median(phi)) if phi.size else float("nan"),
          "phi_positive": (f"{int(np.sum(phi > 0))}/"
                           f"{int(np.isfinite(phi).sum())}") if phi.size
          else "0/0",
          "group": _auc_row(cl.get("phi_test"))}
    if phi.size and y.size == phi.size:
        fl["phi_median_normal"] = _f(np.median(phi[y == 0]))
        fl["phi_median_abnormal"] = _f(np.median(phi[y == 1]))
    ch = cl.get("channel_split", {})
    if ch:
        fl["channel_split"] = {k: {"auc": _f(v.get("auc")), "p": _f(v.get("p"))}
                               for k, v in ch.items()}
    # Sensitivity: Phi under the unconstrained permutation null. Not the
    # reported statistic -- it reorders the visits over all n! permutations,
    # which admits adjacent-equal pairs a visit sequence cannot contain -- but
    # carried here so the size of the correction is on the record rather than
    # asserted. The offset is per-recording, so its spread is the point.
    ph = res.get("phi", {})
    uni = np.asarray(ph.get("excess_uniform", []), float)
    gap = (np.asarray(ph.get("null_uniform", []), float)
           - np.asarray(ph.get("null_mean", []), float))
    if uni.size:
        fl["uniform_null_sensitivity"] = {
            "phi_median": _f(np.nanmedian(uni)),
            "null_offset_median": _f(np.nanmedian(gap)),
            "null_offset_range": [_f(np.nanmin(gap)), _f(np.nanmax(gap))],
            "repeat_rate_median": _f(np.nanmedian(
                np.asarray(ph.get("null_repeat_rate", []), float))),
            "note": ("Phi under the unconstrained permutation null, for "
                     "comparison only; the reported Phi reorders the visits "
                     "without ever repeating a state")}
    out["fluency"] = fl

    # 2. fluency curve -------------------------------------------------------
    csv = os.path.join(outdir, "results", "fluency_curve.csv")
    panels = sorted(glob.glob(os.path.join(outdir, "figures", "fluency_curve",
                                           "*.png")))
    out["fluency_curve"] = {
        "written": bool(panels), "n_panels": len(panels),
        "csv": csv if os.path.exists(csv) else None,
        "sigmas_transitions": [3, 5],
        "note": ("the flat-kernel limit of the curve is Phi itself; the run "
                 "asserts that identity before plotting")}

    # 3. mixing --------------------------------------------------------------
    kem = np.asarray(res.get("kemeny_per_subject", []), float)
    cf, cj = res.get("companions_full", {}), res.get("companions_jump", {})
    km = {"kemeny_full_windows": _f(cf.get("kemeny")),
          "kemeny_full_seconds": _f(cf.get("kemeny_seconds")),
          "kemeny_jump_chain": _f(cj.get("kemeny")),
          "per_subject_median": _f(np.median(kem)) if kem.size else float("nan"),
          "shrinkage_alpha": _f(res.get("alpha", {}).get("alpha")),
          "identity_check": _f(res.get("kemeny_identity", {}).get("abs_diff")),
          "group": _auc_row(cl.get("kemeny_test"))}
    if kem.size and y.size == kem.size:
        km["median_normal"] = _f(np.median(kem[y == 0]))
        km["median_abnormal"] = _f(np.median(kem[y == 1]))
    out["kemeny"] = km

    # 4. synchrony -----------------------------------------------------------
    wc = results.get("wclrpp")
    if wc:
        F = np.asarray(wc.get("F", []), float)
        pairs = list(wc.get("pairs", []))
        ptests = wc.get("pair_tests") or []
        per_pair = []
        for i, nm in enumerate(pairs):
            row = _auc_row(ptests[i] if i < len(ptests) else None)
            row.update({
                "pair": nm, "class": list(wc.get("pair_class", []))[i]
                if wc.get("pair_class") is not None else "",
                "median_normal": _f(np.nanmedian(F[y == 0, i]))
                if F.size else float("nan"),
                "median_abnormal": _f(np.nanmedian(F[y == 1, i]))
                if F.size else float("nan")})
            row["delta"] = _f(row["median_abnormal"] - row["median_normal"])
            per_pair.append(row)
        out["synchrony"] = {
            "limb_signal": wc.get("params", {}).get("limb_signal"),
            "window_frames": wc.get("params", {}).get("w"),
            "per_pair": per_pair,
            "whole_body": _auc_row(wc.get("mean_F_test"))}
    else:
        out["synchrony"] = {"skipped": True}

    # 5. FidgetyFind ---------------------------------------------------------
    ff = results.get("fidgetyfind")
    if ff:
        cov = np.asarray(ff.get("coverage", []), float)
        chains = list(ff.get("chains", []))
        per_chain = []
        for i, nm in enumerate(chains):
            per_chain.append({
                "chain": nm,
                "class": list(ff.get("chain_class", []))[i]
                if ff.get("chain_class") is not None else "",
                "assessable": _f(np.nanmedian(cov[:, i]))
                if cov.size else float("nan")})
        cal = ff.get("calibration", {})
        endpoints = {}
        for g in ("FF", "FF_hip", "FF_dist"):
            v = np.asarray(ff.get(g, []), float)
            row = _auc_row(ff.get("tests", {}).get(g))
            row.update({
                "median_normal": _f(np.nanmedian(v[y == 0])) if v.size
                else float("nan"),
                "median_abnormal": _f(np.nanmedian(v[y == 1])) if v.size
                else float("nan"),
                "n_scored": int(np.isfinite(v).sum()) if v.size else 0,
                "n_total": int(v.size)})
            endpoints[g] = row
        out["fidgetyfind"] = {
            "endpoints": endpoints,
            "group": endpoints.get("FF", {}),
            "per_chain": per_chain,
            "band": ff.get("band_label", ""),
            "calibration": {"scale": _f(cal.get("scale")),
                            "q75": _f(cal.get("q75")),
                            "percentile": _f(cal.get("percentile")),
                            "window": cal.get("window"),
                            "window_reached": bool(cal.get("window_reached",
                                                           False))},
            "window_frames": ff.get("params", {}).get("window"),
            "stride": ff.get("params", {}).get("stride"),
            "nu": _f(ff.get("params", {}).get("nu")),
            "direction": ("AUC below 0.5 is the expected direction: absent "
                          "fidgety movement means less direction variety")}
    else:
        out["fidgetyfind"] = {"skipped": True}

    out["correlations"] = results.get("correlations", {})
    out["checks"] = results.get("checks", {})
    return out


def _fmt(v, nd=3):
    x = _f(v)
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"


def _fmt_p(v):
    x = _f(v)
    return "n/a" if not np.isfinite(x) else (f"{x:.2e}" if x < 1e-3
                                             else f"{x:.4f}")


def summary_markdown(s: dict) -> str:
    """The summary as a short report, in the order the constructs are read."""
    c = s["cohort"]
    L = ["# RVI-38 analysis summary",
         "",
         f"{c['n_recordings']} recordings, {c['n_abnormal']} labelled abnormal "
         f"(absent fidgety movement), on the {c['stream']} stream at "
         f"{c['fps']} fps.",
         ""]
    models = c.get("models") or []
    if models:
        L += ["| model | states | role | file |", "|---|---|---|---|"]
        for m in models:
            L += [f"| {m['name']} | {m.get('k', '')} | "
                  f"{'primary' if m['primary'] else 'replication'} | "
                  f"`{m.get('path') or ''}` |"]
        L += [""]
    rep = c.get("replication") or {}
    if rep:
        L += ["Agreement with the primary model, across the whole cohort "
              "(Spearman): "
              + "; ".join(f"**{nm}** Phi {_fmt(v['phi_rho'], 2)} "
                          f"(p {_fmt_p(v['phi_p'])}), Kemeny "
                          f"{_fmt(v['kemeny_rho'], 2)} "
                          f"(p {_fmt_p(v['kemeny_p'])})"
                          for nm, v in rep.items()),
              ""]
    L += ["Every endpoint is one scalar per recording, contrasted between the "
          "groups with the exact Mann-Whitney permutation null over all "
          "C(38,6) = 2,760,681 label assignments. The effect size is the AUC "
          "(the probability that a random abnormal infant exceeds a random "
          "normal one; 0.5 is no separation), the p-value is two-sided, and "
          "the interval is the percentile interval of a stratified "
          "nonparametric bootstrap. Nothing is corrected for multiplicity, no "
          "nuisance is partialled out of a contrast, and no endpoint has to "
          "clear an admission gate to be reported; the nuisances are reported "
          "as correlations, where the label enters no fit.",
          ""]

    f = s["fluency"]
    g = f.get("group", {})
    L += ["## 1. Fluency (Phi)", "",
          f"- within infants: Phi > 0 in {f['phi_positive']}, median "
          f"{_fmt(f['phi_median'])}",
          f"- normal {_fmt(f.get('phi_median_normal'))} vs abnormal "
          f"{_fmt(f.get('phi_median_abnormal'))}",
          f"- group contrast: AUC {_fmt(g.get('auc'))} "
          f"[{_fmt(g.get('auc_ci', [np.nan])[0])}, "
          f"{_fmt(g.get('auc_ci', [np.nan, np.nan])[1])}], "
          f"p = {_fmt_p(g.get('p'))} ({g.get('method', '')})"]
    if "channel_split" in f:
        parts = ", ".join(f"{k} AUC {_fmt(v['auc'])} (p {_fmt_p(v['p'])})"
                          for k, v in f["channel_split"].items())
        L += [f"- magnitude vs direction: {parts}"]
    sens = f.get("uniform_null_sensitivity")
    if sens:
        lo, hi = sens["null_offset_range"]
        L += [f"- null: the visits are reordered without ever repeating a "
              f"state, the constraint the visit sequence itself carries. "
              f"*Sensitivity* — reordering over all `n!` permutations instead "
              f"admits adjacent-equal pairs ({_fmt(sens['repeat_rate_median'])} "
              f"of them at the median), each worth `S_kk = 1`, which lifts the "
              f"null by {_fmt(sens['null_offset_median'])} at the median and "
              f"by between {_fmt(lo)} and {_fmt(hi)} across recordings; median "
              f"Phi under it is {_fmt(sens['phi_median'])}. That offset varies "
              f"with each infant's own occupancy concentration, which is what "
              f"the reported null holds fixed."]
    L += [""]

    fc = s["fluency_curve"]
    L += ["## 2. Fluency curve", "",
          f"- {fc['n_panels']} per-recording panels"
          + (f", table at `{os.path.basename(fc['csv'])}`" if fc.get("csv")
             else ""),
          f"- smoothing sigma {fc['sigmas_transitions']} transitions; "
          f"{fc['note']}", ""]

    k = s["kemeny"]
    g = k.get("group", {})
    L += ["## 3. Mixing (Kemeny constant)", "",
          f"- cohort chain: {_fmt(k['kemeny_full_windows'], 2)} windows = "
          f"{_fmt(k['kemeny_full_seconds'], 2)} s; jump chain "
          f"{_fmt(k['kemeny_jump_chain'], 2)} jumps",
          f"- per infant (jump chain): median "
          f"{_fmt(k['per_subject_median'], 2)}, normal "
          f"{_fmt(k.get('median_normal'), 2)} vs abnormal "
          f"{_fmt(k.get('median_abnormal'), 2)}",
          f"- group contrast: AUC {_fmt(g.get('auc'))} "
          f"[{_fmt(g.get('auc_ci', [np.nan])[0])}, "
          f"{_fmt(g.get('auc_ci', [np.nan, np.nan])[1])}], "
          f"p = {_fmt_p(g.get('p'))}",
          f"- shrinkage alpha {_fmt(k['shrinkage_alpha'])}, spectral "
          f"identity |diff| = {k['identity_check']:.1e}", ""]

    sy = s["synchrony"]
    L += ["## 4. Synchrony (WCLR-PP inter-limb coordination)", ""]
    if sy.get("skipped"):
        L += ["- skipped", ""]
    else:
        g = sy.get("whole_body", {})
        L += [f"- limb signal `{sy['limb_signal']}`, window "
              f"{sy['window_frames']} frames",
              f"- whole-body coupling (mean F): AUC {_fmt(g.get('auc'))}, "
              f"p = {_fmt_p(g.get('p'))}",
              "", "| pair | class | normal | abnormal | AUC [95% CI] | p |",
              "|---|---|---|---|---|---|"]
        for r in sy["per_pair"]:
            ci = r.get("auc_ci", [np.nan, np.nan])
            L += [f"| {r['pair']} | {r['class']} | "
                  f"{_fmt(r['median_normal'])} | {_fmt(r['median_abnormal'])} "
                  f"| {_fmt(r.get('auc'))} [{_fmt(ci[0])}, {_fmt(ci[1])}] "
                  f"| {_fmt_p(r.get('p'))} |"]
        L += [""]

    ff = s["fidgetyfind"]
    L += ["## 5. FidgetyFind (Morais et al., 2023)", ""]
    if ff.get("skipped"):
        L += ["- skipped", ""]
    else:
        cal = ff.get("calibration", {})
        L += [f"- {ff.get('band', '')}: the published ladder rescaled by "
              f"varsigma = {_fmt(cal.get('scale'))}, from the "
              f"{_fmt(cal.get('percentile'), 0)}th percentile "
              f"{_fmt(cal.get('q75'), 2)}% of the pooled per-frame amplitudes",
              f"- window L = {ff.get('window_frames')} frames, stride "
              f"{ff.get('stride')}, nu = {_fmt(ff.get('nu'))}"
              + ("" if cal.get("window_reached") else
                 " (no window length in the grid reached ten in-band frames, "
                 "so the longest was taken)"),
              f"- {ff['direction']}",
              "",
              "| endpoint | normal | abnormal | AUC [95% CI] | p | scored |",
              "|---|---|---|---|---|---|"]
        for g, lab in (("FF", "FF (all six chains)"),
                       ("FF_hip", "FF_hip (hip chains)"),
                       ("FF_dist", "FF_dist (limb chains)")):
            r = ff.get("endpoints", {}).get(g, {})
            ci = r.get("auc_ci", [np.nan, np.nan])
            L += [f"| {lab} | {_fmt(r.get('median_normal'))} | "
                  f"{_fmt(r.get('median_abnormal'))} | "
                  f"{_fmt(r.get('auc'))} [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
                  f"{_fmt_p(r.get('p'))} | "
                  f"{r.get('n_scored', 0)}/{r.get('n_total', 0)} |"]
        L += ["",
              "Windows in which each chain's amplitude gate left it assessable "
              "(median over recordings):", "",
              "| chain | class | assessable |", "|---|---|---|"]
        for r in ff["per_chain"]:
            L += [f"| {r['chain']} | {r['class']} | "
                  f"{_fmt(r['assessable'], 2)} |"]
        L += [""]

    co = s.get("correlations") or {}
    if co.get("table"):
        covs = co["covariates"]
        L += ["## Correlation analysis", "",
              f"Reported, not adjusted for; the label enters no fit. State "
              f"quantities come from {co.get('model', 'the primary model')}.",
              "",
              "| endpoint | " + " | ".join(covs) + " |",
              "|---" * (len(covs) + 1) + "|"]
        for e, row in co["table"].items():
            cells = []
            for cv in covs:
                r = row.get(cv, {})
                cells.append(f"r {_fmt(r.get('pearson_r'), 2)} "
                             f"(p {_fmt_p(r.get('pearson_p'))}), "
                             f"rho {_fmt(r.get('spearman_rho'), 2)} "
                             f"(p {_fmt_p(r.get('spearman_p'))})")
            L += [f"| {e} | " + " | ".join(cells) + " |"]
        L += [""]

    return "\n".join(L)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def collect_figures(outdir: str) -> dict:
    """Every figure the run wrote, grouped by construct and in reading order."""
    fdir = os.path.join(outdir, "figures")
    found: dict[str, list[str]] = {}
    for group, names in FIGURE_GROUPS.items():
        paths: list[str] = []
        for nm in names:
            if nm.endswith("/"):
                paths += sorted(glob.glob(os.path.join(fdir, nm[:-1], "*.png")))
            else:
                p = os.path.join(fdir, f"{nm}.png")
                if os.path.exists(p):
                    paths.append(p)
        if paths:
            found[group] = paths
    listed = {p for ps in found.values() for p in ps}
    other = [p for p in sorted(glob.glob(os.path.join(fdir, "*.png")))
             if p not in listed]
    if other:
        found["other"] = other
    return found


def is_per_recording(path: str) -> bool:
    """Is this one of the per-recording panels rather than a cohort figure?"""
    return os.path.basename(os.path.dirname(path)) in {d.rstrip("/")
                                                       for d in PER_RECORDING}


def show_figures(figures: dict, which: str = "main", width: int = 900) -> int:
    """Display the figures inline in a notebook. Returns how many were shown.

    ``which`` is ``"main"`` (the cohort figures, the default), ``"all"`` (also
    every per-recording panel -- dozens of images) or a group name from
    :data:`FIGURE_GROUPS`.
    """
    try:
        from IPython.display import Image, Markdown, display
    except ImportError:                                   # noqa: BLE001
        print("  (not in a notebook: figures are on disk, nothing displayed)")
        return 0
    groups = [which] if which in FIGURE_GROUPS else list(figures)
    shown = 0
    for group in groups:
        paths = figures.get(group, [])
        if not paths:
            continue
        display(Markdown(f"### {group.replace('_', ' ')}"))
        panels = [p for p in paths if is_per_recording(p)]
        for p in paths:
            if which == "main" and is_per_recording(p):
                continue
            display(Image(filename=p, width=width))
            shown += 1
        if which == "main" and panels:
            display(Markdown(f'*{len(panels)} per-recording panels written but '
                             f'not shown; `show="all"` displays them, and they '
                             f'are on disk either way.*'))
    return shown


# ---------------------------------------------------------------------------
# the one call
# ---------------------------------------------------------------------------
def model_flags(models=None, arhmm=None, hmm=None) -> list[str]:
    """Turn the ``models`` argument into ``--model`` flags for the runner.

    Accepts whatever is natural to write: one path, a list of paths, a list of
    ``(name, path)`` pairs, or a ``{name: path}`` dict. Unnamed models are
    named after their file, which is what keeps two fits of the same kind
    apart. ``arhmm`` and ``hmm`` are the old two-model shorthands.
    """
    if models is None:
        specs = []
    elif isinstance(models, str):
        specs = [(None, models)]
    elif isinstance(models, dict):
        specs = list(models.items())
    else:
        specs = []
        for m in models:
            if isinstance(m, str):
                specs.append((None, m))
            else:
                pair = tuple(m)
                if len(pair) != 2:
                    raise ValueError(f"expected a (name, path) pair, got {m!r}")
                specs.append(pair)
    flags: list[str] = []
    for name, path in specs:
        flags += ["--model", f"{name}={path}" if name else str(path)]
    if arhmm:
        flags += ["--arhmm", str(arhmm)]
    if hmm:
        flags += ["--hmm", str(hmm)]
    return flags


def run_report(csv: str = "rvi38_analysis.csv",
               models=None,
               primary: str | None = None,
               arhmm: str | None = None,
               hmm: str | None = None,
               labels: str | None = "RVI_38_labels.mat",
               outdir: str = "rvi38_out",
               fast: bool = False,
               fps: float = 25.0,
               stream: str = "auto",
               controls: str = "full",
               synchrony: bool = True,
               fidgetyfind: bool = True,
               fluency_curve: bool = True,
               fidgetyfind_panels: bool = True,
               figures: bool = True,
               show: str | bool = False,
               extra_args: list | None = None,
               **overrides) -> dict:
    """Run every construct and return results, summary and figure paths.

    ``models`` says which fitted models to analyse, and takes whatever is
    natural to write::

        models="arhmm_k11.pkl"                        # one model
        models=["arhmm_k11.pkl", "arhmm_k14.pkl"]     # two, named after files
        models={"K=11": "arhmm_k11.pkl",              # two, named by you
                "K=14": "arhmm_k14.pkl"}

    They may be any mix of kinds -- two AR-HMMs, an AR-HMM and a Gaussian HMM,
    five of anything. The first is the primary model (the clinical layer and
    the figures are computed on it) unless ``primary`` names another; every
    other is a replication, its fluency and mixing time correlated against the
    primary's. With nothing given, the two legacy default filenames are tried.

    The remaining parameters mirror the command line of :mod:`run_analysis`;
    the defaults are the ones that produce the complete picture. Set
    ``fast=True`` for a smoke run (every resampling count is cut ~20x, so the
    p-values are coarse), and ``synchrony=False`` to skip WCLR-PP, which is by
    far the slowest block.

    ``show`` displays the figures when called from a notebook: ``"main"`` for
    the cohort figures, ``"all"`` to include every per-recording panel, or a
    single group name (``"fluency"``, ``"fluency_curve"``, ``"kemeny"``,
    ``"synchrony"``, ``"fidgetyfind"``, ``"clinical"``).

    Any further keyword goes straight through to the runner:
    ``fluency_omega=0.7`` becomes ``--fluency-omega 0.7`` and
    ``wclr_limb_signal="distal"`` becomes ``--wclr-limb-signal distal``. A
    boolean keyword becomes a bare flag when it is true.
    """
    argv = ["--csv", str(csv), "--outdir", str(outdir), "--fps", str(fps),
            "--stream", str(stream), "--controls", str(controls)]
    argv += model_flags(models, arhmm, hmm)
    if primary:
        argv += ["--primary", str(primary)]
    if labels:
        argv += ["--labels", str(labels)]
    if fast:
        argv += ["--fast"]
    if fluency_curve:
        argv += ["--fluency-curve"]
    if not synchrony:
        argv += ["--skip-raw-kinematics"]
    if not fidgetyfind:
        argv += ["--skip-fidgetyfind"]
    if fidgetyfind and fidgetyfind_panels:
        argv += ["--ff-panels"]
    if not figures:
        argv += ["--no-figures"]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv += [flag]
        elif value is not None:
            argv += [flag, str(value)]
    argv += list(extra_args or [])

    prev_stdout = sys.stdout
    try:
        results = run_analysis.main(argv)
    finally:
        tee = sys.stdout
        if isinstance(tee, run_analysis.Tee):
            tee.close()
        sys.stdout = prev_stdout

    summary = summarise(results, outdir)
    figs = collect_figures(outdir)
    summary["figures"] = {k: [os.path.relpath(p, outdir) for p in v]
                          for k, v in figs.items()}

    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(run_analysis._json_safe(summary), fh, indent=1)
    md = summary_markdown(summary)
    with open(os.path.join(outdir, "summary.md"), "w") as fh:
        fh.write(md + "\n")

    print(md)
    n_figs = sum(len(v) for v in figs.values())
    print(f"\n{n_figs} figures under {os.path.join(outdir, 'figures')}/ "
          f"({', '.join(f'{k}: {len(v)}' for k, v in figs.items())})")
    print(f"run log: {os.path.join(outdir, 'run.log')}; "
          f"summary: {os.path.join(outdir, 'summary.md')}")

    if show:
        show_figures(figs, "main" if show is True else str(show))

    return {"results": results, "summary": summary, "figures": figs,
            "outdir": outdir, "log": os.path.join(outdir, "run.log"),
            "markdown": md}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="rvi38_analysis.csv")
    ap.add_argument("--model", action="append", metavar="[NAME=]PATH",
                    help="a fitted model to analyse; repeat for as many as "
                         "you like. The first is the primary model.")
    ap.add_argument("--primary", metavar="NAME")
    ap.add_argument("--labels", default="RVI_38_labels.mat")
    ap.add_argument("--outdir", default="rvi38_out")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-synchrony", action="store_true")
    a = ap.parse_args()
    spec = [tuple(m.split("=", 1)) if "=" in m else m for m in (a.model or [])]
    run_report(csv=a.csv, models=spec or None, primary=a.primary,
               labels=a.labels, outdir=a.outdir, fast=a.fast,
               synchrony=not a.no_synchrony)
