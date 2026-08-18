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
                "fluency_per_subject", "fluency_by_dwell",
                "split_half_reliability"),
    "fluency_curve": ("fluency_curve/",),
    "kemeny": ("degenerate_statistics", "mfpt_matrix", "graph_companions",
               "kemeny_per_subject", "shrinkage_selection"),
    "synchrony": ("wclrpp_pairs", "wclrpp_summary"),
    "fidgetyfind": ("fidgetyfind_subject", "fidgetyfind_chains",
                    "fidgetyfind_windows", "fidgetyfind_agreement",
                    "fidgetyfind/"),
    "clinical": ("auc_effect_sizes", "loo_stability", "redundancy", "gates",
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
    """AUC, its bootstrap interval and p, from a :func:`a1_stats.mannwhitney`."""
    if not t:
        return {}
    return {"auc": _f(t.get("auc")), "p": _f(t.get("p")),
            "auc_ci": [_f(t.get("auc_lo_boot")), _f(t.get("auc_hi_boot"))],
            "n_pos": int(t.get("n1", 0)), "n_neg": int(t.get("n2", 0)),
            "method": t.get("method", "")}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarise(results: dict, outdir: str) -> dict:
    """The headline numbers of every construct, in one dict."""
    primary = results.get("primary")
    res = results.get(primary, {})
    cl = results.get("clinical", {})
    out: dict = {"cohort": {
        "n_recordings": len(results.get("video_names", [])),
        "n_abnormal": int(np.sum(np.asarray(results.get("labels", [])))),
        "primary_model": primary, "k_states": res.get("k"),
        "stream": results.get("stream"),
        "fps": results.get("geometry", {}).get("fps")}}

    # 1. fluency -------------------------------------------------------------
    phi = np.asarray(res.get("phi", {}).get("excess", []), float)
    w = res.get("phi_wilcoxon", {})
    y = np.asarray(results.get("labels", []), int)
    fl = {"phi_median": _f(np.median(phi)) if phi.size else float("nan"),
          "phi_positive": f"{w.get('n_positive', 0)}/{w.get('n', 0)}",
          "within_infant_p": _f(w.get("p")),
          "split_half_r_sb": _f(cl.get("phi_split_half", {}).get("r_sb")),
          "icc": _f(cl.get("phi_split_half", {}).get("icc_a1")),
          "group": _auc_row(cl.get("phi_test")),
          "maxT_p": _f(cl.get("maxt", {}).get("phi", {}).get("p_maxT")),
          "holm_p": _f(cl.get("holm", {}).get("phi"))}
    if phi.size and y.size == phi.size:
        fl["phi_median_normal"] = _f(np.median(phi[y == 0]))
        fl["phi_median_abnormal"] = _f(np.median(phi[y == 1]))
    ch = cl.get("channel_split", {})
    if ch:
        fl["channel_split"] = {k: {"auc": _f(v.get("auc")), "p": _f(v.get("p"))}
                               for k, v in ch.items()}
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
          "estimability_ratio": _f(res.get("estimability", {}).get("ratio")),
          "estimability_passed": bool(res.get("estimability", {})
                                      .get("passed", False)),
          "identity_check": _f(res.get("kemeny_identity", {}).get("abs_diff")),
          "group": _auc_row(cl.get("kemeny_test")),
          "maxT_p": _f(cl.get("maxt", {}).get("kemeny", {}).get("p_maxT")),
          "holm_p": _f(cl.get("holm", {}).get("kemeny"))}
    if kem.size and y.size == kem.size:
        km["median_normal"] = _f(np.median(kem[y == 0]))
        km["median_abnormal"] = _f(np.median(kem[y == 1]))
    out["kemeny"] = km

    # 4. synchrony -----------------------------------------------------------
    wc = results.get("wclrpp")
    if wc:
        F = np.asarray(wc.get("F", []), float)
        t = wc.get("test", {})
        pairs = list(wc.get("pairs", []))
        per_pair = []
        for i, nm in enumerate(pairs):
            per_pair.append({
                "pair": nm, "class": list(wc.get("pair_class", []))[i]
                if wc.get("pair_class") is not None else "",
                "median_normal": _f(np.nanmedian(F[y == 0, i]))
                if F.size else float("nan"),
                "median_abnormal": _f(np.nanmedian(F[y == 1, i]))
                if F.size else float("nan"),
                "delta": _f(np.asarray(t.get("observed", []), float)[i])
                if t else float("nan"),
                "p_corrected": _f(np.asarray(t.get("p_corrected", []),
                                             float)[i]) if t else float("nan")})
        out["synchrony"] = {
            "limb_signal": wc.get("params", {}).get("limb_signal"),
            "window_frames": wc.get("params", {}).get("w"),
            "per_pair": per_pair,
            "whole_body": _auc_row(wc.get("mean_F_test")),
            "n_perm": int(t.get("n_perm", 0)) if t else 0}
    else:
        out["synchrony"] = {"skipped": True}

    # 5. FidgetyFind ---------------------------------------------------------
    ff = results.get("fidgetyfind")
    if ff:
        M = np.asarray(ff.get("median_entropy", []), float)
        t = ff.get("chain_test", {})
        chains = list(ff.get("chains", []))
        per_chain = []
        for i, nm in enumerate(chains):
            per_chain.append({
                "chain": nm,
                "class": list(ff.get("chain_class", []))[i] if
                ff.get("chain_class") is not None else "",
                "median_normal": _f(np.nanmedian(M[y == 0, i]))
                if M.size else float("nan"),
                "median_abnormal": _f(np.nanmedian(M[y == 1, i]))
                if M.size else float("nan"),
                "delta": _f(np.asarray(t.get("observed", []), float)[i])
                if t else float("nan"),
                "p_corrected": _f(np.asarray(t.get("p_corrected", []),
                                             float)[i]) if t else float("nan"),
                "coverage": _f(np.nanmedian(
                    np.asarray(ff.get("coverage", []), float)[:, i]))})
        score = np.asarray(ff.get("score", []), float)
        out["fidgetyfind"] = {
            "score_median_normal": _f(np.nanmedian(score[y == 0]))
            if score.size else float("nan"),
            "score_median_abnormal": _f(np.nanmedian(score[y == 1]))
            if score.size else float("nan"),
            "group": _auc_row(ff.get("tests", {}).get("score")),
            "group_hips_only": _auc_row(ff.get("tests", {})
                                        .get("score_proximal")),
            "group_window_rate": _auc_row(ff.get("tests", {})
                                          .get("positive_rate_mean")),
            "per_chain": per_chain,
            "assessable_fraction": _f(np.nanmedian(
                np.asarray(ff.get("coverage_mean", []), float))),
            "agreement": {k: {"rho": _f(v.get("rho")), "p": _f(v.get("p"))}
                          for k, v in ff.get("agreement", {}).items()},
            "direction": ("AUC below 0.5 is the expected direction: absent "
                          "fidgety movement means less direction variety")}
    else:
        out["fidgetyfind"] = {"skipped": True}

    out["gates"] = cl.get("gates", {})
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
    L = [f"# RVI-38 analysis summary",
         "",
         f"{c['n_recordings']} recordings, {c['n_abnormal']} labelled abnormal "
         f"(absent fidgety movement). Primary model {c['primary_model']} with "
         f"K = {c['k_states']} states on the {c['stream']} stream at "
         f"{c['fps']} fps.",
         "",
         "Effect sizes are AUC: the probability that a random abnormal infant "
         "scores above a random normal one. 0.5 is no separation.",
         ""]

    f = s["fluency"]
    g = f.get("group", {})
    L += ["## 1. Fluency (Phi)", "",
          f"- within infants: Phi > 0 in {f['phi_positive']}, median "
          f"{_fmt(f['phi_median'])}, p = {_fmt_p(f['within_infant_p'])}",
          f"- normal {_fmt(f.get('phi_median_normal'))} vs abnormal "
          f"{_fmt(f.get('phi_median_abnormal'))}",
          f"- group contrast: AUC {_fmt(g.get('auc'))} "
          f"[{_fmt(g.get('auc_ci', [np.nan])[0])}, "
          f"{_fmt(g.get('auc_ci', [np.nan, np.nan])[1])}], "
          f"p = {_fmt_p(g.get('p'))} ({g.get('method', '')})",
          f"- corrected over the two confirmatory endpoints: maxT p = "
          f"{_fmt_p(f['maxT_p'])}, Holm p = {_fmt_p(f['holm_p'])}",
          f"- reliability: split-half r_SB = {_fmt(f['split_half_r_sb'])}, "
          f"ICC(A,1) = {_fmt(f['icc'])}"]
    if "channel_split" in f:
        parts = ", ".join(f"{k} AUC {_fmt(v['auc'])} (p {_fmt_p(v['p'])})"
                          for k, v in f["channel_split"].items())
        L += [f"- magnitude vs direction: {parts}"]
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
          f"p = {_fmt_p(g.get('p'))}; maxT p = {_fmt_p(k['maxT_p'])}",
          f"- shrinkage alpha {_fmt(k['shrinkage_alpha'])}, estimability "
          f"ratio {_fmt(k['estimability_ratio'], 2)} "
          f"({'PASS' if k['estimability_passed'] else 'FAIL'}), spectral "
          f"identity |diff| = {k['identity_check']:.1e}", ""]

    sy = s["synchrony"]
    L += ["## 4. Synchrony (WCLR-PP inter-limb coordination)", ""]
    if sy.get("skipped"):
        L += ["- skipped", ""]
    else:
        g = sy.get("whole_body", {})
        L += [f"- limb signal `{sy['limb_signal']}`, window "
              f"{sy['window_frames']} frames, {sy['n_perm']:,} label "
              f"permutations",
              f"- whole-body coupling (mean F): AUC {_fmt(g.get('auc'))}, "
              f"p = {_fmt_p(g.get('p'))}",
              "", "| pair | class | normal | abnormal | delta | p (corrected) |",
              "|---|---|---|---|---|---|"]
        for r in sy["per_pair"]:
            L += [f"| {r['pair']} | {r['class']} | "
                  f"{_fmt(r['median_normal'])} | {_fmt(r['median_abnormal'])} "
                  f"| {r['delta']:+.3f} | {_fmt_p(r['p_corrected'])} |"]
        L += [""]

    ff = s["fidgetyfind"]
    L += ["## 5. FidgetyFind (Morais et al., 2023)", ""]
    if ff.get("skipped"):
        L += ["- skipped", ""]
    else:
        g = ff.get("group", {})
        h = ff.get("group_hips_only", {})
        L += [f"- score: normal {_fmt(ff['score_median_normal'])} vs abnormal "
              f"{_fmt(ff['score_median_abnormal'])}",
              f"- group contrast: AUC {_fmt(g.get('auc'))} "
              f"[{_fmt(g.get('auc_ci', [np.nan])[0])}, "
              f"{_fmt(g.get('auc_ci', [np.nan, np.nan])[1])}], "
              f"p = {_fmt_p(g.get('p'))}",
              f"- hips only (the unadapted published path): AUC "
              f"{_fmt(h.get('auc'))}, p = {_fmt_p(h.get('p'))}",
              f"- {ff['direction']}",
              f"- median fraction of windows assessable: "
              f"{_fmt(ff['assessable_fraction'])}"]
        if ff.get("agreement"):
            L += ["- agreement with the other constructs (Spearman): "
                  + ", ".join(f"{k} rho {_fmt(v['rho'], 2)} "
                              f"(p {_fmt_p(v['p'])})"
                              for k, v in ff["agreement"].items())]
        L += ["", "| chain | class | normal | abnormal | delta | "
              "p (corrected) | assessable |", "|---|---|---|---|---|---|---|"]
        for r in ff["per_chain"]:
            L += [f"| {r['chain']} | {r['class']} | "
                  f"{_fmt(r['median_normal'])} | {_fmt(r['median_abnormal'])} "
                  f"| {r['delta']:+.3f} | {_fmt_p(r['p_corrected'])} | "
                  f"{_fmt(r['coverage'], 2)} |"]
        L += [""]

    if s.get("gates"):
        L += ["## Pre-specified gates", ""]
        for name, ok in s["gates"].items():
            L += [f"- {'PASS' if ok else 'FAIL'} — {name}"]
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
def run_report(csv: str = "rvi38_analysis.csv",
               arhmm: str | None = "arhmm_rvi38_stream_delta.pkl",
               hmm: str | None = "hmm_rvi38_stream_delta.pkl",
               labels: str | None = "RVI_38_labels.mat",
               outdir: str = "rvi38_out",
               fast: bool = False,
               fps: float = 25.0,
               stream: str = "auto",
               models: str = "both",
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

    Parameters mirror the command line of :mod:`run_analysis`; the defaults are
    the ones that produce the complete picture. Set ``fast=True`` for a smoke
    run (every resampling count is cut ~20x, so the p-values are coarse), and
    ``synchrony=False`` to skip WCLR-PP, which is by far the slowest block.

    ``show`` displays the figures when called from a notebook: ``"main"`` for
    the cohort figures, ``"all"`` to include every per-recording panel, or a
    single group name (``"fluency"``, ``"fluency_curve"``, ``"kemeny"``,
    ``"synchrony"``, ``"fidgetyfind"``, ``"clinical"``).

    Any further keyword goes straight through to the runner:
    ``fluency_omega=0.7`` becomes ``--fluency-omega 0.7``, ``ff_theta=0.6``
    becomes ``--ff-theta 0.6``, ``wclr_limb_signal="distal"`` becomes
    ``--wclr-limb-signal distal``. A boolean keyword becomes a bare flag when
    it is true.
    """
    argv = ["--csv", str(csv), "--outdir", str(outdir), "--fps", str(fps),
            "--stream", str(stream), "--models", str(models),
            "--controls", str(controls)]
    if arhmm:
        argv += ["--arhmm", str(arhmm)]
    if hmm:
        argv += ["--hmm", str(hmm)]
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
    ap.add_argument("--arhmm", default="arhmm_rvi38_stream_delta.pkl")
    ap.add_argument("--hmm", default="hmm_rvi38_stream_delta.pkl")
    ap.add_argument("--labels", default="RVI_38_labels.mat")
    ap.add_argument("--outdir", default="rvi38_out")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-synchrony", action="store_true")
    a = ap.parse_args()
    run_report(csv=a.csv, arhmm=a.arhmm, hmm=a.hmm, labels=a.labels,
               outdir=a.outdir, fast=a.fast, synchrony=not a.no_synchrony)
