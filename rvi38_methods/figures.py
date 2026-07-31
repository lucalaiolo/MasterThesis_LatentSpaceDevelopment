"""Figures, one per file, sized for direct \\includegraphics in LaTeX.

Every annotation is computed from the results object passed in — nothing is
hard-coded, so a stale number cannot survive a change upstream. Titles carry no
document section references; each figure is self-describing.

Each function returns the PNG path (a PDF of the same name is written beside
it) or ``None`` when the underlying quantity was not computed.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from build_pose import FREE, JOINTS   # noqa: E402

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight"})

POS, NEG = "#c0392b", "#5b7fa6"
BLUE, GREEN, GREY = "#2b6cb0", "#2e7d32", "#8a8a8a"


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{name}.png")
    fig.savefig(png, dpi=300)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# state description
# ---------------------------------------------------------------------------
def fig_state_signature(res, results, outdir):
    """Which joints move in each state, as log RMS speed centred per joint."""
    amp = np.asarray(res["amplitude"])
    lab = res["state_labels"]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    lg = np.log(np.clip(amp[:, FREE], 1e-12, None))
    lg = lg - lg.mean(0, keepdims=True)
    v = float(np.abs(lg).max())
    im = ax.imshow(lg, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(FREE)))
    ax.set_xticklabels([JOINTS[j] for j in FREE], rotation=90, fontsize=7)
    ax.set_yticks(range(res["k"]))
    ax.set_yticklabels(lab, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=.046, label="log RMS speed (joint-centred)")
    ax.set_title("State kinematic signature", loc="left", fontsize=10)
    return _save(fig, outdir, "state_signature")


def fig_similarity(res, results, outdir):
    """Pairwise kinematic similarity between states."""
    S = np.asarray(res["S"])
    K = res["k"]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(range(K), fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels(res["state_labels"], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=.046, label="$S_{kk'}$")
    off = S[~np.eye(K, dtype=bool)]
    ax.set_title(f"Kinematic similarity between states\n"
                 f"off-diagonal range [{off.min():+.2f}, {off.max():+.2f}]",
                 loc="left", fontsize=10)
    return _save(fig, outdir, "similarity_matrix")


# ---------------------------------------------------------------------------
# fluency
# ---------------------------------------------------------------------------
def fig_fluency_per_subject(res, results, outdir):
    """Excess similarity of consecutive movements, per infant."""
    labels = np.asarray(results["labels"])
    phi = np.asarray(res["phi"]["excess"], float)
    w = res["phi_wilcoxon"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    o = np.argsort(phi)
    ax.bar(range(len(phi)), phi[o],
           color=[POS if labels[i] else NEG for i in o], width=.82)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xlabel("infant (sorted)")
    ax.set_ylabel("excess similarity $\\Phi$")
    ax.plot([], [], "s", color=POS, label=f"abnormal (n={int(labels.sum())})")
    ax.plot([], [], "s", color=NEG,
            label=f"normal (n={int((1 - labels).sum())})")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title(f"Fluency per infant\n{w['n_positive']}/{w['n']} above zero, "
                 f"Wilcoxon signed-rank $p$ = {w['p']:.2g}", loc="left",
                 fontsize=10)
    return _save(fig, outdir, "fluency_per_subject")


def fig_fluency_by_dwell(res, results, outdir):
    """Excess similarity split by how long the source state was held.

    This is the test for over-segmentation. If the HMM had fragmented one
    continuous movement into several kinematically similar states, consecutive
    visits would look alike only because they are pieces of the same movement,
    and the effect would sit entirely in the shortest dwells. An effect that
    survives or grows at long dwells cannot be explained that way.
    """
    ds = [d for d in res.get("dwell_strat", []) if np.isfinite(d["excess"])]
    if not ds:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    vals = [d["excess"] for d in ds]
    ax.bar(range(len(ds)), vals, color=BLUE, width=.62)
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([d["bin"] for d in ds])
    for i, d in enumerate(ds):
        ax.text(i, d["excess"], f"n={d['n']:,}", ha="center", va="bottom",
                fontsize=6.5)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xlabel("dwell of the source state (windows)")
    ax.set_ylabel("excess similarity $\\Phi$")
    trend = ("rises with dwell, so not over-segmentation"
             if vals[-1] > vals[0] else "concentrated in short dwells")
    ax.set_title(f"Fluency by dwell length\n{trend}", loc="left", fontsize=10)
    return _save(fig, outdir, "fluency_by_dwell")


# ---------------------------------------------------------------------------
# mixing structure
# ---------------------------------------------------------------------------
def fig_degenerate(res, results, outdir):
    """Graph statistics that are uninformative on a row-stochastic matrix."""
    d = res["degenerate"]
    K = res["k"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    x = np.arange(K)
    ax.plot(x, np.asarray(d["out_degree"], float), "o-", ms=3, lw=1, color=GREY,
            label="weighted out-degree")
    ax.plot(x, np.asarray(d["right_perron"], float), "s-", ms=3, lw=1,
            color=BLUE, label="eigenvector centrality")
    ax.plot(x, np.asarray(d["stationary"], float) * K, "^-", ms=3, lw=1,
            color=POS, label="stationary $\\times K$")
    ax.set_xlabel("state")
    ax.set_ylabel("value")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Degenerate graph statistics\nout-degree and centrality are "
                 "constant by construction", loc="left", fontsize=10)
    return _save(fig, outdir, "degenerate_statistics")


def fig_mfpt(res, results, outdir):
    """Expected number of jumps to first reach each state."""
    M = np.asarray(res["companions_jump"]["mfpt"], float)
    K = res["k"]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(M, cmap="magma_r")
    plt.colorbar(im, ax=ax, fraction=.046, label="jumps")
    ax.set_xticks(range(K)); ax.set_xticklabels(range(K), fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels(res["state_labels"], fontsize=7)
    ax.set_xlabel("to state")
    hardest = int(np.argmax(np.asarray(res["companions_jump"]["mean_inbound"])))
    ax.set_title(f"Mean first passage time\nhardest to reach: "
                 f"{res['state_labels'][hardest]}", loc="left", fontsize=10)
    return _save(fig, outdir, "mfpt_matrix")


def fig_companions(res, results, outdir):
    """How central each state is, and how hard it is to reach."""
    K = res["k"]
    clo = np.asarray(res["companions_jump"]["closeness"], float)
    inb = np.asarray(res["companions_jump"]["mean_inbound"], float)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.barh(np.arange(K) - .2, clo / clo.max(), height=.38, color=BLUE,
            label="dynamical closeness (scaled)")
    ax.barh(np.arange(K) + .2, inb / inb.max(), height=.38, color="#d9a441",
            label="mean inbound passage (scaled)")
    ax.set_yticks(range(K)); ax.set_yticklabels(res["state_labels"], fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"State centrality in movement-time\nKemeny constant "
                 f"{res['companions_jump']['kemeny']:.1f} jumps / "
                 f"{res['companions_full']['kemeny_seconds']:.1f} s",
                 loc="left", fontsize=10)
    return _save(fig, outdir, "graph_companions")


def fig_kemeny_per_subject(res, results, outdir):
    """Per-infant mixing time with its block-bootstrap interval."""
    labels = np.asarray(results["labels"])
    kem = np.asarray(res["kemeny_per_subject"], float)
    lo = np.asarray(res["kemeny_lo"], float)
    hi = np.asarray(res["kemeny_hi"], float)
    e = res["estimability"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    o = np.argsort(kem)
    ax.errorbar(range(len(kem)), kem[o],
                yerr=[np.clip(kem[o] - lo[o], 0, None),
                      np.clip(hi[o] - kem[o], 0, None)],
                fmt="none", ecolor="#c8d3de", lw=1)
    ax.scatter(range(len(kem)), kem[o],
               c=[POS if labels[i] else NEG for i in o], s=20, zorder=3,
               edgecolors="none")
    ax.plot([], [], "o", color=POS, label="abnormal")
    ax.plot([], [], "o", color=NEG, label="normal")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_xlabel("infant (sorted)")
    ax.set_ylabel("Kemeny constant (jumps)")
    ax.set_title(f"Per-infant mixing time\nmedian interval width / "
                 f"between-infant range = {e['ratio']:.2f}", loc="left",
                 fontsize=10)
    return _save(fig, outdir, "kemeny_per_subject")


def fig_shrinkage(res, results, outdir):
    """Held-out likelihood against the shrinkage weight."""
    al = res["alpha"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(np.asarray(al["grid"], float), np.asarray(al["loglik_grid"], float),
            "o-", ms=3, lw=1.2, color=BLUE)
    ax.axvline(al["alpha"], color=POS, lw=1.5,
               label=f"selected $\\alpha$ = {al['alpha']:.2f}")
    ax.set_xlabel("shrinkage weight $\\alpha$ towards the group matrix")
    ax.set_ylabel("held-out log-likelihood")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Shrinkage selection\n"
                 + ("degenerate: infants carry little individual information"
                    if al["degenerate"]
                    else "per-infant matrices carry information"),
                 loc="left", fontsize=10)
    return _save(fig, outdir, "shrinkage_selection")


# ---------------------------------------------------------------------------
# inference and quality control
# ---------------------------------------------------------------------------
def fig_split_half(res, results, outdir):
    """Reliability: the measure computed on each half of the recording."""
    cl = results.get("clinical")
    h = results.get("_halves")
    if not cl or h is None:
        return None
    labels = np.asarray(results["labels"])
    sh = cl["phi_split_half"]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.scatter(h[0], h[1], s=26, c=[POS if v else NEG for v in labels],
               edgecolors="none")
    both = np.concatenate([np.asarray(h[0], float), np.asarray(h[1], float)])
    lim = [np.nanmin(both), np.nanmax(both)]
    ax.plot(lim, lim, "k--", lw=.8)
    ax.set_xlabel("$\\Phi$, first half of recording")
    ax.set_ylabel("$\\Phi$, second half of recording")
    ax.set_aspect("equal")
    ax.set_title(f"Split-half reliability\nSpearman-Brown $r$ = "
                 f"{sh['r_sb']:.3f} [{sh['r_sb_lo']:.2f}, {sh['r_sb_hi']:.2f}]",
                 loc="left", fontsize=10)
    return _save(fig, outdir, "split_half_reliability")


def fig_auc(res, results, outdir):
    """Group separation for each endpoint, with both interval estimates."""
    cl = results.get("clinical")
    if not cl:
        return None
    rows = [("Fluency $\\Phi$", cl["phi_test"], "phi"),
            ("Kemeny $\\mathcal{K}$", cl["kemeny_test"], "kemeny")]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, (nm, r, key) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.errorbar(r["auc"], y + .12,
                    xerr=[[r["auc"] - r["auc_lo"]], [r["auc_hi"] - r["auc"]]],
                    fmt="o", color=BLUE, ms=6, capsize=3,
                    label="normal-approximation CI" if i == 0 else None)
        if np.isfinite(r.get("auc_lo_boot", np.nan)):
            ax.errorbar(r["auc"], y - .12,
                        xerr=[[r["auc"] - r["auc_lo_boot"]],
                              [r["auc_hi_boot"] - r["auc"]]],
                        fmt="s", color=GREY, ms=5, capsize=3,
                        label="bootstrap CI" if i == 0 else None)
        ax.text(0.02, y + .3, f"$p$ = {r['p']:.3g}   "
                f"(corrected {cl['maxt'][key]['p_maxT']:.3g})", fontsize=7.5)
    mde = cl["mde"]["auc"]
    ax.axvline(0.5, color="k", lw=.8)
    for mline, lbl in ((mde, "minimum detectable effect"), (1 - mde, None)):
        ax.axvline(mline, color=POS, ls=":", lw=1, label=lbl)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xlabel("AUC, abnormal versus normal")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Group separation for the two endpoints", loc="left",
                 fontsize=10)
    return _save(fig, outdir, "auc_effect_sizes")


def fig_loo(res, results, outdir):
    """Sensitivity of each result to any single infant."""
    cl = results.get("clinical")
    if not cl or not cl.get("loo"):
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for nm, col, lbl in (("phi", BLUE, "Fluency $\\Phi$"),
                         ("kemeny", POS, "Kemeny $\\mathcal{K}$")):
        if not cl["loo"].get(nm):
            continue
        ps = [r["p"] for r in cl["loo"][nm]]
        ax.scatter(np.arange(len(ps)), ps, s=14, c=col, label=lbl)
    ax.axhline(0.05, color="k", ls="--", lw=.9)
    ax.text(0.99, 0.052, "significance threshold $p$ = 0.05", ha="right",
            va="bottom", fontsize=7, transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel("index of the omitted infant")
    ax.set_ylabel("$p$ after omitting that infant")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Leave-one-out sensitivity\npoints above the line mark an "
                 "infant carrying the result", loc="left", fontsize=10)
    return _save(fig, outdir, "loo_stability")


def fig_redundancy(res, results, outdir):
    """Whether either endpoint merely restates occupancy or dwell time."""
    cl = results.get("clinical")
    if not cl:
        return None
    rd = cl["redundancy"]
    pretty = {"phi_vs_entropy": "$\\Phi$ vs occupancy entropy",
              "phi_vs_dwell": "$\\Phi$ vs mean dwell",
              "kemeny_vs_entropy": "$\\mathcal{K}$ vs occupancy entropy",
              "kemeny_vs_dwell": "$\\mathcal{K}$ vs mean dwell"}
    ks = list(rd)
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    vals = [abs(rd[k]["rho"]) for k in ks]
    ax.barh(range(len(ks))[::-1], vals,
            color=[POS if v >= 0.8 else BLUE for v in vals])
    ax.axvline(0.8, color="k", ls="--", lw=.9)
    ax.text(0.8, len(ks) - 0.4, " threshold", fontsize=7, va="top")
    ax.set_yticks(range(len(ks))[::-1])
    ax.set_yticklabels([pretty.get(k, k) for k in ks], fontsize=8)
    ax.set_xlabel("|Spearman $\\rho$|")
    ax.set_xlim(0, 1)
    ax.set_title("Redundancy against simpler quantities", loc="left",
                 fontsize=10)
    return _save(fig, outdir, "redundancy")


def fig_gates(res, results, outdir):
    """Pass/fail of the pre-specified admission criteria."""
    cl = results.get("clinical")
    if not cl or not cl.get("gates"):
        return None
    gates = cl["gates"]
    ks = list(gates)
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(ks) + 1.1))
    ax.barh(range(len(ks))[::-1], [1] * len(ks),
            color=[GREEN if gates[k] else POS for k in ks])
    ax.set_yticks(range(len(ks))[::-1])
    ax.set_yticklabels([k.replace(") [", ")\n[") for k in ks], fontsize=7)
    for i, k in enumerate(ks):
        ax.text(1.03, len(ks) - 1 - i, "PASS" if gates[k] else "FAIL",
                va="center", fontsize=8, fontweight="bold",
                color=GREEN if gates[k] else POS)
    ax.set_xticks([]); ax.set_xlim(0, 1.35)
    ax.set_title("Pre-specified admission criteria", loc="left", fontsize=10)
    return _save(fig, outdir, "gates")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# raw kinematics: limb co-movement and the fidgety band
# ---------------------------------------------------------------------------
def _comovement_panel(results, per_key, med_key, test_key, subtitle, name,
                      outdir):
    """Six panels, one per limb pair; each recording is a strip of its epochs.

    Within-recording spread separates consistent from intermittent co-movement,
    and between-recording differences are read across strips, so consistency and
    magnitude appear in one display. A positive value is the
    cramped-synchronised direction, a negative one the reciprocal pattern.
    """
    cm = results.get("comovement")
    if not cm or per_key not in cm:
        return None
    per = cm[per_key]
    med = np.asarray(cm[med_key], float)
    test = cm.get(test_key)
    y = np.asarray(results["labels"]).astype(int)
    names, klass = cm["pairs"], cm["pair_class"]
    n = len(per)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), sharey=True)
    rng = np.random.default_rng(0)
    for p in range(len(names)):
        ax = axes[p // 2][p % 2]
        order = np.lexsort((np.nan_to_num(med[:, p], nan=-9), y))
        for slot, i in enumerate(order):
            v = np.asarray(per[i], float)
            if v.size == 0:
                continue
            v = v[:, p]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            col = POS if y[i] == 1 else NEG
            ax.scatter(slot + rng.uniform(-0.28, 0.28, len(v)), v, s=5,
                       color=col, alpha=0.35, edgecolor="none", zorder=2)
            ax.plot([slot - 0.42, slot + 0.42], [np.median(v)] * 2, color=col,
                    lw=1.6, zorder=3)
        ax.axhline(0.0, color="0.5", lw=0.8, ls="--", zorder=1)
        extra = (f"   p={test['p_corrected'][p]:.3f}" if test is not None else "")
        ax.set_title(f"{names[p]}  ({klass[p]}){extra}", fontsize=9, loc="left")
        ax.set_xlim(-1, n)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        if p % 2 == 0:
            ax.set_ylabel("co-movement $D$")
    fig.suptitle(f"Limb co-movement per epoch — {subtitle}\n"
                 "(+ together, − opposition); one strip per infant, "
                 "red = abnormal", fontsize=10)
    return _save(fig, outdir, name)


def fig_comovement(res, results, outdir):
    """Co-movement exactly as specified (unfiltered displacements)."""
    return _comovement_panel(results, "per_recording", "medians", "test",
                             "as specified, unfiltered", "comovement", outdir)


def fig_comovement_bandlimited(res, results, outdir):
    """The same construct with the displacements band-limited first.

    Differencing is high-pass, so keypoint noise attenuates |D| toward zero
    without biasing its sign; this panel is the sensitivity analysis and is
    plotted beside the pre-specified one rather than replacing it.
    """
    cm = results.get("comovement", {})
    band = cm.get("band_hz")
    sub = (f"band-limited {band[0]}–{band[1]} Hz" if band
           else "band-limited (sensitivity analysis)")
    return _comovement_panel(results, "per_recording_bandlimited",
                             "medians_bandlimited", "test_bandlimited",
                             sub, "comovement_bandlimited", outdir)


def _velocity_panel(prof, res, outdir, name, colors, hatch, title):
    K, groups = prof["k"], prof["groups"]
    fig, ax = plt.subplots(figsize=(1.5 * K + 2.5, 4.0))
    w = 0.82 / len(groups)
    rng = np.random.default_rng(0)
    for gi, g in enumerate(groups):
        pos, data = [], []
        for s in range(K):
            vals = np.asarray(prof[s][g], float)
            vals = vals[np.isfinite(vals)]
            x = s + (gi - (len(groups) - 1) / 2) * w
            pos.append(x)
            data.append(vals)
            ax.scatter(x + rng.uniform(-w * 0.26, w * 0.26, len(vals)), vals,
                       s=8, color=colors[gi], alpha=0.45, edgecolor="none",
                       zorder=3)
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black"),
                        zorder=2)
        for b in bp["boxes"]:
            b.set(facecolor=colors[gi], alpha=0.45, edgecolor="grey",
                  hatch=hatch[gi])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=colors[i], alpha=0.6, hatch=hatch[i],
                             label=g) for i, g in enumerate(groups)],
              frameon=False, fontsize=8, ncol=min(len(groups), 4),
              loc="upper left")
    ax.set_xticks(range(K))
    ax.set_xticklabels([str(s) for s in range(K)])
    ax.set_xlabel("state")
    ax.set_ylabel("% high velocity frames")
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=9, loc="left")
    return _save(fig, outdir, name)


def fig_velocity_regions(res, results, outdir):
    """Per-state high-velocity fraction by head / arms / legs."""
    prof = res.get("velocity_profile_regions")
    if not prof:
        return None
    return _velocity_panel(
        prof, res, outdir, "state_velocity_regions",
        ["#E8998D", "#EAD7A0", "#8FBF9F"], ["", "", ""],
        "State movement dynamics — head / arms / legs (one dot per infant)")


def fig_velocity_lateral(res, results, outdir):
    """Per-state high-velocity fraction, left versus right."""
    prof = res.get("velocity_profile_lateral")
    if not prof:
        return None
    order = prof["groups"]
    colors = ["#3B7DD8" if g.startswith("left") else "#D8503B" for g in order]
    hatch = ["" if g.endswith("arm") else "//" for g in order]
    return _velocity_panel(
        prof, res, outdir, "state_velocity_lateral", colors, hatch,
        "State movement dynamics — left vs right (blue left, red right; "
        "hatched = leg)")


ALL = [fig_state_signature, fig_similarity, fig_fluency_per_subject,
       fig_fluency_by_dwell, fig_degenerate, fig_mfpt,
       fig_companions, fig_kemeny_per_subject, fig_shrinkage, fig_split_half,
       fig_auc, fig_loo, fig_redundancy, fig_gates,
       fig_comovement, fig_comovement_bandlimited,
       fig_velocity_regions, fig_velocity_lateral]


def make_all(results, outdir):
    """Write every figure as its own file under ``<outdir>/figures``."""
    primary = results["primary"]
    fdir = os.path.join(outdir, "figures")
    made = []
    for fn in ALL:
        try:
            p = fn(results[primary], results, fdir)
        except Exception as exc:                            # noqa: BLE001
            print(f"  figure {fn.__name__} failed: {exc}")
            continue
        if p:
            made.append(p)
    return made
