"""Figure panels for A1, A5 and A7.

Every annotation is computed from the results object passed in. Nothing is
hard-coded: a figure that reports ``ARI = 1.000`` does so because this run
produced it, so a stale number can never survive a change upstream.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from build_pose import FREE, JOINTS   # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight"})

POS, NEG = "#c0392b", "#5b7fa6"
BLUE, GREEN, GREY = "#2b6cb0", "#2e7d32", "#8a8a8a"


def _save(fig, outdir, name):
    png = os.path.join(outdir, f"{name}.png")
    fig.savefig(png, dpi=180)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)
    return png


def _title(ax, letter, main, sub=""):
    ax.set_title(f"{letter}  {main}" + (f"\n{sub}" if sub else ""),
                 loc="left", fontsize=8.5)


# ---------------------------------------------------------------------------
def fig_a1(res, results, outdir, name="A1_fluency"):
    """§5 signatures, §6 similarity, §7 fluency and its controls."""
    S = np.asarray(res["S"])
    amp = np.asarray(res["amplitude"])
    lab = res["state_labels"]
    K = res["k"]
    labels = np.asarray(results["labels"])
    phi = np.asarray(res["phi"]["excess"], float)

    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=.55, wspace=.36)

    # a: state kinematic signature
    ax = fig.add_subplot(gs[0, 0])
    lg = np.log(np.clip(amp[:, FREE], 1e-12, None))
    lg = lg - lg.mean(0, keepdims=True)
    v = float(np.abs(lg).max())
    im = ax.imshow(lg, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(FREE)))
    ax.set_xticklabels([JOINTS[j] for j in FREE], rotation=90, fontsize=6)
    ax.set_yticks(range(K))
    ax.set_yticklabels(lab, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=.046)
    _title(ax, "a", "State kinematic signature",
           f"log RMS speed, per-joint centred ({int(np.min(res['state_frames'])):,}"
           f"+ frames/state)")

    # b: similarity matrix
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(range(K), fontsize=6)
    ax.set_yticks(range(K)); ax.set_yticklabels(lab, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=.046)
    off = S[~np.eye(K, dtype=bool)]
    _title(ax, "b", "Kinematic similarity $S_{kk'}$",
           f"double-centred, off-diag [{off.min():+.2f}, {off.max():+.2f}]")

    # c: group jump chain vs similarity
    ax = fig.add_subplot(gs[0, 2])
    if "group_jump_empirical" in res:
        P = np.asarray(res["group_jump_empirical"])
        m = ~np.eye(K, dtype=bool)
        ax.scatter(S[m], P[m], s=14, c=BLUE, alpha=.65, edgecolors="none")
        b = np.polyfit(S[m], P[m], 1)
        xx = np.linspace(S[m].min(), S[m].max(), 50)
        ax.plot(xx, np.polyval(b, xx), "k-", lw=1.2)
        mt = res.get("mantel", {})
        _title(ax, "c", "Group jump chain vs similarity",
               f"Mantel r = {mt.get('r', float('nan')):+.3f}, "
               f"p = {mt.get('p', float('nan')):.2g}")
    ax.set_xlabel("kinematic similarity $S_{kk'}$")
    ax.set_ylabel("jump probability $\\tilde{A}_{kk'}$")

    # d: per-subject fluency
    ax = fig.add_subplot(gs[1, 0])
    o = np.argsort(phi)
    ax.bar(range(len(phi)), phi[o],
           color=[POS if labels[i] else NEG for i in o], width=.82)
    ax.axhline(0, color="k", lw=.7)
    w = res["phi_wilcoxon"]
    ax.set_xlabel("subject (sorted)")
    ax.set_ylabel("$\\Phi_{\\mathrm{excess}}$")
    ax.plot([], [], "s", color=POS, label=f"abnormal ({int(labels.sum())})")
    ax.plot([], [], "s", color=NEG, label=f"normal ({int((1 - labels).sum())})")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    _title(ax, "d", "Fluency per subject",
           f"{w['n_positive']}/{w['n']} positive, Wilcoxon p = {w['p']:.2g}")

    # e: correlogram
    ax = fig.add_subplot(gs[1, 1])
    lags = np.asarray(res["correlogram"]["lags"])
    E = np.asarray(res["correlogram"]["excess"], float)
    if lags.size == 0:
        ax.text(.5, .5, "skipped\n(--controls core)", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color=GREY)
        _title(ax, "e", "Fluency correlogram (§7.6)", "not run")
        lags = None
    if lags is not None:
        mu = np.nanmean(E, 0)
        se = np.nanstd(E, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(E), 0))
        ax.fill_between(lags, mu - 1.96 * se, mu + 1.96 * se, color=BLUE,
                        alpha=.22)
        ax.plot(lags, mu, "o-", color=BLUE, ms=4, lw=1.3)
        ax.axhline(0, color="k", lw=.7, ls="--")
        ax.set_xlabel("lag (state visits)")
        ax.set_ylabel("excess similarity")
        shape = ("smooth decay" if len(mu) > 2 and mu[1] > 0.35 * mu[0]
                 else "lag-1 spike then collapse")
        _title(ax, "e", "Fluency correlogram (§7.6)", shape)

    # f: dwell stratification
    ax = fig.add_subplot(gs[1, 2])
    ds = [d for d in res["dwell_strat"] if np.isfinite(d["excess"])]
    if ds:
        ax.bar(range(len(ds)), [d["excess"] for d in ds], color=BLUE, width=.65)
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels([d["bin"] for d in ds])
        for i, d in enumerate(ds):
            ax.text(i, d["excess"], f"n={d['n']}", ha="center", va="bottom",
                    fontsize=5.5)
        trend = ("increases with dwell" if ds[-1]["excess"] > ds[0]["excess"]
                 else "concentrated in short dwells")
    else:
        trend = "no bin had enough transitions"
    ax.axhline(0, color="k", lw=.7)
    ax.set_xlabel("dwell of source state (windows)")
    ax.set_ylabel("excess similarity")
    _title(ax, "f", "Fluency by dwell (§7.6)", trend)

    fig.suptitle(f"A1 fluency — {res['tag']}, K = {K}", x=.01, ha="left",
                 fontsize=11, weight="bold")
    return _save(fig, outdir, name)


# ---------------------------------------------------------------------------
def fig_a5(res, results, outdir, name="A5_metastable"):
    """§8 metastable decomposition and its agreement with raw kinematics."""
    from scipy.cluster.hierarchy import dendrogram
    import a57_graph as G

    S = np.asarray(res["S"])
    K = res["k"]
    lab = res["state_labels"]
    sweep = res.get("pcca_sweep", [])
    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=.6, wspace=.4)

    # a: implied timescales
    ax = fig.add_subplot(gs[0, 0])
    for nm, key, c in (("full chain", "timescales_full", BLUE),
                       ("jump chain", "timescales_jump", POS)):
        t = np.asarray(res[key]["timescales_steps"], float)
        ax.plot(range(2, 2 + len(t)), t, "o-", color=c, ms=4, lw=1.2, label=nm)
    ax.set_xlabel("eigenvalue index $m$")
    ax.set_ylabel("implied timescale (steps)")
    ax.legend(fontsize=6.5, frameon=False)
    gr = np.asarray(res["timescales_jump"]["gap_ratios"], float)
    big = np.isfinite(gr).any() and np.nanmax(gr) >= 2.0
    _title(ax, "a", "Implied timescales (§8.2)",
           f"max gap ratio {np.nanmax(gr):.2f} at m="
           f"{res['timescales_jump']['m_at_max_gap']}"
           f" — {'clean gap' if big else 'no clean gap'}")

    # b: PCCA+ memberships at m*
    ax = fig.add_subplot(gs[0, 1])
    best = max(sweep, key=lambda r: r["ari_pcca_kin"]) if sweep else None
    if best:
        chi = np.asarray(best["chi"], float)
        o = np.argsort(chi.argmax(1))
        im = ax.imshow(chi[o], aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(K)); ax.set_yticklabels([lab[i] for i in o],
                                                    fontsize=6)
        ax.set_xticks(range(best["m"]))
        ax.set_xlabel("metastable set")
        plt.colorbar(im, ax=ax, fraction=.046)
        _title(ax, "b", f"PCCA+ memberships, $m={best['m']}$",
               f"crispness {best['crispness']:.2f}, metastability "
               f"{best['metastability']:.2f}/{best['m']}")

    # c: agreement sweep
    ax = fig.add_subplot(gs[0, 2])
    if sweep:
        ms = [r["m"] for r in sweep]
        ax.plot(ms, [r["ari_pcca_kin"] for r in sweep], "o-", color=GREEN,
                ms=5, lw=1.4, label="PCCA+ vs kinematic")
        ax.plot(ms, [r["ari_kmeans_kin"] for r in sweep], "s--", color=GREY,
                ms=4, lw=1, label="k-means vs kinematic")
        ax.plot(ms, [r["ami_pcca_kin"] for r in sweep], "^:", color=BLUE,
                ms=4, lw=1, label="AMI (PCCA+)")
        ax.axhline(0, color="k", lw=.6)
        ax.legend(fontsize=6, frameon=False)
        _title(ax, "c", "Agreement vs number of sets",
               f"peak ARI {best['ari_pcca_kin']:.3f} at m={best['m']}")
    ax.set_xlabel("number of sets $m$")
    ax.set_ylabel("agreement index")

    # d: kinematic dendrogram
    ax = fig.add_subplot(gs[1, 0])
    Z = G.kinematic_linkage(S)
    thr = Z[-(best["m"] - 1), 2] if best and best["m"] > 1 else None
    dendrogram(Z, labels=lab, ax=ax, leaf_font_size=6,
               color_threshold=thr)
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("1 - kinematic similarity")
    _title(ax, "d", "Kinematic clustering (§8.4)",
           "raw velocities only — no transition information")

    # e: permutation null
    ax = fig.add_subplot(gs[1, 1])
    ag = res.get("agreement")
    if ag and best:
        from sklearn.metrics import adjusted_rand_score
        rng = np.random.default_rng(1)
        draw = min(int(ag["n_perm"]), 30_000)
        nul = np.array([adjusted_rand_score(
            best["assign_pcca"], rng.permutation(best["assign_kin"]))
            for _ in range(draw)])
        ax.hist(nul, bins=60, color="#b8c6d6", edgecolor="none")
        ax.axvline(ag["ari"], color=POS, lw=1.8,
                   label=f"observed = {ag['ari']:.3f}")
        ax.set_yscale("log")
        ax.legend(fontsize=6.5, frameon=False)
        _title(ax, "e", "Permutation null (§10.6)",
               f"p = {ag['p_ari']:.2g} over {ag['n_perm']:,} draws "
               f"(shown: {draw:,})")
    ax.set_xlabel("ARI under label permutation")
    ax.set_ylabel("count")

    # f: subject-bootstrap stability
    ax = fig.add_subplot(gs[1, 2])
    stab = res.get("stability")
    if stab:
        ax.hist(np.asarray(stab["ari"], float), bins=30, color="#cfe0cf",
                edgecolor="none")
        ax.axvline(stab["mean"], color=GREEN, lw=1.8,
                   label=f"mean = {stab['mean']:.3f}")
        ax.axvspan(stab["lo"], stab["hi"], color=GREEN, alpha=.12)
        ax.legend(fontsize=6.5, frameon=False)
        _title(ax, "f", "Subject bootstrap (§8.5)",
               f"{stab['n']} refits, 95% [{stab['lo']:.2f}, {stab['hi']:.2f}]")
    ax.set_xlabel("ARI vs fixed kinematic partition")
    ax.set_ylabel("count")

    fig.suptitle(f"A5 metastable decomposition — {res['tag']}", x=.01,
                 ha="left", fontsize=11, weight="bold")
    return _save(fig, outdir, name)


# ---------------------------------------------------------------------------
def fig_a7(res, results, outdir, name="A7_mixing"):
    """§9 mixing structure, per-subject estimability, and the §11 gates."""
    K = res["k"]
    lab = res["state_labels"]
    labels = np.asarray(results["labels"])
    cl = results.get("clinical", {})
    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=.6, wspace=.42)

    # a: degeneracy demonstration
    ax = fig.add_subplot(gs[0, 0])
    d = res["degenerate"]
    x = np.arange(K)
    ax.plot(x, np.asarray(d["out_degree"], float), "o-", ms=3, lw=1,
            color=GREY, label="out-degree")
    rp = np.asarray(d["right_perron"], float)
    ax.plot(x, rp, "s-", ms=3, lw=1, color=BLUE, label="right Perron")
    ax.plot(x, np.asarray(d["stationary"], float) * K, "^-", ms=3, lw=1,
            color=POS, label="stationary x K")
    ax.legend(fontsize=6, frameon=False)
    ax.set_xlabel("state")
    ax.set_ylabel("value")
    _title(ax, "a", "Degenerate statistics (§9.1)",
           f"out-degree ≡ 1: {d['out_degree_is_one']}; Perron constant: "
           f"{d['right_perron_is_constant']}")

    # b: MFPT on the jump chain
    ax = fig.add_subplot(gs[0, 1])
    M = np.asarray(res["companions_jump"]["mfpt"], float)
    im = ax.imshow(M, cmap="magma_r")
    plt.colorbar(im, ax=ax, fraction=.046, label="jumps")
    ax.set_xticks(range(K)); ax.set_xticklabels(range(K), fontsize=6)
    ax.set_yticks(range(K)); ax.set_yticklabels(lab, fontsize=6)
    ax.set_xlabel("to state")
    hardest = int(np.argmax(np.asarray(res["companions_jump"]["mean_inbound"])))
    _title(ax, "b", "Mean first passage (jump chain)",
           f"hardest to reach: {lab[hardest]}")

    # c: companions
    ax = fig.add_subplot(gs[0, 2])
    clo = np.asarray(res["companions_jump"]["closeness"], float)
    inb = np.asarray(res["companions_jump"]["mean_inbound"], float)
    ax.barh(np.arange(K) - .2, clo / clo.max(), height=.38, color=BLUE,
            label="closeness (scaled)")
    ax.barh(np.arange(K) + .2, inb / inb.max(), height=.38, color="#d9a441",
            label="mean inbound (scaled)")
    ax.set_yticks(range(K)); ax.set_yticklabels(lab, fontsize=6)
    ax.legend(fontsize=6, frameon=False)
    _title(ax, "c", "Graph companions (§9.2)",
           f"Kemeny = {res['companions_jump']['kemeny']:.1f} jumps / "
           f"{res['companions_full']['kemeny_seconds']:.1f} s")

    # d: per-subject Kemeny with block-bootstrap CI
    ax = fig.add_subplot(gs[1, 0])
    kem = np.asarray(res["kemeny_per_subject"], float)
    lo = np.asarray(res["kemeny_lo"], float)
    hi = np.asarray(res["kemeny_hi"], float)
    o = np.argsort(kem)
    ax.errorbar(range(len(kem)), kem[o],
                yerr=[np.clip(kem[o] - lo[o], 0, None),
                      np.clip(hi[o] - kem[o], 0, None)],
                fmt="none", ecolor="#c8d3de", lw=1)
    ax.scatter(range(len(kem)), kem[o],
               c=[POS if labels[i] else NEG for i in o], s=18, zorder=3,
               edgecolors="none")
    e = res["estimability"]
    ax.set_xlabel("subject (sorted)")
    ax.set_ylabel("Kemeny constant (jumps)")
    _title(ax, "d", "Per-subject mixing (§9.3)",
           f"CI/range = {e['ratio']:.2f} -> "
           f"{'PASS' if e['passed'] else 'FAIL'} estimability gate")

    # e: shrinkage selection
    ax = fig.add_subplot(gs[1, 1])
    al = res["alpha"]
    ax.plot(np.asarray(al["grid"], float), np.asarray(al["loglik_grid"], float),
            "o-", ms=3, lw=1.2, color=BLUE)
    ax.axvline(al["alpha"], color=POS, lw=1.5,
               label=f"$\\alpha$ = {al['alpha']:.2f}")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_xlabel("shrinkage $\\alpha$")
    ax.set_ylabel("held-out log-likelihood")
    _title(ax, "e", "Shrinkage selection (§9.3)",
           "degenerate (alpha >= 0.9)" if al["degenerate"]
           else "per-subject matrices carry information")

    # f: gates
    ax = fig.add_subplot(gs[1, 2])
    gates = cl.get("gates", {})
    if gates:
        ks = list(gates)
        ax.barh(range(len(ks))[::-1], [1] * len(ks),
                color=[GREEN if gates[k] else POS for k in ks])
        ax.set_yticks(range(len(ks))[::-1])
        ax.set_yticklabels([k.replace(") [", ")\n[") for k in ks], fontsize=5.5)
        for i, k in enumerate(ks):
            ax.text(1.03, len(ks) - 1 - i, "PASS" if gates[k] else "FAIL",
                    va="center", fontsize=7, fontweight="bold",
                    color=GREEN if gates[k] else POS)
        ax.set_xticks([]); ax.set_xlim(0, 1.35)
    _title(ax, "f", "Pre-specified gates (§11)",
           "a failing gate blocks the claim regardless of p")

    fig.suptitle(f"A7 mixing structure — {res['tag']}", x=.01, ha="left",
                 fontsize=11, weight="bold")
    return _save(fig, outdir, name)


# ---------------------------------------------------------------------------
def fig_controls(res, results, outdir, name="A0_controls"):
    """§2.3 duration, §6.3 centring, §7.4 estimator choice, §10 inference."""
    cl = results.get("clinical")
    if not cl:
        return None
    labels = np.asarray(results["labels"])
    phi = np.asarray(res["phi"]["excess"], float)
    phiz = np.asarray(res["phi_z"], float)
    kem = np.asarray(res["kemeny_per_subject"], float)
    lengths = np.asarray([len(np.atleast_1d(s)) for s in res["visit_sequences"]],
                         float)
    logL = np.log(np.clip(lengths, 1, None))

    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=.55, wspace=.4)
    du = cl["duration"]

    # a: duration bias, excess vs z-score
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(logL, (phi - np.nanmean(phi)) / np.nanstd(phi), s=20, c=BLUE,
               label=f"excess  $\\rho$={du['phi_vs_logL']['rho']:+.2f}")
    if np.isfinite(phiz).any():
        ax.scatter(logL, (phiz - np.nanmean(phiz)) / np.nanstd(phiz), s=20,
                   c=POS, marker="^",
                   label=f"z-score  $\\rho$={du['phi_z_vs_logL']['rho']:+.2f}")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_xlabel("log number of visits")
    ax.set_ylabel("standardised value")
    _title(ax, "a", "Why the z-score form is rejected (§7.4)",
           "the z-score scales with recording length")

    # b: split-half
    ax = fig.add_subplot(gs[0, 1])
    sh = cl["phi_split_half"]
    h = results.get("_halves")
    if h is not None:
        ax.scatter(h[0], h[1], s=22, c=[POS if v else NEG for v in labels])
        lim = [np.nanmin(h), np.nanmax(h)]
        ax.plot(lim, lim, "k--", lw=.8)
    ax.set_xlabel("$\\Phi$ first half")
    ax.set_ylabel("$\\Phi$ second half")
    _title(ax, "b", "Split-half reliability (§10.8)",
           f"$r_{{SB}}$ = {sh['r_sb']:+.3f} "
           f"[{sh['r_sb_lo']:+.2f}, {sh['r_sb_hi']:+.2f}] — gate "
           f"{'PASS' if np.isfinite(sh['r_sb']) and sh['r_sb'] >= 0.6 else 'FAIL'}")

    # c: centring comparison
    ax = fig.add_subplot(gs[0, 2])
    c = res["centring"]
    for i, (mode, col) in enumerate((("single", GREY), ("double", BLUE))):
        ax.barh(i, c[mode]["max"] - c[mode]["min"], left=c[mode]["min"],
                color=col, height=.5)
        ax.plot(c[mode]["mean"], i, "k|", ms=14)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["single-centred",
                                               "double-centred"], fontsize=7)
    ax.axvline(0, color="k", lw=.7)
    ax.set_xlabel("off-diagonal range of $S$")
    _title(ax, "c", "Why double centring (§6.3)",
           "single centring cannot express dissimilarity")

    # d: effect sizes with intervals
    ax = fig.add_subplot(gs[1, 0])
    rows = [("$\\Phi$ (A1)", cl["phi_test"]), ("$\\mathcal{K}$ (A7)",
                                               cl["kemeny_test"])]
    for i, (nm, r) in enumerate(rows):
        ax.errorbar(r["auc"], i, xerr=[[r["auc"] - r["auc_lo"]],
                                       [r["auc_hi"] - r["auc"]]],
                    fmt="o", color=BLUE, ms=6, capsize=3)
        ax.text(r["auc"], i + .18, f"p={r['p']:.3g} (maxT "
                f"{cl['maxt'][['phi', 'kemeny'][i]]['p_maxT']:.3g})",
                ha="center", fontsize=6)
    mde = cl["mde"]["auc"]
    ax.axvline(0.5, color="k", lw=.8)
    for m in (mde, 1 - mde):
        ax.axvline(m, color=POS, ls=":", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC (abnormal vs normal)")
    _title(ax, "d", "Confirmatory family (§10.2, §10.7)",
           f"dotted: minimum detectable AUC {mde:.2f} at 80% power")

    # e: leave-one-out stability
    ax = fig.add_subplot(gs[1, 1])
    for i, (nm, col) in enumerate((("phi", BLUE), ("kemeny", POS))):
        if not cl.get("loo", {}).get(nm):
            continue
        ps = [r["p"] for r in cl["loo"][nm]]
        ax.scatter(np.arange(len(ps)) + .0, ps, s=12, c=col, label=nm)
    ax.axhline(0.05, color="k", ls="--", lw=.8)
    ax.set_yscale("log")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6.5, frameon=False)
    else:
        ax.text(.5, .5, "skipped\n(--controls core)", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color=GREY)
    ax.set_xlabel("dropped subject")
    ax.set_ylabel("p after dropping")
    _title(ax, "e", "Leave-one-out stability",
           "a result driven by one infant moves above the line")

    # f: redundancy
    ax = fig.add_subplot(gs[1, 2])
    rd = cl["redundancy"]
    ks = list(rd)
    ax.barh(range(len(ks))[::-1], [abs(rd[k]["rho"]) for k in ks],
            color=[POS if abs(rd[k]["rho"]) >= 0.8 else BLUE for k in ks])
    ax.axvline(0.8, color="k", ls="--", lw=.8)
    ax.set_yticks(range(len(ks))[::-1])
    ax.set_yticklabels([k.replace("_", " ") for k in ks], fontsize=6)
    ax.set_xlabel("|Spearman rho|")
    ax.set_xlim(0, 1)
    _title(ax, "f", "Redundancy gate (§11)",
           "collinear with occupancy entropy or dwell?")

    fig.suptitle(f"Controls and inference — {res['tag']}", x=.01, ha="left",
                 fontsize=11, weight="bold")
    return _save(fig, outdir, name)


# ---------------------------------------------------------------------------
def make_all(results, outdir):
    primary = results["primary"]
    res = results[primary]
    made = [fig_a1(res, results, outdir),
            fig_a5(res, results, outdir),
            fig_a7(res, results, outdir)]
    c = fig_controls(res, results, outdir)
    if c:
        made.append(c)
    for tag in results.get("models_loaded", []):
        if tag != primary and tag in results:
            made.append(fig_a1(results[tag], results, outdir,
                               f"A1_fluency_{tag.replace(' ', '_')}"))
            made.append(fig_a5(results[tag], results, outdir,
                               f"A5_metastable_{tag.replace(' ', '_')}"))
    return [m for m in made if m]
