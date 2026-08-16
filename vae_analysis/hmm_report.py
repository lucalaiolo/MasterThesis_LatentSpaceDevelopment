"""One-call HMM analysis report over the temporal-latent motion VAE.

:func:`run_hmm_report` runs the whole Stage-4 pipeline from a frozen model and a
list of videos: stitch the latent window trajectory (pose or delta stream),
check the seam, fit the Gaussian HMM (video-wise K selection), summarise each
state's dwell time, and render every figure — transition matrix + stationary
distribution, per-subject occupancy / dwell, decoded state appearances (pose
stream), the Fig-3a state movement-dynamics panel, and, when labels are given,
the clinical contrast (per-state occupancy and per-state mean dwell time with
Mann-Whitney U, effect size, exact p, Holm correction over the states, and
leave-one-subject-out).

The heavy lifting lives in :mod:`vae_analysis.hmm_pipeline`; this module only
orchestrates it and draws. All figures optionally save to ``out_dir``.

Example::

    from vae_analysis.hmm_report import run_hmm_report
    out = run_hmm_report(adapter, rvi.videos, bones=rvi.bones, limbs=rvi.limbs,
                         clip_len=cfg.clip_length, fps=25, stream="pose",
                         video_names=rvi.video_names,
                         positive_ids={"0005","0009","0010","0011","0018","0019"})
"""

from __future__ import annotations

import numpy as np

from . import hmm_pipeline as H


# ---------------------------------------------------------------------------
# small shared computations
# ---------------------------------------------------------------------------
def _stationary(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eig(A.T)
    s = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return s / s.sum()


def _per_subject_occ_dwell(states, lengths, K, f_win):
    offs = np.cumsum(np.r_[0, lengths])
    n = len(lengths)
    occ = np.zeros((n, K)); dwell = np.full((n, K), np.nan)
    pooled = {s: [] for s in range(K)}
    for b in range(n):
        seq = states[offs[b]:offs[b + 1]]
        occ[b] = np.bincount(seq, minlength=K) / max(len(seq), 1)
        runs = H._runs(seq, K) if hasattr(H, "_runs") else _runs(seq, K)
        for s in range(K):
            if runs[s]:
                dwell[b, s] = np.mean(runs[s]) / f_win
                pooled[s] += runs[s]
    mean_dwell = np.array([np.mean(pooled[s]) / f_win if pooled[s] else np.nan
                           for s in range(K)])
    return occ, dwell, mean_dwell


def _runs(seq, K):
    runs = {s: [] for s in range(K)}
    if len(seq) == 0:
        return runs
    cur, cnt = seq[0], 1
    for x in seq[1:]:
        if x == cur:
            cnt += 1
        else:
            runs[cur].append(cnt); cur, cnt = x, 1
    runs[cur].append(cnt)
    return runs


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_transition(res, *, save=None):
    import matplotlib.pyplot as plt
    A = np.asarray(res["transition"]); K = res["k"]; stat = _stationary(A)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2),
                           gridspec_kw=dict(width_ratios=[3, 1]))
    im = ax[0].imshow(A, cmap="magma", vmin=0, vmax=1)
    ax[0].set(title="transition matrix A", xlabel="to state", ylabel="from state")
    ax[0].set_xticks(range(K)); ax[0].set_yticks(range(K))
    fig.colorbar(im, ax=ax[0], fraction=.046)
    ax[1].barh(range(K), stat, color="0.4"); ax[1].invert_yaxis()
    ax[1].set(title="stationary distribution", xlabel="prob", ylabel="state")
    ax[1].set_yticks(range(K))
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plot_occupancy_dwell(occ, dwell, *, save=None):
    import matplotlib.pyplot as plt
    K = occ.shape[1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    im0 = ax[0].imshow(occ, aspect="auto", cmap="viridis")
    ax[0].set(title="per-subject state occupancy", xlabel="state", ylabel="subject")
    ax[0].set_xticks(range(K)); fig.colorbar(im0, ax=ax[0], fraction=.046)
    im1 = ax[1].imshow(dwell, aspect="auto", cmap="cividis")
    ax[1].set(title="per-subject mean dwell (s)", xlabel="state", ylabel="subject")
    ax[1].set_xticks(range(K)); fig.colorbar(im1, ax=ax[1], fraction=.046)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def _dwell_label(dwell, s):
    """``"1.23 s dwell"`` for state ``s``, or ``""`` when no dwell is available."""
    if dwell is None:
        return ""
    d = np.asarray(dwell["dwell_seconds"], float)
    if s >= len(d) or not np.isfinite(d[s]):
        return ""
    return f"{d[s]:.2f} s dwell"


def plot_state_dwell(dwell, *, save=None):
    """Mean dwell time per state: measured (Viterbi runs) vs implied by ``A_kk``.

    The bars are the mean run length of the decoded path; the markers are the
    geometric holding time ``1/(1-A_kk)`` the fitted chain implies. A state whose
    marker sits well below its bar is more persistent than a first-order chain
    predicts. ``dwell`` is :func:`hmm_pipeline.state_dwell_times` output.
    """
    import matplotlib.pyplot as plt
    meas = np.asarray(dwell["dwell_seconds"], float)
    impl = np.asarray(dwell["implied_seconds"], float)
    K = len(meas)
    fig, ax = plt.subplots(figsize=(1.0 * K + 2.6, 4.0))
    ax.bar(range(K), np.nan_to_num(meas), color="0.55", width=0.66,
           label="measured (Viterbi runs)")
    finite = np.isfinite(impl)
    ax.scatter(np.arange(K)[finite], impl[finite], s=46, color="crimson",
               zorder=3, label="implied by $A_{kk}$")
    for s in range(K):
        if np.isfinite(meas[s]):
            ax.annotate(f"{meas[s]:.2f}", (s, meas[s]), ha="center",
                        va="bottom", fontsize=8, xytext=(0, 2),
                        textcoords="offset points")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"state {s}" for s in range(K)])
    ax.set_ylabel("mean dwell time (s)")
    top = np.nanmax(np.r_[meas[np.isfinite(meas)], impl[finite], 1e-6])
    ax.set_ylim(0, top * 1.32)                      # headroom for the legend
    ax.set_title("state dwell time")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plot_state_appearances(adapter, res, dwell, bones, *, n_cols=4, save=None):
    """Decoded appearance of each state (pose stream only).

    ``dwell`` is :func:`hmm_pipeline.state_dwell_times` output (or None); each
    panel is titled with the state's measured mean dwell time.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    K = res["k"]; rows = int(np.ceil(K / n_cols))
    fig = plt.figure(figsize=(2.6 * n_cols, 2.9 * rows))
    gs = gridspec.GridSpec(rows, 2 * n_cols, figure=fig)
    axl = []
    for r in range(rows):
        items = min(n_cols, K - r * n_cols); pad = n_cols - items
        for j in range(items):
            c0 = pad + 2 * j; axl.append(fig.add_subplot(gs[r, c0:c0 + 2]))
    for s in range(K):
        pose = H.decode_state_appearance(adapter, res, s); f = pose.shape[0] // 2
        ax = axl[s]
        for a, b in bones:
            ax.plot([pose[f, a, 0], pose[f, b, 0]], [pose[f, a, 1], pose[f, b, 1]],
                    "-", lw=2)
        ax.scatter(pose[f, :, 0], pose[f, :, 1], s=10)
        ax.set_aspect("equal"); ax.axis("off")
        ttl = f"state {s}"
        d = _dwell_label(dwell, s)
        if d:
            ttl += f"\n{d}"
        ax.set_title(ttl, fontsize=9)
    fig.suptitle("decoded state appearance", weight="bold", x=0.02, ha="left")
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plot_movement_dynamics(videos, res, lengths, bones, *, clip_len, stride,
                           n_win, stream="pose", dwell=None, anchor="auto",
                           mean_arrows=True, n_sample=5000, alpha=0.05, lw=0.4,
                           clip_pctl=98, reach_bones=1.4, n_cols=4, seed=0,
                           invert_y=False, save=None):
    """Fig-3a: per-state raw-velocity cloud on the (state or global) mean pose.

    ``dwell`` is :func:`hmm_pipeline.state_dwell_times` output (or None); when
    given, each panel's title carries the state's measured mean dwell time
    alongside its occupancy.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.gridspec as gridspec

    if anchor == "auto":
        anchor = "global" if stream == "delta" else "state"
    l = clip_len // n_win
    lo = (n_win - (stride // l)) // 2; f0 = lo * l
    states = np.asarray(res["states"]); K = res["k"]
    kept = [v for v in videos if len(v) >= clip_len]
    offs = np.cumsum(np.r_[0, lengths]); J = kept[0].shape[1]

    state_vel = {s: [] for s in range(K)}
    pos_sum = {s: np.zeros((J, 2)) for s in range(K)}; pos_cnt = {s: 0 for s in range(K)}
    for i, vp in enumerate(kept):
        F, Li = len(vp), lengths[i]
        if Li == 0: continue
        vel = np.diff(vp, axis=0)
        fr = np.arange(f0, f0 + l * Li)
        flab = np.repeat(states[offs[i]:offs[i + 1]], l)
        m = fr < F - 1; fr, flab = fr[m], flab[m]
        vsel, psel = vel[fr], vp[fr]
        ok = np.isfinite(vsel).all(axis=(1, 2))
        fr, flab, vsel, psel = fr[ok], flab[ok], vsel[ok], psel[ok]
        for s in range(K):
            sm = flab == s
            if sm.any():
                state_vel[s].append(vsel[sm])
                pos_sum[s] += psel[sm].sum(0); pos_cnt[s] += int(sm.sum())
    state_vel = {s: (np.concatenate(v) if v else np.zeros((0, J, 2)))
                 for s, v in state_vel.items()}
    mean_vel = {s: (state_vel[s].mean(0) if len(state_vel[s]) else np.zeros((J, 2)))
                for s in range(K)}
    meanpose = np.concatenate(kept, axis=0).mean(0)
    statepose = {s: (pos_sum[s] / pos_cnt[s] if pos_cnt[s] else meanpose) for s in range(K)}
    anchor_of = (lambda s: statepose[s]) if anchor == "state" else (lambda s: meanpose)

    speeds = np.concatenate([np.linalg.norm(v, axis=2).ravel()
                             for v in state_vel.values() if len(v)])
    cap = np.percentile(speeds, clip_pctl)
    d_ref = np.median([np.linalg.norm(meanpose[a] - meanpose[b]) for a, b in bones])
    scale = reach_bones * d_ref / cap; reach = reach_bones * d_ref
    anchors_all = np.stack([anchor_of(s) for s in range(K)]).reshape(-1, 2)
    xlim = (anchors_all[:, 0].min() - reach - .3 * d_ref, anchors_all[:, 0].max() + reach + .3 * d_ref)
    ylim = (anchors_all[:, 1].min() - reach - .3 * d_ref, anchors_all[:, 1].max() + reach + .3 * d_ref)
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, K))

    rng = np.random.default_rng(seed)
    rows = int(np.ceil(K / n_cols))
    fig = plt.figure(figsize=(2.8 * n_cols, 3.3 * rows))
    gs = gridspec.GridSpec(rows, 2 * n_cols, figure=fig)
    axl = []
    for r in range(rows):
        items = min(n_cols, K - r * n_cols); pad = n_cols - items
        for j in range(items):
            c0 = pad + 2 * j; axl.append(fig.add_subplot(gs[r, c0:c0 + 2]))
    for s in range(K):
        ax = axl[s]; ap = anchor_of(s)
        ax.add_collection(LineCollection([[ap[a], ap[b]] for a, b in bones],
                                         colors="0.5", lw=1.0, alpha=0.4, zorder=1))
        sv = state_vel[s]
        if len(sv):
            idx = rng.choice(len(sv), size=min(n_sample, len(sv)), replace=False)
            vv = sv[idx].copy()
            nrm = np.linalg.norm(vv, axis=2, keepdims=True)
            vv *= np.minimum(1.0, cap / np.clip(nrm, 1e-9, None))
            stt = np.broadcast_to(ap, vv.shape)
            segs = np.stack([stt, stt + scale * vv], axis=2).reshape(-1, 2, 2)
            ax.add_collection(LineCollection(segs, colors=[colors[s]], lw=lw,
                                             alpha=alpha, zorder=2))
        if mean_arrows and len(sv):
            mv = np.clip(np.linalg.norm(mean_vel[s], axis=1, keepdims=True), 1e-9, None)
            mvv = mean_vel[s] * np.minimum(1.0, cap / mv)
            for j in range(J):
                ax.annotate("", xy=ap[j] + scale * mvv[j], xytext=ap[j],
                            arrowprops=dict(arrowstyle="-|>", color="0.1", lw=1.1,
                                            alpha=0.9), zorder=4)
        ax.scatter(ap[:, 0], ap[:, 1], s=7, color="0.12", zorder=3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if invert_y: ax.invert_yaxis()
        ttl = f"state {s}  (occ {res['occupancy'][s] * 100:.0f}%)"
        d = _dwell_label(dwell, s)
        if d: ttl += f"\n{d}"
        ax.set_title(ttl, fontsize=10)
    fig.suptitle(f"state movement dynamics  (stream: {stream}, anchor: {anchor})",
                 x=0.02, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plot_velocity_boxplot(dyn, *, group_colors, group_hatch=None,
                          title="state movement dynamics", save=None):
    """Fig-3b: grouped boxplots of % high-velocity frames per body group / state.

    ``dyn`` is the output of :func:`hmm_pipeline.state_movement_dynamics`; one dot
    per video, box = median + IQR. ``group_colors`` (and optional ``group_hatch``)
    map each group name to its style — e.g. head/arms/legs, or the lateral
    left_arm/right_arm/left_leg/right_leg for a left-vs-right asymmetry read.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    K = dyn["k"]; gnames = dyn["groups"]; group_hatch = group_hatch or {}
    fig, ax = plt.subplots(figsize=(1.7 * K + 2, 4.4))
    width = 0.82 / len(gnames); rng = np.random.default_rng(0)
    for gi, g in enumerate(gnames):
        col = group_colors.get(g, "0.6"); hatch = group_hatch.get(g, "")
        positions, data = [], []
        for s in range(K):
            vals = dyn[s][g]; vals = vals[~np.isnan(vals)]
            pos = s + (gi - (len(gnames) - 1) / 2) * width
            positions.append(pos); data.append(vals)
            ax.scatter(pos + rng.uniform(-width * 0.26, width * 0.26, len(vals)),
                       vals, s=13, color=col, edgecolor="none", alpha=0.5, zorder=3)
        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black"), zorder=2)
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.45, edgecolor="grey", hatch=hatch)
    handles = [Patch(facecolor=group_colors.get(g, "0.6"), alpha=.6,
                     hatch=group_hatch.get(g, ""), label=g) for g in gnames]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              ncol=min(len(gnames), 4), fontsize=8)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"state {s}" for s in range(K)])
    ax.set_ylabel("% high velocity frames"); ax.set_ylim(bottom=0)
    ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# clinical test (labels enter here only)
# ---------------------------------------------------------------------------
def clinical_test(values, y):
    """Mann-Whitney U with AUC / rank-biserial, exact p, and LOO stability."""
    from scipy.stats import mannwhitneyu
    v = np.asarray(values, float); ok = np.isfinite(v); v, yy = v[ok], np.asarray(y)[ok]
    pos, neg = v[yy == 1], v[yy == 0]
    if len(pos) < 1 or len(neg) < 1:
        return None
    U = mannwhitneyu(pos, neg, alternative="two-sided")
    auc = U.statistic / (len(pos) * len(neg)); rb = 2 * auc - 1
    try:
        p = mannwhitneyu(pos, neg, alternative="two-sided", method="exact").pvalue
        method = "exact"
    except Exception:
        p, method = U.pvalue, "asymptotic(ties)"
    loo_p, loo_auc = [], []
    for d in range(len(v)):
        k = np.ones(len(v), bool); k[d] = False
        a, b = v[k][yy[k] == 1], v[k][yy[k] == 0]
        if len(a) and len(b):
            uu = mannwhitneyu(a, b, alternative="two-sided")
            loo_p.append(uu.pvalue); loo_auc.append(uu.statistic / (len(a) * len(b)))
    return dict(U=float(U.statistic), auc=float(auc), rank_biserial=float(rb),
                p=float(p), p_method=method, direction=("abnormal>normal" if auc > .5
                else "normal>abnormal"), median_pos=float(np.median(pos)),
                median_neg=float(np.median(neg)), loo_p=loo_p, loo_auc=loo_auc,
                n_pos=int(len(pos)), n_neg=int(len(neg)))


def holm_bonferroni(pvals):
    """Holm step-down adjusted p-values (monotone, same order as the input).

    The per-state contrasts are ``2K`` tests over one dataset, so the raw p's are
    not the whole story; Holm is uniformly more powerful than Bonferroni and
    makes no independence assumption. ``NaN`` entries pass through untouched.
    """
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        out[idx] = running
    return out


def _labels_from_names(names, positive_ids):
    import re
    pos = {str(p).zfill(4) for p in positive_ids}
    y = np.array([int(any(t.zfill(4) in pos for t in re.findall(r"\d+", str(n))))
                  for n in names])
    found = {t.zfill(4) for n in names for t in re.findall(r"\d+", str(n))}
    return y, (pos - found)


def plot_clinical(panels, y, *, n_cols=4, ylabel=None,
                  suptitle="clinical contrast (labels; exploratory)",
                  subtitles=None, save=None):
    """Group contrast for each panel: one dot per subject, box = median + IQR.

    ``panels`` is a list of ``(title, values)``; ``subtitles`` optionally maps a
    panel index to a second title line (e.g. the test's AUC and p).
    """
    import matplotlib.pyplot as plt
    y = np.asarray(y)
    n = max(len(panels), 1)
    n_cols = max(1, min(n_cols, n))
    rows = int(np.ceil(n / n_cols))
    fig, ax = plt.subplots(rows, n_cols, figsize=(3.5 * n_cols, 3.8 * rows),
                           squeeze=False)
    rng = np.random.default_rng(0)
    for k, (ttl, vals) in enumerate(panels):
        a = ax[k // n_cols][k % n_cols]; vv = np.asarray(vals, float)
        g0 = vv[(y == 0) & np.isfinite(vv)]; g1 = vv[(y == 1) & np.isfinite(vv)]
        a.boxplot([g0, g1], showfliers=False, widths=.5)
        a.set_xticks([1, 2]); a.set_xticklabels(["normal (0)", "abnormal (1)"])
        for xi, g in [(1, g0), (2, g1)]:
            a.scatter(np.full(len(g), xi) + rng.uniform(-.08, .08, len(g)),
                      g, s=18, alpha=.6, color="crimson" if xi == 2 else "0.3", zorder=3)
        sub = (subtitles or {}).get(k)
        a.set_title(f"{ttl}\n{sub}" if sub else ttl, fontsize=9)
        a.set_ylabel(ylabel or ttl)
    for k in range(len(panels), rows * n_cols):
        ax[k // n_cols][k % n_cols].axis("off")
    fig.suptitle(suptitle, weight="bold")
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# persist / restore the fitted HMM (joblib bundle)
# ---------------------------------------------------------------------------
def save_hmm(path, res, Z, lengths, vidid, *, stream=None, fps=None,
             f_win=None, clip_len=None, n_win=None, compress=3):
    """Dump the fitted HMM + stitch outputs so nothing has to re-run.

    Matches the established bundle layout: the whole ``res`` (hmmlearn model +
    states / means / covars / occupancy / dwell / ...), the stitched trajectory
    ``Z`` / ``lengths`` / ``vidid`` (tiny at ``d=8``, so you skip the
    encode-stitch on reload too), a version-proof ``model_params`` backup to
    rebuild the emissions without a refit, and a ``meta`` block.
    """
    import os, joblib, hmmlearn
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    m = res["model"]
    try:                                       # hmmlearn GaussianHMM
        model_params = {
            "startprob": m.startprob_, "transmat": m.transmat_,
            "means": m.means_, "covars": m.covars_,
            "covariance_type": m.covariance_type}
    except AttributeError:                     # ssm AR-HMM (no hmmlearn attrs)
        model_params = {
            "transmat": np.asarray(res["transition"]),
            "ar_As": res.get("ar_As"), "ar_bs": res.get("ar_bs"),
            "ar_Sigmas": res.get("ar_Sigmas"),
            "covariance_type": res.get("regularisation", {}).get(
                "final_covariance_type", "ar")}
    joblib.dump({
        "res": res,
        "Z": Z, "lengths": lengths, "vidid": vidid,
        "model_params": model_params,
        "meta": {"k": res["k"], "hmmlearn": hmmlearn.__version__,
                 "stream": stream, "fps": fps, "f_win": f_win,
                 "clip_len": clip_len, "n_win": n_win,
                 "lags": res.get("lags")},
    }, path, compress=compress)
    print(f"[saved] {path}  ({os.path.getsize(path)/1e6:.1f} MB)  K={res['k']}")
    return path


def load_hmm(path):
    """Load a :func:`save_hmm` bundle. ``d['res']['model']`` is ready to use."""
    import joblib
    return joblib.load(path)


def rebuild_hmm(model_params):
    """Reconstruct a GaussianHMM from ``model_params`` without a refit.

    The version-proof fallback if ``res['model']`` ever fails to unpickle across
    hmmlearn versions.
    """
    from hmmlearn.hmm import GaussianHMM
    mp = model_params
    ct = mp["covariance_type"]
    m = GaussianHMM(n_components=len(mp["startprob"]), covariance_type=ct)
    m.startprob_ = np.asarray(mp["startprob"])
    m.transmat_ = np.asarray(mp["transmat"])
    m.means_ = np.asarray(mp["means"])
    # The covars_ getter always returns per-state (K, d, d) / (K, d), but the
    # setter wants the native shape for the covariance type.
    cov = np.asarray(mp["covars"])
    if ct == "tied" and cov.ndim == 3:
        cov = cov[0]                       # (d, d), shared across states
    elif ct == "spherical" and cov.ndim == 2:
        cov = cov[:, 0]                    # (K,)
    m.covars_ = cov
    return m


# ---------------------------------------------------------------------------
# the one call
# ---------------------------------------------------------------------------
def run_hmm_report(adapter, videos, *, bones, limbs, clip_len, stride=None,
                   stream="pose", n_win=None, k_range=range(2, 9), fps=25,
                   selection="cv", n_splits=5, n_restarts=5,
                   n_iter=200, n_jobs=1, seed=0, top_frac=0.10,
                   model="hmm", lags=1, velocity_grouping="regions",
                   video_names=None, labels=None, positive_ids=None,
                   out_dir=None, save_hmm_to=None, reuse=None,
                   show=True) -> dict:
    """Fit the HMM and render every figure in one call.

    Args:
        adapter: :class:`ArchitecturesAdapter` around a frozen temporal VAE.
        videos: list of ``(F, J, D)`` recordings.
        bones, limbs: skeleton (e.g. ``bundle.bones`` / ``bundle.limbs``).
        clip_len: VAE input length; stride defaults to ``clip_len // 2``.
        stream: ``"pose"`` or ``"delta"``.
        k_range / selection / n_splits / n_restarts: passed to :func:`fit_hmm`.
        fps: native frame rate; ``f_win = fps / (clip_len / n_windows)``.
        video_names / labels / positive_ids: supply either an explicit ``labels``
            array (aligned to kept-video order) or ``video_names`` + a set of
            ``positive_ids`` to derive labels; omit all three to skip the
            clinical test.
        model: ``"hmm"`` (static-Gaussian HMM via hmmlearn, default) or
            ``"arhmm"`` (autoregressive HMM via ssm — each state a linear
            dynamical regime). Everything downstream is identical; the AR path
            skips the decoded state-appearance figure (AR states have no single
            pose) and uses ``lags``. ``selection`` is mapped to ``"cv"``/``"none"``
            for the AR path, and ``n_jobs`` does not apply to it.
        lags: AR order for ``model="arhmm"`` — an int or a list to sweep
            (jointly selected with K). Ignored for ``model="hmm"``.
        velocity_grouping: body grouping for the Fig-3b velocity boxplot —
            ``"regions"`` (head/arms/legs, default), ``"lateral"`` (left_arm/
            right_arm/left_leg/right_leg, for left-vs-right asymmetry), or
            ``"side"`` (whole left vs whole right).
        out_dir: if set, every figure is saved there as PNG.
        save_hmm_to: if set, the fitted model + stitch outputs are dumped there as
            a joblib bundle (see :func:`save_hmm`) so a reload skips both the
            encode-stitch and the refit. Works for both model types.
        reuse: a :func:`save_hmm` bundle — its path, or the already-loaded dict —
            to rebuild the report from. Skips the encode-stitch **and** the K
            sweep entirely and redraws every figure from the stored fit, which
            turns an hours-long run into seconds. ``videos`` is still needed for
            the raw-velocity figures (they read the recordings, not the latent),
            and the stored geometry wins over the arguments passed here so the
            Viterbi states stay aligned to the frames they came from; any
            disagreement is printed. ``k_range`` / ``selection`` / ``n_restarts``
            / ``n_iter`` / ``n_jobs`` / ``seed`` are unused in this mode.
        show: call ``plt.show()`` on each figure (notebook display).

    Returns:
        Dict with ``res``, ``dwell_times`` (per-state mean dwell, measured and
        ``A_kk``-implied), ``occ``, ``dwell``, ``mean_dwell``, ``feats``
        (occupancy|dwell), ``seam``, ``clinical`` (or None), and ``figures``
        (name -> Figure).
    """
    import os, time
    import matplotlib.pyplot as plt
    stride_arg = stride            # kept so a reused bundle can re-derive it
    stride = stride or clip_len // 2
    n_win = n_win or adapter.n_windows()
    l = clip_len // n_win
    f_win = fps / l
    figs = {}
    t_run = time.time()

    def _save(name):
        return (os.path.join(out_dir, f"{name}.png") if out_dir else None)

    def _stage(msg):
        print(f"\n=== {msg}  [+{H._fmt_dur(time.time() - t_run)}] ===", flush=True)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[setup]  {len(videos)} videos | stream={stream} | clip_len={clip_len} "
          f"stride={stride} n_win={n_win} | fps={fps} -> f_win={f_win:.3f} Hz "
          f"(1 window = {1 / f_win:.3f} s) | model={model}", flush=True)

    if reuse is not None:
        # 1-2. reload a finished fit instead of redoing the encode and the sweep.
        _stage("1/5  reuse a saved fit (no encode, no refit)")
        bundle = (reuse if isinstance(reuse, dict) else load_hmm(reuse))
        try:
            res = bundle["res"]; Z = bundle["Z"]
            lengths = bundle["lengths"]; vidid = bundle["vidid"]
        except (KeyError, TypeError) as e:
            raise ValueError(
                "reuse must be a save_hmm bundle (or its path) carrying "
                f"'res'/'Z'/'lengths'/'vidid'; got {type(bundle).__name__} "
                f"missing {e}.") from None
        # The window->frame map of every figure depends on this geometry, so
        # take it from the bundle and say so when the call disagrees: silently
        # honouring the caller would misalign states against frames.
        meta = bundle.get("meta") or {}
        for name, passed in (("stream", stream), ("clip_len", clip_len),
                             ("n_win", n_win), ("fps", fps)):
            stored = meta.get(name)
            if stored is not None and stored != passed:
                print(f"[reuse]  !! {name}={passed!r} in this call but "
                      f"{stored!r} in the bundle — using the bundle's value "
                      f"so the states line up with the frames.", flush=True)
        stream = meta.get("stream", stream)
        clip_len = meta.get("clip_len", clip_len)
        n_win = meta.get("n_win", n_win)
        fps = meta.get("fps", fps)
        l = clip_len // n_win
        f_win = meta.get("f_win") or fps / l
        # re-derive from the bundle's clip_len unless the caller named a stride
        stride = stride_arg or clip_len // 2
        print(f"[reuse]  {Z.shape} over {len(lengths)} videos | K={res['k']} | "
              f"stream={stream} f_win={f_win:.3f} Hz "
              f"(hmmlearn {meta.get('hmmlearn', '?')})", flush=True)
        seam = H.seam_diagnostic(Z, lengths, clip_len=clip_len, n_win=n_win,
                                 f_win=f_win)
        _stage("2/5  fit reused")
    else:
        # 1. stitch + seam
        _stage(f"1/5  encode + stitch ({len(videos)} recordings through the VAE)")
        Z, lengths, vidid = H.stitch_dataset(adapter, videos, clip_len=clip_len,
                                             stride=stride, stream=stream,
                                             verbose=True)
        print(f"[stitch] {Z.shape} over {len(lengths)} videos "
              f"({np.sum(lengths) / f_win / 60:.1f} min of windowed motion)",
              flush=True)
        seam = H.seam_diagnostic(Z, lengths, clip_len=clip_len, n_win=n_win,
                                 f_win=f_win)
        print(f"[seam]   f={seam['f_seam']:.3f}Hz ratio={seam['max_ratio']:.1f} "
              f"passed={seam['passed']}", flush=True)

        # 2. fit the state model (static HMM or autoregressive HMM)
        _stage(f"2/5  fit {model} and select K")
        if model == "arhmm":
            from . import arhmm as _arhmm
            ar_sel = "cv" if selection == "cv" else "none"
            res = _arhmm.fit_arhmm(Z, lengths, k_range=k_range, lags=lags,
                                   f_win=f_win, selection=ar_sel,
                                   n_splits=n_splits, n_restarts=n_restarts,
                                   n_iters=n_iter, seed=seed, verbose=True)
        else:
            res = H.fit_hmm(Z, lengths, k_range=k_range, f_win=f_win,
                            selection=selection, n_splits=n_splits,
                            n_restarts=n_restarts, n_iter=n_iter, seed=seed,
                            n_jobs=n_jobs, verbose=True)
    dwl = H.state_dwell_times(res)
    K = res["k"]
    print(f"[{model}]  K={K} "
          f"cov={res.get('regularisation', {}).get('final_covariance_type', '?')} "
          f"occ={np.round(res['occupancy'], 2)}", flush=True)
    print(f"[dwell]  mean dwell (s)={np.round(dwl['dwell_seconds'], 2)} "
          f"| implied by A_kk={np.round(dwl['implied_seconds'], 2)}", flush=True)

    # 3. per-subject phenotype features
    _stage("3/5  per-subject occupancy and dwell")
    occ, dwell, mean_dwell = _per_subject_occ_dwell(res["states"], lengths, K, f_win)
    feats = np.concatenate([occ, np.nan_to_num(dwell)], axis=1)
    print(f"[pheno]  feature matrix {feats.shape} (occupancy K + dwell K)",
          flush=True)

    # optional: persist the fitted HMM + stitch outputs (joblib bundle)
    if save_hmm_to:
        save_hmm(save_hmm_to, res, Z, lengths, vidid, stream=stream,
                 fps=fps, f_win=f_win, clip_len=clip_len, n_win=n_win)

    # 4. figures
    _stage("4/5  figures")

    def _fig(name, fn, *a, **kw):
        """Draw one figure, timed, and never let a plot failure kill the run."""
        t0 = time.time()
        print(f"[plots]  {name} ...", end="", flush=True)
        try:
            figs[name] = fn(*a, save=_save(name), **kw)
        except Exception as e:  # noqa: BLE001
            print(f" skipped: {type(e).__name__}: {e}", flush=True)
            return None
        print(f" done ({time.time() - t0:.1f}s)", flush=True)
        return figs[name]

    _fig("transition", plot_transition, res)
    _fig("state_dwell", plot_state_dwell, dwl)
    _fig("occupancy_dwell", plot_occupancy_dwell, occ, dwell)
    # Decoded state appearance needs a per-state mean pose — pose stream, static
    # HMM only. AR states are dynamics (no single pose), delta states are changes.
    if stream == "pose" and model != "arhmm":
        _fig("state_appearance", plot_state_appearances, adapter, res, dwl, bones)
    _fig("movement_dynamics", plot_movement_dynamics, videos, res, lengths, bones,
         clip_len=clip_len, stride=stride, n_win=n_win, stream=stream, dwell=dwl)

    # velocity boxplot (Fig-3b): % high-velocity frames per body group per state
    try:
        if velocity_grouping == "lateral":
            vgroups = H.lateral_groups(limbs)
            vcolors = {"left_arm": "#3B7DD8", "right_arm": "#D8503B",
                       "left_leg": "#3B7DD8", "right_leg": "#D8503B"}
            vhatch = {"left_leg": "//", "right_leg": "//"}
            vtitle = "state movement dynamics — left vs right (arm / leg)"
        elif velocity_grouping == "side":
            vgroups = H.side_groups(limbs)
            vcolors = {"left": "#3B7DD8", "right": "#D8503B"}; vhatch = None
            vtitle = "state movement dynamics — left vs right"
        else:  # "regions"
            vgroups = H.body_groups(limbs)
            vcolors = {"head": "#E8998D", "arms": "#EAD7A0", "legs": "#8FBF9F"}
            vhatch = None
            vtitle = "state movement dynamics — head / arms / legs"
        print(f"[plots]  high-velocity frames per {velocity_grouping} group ...",
              end="", flush=True)
        _t0 = time.time()
        dyn = H.state_movement_dynamics(videos, res, lengths, groups=vgroups,
                                        clip_len=clip_len, stride=stride,
                                        n_win=n_win, top_frac=top_frac)
        print(f" done ({time.time() - _t0:.1f}s)", flush=True)
        _fig("velocity_boxplot", plot_velocity_boxplot, dyn,
             group_colors=vcolors, group_hatch=vhatch, title=vtitle)
    except Exception as e:  # noqa: BLE001
        print(f"[plots] velocity_boxplot skipped: {e}", flush=True)

    # 5. clinical test (labels enter here only)
    _stage("5/5  clinical contrast")
    clinical = None
    if labels is None and positive_ids is not None and video_names is not None:
        kept_idx = [i for i, v in enumerate(videos)
                    if len(v) >= clip_len]
        names = [str(video_names[i]) for i in kept_idx]
        labels, missing = _labels_from_names(names, positive_ids)
        print(f"[labels] {len(labels)} subjects: {int(labels.sum())} positive / "
              f"{int((labels == 0).sum())} negative")
        if missing:
            print(f"[labels] !! positive IDs not found among kept subjects: {sorted(missing)}")
    if labels is not None:
        labels = np.asarray(labels)
        # Two per-state readouts, both straight off the Viterbi path: how much of
        # a recording sits in a state, and how long the state is held when
        # entered. 2K tests over one dataset, so Holm-corrected across them.
        names, values = [], []
        for s in range(K):
            names.append(f"occupancy_s{s}"); values.append(occ[:, s])
        for s in range(K):
            names.append(f"dwell_s{s}"); values.append(dwell[:, s])
        tests = {n: clinical_test(v, labels) for n, v in zip(names, values)}
        p_adj = holm_bonferroni([tests[n]["p"] if tests[n] else np.nan
                                 for n in names])
        clinical = {"tests": tests,
                    "p_holm": {n: (float(q) if np.isfinite(q) else None)
                               for n, q in zip(names, p_adj)},
                    "values": dict(zip(names, values))}
        print(f"[clinical] {len(names)} per-state contrasts "
              f"(occupancy and mean dwell), Holm-corrected:")
        for n, q in zip(names, p_adj):
            r = tests[n]
            if r is None:
                print(f"  {n:14s} skipped (a group has no finite value here)")
                continue
            # A state only one subject per group ever visits leaves no LOO fold
            # with both groups populated — that emptiness is itself the finding.
            loo = (f"LOO p[{min(r['loo_p']):.4f},{max(r['loo_p']):.4f}]"
                   if r["loo_p"] else "LOO n/a (too few subjects visit it)")
            print(f"  {n:14s} AUC={r['auc']:.3f} rb={r['rank_biserial']:+.3f} "
                  f"p={r['p']:.4f}({r['p_method']}) p_holm={q:.4f} "
                  f"{r['direction']}  n={r['n_pos']}/{r['n_neg']}  {loo}")

        def _sub(n):
            r = tests[n]
            i = names.index(n)
            return (f"AUC {r['auc']:.2f}, p {r['p']:.3f} "
                    f"(Holm {p_adj[i]:.3f})") if r else "not testable"
        _fig("clinical_occupancy", plot_clinical,
             [(f"state {s}", occ[:, s]) for s in range(K)], labels,
             ylabel="occupancy",
             subtitles={s: _sub(f"occupancy_s{s}") for s in range(K)},
             suptitle="clinical contrast — per-state occupancy (exploratory)")
        _fig("clinical_dwell", plot_clinical,
             [(f"state {s}", dwell[:, s]) for s in range(K)], labels,
             ylabel="mean dwell (s)",
             subtitles={s: _sub(f"dwell_s{s}") for s in range(K)},
             suptitle="clinical contrast — per-state mean dwell time (exploratory)")
        print("[caveat] wide CIs at few positives; exploratory. LOO ranges show "
              "fragility.", flush=True)
    else:
        print("[clinical] skipped (no labels / positive_ids given)", flush=True)

    print(f"\n=== done in {H._fmt_dur(time.time() - t_run)} | "
          f"{len(figs)} figures"
          + (f" saved to {out_dir}" if out_dir else "") + " ===", flush=True)

    if show:
        for f in figs.values():
            plt.figure(f.number); plt.show()

    return dict(res=res, dwell_times=dwl, occ=occ, dwell=dwell,
                mean_dwell=mean_dwell, feats=feats, seam=seam,
                clinical=clinical, lengths=lengths, figures=figs)
