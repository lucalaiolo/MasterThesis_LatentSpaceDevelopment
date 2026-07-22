"""One-call HMM analysis report over the temporal-latent motion VAE.

:func:`run_hmm_report` runs the whole Stage-4 pipeline from a frozen model and a
list of videos: stitch the latent window trajectory (pose or delta stream),
check the seam, fit the Gaussian HMM (video-wise K selection), label state
frequencies, and render every figure — transition matrix + stationary
distribution, per-subject occupancy / dwell, decoded state appearances (pose
stream), the Fig-3a state movement-dynamics panel, and, when labels are given,
the clinical contrast (raw-velocity FBR and fidgety-band-state occupancy with
Mann-Whitney U, effect size, exact permutation p, and leave-one-subject-out).

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


def plot_state_appearances(adapter, res, lab, bones, *, n_cols=4, save=None):
    """Decoded appearance of each state (pose stream only)."""
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
        ttl = f"state {s}\n{lab['implied_hz'][s]:.2f}Hz{' band' if lab['in_band'][s] else ''}"
        ax.set_title(ttl, fontsize=9)
    fig.suptitle("decoded state appearance", weight="bold", x=0.02, ha="left")
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plot_movement_dynamics(videos, res, lengths, bones, *, clip_len, stride,
                           n_win, stream="pose", lab=None, anchor="auto",
                           mean_arrows=True, n_sample=5000, alpha=0.05, lw=0.4,
                           clip_pctl=98, reach_bones=1.4, n_cols=4, seed=0,
                           invert_y=False, save=None):
    """Fig-3a: per-state raw-velocity cloud on the (state or global) mean pose."""
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
        if lab is not None: ttl += f"\n{lab['implied_hz'][s]:.2f} Hz"
        ax.set_title(ttl, fontsize=10)
    fig.suptitle(f"state movement dynamics  (stream: {stream}, anchor: {anchor})",
                 x=0.02, ha="left", fontsize=13, weight="bold")
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


def _labels_from_names(names, positive_ids):
    import re
    pos = {str(p).zfill(4) for p in positive_ids}
    y = np.array([int(any(t.zfill(4) in pos for t in re.findall(r"\d+", str(n))))
                  for n in names])
    found = {t.zfill(4) for n in names for t in re.findall(r"\d+", str(n))}
    return y, (pos - found)


def plot_clinical(panels, y, *, save=None):
    import matplotlib.pyplot as plt
    y = np.asarray(y)
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), squeeze=False)
    for k, (ttl, vals) in enumerate(panels):
        a = ax[0][k]; vv = np.asarray(vals, float)
        g0 = vv[(y == 0) & np.isfinite(vv)]; g1 = vv[(y == 1) & np.isfinite(vv)]
        a.boxplot([g0, g1], showfliers=False, widths=.5)
        a.set_xticks([1, 2]); a.set_xticklabels(["normal (0)", "abnormal (1)"])
        for xi, g in [(1, g0), (2, g1)]:
            a.scatter(np.full(len(g), xi) + np.random.default_rng(0).uniform(-.08, .08, len(g)),
                      g, s=18, alpha=.6, color="crimson" if xi == 2 else "0.3", zorder=3)
        a.set_title(ttl); a.set_ylabel(ttl)
    fig.suptitle("clinical contrast (labels; exploratory)", weight="bold")
    fig.tight_layout()
    if save: fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# persist / restore the fitted HMM (joblib bundle)
# ---------------------------------------------------------------------------
def save_hmm(path, res, Z, lengths, vidid, *, stream=None, band=None, fps=None,
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
    joblib.dump({
        "res": res,
        "Z": Z, "lengths": lengths, "vidid": vidid,
        "model_params": {
            "startprob": m.startprob_, "transmat": m.transmat_,
            "means": m.means_, "covars": m.covars_,
            "covariance_type": m.covariance_type},
        "meta": {"k": res["k"], "hmmlearn": hmmlearn.__version__,
                 "stream": stream, "band": band, "fps": fps, "f_win": f_win,
                 "clip_len": clip_len, "n_win": n_win},
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
                   band=(0.5, 2.0), selection="cv", n_splits=5, n_restarts=5,
                   n_iter=200, seed=0, top_frac=0.10,
                   video_names=None, labels=None, positive_ids=None,
                   out_dir=None, save_hmm_to=None, show=True) -> dict:
    """Fit the HMM and render every figure in one call.

    Args:
        adapter: :class:`ArchitecturesAdapter` around a frozen temporal VAE.
        videos: list of ``(F, J, D)`` recordings.
        bones, limbs: skeleton (e.g. ``bundle.bones`` / ``bundle.limbs``).
        clip_len: VAE input length; stride defaults to ``clip_len // 2``.
        stream: ``"pose"`` or ``"delta"``.
        k_range / selection / n_splits / n_restarts: passed to :func:`fit_hmm`.
        fps: native frame rate; ``f_win = fps / (clip_len / n_windows)``.
        band: fidgety band for the frequency flags.
        video_names / labels / positive_ids: supply either an explicit ``labels``
            array (aligned to kept-video order) or ``video_names`` + a set of
            ``positive_ids`` to derive labels; omit all three to skip the
            clinical test.
        out_dir: if set, every figure is saved there as PNG.
        save_hmm_to: if set, the fitted HMM + stitch outputs are dumped there as
            a joblib bundle (see :func:`save_hmm`) so a reload skips both the
            encode-stitch and the refit.
        show: call ``plt.show()`` on each figure (notebook display).

    Returns:
        Dict with ``res``, ``lab``, ``occ``, ``dwell``, ``mean_dwell``,
        ``feats`` (occupancy|dwell), ``band_states``, ``seam``, ``clinical``
        (or None), and ``figures`` (name -> Figure).
    """
    import os
    import matplotlib.pyplot as plt
    stride = stride or clip_len // 2
    n_win = n_win or adapter.n_windows()
    l = clip_len // n_win
    f_win = fps / l
    figs = {}
    def _save(name):
        return (os.path.join(out_dir, f"{name}.png") if out_dir else None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. stitch + seam
    Z, lengths, vidid = H.stitch_dataset(adapter, videos, clip_len=clip_len,
                                         stride=stride, stream=stream)
    seam = H.seam_diagnostic(Z, lengths, clip_len=clip_len, n_win=n_win, f_win=f_win)
    print(f"[stitch] {Z.shape} over {len(lengths)} videos (stream={stream})")
    print(f"[seam]   f={seam['f_seam']:.3f}Hz ratio={seam['max_ratio']:.1f} "
          f"passed={seam['passed']}")

    # 2. fit HMM + frequency labels
    res = H.fit_hmm(Z, lengths, k_range=k_range, f_win=f_win, selection=selection,
                    n_splits=n_splits, n_restarts=n_restarts, n_iter=n_iter, seed=seed)
    lab = H.label_state_frequencies(res, band=band)
    K = res["k"]
    print(f"[hmm]    K={K} cov={res['regularisation']['final_covariance_type']} "
          f"occ={np.round(res['occupancy'], 2)}")
    print(f"[freq]   implied Hz={np.round(lab['implied_hz'], 2)} "
          f"band states={lab['in_band_states']}")

    # 3. per-subject phenotype features
    occ, dwell, mean_dwell = _per_subject_occ_dwell(res["states"], lengths, K, f_win)
    feats = np.concatenate([occ, np.nan_to_num(dwell)], axis=1)
    band_states = lab["in_band_states"]
    print(f"[pheno]  feature matrix {feats.shape} (occupancy K + dwell K)")

    # optional: persist the fitted HMM + stitch outputs (joblib bundle)
    if save_hmm_to:
        save_hmm(save_hmm_to, res, Z, lengths, vidid, stream=stream, band=band,
                 fps=fps, f_win=f_win, clip_len=clip_len, n_win=n_win)

    # 4. figures
    figs["transition"] = plot_transition(res, save=_save("transition"))
    figs["occupancy_dwell"] = plot_occupancy_dwell(occ, dwell, save=_save("occupancy_dwell"))
    if stream == "pose":
        try:
            figs["state_appearance"] = plot_state_appearances(
                adapter, res, lab, bones, save=_save("state_appearance"))
        except Exception as e:  # noqa: BLE001
            print(f"[plots] state_appearance skipped: {e}")
    figs["movement_dynamics"] = plot_movement_dynamics(
        videos, res, lengths, bones, clip_len=clip_len, stride=stride, n_win=n_win,
        stream=stream, lab=lab, save=_save("movement_dynamics"))

    # 5. clinical test (labels enter here only)
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
        kept_idx = [i for i, v in enumerate(videos) if len(v) >= clip_len]
        fbr = np.array([H.raw_velocity_fbr(videos[i], fps, band) for i in kept_idx])
        band_occ = occ[:, band_states].sum(1) if band_states else np.zeros(len(occ))
        clinical = {"FBR": clinical_test(fbr, labels),
                    "band_occupancy": (clinical_test(band_occ, labels)
                                       if band_states else None)}
        for name, r in clinical.items():
            if r is None:
                print(f"[{name}] skipped"); continue
            print(f"[{name}] AUC={r['auc']:.3f} rb={r['rank_biserial']:+.3f} "
                  f"p={r['p']:.4f}({r['p_method']}) {r['direction']}  "
                  f"LOO p[{min(r['loo_p']):.4f},{max(r['loo_p']):.4f}]")
        panels = [("raw-velocity FBR", fbr)]
        if band_states:
            panels.append(("fidgety-band occupancy", band_occ))
        figs["clinical"] = plot_clinical(panels, labels, save=_save("clinical"))
        print("[caveat] wide CIs at few positives; exploratory. LOO ranges show fragility.")

    if show:
        for f in figs.values():
            plt.figure(f.number); plt.show()

    return dict(res=res, lab=lab, occ=occ, dwell=dwell, mean_dwell=mean_dwell,
                feats=feats, band_states=band_states, seam=seam,
                clinical=clinical, lengths=lengths, figures=figs)
