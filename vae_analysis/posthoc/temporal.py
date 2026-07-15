"""Temporal structure of the outer-loop trajectory ([post-hoc plan §4]).

The static per-clip latent folds fast dynamics into one vector; slow
structure between clips (and freezing events) lives in the ordered sequence
of window latents per walk. Two complementary readouts:

    §4.2  a Gaussian HMM over the concatenated trajectories — behavioural
          regimes, dwell times, transition structure, and how state usage
          differs across cohort / freezer / medication;
    §4.3  PELT change points, with the clean external validation: do the
          detected changes line up with E-LC's annotated FoG events?

Both are dependency-guarded (``hmmlearn`` for the HMM, ``ruptures`` for
PELT) and degrade to a clear skip when the package is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import palette as pal


# ============================================================================
# §4.2  Gaussian HMM — behavioural regimes
# ============================================================================

@dataclass
class HMMResult:
    k: int
    transmat: np.ndarray
    means: np.ndarray
    occupancy: np.ndarray                    # overall state occupancy
    dwell_mean_seconds: np.ndarray           # analytic per-state mean dwell
    empirical_dwell: dict                    # state -> array of run durations
    state_paths: list                        # per included walk (viterbi)
    included: list                           # walk indices included
    per_window_state: np.ndarray
    per_window_walk: np.ndarray
    bic_curve: dict
    stride_seconds: float
    pca_basis: tuple                         # (mean, Vt) shared PCA for plots


def _hmm_n_params(k: int, d: int) -> int:
    """Free parameters of a full-covariance Gaussian HMM (for BIC)."""
    means = k * d
    covs = k * d * (d + 1) // 2
    start = k - 1
    trans = k * (k - 1)
    return means + covs + start + trans


def fit_hmm(data, model: str, states_range=range(2, 9),
            stride_seconds: float = 1.0, min_len: int = 3,
            seed: int = 0) -> HMMResult | None:
    """Fit a Gaussian HMM on the concatenated outer-loop trajectories ([§4.2]).

    Sequence boundaries are respected (``lengths`` passed to hmmlearn, walks
    are never shuffled together). The state count is chosen by BIC over
    ``states_range``. Returns ``None`` if hmmlearn is missing or there is
    too little data.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None

    traj = data.traj.get(model, [])
    seqs, lengths, walk_of_seq = [], [], []
    for wi, z in zip([m.walk_index for m in data.traj_meta], traj):
        if len(z) >= min_len:
            seqs.append(np.asarray(z, dtype=np.float64))
            lengths.append(len(z))
            walk_of_seq.append(wi)
    if len(seqs) < 2 or sum(lengths) < max(states_range) * 3:
        return None

    X = np.concatenate(seqs, axis=0)
    lengths = np.asarray(lengths)
    d = X.shape[1]

    best, best_bic, best_k = None, np.inf, None
    bic_curve = {}
    for k in states_range:
        if k >= len(X):
            continue
        try:
            hmm = GaussianHMM(n_components=k, covariance_type="full",
                              n_iter=100, random_state=seed)
            hmm.fit(X, lengths)
            ll = hmm.score(X, lengths)
        except Exception:
            continue
        bic = -2 * ll + _hmm_n_params(k, d) * np.log(len(X))
        bic_curve[k] = float(bic)
        if bic < best_bic:
            best, best_bic, best_k = hmm, bic, k
    if best is None:
        return None

    # Decode per walk (respecting boundaries), collect per-window states.
    state_paths, per_state, per_walk = [], [], []
    offset = 0
    for L, wi in zip(lengths, walk_of_seq):
        seg = X[offset:offset + L]
        states = best.predict(seg)
        state_paths.append(states)
        per_state.append(states)
        per_walk.append(np.full(L, wi))
        offset += L
    per_window_state = np.concatenate(per_state)
    per_window_walk = np.concatenate(per_walk)

    occ = np.array([np.mean(per_window_state == s) for s in range(best_k)])
    dwell = stride_seconds / (1.0 - np.clip(np.diag(best.transmat_), 0, 1 - 1e-9))
    empirical = _empirical_dwell(state_paths, best_k, stride_seconds)

    # Shared PCA basis (fit on all windows) for the trajectory plot.
    mean = X.mean(axis=0)
    Vt = np.linalg.svd(X - mean, full_matrices=False)[2]

    return HMMResult(
        k=best_k, transmat=best.transmat_, means=best.means_,
        occupancy=occ, dwell_mean_seconds=dwell, empirical_dwell=empirical,
        state_paths=state_paths, included=walk_of_seq,
        per_window_state=per_window_state, per_window_walk=per_window_walk,
        bic_curve=bic_curve, stride_seconds=stride_seconds,
        pca_basis=(mean, Vt))


def _empirical_dwell(state_paths, k, stride_seconds) -> dict:
    """Run-length dwell durations (seconds) per state, from the Viterbi paths."""
    runs = {s: [] for s in range(k)}
    for path in state_paths:
        if len(path) == 0:
            continue
        cur, length = path[0], 1
        for s in path[1:]:
            if s == cur:
                length += 1
            else:
                runs[int(cur)].append(length * stride_seconds)
                cur, length = s, 1
        runs[int(cur)].append(length * stride_seconds)
    return {s: np.asarray(v, dtype=np.float64) for s, v in runs.items()}


def hmm_state_usage(hmm: HMMResult, data, group_key: str) -> dict:
    """Per-group state occupancy ([post-hoc plan §4.2]).

    ``group_key`` is ``"cohort"``, ``"freezer"``, or ``"med"``. Returns
    ``{group_value: occupancy_vector}`` where each vector sums to 1 over the
    HMM states. Freezers spending more time in a particular state, or OFF
    walks showing different usage, would be a clean result.
    """
    meta_by_walk = {m.walk_index: m for m in data.traj_meta}
    groups: dict = {}
    for wi, state in zip(hmm.per_window_walk, hmm.per_window_state):
        m = meta_by_walk.get(int(wi))
        if m is None:
            continue
        if group_key == "cohort":
            g = m.cohort_name
        else:
            from .data import normalise_walk_label
            g = normalise_walk_label(group_key, m.labels)
        if g is None:
            continue
        groups.setdefault(g, []).append(int(state))
    usage = {}
    for g, states in groups.items():
        counts = np.bincount(states, minlength=hmm.k).astype(np.float64)
        usage[g] = counts / max(counts.sum(), 1)
    return usage


def plot_hmm_transition(hmm: HMMResult, out_dir, model: str,
                        name: str | None = None):
    """Transition-matrix heatmap, probabilities annotated ([§4.2])."""
    plt = pal.import_matplotlib()
    T = hmm.transmat
    fig, ax = plt.subplots(figsize=(1.1 * hmm.k + 2, 1.1 * hmm.k + 1.5))
    im = ax.imshow(T, cmap=pal.HEATMAP_CMAP, vmin=0, vmax=1)
    for i in range(hmm.k):
        for j in range(hmm.k):
            ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center",
                    color="white" if T[i, j] < 0.6 else "black", fontsize=8)
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    ax.set_xticks(range(hmm.k))
    ax.set_yticks(range(hmm.k))
    ax.set_title(f"{model} HMM transition matrix (K={hmm.k})")
    fig.colorbar(im, ax=ax, label="transition probability")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name or f"hmm_transition_{model}.png")


def plot_hmm_trajectories(hmm: HMMResult, data, out_dir, model: str,
                          n_walks: int = 6, name: str | None = None):
    """State-coloured outer-loop trajectories for a few walks ([§4.2])."""
    plt = pal.import_matplotlib()
    mean, Vt = hmm.pca_basis
    traj_by_walk = {m.walk_index: z for m, z in
                    zip(data.traj_meta, data.traj[model])}
    picks = hmm.included[:n_walks]
    if not picks:
        return None
    ncol = min(3, len(picks))
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (wi, states) in zip(axes.flat, zip(picks, hmm.state_paths)):
        ax.axis("on")
        z = traj_by_walk[wi]
        proj = (z - mean) @ Vt[:2].T
        ax.plot(proj[:, 0], proj[:, 1], "-", color="#BBBBBB", alpha=0.7,
                linewidth=1, zorder=1)
        ax.scatter(proj[:, 0], proj[:, 1],
                   c=[pal.state_color(s) for s in states], s=22, zorder=2)
        ax.scatter(proj[0, 0], proj[0, 1], marker="o", s=60,
                   facecolor="none", edgecolor="black", label="start")
        m = next(mm for mm in data.traj_meta if mm.walk_index == wi)
        ax.set_title(f"walk {wi} ({m.cohort_name})", fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=pal.state_color(s), markersize=8,
                          label=f"state {s}") for s in range(hmm.k)]
    fig.legend(handles=handles, loc="upper right", fontsize=8, title="state")
    fig.suptitle(f"{model} — outer-loop trajectories coloured by HMM state",
                 y=1.02)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name or f"hmm_trajectories_{model}.png")


def plot_hmm_occupancy(hmm: HMMResult, data, out_dir, model: str,
                       name: str | None = None):
    """Stacked state-occupancy bars grouped by cohort and by label ([§4.2])."""
    plt = pal.import_matplotlib()
    group_keys = [k for k in ("cohort", "freezer", "med")
                  if _group_present(hmm, data, k)]
    if not group_keys:
        return None
    fig, axes = plt.subplots(1, len(group_keys),
                             figsize=(4.6 * len(group_keys), 4.2),
                             squeeze=False)
    for ax, gk in zip(axes[0], group_keys):
        usage = hmm_state_usage(hmm, data, gk)
        groups = list(usage.keys())
        bottom = np.zeros(len(groups))
        for s in range(hmm.k):
            vals = np.array([usage[g][s] for g in groups])
            ax.bar(range(len(groups)), vals, bottom=bottom,
                   color=pal.state_color(s), label=f"state {s}")
            bottom += vals
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([str(g) for g in groups], rotation=30, ha="right")
        ax.set_ylabel("state occupancy")
        ax.set_title(f"by {gk}")
        ax.set_ylim(0, 1)
    axes[0][-1].legend(fontsize=7, title="state", bbox_to_anchor=(1.02, 1),
                       loc="upper left")
    fig.suptitle(f"{model} — HMM state usage across groups", y=1.03)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name or f"hmm_occupancy_{model}.png")


def _group_present(hmm, data, gk) -> bool:
    return len(hmm_state_usage(hmm, data, gk)) >= 2


def plot_hmm_dwell(hmm: HMMResult, out_dir, model: str,
                   name: str | None = None):
    """Per-state dwell-time distributions (violin), coloured by state ([§4.2])."""
    plt = pal.import_matplotlib()
    states = [s for s in range(hmm.k) if len(hmm.empirical_dwell[s]) > 0]
    if not states:
        return None
    fig, ax = plt.subplots(figsize=(1.1 * len(states) + 2, 4.0))
    data_v = [hmm.empirical_dwell[s] for s in states]
    parts = ax.violinplot(data_v, showmeans=True, showextrema=False)
    for pc, s in zip(parts["bodies"], states):
        pc.set_facecolor(pal.state_color(s))
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(states) + 1))
    ax.set_xticklabels([f"state {s}" for s in states])
    ax.set_ylabel("dwell time (seconds)")
    ax.set_title(f"{model} — HMM per-state dwell-time distributions")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name or f"hmm_dwell_{model}.png")


# ============================================================================
# §4.3  PELT change points, validated on E-LC FoG annotations
# ============================================================================

def _change_point_times(traj: np.ndarray, window_starts: np.ndarray,
                        fps: float, penalty: float) -> np.ndarray:
    """Detected change-point times (seconds) for one walk's trajectory.

    A change point at window index ``w`` marks a regime boundary; its time
    is the start of window ``w`` in seconds. Uses the existing PELT pipeline
    (``dynamics.change_points``), which falls back to a light detector if
    ruptures is absent.
    """
    from ..dynamics import change_points
    if len(traj) < 2:
        return np.empty(0)
    res = change_points(traj, penalty=penalty)
    breaks = [b for b in res["breaks"] if 0 < b < len(traj)]
    return np.asarray([window_starts[b] / float(fps) for b in breaks
                       if b < len(window_starts)])


def _boundaries_from_intervals(intervals) -> np.ndarray:
    """Onset+offset times of annotated FoG intervals, sorted."""
    b = []
    for s, e in intervals:
        b.extend([s, e])
    return np.asarray(sorted(b))


def _match_counts(detected: np.ndarray, annotated: np.ndarray,
                  tol: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching within ``tol``.

    Returns ``(n_matched_annotated, n_detected, n_annotated)`` — the pieces
    precision (matched / detected) and recall (matched / annotated) need.
    """
    if len(annotated) == 0:
        return 0, len(detected), 0
    used = np.zeros(len(detected), dtype=bool)
    matched = 0
    for a in annotated:
        if len(detected) == 0:
            break
        d = np.abs(detected - a)
        d[used] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]) and d[j] <= tol:
            used[j] = True
            matched += 1
    return matched, len(detected), len(annotated)


@dataclass
class PeltResult:
    penalty: float
    tolerances: tuple
    precision: dict                 # tol -> precision
    recall: dict                    # tol -> recall
    f1: dict                        # tol -> f1
    per_walk_hit_rate: dict         # walk_index -> recall at ref tol
    report_walks: list              # walk indices used for reporting
    tune_walks: list
    detected_by_walk: dict          # walk_index -> detected times
    ref_tol: float


def pelt_fog_validation(data, model: str, tolerances=(0.5, 1.0),
                        penalty_grid=(2.0, 5.0, 10.0, 20.0, 40.0),
                        ref_tol: float = 1.0, tune_frac: float = 0.4,
                        fog_cohort: str = "E-LC", seed: int = 0
                        ) -> PeltResult | None:
    """Validate PELT change points against E-LC FoG annotations ([§4.3]).

    The penalty is tuned on a held-out set of E-LC walks (chosen by
    *subject* so no leakage) to maximise F1 at ``ref_tol``, then applied
    frozen to the reported walks — the plan is explicit that the reported
    agreement walks must not be tuned on ([post-hoc plan §4.3, §8]).

    Returns a :class:`PeltResult`, or ``None`` if E-LC has no usable FoG
    annotations in this bundle.
    """
    meta_by_walk = {m.walk_index: m for m in data.traj_meta}
    elc = [m for m in data.traj_meta
           if m.cohort_name == fog_cohort and len(m.fog_intervals) > 0
           and len(data.traj[model][_pos(data, m.walk_index)]) >= 2]
    if len(elc) < 2:
        return None

    # Split by subject into tune / report.
    subjects = sorted({m.subject for m in elc})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_tune = max(1, int(round(len(subjects) * tune_frac)))
    tune_subj = set(subjects[:n_tune])
    tune = [m for m in elc if m.subject in tune_subj]
    report = [m for m in elc if m.subject not in tune_subj]
    if not report:                       # tiny cohort — fall back to all
        report, tune = elc, elc

    def _walk_traj(m):
        return np.asarray(data.traj[model][_pos(data, m.walk_index)])

    # Tune the penalty on the tune walks (maximise F1 at ref_tol).
    best_pen, best_f1 = penalty_grid[0], -1.0
    for pen in penalty_grid:
        tm, td, ta = 0, 0, 0
        for m in tune:
            det = _change_point_times(_walk_traj(m), m.window_starts, m.fps, pen)
            ann = _boundaries_from_intervals(m.fog_intervals)
            a, dd, aa = _match_counts(det, ann, ref_tol)
            tm += a; td += dd; ta += aa
        prec = tm / td if td else 0.0
        rec = tm / ta if ta else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_pen = f1, pen

    # Evaluate frozen penalty on the report walks, at each tolerance.
    detected_by_walk = {m.walk_index:
                        _change_point_times(_walk_traj(m), m.window_starts,
                                            m.fps, best_pen) for m in report}
    precision, recall, f1 = {}, {}, {}
    for tol in tolerances:
        tm, td, ta = 0, 0, 0
        for m in report:
            det = detected_by_walk[m.walk_index]
            ann = _boundaries_from_intervals(m.fog_intervals)
            a, dd, aa = _match_counts(det, ann, tol)
            tm += a; td += dd; ta += aa
        precision[tol] = tm / td if td else float("nan")
        recall[tol] = tm / ta if ta else float("nan")
        p, r = precision[tol], recall[tol]
        f1[tol] = (2 * p * r / (p + r)) if (p and r and (p + r)) else 0.0

    per_walk_hit = {}
    for m in report:
        det = detected_by_walk[m.walk_index]
        ann = _boundaries_from_intervals(m.fog_intervals)
        a, _, aa = _match_counts(det, ann, ref_tol)
        per_walk_hit[m.walk_index] = a / aa if aa else float("nan")

    return PeltResult(
        penalty=best_pen, tolerances=tuple(tolerances),
        precision=precision, recall=recall, f1=f1,
        per_walk_hit_rate=per_walk_hit,
        report_walks=[m.walk_index for m in report],
        tune_walks=[m.walk_index for m in tune],
        detected_by_walk=detected_by_walk, ref_tol=ref_tol)


def _pos(data, walk_index: int) -> int:
    """Position of a walk in the trajectory lists (index-aligned to traj_meta)."""
    for i, m in enumerate(data.traj_meta):
        if m.walk_index == walk_index:
            return i
    raise KeyError(walk_index)


def plot_pelt_timelines(pelt: PeltResult, data, model: str, out_dir,
                        n_walks: int = 8, name: str = "pelt_timelines.png"):
    """Per-walk timeline strips: FoG bands + detected change points ([§4.3])."""
    plt = pal.import_matplotlib()
    meta_by_walk = {m.walk_index: m for m in data.traj_meta}
    picks = pelt.report_walks[:n_walks]
    if not picks:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(picks) + 1.5))
    for row, wi in enumerate(picks):
        m = meta_by_walk[wi]
        # Annotated FoG intervals as translucent bands.
        for (s, e) in m.fog_intervals:
            ax.axhspan(row - 0.4, row + 0.4, xmin=0, xmax=1, alpha=0)  # noop keep scale
            ax.fill_betweenx([row - 0.4, row + 0.4], s, e,
                             color="#4C72B0", alpha=0.30,
                             label="_" if row else "annotated FoG")
        # Detected change points as vertical marks.
        for t in pelt.detected_by_walk[wi]:
            ax.plot([t, t], [row - 0.4, row + 0.4], color="#C44E52",
                    linewidth=1.8, label="_" if row else "PELT change point")
    ax.set_yticks(range(len(picks)))
    ax.set_yticklabels([f"walk {wi}" for wi in picks])
    ax.set_xlabel("time (seconds)")
    ax.set_title(f"{model} — E-LC PELT change points vs annotated FoG "
                 f"(penalty={pelt.penalty:g}, report split)")
    # De-duplicate legend entries.
    handles = [plt.Rectangle((0, 0), 1, 1, color="#4C72B0", alpha=0.3),
               plt.Line2D([0], [0], color="#C44E52", linewidth=1.8)]
    ax.legend(handles, ["annotated FoG", "PELT change point"],
              loc="upper right", fontsize=8)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)


def plot_pelt_pr(pelt: PeltResult, out_dir, model: str,
                 name: str = "pelt_precision_recall.png"):
    """Precision–recall across tolerance windows, coloured by tolerance ([§4.3])."""
    plt = pal.import_matplotlib()
    tols = list(pelt.tolerances)
    rec = [pelt.recall[t] for t in tols]
    prec = [pelt.precision[t] for t in tols]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sc = ax.scatter(rec, prec, c=tols, cmap=pal.SEQUENTIAL_CMAP, s=120,
                    zorder=3, edgecolor="black", linewidth=0.5)
    ax.plot(rec, prec, "-", color="#999999", alpha=0.6, zorder=2)
    for t, r, p in zip(tols, rec, prec):
        ax.annotate(f"±{t:g}s", (r, p), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xlabel("recall (annotated FoG boundaries hit)")
    ax.set_ylabel("precision (change points that hit)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{model} — PELT vs FoG: precision–recall by tolerance")
    fig.colorbar(sc, ax=ax, label="tolerance (seconds)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)


def plot_pelt_single_walk(pelt: PeltResult, data, model: str, out_dir,
                          name: str = "pelt_single_walk.png"):
    """PC1 of one walk's outer-loop trajectory with change points ([§4.3])."""
    plt = pal.import_matplotlib()
    if not pelt.report_walks:
        return None
    # Pick the report walk with the most annotated boundaries.
    meta_by_walk = {m.walk_index: m for m in data.traj_meta}
    wi = max(pelt.report_walks,
             key=lambda w: len(meta_by_walk[w].fog_intervals))
    m = meta_by_walk[wi]
    z = np.asarray(data.traj[model][_pos(data, wi)])
    if len(z) < 2:
        return None
    t = m.window_starts / m.fps
    pc1 = (z - z.mean(0)) @ np.linalg.svd(z - z.mean(0),
                                          full_matrices=False)[2][0]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, pc1, "-o", color="#333333", markersize=3, label="latent PC1")
    for (s, e) in m.fog_intervals:
        ax.axvspan(s, e, color="#4C72B0", alpha=0.25,
                   label="_" if s != m.fog_intervals[0][0] else "annotated FoG")
    for ct in pelt.detected_by_walk[wi]:
        ax.axvline(ct, color="#C44E52", linewidth=1.6,
                   label="_")
    ax.axvline(np.nan, color="#C44E52", linewidth=1.6, label="PELT change point")
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("outer-loop latent PC1")
    ax.set_title(f"{model} — walk {wi} ({m.cohort_name}): "
                 "what the segmentation responds to")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)
