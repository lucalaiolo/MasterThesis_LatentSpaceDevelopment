"""HMM interpretability layer over the temporal-latent motion VAE.

Build document: ``docs/HMM_RVI38_BUILD.md`` (the RVI-38 spec). This module
implements that pipeline against the *frozen* temporal VAE, model- and
dataset-agnostic: it runs on any ``temporal_conv`` / ``temporal_transformer``
checkpoint (via :class:`ArchitecturesAdapter`) and any list of videos, so the
RVI-38 recordings plug in where synthetic clips do in the tests.

Pipeline, in order:

1. :func:`encode_window_sequence` / :func:`stitch_dataset` — the overlap-crop
   stitcher (§3, §5.1). The VAE sees only ``clip_len`` frames, so a recording
   becomes a run of clips; naive concatenation injects a seam every clip
   (a comb at ``f_frame/clip_len`` inside the fidgety band). Encoding at 50%
   overlap and keeping each clip's central windows tiles the recording with no
   seam.
2. :func:`seam_diagnostic` — PSD check that the ``f_frame/clip_len`` comb is
   gone before per-video ``lengths`` are trusted (Guardrail 3.1).
3. :func:`fit_hmm` — full-covariance Gaussian HMM with a ridge floor, shrinkage
   triggers, k-means restarts, subject-disjoint ``K`` selection, and a Viterbi
   decode (§2, §5.2; Guardrails 2.2, 2.4, 2.5).
4. :func:`akk_from_frequency` / :func:`band_power_ratio` — the fidgety-band
   frequency layer (§4), FBR on raw keypoint velocities as the headline.
5. :func:`decode_state_appearance` / :func:`phenotype_features` — interpretability
   and per-video features for clustering (§5.3-5.5).

Everything uses the posterior **mean** (Guardrail 5.0); never a sampled ``z``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Overlap-crop stitcher  (§3, §5.1)
# ---------------------------------------------------------------------------
def _clip_starts(F: int, clip_len: int, stride: int) -> list[int]:
    """Start frames of every full ``clip_len`` clip at the given stride."""
    return list(range(0, max(F - clip_len + 1, 0), stride))


def encode_window_sequence(adapter, video: np.ndarray, *, clip_len: int = 64,
                           stride: int = 32, keep: tuple[int, int] | None = None,
                           stream: str = "pose", mask=None
                           ) -> np.ndarray:
    """Per-video latent-window trajectory via overlap-crop stitching.

    Encode ``clip_len``-frame clips at ``stride`` frames, keep only the central
    windows ``[keep[0]:keep[1])`` of each clip's ``(n_win, d_z)`` block, and
    concatenate in temporal order. With ``stride`` chosen so its window count
    equals the kept-window count, the retained regions tile the recording with
    no gap and no seam, and every kept window carries intra-clip context on both
    sides.

    Args:
        adapter: an :class:`ArchitecturesAdapter` around a temporal VAE.
        video: one recording, shape ``(F, J, D)``.
        clip_len: VAE input length (frozen; 64 in the spec).
        stride: hop between clip starts, in frames. Default ``clip_len//2``
            (50% overlap). Must be a multiple of the temporal downsample ``l``.
        keep: ``(lo, hi)`` window indices to retain per clip. Default centres a
            block of ``stride/l`` windows — the value that tiles gaplessly.
        stream: ``"pose"`` returns the window means ``z_w``; ``"delta"`` returns
            the change stream ``z_{w+1}-z_w`` over the stitched trajectory.
        mask: optional ``(F, J)`` visibility; defaults to all-visible.

    Returns:
        Trajectory ``(M_v, d_z)`` of posterior means. Empty ``(0, d_z)`` when the
        video is shorter than one clip.
    """
    if stream not in ("pose", "delta"):
        raise ValueError(f"stream must be 'pose' or 'delta', got {stream!r}.")
    if not adapter.is_temporal():
        raise ValueError("encode_window_sequence needs a temporal_* model "
                         "(one exposing window_latents).")

    n_win = adapter.n_windows()
    d_z = adapter.d_z
    l = clip_len // n_win                         # frames per window
    if stride % l != 0:
        raise ValueError(f"stride ({stride}) must be a multiple of the window "
                         f"length l={l} so kept regions tile on window bounds.")
    step_win = stride // l                         # windows advanced per clip

    if keep is None:
        # Centre a block of `step_win` windows: the count that tiles gaplessly.
        lo = (n_win - step_win) // 2
        keep = (lo, lo + step_win)
    lo, hi = keep
    if not (0 <= lo < hi <= n_win):
        raise ValueError(f"keep={keep} out of range for n_win={n_win}.")
    if (hi - lo) != step_win:
        # Not fatal (caller may want overlap/gaps on purpose) but warn loudly:
        # the seam-free guarantee only holds when kept-count == stride-in-windows.
        import warnings
        warnings.warn(
            f"keep width {hi-lo} != stride-in-windows {step_win}: the stitched "
            f"trajectory will have gaps or overlaps between clips.", stacklevel=2)

    F = video.shape[0]
    J = video.shape[1]
    starts = _clip_starts(F, clip_len, stride)
    if not starts:
        return np.empty((0, d_z), np.float32)

    clips = np.stack([video[s:s + clip_len] for s in starts]).astype(np.float32)
    if mask is None:
        M = np.ones(clips.shape[:3], np.float32)
    else:
        M = np.stack([mask[s:s + clip_len] for s in starts]).astype(np.float32)

    mu_flat, _ = adapter.encode(clips, M)          # (n_clips, d_z*n_win) MEAN
    win = adapter.window_latents(mu_flat)          # (n_clips, n_win, d_z)
    kept = win[:, lo:hi, :]                         # (n_clips, step_win, d_z)
    traj = kept.reshape(-1, d_z)                    # (M_v, d_z)

    if stream == "delta":
        if len(traj) < 2:
            return np.empty((0, d_z), np.float32)
        traj = np.diff(traj, axis=0)               # continuous across the seam-free stitch
    return traj.astype(np.float32)


def stitch_dataset(adapter, videos: list[np.ndarray], *, clip_len: int = 64,
                   stride: int = 32, keep: tuple[int, int] | None = None,
                   stream: str = "pose", masks=None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch every video and stack into the HMM's ``(Z, lengths, video_id)``.

    The ``lengths`` array is what makes the HMM sum per-video log-likelihoods
    rather than concatenate windows across recording boundaries (§2.2). Videos
    too short for one clip are skipped.

    Returns:
        Z: ``(sum M_v, d_z)`` stacked trajectories.
        lengths: ``(n_kept_videos,)`` window count per retained video.
        video_id: ``(sum M_v,)`` retained-video index per window.
    """
    parts, lengths, ids = [], [], []
    kept_idx = 0
    for v, video in enumerate(videos):
        m = None if masks is None else masks[v]
        traj = encode_window_sequence(adapter, video, clip_len=clip_len,
                                      stride=stride, keep=keep, stream=stream,
                                      mask=m)
        if len(traj) == 0:
            continue
        parts.append(traj)
        lengths.append(len(traj))
        ids.append(np.full(len(traj), kept_idx, dtype=np.int64))
        kept_idx += 1
    if not parts:
        raise ValueError("No windows stitched. Are all videos shorter than "
                         "clip_len?")
    return (np.concatenate(parts), np.asarray(lengths, np.int64),
            np.concatenate(ids))


# ---------------------------------------------------------------------------
# 2. Seam diagnostic  (§3; Guardrail 3.1)
# ---------------------------------------------------------------------------
def seam_diagnostic(Z: np.ndarray, lengths: np.ndarray, *, clip_len: int,
                    n_win: int, f_win: float, tol: float = 3.0) -> dict:
    """Check the stitched trajectory for the ``f_frame/clip_len`` seam comb.

    A clip boundary every ``clip_len`` frames is a boundary every
    ``clip_len/l = n_win`` windows, i.e. a spectral line at ``f_win/n_win`` Hz
    and its harmonics in the window-sampled trajectory. This is exactly the
    ``f_frame/clip_len`` comb — inside the fidgety band — that naive
    concatenation injects. The overlap-crop stitcher should leave no such line.

    The check compares band power in a narrow bin around the seam fundamental to
    the median PSD level, per active dimension, averaged over videos.

    Args:
        Z, lengths: output of :func:`stitch_dataset`.
        clip_len, n_win: VAE geometry.
        f_win: window sampling rate (Hz) = ``f_frame / l``.
        tol: pass threshold — flag when seam-bin power exceeds ``tol`` x median.

    Returns:
        Dict with the seam frequency, the per-video ratio, the max ratio, and a
        boolean ``passed`` (max ratio below ``tol``).
    """
    from scipy.signal import welch

    f_seam = f_win / n_win                          # == f_frame / clip_len
    ratios = []
    offset = 0
    for L in lengths:
        seg = Z[offset:offset + L]
        offset += L
        if L < 2 * n_win:                           # too short to resolve the line
            continue
        nper = min(L, 4 * n_win)
        f, P = welch(seg, fs=f_win, axis=0, nperseg=nper)
        P = P.mean(axis=1)                          # average PSD over dims
        med = np.median(P[1:]) + 1e-12
        # power in the bin nearest the seam fundamental
        j = int(np.argmin(np.abs(f - f_seam)))
        ratios.append(P[j] / med)
    ratios = np.asarray(ratios) if ratios else np.array([np.nan])
    max_ratio = float(np.nanmax(ratios))
    return {"f_seam": f_seam, "per_video_ratio": ratios,
            "max_ratio": max_ratio, "passed": bool(max_ratio < tol),
            "tol": tol}

# === SOUND seam gate — replaces `assert seam["passed"]` =====================
def seam_gate(Z, lengths, *, n_win, l, f_win, stride,
              jump_tol=1.5, comb_tol=3.0, min_harm=2):
    import numpy as np
    from scipy.signal import welch
    step, blocks = stride // l, np.cumsum(np.r_[0, lengths])
    # (a) boundary-jump: extra step size at clip seams vs interior
    bnd, itr = [], []
    for a, b in zip(blocks[:-1], blocks[1:]):
        seg = Z[a:b]
        if len(seg) < 2: continue
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        m = (np.arange(1, len(seg)) % step == 0)
        bnd.append(d[m]); itr.append(d[~m])
    jump = np.concatenate(bnd).mean() / np.concatenate(itr).mean()
    # (b) local-baseline comb test at the seam harmonics
    Ps = []
    for a, b in zip(blocks[:-1], blocks[1:]):
        seg = Z[a:b]
        if len(seg) < 4 * n_win: continue
        f, P = welch(seg, fs=f_win, axis=0, nperseg=4 * n_win); Ps.append(P.mean(1))
    f_seam = f_win / n_win
    harm = f_seam * np.arange(1, 6); harm = harm[harm < f_win / 2]
    locs = []
    if Ps:
        Pavg = np.mean(Ps, axis=0)
        hbins = {int(np.argmin(np.abs(f - h))) for h in harm}
        for h in harm:
            b0 = int(np.argmin(np.abs(f - h)))
            nb = [j for j in range(max(1, b0-3), min(len(f), b0+4))
                  if j != b0 and j not in hbins]
            locs.append(Pavg[b0] / (np.median(Pavg[nb]) + 1e-12))
    locs = np.array(locs) if locs else np.array([np.nan])
    n_comb = int(np.sum(locs > comb_tol))
    return {"jump": jump, "local_harmonic": locs, "n_comb_lines": n_comb,
            "passed": bool(jump < jump_tol and n_comb < min_harm)}

# ---------------------------------------------------------------------------
# 3. Full-covariance HMM fit  (§2, §5.2; Guardrails 2.2, 2.4, 2.5)
# ---------------------------------------------------------------------------
def _cov_n_params(d: int, covariance_type: str) -> int:
    """Free covariance parameters per state at this covariance family."""
    if covariance_type == "full":
        return d * (d + 1) // 2
    if covariance_type == "tied":
        return d * (d + 1) // 2          # shared, counted once (handled by caller)
    if covariance_type == "diag":
        return d
    if covariance_type == "spherical":
        return 1
    raise ValueError(covariance_type)


def hmm_n_params(k: int, d: int, covariance_type: str = "full") -> int:
    """Total free scalar parameters of a ``k``-state Gaussian HMM (for BIC)."""
    trans = (k - 1) + k * (k - 1)                    # startprob + transmat
    means = k * d
    if covariance_type == "tied":
        cov = _cov_n_params(d, "tied")               # one shared matrix
    else:
        cov = k * _cov_n_params(d, covariance_type)
    return trans + means + cov


def _video_blocks(lengths: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) row spans of each video in the stacked trajectory."""
    ends = np.cumsum(lengths)
    starts = ends - lengths
    return list(zip(starts.tolist(), ends.tolist()))


def _subset(Z: np.ndarray, lengths: np.ndarray, keep: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """Gather the windows of the videos in ``keep`` (a boolean/index over videos).

    Videos are contiguous in ``Z`` (stitch order), so this concatenates whole
    blocks and returns the matching per-video ``lengths`` sub-array.
    """
    blocks = _video_blocks(lengths)
    keep = np.atleast_1d(keep)
    if keep.dtype == bool:
        keep_idx = np.where(keep)[0]
    else:
        keep_idx = keep
    rows = [np.arange(s, e) for i, (s, e) in enumerate(blocks) if i in set(keep_idx.tolist())]
    sub_lengths = lengths[keep_idx]
    return Z[np.concatenate(rows)], sub_lengths


def _fit_once(Z, lengths, k, covariance_type, min_covar, n_iter, seed):
    """One GaussianHMM fit; returns (model, train_loglik) or (None, -inf)."""
    from hmmlearn.hmm import GaussianHMM
    import logging, warnings
    hmm_logger = logging.getLogger("hmmlearn")
    prev = hmm_logger.level
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="hmmlearn")
            hmm_logger.setLevel(logging.ERROR)
            model = GaussianHMM(n_components=k, covariance_type=covariance_type,
                                min_covar=min_covar, n_iter=n_iter,
                                random_state=seed, init_params="stmc")
            model.fit(Z, lengths)
            ll = model.score(Z, lengths)
    except Exception:  # noqa: BLE001 — degenerate init; caller retries/skips
        return None, -np.inf
    finally:
        hmm_logger.setLevel(prev)
    if not np.isfinite(ll):
        return None, -np.inf
    return model, ll


def _best_of_restarts(Z, lengths, k, covariance_type, min_covar, n_iter,
                      n_restarts, seed):
    """Best-training-loglik model over ``n_restarts`` k-means-seeded inits."""
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        model, ll = _fit_once(Z, lengths, k, covariance_type, min_covar,
                              n_iter, seed + 1000 * r)
        if ll > best_ll:
            best, best_ll = model, ll
    return best, best_ll


def _state_condition_numbers(model) -> np.ndarray:
    """Condition number of each state's covariance (Guardrail 2.4 telemetry)."""
    covs = model.covars_
    if covs.ndim == 2:                               # diag: (k, d)
        return (covs.max(1) / np.clip(covs.min(1), 1e-30, None))
    conds = []
    for C in covs:                                   # full/tied: (k, d, d)
        ev = np.linalg.eigvalsh(C)
        conds.append(ev.max() / max(ev.min(), 1e-30))
    return np.asarray(conds)


def _viterbi_summary(model, Z, lengths, f_win: float):
    """Viterbi decode + per-video occupancy / dwell (Guardrail 2.2).

    Dwell runs are counted **within** each video so a run is never stitched
    across a recording boundary.
    """
    states = model.predict(Z, lengths)               # hmmlearn: Viterbi by default
    k = model.n_components
    occ = np.array([(states == s).mean() for s in range(k)])

    # per-video occupancy + dwell (for phenotype features) and pooled dwell runs
    runs = {s: [] for s in range(k)}
    per_video_occ, per_video_dwell = [], []
    for (s0, s1) in _video_blocks(lengths):
        seg = states[s0:s1]
        per_video_occ.append(np.array([(seg == s).mean() for s in range(k)]))
        vruns = {s: [] for s in range(k)}
        cur, cnt = seg[0], 1
        for st in seg[1:]:
            if st == cur:
                cnt += 1
            else:
                runs[cur].append(cnt); vruns[cur].append(cnt)
                cur, cnt = st, 1
        runs[cur].append(cnt); vruns[cur].append(cnt)
        per_video_dwell.append(np.array(
            [np.mean(vruns[s]) if vruns[s] else np.nan for s in range(k)]))
    dwell_win = np.array([np.mean(runs[s]) if runs[s] else np.nan
                          for s in range(k)])
    return {"states": states, "occupancy": occ,
            "per_video_occupancy": np.stack(per_video_occ),
            "per_video_dwell_windows": np.stack(per_video_dwell),
            "dwell_windows": dwell_win,
            "dwell_seconds": dwell_win / f_win}


def fit_hmm(Z: np.ndarray, lengths: np.ndarray, *, k_range=range(2, 11),
            f_win: float = 7.5, covariance_type: str = "full",
            min_covar: float = 1e-3, n_restarts: int = 5, n_iter: int = 200,
            selection: str = "cv", n_splits: int = 5, val_fraction: float = 0.2,
            seed: int = 0, cond_ceiling: float = 1e8,
            occupancy_floor_factor: float = 10.0) -> dict:
    """Fit a Gaussian HMM over the stitched window trajectory.

    Full covariance is the a-priori family (Proposition 2.3 — affine-invariant,
    so whitening is a no-op and the diagonal model's rotation-dependent
    misspecification is avoided). ``K`` is chosen by held-out log-likelihood
    under a **video-wise** split (Guardrail 2.5), never a clip split. BIC is
    reported alongside but not used to select. The returned state summaries come
    from a Viterbi decode (Guardrail 2.2).

    Guardrail 2.4: per state and per fit, occupancy count and covariance
    condition number are logged; when a state's window budget falls below
    ``occupancy_floor_factor x`` its covariance parameter count, or its condition
    number exceeds ``cond_ceiling``, the ridge floor is escalated and, if still
    triggered, the covariance family drops to ``"tied"`` (the first rung of the
    §2.6 fallback ladder). The choice is logged in ``regularisation``.

    Args:
        Z, lengths: output of :func:`stitch_dataset`.
        k_range: candidate state counts.
        f_win: window sampling rate (Hz), for dwell seconds and the frequency map.
        selection: ``"cv"`` (mean held-out LL over ``n_splits`` seeded video-wise
            splits — the default), ``"loo"`` (leave-one-video-out), or ``"bic"``.
        cond_ceiling, occupancy_floor_factor: Guardrail 2.4 triggers.

    Returns:
        Dict with ``model``, ``k``, ``transition``, ``stationary``, ``means``,
        ``covars``, Viterbi ``states`` / ``occupancy`` / ``dwell_*`` /
        ``per_video_occupancy``, per-``K`` selection scores, BIC, and the
        ``regularisation`` decision log.
    """
    Z = np.asarray(Z, np.float64)
    d = Z.shape[1]
    n_videos = len(lengths)
    M = len(Z)
    rng = np.random.default_rng(seed)

    # ---- Guardrail 2.5: candidate K capped by soft data limits -------------
    # At least ~5 windows/state (else a state goes unvisited and its transition
    # row is undefined) and K < n_videos (else a video-wise held-out split has
    # no training videos). Covariance *capacity* is not capped here: a thin
    # state escalates to the tied-covariance rung below rather than being
    # forbidden, so K stays free to explore.
    cov_pp = _cov_n_params(d, covariance_type)
    k_hi = max(2, min(n_videos, M // 5))
    ks = [k for k in k_range if 2 <= k <= k_hi]
    if not ks:
        raise ValueError(
            f"Not enough data/videos to fit an HMM: {M} windows over "
            f"{n_videos} videos allows no K in {list(k_range)}.")

    # ---- video-wise CV splits ----------------------------------------------
    def make_splits():
        if selection == "loo":
            return [np.array([v]) for v in range(n_videos)]
        if selection == "cv":
            out = []
            for s in range(n_splits):
                perm = np.random.default_rng(seed + s).permutation(n_videos)
                n_val = max(1, int(round(n_videos * val_fraction)))
                out.append(perm[:n_val])
            return out
        return []                                    # bic: no held-out

    splits = make_splits()

    # ---- select K -----------------------------------------------------------
    scores, bics = {}, {}
    for k in ks:
        # BIC on a full-data fit (reported alongside).
        full, ll_full = _best_of_restarts(Z, lengths, k, covariance_type,
                                          min_covar, n_iter, n_restarts, seed)
        bics[k] = (-2 * ll_full + hmm_n_params(k, d, covariance_type) * np.log(M)
                   if full is not None else np.inf)
        if selection == "bic":
            scores[k] = -bics[k]                     # higher is better
            continue
        # held-out mean LL per window over video-wise splits
        fold_scores = []
        for val_videos in splits:
            val_set = set(val_videos.tolist())
            train_videos = np.array([v for v in range(n_videos) if v not in val_set])
            if len(train_videos) < 1 or len(val_videos) < 1:
                continue
            Ztr, ltr = _subset(Z, lengths, train_videos)
            Zva, lva = _subset(Z, lengths, val_videos)
            model, _ = _best_of_restarts(Ztr, ltr, k, covariance_type,
                                        min_covar, n_iter, n_restarts, seed)
            if model is None:
                continue
            try:
                ll = model.score(Zva, lva)
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(ll):
                fold_scores.append(ll / len(Zva))    # per-window, comparable across folds
        scores[k] = float(np.mean(fold_scores)) if fold_scores else -np.inf
    if not scores or all(v == -np.inf for v in scores.values()):
        raise ValueError("No HMM converged for any candidate K.")
    k_best = max(scores, key=scores.get)

    # ---- final fit at K*, with Guardrail 2.4 escalation --------------------
    reg = {"covariance_type": covariance_type, "min_covar": min_covar,
           "escalations": []}
    cov_type = covariance_type
    mc = min_covar
    for attempt in range(3):
        model, _ = _best_of_restarts(Z, lengths, k_best, cov_type, mc, n_iter,
                                    n_restarts, seed)
        if model is None:
            mc *= 10
            reg["escalations"].append(f"no-converge -> min_covar={mc:g}")
            continue
        summ = _viterbi_summary(model, Z, lengths, f_win)
        conds = _state_condition_numbers(model)
        occ_counts = summ["occupancy"] * M
        floor = occupancy_floor_factor * cov_pp
        reg["state_condition"] = conds.tolist()
        reg["state_occupancy_counts"] = occ_counts.tolist()
        # Two independent triggers with distinct remedies (Guardrail 2.4):
        #  * ill-conditioning is a numerical defect -> raise the ridge floor;
        #  * an occupancy shortfall is a capacity defect (too few windows for
        #    a full matrix) -> the ridge cannot fix it, drop to tied covariance.
        cond_triggered = conds.max() > cond_ceiling
        occ_triggered = occ_counts.min() < floor
        if cond_triggered and mc < 1e-1:             # rung 0: ridge floor
            mc *= 10
            reg["escalations"].append(
                f"ill-conditioned (cond={conds.max():.1e}) -> min_covar={mc:g}")
            continue
        if occ_triggered and cov_type == "full":     # rung 1: tied covariance
            cov_type, mc = "tied", min_covar
            reg["escalations"].append(
                f"thin state (min_occ={occ_counts.min():.0f}<{floor:.0f}) -> "
                f"covariance_type='tied' (fallback ladder §2.6)")
            continue
        if cond_triggered or occ_triggered:
            reg["escalations"].append(
                "trigger persists after ladder; accepting fit (see §2.6 rungs "
                "semi-tied / factor-analysed for further reduction)")
        break
    reg["final_covariance_type"] = cov_type
    reg["final_min_covar"] = mc

    # ---- stationary distribution (left eigenvector for eigenvalue 1) --------
    A = model.transmat_
    w, V = np.linalg.eig(A.T)
    stat = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    stat = stat / stat.sum()

    return {"model": model, "k": k_best, "d": d, "f_win": f_win,
            "transition": A, "stationary": stat,
            "means": model.means_, "covars": model.covars_,
            "selection": selection, "selection_scores": scores, "bic": bics,
            "regularisation": reg, **summ}


# ---------------------------------------------------------------------------
# 4. Fidgety-band frequency layer  (§4; Guardrail 4.1)
# ---------------------------------------------------------------------------
def akk_from_frequency(f: float, f_win: float) -> float:
    """Self-transition ``A_kk`` implied by an oscillation at ``f`` Hz.

    A periodic movement reaches its two extremes at rate ``2f`` (one cycle is
    two half-cycle dwells, §4.2), so the mean half-cycle dwell is
    ``tau = f_win/(2f)`` windows and ``A_kk = 1 - 1/tau = 1 - 2f/f_win``.
    """
    return 1.0 - 2.0 * f / f_win


def frequency_from_akk(a_kk: float, f_win: float) -> float:
    """Oscillation frequency (Hz) implied by a fitted ``A_kk`` (inverse of above).

    ``f = f_win (1 - A_kk) / 2``. Returns ``inf`` for the degenerate
    ``A_kk >= 1`` (a never-leaving state).
    """
    if a_kk >= 1.0:
        return np.inf
    return f_win * (1.0 - a_kk) / 2.0


def dwell_windows_from_frequency(f: float, f_win: float) -> float:
    """Mean half-cycle dwell in windows for an oscillation at ``f`` Hz."""
    return f_win / (2.0 * f)


def label_state_frequencies(res: dict, band: tuple[float, float] = (0.5, 2.0)
                            ) -> dict:
    """Map every fitted state's ``A_kk`` to an implied frequency and flag in-band.

    Uses the diagonal of the transition matrix from :func:`fit_hmm`. The band
    edges default to a nominal fidgety window; take the exact clinical edges from
    Einspieler and Prechtl for a thesis figure (§4.3).

    Returns:
        Dict with per-state ``implied_hz``, a boolean ``in_band`` mask, the
        band, and the list of in-band state indices.
    """
    f_win = res["f_win"]
    a = np.diag(res["transition"])
    implied = np.array([frequency_from_akk(ak, f_win) for ak in a])
    lo, hi = band
    in_band = (implied >= lo) & (implied <= hi)
    return {"implied_hz": implied, "in_band": in_band, "band": band,
            "in_band_states": np.where(in_band)[0].tolist(),
            "akk": a}


def band_power_ratio(signal: np.ndarray, fs: float,
                     band: tuple[float, float] = (0.5, 2.0),
                     nperseg: int | None = None) -> float:
    """Fidgety-band ratio: band power / total power of a multichannel signal.

    ``FBR = (integral of P(f) over the band) / (integral over [0, fs/2])``,
    with ``P`` the Welch PSD averaged over channels (§4.4). ``signal`` is
    ``(N,)`` or ``(N, C)``.
    """
    from scipy.signal import welch

    x = np.asarray(signal, float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 8:
        return np.nan
    if nperseg is None:
        nperseg = min(len(x), 256)
    f, P = welch(x, fs=fs, axis=0, nperseg=nperseg)
    P = P.mean(axis=1)
    lo, hi = band
    band_mask = (f >= lo) & (f <= hi)
    total = np.trapezoid(P, f) if hasattr(np, "trapezoid") else np.trapz(P, f)
    if total <= 0:
        return np.nan
    num = (np.trapezoid(P[band_mask], f[band_mask]) if hasattr(np, "trapezoid")
           else np.trapz(P[band_mask], f[band_mask]))
    return float(num / total)


def raw_velocity_fbr(video: np.ndarray, fs: float,
                     band: tuple[float, float] = (0.5, 2.0)) -> float:
    """Headline fidgety-band ratio, on **raw keypoint velocities** (§4.4).

    Velocities are frame differences of the pose, continuous across the whole
    recording with no encoder seams — so this figure can never be corrupted by a
    stitching artifact (Guardrail 4.1). Each joint-coordinate velocity series
    contributes a channel; the FBR is their pooled band fraction.

    Args:
        video: one recording ``(F, J, D)``.
        fs: native frame rate (Hz) — 25 for RVI-38, *not* the window rate.
        band: fidgety band edges (Hz).
    """
    v = np.diff(np.asarray(video, float), axis=0)      # (F-1, J, D)
    v = v.reshape(v.shape[0], -1)                        # (F-1, J*D) channels
    return band_power_ratio(v, fs, band)


def latent_band_power(trajectory: np.ndarray, f_win: float,
                      band: tuple[float, float] = (0.5, 2.0)) -> float:
    """Corroborating FBR on a seam-handled latent window trajectory (§4.4).

    Read alongside :func:`raw_velocity_fbr`, never as the headline (Guardrail
    4.1). ``trajectory`` is a per-video ``(M_v, d_z)`` block from the stitcher;
    sampling rate is the window rate ``f_win``.
    """
    return band_power_ratio(trajectory, f_win, band)


# ---------------------------------------------------------------------------
# 5. Interpretability + phenotype clustering  (§5.3-5.5; Guardrail 5.1)
# ---------------------------------------------------------------------------
def decode_state_appearance(adapter, res: dict, state: int) -> np.ndarray:
    """Decode a state's mean into its rendered pose sequence (§5.3.1).

    Builds a constant-state latent block ``Z_k = [mu_k, ..., mu_k]`` of shape
    ``(n_win, d_z)``, flattens it in the model's own window order, and pushes it
    through the frozen decoder. A state that decodes to nothing recognisable is
    a sign ``K`` is too high.

    For a **difference-stream** model (``stream="delta"``) ``mu_k`` is a change,
    not a pose, so the constant-block decode is not a literal appearance — render
    the integrated trajectory instead. This helper handles the pose stream.

    Returns:
        Pose sequence ``(T, J, D)``.
    """
    n_win = adapter.n_windows()
    mu_k = np.asarray(res["means"][state], np.float32)         # (d_z,)
    block = np.tile(mu_k, (1, n_win, 1))                        # (1, n_win, d_z)
    z_flat = adapter.flatten_windows(block)                    # (1, d_z*n_win)
    return adapter.decode(z_flat)[0]                           # (T, J, D)


def phenotype_features(res: dict, fbr_per_video: np.ndarray | None = None
                       ) -> tuple[np.ndarray, list[str]]:
    """Assemble one feature vector per video for phenotype clustering (§5.4).

    Concatenates the ``K``-dim state-occupancy histogram, the ``K``-dim
    mean-dwell vector (windows; absent states filled with 0 dwell), and,
    optionally, the scalar raw-velocity FBR. Rows align with the stitched
    ``video_id`` order.

    Args:
        res: :func:`fit_hmm` output.
        fbr_per_video: optional ``(n_videos,)`` raw-velocity FBR per retained
            video (the headline frequency feature).

    Returns:
        (features ``(n_videos, 2K[+1])``, column names).
    """
    occ = res["per_video_occupancy"]                           # (n_vid, K)
    dwell = np.nan_to_num(res["per_video_dwell_windows"], nan=0.0)
    k = occ.shape[1]
    cols = [f"occ_s{s}" for s in range(k)] + [f"dwell_s{s}" for s in range(k)]
    feats = [occ, dwell]
    if fbr_per_video is not None:
        feats.append(np.asarray(fbr_per_video, float)[:, None])
        cols.append("raw_velocity_fbr")
    return np.concatenate(feats, axis=1), cols


def cluster_phenotypes(features: np.ndarray, *, k_range=range(2, 7),
                       standardize: bool = True, seed: int = 0,
                       labels: np.ndarray | None = None) -> dict:
    """Cluster the per-video phenotype vectors, with small-``n`` honesty (§5.4).

    Runs TwoNN intrinsic dimension first: if it exceeds the apparent cluster
    count, the structure is better read as continuous than partitioned. Then
    fits a Gaussian mixture, selecting the component count by BIC and reporting
    the silhouette internally. If ground-truth ``labels`` are given they are used
    **only** post hoc (ARI / AMI), never in the fit (Guardrail 5.1).

    With ``n`` on the order of tens this is exploratory; report effect sizes and
    stability, not a headline accuracy.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from .posterior_geometry import intrinsic_dimension_twonn

    X = np.asarray(features, float)
    n = len(X)
    if standardize:
        mu, sd = X.mean(0), X.std(0) + 1e-9
        X = (X - mu) / sd

    twonn = intrinsic_dimension_twonn(X)

    ks = [k for k in k_range if 2 <= k <= max(2, n // 2)]
    fits = {}
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             reg_covar=1e-4, random_state=seed, n_init=5)
        gm.fit(X)
        lab = gm.predict(X)
        sil = (silhouette_score(X, lab) if len(np.unique(lab)) > 1 else np.nan)
        fits[k] = {"bic": gm.bic(X), "silhouette": sil, "labels": lab,
                   "model": gm}
    if not fits:
        raise ValueError(f"too few samples ({n}) to cluster.")
    k_best = min(fits, key=lambda k: fits[k]["bic"])
    best = fits[k_best]

    out = {"k": k_best, "labels": best["labels"],
           "silhouette": best["silhouette"],
           "bic": {k: v["bic"] for k, v in fits.items()},
           "silhouette_by_k": {k: v["silhouette"] for k, v in fits.items()},
           "intrinsic_dimension": twonn.get("d_hat", twonn.get("dimension")),
           "n": n, "exploratory": True}
    if labels is not None:
        from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
        labels = np.asarray(labels)
        out["ari"] = float(adjusted_rand_score(labels, best["labels"]))
        out["ami"] = float(adjusted_mutual_info_score(labels, best["labels"]))
    return out
