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
   (a comb at ``f_frame/clip_len``). Encoding at 50% overlap and keeping each
   clip's central windows tiles the recording with no seam.
2. :func:`seam_diagnostic` — PSD check that the ``f_frame/clip_len`` comb is
   gone before per-video ``lengths`` are trusted (Guardrail 3.1).
3. :func:`fit_hmm` — full-covariance Gaussian HMM with a ridge floor, shrinkage
   triggers, k-means restarts, subject-disjoint ``K`` selection, and a Viterbi
   decode (§2, §5.2; Guardrails 2.2, 2.4, 2.5).
4. :func:`state_dwell_times` — the timescale of each state: how long the chain
   holds it, empirically (Viterbi runs) and as implied by ``A_kk``.
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
                   stream: str = "pose", masks=None, verbose: bool = False
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch every video and stack into the HMM's ``(Z, lengths, video_id)``.

    The ``lengths`` array is what makes the HMM sum per-video log-likelihoods
    rather than concatenate windows across recording boundaries (§2.2). Videos
    too short for one clip are skipped.

    Args:
        verbose: print one line per recording as it is encoded — this pass runs
            the VAE over every clip of every video, so on a long cohort it is
            worth seeing move.

    Returns:
        Z: ``(sum M_v, d_z)`` stacked trajectories.
        lengths: ``(n_kept_videos,)`` window count per retained video.
        video_id: ``(sum M_v,)`` retained-video index per window.
    """
    import time
    parts, lengths, ids = [], [], []
    kept_idx = 0
    t0 = time.time()
    n = len(videos)
    for v, video in enumerate(videos):
        m = None if masks is None else masks[v]
        traj = encode_window_sequence(adapter, video, clip_len=clip_len,
                                      stride=stride, keep=keep, stream=stream,
                                      mask=m)
        if len(traj) == 0:
            if verbose:
                print(f"[stitch]  {v + 1:>3}/{n}  skipped "
                      f"({len(video)} frames < clip_len={clip_len})", flush=True)
            continue
        parts.append(traj)
        lengths.append(len(traj))
        ids.append(np.full(len(traj), kept_idx, dtype=np.int64))
        kept_idx += 1
        if verbose:
            el = time.time() - t0
            print(f"[stitch]  {v + 1:>3}/{n}  {len(video):>6} frames -> "
                  f"{len(traj):>5} windows | elapsed {_fmt_dur(el)} "
                  f"eta {_fmt_dur(el / (v + 1) * (n - v - 1))}", flush=True)
    if not parts:
        raise ValueError("No windows stitched. Are all videos shorter than "
                         "clip_len?")
    return (np.concatenate(parts), np.asarray(lengths, np.int64),
            np.concatenate(ids))


# ---------------------------------------------------------------------------
# 2. Seam diagnostic  (§3; Guardrail 3.1)
# ---------------------------------------------------------------------------
def seam_diagnostic(Z: np.ndarray, lengths: np.ndarray, *, clip_len: int,
                    n_win: int, f_win: float, stride: int | None = None,
                    tol: float = 3.0) -> dict:
    """Check the stitched trajectory for a spectral line at the clip boundary.

    Each clip contributes ``stride/l`` windows to the stitched trajectory, so
    clip boundaries recur with that period and would show as a line at
    ``f_win * l / stride`` and its harmonics. For naive non-overlapping tiling
    (``stride == clip_len``) that reduces to the familiar ``f_win/n_win``, i.e.
    the ``f_frame/clip_len`` comb. **The default overlap-crop stitcher runs at
    ``stride = clip_len/2``, where boundaries recur every ``n_win/2`` windows —
    one octave above ``f_win/n_win``**, so probing ``f_win/n_win`` looks in the
    wrong bin. Pass the ``stride`` you stitched with.

    The bin is compared against a **local** baseline — the neighbouring bins,
    excluding the harmonics themselves — not the median of the whole spectrum.
    A latent motion trajectory is strongly autocorrelated, so its PSD is red;
    against a global median any low-frequency bin scores an order of magnitude
    high whether or not a seam exists, which makes such a test fire on every
    real recording and separate nothing.

    This is the spectral half of :func:`seam_gate`, which is the gate to trust:
    it pairs this with a time-domain boundary-jump test that detects a seam
    whose energy is spread rather than concentrated in a line.

    Args:
        Z, lengths: output of :func:`stitch_dataset`.
        clip_len, n_win: VAE geometry.
        f_win: window sampling rate (Hz) = ``f_frame / l``.
        stride: the stride used when stitching. Defaults to ``clip_len``
            (naive tiling), which is the only case where the boundary period is
            ``n_win``.
        tol: pass threshold — flag when the boundary bin exceeds ``tol`` x its
            local baseline.

    Returns:
        Dict with the boundary frequency, the per-video ratio, the max ratio,
        and a boolean ``passed`` (max ratio below ``tol``).
    """
    from scipy.signal import welch

    l = clip_len // n_win
    step_win = max(1, (stride // l) if stride else n_win)   # windows per clip
    f_seam = f_win / step_win
    harm = f_seam * np.arange(1, 6)
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
        j = int(np.argmin(np.abs(f - f_seam)))
        # local baseline: nearby bins, minus the harmonic bins themselves
        hbins = {int(np.argmin(np.abs(f - h))) for h in harm if h < f_win / 2}
        nb = [i for i in range(max(1, j - 3), min(len(f), j + 4))
              if i != j and i not in hbins]
        if not nb:
            continue
        ratios.append(P[j] / (np.median(P[nb]) + 1e-12))
    ratios = np.asarray(ratios) if ratios else np.array([np.nan])
    max_ratio = float(np.nanmax(ratios))
    return {"f_seam": f_seam, "per_video_ratio": ratios,
            "max_ratio": max_ratio, "passed": bool(max_ratio < tol),
            "tol": tol, "boundary_period_windows": step_win}

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
    """One GaussianHMM fit.

    Returns ``(model, train_loglik, info)``, or ``(None, -inf, info)`` on a
    degenerate init. ``info`` is the per-restart telemetry the progress log
    prints: EM iterations actually run, whether EM met its tolerance, whether it
    stopped only because it ran out of iterations, wall seconds, and the
    exception type if the fit failed.

    Convergence is judged here rather than read off ``monitor_.converged``:
    hmmlearn defines that property as ``iter == n_iter or delta < tol``, so it
    reports ``True`` precisely when the iteration cap is hit — a fit stopped
    dead at ``n_iter`` with a log-likelihood still climbing by hundreds is
    called "converged". The cap is the thing worth surfacing, so it is measured
    directly.
    """
    from hmmlearn.hmm import GaussianHMM
    import logging, warnings, time
    hmm_logger = logging.getLogger("hmmlearn")
    prev = hmm_logger.level
    t0 = time.time()
    info = {"n_iter": 0, "converged": False, "hit_cap": False,
            "last_delta": np.nan, "seconds": 0.0, "error": None}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="hmmlearn")
            hmm_logger.setLevel(logging.ERROR)
            model = GaussianHMM(n_components=k, covariance_type=covariance_type,
                                min_covar=min_covar, n_iter=n_iter,
                                random_state=seed, init_params="stmc")
            model.fit(Z, lengths)
            ll = model.score(Z, lengths)
        mon = getattr(model, "monitor_", None)
        ran = int(getattr(mon, "iter", 0) or 0)
        hist = list(getattr(mon, "history", []) or [])
        tol = float(getattr(mon, "tol", 0.0) or 0.0)
        delta = (hist[-1] - hist[-2]) if len(hist) >= 2 else np.nan
        tol_met = bool(len(hist) >= 2 and delta < tol)
        info["n_iter"] = ran
        info["last_delta"] = float(delta)
        info["converged"] = tol_met
        # Capped only when it used every iteration *and* still had not met tol;
        # a fit that converges exactly on its last allowed step is not capped.
        info["hit_cap"] = bool(ran >= n_iter and not tol_met)
    except Exception as e:  # noqa: BLE001 — degenerate init; caller retries/skips
        info["error"] = type(e).__name__
        info["seconds"] = time.time() - t0
        return None, -np.inf, info
    finally:
        hmm_logger.setLevel(prev)
    info["seconds"] = time.time() - t0
    if not np.isfinite(ll):
        info["error"] = "non-finite loglik"
        return None, -np.inf, info
    return model, ll, info


def _best_of_restarts(Z, lengths, k, covariance_type, min_covar, n_iter,
                      n_restarts, seed):
    """Best-training-loglik model over ``n_restarts`` k-means-seeded inits.

    Returns ``(model, train_loglik, restarts)`` with ``restarts`` the list of
    per-restart ``info`` dicts from :func:`_fit_once`, in restart order.
    """
    best, best_ll, restarts = None, -np.inf, []
    for r in range(n_restarts):
        model, ll, info = _fit_once(Z, lengths, k, covariance_type, min_covar,
                                    n_iter, seed + 1000 * r)
        info["loglik"] = float(ll)
        restarts.append(info)
        if ll > best_ll:
            best, best_ll = model, ll
    return best, best_ll, restarts


def _fmt_restarts(restarts) -> str:
    """``"41,80*,fail"`` — EM iterations per restart, ``*`` = stopped at the cap."""
    out = []
    for r in restarts:
        if r.get("error"):
            out.append("fail")
        else:
            out.append(f"{r['n_iter']}{'*' if r.get('hit_cap') else ''}")
    return ",".join(out)


def _fmt_dur(seconds: float) -> str:
    """``"48s"`` / ``"6.4m"`` / ``"1.2h"``."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _selection_job(Z, lengths, k, fold, val_videos, covariance_type, min_covar,
                   n_iter, n_restarts, seed):
    """One unit of K-selection work, sized so progress is reported often.

    ``fold is None`` is the full-data fit that yields BIC; otherwise it is one
    held-out fold, scored as validation log-likelihood **per window** so folds of
    different size are comparable. Runs in a joblib worker, so it returns
    telemetry rather than printing — worker stdout does not reach a notebook.
    """
    import time
    t0 = time.time()
    out = {"k": k, "fold": fold, "score": None, "bic": None, "train_ll": None,
           "restarts": [], "n_val": 0, "error": None}
    if fold is None:
        model, ll, restarts = _best_of_restarts(Z, lengths, k, covariance_type,
                                                min_covar, n_iter, n_restarts,
                                                seed)
        out["restarts"] = restarts
        out["train_ll"] = float(ll)
        out["bic"] = (float(-2 * ll + hmm_n_params(k, Z.shape[1],
                                                   covariance_type)
                            * np.log(len(Z)))
                      if model is not None else np.inf)
        if model is None:
            out["error"] = "no restart converged"
    else:
        n_videos = len(lengths)
        val_set = set(np.asarray(val_videos).tolist())
        train_videos = np.array([v for v in range(n_videos) if v not in val_set])
        if len(train_videos) < 1 or len(val_videos) < 1:
            out["error"] = "empty split"
        else:
            Ztr, ltr = _subset(Z, lengths, train_videos)
            Zva, lva = _subset(Z, lengths, np.asarray(val_videos))
            model, _, restarts = _best_of_restarts(Ztr, ltr, k, covariance_type,
                                                   min_covar, n_iter,
                                                   n_restarts, seed)
            out["restarts"] = restarts
            out["n_val"] = int(len(Zva))
            if model is None:
                out["error"] = "no restart converged"
            else:
                try:
                    ll = model.score(Zva, lva)
                except Exception as e:  # noqa: BLE001
                    out["error"] = f"score failed: {type(e).__name__}"
                else:
                    if np.isfinite(ll):
                        out["score"] = float(ll / len(Zva))
                    else:
                        out["error"] = "non-finite held-out loglik"
    out["seconds"] = time.time() - t0
    return out


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


def kfold_subject_splits(n_videos: int, n_splits: int,
                         seed: int = 0) -> list[np.ndarray]:
    """Partition subject indices ``0..n_videos-1`` into ``n_splits`` folds.

    One permutation seeded by ``seed``, sliced by :func:`numpy.array_split`, so
    every subject falls in exactly one validation fold and fold sizes differ by
    at most one. Genuine k-fold cross-validation, not repeated random holdout.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
    if n_splits > n_videos:
        raise ValueError(
            f"n_splits={n_splits} exceeds n_videos={n_videos}; "
            f"cannot form disjoint folds.")
    perm = np.random.default_rng(seed).permutation(n_videos)
    return [np.asarray(fold) for fold in np.array_split(perm, n_splits)]


def fit_hmm(Z: np.ndarray, lengths: np.ndarray, *, k_range=range(2, 11),
            f_win: float = 7.5, covariance_type: str = "full",
            min_covar: float = 1e-3, n_restarts: int = 5, n_iter: int = 200,
            selection: str = "cv", n_splits: int = 5,
            seed: int = 0, cond_ceiling: float = 1e8,
            occupancy_floor_factor: float = 10.0, verbose: bool = False,
            n_jobs: int = 1) -> dict:
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
        selection: ``"cv"`` (``n_splits``-fold partition over subjects, each
            validated once — the default), ``"loo"`` (leave-one-video-out), or
            ``"bic"``.
        cond_ceiling, occupancy_floor_factor: Guardrail 2.4 triggers.
        n_jobs: processes for the K sweep. Work is split one job per
            ``(K, fold)``, so every core stays busy and progress is reported
            many times per ``K``.
        verbose: stream a progress log — one line per completed job with its
            held-out log-likelihood per window, the EM iteration count of each
            restart (``*`` marks a restart that hit the ``n_iter`` cap), wall
            time, and a running ETA; then a summary line per ``K``, the
            selection, and the final fit's escalation ladder. Lines are printed
            from the **calling** process, so they appear in a notebook cell even
            when the fits run in joblib workers.

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
            return kfold_subject_splits(n_videos, n_splits, seed)
        return []                                    # bic: no held-out

    splits = make_splits()

    # ---- select K -----------------------------------------------------------
    # One job per (K, fold) rather than per K: the unit of parallel work is small
    # enough to keep every core busy and, more importantly, to report progress
    # many times per K instead of once. Job order is K-major so early lines cover
    # a whole K rather than one fold of every K.
    import time as _time
    jobs = []
    for k in ks:
        jobs.append((k, None, None))                  # full-data fit -> BIC
        if selection != "bic":
            for i, val_videos in enumerate(splits):
                jobs.append((k, i, val_videos))
    n_total = len(jobs)
    t_start = _time.time()

    if verbose:
        print(f"[fit_hmm] {M} windows, d={d}, {n_videos} videos | "
              f"K in {list(ks)} | selection={selection}"
              + (f" ({len(splits)} folds)" if splits else "")
              + f" | n_jobs={n_jobs}", flush=True)
        print(f"[fit_hmm] {n_total} jobs x {n_restarts} restarts x <={n_iter} EM "
              f"iters = up to {n_total * n_restarts} fits. "
              f"'iters' below lists EM iterations per restart; "
              f"* = stopped at the {n_iter}-iteration cap without meeting "
              f"tolerance (raise n_iter).", flush=True)

    fold_scores = {k: [] for k in ks}
    bics = {k: np.inf for k in ks}
    pending = {k: sum(1 for j in jobs if j[0] == k) for k in ks}
    done = 0
    n_restarts_run = 0
    n_capped = 0

    def _report(out):
        """Print one completed job from the **parent** process, with an ETA."""
        nonlocal done, n_restarts_run, n_capped
        done += 1
        n_restarts_run += len(out["restarts"])
        n_capped += sum(1 for r in out["restarts"] if r.get("hit_cap"))
        k, fold = out["k"], out["fold"]
        if fold is None:
            bics[k] = out["bic"]
            what = "full"
            val = (f"bic={out['bic']:.0f}" if np.isfinite(out["bic"])
                   else "bic=inf")
        else:
            what = f"fold {fold + 1}/{len(splits)}"
            if out["score"] is not None:
                fold_scores[k].append(out["score"])
                val = f"ll/win={out['score']:+.4f}"
            else:
                val = f"FAILED ({out['error']})"
        pending[k] -= 1
        if verbose:
            elapsed = _time.time() - t_start
            eta = elapsed / done * (n_total - done)
            print(f"[fit_hmm] {done:>3}/{n_total} "
                  f"K={k:<2} {what:<12} {val:<22} "
                  f"iters {_fmt_restarts(out['restarts']):<14} "
                  f"{out['seconds']:5.1f}s | elapsed {_fmt_dur(elapsed)} "
                  f"eta {_fmt_dur(eta)}", flush=True)
            if pending[k] == 0:                        # this K is fully scored
                sc = (float(np.mean(fold_scores[k])) if fold_scores[k]
                      else -np.inf)
                tag = (f"cv ll/win = {sc:+.4f} over {len(fold_scores[k])} folds"
                       if selection != "bic" else "")
                print(f"[fit_hmm]   -> K={k} complete: {tag}"
                      f"{'  ' if tag else ''}bic={bics[k]:.0f}", flush=True)

    # Each hmmlearn fit is single-threaded, so process parallelism is the real
    # win. Results are consumed as they land (`generator_unordered`) so the
    # progress log comes from this process — a worker's stdout never reaches a
    # notebook cell, which is exactly where this is run.
    _args = (covariance_type, min_covar, n_iter, n_restarts, seed)
    if n_jobs == 1:
        for k, fold, val_videos in jobs:
            _report(_selection_job(Z, lengths, k, fold, val_videos, *_args))
    else:
        from joblib import Parallel, delayed

        def _tasks():
            return (delayed(_selection_job)(Z, lengths, k, fold, val_videos,
                                            *_args)
                    for k, fold, val_videos in jobs)

        try:                              # joblib >= 1.4 streams completions
            runner = Parallel(n_jobs=n_jobs, prefer="processes",
                              return_as="generator_unordered")
        except (TypeError, ValueError):   # older joblib: collect, then report
            runner = None
        if runner is not None:
            for out in runner(_tasks()):
                _report(out)
        else:
            if verbose:
                print("[fit_hmm] joblib<1.4: results arrive only at the end of "
                      "the sweep, so no per-job progress below.", flush=True)
            for out in Parallel(n_jobs=n_jobs, prefer="processes")(_tasks()):
                _report(out)

    if selection == "bic":
        scores = {k: -bics[k] for k in ks}
    else:
        scores = {k: (float(np.mean(fold_scores[k])) if fold_scores[k]
                      else -np.inf) for k in ks}
    if not scores or all(v == -np.inf for v in scores.values()):
        raise ValueError("No HMM converged for any candidate K.")
    k_best = max(scores, key=scores.get)
    if verbose:
        ranked = sorted(scores, key=scores.get, reverse=True)
        runner_up = (f", runner-up K={ranked[1]} ({scores[ranked[1]]:+.4f})"
                     if len(ranked) > 1 else "")
        print(f"[fit_hmm] selection took {_fmt_dur(_time.time() - t_start)}; "
              f"K*={k_best} ({scores[k_best]:+.4f}){runner_up}", flush=True)
        if n_capped:
            frac = n_capped / max(n_restarts_run, 1)
            note = ("— K selection is comparing under-fitted models; raise "
                    "n_iter" if frac > 0.25 else
                    "— tolerable, but raise n_iter if K* sits near a capped K")
            print(f"[fit_hmm] !! {n_capped}/{n_restarts_run} restarts "
                  f"({frac:.0%}) stopped at the n_iter={n_iter} cap {note}.",
                  flush=True)

    # ---- final fit at K*, with Guardrail 2.4 escalation --------------------
    reg = {"covariance_type": covariance_type, "min_covar": min_covar,
           "escalations": []}
    cov_type = covariance_type
    mc = min_covar
    for attempt in range(3):
        if verbose:
            print(f"[fit_hmm] final fit at K={k_best} "
                  f"(attempt {attempt + 1}/3, cov={cov_type}, "
                  f"min_covar={mc:g}) ...", flush=True)
        _t_final = _time.time()
        model, ll_final, restarts = _best_of_restarts(
            Z, lengths, k_best, cov_type, mc, n_iter, n_restarts, seed)
        if verbose:
            print(f"[fit_hmm]   iters {_fmt_restarts(restarts)}  "
                  f"train ll={ll_final:.1f}  "
                  f"({_time.time() - _t_final:.1f}s)", flush=True)
        if model is None:
            mc *= 10
            reg["escalations"].append(f"no-converge -> min_covar={mc:g}")
            if verbose:
                print(f"[fit_hmm]   no restart converged -> "
                      f"min_covar={mc:g}", flush=True)
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
            msg = (f"ill-conditioned (cond={conds.max():.1e}) -> "
                   f"min_covar={mc:g}")
            reg["escalations"].append(msg)
            if verbose: print(f"[fit_hmm]   escalate: {msg}", flush=True)
            continue
        if occ_triggered and cov_type == "full":     # rung 1: tied covariance
            cov_type, mc = "tied", min_covar
            msg = (f"thin state (min_occ={occ_counts.min():.0f}<{floor:.0f}) -> "
                   f"covariance_type='tied' (fallback ladder §2.6)")
            reg["escalations"].append(msg)
            if verbose: print(f"[fit_hmm]   escalate: {msg}", flush=True)
            continue
        if cond_triggered or occ_triggered:
            msg = ("trigger persists after ladder; accepting fit (see §2.6 rungs "
                   "semi-tied / factor-analysed for further reduction)")
            reg["escalations"].append(msg)
            if verbose: print(f"[fit_hmm]   {msg}", flush=True)
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
# 4. State timescales: mean dwell time
# ---------------------------------------------------------------------------
def state_dwell_times(res: dict) -> dict:
    """Mean dwell time of every fitted state, measured and model-implied.

    Dwell time is the only timescale a first-order HMM actually asserts about a
    state. Under the fitted chain the holding time of state ``k`` is geometric
    with parameter ``1 - A_kk``, so its mean is ``1/(1 - A_kk)`` windows — a
    property of the transition matrix alone, carrying no assumption that the
    state recurs, alternates, or oscillates. This is reported next to the
    **measured** dwell, the mean run length of the Viterbi path (runs are counted
    within a recording and never stitched across a boundary; see
    :func:`_viterbi_summary`).

    The two agree when the geometric holding-time model fits, and diverge when
    the decoded runs are more or less persistent than a first-order chain
    predicts — a gap worth reading rather than hiding, so both are returned.

    Args:
        res: :func:`fit_hmm` output.

    Returns:
        Dict with ``akk`` (the transition diagonal), the implied dwell in
        ``implied_windows`` / ``implied_seconds``, and the measured Viterbi
        dwell in ``dwell_windows`` / ``dwell_seconds``. A never-leaving state
        (``A_kk >= 1``) gets an infinite implied dwell.
    """
    f_win = res["f_win"]
    a = np.asarray(np.diag(res["transition"]), float)
    implied_win = np.where(a >= 1.0, np.inf,
                           1.0 / np.clip(1.0 - a, 1e-12, None))
    return {"akk": a,
            "implied_windows": implied_win,
            "implied_seconds": implied_win / f_win,
            "dwell_windows": np.asarray(res["dwell_windows"], float),
            "dwell_seconds": np.asarray(res["dwell_seconds"], float),
            "f_win": f_win}


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


def phenotype_features(res: dict, extra: np.ndarray | None = None,
                       extra_names: list[str] | None = None
                       ) -> tuple[np.ndarray, list[str]]:
    """Assemble one feature vector per video for phenotype clustering (§5.4).

    Concatenates the ``K``-dim state-occupancy histogram and the ``K``-dim
    mean-dwell vector (windows; absent states filled with 0 dwell). Rows align
    with the stitched ``video_id`` order.

    Args:
        res: :func:`fit_hmm` output.
        extra: optional ``(n_videos,)`` or ``(n_videos, C)`` block of extra
            per-video covariates to append.
        extra_names: column names for ``extra``; defaults to ``extra_0..``.

    Returns:
        (features ``(n_videos, 2K[+C])``, column names).
    """
    occ = res["per_video_occupancy"]                           # (n_vid, K)
    dwell = np.nan_to_num(res["per_video_dwell_windows"], nan=0.0)
    k = occ.shape[1]
    cols = [f"occ_s{s}" for s in range(k)] + [f"dwell_s{s}" for s in range(k)]
    feats = [occ, dwell]
    if extra is not None:
        E = np.asarray(extra, float)
        if E.ndim == 1:
            E = E[:, None]
        feats.append(E)
        cols += (list(extra_names) if extra_names is not None
                 else [f"extra_{c}" for c in range(E.shape[1])])
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


# ---------------------------------------------------------------------------
# 6. State movement dynamics — high-velocity frames per body group per state
#    (the "what does each state mean, kinematically" figure)
# ---------------------------------------------------------------------------
def body_groups(limbs: dict[str, list[int]]) -> dict[str, list[int]]:
    """Collapse the skeleton's limb map into head / arms / legs joint groups.

    Merges left+right arms and left+right legs; head is whatever the skeleton
    labels ``head``. Works for BODY-15 / COCO-18 limb maps.
    """
    def _merge(*names):
        out = []
        for n in names:
            out += list(limbs.get(n, []))
        return sorted(set(out))
    return {"head": _merge("head"),
            "arms": _merge("right_arm", "left_arm"),
            "legs": _merge("right_leg", "left_leg")}


def lateral_groups(limbs: dict[str, list[int]]) -> dict[str, list[int]]:
    """Four lateralized limb groups: left_arm, right_arm, left_leg, right_leg.

    For reading left/right movement **asymmetry** per state: typical spontaneous
    movement is roughly symmetric, so a consistent left-right gap is worth
    flagging.
    """
    out = {}
    for name in ("left_arm", "right_arm", "left_leg", "right_leg"):
        if limbs.get(name):
            out[name] = sorted(set(limbs[name]))
    return out


def side_groups(limbs: dict[str, list[int]], include_head: bool = False
                ) -> dict[str, list[int]]:
    """Whole-side groups: all left limbs vs all right limbs (optional head)."""
    L = sorted(set(list(limbs.get("left_arm", [])) + list(limbs.get("left_leg", []))))
    R = sorted(set(list(limbs.get("right_arm", [])) + list(limbs.get("right_leg", []))))
    out = {"left": L, "right": R}
    if include_head and limbs.get("head"):
        out["head"] = sorted(set(limbs["head"]))
    return out


def _kept_window_frame_spans(F: int, clip_len: int, stride: int, n_win: int,
                             keep: tuple[int, int] | None) -> list[tuple[int, int]]:
    """Frame span (start, end) of every kept window, in the trajectory's order.

    Mirrors :func:`encode_window_sequence`'s crop/keep logic exactly (clip-major,
    then the kept window range within each clip), so the spans line up 1:1 with
    the stitched trajectory and hence with ``res['states']``.
    """
    l = clip_len // n_win
    step_win = stride // l
    if keep is None:
        lo = (n_win - step_win) // 2
        keep = (lo, lo + step_win)
    lo, hi = keep
    spans = []
    for s in _clip_starts(F, clip_len, stride):
        for w in range(lo, hi):
            spans.append((s + w * l, s + (w + 1) * l))
    return spans


def state_movement_dynamics(videos: list[np.ndarray], res: dict, lengths: np.ndarray,
                            *, groups: dict[str, list[int]], clip_len: int,
                            stride: int, n_win: int, keep: tuple[int, int] | None = None,
                            top_frac: float = 0.10) -> dict:
    """Per-state % of high-velocity frames for each body-point group, per video.

    Reproduces the "state movement dynamics" figure (kinematic meaning of each
    HMM state). For each video and body point, the frames in its top
    ``top_frac`` by speed are the "high-velocity" frames. Each Viterbi window
    state is mapped to its ``l`` frames; then for each ``(state, group)`` the
    metric is **the percentage of that state's frames that are high-velocity,
    averaged over the group's body points** — one value per video.

    Requires the **pose** stream (window state ~ pose window). ``videos`` must be
    the same list passed to :func:`stitch_dataset`; videos too short for one clip
    are skipped identically here so the blocks line up with ``lengths``.

    Returns:
        Dict ``{state: {group: np.ndarray of per-video percentages}}`` (NaN for a
        video that never visits that state), plus ``"k"`` and ``"groups"``.
    """
    states = np.asarray(res["states"])
    K = res["k"]
    kept = [v for v in videos
            if len(_clip_starts(len(v), clip_len, stride)) > 0]
    if len(kept) != len(lengths):
        raise ValueError(
            f"{len(kept)} stitchable videos but {len(lengths)} trajectory "
            f"blocks — pass the same videos / clip_len / stride as stitch_dataset.")

    out = {s: {g: [] for g in groups} for s in range(K)}
    offset = 0
    for video, L in zip(kept, lengths):
        st = states[offset:offset + L]
        offset += L
        spans = _kept_window_frame_spans(len(video), clip_len, stride, n_win, keep)
        # A pose stream has one state per window (len(spans) == L); the delta
        # stream has one fewer (Δz between consecutive windows, L == n_win-1), so
        # map each state to the first L window spans (the "from" window).
        if len(spans) < L:
            raise ValueError(f"fewer frame spans ({len(spans)}) than states ({L}).")
        spans = spans[:L]
        # per-frame, per-joint speed; high-velocity = top `top_frac` per joint.
        speed = np.linalg.norm(np.diff(np.asarray(video, float), axis=0), axis=-1)
        speed = np.vstack([speed, speed[-1:]])           # pad to F frames
        thr = np.quantile(speed, 1.0 - top_frac, axis=0)  # (J,)
        hv = speed >= thr[None, :]                        # (F, J) bool
        F = len(video)
        for s in range(K):
            frames_s = []
            for (a, b), ws in zip(spans, st):
                if ws == s:
                    frames_s.extend(range(a, min(b, F)))
            for g, joints in groups.items():
                if frames_s:
                    pct = 100.0 * hv[np.ix_(np.asarray(frames_s), joints)].mean()
                else:
                    pct = np.nan
                out[s][g].append(pct)
    for s in range(K):
        for g in groups:
            out[s][g] = np.asarray(out[s][g], float)
    out["k"] = K
    out["groups"] = list(groups)
    return out
