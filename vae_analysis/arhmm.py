"""Autoregressive HMM over the temporal-latent trajectory, via the `ssm` package.

Each state is a linear **autoregressive** Gaussian process

    x_t = A_k x_{t-1} + ... + A_k^{(p)} x_{t-p} + b_k + e,   e ~ N(0, Sigma_k)

so a state is a *dynamical regime* — how the latent evolves — not a static
cluster of poses. This is the MoSeq / Datta-lab behavioural AR-HMM; for a motion
latent it captures movement dynamics that a static-Gaussian HMM (means only)
cannot. EM is done by ``ssm`` (Linderman lab); we never implement it here — this
module only splits the stitched trajectory into per-video sequences, calls ssm,
and wraps the result into the same ``res`` dict :mod:`vae_analysis.hmm_report`
already plots (states / transition / occupancy / f_win), plus the AR parameters.

Install (Colab / fresh env)::

    pip install numpy cython
    pip install --no-build-isolation "ssm @ git+https://github.com/lindermanlab/ssm.git"

Usage::

    from vae_analysis.arhmm import fit_arhmm
    res = fit_arhmm(Z, lengths, k_range=range(4, 10), f_win=F_WIN, lags=1)
    # res drops into the hmm_report figures + state_dwell_times unchanged:
    from vae_analysis import hmm_report as HR
    HR.plot_transition(res); HR.plot_movement_dynamics(videos, res, lengths, bones,
        clip_len=cfg.clip_length, stride=cfg.clip_length//2, n_win=adapter.n_windows(),
        stream="pose")

Note: a state has no single decodable pose (it is dynamics, not a location), so
``decode_state_appearance`` does not apply — use the movement-dynamics figure.
"""

from __future__ import annotations

import numpy as np

from .hmm_pipeline import kfold_subject_splits, _fmt_dur


def _split(Z, lengths):
    """Concatenated ``(M, d)`` -> list of per-video ``(M_v, d)`` arrays."""
    offs = np.cumsum(np.r_[0, np.asarray(lengths)])
    return [np.ascontiguousarray(np.asarray(Z, float)[offs[i]:offs[i + 1]])
            for i in range(len(lengths))]


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


# Captures the last exception text so an all-``-inf`` sweep can explain itself
# instead of silently reporting no score (the errors are otherwise swallowed).
_ERRBOX = {"msg": None}


def _fit_one(datas, K, D, lags, n_iters, tol, seed):
    """One ssm AR-HMM EM fit.

    Returns ``(model, train_loglik, info)``; ``(None, -inf, info)`` on a
    degenerate init. ``info`` carries the telemetry the progress log prints —
    EM iterations run, whether the ``n_iters`` cap was reached, wall seconds,
    and the error text. The error travels **in the return value**, not only in
    ``_ERRBOX``: this runs in a joblib worker, whose module globals are a
    separate copy the parent never sees.
    """
    import ssm, time
    np.random.seed(seed)
    t0 = time.time()
    info = {"n_iter": 0, "hit_cap": False, "seconds": 0.0, "error": None,
            "loglik": -np.inf}
    try:
        model = ssm.HMM(K, D, observations="ar",
                        observation_kwargs=dict(lags=lags))
        lls = model.fit(datas, method="em", num_iters=n_iters, tolerance=tol,
                        verbose=0)
    except Exception as e:  # noqa: BLE001 — degenerate init; caller retries/skips
        info["error"] = f"{type(e).__name__}: {e}"
        info["seconds"] = time.time() - t0
        _ERRBOX["msg"] = info["error"]
        return None, -np.inf, info
    lls = np.asarray(lls, float)
    info["n_iter"] = int(len(lls))
    # ssm's EM returns one entry per iteration and stops early on tolerance, so
    # a full-length trace means it ran out rather than converged.
    info["hit_cap"] = bool(len(lls) >= n_iters)
    info["seconds"] = time.time() - t0
    ll = float(lls[-1])
    info["loglik"] = ll
    if not np.isfinite(ll):
        info["error"] = f"train log-likelihood was {ll} (non-finite)"
        _ERRBOX["msg"] = info["error"]
        return None, -np.inf, info
    return model, ll, info


def _best_of_restarts(datas, K, D, lags, n_iters, tol, n_restarts, seed):
    """Best-training-loglik model over ``n_restarts`` inits, plus telemetry."""
    best, best_ll, restarts = None, -np.inf, []
    for r in range(n_restarts):
        m, ll, info = _fit_one(datas, K, D, lags, n_iters, tol, seed + 1000 * r)
        restarts.append(info)
        if ll > best_ll:
            best, best_ll = m, ll
    return best, best_ll, restarts


def _selection_job(Z, lengths, k, lags, fold, val_videos, n_iters, tol,
                   n_restarts, seed):
    """One unit of ``(K, lags, fold)`` selection work, sized for frequent progress.

    Returns **only picklable scalars and telemetry** — never the fitted ssm
    model. ssm objects carry autograd closures that do not round-trip through
    joblib reliably, so the winning model is refitted once in the parent at
    ``(K*, lags*)`` instead of being shipped back from a worker.

    ``fold is None`` scores on all data (``selection="none"``); otherwise it
    trains on the complement of ``val_videos`` and scores the held-out fold as
    log-likelihood **per window**, so folds of different size are comparable.
    """
    import time
    t0 = time.time()
    out = {"k": k, "lags": lags, "fold": fold, "score": None, "restarts": [],
           "n_val": 0, "error": None}
    datas = _split(Z, lengths)
    D = np.asarray(Z).shape[1]
    n_videos = len(lengths)
    if fold is None:
        model, ll, restarts = _best_of_restarts(datas, k, D, lags, n_iters, tol,
                                                n_restarts, seed)
        out["restarts"] = restarts
        if model is None:
            out["error"] = _first_error(restarts)
        else:
            out["score"] = float(ll)
    else:
        val = set(np.asarray(val_videos).tolist())
        tr = [datas[i] for i in range(n_videos) if i not in val]
        va = [datas[i] for i in range(n_videos) if i in val]
        if not tr or not va:
            out["error"] = "empty split"
        else:
            model, _, restarts = _best_of_restarts(tr, k, D, lags, n_iters, tol,
                                                   n_restarts, seed)
            out["restarts"] = restarts
            out["n_val"] = int(sum(len(v) for v in va))
            if model is None:
                out["error"] = _first_error(restarts)
            else:
                try:
                    ll = model.log_likelihood(va)
                except Exception as e:  # noqa: BLE001
                    out["error"] = f"log_likelihood raised {type(e).__name__}: {e}"
                else:
                    if np.isfinite(ll):
                        out["score"] = float(ll / max(out["n_val"], 1))
                    else:
                        out["error"] = f"held-out log-likelihood was {ll}"
    out["seconds"] = time.time() - t0
    return out


def _first_error(restarts):
    """The first real error text across restarts, for the all--inf diagnosis."""
    for r in restarts:
        if r.get("error"):
            return r["error"]
    return "no restart converged"


def _fmt_restarts(restarts) -> str:
    """``"41,200*,fail"`` — EM iterations per restart, ``*`` = hit the cap."""
    out = []
    for r in restarts:
        if r.get("error"):
            out.append("fail")
        else:
            out.append(f"{r['n_iter']}{'*' if r.get('hit_cap') else ''}")
    return ",".join(out)


def _res_from_model(model, datas, lengths, K, f_win):
    """Wrap a fitted ssm AR-HMM into the plot-compatible ``res`` dict."""
    states = np.concatenate([model.most_likely_states(d) for d in datas]).astype(int)
    A = np.asarray(model.transitions.transition_matrix)
    occ = np.bincount(states, minlength=K) / max(len(states), 1)
    # dwell (pooled per-video runs, seconds) — never stitched across a boundary
    offs = np.cumsum(np.r_[0, np.asarray(lengths)])
    pooled = {s: [] for s in range(K)}
    for i in range(len(lengths)):
        for s, r in _runs(states[offs[i]:offs[i + 1]], K).items():
            pooled[s] += r
    dwell_win = np.array([np.mean(pooled[s]) if pooled[s] else np.nan
                          for s in range(K)])
    # stationary distribution (left eigenvector for eigenvalue 1)
    w, V = np.linalg.eig(A.T)
    stat = np.real(V[:, np.argmin(np.abs(w - 1.0))]); stat = stat / stat.sum()
    return {"model": model, "k": K, "states": states, "transition": A,
            "stationary": stat, "occupancy": occ, "f_win": f_win,
            "dwell_windows": dwell_win, "dwell_seconds": dwell_win / f_win,
            "ar_As": np.asarray(model.observations.As),
            "ar_bs": np.asarray(model.observations.bs),
            "ar_Sigmas": np.asarray(model.observations.Sigmas),
            "regularisation": {"final_covariance_type": f"ar(lags={model.observations.lags})"}}


def fit_arhmm(Z, lengths, *, k_range=range(2, 9), lags=1, f_win=6.25,
              n_iters=50, tol=1e-4, n_restarts=1, selection="cv", n_splits=5,
              seed=0, n_jobs=1, verbose=True) -> dict:
    """Fit an autoregressive HMM and select K, mirroring :func:`fit_hmm`.

    Args:
        Z, lengths: stitched trajectory + per-video window counts (from
            :func:`stitch_dataset`). Videos are the AR sequences (ssm takes a
            list, so variable lengths are native — no boundary stitching).
        k_range: candidate state counts. A single value (e.g. ``range(9, 10)``)
            forces K.
        lags: AR order p — an ``int`` (fixed) or an iterable such as ``[1, 2, 3]``
            to sweep. When a list is given, ``(K, lags)`` are selected **jointly**
            by the same held-out criterion (predictive LL, so a higher-order
            model only wins if it generalises). Each extra lag adds ``D`` columns
            to every state's dynamics matrix, so higher orders need more windows
            per state and cost more per fit.
        f_win: window sampling rate (Hz), for the dwell times in seconds.
        selection: ``"cv"`` (``n_splits``-fold partition over subjects, each
            validated once) or ``"none"`` (fit each K on all data, pick best
            train LL).
        n_restarts: EM restarts per fit (best train LL kept).
        n_jobs: processes for the sweep. Work is split one job per
            ``(K, lags, fold)``, so a candidate no longer has to finish before
            anything is reported. Workers return scores only — the model at
            ``(K*, lags*)`` is refitted once here, since ssm objects do not
            round-trip through joblib reliably.
        verbose: stream a progress log — one line per completed job with its
            held-out log-likelihood per window, the EM iterations of each
            restart (``*`` = hit the ``n_iters`` cap), wall time and a running
            ETA, then a summary per candidate. Printed from the **calling**
            process, so it reaches a notebook cell even when fits run in
            workers.

    Returns:
        A ``res`` dict compatible with :mod:`vae_analysis.hmm_report` and
        :func:`~vae_analysis.hmm_pipeline.state_dwell_times` — plus ``ar_As`` /
        ``ar_bs`` / ``ar_Sigmas`` (the per-state dynamics), ``selection_scores``.
    """
    import time as _time
    Z = np.asarray(Z, float); D = Z.shape[1]
    n_videos = len(lengths)
    datas_all = _split(Z, lengths)
    ks = [k for k in k_range if 2 <= k <= max(2, min(n_videos, len(Z) // 5))]
    if not ks:
        raise ValueError(f"no candidate K in {list(k_range)} fits the data budget.")
    lag_list = [int(lags)] if np.isscalar(lags) else [int(p) for p in lags]
    candidates = [(k, p) for k in ks for p in lag_list]

    # ---- fail loud on the two things that turn every fit into -inf ----------
    if not np.isfinite(Z).all():
        bad = int((~np.isfinite(Z)).any(axis=1).sum())
        raise ValueError(
            f"Z has non-finite values ({bad} rows with NaN/Inf). The AR-HMM "
            f"cannot fit — check the stitched latent / encoder output before "
            f"fitting (a static HMM may tolerate what ssm does not).")
    lag_max = max(lag_list)
    short = [i for i, dd in enumerate(datas_all) if len(dd) <= lag_max]
    if short:
        raise ValueError(
            f"{len(short)} video(s) have <= {lag_max} windows — too short for AR "
            f"lags={lag_max} (each sequence needs > lags points). Lower `lags`, "
            f"or drop the short recordings.")
    _ERRBOX["msg"] = None

    splits = (kfold_subject_splits(n_videos, n_splits, seed)
              if selection == "cv" else [])

    # One job per (K, lags, fold): small enough to keep every core busy and to
    # report progress many times per candidate rather than once at its end.
    jobs = []
    for k, p in candidates:
        if selection == "cv":
            for i, val_videos in enumerate(splits):
                jobs.append((k, p, i, val_videos))
        else:
            jobs.append((k, p, None, None))
    n_total = len(jobs)
    t_start = _time.time()

    if verbose:
        print(f"[arhmm] M={len(Z)} d={D} | K in {ks} | lags in {lag_list} | "
              f"selection={selection} | {len(candidates)} candidates, "
              f"{n_total} jobs | n_jobs={n_jobs}", flush=True)
        print(f"[arhmm] up to {n_total * n_restarts} EM fits "
              f"({n_restarts} restarts x <={n_iters} iters each). 'iters' below "
              f"is per restart; * = hit the {n_iters}-iteration cap.", flush=True)

    fold_scores = {c: [] for c in candidates}
    pending = {c: sum(1 for j in jobs if (j[0], j[1]) == c) for c in candidates}
    done = 0

    def _report(out):
        """Print one completed job from the **parent** process, with an ETA."""
        nonlocal done
        done += 1
        c = (out["k"], out["lags"])
        if out["score"] is not None:
            fold_scores[c].append(out["score"])
            val = (f"ll/win={out['score']:+.4f}" if out["fold"] is not None
                   else f"train ll={out['score']:.1f}")
        else:
            val = f"FAILED ({out['error']})"
            if out["error"]:
                _ERRBOX["msg"] = out["error"]     # workers cannot set ours
        pending[c] -= 1
        if verbose:
            what = ("all data" if out["fold"] is None
                    else f"fold {out['fold'] + 1}/{len(splits)}")
            elapsed = _time.time() - t_start
            eta = elapsed / done * (n_total - done)
            print(f"[arhmm] {done:>3}/{n_total} K={out['k']:<2} "
                  f"lags={out['lags']} {what:<11} {val:<24} "
                  f"iters {_fmt_restarts(out['restarts']):<14} "
                  f"{out['seconds']:6.1f}s | elapsed {_fmt_dur(elapsed)} "
                  f"eta {_fmt_dur(eta)}", flush=True)
            if pending[c] == 0:
                sc = float(np.mean(fold_scores[c])) if fold_scores[c] else -np.inf
                print(f"[arhmm]   -> K={c[0]} lags={c[1]} complete: "
                      f"score={sc:+.4f} over {len(fold_scores[c])} fits",
                      flush=True)

    _args = (n_iters, tol, n_restarts, seed)
    if n_jobs == 1:
        for k, p, fold, val_videos in jobs:
            _report(_selection_job(Z, lengths, k, p, fold, val_videos, *_args))
    else:
        from joblib import Parallel, delayed

        def _tasks():
            return (delayed(_selection_job)(Z, lengths, k, p, fold, val_videos,
                                            *_args)
                    for k, p, fold, val_videos in jobs)

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
                print("[arhmm] joblib<1.4: results arrive only at the end of "
                      "the sweep, so no per-job progress below.", flush=True)
            for out in Parallel(n_jobs=n_jobs, prefer="processes")(_tasks()):
                _report(out)

    scores = {c: (float(np.mean(fold_scores[c])) if fold_scores[c] else -np.inf)
              for c in candidates}

    if not scores or all(v == -np.inf for v in scores.values()):
        why = _ERRBOX["msg"] or "unknown (no exception captured)"
        raise ValueError(
            f"AR-HMM scored -inf for every (K, lags): the fits ran but produced "
            f"no finite score. Underlying cause: {why}. "
            f"Most common: an incompatible `ssm` install — verify "
            f"ssm.__file__ points at the git build and its primitives import "
            f"logsumexp from autograd.scipy.special (not scipy.misc); reinstall "
            f"from git and RESTART the runtime. Otherwise check Z for degenerate "
            f"(near-constant) dimensions.")
    k_best, lag_best = max(scores, key=scores.get)
    if verbose:
        ranked = sorted(scores, key=scores.get, reverse=True)
        runner_up = (f", runner-up K={ranked[1][0]} lags={ranked[1][1]} "
                     f"({scores[ranked[1]]:+.4f})" if len(ranked) > 1 else "")
        print(f"[arhmm] selection took {_fmt_dur(_time.time() - t_start)}; "
              f"K*={k_best} lags*={lag_best} ({scores[(k_best, lag_best)]:+.4f})"
              f"{runner_up}", flush=True)
        print(f"[arhmm] final fit at K={k_best} lags={lag_best} on all "
              f"{n_videos} recordings ...", flush=True)

    # Final fit on ALL data at (K*, lags*). Done here rather than reused from a
    # worker: the sweep deliberately returns scores only, never ssm models.
    _t_final = _time.time()
    model, ll_final, restarts = _best_of_restarts(
        datas_all, k_best, D, lag_best, n_iters, tol, max(n_restarts, 2), seed)
    if verbose:
        print(f"[arhmm]   iters {_fmt_restarts(restarts)}  "
              f"train ll={ll_final:.1f}  "
              f"({_time.time() - _t_final:.1f}s)", flush=True)
    if model is None:
        raise ValueError(
            "final AR-HMM fit failed at (K*, lags*): "
            f"{_first_error(restarts)}")
    res = _res_from_model(model, datas_all, lengths, k_best, f_win)
    res["selection"] = selection
    res["selection_scores"] = scores        # keyed by (K, lags)
    res["lags"] = lag_best
    if verbose:
        print(f"[arhmm] K*={k_best} lags*={lag_best} "
              f"occ={np.round(res['occupancy'], 2)} "
              f"dwell_s={np.round(res['dwell_seconds'], 2)}", flush=True)
    return res
