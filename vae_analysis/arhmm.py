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
    # res drops into the hmm_report figures + label_state_frequencies unchanged:
    from vae_analysis import hmm_report as HR
    HR.plot_transition(res); HR.plot_movement_dynamics(videos, res, lengths, bones,
        clip_len=cfg.clip_length, stride=cfg.clip_length//2, n_win=adapter.n_windows(),
        stream="pose")

Note: a state has no single decodable pose (it is dynamics, not a location), so
``decode_state_appearance`` does not apply — use the movement-dynamics figure.
"""

from __future__ import annotations

import numpy as np


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


def _fit_one(datas, K, D, lags, n_iters, tol, seed):
    """One ssm AR-HMM EM fit; returns (model, train_loglik) or (None, -inf)."""
    import ssm
    np.random.seed(seed)
    try:
        model = ssm.HMM(K, D, observations="ar",
                        observation_kwargs=dict(lags=lags))
        lls = model.fit(datas, method="em", num_iters=n_iters, tolerance=tol,
                        verbose=0)
    except Exception:  # noqa: BLE001 — degenerate init; caller retries/skips
        return None, -np.inf
    ll = float(lls[-1])
    return (model, ll) if np.isfinite(ll) else (None, -np.inf)


def _best_of_restarts(datas, K, D, lags, n_iters, tol, n_restarts, seed):
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        m, ll = _fit_one(datas, K, D, lags, n_iters, tol, seed + 1000 * r)
        if ll > best_ll:
            best, best_ll = m, ll
    return best, best_ll


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
              val_fraction=0.2, seed=0, verbose=True) -> dict:
    """Fit an autoregressive HMM and select K, mirroring :func:`fit_hmm`.

    Args:
        Z, lengths: stitched trajectory + per-video window counts (from
            :func:`stitch_dataset`). Videos are the AR sequences (ssm takes a
            list, so variable lengths are native — no boundary stitching).
        k_range: candidate state counts. A single value (e.g. ``range(9, 10)``)
            forces K.
        lags: AR order p (1 = depends on the previous window only).
        f_win: window sampling rate (Hz), for dwell seconds / the frequency map.
        selection: ``"cv"`` (mean held-out LL/window over ``n_splits`` video-wise
            splits) or ``"none"`` (fit each K on all data, pick best train LL).
        n_restarts: EM restarts per fit (best train LL kept).

    Returns:
        A ``res`` dict compatible with :mod:`vae_analysis.hmm_report` and
        :func:`label_state_frequencies` — plus ``ar_As`` / ``ar_bs`` /
        ``ar_Sigmas`` (the per-state dynamics), ``selection_scores``.
    """
    import time as _time
    Z = np.asarray(Z, float); D = Z.shape[1]
    n_videos = len(lengths)
    datas_all = _split(Z, lengths)
    ks = [k for k in k_range if 2 <= k <= max(2, min(n_videos, len(Z) // 5))]
    if not ks:
        raise ValueError(f"no candidate K in {list(k_range)} fits the data budget.")

    if verbose:
        print(f"[arhmm] M={len(Z)} d={D} lags={lags} | K in {ks} | "
              f"selection={selection}", flush=True)

    def video_splits():
        out = []
        for s in range(n_splits):
            perm = np.random.default_rng(seed + s).permutation(n_videos)
            n_val = max(1, int(round(n_videos * val_fraction)))
            out.append(set(perm[:n_val].tolist()))
        return out

    scores = {}
    for k in ks:
        t0 = _time.time()
        if selection == "cv":
            fold = []
            for val_set in video_splits():
                tr = [datas_all[i] for i in range(n_videos) if i not in val_set]
                va = [datas_all[i] for i in range(n_videos) if i in val_set]
                m, _ = _best_of_restarts(tr, k, D, lags, n_iters, tol, n_restarts, seed)
                if m is None:
                    continue
                try:
                    ll = m.log_likelihood(va); nva = sum(len(v) for v in va)
                    fold.append(ll / max(nva, 1))
                except Exception:  # noqa: BLE001
                    continue
            scores[k] = float(np.mean(fold)) if fold else -np.inf
        else:
            _, ll = _best_of_restarts(datas_all, k, D, lags, n_iters, tol,
                                      n_restarts, seed)
            scores[k] = ll
        if verbose:
            print(f"[arhmm]   K={k}: score={scores[k]:.3f}  ({_time.time()-t0:.1f}s)",
                  flush=True)

    if not scores or all(v == -np.inf for v in scores.values()):
        raise ValueError("no AR-HMM converged for any candidate K.")
    k_best = max(scores, key=scores.get)

    # final fit on ALL data at K*
    model, _ = _best_of_restarts(datas_all, k_best, D, lags, n_iters, tol,
                                 max(n_restarts, 2), seed)
    if model is None:
        raise ValueError("final AR-HMM fit failed at K*.")
    res = _res_from_model(model, datas_all, lengths, k_best, f_win)
    res["selection"] = selection
    res["selection_scores"] = scores
    if verbose:
        print(f"[arhmm] K*={k_best} occ={np.round(res['occupancy'], 2)} "
              f"dwell_s={np.round(res['dwell_seconds'], 2)}", flush=True)
    return res
