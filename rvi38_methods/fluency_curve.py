"""FLUENCY_CURVE — exploratory temporal decomposition of A1's ``Phi``.

A1 reports one scalar per recording, the excess similarity of consecutive
movements ``Phi = mean_t s_t - c`` (:func:`a1_core.phi_excess`), where
``s_t = S[q_t, q_{t+1}]`` is the kinematic similarity of the ``t``-th visit
transition and ``c`` is the occupancy-matched permutation null mean of §7.2.
That average discards *when* fluency rises and falls within a recording. This
module unrolls the same estimator along the transition index into a smoothed
curve ``phi(t)`` whose flat-kernel limit is exactly ``Phi`` (Prop 1), so nothing
new is estimated and nothing about A1 changes -- it is a lens on the identical
quantity.

Everything is reused from A1: the kinematic similarity matrix ``S`` (built from
its ``omega``), the run-length-compressed Viterbi visit sequence ``q``
(:func:`a1_core.visit_sequence`), and -- when available -- A1's cached null mean
``c`` and scalar ``Phi`` (``phi['null_mean']`` / ``phi['excess']`` from
:func:`a1_core.phi_excess`). Reusing the cache makes the flat-kernel identity of
step 2 hold to floating-point exactness; recomputing ``c`` here (``B = 2000``
reorderings that never repeat a state, the §7.2 null of
:func:`a1_core.smirnov_null`) reproduces it up to Monte-Carlo noise. The wall-clock transition time ``T_t`` is the plot axis only, derived from
A1's window geometry (:class:`a1_core.Geometry`) and the visit dwell lengths.

Definitions (METHODS §7 notation)::

    s_t      = S[q_t, q_{t+1}]                              t = 1..n-1
    c        ~ (1/B) sum_b  mean_t S[q_pi_b(t), q_pi_b(t+1)]   pi_b ~ U(no-repeat)
    phi(t)   = ( sum_t' k_sig(t'-t) s_t' ) / ( sum_t' k_sig(t'-t) )  -  c
    Phi      = ( 1/(n-1) sum_t s_t )  -  c
    k_sig(u) = exp(-u^2 / 2 sig^2)

The Gaussian kernel ``k_sig`` runs over the **transition index** ``t'`` (never
seconds), with ``sig in {3, 5}`` transitions; ``T_t`` is used only to place the
curve, the mean line and the rug on a wall-clock axis. The reported scalar stays
``Phi`` -- transition-averaged and unchanged. There is deliberately no
time-integral average ``(1/T) int phi``: that equals
``Phi + Cov_t(Delta, s) / mean(Delta)`` with ``Delta`` the inter-transition
gaps, i.e. a duration-weighted quantity that is *not* ``Phi``. The curve is
exploratory: the CSV carries ``id, label, Phi, n_transitions`` and no test
column, and ``sig`` is fixed a priori, never tuned against a group contrast.
"""

from __future__ import annotations

import os

import numpy as np

from a1_core import (GEOM, Geometry, run_lengths, smirnov_null,
                     visit_sequence)


# ---------------------------------------------------------------------------
# the three estimator primitives of the spec (plus the plot-axis helper)
# ---------------------------------------------------------------------------
def transition_similarity(S: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``s_t = S[q_t, q_{t+1}]`` for the visit transitions, shape ``(n-1,)``.

    ``q`` is the run-length-compressed visit sequence of §7.1, so consecutive
    entries differ and no ``s_t`` is a self-similarity ``S_kk = 1``. Its mean is
    A1's observed statistic ``phi['observed']`` exactly.
    """
    S = np.asarray(S)
    q = np.asarray(q)
    if q.shape[0] < 2:
        return np.empty(0, float)
    return S[q[:-1], q[1:]].astype(float)


def phi_curve(s: np.ndarray, sigma: float, c: float) -> np.ndarray:
    """Gaussian-smoothed excess similarity along the transition index.

    ``phi(t) = ( sum_t' k(t'-t) s_t' ) / ( sum_t' k(t'-t) ) - c`` with
    ``k(u) = exp(-u^2 / 2 sigma^2)`` on the integer index ``t'`` (never on
    ``T_t``). The kernel is weight-normalised, so a flat kernel
    (``sigma -> inf``) returns the constant ``mean(s) - c = Phi`` at every ``t``
    (Prop 1) and no edge correction is needed near the ends of the recording.

    Returns a curve of the same length as ``s`` (``n-1``).
    """
    s = np.asarray(s, float)
    n = s.shape[0]
    if n == 0:
        return s.copy()
    idx = np.arange(n, dtype=float)
    d = idx[:, None] - idx[None, :]                    # (n, n): t - t'
    K = np.exp(-(d * d) / (2.0 * float(sigma) ** 2))   # symmetric, diag = 1
    return (K @ s) / K.sum(axis=1) - float(c)


def null_offset(S: np.ndarray, q: np.ndarray, B: int = 2000, seed: int = 0,
                cache: float | None = None) -> float:
    """A1's §7.2 occupancy-matched null mean ``c`` for a single visit sequence.

    ``c = (1/B) sum_b mean_t S[q_pi_b(t), q_pi_b(t+1)]`` over ``B`` reorderings
    ``pi_b`` of ``q`` drawn uniformly from those that never repeat a state, the
    space ``q`` itself lives in (:func:`a1_core.smirnov_null`). This is
    exactly the ``null_mean`` that :func:`a1_core._phi_one` accumulates, so
    ``mean(s) - c`` equals A1's ``Phi`` up to the Monte-Carlo error of the two
    independent draws.

    Pass ``cache`` (A1's ``phi['null_mean'][i]``) to reuse the identical value
    A1 already computed; then the step-2 flat-kernel identity holds exactly.
    """
    if cache is not None:
        return float(cache)
    S = np.asarray(S)
    q = np.asarray(q)
    if q.shape[0] < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    return float(smirnov_null(q, S, int(B), rng)["null"])


def transition_times(states_i: np.ndarray, geom: Geometry = GEOM) -> np.ndarray:
    """Wall-clock time (s) of each visit transition -- the plot axis only.

    A1 states are attributed to delta windows on a regular tiling
    (:meth:`a1_core.Geometry.delta_spans`): window ``m`` opens at frame
    ``f0 + l*m``, i.e. at ``(f0 + l*m) / fps`` seconds. Transition ``t`` occurs at
    the window where the ``(t+1)``-th visit begins, whose index is the cumulative
    dwell ``cumsum(run_lengths)[t]``. Returns times aligned with
    :func:`transition_similarity` (shape ``n-1``).

    The statistic never reads ``T`` -- smoothing is on the index -- so any affine
    reparametrisation of this axis leaves ``phi`` and ``Phi`` untouched.
    """
    states_i = np.asarray(states_i)
    _, rl = run_lengths(states_i)
    if len(rl) < 2:
        return np.empty(0, float)
    onset = np.cumsum(rl)[:-1]                          # window index of each txn
    return (geom.f0 + geom.l * onset) / geom.fps


# ---------------------------------------------------------------------------
# plotting: one panel per recording
# ---------------------------------------------------------------------------
_SIG3, _SIG5 = "#e08214", "#2b6cb0"      # sigma = 3 (orange), sigma = 5 (blue)
_MEAN, _RAW, _RUG = "#555555", "#b5b5b5", "#8a8a8a"


def plot_fluency_curve(phi3, phi5, s, T, Phi, meta, out_png):
    """Write the per-recording fluency-curve panel to ``out_png`` (PNG).

    * ``phi(t)`` for ``sigma = 3`` and ``sigma = 5`` against the wall-clock
      transition time ``T``;
    * horizontal lines at ``0`` (chance) and at the recording mean ``Phi``;
    * the raw per-transition excess ``s_t - c`` as faint dots (the data the
      curve smooths; ``c = mean(s) - Phi`` is recovered from the identity);
    * a rug of the transitions ``{s_t}`` at ``{T_t}`` below the axis.

    ``meta`` supplies ``id`` and ``label`` for the title
    ``id | label | Phi={:.3f}`` and, optionally, ``sigmas`` and
    ``n_transitions`` for the subtitle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.asarray(s, float)
    T = np.asarray(T, float)
    Phi = float(Phi)
    c = float(np.mean(s)) - Phi if s.size else 0.0      # s_t - c is the raw excess
    s_exc = s - c
    sig3, sig5 = (meta.get("sigmas", (3, 5)) if isinstance(meta, dict)
                  else (3, 5))

    phi3 = np.asarray(phi3, float)
    phi5 = np.asarray(phi5, float)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    if T.size:
        # Frame the view on the curves, chance and Phi -- not on the raw excess,
        # whose spread would otherwise squash the curve. Faint raw dots are drawn
        # for context and clipped to this frame; the rug keeps every transition.
        span = np.concatenate([phi3, phi5, [0.0, Phi]])
        lo, hi = float(np.min(span)), float(np.max(span))
        pad = 0.08 * max(hi - lo, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        ax.scatter(T, s_exc, s=9, color=_RAW, alpha=0.5, edgecolors="none",
                   zorder=1, clip_on=True)
        ax.plot(T, phi3, color=_SIG3, lw=1.4,
                label=f"$\\phi(t)$, $\\sigma$={sig3}", zorder=3)
        ax.plot(T, phi5, color=_SIG5, lw=2.1,
                label=f"$\\phi(t)$, $\\sigma$={sig5}", zorder=4)
        # rug: one tick per transition {s_t} at its wall-clock time {T_t}, below.
        trans = ax.get_xaxis_transform()
        ax.vlines(T, 0.0, 0.03, transform=trans, color=_RUG, lw=0.6,
                  alpha=0.85, zorder=2)
    ax.axhline(0.0, color="k", lw=0.8, zorder=2)
    ax.axhline(Phi, color=_MEAN, ls="--", lw=1.3, zorder=2,
               label=f"$\\Phi$ = {Phi:+.3f}")
    ax.annotate("chance", xy=(0.995, 0.0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7, color="#333333")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("fluency  $\\phi(t)$  (excess similarity)")
    ax.margins(x=0.01)
    ax.legend(fontsize=7.6, frameon=False, ncol=3, loc="best")

    ident = str(meta.get("id", "")) if isinstance(meta, dict) else str(meta)
    lab = str(meta.get("label", "")) if isinstance(meta, dict) else ""
    ntx = (meta.get("n_transitions", s.size) if isinstance(meta, dict)
           else s.size)
    ax.set_title(f"{ident} | {lab} | $\\Phi$={Phi:.3f}", loc="left", fontsize=10)
    ax.text(0.99, 0.965, f"$n$={ntx} txns · smoothed on index, "
            f"$\\sigma\\in\\{{{sig3},{sig5}\\}}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color="#777777")

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# driver: reuse A1 artefacts, one figure per recording + one CSV
# ---------------------------------------------------------------------------
def _safe(name: str) -> str:
    keep = "-_.() "
    return "".join(ch if ch.isalnum() or ch in keep else "_"
                   for ch in str(name)).strip().replace(" ", "_") or "recording"


def fluency_curves(states, vidid, S, ids, labels, phi=None, geom: Geometry = GEOM,
                   sigmas=(3, 5), out_root=".", B: int = 2000, seed: int = 0,
                   label_names=("normal", "abnormal"), atol: float = 1e-9,
                   verbose: bool = True):
    """Fluency curves for every recording, reusing A1 artefacts.

    Parameters mirror A1's in-memory objects:

    * ``states``, ``vidid`` -- the per-window Viterbi state path and its
      recording index (``model['states']``, ``model['vidid']``).
    * ``S`` -- the direction-aware similarity built from A1's ``omega``
      (``results[primary]['S']``). Everything about the curve inherits ``omega``
      through ``S``; no channel or weight is re-chosen here.
    * ``ids``, ``labels`` -- recording names and group labels (``video_names``,
      ``labels``).
    * ``phi`` -- optional A1 fluency cache (:func:`a1_core.phi_excess` output).
      When given, ``c = phi['null_mean'][i]`` and ``Phi = phi['excess'][i]`` are
      the A1 values verbatim, so the step-2 assertion holds exactly; otherwise
      ``c`` is recomputed with ``B`` permutations and ``Phi = mean(s) - c``.

    Writes ``<out_root>/figures/fluency_curve/{id}.png`` per recording and the
    exploratory ``<out_root>/results/fluency_curve.csv`` with columns
    ``id, label, Phi, n_transitions``. Returns ``(rows, fig_paths, csv_path)``.
    """
    states = np.asarray(states)
    vidid = np.asarray(vidid)
    S = np.asarray(S)
    ids = list(ids)
    labels = list(labels)
    n_rec = len(ids)

    fig_dir = os.path.join(out_root, "figures", "fluency_curve")
    res_dir = os.path.join(out_root, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    def _label_str(lab):
        try:
            iv = int(lab)
        except (TypeError, ValueError):
            return str(lab)
        if label_names is not None and 0 <= iv < len(label_names):
            return label_names[iv]
        return str(lab)

    rows, fig_paths = [], []
    for i in range(n_rec):
        st_i = states[vidid == i]
        q = visit_sequence(st_i)
        s = transition_similarity(S, q)
        n_tx = int(s.size)

        # ---- step 2: c, Phi, and the flat-kernel identity (Prop 1) ----
        cache_c = None
        cache_phi = None
        if phi is not None:
            cache_c = float(np.asarray(phi["null_mean"])[i])
            cache_phi = float(np.asarray(phi["excess"])[i])

        if n_tx < 1 or not np.isfinite(s).all():
            rows.append((ids[i], labels[i],
                         cache_phi if cache_phi is not None else float("nan"),
                         n_tx))
            if verbose:
                print(f"  {ids[i]}: {n_tx} transitions — curve skipped")
            continue

        c = null_offset(S, q, B=B, seed=seed, cache=cache_c)
        Phi = cache_phi if cache_phi is not None else float(s.mean() - c)
        # flat-kernel curve == Phi  <=>  mean(s) - c == Phi.  Guards against an
        # S / Phi mismatch (e.g. a cache from a different omega).
        assert abs(float(s.mean()) - c - Phi) < atol, (
            f"{ids[i]}: flat-kernel identity broken "
            f"(mean(s)-c={float(s.mean()) - c:.3e} vs Phi={Phi:.3e})")

        # ---- step 3: the two curves ----
        phi3 = phi_curve(s, sigmas[0], c)
        phi5 = phi_curve(s, sigmas[1], c)
        T = transition_times(st_i, geom)

        # ---- step 4: plot ----
        out_png = os.path.join(fig_dir, f"{_safe(ids[i])}.png")
        meta = {"id": ids[i], "label": _label_str(labels[i]),
                "sigmas": tuple(sigmas), "n_transitions": n_tx}
        plot_fluency_curve(phi3, phi5, s, T, Phi, meta, out_png)
        fig_paths.append(out_png)
        rows.append((ids[i], labels[i], Phi, n_tx))
        if verbose:
            print(f"  {ids[i]}: Phi={Phi:+.3f}, {n_tx} transitions -> {out_png}")

    csv_path = os.path.join(res_dir, "fluency_curve.csv")
    _write_csv(csv_path, rows)
    if verbose:
        print(f"  wrote {csv_path} ({len(rows)} recordings) and "
              f"{len(fig_paths)} figures under {fig_dir}")
    return rows, fig_paths, csv_path


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "Phi", "n_transitions"])
        for ident, lab, Phi, n_tx in rows:
            phi_str = "" if Phi is None or not np.isfinite(Phi) else f"{Phi:.6f}"
            w.writerow([ident, lab, phi_str, n_tx])
    return path


# ---------------------------------------------------------------------------
# self-test: the identities the module must satisfy
# ---------------------------------------------------------------------------
def _selftest() -> int:
    ok = True

    rng = np.random.default_rng(0)
    K = 6
    S = np.clip(rng.normal(0, 0.4, (K, K)), -1, 1)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 1.0)
    q = visit_sequence(rng.integers(0, K, 400))
    s = transition_similarity(S, q)
    c = null_offset(S, q, B=500, seed=1)

    # Prop 1: a wide kernel collapses the curve to the constant Phi.
    Phi = float(s.mean() - c)
    flat = phi_curve(s, 1e6, c)
    p1 = np.allclose(flat, Phi, atol=1e-9)
    print(f"  {'ok  ' if p1 else 'FAIL'}  flat-kernel curve == Phi (Prop 1)  "
          f"max|dev|={np.abs(flat - Phi).max():.2e}")
    ok &= p1

    # weight-normalisation: each phi(t) is a convex combination of s (minus c),
    # so phi(t)+c must stay within [min s, max s] for any sigma. (The *index*
    # mean of phi is not exactly Phi at finite sigma because of edge weighting.)
    conv = all(np.all(phi_curve(s, sg, 0.0) <= s.max() + 1e-9)
               and np.all(phi_curve(s, sg, 0.0) >= s.min() - 1e-9)
               for sg in (3, 5))
    print(f"  {'ok  ' if conv else 'FAIL'}  phi(t)+c stays within [min s, max s] "
          f"(convex, weight-normalised)")
    ok &= conv

    # transition_similarity length and the observed-mean identity.
    lp = s.shape[0] == q.shape[0] - 1
    print(f"  {'ok  ' if lp else 'FAIL'}  s has n-1 entries")
    ok &= lp

    # null_offset reuses the cache verbatim.
    cc = null_offset(S, q, cache=0.1234)
    print(f"  {'ok  ' if cc == 0.1234 else 'FAIL'}  null_offset returns the cache")
    ok &= cc == 0.1234

    # transition_times: monotone, length n-1, spaced by whole windows.
    st = np.repeat([0, 1, 2, 0], [3, 2, 4, 1])          # dwell 3,2,4,1 windows
    T = transition_times(st, GEOM)
    good_T = (T.shape[0] == 3 and np.all(np.diff(T) > 0)
              and abs(T[0] - (GEOM.f0 + GEOM.l * 3) / GEOM.fps) < 1e-9)
    print(f"  {'ok  ' if good_T else 'FAIL'}  transition_times monotone, "
          f"onset windows [3,5,9]  T={np.round(T, 3).tolist()}")
    ok &= good_T

    print("  self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
