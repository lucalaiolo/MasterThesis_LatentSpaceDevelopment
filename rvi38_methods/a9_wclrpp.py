"""WCLR-PP: windowed conditional limb regression, phase-preserving.

A vector-valued directed measure of inter-limb coordination on the raw
registered keypoints. It replaces the earlier cosine-Gram co-movement construct
(``a8_movement``'s ``comovement_*``): both target the cramped-synchronised
signature, but WCLR-PP measures the variance in one limb's *future* 2D velocity
that the other limb explains **beyond that limb's own past**, so it separates
genuine coupling from shared autocorrelation and keeps direction. In-phase
co-contraction and anti-phase alternation are distinct outcomes here, which is
the clinical point: anti-phase alternation is the healthy fidgety pattern and
in-phase co-contraction the cramped-synchronised one, so the sign of coupling
flips relative to the dyad-synchrony literature and *high* coupling is the
pathological pole.

Like everything in :mod:`a8_movement` this reads raw keypoint displacements
only — never the encoder or the state model — so it is an independent estimator
alongside the state-based quantities of A1/A7.

Two nested multivariate OLS models are fit in every window, predicting limb A's
future 2D velocity from A's own past (M1) and from A's past plus B's (M2):

.. math::
    R^2_{M\\bullet} = 1 - \\frac{\\lVert U\\rVert_F^2}{\\lVert Y-\\bar Y\\rVert_F^2},
    \\qquad \\Delta R^2(t,\\tau) = R^2_{M2}-R^2_{M1} \\ge 0 .

The Frobenius (trace) norm makes the measure rotation- and scale-invariant. The
``(t,\\tau)`` map of ``ΔR²`` is reduced by the peak-picking of the spec to two
per-pair scalars: ``F`` (fraction of assessable time spent coupled, the
pathological quantity) and ``R2`` (mean coupling strength when coupled).

What counts as "a limb's velocity" is a construct choice, exposed as
:data:`LIMB_SIGNALS` and selected by ``WCLRParams.limb_signal``: the mean over
the distal two joints (``"distal"``, elbow+wrist and knee+ankle — the specified
construct, METHODS §Synchrony, which fixes ``J_L`` as the distal joints a
priori), the distal end-effector alone (``"end_effector"``), or the mean over
the whole chain including shoulder or hip (``"limb"``). The proximal joints are
not a free choice here — see the warning on :data:`LIMB_SIGNALS`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np



def _nanmean(a, axis=None):
    """``np.nanmean`` that returns NaN for an all-NaN slice without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)

# ---------------------------------------------------------------------------
# limb velocity signals and the six clinically-motivated pairs (BODY-15 layout)
# ---------------------------------------------------------------------------
LIMB_ORDER: tuple[str, ...] = ("RA", "LA", "RL", "LL")

#: A limb's velocity signal is the **mean signed** frame-to-frame displacement
#: of the joints listed here, so which joints are listed is a construct choice:
#:
#: ``"distal"``
#:     elbow+wrist and knee+ankle — the limb average without the proximal
#:     joints, and the specified construct: METHODS §Synchrony fixes ``J_L`` as
#:     the distal joints a priori;
#: ``"end_effector"``
#:     the distal keypoint alone (wrist, ankle);
#: ``"limb"``
#:     the whole chain, shoulder or hip included.
#:
#: **Averaging buys less than it appears to.** The motive for a limb average is
#: that it should suppress per-keypoint detector noise, which differencing
#: amplifies and which is the binding constraint on this measure. It does, but
#: weakly: torso normalisation pins MidHip and Neck, so proximal joints must
#: move *less* than distal ones, and they therefore contribute little signal
#: while contributing full noise. Simulated at a plausible chain gain
#: (0.25/0.6/1.0 from proximal to distal), the velocity SNR of the leg signal
#: relative to the ankle alone is **1.18x** for ``"distal"`` and **1.03x** for
#: ``"limb"`` under iid per-keypoint noise -- and *below* 1 (0.74x and 0.47x)
#: once the detector's errors are correlated along the limb, as they are
#: whenever a mislocated segment moves its joints together. So ``"distal"`` is
#: the specified construct on anatomical grounds, not because averaging buys
#: much noise suppression; read ``"end_effector"`` as the robustness check
#: against it rather than as a degraded version of it.
#:
#: **The proximal joints are not a free choice.** The pose is torso-normalised
#: (:mod:`build_pose`: MidHip pinned at ``(0,0)``, Neck at ``(0,1)``), which
#: leaves RHip/LHip placed about a fixed MidHip and RShoulder/LShoulder about a
#: fixed Neck. Wherever that placement is symmetric — MidHip at the midpoint of
#: the two hips is exactly the case in which it is — the two hips carry velocity
#: components that are a *negation* of one another, and likewise the two
#: shoulders: a deterministic linear relation that is a property of the
#: normalisation, not of the infant. ``ΔR²`` is explained variance and is blind
#: to the sign of the relation, so folding those joints into the limb means
#: injects near-perfectly predictable common variance into precisely the two
#: homologous pairs (RA-LA, RL-LL) that carry the cramped-synchronised
#: signature, driving their ``F`` toward the ceiling for every infant and
#: flattening the group contrast.
LIMB_SIGNALS: dict[str, dict[str, tuple[int, ...]]] = {
    "end_effector": {"RA": (4,), "LA": (7,), "RL": (11,), "LL": (14,)},
    "distal": {"RA": (3, 4), "LA": (6, 7), "RL": (10, 11), "LL": (13, 14)},
    "limb": {"RA": (2, 3, 4), "LA": (5, 6, 7),
             "RL": (9, 10, 11), "LL": (12, 13, 14)},
}
DEFAULT_LIMB_SIGNAL = "distal"

END_EFFECTORS: dict[str, int] = {
    "RWrist": 4, "LWrist": 7, "RAnkle": 11, "LAnkle": 14,
}

# Each pair is (limb A, limb B, anatomical class). The class is the interpretive
# key: a homologous or ipsilateral coupling that is high on every pair at once is
# the whole-body cramped-synchronised pattern.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("RA", "LA", "homologous"),      # homologous arms
    ("RL", "LL", "homologous"),      # homologous legs
    ("RA", "RL", "ipsilateral"),     # ipsilateral right
    ("LA", "LL", "ipsilateral"),     # ipsilateral left
    ("RA", "LL", "contralateral"),   # contralateral
    ("LA", "RL", "contralateral"),   # contralateral
)
PAIR_CLASSES: tuple[str, ...] = tuple(p[2] for p in PAIRS)

# Labels follow the signal: an end-effector run is named for the keypoint it
# actually uses, a limb-average run for the limb.
_SIGNAL_ABBREV: dict[str, dict[str, str]] = {
    "end_effector": {"RA": "RWr", "LA": "LWr", "RL": "RAn", "LL": "LAn"},
}


def pair_names(limb_signal: str = DEFAULT_LIMB_SIGNAL) -> tuple[str, ...]:
    """The six pair labels for a limb signal, in :data:`PAIRS` order."""
    ab = _SIGNAL_ABBREV.get(limb_signal, {})
    return tuple(f"{ab.get(a, a)}-{ab.get(b, b)}" for a, b, _ in PAIRS)


PAIR_NAMES: tuple[str, ...] = pair_names()
PAIR_CLASS: dict[str, str] = dict(zip(PAIR_NAMES, PAIR_CLASSES))


@dataclass(frozen=True)
class WCLRParams:
    """Peak-picking and regression parameters (25 fps, frame units).

    Defaults are the spec values: ``w`` = 2 s window, ``tau_max`` ≈ 0.52 s of
    lag either way, ``ell_min`` = 0.76 s minimum coupled run, and the inherited
    ``c`` = 0.25 magnitude cutoff (kept as a heuristic gate, not compared across
    methods). ``h`` = 1 means one assessable row per frame, so ``F`` is a
    frame-fraction.

    ``dtau`` = 3 is the specified peak-lag continuity tolerance: consecutive
    rows chain into one coupled run only while their peak lags stay within this
    many frames, so lowering it breaks runs at every lag step and ``F`` falls.

    ``limb_signal`` selects which joints define a limb's velocity — a key of
    :data:`LIMB_SIGNALS` or an explicit ``{limb: joint indices}`` map. It
    defaults to the specified ``"distal"`` (elbow+wrist, knee+ankle); changing
    it changes the construct, not just an estimator setting.
    """

    w: int = 50
    tau_max: int = 13
    h: int = 1
    dtau: int = 3
    ell_min: int = 19
    c: float = 0.25
    fps: float = 25.0
    eps: float = 1e-12
    limb_signal: str = DEFAULT_LIMB_SIGNAL


# ---------------------------------------------------------------------------
# 0. limb velocity signals (the input to everything below)
# ---------------------------------------------------------------------------
def limb_velocities(video: np.ndarray,
                    limb_signal: str | dict[str, tuple[int, ...]]
                    = DEFAULT_LIMB_SIGNAL) -> np.ndarray:
    """Per-limb velocity series ``(L, 4, 2)``, limbs in :data:`LIMB_ORDER`.

    The mean is taken over the **signed** displacement, so an oscillation
    survives it — averaging speeds instead would destroy the very alternation
    the measure exists to read.
    """
    x = np.asarray(video, float)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"video must be (F, J, 2); got {x.shape}")
    if isinstance(limb_signal, str):
        if limb_signal not in LIMB_SIGNALS:
            raise ValueError(f"unknown limb signal {limb_signal!r}; choose "
                             f"from {sorted(LIMB_SIGNALS)}")
        joints = LIMB_SIGNALS[limb_signal]
    else:
        joints = limb_signal
    missing = [L for L in LIMB_ORDER if not len(joints.get(L, ()))]
    if missing:
        raise ValueError(f"limb signal defines no joints for {missing}")
    vel = np.diff(x, axis=0)                                    # (L, J, 2)
    return np.stack([vel[:, list(joints[L]), :].mean(axis=1)
                     for L in LIMB_ORDER], axis=1)              # (L, 4, 2)


# ---------------------------------------------------------------------------
# 1. the vector-valued ΔR² map
# ---------------------------------------------------------------------------
def delta_r2_matrix(vA: np.ndarray, vB: np.ndarray, w: int, tau_max: int,
                    eps: float = 1e-12
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``ΔR²(t,τ)`` for A predicted from A's past (M1) and A+B (M2).

    ``vA``, ``vB`` are ``(L, 2)`` velocity series (frame-to-frame displacement).
    Rows are window starts ``t``, columns are the ``2τ_max+1`` lags. Only starts
    that admit every lag are assessable::

        τ_max ≤ t ≤ L - w - τ_max,  D = N_rows = L - w + 1 - 2τ_max.

    Returns ``(M, taus, SSR1, TSS)``: ``M`` is ``(D, 2τ_max+1)`` of ``ΔR²``,
    ``taus`` the lag axis, and ``SSR1``/``TSS`` are the M1 residual and total
    scatter per cell — returned so a circular-shift surrogate can reuse the
    fixed M1 branch and only recompute M2.

    The design uses the pseudo-inverse of the Gram matrix, so a window in which
    a limb is momentarily still (a rank-deficient design, e.g. the 1-D reduction
    check where every ``Δy ≡ 0``) yields the minimum-norm fit and hence the
    correct residuals rather than a singular-matrix error.
    """
    vA = np.asarray(vA, float)
    vB = np.asarray(vB, float)
    L = len(vA)
    lo = tau_max
    hi = L - w - tau_max                       # inclusive last valid start
    if hi < lo:
        raise ValueError(
            f"clip too short: L={L}, w={w}, tau_max={tau_max} leaves no rows")
    rows = np.arange(lo, hi + 1)
    D = len(rows)
    k = rows[:, None] + np.arange(w)[None, :]  # (D, w) predictor indices
    PA = vA[k]                                  # (D, w, 2)
    PB = vB[k]                                  # (D, w, 2)
    ones = np.ones((D, w, 1))
    X1 = np.concatenate([ones, PA], axis=-1)             # (D, w, 3)
    X2 = np.concatenate([ones, PA, PB], axis=-1)         # (D, w, 5)
    P1 = np.linalg.pinv(np.einsum("dwi,dwj->dij", X1, X1))   # (D, 3, 3)
    P2 = np.linalg.pinv(np.einsum("dwi,dwj->dij", X2, X2))   # (D, 5, 5)

    taus = np.arange(-tau_max, tau_max + 1)
    M = np.empty((D, len(taus)), float)
    SSR1 = np.empty((D, len(taus)), float)
    TSS = np.empty((D, len(taus)), float)
    for j, tau in enumerate(taus):
        Y = vA[k + tau]                                   # (D, w, 2)
        tss = ((Y - Y.mean(1, keepdims=True)) ** 2).sum((1, 2))
        B1 = P1 @ np.einsum("dwi,dwc->dic", X1, Y)
        B2 = P2 @ np.einsum("dwi,dwc->dic", X2, Y)
        s1 = ((Y - np.einsum("dwi,dic->dwc", X1, B1)) ** 2).sum((1, 2))
        s2 = ((Y - np.einsum("dwi,dic->dwc", X2, B2)) ** 2).sum((1, 2))
        SSR1[:, j] = s1
        TSS[:, j] = tss
        with np.errstate(invalid="ignore", divide="ignore"):
            dr2 = np.where(tss > eps, (s1 - s2) / tss, 0.0)
        M[:, j] = dr2
    return M, taus, SSR1, TSS


def _m2_from_shift(vA, vB, rows, k, w, taus, P1_unused, SSR1, TSS,
                   eps=1e-12) -> np.ndarray:
    """ΔR² with a fixed M1 branch (``SSR1``, ``TSS``) and a fresh M2 on ``vB``.

    Used by the surrogate: M1 depends only on A and is invariant to shifting B,
    so only M2's design, coefficients and residuals are recomputed.
    """
    PA = vA[k]
    PB = vB[k]
    D = len(rows)
    ones = np.ones((D, w, 1))
    X2 = np.concatenate([ones, PA, PB], axis=-1)
    P2 = np.linalg.pinv(np.einsum("dwi,dwj->dij", X2, X2))
    M = np.empty((D, len(taus)), float)
    for j, tau in enumerate(taus):
        Y = vA[k + tau]
        B2 = P2 @ np.einsum("dwi,dwc->dic", X2, Y)
        s2 = ((Y - np.einsum("dwi,dic->dwc", X2, B2)) ** 2).sum((1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            M[:, j] = np.where(TSS[:, j] > eps,
                               (SSR1[:, j] - s2) / TSS[:, j], 0.0)
    return M


# ---------------------------------------------------------------------------
# 2. peak-picking (method-agnostic; operates on any ΔR² matrix)
# ---------------------------------------------------------------------------
def peak_pick(M: np.ndarray, c: float = 0.25, dtau: int = 3,
              ell_min: int = 19, h: int = 1, D: int | None = None) -> dict:
    """Reduce a ``ΔR²`` matrix to the two per-pair scalars ``F`` and ``R2``.

    Per row keep the largest-``ΔR²`` cell; drop rows whose peak is ``≤ c``; chain
    surviving consecutive rows whose peak lags are within ``dtau``; keep a run
    only if its mean peak exceeds ``c`` and its length is at least ``ell_min``.
    Then ``F = h·Σℓ_s / D`` (fraction of assessable time coupled) and
    ``R2 = mean over kept runs of the run mean`` (strength when coupled).

    ``D`` defaults to the number of assessable rows, ``M.shape[0]`` — the correct
    F denominator per the spec (``N_rows``, not wall-clock ``T``).
    """
    M = np.asarray(M, float)
    n = M.shape[0]
    D = n if D is None else D
    if n == 0 or D <= 0:
        return {"F": 0.0, "R2": np.nan, "n_intervals": 0, "intervals": [],
                "tau_star": np.empty(0, int), "peak": np.empty(0)}
    star = np.nanargmax(M, axis=1)
    peak = M[np.arange(n), star]
    keep = peak > c

    intervals: list[tuple[int, int, int, float]] = []
    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while (j + 1 < n and keep[j + 1]
               and abs(int(star[j + 1]) - int(star[j])) <= dtau):
            j += 1
        length = j - i + 1
        mean = float(peak[i:j + 1].mean())
        intervals.append((i, j, length, mean))
        i = j + 1

    kept = [iv for iv in intervals if iv[3] > c and iv[2] >= ell_min]
    total = sum(iv[2] for iv in kept)
    F = h * total / D
    R2 = float(np.mean([iv[3] for iv in kept])) if kept else np.nan
    return {"F": float(F), "R2": R2, "n_intervals": len(kept),
            "intervals": kept, "tau_star": star, "peak": peak}


# ---------------------------------------------------------------------------
# 3. directed and symmetric per-pair coupling
# ---------------------------------------------------------------------------
def directed_wclrpp(vA: np.ndarray, vB: np.ndarray,
                    params: WCLRParams = WCLRParams()) -> dict:
    """Coupling of B *into* A: predict A's future from A's past plus B."""
    M, _, _, _ = delta_r2_matrix(vA, vB, params.w, params.tau_max, params.eps)
    Mc = np.clip(M, 0.0, None)                 # remove fp negatives before pick
    res = peak_pick(Mc, params.c, params.dtau, params.ell_min, params.h)
    res["min_dr2"] = float(np.nanmin(M))       # for the ΔR²≥0 assertion
    return res


def pair_wclrpp(v1: np.ndarray, v2: np.ndarray,
                params: WCLRParams = WCLRParams()) -> dict:
    """Symmetric coupling of a limb pair: average both directions.

    The regression is directional (B→A ≠ A→B), so a single per-pair score
    averages the two, as the spec prescribes for a symmetric summary.
    """
    d12 = directed_wclrpp(v1, v2, params)      # v2 into v1
    d21 = directed_wclrpp(v2, v1, params)      # v1 into v2
    F = 0.5 * (d12["F"] + d21["F"])
    R2 = float(_nanmean([d12["R2"], d21["R2"]]))
    return {"F": F, "R2": R2, "directed": (d12, d21),
            "min_dr2": min(d12["min_dr2"], d21["min_dr2"])}


# ---------------------------------------------------------------------------
# 4. per-recording and per-cohort
# ---------------------------------------------------------------------------
def wclrpp_recording(video: np.ndarray, params: WCLRParams = WCLRParams()
                     ) -> tuple[np.ndarray, np.ndarray]:
    """The six per-pair ``F`` and ``R2`` values for one recording.

    ``video`` is ``(F, 15, 2)`` registered keypoints; each limb's velocity is
    built by :func:`limb_velocities` under ``params.limb_signal``. Returns
    ``(F[6], R2[6])`` in ``pair_names(params.limb_signal)`` order.
    """
    V = limb_velocities(video, params.limb_signal)     # (L, 4, 2)
    idx = {L: i for i, L in enumerate(LIMB_ORDER)}
    F = np.full(len(PAIRS), np.nan)
    R2 = np.full(len(PAIRS), np.nan)
    for p, (a, b, _) in enumerate(PAIRS):
        res = pair_wclrpp(V[:, idx[a], :], V[:, idx[b], :], params)
        F[p] = res["F"]
        R2[p] = res["R2"]
    return F, R2


def wclrpp_dataset(vids, params: WCLRParams = WCLRParams()) -> dict:
    """WCLR-PP for every recording.

    Returns per-pair ``F``/``R2`` matrices ``(n, 6)`` plus the per-infant
    aggregation: whole-body coupling ``mean_F`` (mean over the six pairs), its
    ``spread_F`` (across-pair standard deviation, which separates a whole-body
    pattern from a single-pair one), and the mean strength ``mean_R2``.
    """
    Fmat, R2mat = [], []
    for v in vids:
        F, R2 = wclrpp_recording(v, params)
        Fmat.append(F)
        R2mat.append(R2)
    Fmat = np.asarray(Fmat, float)
    R2mat = np.asarray(R2mat, float)
    with np.errstate(invalid="ignore"):
        agg = {
            "mean_F": Fmat.mean(axis=1),
            "spread_F": Fmat.std(axis=1),
            "mean_R2": _nanmean(R2mat, axis=1),
        }
    return {"F": Fmat, "R2": R2mat, "pairs": pair_names(params.limb_signal),
            "pair_class": list(PAIR_CLASSES),
            "limb_signal": params.limb_signal, **agg}


# ---------------------------------------------------------------------------
# 6. circular-shift surrogate null (per recording, per pair)
# ---------------------------------------------------------------------------
def surrogate_pvalue(v1: np.ndarray, v2: np.ndarray,
                     params: WCLRParams = WCLRParams(),
                     n_surr: int = 1000, seed: int = 0) -> dict:
    """One-sided surrogate p-value for a single limb pair in one recording.

    Circular-shift ``v2`` against ``v1`` by a random wrap offset and recompute
    the symmetric ``F``. The shift destroys coupling while preserving each limb's
    marginal and autocorrelation exactly, so it is a valid null for coupling.
    The M1 branch (A's own autocorrelation) is invariant to shifting B and is
    reused, so each surrogate only refits M2. ``p = (1 + #{F_surr ≥ F_obs})/(B+1)``.
    """
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    obs = pair_wclrpp(v1, v2, params)["F"]

    # Precompute the fixed M1 branch for both directions.
    def prep(vA):
        L = len(vA)
        lo, hi = params.tau_max, L - params.w - params.tau_max
        rows = np.arange(lo, hi + 1)
        k = rows[:, None] + np.arange(params.w)[None, :]
        _, taus, SSR1, TSS = delta_r2_matrix(vA, vA, params.w, params.tau_max,
                                             params.eps)
        return rows, k, taus, SSR1, TSS

    r1, k1, taus1, SSR1_1, TSS1 = prep(v1)     # A = v1, B shifted
    r2, k2, taus2, SSR1_2, TSS2 = prep(v2)     # A = v2, B shifted
    rng = np.random.default_rng(seed)
    L = len(v1)
    ge = 0
    for _ in range(n_surr):
        s = int(rng.integers(1, L))
        v2s = np.roll(v2, s, axis=0)
        v1s = np.roll(v1, s, axis=0)
        m12 = _m2_from_shift(v1, v2s, r1, k1, params.w, taus1, None,
                             SSR1_1, TSS1, params.eps)
        m21 = _m2_from_shift(v2, v1s, r2, k2, params.w, taus2, None,
                             SSR1_2, TSS2, params.eps)
        f12 = peak_pick(np.clip(m12, 0, None), params.c, params.dtau,
                        params.ell_min, params.h)["F"]
        f21 = peak_pick(np.clip(m21, 0, None), params.c, params.dtau,
                        params.ell_min, params.h)["F"]
        if 0.5 * (f12 + f21) >= obs:
            ge += 1
    return {"F_obs": float(obs), "p": (1 + ge) / (n_surr + 1),
            "n_surr": n_surr}
