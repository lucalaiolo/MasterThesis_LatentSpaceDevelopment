"""Nonparametric inference: the whole of the reported procedure, and nothing else.

The sample is small (N = 38), the outcome is rare and imbalanced (6 against 32),
and the features are bounded and skewed, so asymptotic calibration is
inappropriate: every null below is the exact permutation law and every interval
is a resampling one.

**The reported procedure** (METHODS §Inference). Every clinical endpoint is a
single scalar per recording, contrasted between the ``n1 = 6`` abnormal and
``n0 = 32`` normal recordings by :func:`mannwhitney`, which is the whole of it:

* the null is equality of the two distributions, ``H0 : F_X = F_Y``, which makes
  the labels exchangeable and therefore makes the exact permutation law the true
  null law of ``U``;
* pooled mid-ranks give ``U = R1 - n1(n1+1)/2`` and ``AUC = U / (n1 n0)``, the
  probability that a random abnormal recording exceeds a random normal one,
  equal to 1/2 under the null;
* the null law of ``U`` is its distribution over all ``C(38,6) = 2,760,681``
  label assignments, enumerated exactly by dynamic programming over the rank
  multiset, and the two-sided p-value is the exact-null probability of an AUC at
  least as far from 1/2 as the observed one;
* the interval is the percentile interval of a **stratified nonparametric
  bootstrap** (:func:`bootstrap_auc_ci`): ``B = 10,000`` resamples of each group
  independently and at its original size, so the 6/32 allocation is held fixed
  and the interval cannot leave ``[0, 1]``.

Alongside the endpoints, :func:`correlation_table` reports each one's Pearson
and Spearman correlation with occupancy entropy, mean dwell time and log
recording length (METHODS §Correlation analysis). The label enters no fit.

Nothing else is reported and nothing else lives here. There is no multiplicity
correction, no nuisance adjustment, no normal approximation, no power
calculation, and no admission gate: an endpoint is contrasted, its interval is
given, and its correlations with the three nuisances are stated.
"""

from __future__ import annotations

from math import comb

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# the endpoint contrast
# ---------------------------------------------------------------------------
def _auc_from_ranksum(R1, n1, n2):
    return (R1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def exact_ranksum_null(ranks: np.ndarray, n1: int):
    """Exact distribution of the size-``n1`` subset rank-sum, by DP.

    Mid-ranks are multiples of one half, so the ranks are doubled to integers
    and the convolution is carried out on that lattice. Returns ``(sums,
    counts)`` with ``sums`` on the original (undoubled) scale.
    """
    r2 = np.rint(np.asarray(ranks, float) * 2).astype(np.int64)
    top = int(np.sort(r2)[-n1:].sum())
    dp = np.zeros((n1 + 1, top + 1), dtype=np.float64)   # float64: exact to 2^53
    dp[0, 0] = 1.0
    for v in r2:
        v = int(v)
        hi = min(n1, len(r2))
        for j in range(hi, 0, -1):
            if v:
                dp[j, v:] += dp[j - 1, :-v]
            else:
                dp[j] += dp[j - 1]
    counts = dp[n1]
    nz = np.flatnonzero(counts)
    return nz / 2.0, counts[nz]


def mannwhitney(x, y, exact: bool | None = None, n_perm: int = 200_000,
                seed: int = 0, boot: int = 10_000, alpha: float = 0.05) -> dict:
    """The reported endpoint contrast: exact Mann-Whitney U, AUC, bootstrap CI.

    ``x`` is the abnormal group, ``y`` the normal one. Ranks are computed once
    on the pooled sample with mid-ranks for ties and only the labels are
    permuted, which is valid under exchangeability and preserves the tie
    pattern. ``exact`` defaults to enumerating every label assignment whenever
    there are at most 5,000,000 of them, which covers the C(38,6) = 2,760,681
    of this cohort; below that fallback the null is Monte Carlo and the
    returned ``method`` says so.

    Returns ``auc``, the two-sided exact p-value, and ``[auc_lo, auc_hi]``: the
    percentile interval of the stratified bootstrap of
    :func:`bootstrap_auc_ci`. That interval is the only one reported. Recordings
    the endpoint did not score (``NaN``) are dropped, and the returned ``n1`` /
    ``n2`` say how many entered.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return {"auc": np.nan, "rank_biserial": np.nan, "p": np.nan,
                "n1": n1, "n2": n2, "method": "empty"}

    r = stats.rankdata(np.concatenate([x, y]))
    auc = _auc_from_ranksum(r[:n1].sum(), n1, n2)
    obs = abs(auc - 0.5)

    if exact is None:
        exact = comb(n1 + n2, n1) <= 5_000_000
    if exact:
        sums, counts = exact_ranksum_null(r, n1)
        aucs = _auc_from_ranksum(sums, n1, n2)
        total = counts.sum()
        hit = counts[np.abs(aucs - 0.5) >= obs - 1e-12].sum()
        p = float(hit / total)
        method = f"exact enumeration ({int(total):,} assignments)"
    else:
        rng = np.random.default_rng(seed)
        idx = np.argsort(rng.random((n_perm, n1 + n2)), axis=1)[:, :n1]
        aucs = _auc_from_ranksum(r[idx].sum(axis=1), n1, n2)
        p = float((1 + np.sum(np.abs(aucs - 0.5) >= obs - 1e-12)) / (n_perm + 1))
        method = f"Monte Carlo ({n_perm:,} draws)"

    bs = (bootstrap_auc_ci(x, y, n_boot=boot, alpha=alpha, seed=seed) if boot
          else {"lo": np.nan, "hi": np.nan, "n_boot": 0})
    return {"auc": float(auc), "rank_biserial": float(2 * auc - 1), "p": p,
            "n1": n1, "n2": n2, "method": method,
            "auc_lo": bs["lo"], "auc_hi": bs["hi"],
            "ci_method": (f"stratified percentile bootstrap "
                          f"({bs['n_boot']:,} resamples, "
                          f"{100 * (1 - alpha):.0f}%)"),
            "n_boot": bs["n_boot"], "alpha": alpha,
            "U": float(auc * n1 * n2)}


def bootstrap_auc_ci(x, y, n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """The reported AUC interval: stratified nonparametric percentile bootstrap.

    Each group is resampled independently, with replacement, at its original
    size, so the 6/32 allocation is held fixed; the AUC is recomputed on every
    resample and the interval is the ``[alpha/2, 1 - alpha/2]`` quantile pair of
    those replicates. It cannot leave [0, 1] by construction, which a normal
    approximation at n1 = 6 cannot promise.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return {"lo": np.nan, "hi": np.nan, "n_boot": 0}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        xb = x[rng.integers(0, n1, n1)]
        yb = y[rng.integers(0, n2, n2)]
        r = stats.rankdata(np.concatenate([xb, yb]))
        out[b] = _auc_from_ranksum(r[:n1].sum(), n1, n2)
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "n_boot": n_boot,
            "se": float(out.std(ddof=1))}


# ---------------------------------------------------------------------------
# the correlation analysis
# ---------------------------------------------------------------------------
def correlate(a, b) -> dict:
    """Pearson and Spearman correlation of two per-recording vectors.

    Both are reported for every endpoint: Pearson answers "does it move
    linearly with this?", Spearman "does it order the infants the same way?",
    and at n = 38 with bounded, skewed endpoints the two can disagree in ways
    worth seeing. Non-finite entries are dropped pairwise.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    out = {"pearson_r": np.nan, "pearson_p": np.nan,
           "spearman_rho": np.nan, "spearman_p": np.nan, "n": n}
    if n < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return out
    r, pr = stats.pearsonr(a[ok], b[ok])
    rho, ps = stats.spearmanr(a[ok], b[ok])
    out.update({"pearson_r": float(r), "pearson_p": float(pr),
                "spearman_rho": float(rho), "spearman_p": float(ps)})
    return out


def correlation_table(endpoints: dict, covariates: dict) -> dict:
    """Every endpoint against every covariate, Pearson and Spearman.

    ``endpoints`` and ``covariates`` map a name to one value per recording.
    Returns ``{endpoint: {covariate: correlate(...)}}``. The label enters
    nothing here.
    """
    return {e: {c: correlate(v, u) for c, u in covariates.items()}
            for e, v in endpoints.items()}
