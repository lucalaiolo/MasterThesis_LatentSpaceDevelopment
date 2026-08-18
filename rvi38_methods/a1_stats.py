"""Nonparametric inference.

Everything here is permutation- or resampling-based: the sample is small
(N = 38), the outcome is rare and imbalanced (6 against 32), and the features
are bounded and skewed, so asymptotic calibration is inappropriate.

**The reported procedure.** Every clinical endpoint is a single scalar per
recording, contrasted between the n1 = 6 abnormal and n0 = 32 normal recordings
by :func:`mannwhitney`, which is the whole of it:

* the null is equality of the two distributions, which makes the labels
  exchangeable, so the exact permutation law is the true null law of ``U``;
* pooled mid-ranks give ``U = R1 - n1(n1+1)/2`` and
  ``AUC = U / (n1 n0)``, the probability that a random abnormal recording
  exceeds a random normal one, equal to 1/2 under the null;
* the null law of ``U`` is its distribution over all ``C(38,6) = 2,760,681``
  label assignments, enumerated exactly by dynamic programming over the rank
  multiset, and the two-sided p-value is the exact-null probability of an AUC
  at least as far from 1/2 as the observed one;
* the interval is the percentile interval of a **stratified nonparametric
  bootstrap** (:func:`bootstrap_auc_ci`): B = 10,000 resamples of each group
  independently and at its original size, so the 6/32 allocation is held fixed
  and the interval cannot leave [0, 1].

Alongside the endpoints, :func:`correlation_table` reports each one's Pearson
and Spearman correlation with occupancy entropy, mean dwell time and log
recording length. The label enters no fit.

Functions below the endpoint machinery -- multiplicity corrections, the
Hanley-McNeil normal approximation, the minimum detectable effect, the
Freedman-Lane nuisance adjustment -- are **not part of the reported procedure**
and nothing in the pipeline calls them. They are kept because they were used
while the analysis was being developed and are cheap to keep correct; see the
note above each.
"""

from __future__ import annotations

from math import comb

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# §10.2 two-group comparison
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
    :func:`bootstrap_auc_ci`. That interval is the only one reported. The
    Hanley-McNeil normal approximation is not used: at n1 = 6 it routinely runs
    past 1 and has to be clipped, which is not an interval so much as an
    apology for one.
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


# ---------------------------------------------------------------------------
# Retained, not reported. Nothing in the pipeline calls anything from here to
# the end of this block: the reported interval is the stratified bootstrap
# above, the reported p-value is the exact permutation one, and neither
# multiplicity correction nor nuisance adjustment nor a power calculation is
# part of the procedure. Kept because they were used while the analysis was
# being developed, and are cheap to keep correct.
# ---------------------------------------------------------------------------
def hanley_mcneil_se(auc: float, n1: int, n2: int) -> float:
    """Distribution-free SE of the AUC (Hanley & McNeil, 1982). Not reported."""
    a = min(max(auc, 1e-9), 1 - 1e-9)
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a))
    return float(np.sqrt(max(var, 0.0) / (n1 * n2)))


def hanley_mcneil_ci(auc, n1, n2, alpha=0.05):
    se = hanley_mcneil_se(auc, n1, n2)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(np.clip(auc - z * se, 0, 1)), float(np.clip(auc + z * se, 0, 1))


def bootstrap_auc_ci(x, y, n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """The reported AUC interval: stratified nonparametric percentile bootstrap.

    Each group is resampled independently, with replacement, at its original
    size, so the 6/32 allocation is held fixed; the AUC is recomputed on every
    resample and the interval is the ``[alpha/2, 1 - alpha/2]`` quantile pair of
    those replicates. It cannot leave [0, 1] by construction, which the normal
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


def min_detectable_effect(n1: int, n2: int, alpha: float = 0.05,
                          power: float = 0.80) -> dict:
    """Smallest AUC reaching ``power`` at ``alpha``, at this allocation.

    Not reported: post-hoc power is a deterministic function of the p-value
    (Hoenig & Heisey, 2001) and the design power adds nothing the interval does
    not already say.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    se0 = hanley_mcneil_se(0.5, n1, n2)
    grid = np.linspace(0.5 + 1e-4, 0.9999, 20_000)
    for a in grid:
        if (a - 0.5) >= z_a * se0 + z_b * hanley_mcneil_se(a, n1, n2):
            return {"auc": float(a), "rank_biserial": float(2 * a - 1),
                    "alpha": alpha, "power": power, "n1": n1, "n2": n2}
    return {"auc": np.nan, "alpha": alpha, "power": power, "n1": n1, "n2": n2}


def required_n(auc: float, ratio: float, alpha: float = 0.05,
               power: float = 0.80, n_max: int = 100_000) -> int:
    """Smallest total N reaching ``power`` for a true ``auc`` at ``ratio``
    positives (§10.12)."""
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    n = 4
    while n < n_max:
        n1 = max(int(round(n * ratio)), 1)
        n2 = n - n1
        if n2 >= 1:
            if (auc - 0.5) >= (z_a * hanley_mcneil_se(0.5, n1, n2)
                               + z_b * hanley_mcneil_se(auc, n1, n2)):
                return n
        n += 1
    return -1


# ---------------------------------------------------------------------------
# §10.7 multiplicity
# ---------------------------------------------------------------------------
def holm(pvals) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values. Not reported."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def maxt(features: dict, labels, n_perm: int = 200_000, seed: int = 0) -> dict:
    """maxT family-wise correction over a declared family. Not reported.

    The null maximum of ``|AUC - 0.5|`` across the family is accumulated inside
    the same permutation loop that produces the marginal nulls, so the
    correction exploits the observed correlation between the members and is
    less conservative than assuming independence.
    """
    labels = np.asarray(labels).astype(bool)
    names = list(features)
    rng = np.random.default_rng(seed)
    n = len(labels)
    n1 = int(labels.sum())
    n2 = n - n1

    obs, ranks = {}, {}
    for k in names:
        v = np.asarray(features[k], float)
        ok = np.isfinite(v)
        r = np.full(n, np.nan)
        r[ok] = stats.rankdata(v[ok])
        ranks[k] = r
        obs[k] = abs(_auc_from_ranksum(np.nansum(r[labels & ok]),
                                       int((labels & ok).sum()),
                                       int(((~labels) & ok).sum())) - 0.5)

    idx = np.argsort(rng.random((n_perm, n)), axis=1)[:, :n1]
    null_max = np.zeros(n_perm)
    marg = {}
    for k in names:
        r = ranks[k]
        if np.isnan(r).any():                      # fall back to complete cases
            ok = np.isfinite(r)
            rr = stats.rankdata(np.asarray(features[k], float)[ok])
            m1 = int((labels & ok).sum())
            m2 = int(ok.sum()) - m1
            sub = np.argsort(rng.random((n_perm, int(ok.sum()))), axis=1)[:, :m1]
            a = np.abs(_auc_from_ranksum(rr[sub].sum(1), m1, m2) - 0.5)
        else:
            a = np.abs(_auc_from_ranksum(r[idx].sum(1), n1, n2) - 0.5)
        marg[k] = float((1 + np.sum(a >= obs[k] - 1e-12)) / (n_perm + 1))
        null_max = np.maximum(null_max, a)

    return {k: {"p_marginal": marg[k],
                "p_maxT": float((1 + np.sum(null_max >= obs[k] - 1e-12))
                                / (n_perm + 1)),
                "statistic": float(obs[k])} for k in names}


# ---------------------------------------------------------------------------
# §10.3 nuisance adjustment
# ---------------------------------------------------------------------------
def freedman_lane(y, labels, nuisance, n_perm: int = 200_000, seed: int = 0):
    """Freedman-Lane adjustment for a between-subject nuisance. Not reported:
    the nuisances are reported as correlations, and the label enters no fit.

    ``y`` is regressed on the nuisance, the residuals carry the tested signal,
    and the label permutation is applied to them, so the nuisance structure is
    held fixed and only the label-residual association is tested.
    """
    y = np.asarray(y, float)
    labels = np.asarray(labels).astype(bool)
    Xn = np.asarray(nuisance, float)
    if Xn.ndim == 1:
        Xn = Xn[:, None]
    ok = np.isfinite(y) & np.isfinite(Xn).all(1)
    y, Xn, labels = y[ok], Xn[ok], labels[ok]
    X = np.column_stack([np.ones(len(y)), Xn])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r = mannwhitney(resid[labels], resid[~labels], n_perm=n_perm, seed=seed)
    r["nuisance_beta"] = beta[1:].tolist()
    r["n_used"] = int(ok.sum())
    return r


# ---------------------------------------------------------------------------
# §10.8 reliability
# ---------------------------------------------------------------------------
def spearman_brown(r_half: float) -> float:
    if not np.isfinite(r_half):
        return np.nan
    denom = 1 + r_half
    return float(2 * r_half / denom) if abs(denom) > 1e-12 else np.nan


def split_half(first, second, n_boot: int = 2000, seed: int = 0) -> dict:
    """§10.8 contiguous split-half reliability with the Spearman-Brown step-up.

    Halves are contiguous rather than interleaved: interleaved halves share
    local temporal context and inflate the estimate.
    """
    a, b = np.asarray(first, float), np.asarray(second, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return {"r_half": np.nan, "r_sb": np.nan, "n": int(ok.sum())}
    rho, p = stats.spearmanr(a[ok], b[ok])
    rng = np.random.default_rng(seed)
    idx = np.arange(int(ok.sum()))
    boot = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(s)) < 3:
            continue
        rr, _ = stats.spearmanr(a[ok][s], b[ok][s])
        boot.append(spearman_brown(rr))
    boot = np.array([v for v in boot if np.isfinite(v)])
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 10
              else (np.nan, np.nan))
    return {"r_half": float(rho), "p_half": float(p),
            "r_sb": spearman_brown(rho), "r_sb_lo": float(lo),
            "r_sb_hi": float(hi), "n": int(ok.sum()),
            "icc_a1": icc_a1(np.column_stack([a[ok], b[ok]]))}


def icc_a1(M: np.ndarray) -> float:
    """§10.8 two-way ICC(A,1), single measure, absolute agreement."""
    M = np.asarray(M, float)
    n, k = M.shape
    if n < 2 or k < 2:
        return np.nan
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2
           ).sum() / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if abs(denom) > 1e-12 else np.nan


# ---------------------------------------------------------------------------
# §10.9 confidence intervals
# ---------------------------------------------------------------------------
def bca_ci(v, fn=np.median, n_boot: int = 10_000, alpha: float = 0.05,
           seed: int = 0) -> dict:
    """§10.9 bias-corrected and accelerated bootstrap interval."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "n": n}
    theta = float(fn(v))
    rng = np.random.default_rng(seed)
    boot = np.array([fn(v[rng.integers(0, n, n)]) for _ in range(n_boot)])
    prop = np.mean(boot < theta)
    z0 = stats.norm.ppf(np.clip(prop, 1e-9, 1 - 1e-9))
    jack = np.array([fn(np.delete(v, i)) for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    acc = num / den if abs(den) > 1e-30 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = stats.norm.ppf(q)
        adj = z0 + (z0 + z) / max(1 - acc * (z0 + z), 1e-9)
        out.append(float(np.percentile(boot, 100 * stats.norm.cdf(adj))))
    return {"estimate": theta, "lo": out[0], "hi": out[1], "n": n,
            "z0": float(z0), "acceleration": float(acc)}


# ---------------------------------------------------------------------------
# §10.4 / §10.6 one-sample and partition agreement
# ---------------------------------------------------------------------------
def wilcoxon_signed(v, alternative: str = "greater") -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or np.allclose(v, 0):
        return {"p": np.nan, "n": len(v), "n_positive": int(np.sum(v > 0))}
    r = stats.wilcoxon(v, alternative=alternative)
    return {"statistic": float(r.statistic), "p": float(r.pvalue),
            "n": len(v), "n_positive": int(np.sum(v > 0)),
            "median": float(np.median(v))}


def partition_agreement(a, b, n_perm: int = 200_000, seed: int = 0) -> dict:
    """§10.6 ARI and AMI with a label-permutation null on one partition."""
    from sklearn.metrics import (adjusted_mutual_info_score,
                                 adjusted_rand_score)
    a, b = np.asarray(a), np.asarray(b)
    ari = float(adjusted_rand_score(a, b))
    ami = float(adjusted_mutual_info_score(a, b))
    rng = np.random.default_rng(seed)
    null_ari = np.empty(n_perm)
    for i in range(n_perm):
        null_ari[i] = adjusted_rand_score(a, rng.permutation(b))
    p = float((1 + np.sum(null_ari >= ari - 1e-12)) / (n_perm + 1))
    return {"ari": ari, "ami": ami, "p_ari": p, "n_perm": n_perm,
            "null_mean": float(null_ari.mean()),
            "null_p95": float(np.percentile(null_ari, 95))}


# ---------------------------------------------------------------------------
# §2.3 / §11 duration controls
# ---------------------------------------------------------------------------
def correlate(a, b) -> dict:
    """Pearson and Spearman correlation of two per-recording vectors.

    Both are reported for every endpoint: Pearson answers "does it move
    linearly with this?", Spearman "does it order the infants the same way?",
    and at n = 38 with bounded, skewed endpoints the two can disagree in ways
    worth seeing.
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


def duration_control(feature, log_len) -> dict:
    """Spearman correlation of a feature with ``log L_i`` (§2.3 control 1)."""
    f, L = np.asarray(feature, float), np.asarray(log_len, float)
    ok = np.isfinite(f) & np.isfinite(L)
    if ok.sum() < 4:
        return {"rho": np.nan, "p": np.nan, "n": int(ok.sum())}
    rho, p = stats.spearmanr(f[ok], L[ok])
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


def ordering_stability(full, truncated) -> dict:
    """§2.3 control 2: subject ordering under truncation to a common budget."""
    a, b = np.asarray(full, float), np.asarray(truncated, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return {"rho": np.nan, "p": np.nan, "n": int(ok.sum())}
    rho, p = stats.spearmanr(a[ok], b[ok])
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


def loo_auc(x, y, exact: bool = True) -> list:
    """Leave-one-out AUC and p, dropping each subject in turn."""
    x, y = list(np.asarray(x, float)), list(np.asarray(y, float))
    out = []
    for i in range(len(x)):
        r = mannwhitney(x[:i] + x[i + 1:], y, exact=exact)
        out.append({"dropped": "pos", "index": i, "auc": r["auc"], "p": r["p"]})
    for i in range(len(y)):
        r = mannwhitney(x, y[:i] + y[i + 1:], exact=exact)
        out.append({"dropped": "neg", "index": i, "auc": r["auc"], "p": r["p"]})
    return out


# ---------------------------------------------------------------------------
# family of features tested together: Westfall-Young maximum-statistic
# ---------------------------------------------------------------------------
def median_contrast(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Median difference (positive − negative), one value per column."""
    with np.errstate(invalid="ignore"):
        return np.nanmedian(M[y == 1], axis=0) - np.nanmedian(M[y == 0], axis=0)


def maxstat_label_test(M, labels, n_perm: int = 10_000, seed: int = 0,
                       names=None, contrast=median_contrast) -> dict:
    """Label-permutation test over a family of features, max-statistic corrected.

    ``M`` is ``(n_subjects, n_features)`` and the recording is the exchangeable
    unit, so only the labels are permuted. Family-wise error across the columns
    is controlled by the Westfall-Young maximum-statistic procedure, and **every
    column is reported whatever any one of them shows**. The columns must share
    a scale (the maximum is taken without standardisation), which holds for the
    families this package tests: coupled-time fractions and normalised
    entropies both live on ``[0, 1]``.
    """
    M = np.asarray(M, float)
    y = np.asarray(labels).astype(int)
    if len(y) != len(M):
        raise ValueError(f"{len(M)} recordings but {len(y)} labels")
    n, n1 = len(y), int(y.sum())
    if n1 == 0 or n1 == n:
        raise ValueError("labels are degenerate (one group is empty)")
    observed = contrast(M, y)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, M.shape[1]))
    for b in range(n_perm):
        yp = np.zeros(n, int)
        yp[rng.choice(n, n1, replace=False)] = 1
        null[b] = contrast(M, yp)
    null_max = np.nanmax(np.abs(null), axis=1)
    a = np.abs(observed)
    return {"observed": observed,
            "p_corrected": (1 + (null_max[:, None] >= a[None, :]).sum(0)) / (1 + n_perm),
            "p_uncorrected": (1 + (np.abs(null) >= a[None, :]).sum(0)) / (1 + n_perm),
            "null_max": null_max, "n_perm": n_perm,
            "names": None if names is None else tuple(names),
            "n_pos": n1, "n_neg": n - n1}
