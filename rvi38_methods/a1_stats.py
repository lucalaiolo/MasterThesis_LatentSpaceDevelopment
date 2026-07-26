"""Nonparametric inference (METHODS §10).

Everything here is permutation- or resampling-based, as §10 requires: the
sample is small (N = 38), the outcome is rare and imbalanced (6 against 32),
and the features are bounded and skewed, so asymptotic calibration is
inappropriate.

:func:`mannwhitney` performs **exact enumeration** of all ``C(38,6) =
2,760,681`` label assignments by dynamic programming over the rank multiset
(§10.1 notes this is feasible; enumerating it removes Monte Carlo error from the
headline contrasts entirely). Monte Carlo is retained for the statistics where
enumeration does not apply.
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
                seed: int = 0) -> dict:
    """Permutation Mann-Whitney with AUC and rank-biserial effect size.

    Ranks are computed once on the pooled sample and only the labels are
    permuted, which is valid under exchangeability and preserves the tie
    pattern (§10.2). ``exact`` defaults to enumeration whenever the label space
    is at most 5,000,000 assignments.
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

    lo, hi = hanley_mcneil_ci(auc, n1, n2)
    return {"auc": float(auc), "rank_biserial": float(2 * auc - 1), "p": p,
            "n1": n1, "n2": n2, "method": method, "auc_lo": lo, "auc_hi": hi,
            "U": float(auc * n1 * n2)}


def hanley_mcneil_se(auc: float, n1: int, n2: int) -> float:
    """§10.2 / §10.12 distribution-free SE of the AUC (Hanley & McNeil, 1982)."""
    a = min(max(auc, 1e-9), 1 - 1e-9)
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a))
    return float(np.sqrt(max(var, 0.0) / (n1 * n2)))


def hanley_mcneil_ci(auc, n1, n2, alpha=0.05):
    se = hanley_mcneil_se(auc, n1, n2)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(np.clip(auc - z * se, 0, 1)), float(np.clip(auc + z * se, 0, 1))


def min_detectable_effect(n1: int, n2: int, alpha: float = 0.05,
                          power: float = 0.80) -> dict:
    """§10.12 smallest AUC reaching ``power`` at ``alpha``, at this allocation."""
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
    """Holm (1979) step-down adjusted p-values."""
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
    """§10.7 maxT family-wise correction over a declared family.

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
    """§10.3 Freedman-Lane adjustment for a between-subject nuisance.

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
# §10.5 matrix association
# ---------------------------------------------------------------------------
def mantel(A: np.ndarray, B: np.ndarray, n_perm: int = 20_000,
           seed: int = 0) -> dict:
    """§10.5 Mantel test: correlation of off-diagonal entries, states permuted
    jointly on rows and columns."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    K = len(A)
    off = ~np.eye(K, dtype=bool)
    ok = np.isfinite(A) & np.isfinite(B) & off
    r_obs = float(np.corrcoef(A[ok], B[ok])[0, 1])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(K)
        Bp = B[np.ix_(p, p)]
        m = np.isfinite(A) & np.isfinite(Bp) & off
        r = np.corrcoef(A[m], Bp[m])[0, 1]
        cnt += abs(r) >= abs(r_obs) - 1e-12
    return {"r": r_obs, "p": float((1 + cnt) / (n_perm + 1)), "n_perm": n_perm}


def partial_mantel(A, B, C, n_perm: int = 20_000, seed: int = 0) -> dict:
    """Mantel statistic on matrices residualised against a third (§10.5)."""
    def resid(M, Cm, mask):
        X = np.column_stack([np.ones(mask.sum()), Cm[mask]])
        b, *_ = np.linalg.lstsq(X, M[mask], rcond=None)
        return M[mask] - X @ b
    A, B, C = map(lambda M: np.asarray(M, float), (A, B, C))
    K = len(A)
    off = ~np.eye(K, dtype=bool)
    ra, rb = resid(A, C, off), resid(B, C, off)
    r_obs = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(K)
        rbp = resid(B[np.ix_(p, p)], C, off)
        cnt += abs(np.corrcoef(ra, rbp)[0, 1]) >= abs(r_obs) - 1e-12
    return {"r": r_obs, "p": float((1 + cnt) / (n_perm + 1))}


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
