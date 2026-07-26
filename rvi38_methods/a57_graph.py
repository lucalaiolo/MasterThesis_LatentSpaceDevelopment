"""A5 metastable decomposition (§8) and A7 mixing structure (§9).

Two chains are analysed throughout:

``A``          the full chain, whose timescales mix dwell with sequencing;
``A_jump``     the embedded jump chain, ``A_kk' / (1 - A_kk)`` off-diagonal with
               zero diagonal, which removes dwell and isolates sequencing.

The jump chain is the object of interest for repertoire structure, because
metastability on ``A`` is dominated by self-transitions and merely restates
dwell time (§8.1).
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# §8.1 basic chain quantities
# ---------------------------------------------------------------------------
def jump_chain(A: np.ndarray) -> np.ndarray:
    """Embedded jump chain: the probability that the next *different* state."""
    J = np.array(A, float, copy=True)
    np.fill_diagonal(J, 0.0)
    r = J.sum(1, keepdims=True)
    return np.divide(J, r, out=np.zeros_like(J), where=r > 0)


def stationary(A: np.ndarray) -> np.ndarray:
    """Left Perron vector, normalised to a probability distribution."""
    w, V = np.linalg.eig(np.asarray(A, float).T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    s = v.sum()
    if abs(s) < 1e-12:
        return np.full(len(A), 1.0 / len(A))
    v = v / s
    return np.clip(v, 1e-15, None) / np.clip(v, 1e-15, None).sum()


def implied_timescales(A: np.ndarray, f_win: float = 6.25, n: int = 8):
    """§8.2 ``t_m = -1 / ln|lambda_m|`` in steps and seconds, with gap ratios.

    ``lambda_1 = 1`` is discarded. A large ``t_m / t_{m+1}`` is the conventional
    criterion for ``m`` metastable sets, so the ratios are returned for
    inspection rather than a single chosen ``m``.
    """
    lam = np.linalg.eigvals(np.asarray(A, float))
    lam = lam[np.argsort(-np.abs(lam))]
    sub = np.abs(lam[1:n + 1])
    t = -1.0 / np.log(np.clip(sub, 1e-12, 1 - 1e-12))
    gaps = t[:-1] / np.clip(t[1:], 1e-30, None)
    return {"eigenvalues": lam, "timescales_steps": t,
            "timescales_seconds": t / f_win, "gap_ratios": gaps,
            "m_at_max_gap": int(np.argmax(gaps) + 2) if len(gaps) else np.nan}


def spectral_gap(A: np.ndarray) -> float:
    lam = np.sort(np.abs(np.linalg.eigvals(np.asarray(A, float))))[::-1]
    return float(1.0 - lam[1]) if len(lam) > 1 else np.nan


# ---------------------------------------------------------------------------
# §9.1 statistics that must be shown to be degenerate before any graph analysis
# ---------------------------------------------------------------------------
def degenerate_centralities(A: np.ndarray) -> dict:
    """Demonstrates the three uninformative statistics of §9.1."""
    A = np.asarray(A, float)
    K = len(A)
    rho = stationary(A)
    w, V = np.linalg.eig(A)
    right = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    right = right / right[np.argmax(np.abs(right))]      # scale-free comparison
    pr = {a: stationary(a * A + (1 - a) * np.ones((K, K)) / K)
          for a in (0.85, 0.95, 0.99)}
    return {
        "out_degree": A.sum(1),
        "out_degree_is_one": bool(np.allclose(A.sum(1), 1.0)),
        "right_perron": right,
        "right_perron_is_constant": bool(np.allclose(right, right[0], atol=1e-8)),
        "stationary": rho,
        "pagerank": pr,
        "pagerank_to_stationary_l1": {
            a: float(np.abs(v - rho).sum()) for a, v in pr.items()},
    }


# ---------------------------------------------------------------------------
# §9.2 fundamental matrix, mean first passage, Kemeny
# ---------------------------------------------------------------------------
def fundamental(A: np.ndarray, rho=None) -> np.ndarray:
    """``Z = (I - A + 1 rho^T)^{-1}``."""
    A = np.asarray(A, float)
    rho = stationary(A) if rho is None else rho
    K = len(A)
    return np.linalg.inv(np.eye(K) - A + np.outer(np.ones(K), rho))


def mfpt(A: np.ndarray, rho=None) -> np.ndarray:
    """Mean first passage ``M[k,k'] = (Z_k'k' - Z_kk') / rho_k'``, ``M_kk = 0``."""
    A = np.asarray(A, float)
    rho = stationary(A) if rho is None else rho
    Z = fundamental(A, rho)
    M = (np.diag(Z)[None, :] - Z) / rho[None, :]
    np.fill_diagonal(M, 0.0)
    return M


def kemeny(A: np.ndarray, rho=None) -> float:
    """Kemeny's constant, excursion convention ``trace(Z) - 1`` (§9.2).

    This excludes the recurrence term; the alternative convention differs by
    exactly one, so the convention is stated wherever the value is reported.
    """
    A = np.asarray(A, float)
    rho = stationary(A) if rho is None else rho
    return float(np.trace(fundamental(A, rho)) - 1.0)


def kemeny_identity_check(A: np.ndarray) -> dict:
    """Verify ``trace(Z) - 1 == sum_{m>=2} 1/(1 - lambda_m)`` (§9.2 proof)."""
    A = np.asarray(A, float)
    lam = np.linalg.eigvals(A)
    lam = lam[np.argsort(-np.abs(lam))][1:]
    spec = float(np.real(np.sum(1.0 / (1.0 - lam))))
    trc = kemeny(A)
    return {"trace_form": trc, "spectral_form": spec,
            "abs_diff": float(abs(trc - spec)),
            "match": bool(abs(trc - spec) < 1e-6 * max(1.0, abs(trc)))}


def graph_companions(A: np.ndarray, f_win: float = 6.25) -> dict:
    """§9.2 dynamical closeness and mean inbound passage, per state."""
    M = mfpt(A)
    K = len(A)
    closeness = 1.0 / np.clip(M.sum(1), 1e-30, None)
    inbound = M.sum(0) / max(K - 1, 1)
    return {"mfpt": M, "closeness": closeness, "mean_inbound": inbound,
            "kemeny": kemeny(A), "kemeny_seconds": kemeny(A) / f_win}


# ---------------------------------------------------------------------------
# §8.3 PCCA+
# ---------------------------------------------------------------------------
def _inner_simplex(X: np.ndarray) -> np.ndarray:
    """Vertex states of the eigenvector simplex (Deuflhard-Weber)."""
    m = X.shape[1]
    idx = np.zeros(m, int)
    Y = np.array(X, float, copy=True)
    idx[0] = int(np.argmax(np.linalg.norm(Y, axis=1)))
    Y = Y - Y[idx[0]]
    for j in range(1, m):
        for i in range(j):
            nv = np.linalg.norm(Y[idx[i]])
            if nv < 1e-12:
                continue
            d = Y[idx[i]] / nv
            Y = Y - np.outer(Y @ d, d)
        idx[j] = int(np.argmax(np.linalg.norm(Y, axis=1)))
    return idx


def pcca(A: np.ndarray, m: int):
    """§8.3 PCCA+ fuzzy memberships ``chi`` of shape ``(K, m)``."""
    A = np.asarray(A, float)
    w, V = np.linalg.eig(A)
    order = np.argsort(-np.real(w))
    X = np.real(V[:, order[:m]])
    first = X[:, [0]]
    first = np.where(np.abs(first) < 1e-12, 1e-12, first)
    X = X / first
    idx = _inner_simplex(X)
    try:
        inv = np.linalg.inv(X[idx])
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(X[idx])
    chi = np.clip(X @ inv, 0, None)
    s = chi.sum(1, keepdims=True)
    chi = np.divide(chi, s, out=np.full_like(chi, 1.0 / m), where=s > 1e-12)
    return chi, idx


def crispness(chi: np.ndarray) -> float:
    """Mean maximum membership; 1.0 is a hard partition (§8.3)."""
    return float(chi.max(1).mean())


def metastability(A: np.ndarray, assign: np.ndarray, rho=None) -> float:
    """§8.3 occupancy-weighted within-set transition mass; upper bound is ``m``."""
    A = np.asarray(A, float)
    rho = stationary(A) if rho is None else rho
    tot = 0.0
    for s in np.unique(assign):
        msk = assign == s
        w = rho[msk] / max(rho[msk].sum(), 1e-30)
        tot += float(w @ A[np.ix_(msk, msk)].sum(1))
    return tot


def spectral_kmeans(A: np.ndarray, m: int, seed: int = 0, n_init: int = 50):
    """§8.3 control: k-means on the same eigenvector embedding.

    If it recovers the PCCA+ partition, the result is a property of the
    embedding rather than of the PCCA+ construction.
    """
    from sklearn.cluster import KMeans
    A = np.asarray(A, float)
    w, V = np.linalg.eig(A)
    order = np.argsort(-np.real(w))
    lam = np.real(w[order[:m]])
    X = np.real(V[:, order[:m]])
    nrm = np.linalg.norm(X, axis=0, keepdims=True)
    X = X / np.where(nrm < 1e-12, 1.0, nrm)
    E = X[:, 1:] * lam[1:]
    if E.shape[1] == 0:
        return np.zeros(len(A), int)
    return KMeans(m, n_init=n_init, random_state=seed).fit_predict(E)


# ---------------------------------------------------------------------------
# §8.4 the independent kinematic partition
# ---------------------------------------------------------------------------
def kinematic_partition(S: np.ndarray, m: int) -> np.ndarray:
    """Average-linkage clustering of ``D = clip(1 - S, 0, 2)`` into ``m`` sets.

    Average linkage is invariant to monotone transformations of the distances
    and does not impose Ward's compactness assumption. This partition sees no
    transition information whatsoever, which is what makes its agreement with
    the metastable partition informative (§8.4).
    """
    D = np.clip(1.0 - np.asarray(S, float), 0, 2)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, m, "maxclust")


def kinematic_linkage(S: np.ndarray):
    D = np.clip(1.0 - np.asarray(S, float), 0, 2)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    return linkage(squareform(D, checks=False), method="average")


# ---------------------------------------------------------------------------
# §9.3 per-subject estimation: shrinkage and the moving block bootstrap
# ---------------------------------------------------------------------------
def counts_from_path(seq: np.ndarray, K: int) -> np.ndarray:
    C = np.zeros((K, K))
    if len(seq) > 1:
        np.add.at(C, (seq[:-1], seq[1:]), 1.0)
    return C


def row_normalise(C: np.ndarray) -> np.ndarray:
    r = C.sum(1, keepdims=True)
    return np.divide(C, r, out=np.full_like(C, 1.0 / C.shape[0]), where=r > 0)


def shrink(C: np.ndarray, Abar: np.ndarray, alpha: float) -> np.ndarray:
    """``(1 - alpha) * per-subject estimate + alpha * group matrix`` (§9.3)."""
    return (1 - alpha) * row_normalise(C) + alpha * Abar


def choose_alpha(seqs, K: int, grid=None, train_frac: float = 0.7) -> dict:
    """Select shrinkage by held-out predictive likelihood (§9.3).

    Fit on the first 70% of each subject's visit sequence, score the remaining
    30%, and take the ``alpha`` maximising the summed held-out log-likelihood.
    The group matrix is built once from the training portions; recomputing it
    per subject inside the loop (as a naive implementation does) is quadratic
    and changes nothing.
    """
    grid = np.linspace(0, 0.95, 20) if grid is None else np.asarray(grid)
    train = [np.asarray(s)[:int(train_frac * len(s))] for s in seqs]
    test = [np.asarray(s)[int(train_frac * len(s)):] for s in seqs]
    Abar = row_normalise(sum(counts_from_path(t, K) for t in train))
    lls = []
    for a in grid:
        ll = 0.0
        for tr, te in zip(train, test):
            if len(te) < 2:
                continue
            P = np.clip(shrink(counts_from_path(tr, K), Abar, a), 1e-12, None)
            P /= P.sum(1, keepdims=True)
            ll += float(np.log(P[te[:-1], te[1:]]).sum())
        lls.append(ll)
    lls = np.array(lls)
    best = int(np.argmax(lls))
    # Tolerance matters: linspace(0, 0.95, 20) lands on 0.8999999999999999, so
    # a bare `>= 0.9` would report a maximally shrunk fit as informative.
    return {"alpha": float(grid[best]), "loglik": float(lls[best]),
            "grid": grid, "loglik_grid": lls,
            "group_matrix": Abar,
            "degenerate": bool(grid[best] >= 0.9 - 1e-9)}


def moving_block_bootstrap(seq: np.ndarray, block: int, rng) -> np.ndarray:
    """One moving-block resample preserving short-range dependence (§10.9)."""
    seq = np.asarray(seq)
    n = len(seq)
    if n <= block:
        return seq.copy()
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, nb)
    return np.concatenate([seq[t:t + block] for t in starts])[:n]


def subject_kemeny(window_path: np.ndarray, K: int, Abar_jump: np.ndarray,
                   alpha: float, use_jump: bool = True) -> float:
    """Kemeny for one subject from a (possibly resampled) window path.

    With ``use_jump`` the visit sequence is extracted first, so the estimate is
    in *jumps* and dwell does not enter; otherwise the full window chain is used
    and the value is in windows.
    """
    from a1_core import visit_sequence
    seq = visit_sequence(np.asarray(window_path)) if use_jump else np.asarray(
        window_path)
    if len(seq) < 3:
        return np.nan
    P = shrink(counts_from_path(seq, K), Abar_jump, alpha)
    if use_jump:
        np.fill_diagonal(P, 0.0)
        r = P.sum(1, keepdims=True)
        P = np.divide(P, r, out=np.full_like(P, 1.0 / K), where=r > 0)
    try:
        return kemeny(P)
    except np.linalg.LinAlgError:
        return np.nan


def block_bootstrap_kemeny(window_path, K, Abar_jump, alpha, block: int = 50,
                           n_boot: int = 400, seed: int = 0,
                           use_jump: bool = True) -> np.ndarray:
    """Percentile-interval draws for one subject's Kemeny constant (§9.3).

    Blocks of 50 windows (8 s at 6.25 Hz) preserve the autocorrelation of the
    state path; an i.i.d. bootstrap would be invalid because consecutive windows
    are strongly dependent.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = moving_block_bootstrap(window_path, block, rng)
        out[b] = subject_kemeny(s, K, Abar_jump, alpha, use_jump)
    return out


def estimability_gate(values, lo, hi) -> dict:
    """§9.3 / §11: median interval width against the between-subject range.

    A ratio above one half means per-subject values are not resolvable and only
    group-level results should be reported.
    """
    v = np.asarray(values, float)
    w = np.asarray(hi, float) - np.asarray(lo, float)
    ok = np.isfinite(v) & np.isfinite(w)
    if ok.sum() < 3:
        return {"ratio": np.nan, "passed": False, "n": int(ok.sum())}
    med_w = float(np.median(w[ok]))
    rng_v = float(np.nanmax(v[ok]) - np.nanmin(v[ok]))
    ratio = med_w / rng_v if rng_v > 0 else np.inf
    return {"median_width": med_w, "between_subject_range": rng_v,
            "ratio": float(ratio), "passed": bool(ratio < 0.5),
            "n": int(ok.sum())}
