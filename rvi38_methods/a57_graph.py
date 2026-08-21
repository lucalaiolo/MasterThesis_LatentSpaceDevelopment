"""A7 mixing structure (§9): chain quantities, passage times, Kemeny.

Two chains are analysed throughout:

``A``          the full chain, whose behaviour mixes dwell with sequencing;
``A_jump``     the embedded jump chain, ``A_kk' / (1 - A_kk)`` off-diagonal with
               zero diagonal, which removes dwell and isolates sequencing.

The jump chain is the object of interest for repertoire structure, because any
statistic of ``A`` is dominated by self-transitions and merely restates dwell
time.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# basic chain quantities
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
