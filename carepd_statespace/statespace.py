"""State-space models for the CARE-PD pipeline ([guideline §4]).

A self-contained NumPy **ARHMM** (autoregressive hidden Markov model) fit by
EM with log-space forward-backward and AR(L) Gaussian emissions, plus a
Gaussian mixture (no progression) and a plain Gaussian HMM as the ``L=0``
special case — the three model families the paper compares.

The reference uses the ``ssm`` library; it is unmaintained and fussy to
build (needs Cython + ``numpy<1.24``), and ``dynamax`` pulls in JAX, so this
module implements the ARHMM directly. The data here is small (~5 h at
15 Hz), so NumPy EM is fast enough and fully controllable — which matters
for the label-alignment averaging ([guideline §4.4]).

Emission for state ``k``:  ``x_t ~ N(b_k + sum_{l=1}^L A_k^(l) x_{t-l}, Q_k)``
for ``t >= L``; the first ``L`` frames use a fallback ``N(mu0_k, Q_k)``.
``L = 0`` recovers a plain Gaussian HMM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


# ---- Gaussian mixture (no transitions, [guideline §4.1]) ------------------

def fit_gmm(seqs: list[np.ndarray], K: int, seed: int = 0):
    from sklearn.mixture import GaussianMixture
    X = np.concatenate(seqs, axis=0)
    gm = GaussianMixture(K, covariance_type="full", random_state=seed,
                         reg_covar=1e-4).fit(X)
    return gm


def gmm_loglik_per_frame(gm, seqs: list[np.ndarray]) -> float:
    X = np.concatenate(seqs, axis=0)
    return float(gm.score(X))            # mean log-likelihood per frame


# ---- ARHMM ([guideline §4.1-§4.6]) ----------------------------------------

@dataclass
class ARHMMParams:
    pi: np.ndarray            # (K,)
    logP: np.ndarray          # (K, K) log transition
    W: np.ndarray             # (K, L*d+1, d) AR weights [A^1..A^L, b]
    Q_inv: np.ndarray         # (K, d, d)
    Q_logdet: np.ndarray      # (K,)
    mu0: np.ndarray           # (K, d) fallback mean for t<L
    K: int
    L: int
    d: int


def _design(x: np.ndarray, L: int) -> np.ndarray:
    """Lagged design [x_{t-1},...,x_{t-L}, 1] for t in [L, T); (T-L, L*d+1)."""
    T, d = x.shape
    if L == 0:
        return np.ones((T, 1))
    rows = [x[L - l - 1:T - l - 1] for l in range(L)]   # x_{t-1}..x_{t-L}
    Phi = np.concatenate(rows, axis=1)                   # (T-L, L*d)
    return np.concatenate([Phi, np.ones((T - L, 1))], axis=1)


def _emission_ll(x: np.ndarray, p: ARHMMParams) -> np.ndarray:
    """Per-frame per-state emission log-likelihood; (T, K)."""
    T, d, L, K = len(x), p.d, p.L, p.K
    ll = np.zeros((T, K))
    const = -0.5 * d * np.log(2 * np.pi)
    # t < L: fallback N(mu0_k, Q_k)
    for t in range(min(L, T)):
        for k in range(K):
            r = x[t] - p.mu0[k]
            ll[t, k] = const - 0.5 * p.Q_logdet[k] - 0.5 * r @ p.Q_inv[k] @ r
    if T <= L:
        return ll
    Phi = _design(x, L)                                  # (T-L, Ld+1)
    tgt = x[L:]                                           # (T-L, d)
    for k in range(K):
        pred = Phi @ p.W[k]                              # (T-L, d)
        r = tgt - pred
        quad = np.einsum("ti,ij,tj->t", r, p.Q_inv[k], r)
        ll[L:, k] = const - 0.5 * p.Q_logdet[k] - 0.5 * quad
    return ll


def _forward_backward(ll: np.ndarray, logpi: np.ndarray, logP: np.ndarray):
    T, K = ll.shape
    la = np.zeros((T, K))
    la[0] = logpi + ll[0]
    for t in range(1, T):
        la[t] = ll[t] + logsumexp(la[t - 1][:, None] + logP, axis=0)
    lb = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(logP + ll[t + 1] + lb[t + 1], axis=1)
    loglik = logsumexp(la[-1])
    gamma = np.exp(la + lb - loglik)
    # xi summed over t: (K,K)
    xi = np.zeros((K, K))
    for t in range(T - 1):
        m = (la[t][:, None] + logP + (ll[t + 1] + lb[t + 1])[None, :] - loglik)
        xi += np.exp(m)
    return gamma, xi, loglik


def _init_params(seqs, K, L, seed, cov_floor=1e-3):
    from sklearn.cluster import KMeans
    X = np.concatenate(seqs, axis=0)
    d = X.shape[1]
    labels = KMeans(K, random_state=seed, n_init=5).fit_predict(X)
    offset = 0
    seq_labels = []
    for s in seqs:
        seq_labels.append(labels[offset:offset + len(s)])
        offset += len(s)
    p = _m_step(seqs, [_onehot(sl, K) for sl in seq_labels], K, L, d, cov_floor)
    return p


def _onehot(labels, K):
    g = np.zeros((len(labels), K))
    g[np.arange(len(labels)), labels] = 1.0
    return g


def _m_step(seqs, gammas, K, L, d, cov_floor=1e-3, ridge=1e-3):
    pi = np.zeros(K)
    trans = np.full((K, K), 1e-6)
    Sxx = [np.zeros((L * d + 1, L * d + 1)) for _ in range(K)]
    Sxy = [np.zeros((L * d + 1, d)) for _ in range(K)]
    wsum = np.zeros(K)
    mu0_num = np.zeros((K, d))
    mu0_den = np.zeros(K)
    resid = [[] for _ in range(K)]      # store weighted residual moments lazily

    # First pass: accumulate transition + AR sufficient stats.
    for x, g in zip(seqs, gammas):
        pi += g[0]
        Phi = _design(x, L)
        tgt = x[L:]
        w = g[L:]                        # (T-L, K)
        for k in range(K):
            wk = w[:, k]
            Sxx[k] += (Phi * wk[:, None]).T @ Phi
            Sxy[k] += (Phi * wk[:, None]).T @ tgt
            wsum[k] += wk.sum()
        for t in range(min(L, len(x))):
            mu0_num += g[t][:, None] * x[t]
            mu0_den += g[t]
        # transitions via pairwise responsibilities (approx from gamma is
        # insufficient; recompute in EM step where xi is available).
    W = np.zeros((K, L * d + 1, d))
    Q = np.zeros((K, d, d))
    mu0 = np.zeros((K, d))
    for k in range(K):
        A = Sxx[k] + ridge * np.eye(L * d + 1)
        W[k] = np.linalg.solve(A, Sxy[k])
        mu0[k] = (mu0_num[k] / mu0_den[k] if mu0_den[k] > 1e-8
                  else W[k][-1])
    # Second pass: residual covariance per state.
    for x, g in zip(seqs, gammas):
        Phi = _design(x, L)
        tgt = x[L:]
        w = g[L:]
        for k in range(K):
            r = tgt - Phi @ W[k]
            Q[k] += (r * w[:, k][:, None]).T @ r
    for k in range(K):
        Q[k] = Q[k] / max(wsum[k], 1e-8) + cov_floor * np.eye(d)
    pi = pi / max(pi.sum(), 1e-8)
    return _pack(pi, None, W, Q, mu0, K, L, d)


def _pack(pi, logP, W, Q, mu0, K, L, d):
    Q_inv = np.linalg.inv(Q)
    Q_logdet = np.array([np.linalg.slogdet(Q[k])[1] for k in range(K)])
    if logP is None:
        logP = np.log(np.full((K, K), 1.0 / K))
    return ARHMMParams(pi=pi, logP=logP, W=W, Q_inv=Q_inv, Q_logdet=Q_logdet,
                       mu0=mu0, K=K, L=L, d=d)


class ARHMM:
    """Autoregressive HMM fit by EM ([guideline §4])."""

    backend = "numpy"

    def __init__(self, K: int, L: int, seed: int = 0):
        self.K, self.L, self.seed = K, L, seed
        self.params: ARHMMParams | None = None

    def emission_means(self) -> np.ndarray:
        """Per-state mean-pose (AR intercept); (K, d). For state alignment."""
        return self.params.W[:, -1, :]

    def fit(self, seqs, n_iter: int = 50, tol: float = 1e-3, verbose=False):
        p = _init_params(seqs, self.K, self.L, self.seed)
        prev = -np.inf
        for it in range(n_iter):
            gammas, trans, total = [], np.full((self.K, self.K), 1e-6), 0.0
            for x in seqs:
                ll = _emission_ll(x, p)
                g, xi, loglik = _forward_backward(ll, np.log(p.pi + 1e-12), p.logP)
                gammas.append(g)
                trans += xi
                total += loglik
            if verbose:
                print(f"    EM {it:2d}: loglik={total:.1f}")
            if total - prev < tol * abs(prev) and it > 2:
                break
            prev = total
            p = _m_step(seqs, gammas, self.K, self.L, p.d)
            logP = np.log(trans / trans.sum(axis=1, keepdims=True))
            p.logP = logP
        self.params = p
        self.final_loglik = total
        return self

    def log_likelihood(self, seqs) -> float:
        """Mean per-frame log-likelihood over ``seqs``."""
        p, total, n = self.params, 0.0, 0
        for x in seqs:
            ll = _emission_ll(x, p)
            _, _, loglik = _forward_backward(ll, np.log(p.pi + 1e-12), p.logP)
            total += loglik
            n += len(x)
        return total / max(n, 1)

    def decode(self, x) -> np.ndarray:
        """Viterbi most-likely state path for one walk ([guideline §4.6])."""
        p = self.params
        ll = _emission_ll(x, p)
        T, K = ll.shape
        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = np.log(p.pi + 1e-12) + ll[0]
        for t in range(1, T):
            m = delta[t - 1][:, None] + p.logP
            psi[t] = m.argmax(0)
            delta[t] = m.max(0) + ll[t]
        states = np.zeros(T, dtype=int)
        states[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def transition_matrix(self) -> np.ndarray:
        return np.exp(self.params.logP)

    def n_params(self) -> int:
        K, L, d = self.K, self.L, self.params.d
        return (K - 1) + K * (K - 1) + K * (L * d * d + d) + K * d * (d + 1) // 2

    def aic(self, seqs) -> float:
        n = sum(len(s) for s in seqs)
        return -2 * self.log_likelihood(seqs) * n + 2 * self.n_params()

    def sample(self, T: int, rng=None) -> np.ndarray:
        """Generate a synthetic sequence ([guideline §4.5])."""
        rng = np.random.default_rng() if rng is None else rng
        p = self.params
        P = np.exp(p.logP)
        d, L = p.d, p.L
        x = np.zeros((T, d))
        z = rng.choice(p.K, p=p.pi / p.pi.sum())
        for t in range(T):
            if t < L:
                mean = p.mu0[z]
            else:
                Phi = _design(x[:t + 1], L)[-1]
                mean = Phi @ p.W[z]
            Q = np.linalg.inv(p.Q_inv[z])
            x[t] = rng.multivariate_normal(mean, (Q + Q.T) / 2)
            z = rng.choice(p.K, p=P[z])
        return x


# ---- State alignment for averaging refits ([guideline §4.4]) --------------

def align_states(ref, other) -> np.ndarray:
    """Hungarian match of ``other``'s states to ``ref`` by emission mean.

    Returns a permutation ``perm`` s.t. ``other`` state ``perm[k]`` matches
    ``ref`` state ``k`` — required before averaging refits, or the mean of
    label-switched runs is meaningless ([guideline gotchas]). Backend-
    agnostic: uses each model's ``emission_means()``.
    """
    from scipy.optimize import linear_sum_assignment
    a, b = ref.emission_means(), other.emission_means()
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    _, col = linear_sum_assignment(cost)
    return col
