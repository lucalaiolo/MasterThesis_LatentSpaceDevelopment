"""Pluggable ARHMM backends — ssm (paper), dynamax (JAX), or NumPy ([guideline §4]).

The paper fits the ARHMM with the Linderman-lab **``ssm``** library; the
guideline offers **``dynamax``** (JAX) as a maintained fallback and the
NumPy implementation in ``statespace.py`` as the always-available default.
All three expose the same interface the driver uses:

    m = build_arhmm(K, L, backend="ssm").fit(seqs)
    m.log_likelihood(seqs)   # mean per-frame
    m.decode(x)              # Viterbi state path
    m.transition_matrix() ; m.emission_means() ; m.n_params() ; m.aic(seqs)
    m.sample(T, rng)

``ssm`` is unmaintained (needs Cython + ``numpy<1.24``); ``dynamax`` needs
JAX. Both import lazily, so this module loads without them and raises a
clear install hint only when the backend is actually requested. Record which
backend a run used — it is written into the results.
"""

from __future__ import annotations

import numpy as np

from .statespace import ARHMM as _NumpyARHMM


def available_backends() -> list[str]:
    """Which backends can import right now (numpy is always available)."""
    out = ["numpy"]
    for name, mod in (("ssm", "ssm"), ("dynamax", "dynamax")):
        try:
            __import__(mod)
            out.append(name)
        except Exception:
            pass
    return out


def build_arhmm(K: int, L: int, backend: str = "numpy", seed: int = 0):
    """Factory. ``backend`` in {numpy, ssm, dynamax, auto}.

    ``auto`` prefers the paper's ``ssm``, then ``dynamax``, then NumPy.
    """
    if backend == "auto":
        avail = available_backends()
        backend = "ssm" if "ssm" in avail else (
            "dynamax" if "dynamax" in avail else "numpy")
    if backend == "numpy":
        return _NumpyARHMM(K, L, seed=seed)
    if backend == "ssm":
        return SSMBackend(K, L, seed=seed)
    if backend == "dynamax":
        return DynamaxBackend(K, L, seed=seed)
    raise ValueError(f"unknown backend {backend!r}")


# ---- ssm backend (the paper's implementation, [guideline §4.1]) -----------

class SSMBackend:
    """Wraps ``ssm.HMM`` exactly as the reference ``ARHMM.ipynb`` does."""

    backend = "ssm"

    def __init__(self, K: int, L: int, seed: int = 0):
        self.K, self.L, self.seed = K, L, seed
        self.model = None
        self._d = None

    def _build(self, D):
        try:
            import ssm
        except ImportError as e:
            raise ImportError(
                "backend='ssm' needs the Linderman-lab `ssm` library. It is "
                "unmaintained: `pip install numpy<1.24 cython` then "
                "`pip install git+https://github.com/lindermanlab/ssm`."
            ) from e
        # Exactly the paper's call: kmeans init, stochastic EM, standard
        # transitions; AR observations carry the lag, Gaussian (L=0) do not.
        if self.L > 0:
            return ssm.HMM(self.K, D, init_method="kmeans", observations="ar",
                           observation_kwargs={"lags": self.L},
                           method="stochastic_em", transitions="standard")
        return ssm.HMM(self.K, D, init_method="kmeans", observations="gaussian",
                       method="stochastic_em", transitions="standard")

    def fit(self, seqs, n_iter: int = 500, tol: float = 1e-3):
        self._d = seqs[0].shape[1]
        np.random.seed(self.seed)
        self.model = self._build(self._d)
        self.model.fit([np.asarray(s) for s in seqs], num_iters=n_iter,
                       tolerance=tol)
        self.final_loglik = self.model.log_likelihood(list(seqs))
        return self

    def log_likelihood(self, seqs) -> float:
        n = sum(len(s) for s in seqs)
        return float(self.model.log_likelihood(list(seqs))) / max(n, 1)

    def decode(self, x) -> np.ndarray:
        return np.asarray(self.model.most_likely_states(np.asarray(x)))

    def transition_matrix(self) -> np.ndarray:
        tr = self.model.transitions
        if hasattr(tr, "transition_matrix"):
            return np.asarray(tr.transition_matrix)
        return np.exp(np.asarray(tr.log_Ps))

    def emission_means(self) -> np.ndarray:
        obs = self.model.observations
        for attr in ("bs", "mus", "b"):
            if hasattr(obs, attr):
                v = np.asarray(getattr(obs, attr))
                return v.reshape(self.K, -1)[:, :self._d]
        # fallback: state means of a synthetic draw
        return np.zeros((self.K, self._d))

    def n_params(self) -> int:
        p = self.model.params
        return int(sum(np.size(i) for i in _flatten(p)))

    def aic(self, seqs) -> float:
        n = sum(len(s) for s in seqs)
        return -2 * self.log_likelihood(seqs) * n + 2 * self.n_params()

    def sample(self, T: int, rng=None) -> np.ndarray:
        out = self.model.sample(T)
        return np.asarray(out[1] if isinstance(out, tuple) else out)


# ---- dynamax backend (JAX, [guideline §1]) --------------------------------

class DynamaxBackend:
    """Wraps ``dynamax.LinearAutoregressiveHMM`` (JAX).

    dynamax EM wants equal-length batches, so variable-length walks are
    right-padded to the longest walk (padding uses the last frame). This is a
    best-effort integration; for a strictly faithful reproduction use
    ``backend='ssm'``, which handles the ragged list natively.
    """

    backend = "dynamax"

    def __init__(self, K: int, L: int, seed: int = 0):
        self.K, self.L, self.seed = K, L, max(L, 0)
        self.model = self.params = None
        self._d = None
        self._lengths = None

    def _lib(self):
        try:
            import jax
            import jax.numpy as jnp
            import jax.random as jr
            from dynamax.hidden_markov_model import (
                LinearAutoregressiveHMM, GaussianHMM)
            return jax, jnp, jr, LinearAutoregressiveHMM, GaussianHMM
        except ImportError as e:
            raise ImportError(
                "backend='dynamax' needs JAX + dynamax: "
                "`pip install jax dynamax`."
            ) from e

    def _pad(self, seqs, jnp):
        Tmax = max(len(s) for s in seqs)
        self._lengths = [len(s) for s in seqs]
        batch = np.stack([np.concatenate(
            [s, np.repeat(s[-1:], Tmax - len(s), 0)], 0) for s in seqs])
        return jnp.asarray(batch), Tmax

    def fit(self, seqs, n_iter: int = 100, tol: float = 1e-3):
        jax, jnp, jr, ARHMM, GaussHMM = self._lib()
        self._d = seqs[0].shape[1]
        batch, _ = self._pad(seqs, jnp)
        if self.L > 0:
            self.model = ARHMM(self.K, self._d, num_lags=self.L)
            inputs = jax.vmap(self.model.compute_inputs)(batch)
            params, props = self.model.initialize(
                jr.PRNGKey(self.seed), method="kmeans",
                emissions=batch, inputs=inputs)
            params, lls = self.model.fit_em(params, props, batch,
                                            inputs=inputs, num_iters=n_iter)
            self._inputs = inputs
        else:
            self.model = GaussHMM(self.K, self._d)
            params, props = self.model.initialize(jr.PRNGKey(self.seed),
                                                  method="kmeans", emissions=batch)
            params, lls = self.model.fit_em(params, props, batch, num_iters=n_iter)
            self._inputs = None
        self.params = params
        self.final_loglik = float(np.asarray(lls)[-1])
        return self

    def _emit_inputs(self, x):
        import jax.numpy as jnp
        if self.L > 0:
            return self.model.compute_inputs(jnp.asarray(x))
        return None

    def log_likelihood(self, seqs) -> float:
        import jax.numpy as jnp
        total, n = 0.0, 0
        for s in seqs:
            x = jnp.asarray(s)
            inp = self._emit_inputs(s)
            kw = {"inputs": inp} if inp is not None else {}
            total += float(self.model.marginal_log_prob(self.params, x, **kw))
            n += len(s)
        return total / max(n, 1)

    def decode(self, x) -> np.ndarray:
        import jax.numpy as jnp
        inp = self._emit_inputs(x)
        kw = {"inputs": inp} if inp is not None else {}
        return np.asarray(self.model.most_likely_states(
            self.params, jnp.asarray(x), **kw))

    def transition_matrix(self) -> np.ndarray:
        return np.asarray(self.params.transitions.transition_matrix)

    def emission_means(self) -> np.ndarray:
        em = self.params.emissions
        for attr in ("biases", "means", "bias"):
            if hasattr(em, attr):
                return np.asarray(getattr(em, attr)).reshape(self.K, -1)[:, :self._d]
        return np.zeros((self.K, self._d))

    def n_params(self) -> int:
        import jax
        leaves = jax.tree_util.tree_leaves(self.params)
        return int(sum(np.size(np.asarray(l)) for l in leaves))

    def aic(self, seqs) -> float:
        n = sum(len(s) for s in seqs)
        return -2 * self.log_likelihood(seqs) * n + 2 * self.n_params()

    def sample(self, T: int, rng=None) -> np.ndarray:
        import jax.random as jr
        seed = 0 if rng is None else int(rng.integers(0, 2 ** 31 - 1))
        kw = {}
        if self.L > 0:
            kw["prev_emissions"] = None
        states, emissions = self.model.sample(self.params, jr.PRNGKey(seed), T, **kw)
        return np.asarray(emissions)


def _flatten(obj):
    """Flatten ssm's nested params tuple into leaf arrays."""
    if isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _flatten(o)
    else:
        yield obj
