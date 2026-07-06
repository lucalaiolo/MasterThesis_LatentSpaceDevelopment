"""Mask policies for the three training recipes.

Every step, the training loop draws a fresh mask per clip from the
policy. Six policies are available:

    "none"              The mask is all ones. Only valid for Recipe 1,
                        where it reduces the model to a plain-VAE
                        ablation ([MVAE §8]).
    "uniform"           Each joint at each frame is hidden with
                        probability rho, independently. [MVAE §2.2]
    "top_k_speed"       Rank joints by mean speed over the clip and
                        deterministically hide the k = floor(rho * J)
                        fastest. Same mask across all frames. [MVAE §2.3]
    "softmax_speed"     Sample k joints without replacement via
                        Gumbel-Top-k on softmax(s / tau). Temperature
                        tau -> 0 recovers top-k; tau -> infinity
                        recovers uniform. [MVAE §2.4]
    "per_frame_speed"   At each frame, hide the top-k joints by
                        instantaneous speed. Mask varies through time.
                        [MVAE §2.5]
    "limb"              Hide one named limb for the whole clip. Limbs
                        can be sampled uniformly or weighted by their
                        mean speed. [MVAE §2.6]

The four speed-based policies need the clip to compute the score, so
`sample` takes an optional `X` of shape (T, J, 3). Policies that don't
use `X` accept and ignore it. The policies return NumPy arrays of
shape (T, J) with 1 for visible. The training loop converts to torch.
Nothing here needs torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _mean_speed(X: np.ndarray) -> np.ndarray:
    """Clip-level mean speed per joint ([MVAE §2.1]).

    s_j = (1 / (T - 1)) * sum_t || x_{t+1,j} - x_{t,j} ||_2.

    Args:
        X: shape (T, J, 3).
    Returns:
        Vector of shape (J,) in the same units as X.
    """
    diffs = np.diff(X, axis=0)                       # (T - 1, J, 3)
    return np.linalg.norm(diffs, axis=-1).mean(axis=0)


def _instantaneous_speed(X: np.ndarray) -> np.ndarray:
    """Per-frame speed per joint ([MVAE §2.5]).

    s_{t,j} = || x_{t+1,j} - x_{t,j} ||_2. The last row is a copy of
    the second-to-last since there is no x_{T} to diff into.

    Args:
        X: shape (T, J, 3).
    Returns:
        Array of shape (T, J).
    """
    diffs = np.diff(X, axis=0)                       # (T - 1, J, 3)
    S = np.linalg.norm(diffs, axis=-1)               # (T - 1, J)
    # Extend to T rows: the last frame gets the same score as its
    # predecessor. This keeps the mask shape (T, J) without inventing
    # a value for x_T - x_{T-1}.
    return np.vstack([S, S[-1:]])                    # (T, J)


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values in `scores`.

    Uses argpartition for O(n) selection; order within the selected
    block is unspecified and does not matter for our use.
    """
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= scores.shape[-1]:
        return np.arange(scores.shape[-1])
    return np.argpartition(-scores, k - 1, axis=-1)[..., :k]


class MaskPolicy:
    """The mask policy protocol.

    `sample` is the sole hook. Speed-based policies read `X`; others
    ignore it. `sample_batch` iterates `sample` over a batch.
    """

    def sample(self, T: int, J: int, rng: np.random.Generator,
               X: np.ndarray | None = None) -> np.ndarray:
        """Return one mask of shape (T, J)."""
        raise NotImplementedError

    def sample_batch(self, B: int, T: int, J: int,
                     rng: np.random.Generator,
                     X: np.ndarray | None = None) -> np.ndarray:
        """Return a batch of masks of shape (B, T, J).

        If `X` is passed it must be (B, T, J, 3); each clip's mask is
        drawn from the score of that clip. Otherwise no clip is used —
        only makes sense for the score-free policies.
        """
        if X is None:
            return np.stack([self.sample(T, J, rng) for _ in range(B)])
        return np.stack([self.sample(T, J, rng, X=X[b]) for b in range(B)])


@dataclass
class NoMask(MaskPolicy):
    """Every joint visible; the plain-VAE baseline for Recipe 1 ablations."""

    def sample(self, T: int, J: int, rng: np.random.Generator,
               X: np.ndarray | None = None) -> np.ndarray:
        return np.ones((T, J), dtype=np.float32)


@dataclass
class UniformMask(MaskPolicy):
    """Hide each joint at each frame with a fixed probability rho ([MVAE §2.2]).

    Attributes:
        rho: fraction hidden, in [0, 1].
    """

    rho: float = 0.3

    def sample(self, T: int, J: int, rng: np.random.Generator,
               X: np.ndarray | None = None) -> np.ndarray:
        keep = rng.random((T, J)) > self.rho
        return keep.astype(np.float32)


@dataclass
class TopKSpeedMask(MaskPolicy):
    """Deterministic top-k on mean speed ([MVAE §2.3]).

    Ranks joints by clip-level mean speed and hides the k = floor(rho J)
    fastest. The same joints are hidden at every frame — the mask does
    not vary through time.

    Attributes:
        rho: fraction hidden, in [0, 1].
    """

    rho: float = 0.3

    def sample(self, T: int, J: int, rng: np.random.Generator,
               X: np.ndarray | None = None) -> np.ndarray:
        if X is None:
            raise ValueError("TopKSpeedMask needs the clip X to score joints.")
        s = _mean_speed(X)                           # (J,)
        k = int(np.floor(self.rho * J))
        hidden = _top_k_indices(s, k)
        M = np.ones((T, J), dtype=np.float32)
        M[:, hidden] = 0.0
        return M


@dataclass
class SoftmaxSpeedMask(MaskPolicy):
    """Softmax-sampled top-k by speed via Gumbel-Top-k ([MVAE §2.4]).

    p_j = softmax(s_j / tau). We draw k joints without replacement by
    taking the top-k of (log p_j + Gumbel(0, 1)). Temperature tau -> 0
    recovers §2.3; tau -> infinity recovers §2.2.

    Attributes:
        rho: fraction hidden, in [0, 1].
        temperature: softmax temperature. Clamped to at least 1e-6 for
            numerical safety.
    """

    rho: float = 0.3
    temperature: float = 1.0

    def sample(self, T: int, J: int, rng: np.random.Generator,
               X: np.ndarray | None = None) -> np.ndarray:
        if X is None:
            raise ValueError("SoftmaxSpeedMask needs the clip X to score joints.")
        s = _mean_speed(X)                           # (J,)
        tau = max(self.temperature, 1e-6)
        # The softmax normaliser cancels under argmax, so we use the raw
        # logits s / tau. Gumbel noise is added and we take the top-k.
        logits = s / tau
        u = rng.uniform(size=J).clip(1e-12, 1.0)
        g = -np.log(-np.log(u))                       # Gumbel(0, 1)
        k = int(np.floor(self.rho * J))
        hidden = _top_k_indices(logits + g, k)
        M = np.ones((T, J), dtype=np.float32)
        M[:, hidden] = 0.0
        return M


@dataclass
class PerFrameSpeedMask(MaskPolicy):
    """Per-frame top-k by instantaneous speed ([MVAE §2.5]).

    At each frame, mask the k = floor(rho J) joints whose instantaneous
    speed is highest. The mask varies through time, so hidden joints
    scatter across the clip and the model has to lean on temporal
    context to recover motion peaks.

    Attributes:
        rho: fraction hidden per frame, in [0, 1].
    """

    rho: float = 0.3

    def sample(self, T: int, J: int, rng: np.random.Generator,
               X: np.ndarray | None = None) -> np.ndarray:
        if X is None:
            raise ValueError("PerFrameSpeedMask needs the clip X to score joints.")
        S = _instantaneous_speed(X)                  # (T, J)
        k = int(np.floor(self.rho * J))
        M = np.ones((T, J), dtype=np.float32)
        if k <= 0:
            return M
        if k >= J:
            return np.zeros((T, J), dtype=np.float32)
        top = _top_k_indices(S, k)                   # (T, k)
        rows = np.arange(T)[:, None]                 # (T, 1)
        M[rows, top] = 0.0
        return M


@dataclass
class LimbMask(MaskPolicy):
    """Hide one named limb for the whole clip ([MVAE §2.6]).

    Attributes:
        limbs: map from limb name to the joint indices for that limb.
        names: which limbs to pick from. Empty means all.
        speed_weighted: when True, sample the limb with probability
            proportional to its mean speed instead of uniformly.
    """

    limbs: dict[str, list[int]] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    speed_weighted: bool = False

    def sample(self, T: int, J: int, rng: np.random.Generator,
               X: np.ndarray | None = None) -> np.ndarray:
        pool = self.names if self.names else list(self.limbs)
        if not pool:
            raise ValueError("LimbMask has no limbs configured.")

        if self.speed_weighted:
            if X is None:
                raise ValueError(
                    "LimbMask(speed_weighted=True) needs the clip X."
                )
            s = _mean_speed(X)                       # (J,)
            weights = np.array(
                [s[self.limbs[name]].mean() for name in pool],
                dtype=np.float64,
            )
            total = weights.sum()
            p = (weights / total) if total > 0 else None
            pick = pool[int(rng.choice(len(pool), p=p))]
        else:
            pick = pool[int(rng.integers(len(pool)))]

        M = np.ones((T, J), dtype=np.float32)
        M[:, self.limbs[pick]] = 0.0
        return M


def build_policy(config, limbs: dict[str, list[int]] | None = None) -> MaskPolicy:
    """Build a mask policy from a TrainingConfig.

    Args:
        config: a TrainingConfig with `mask_policy` and any policy-specific
            fields set.
        limbs: joint-index lists, needed for "limb".
    Returns:
        A policy that the training loop calls once per step.
    """
    name = config.mask_policy
    if name == "none":
        return NoMask()
    if name == "uniform":
        return UniformMask(rho=config.mask_rho)
    if name == "top_k_speed":
        return TopKSpeedMask(rho=config.mask_rho)
    if name == "softmax_speed":
        return SoftmaxSpeedMask(rho=config.mask_rho,
                                temperature=config.mask_softmax_temperature)
    if name == "per_frame_speed":
        return PerFrameSpeedMask(rho=config.mask_rho)
    if name == "limb":
        if not limbs:
            raise ValueError("Limb policy needs a `limbs` map.")
        return LimbMask(limbs=limbs, names=list(config.mask_limb_names),
                        speed_weighted=config.mask_limb_speed_weighted)
    raise ValueError(f"unknown mask policy: {name!r}")
