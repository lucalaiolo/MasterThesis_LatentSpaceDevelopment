"""Mask policies for the three training recipes.

Every step, the training loop draws a fresh mask per clip from the
policy. Three policies map to the three regimes of [MVAE §3-5]:

    "none"     Recipe 2. The mask is all ones. The encoder sees the
               unmasked clip.
    "uniform"  Recipe 1 or 3. Each joint at each frame is hidden with
               probability rho, independently. [MVAE §2.2]
    "limb"     Recipe 1 or 3. One named limb is hidden for the whole
               clip. Every joint of the limb goes; every other joint
               stays. [MVAE §2.6]

The policies return NumPy arrays of shape (T, J) with 1 for visible.
The training loop converts to torch. Nothing here needs torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class MaskPolicy:
    """The mask policy protocol."""

    def sample(self, T: int, J: int, rng: np.random.Generator) -> np.ndarray:
        """Return one mask of shape (T, J)."""
        raise NotImplementedError

    def sample_batch(self, B: int, T: int, J: int,
                     rng: np.random.Generator) -> np.ndarray:
        """Return a batch of masks of shape (B, T, J)."""
        return np.stack([self.sample(T, J, rng) for _ in range(B)])


@dataclass
class NoMask(MaskPolicy):
    """Recipe 2. Every joint visible."""

    def sample(self, T: int, J: int, rng: np.random.Generator) -> np.ndarray:
        return np.ones((T, J), dtype=np.float32)


@dataclass
class UniformMask(MaskPolicy):
    """Hide each joint at each frame with a fixed probability rho.

    Attributes:
        rho: fraction hidden, in [0, 1].
    """

    rho: float = 0.3

    def sample(self, T: int, J: int, rng: np.random.Generator) -> np.ndarray:
        keep = rng.random((T, J)) > self.rho
        return keep.astype(np.float32)


@dataclass
class LimbMask(MaskPolicy):
    """Hide one named limb for the whole clip.

    Attributes:
        limbs: map from limb name to the joint indices for that limb.
        names: which limbs to pick from. Empty means all.
    """

    limbs: dict[str, list[int]] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)

    def sample(self, T: int, J: int, rng: np.random.Generator) -> np.ndarray:
        pool = self.names if self.names else list(self.limbs)
        if not pool:
            raise ValueError("LimbMask has no limbs configured.")
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
    if config.mask_policy == "none":
        return NoMask()
    if config.mask_policy == "uniform":
        return UniformMask(rho=config.mask_uniform_rho)
    if config.mask_policy == "limb":
        if not limbs:
            raise ValueError("Limb policy needs a `limbs` map.")
        return LimbMask(limbs=limbs, names=list(config.mask_limb_names))
    raise ValueError(f"unknown mask policy: {config.mask_policy!r}")
