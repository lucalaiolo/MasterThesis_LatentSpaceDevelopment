"""Interfaces and data containers for the neonate-motion VAE analysis.

Everything downstream speaks in terms of the objects defined here. Plug
a trained model in through `VAEModel`, describe the skeleton through
`Skeleton`, and hold encoded data in `LatentSet`. All shapes are generic
in the joint count J, the clip length T, and the latent width d_z.

Array-shape conventions (NumPy or torch, as noted per function):
    X       clips              (N, T, J, 3)
    M       masks, 1 = visible (N, T, J)
    mu      posterior means    (N, d_z)
    logvar  posterior log-var  (N, d_z)
    z       latent samples     (N, d_z)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VAEModel(Protocol):
    """The plug point for a trained VAE.

    Implement this thin wrapper around your own model. The analysis code
    calls only these four methods, so nothing else about your training
    setup leaks in. Methods that feed the Jacobian tools (encode_mean,
    decode) must be differentiable and accept torch tensors; the batch
    encode/decode may use NumPy for speed.
    """

    def encode(self, X: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map a batch of clips to posterior mean and log-variance.

        Args:
            X: clips, shape (N, T, J, 3).
            M: masks, shape (N, T, J), 1 for a visible joint.
        Returns:
            (mu, logvar), each shape (N, d_z).
        """
        ...

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Map a batch of latents to reconstructed clips.

        Args:
            z: latents, shape (N, d_z).
        Returns:
            X_hat, shape (N, T, J, 3).
        """
        ...

    def encode_mean_torch(self, X, M):
        """Differentiable posterior mean for one clip, in torch.

        Args:
            X: torch tensor, shape (T, J, 3), requires grad for the
                encoder Jacobian.
            M: torch tensor, shape (T, J).
        Returns:
            mu, torch tensor of shape (d_z,).
        """
        ...

    def decode_torch(self, z):
        """Differentiable decoder for one latent, in torch.

        Args:
            z: torch tensor, shape (d_z,), requires grad for the decoder
                Jacobian.
        Returns:
            X_hat, torch tensor of shape (T, J, 3).
        """
        ...


@dataclass
class Skeleton:
    """A generic skeleton: joint count, bones, and left-right pairing.

    Supply this for your own model. Nothing here assumes J = 22.

    Attributes:
        n_joints: the joint count J.
        bones: list of (parent, child) index pairs, one per rigid link.
        left_right: list of (left_index, right_index) pairs for bilateral
            joints. Midline joints (spine, head) are left out.
        lateral_axis: the coordinate index a left-right mirror negates,
            0 for x by convention.
        limbs: optional map from a limb name to its joint indices, used
            by the kinematic features. Leave empty to skip per-limb work.
    """

    n_joints: int
    bones: list[tuple[int, int]] = field(default_factory=list)
    left_right: list[tuple[int, int]] = field(default_factory=list)
    lateral_axis: int = 0
    limbs: dict[str, list[int]] = field(default_factory=dict)

    def flip_permutation(self) -> np.ndarray:
        """Return the J-by-J permutation matrix of the left-right swap.

        Left and right partners trade places; every other joint stays
        put. Used by the symmetry analysis.
        """
        P = np.eye(self.n_joints)
        for a, b in self.left_right:
            P[[a, b]] = P[[b, a]]
        return P

    def bone_index(self) -> np.ndarray:
        """Return the bones as an integer array of shape (n_bones, 2)."""
        if not self.bones:
            raise ValueError("Skeleton has no bones; set `bones` to use "
                             "bone-length checks.")
        return np.asarray(self.bones, dtype=int)


@dataclass
class LatentSet:
    """Encoded data: posterior parameters, samples, and clip labels.

    Attributes:
        mu: posterior means, shape (N, d_z).
        logvar: posterior log-variances, shape (N, d_z).
        z: one latent sample per clip, shape (N, d_z). Filled by
            `sample` if left as None.
        video_id: integer video label per clip, shape (N,), for the
            per-video and cross-video statistics.
        time_index: clip start frame per clip, shape (N,), for temporal
            hold-outs and sliding-window order.
    """

    mu: np.ndarray
    logvar: np.ndarray
    z: np.ndarray | None = None
    video_id: np.ndarray | None = None
    time_index: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.mu.shape[0]

    @property
    def d_z(self) -> int:
        return self.mu.shape[1]

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw one latent per clip by the reparameterisation trick.

        Stores the draw in `z` and returns it.
        """
        rng = np.random.default_rng() if rng is None else rng
        std = np.exp(0.5 * self.logvar)
        self.z = self.mu + std * rng.standard_normal(self.mu.shape)
        return self.z

    def prior_like(self, n: int | None = None,
                   rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw `n` samples from the standard-normal prior of the same width."""
        rng = np.random.default_rng() if rng is None else rng
        n = self.n if n is None else n
        return rng.standard_normal((n, self.d_z))


def encode_dataset(model: VAEModel, X: np.ndarray, M: np.ndarray,
                   batch: int = 256, **labels) -> LatentSet:
    """Encode a whole dataset into a `LatentSet`.

    Runs the model's batch encoder over the clips and packs the result.
    Pass `video_id` and `time_index` through `labels` to carry them.
    """
    mus, lvs = [], []
    for i in range(0, len(X), batch):
        mu, lv = model.encode(X[i:i + batch], M[i:i + batch])
        mus.append(np.asarray(mu))
        lvs.append(np.asarray(lv))
    return LatentSet(mu=np.concatenate(mus), logvar=np.concatenate(lvs),
                     **labels)
