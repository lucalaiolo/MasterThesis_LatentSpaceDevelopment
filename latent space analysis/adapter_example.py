"""A worked adapter: wrap your trained torch VAE into the analysis interface.

Copy this file, change the two model calls to match your own network, and
every analysis in the toolkit will run against your model. The interface
needs four methods: two batch methods in NumPy for speed, and two single-clip
methods in torch for the Jacobian tools.

This example assumes your network exposes:
    net.encode(x, m) -> (mu, logvar)   with x (B, T, J, 3), m (B, T, J)
    net.decode(z)    -> x_hat          with z (B, d_z)
Adjust the reshapes if your network takes flattened input.
"""

from __future__ import annotations

import numpy as np


class TorchVAEAdapter:
    """Adapter from a trained torch VAE to the analysis `VAEModel` interface."""

    def __init__(self, net, device: str = "cpu"):
        import torch
        self.torch = torch
        self.net = net.to(device).eval()
        self.device = device

    # ---- Batch methods in NumPy, used by most analyses. ----
    def encode(self, X, M):
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
            m = torch.as_tensor(np.asarray(M), dtype=torch.float32, device=self.device)
            mu, logvar = self.net.encode(x, m)
        return np.asarray(mu.cpu()), np.asarray(logvar.cpu())

    def decode(self, z):
        torch = self.torch
        with torch.no_grad():
            zt = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=self.device)
            xh = self.net.decode(zt)
        return np.asarray(xh.cpu())

    # ---- Single-clip methods in torch, used by the Jacobian tools. ----
    def encode_mean_torch(self, X, M):
        # X (T, J, 3), M (T, J); return mu (d_z,).
        mu, _ = self.net.encode(X[None], M[None])
        return mu[0]

    def decode_torch(self, z):
        # z (d_z,); return x_hat (T, J, 3).
        return self.net.decode(z[None])[0]


# ---- Example skeleton for a hypothetical 17-joint model. ----
def example_skeleton():
    """A generic skeleton. Replace indices with your own joint layout."""
    from vae_analysis import Skeleton
    return Skeleton(
        n_joints=17,
        bones=[(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
               (0, 7), (7, 8), (8, 9), (7, 10), (10, 11), (11, 12),
               (7, 13), (13, 14), (14, 15), (15, 16)],
        left_right=[(1, 4), (2, 5), (3, 6), (10, 13), (11, 14), (12, 15)],
        lateral_axis=0,
        limbs={"left_arm": [1, 2, 3], "right_arm": [4, 5, 6],
               "left_leg": [10, 11, 12], "right_leg": [13, 14, 15]},
    )


if __name__ == "__main__":
    print("Import your net, wrap it: model = TorchVAEAdapter(net). "
          "Then follow smoke_test.py, swapping the FakeModel for `model`.")
