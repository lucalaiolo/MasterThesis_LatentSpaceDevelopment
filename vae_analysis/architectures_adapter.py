"""Adapter mapping the `architectures` VAE onto the analysis interface.

Our `ConvVAE` / `TransformerVAE` expose `encode(X, M)` and
`decode_full(z)` (plus `decode_inp(z, M)` for Recipe 3). The analysis
toolkit's `VAEModel` protocol asks for `encode`, `decode`,
`encode_mean_torch`, and `decode_torch` — this shim lines them up.

Usage:

    from architectures.analyze import load_checkpoint
    from vae_analysis.architectures_adapter import ArchitecturesAdapter

    model, config = load_checkpoint(ckpt_path, device="cuda")
    adapter = ArchitecturesAdapter(model, device="cuda")

    from vae_analysis import encode_dataset
    latent = encode_dataset(adapter, X, M, video_id=vid, time_index=t0)
"""

from __future__ import annotations

import numpy as np


class ArchitecturesAdapter:
    """Wrap an `architectures` VAE into the analysis `VAEModel` protocol."""

    def __init__(self, net, device: str = "cpu"):
        import torch
        self.torch = torch
        self.net = net.to(device).eval()
        self.device = device

    # ---- Batch methods in NumPy, used by most analyses. ----
    def encode(self, X, M):
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(X), dtype=torch.float32,
                                device=self.device)
            m = torch.as_tensor(np.asarray(M), dtype=torch.float32,
                                device=self.device)
            mu, logvar = self.net.encode(x, m)
        return np.asarray(mu.cpu()), np.asarray(logvar.cpu())

    def decode(self, z):
        """Full-clip decoder — the toolkit's `decode(z)` maps to `decode_full`."""
        torch = self.torch
        with torch.no_grad():
            zt = torch.as_tensor(np.asarray(z), dtype=torch.float32,
                                 device=self.device)
            xh = self.net.decode_full(zt)
        return np.asarray(xh.cpu())

    # ---- Temporal-latent window layout (temporal_* models only). ----
    def is_temporal(self) -> bool:
        """True when the wrapped net keeps a per-window latent sequence."""
        return hasattr(self.net, "window_latents")

    def n_windows(self) -> int:
        return int(self.net.n_windows())

    @property
    def d_z(self) -> int:
        """Per-window latent width (``d`` in the HMM notation)."""
        return int(self.net.d_z)

    def window_latents(self, z):
        """Flattened latent ``(B, d_z*n_win)`` -> windows ``(B, n_win, d_z)`` (NumPy).

        Delegates to the model so the conv/transformer flatten-order
        difference stays where it belongs.
        """
        torch = self.torch
        zt = torch.as_tensor(np.asarray(z), dtype=torch.float32)
        return np.asarray(self.net.window_latents(zt))

    def flatten_windows(self, w):
        """Windows ``(B, n_win, d_z)`` -> flattened latent ``(B, d_z*n_win)`` (NumPy)."""
        torch = self.torch
        wt = torch.as_tensor(np.asarray(w), dtype=torch.float32)
        return np.asarray(self.net.flatten_windows(wt))

    # ---- Single-clip methods in torch, used by the Jacobian tools. ----
    def encode_mean_torch(self, X, M):
        # X (T, J, 3), M (T, J); return mu (d_z,). jacrev calls this with
        # tensors built from numpy (CPU) even when the net is on CUDA.
        # Move them explicitly so the CUDA matmul does not complain.
        X = X.to(self.device)
        M = M.to(self.device)
        mu, _ = self.net.encode(X[None], M[None])
        return mu[0]

    def decode_torch(self, z):
        # z (d_z,); return x_hat (T, J, 3). See note above on device.
        z = z.to(self.device)
        return self.net.decode_full(z[None])[0]
