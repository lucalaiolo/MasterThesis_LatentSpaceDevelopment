"""Held-out evaluation ([MVAE §7]).

Mean per-joint position error is the average Euclidean distance between
predicted and true joint positions. Two variants:

    mpjpe_reconstruction  averaged over every joint of every clip.
    mpjpe_inpainting      averaged over hidden joints only, useful for
                          Recipe 3.
"""

from __future__ import annotations

import numpy as np


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("Evaluation needs PyTorch.") from e


def evaluate(model, clips: np.ndarray, mask_policy, batch_size: int = 64,
             device: str = "cpu", seed: int = 0,
             recipe: int = 1) -> dict:
    """Compute MPJPE variants on a held-out set of clips.

    Args:
        model: a trained VAE.
        clips: shape (N, T, J, 3).
        mask_policy: draws test-time masks.
        batch_size: minibatch size.
        device: "cuda" or "cpu".
        seed: seeds the mask draws.
        recipe: 1, 2, or 3, matching the model that produced X_hat.
    Returns:
        Dict with mean per-joint position error over all joints, over
        visible joints, and over hidden joints.
    """
    torch = _torch()
    model.eval()
    model.to(device)

    rng = np.random.default_rng(seed)
    T, J = clips.shape[1], clips.shape[2]

    tot_all, tot_vis, tot_inp = 0.0, 0.0, 0.0
    n_all, n_vis, n_inp = 0, 0, 0

    with torch.no_grad():
        for i in range(0, len(clips), batch_size):
            X = torch.from_numpy(clips[i:i + batch_size].astype(np.float32)).to(device)
            B = X.shape[0]

            M = np.stack([mask_policy.sample(T, J, rng) for _ in range(B)])
            Mt = torch.from_numpy(M).to(device)
            M_in = Mt if recipe in (1, 3) else torch.ones_like(Mt)

            X_hat, _, _ = model(X, M_in)
            err = torch.linalg.norm(X_hat - X, dim=-1)   # (B, T, J), per-joint distance

            tot_all += float(err.sum())
            n_all += err.numel()

            vis = Mt > 0.5
            hid = ~vis
            if vis.any():
                tot_vis += float(err[vis].sum())
                n_vis += int(vis.sum())
            if hid.any():
                tot_inp += float(err[hid].sum())
                n_inp += int(hid.sum())

    return {
        "mpjpe_all": tot_all / max(n_all, 1),
        "mpjpe_visible": tot_vis / max(n_vis, 1),
        "mpjpe_inpainted": tot_inp / max(n_inp, 1),
    }
