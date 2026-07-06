"""Held-out evaluation ([MVAE §7]).

Mean per-joint position error is the average Euclidean distance between
predicted and true joint positions. Two variants:

    mpjpe_reconstruction  averaged over every joint of every clip; the
                          model sees the unmasked clip at inference
                          ([MVAE §7.1]).
    mpjpe_inpainting      averaged over hidden joints only; the model
                          sees the masked clip at inference. For
                          Recipe 3 this reads off the mask-conditioned
                          inpainting head ([MVAE §7.2]).
"""

from __future__ import annotations

import numpy as np


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("Evaluation needs PyTorch.") from e


def _decode_full(model, X, M):
    """Return the full-clip reconstruction from `model(X, M)`.

    Handles both the two-tuple return (Recipes 1, 2) and the three-tuple
    return (Recipe 3, where the first element is the full head).
    """
    out = model(X, M)
    return out[0]


def _decode_inpaint(model, X, M):
    """Return the inpainting reconstruction for a masked input.

    For Recipe 3 the model returns (X_hat_full, X_hat_inp, mu, logvar);
    we use the mask-conditioned head. For Recipes 1 and 2 there is only
    one head, so we use it.
    """
    out = model(X, M)
    if len(out) == 4:                       # Recipe 3: (full, inp, mu, logvar)
        return out[1]
    return out[0]


def evaluate(model, clips: np.ndarray, mask_policy, batch_size: int = 64,
             device: str = "cpu", seed: int = 0,
             recipe: int = 1) -> dict:
    """Compute MPJPE variants on a held-out set of clips.

    Args:
        model: a trained VAE.
        clips: shape (N, T, J, 3).
        mask_policy: draws test-time masks for the inpainting metric.
        batch_size: minibatch size.
        device: "cuda" or "cpu".
        seed: seeds the mask draws.
        recipe: 1, 2, or 3, matching the model that produced X_hat. Only
            affects which head answers the inpainting query.
    Returns:
        Dict with mean per-joint position error over all joints, over
        visible joints, and over hidden joints. `mpjpe_all` comes from
        the unmasked-input reconstruction pass; `mpjpe_visible` and
        `mpjpe_inpainted` come from the masked-input inference pass.
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

            # ---- Reconstruction pass: unmasked input, full head -----
            M_ones = torch.ones((B, T, J), device=device)
            X_hat_full = _decode_full(model, X, M_ones)
            err_full = torch.linalg.norm(X_hat_full - X, dim=-1)   # (B, T, J)
            tot_all += float(err_full.sum())
            n_all += err_full.numel()

            # ---- Inpainting pass: masked input, mask-aware head -----
            M = np.stack([mask_policy.sample(T, J, rng) for _ in range(B)])
            Mt = torch.from_numpy(M).to(device)
            X_hat_inp = _decode_inpaint(model, X, Mt)
            err_inp = torch.linalg.norm(X_hat_inp - X, dim=-1)     # (B, T, J)

            vis = Mt > 0.5
            hid = ~vis
            if vis.any():
                tot_vis += float(err_inp[vis].sum())
                n_vis += int(vis.sum())
            if hid.any():
                tot_inp += float(err_inp[hid].sum())
                n_inp += int(hid.sum())

    return {
        "mpjpe_all": tot_all / max(n_all, 1),
        "mpjpe_visible": tot_vis / max(n_vis, 1),
        "mpjpe_inpainted": tot_inp / max(n_inp, 1),
    }
