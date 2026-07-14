"""Encoder-side information geometry (Part II Section 14).

The decoder derivative says what each latent writes to the pose. The
encoder derivative says what each latent reads from the pose. Comparing
the two closes the loop: a joint the decoder reads out of a latent that
the encoder never wrote into it is carried by a training-set correlation,
not by direct coding.
"""

from __future__ import annotations

import numpy as np


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("Encoder Jacobian tools need PyTorch.") from e


def encoder_jacobian(model, X: np.ndarray, M: np.ndarray):
    """Encoder Jacobian of the posterior mean at one clip.

    Args:
        model: a VAEModel with a differentiable `encode_mean_torch`.
        X: clip, shape (T, J, 3).
        M: mask, shape (T, J).
    Returns:
        Jacobian array of shape (d_z, T, J, 3).
    """
    torch = _require_torch()
    from torch.func import jacrev

    X_t = torch.as_tensor(X, dtype=torch.float32)
    M_t = torch.as_tensor(M, dtype=torch.float32)
    J = jacrev(lambda x: model.encode_mean_torch(x, M_t))(X_t)  # (d_z,T,J,3)
    return np.asarray(J.detach().cpu())


def encoder_sensitivity_map(model, clips: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Average latent-by-joint read map over a set of clips (Section 14.1).

    Args:
        model: a VAEModel.
        clips: shape (A, T, J, 3).
        masks: shape (A, T, J).
    Returns:
        Map E of shape (d_z, J): how much each joint moves each latent.
    """
    acc = None
    for X, M in zip(clips, masks):
        Jc = encoder_jacobian(model, X, M)          # (d_z, T, J, 3)
        e = np.sqrt((Jc ** 2).sum(axis=(1, 3)) / Jc.shape[1])  # (d_z, J)
        acc = e if acc is None else acc + e
    return acc / len(clips)


def precision_spectrum(latent) -> dict:
    """Mean posterior precision per latent dimension (Section 14.2).

    Precision is one over the posterior variance. A dimension near one
    sits at the prior and carries nothing; a large value is pinned down
    hard by the data. This grades the active-unit count.

    Returns:
        Dict with `precision` (d_z,) sorted high to low, the sorting
        `order`, and the count of dimensions above 2 (a loose "live" mark).
    """
    prec = np.mean(1.0 / np.exp(latent.logvar), axis=0)  # (d_z,)
    order = np.argsort(prec)[::-1]
    return {"precision": prec[order], "order": order,
            "n_live": int(np.sum(prec > 2.0))}


def read_write_mismatch(read_map: np.ndarray, write_map: np.ndarray) -> np.ndarray:
    """Compare encoder read and decoder write maps (Section 14.1).

    Both maps are joint-by-latent (the decoder map transposed to match).
    Returns the normalised absolute difference per (latent, joint) pair;
    large entries flag correlation-carried joints.

    Args:
        read_map: encoder map E, shape (d_z, J).
        write_map: decoder map S transposed to (d_z, J).
    Returns:
        Mismatch matrix, shape (d_z, J), each map first scaled to unit max.
    """
    r = read_map / (read_map.max() + 1e-12)
    w = write_map / (write_map.max() + 1e-12)
    return np.abs(r - w)
