"""ELBO components and the split reconstruction loss for Recipe 3.

Three functions the training loop calls each step:

    kl_gaussian            KL between the diagonal Gaussian posterior
                           and the standard-normal prior, per sample.
    reconstruction_mse     mean squared error over all joints; used for
                           Recipes 1 and 2.
    split_reconstruction   separate mean squared errors on visible and
                           hidden joints; used for Recipe 3.
"""

from __future__ import annotations


def _import_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("This module needs PyTorch.") from e


torch = _import_torch()


def kl_gaussian(mu, logvar):
    """Kullback-Leibler divergence to a standard-normal prior.

    Closed form for a diagonal Gaussian posterior:
        KL = -1/2 * sum_i (1 + logvar_i - mu_i^2 - exp(logvar_i)).

    Args:
        mu, logvar: (B, d_z).
    Returns:
        Tensor of shape (B,), one KL per sample.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def reconstruction_mse(x_hat, x):
    """Mean squared error over every joint and coordinate.

    Args:
        x_hat, x: (B, T, J, 3).
    Returns:
        Scalar mean over the batch, time, joints, and coordinates.
    """
    return (x_hat - x).pow(2).mean()


def split_reconstruction(x_hat, x, M, lambda_visible: float,
                         lambda_inpainted: float):
    """Weighted mean squared error split by joint visibility (Recipe 3).

    Divides the pixel-wise squared error into visible-joint and
    hidden-joint parts, each normalised by its own count, and weights
    them for the total.

    Args:
        x_hat, x: (B, T, J, 3).
        M: (B, T, J), 1 for visible.
        lambda_visible, lambda_inpainted: weights that should sum to 1.
    Returns:
        (total, visible_mse, inpainted_mse), the total for the optimiser
        and the two parts for logging.
    """
    err_sq = (x_hat - x).pow(2)                       # (B, T, J, 3)
    M_ext = M.unsqueeze(-1)                           # (B, T, J, 1)
    n_vis = (M.sum() * 3).clamp_min(1.0)
    n_inp = ((1 - M).sum() * 3).clamp_min(1.0)
    loss_vis = (err_sq * M_ext).sum() / n_vis
    loss_inp = (err_sq * (1 - M_ext)).sum() / n_inp
    total = lambda_visible * loss_vis + lambda_inpainted * loss_inp
    return total, loss_vis, loss_inp


def beta_schedule(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    """Linear warmup of the KL weight, held at beta_max after warmup."""
    if warmup_epochs <= 0:
        return beta_max
    return min(1.0, epoch / warmup_epochs) * beta_max
