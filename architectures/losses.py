"""ELBO components and the reconstruction losses for the three recipes.

The training loop calls these each step:

    kl_gaussian                KL between the diagonal Gaussian posterior
                               and the standard-normal prior, per sample.
    reconstruction_mse         mean squared error over all joints; the
                               full-clip term used by every recipe.
    reconstruction_mse_hidden  mean squared error over hidden joints only;
                               the inpainting term for Recipe 3's
                               second head ([MVAE §5.2]).
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


def reconstruction_mse_hidden(x_hat, x, M):
    """Mean squared error over hidden joints only ([MVAE §5.2]).

    Normalises by the number of hidden coordinates in this batch so the
    magnitude stays stable across masks with different hide-fractions.

    Args:
        x_hat, x: (B, T, J, 3).
        M: (B, T, J), 1 for visible.
    Returns:
        Scalar mean over hidden joint coordinates.
    """
    err_sq = (x_hat - x).pow(2)                        # (B, T, J, 3)
    hidden = (1 - M).unsqueeze(-1)                     # (B, T, J, 1)
    n_inp = (hidden.sum() * 3).clamp_min(1.0)
    return (err_sq * hidden).sum() / n_inp


def beta_schedule(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    """Linear warmup of the KL weight, held at beta_max after warmup."""
    if warmup_epochs <= 0:
        return beta_max
    return min(1.0, epoch / warmup_epochs) * beta_max
