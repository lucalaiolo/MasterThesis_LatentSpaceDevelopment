"""ELBO components and the reconstruction losses for the three recipes.

The training loop calls these each step:

    kl_gaussian                KL between the diagonal Gaussian posterior
                               and the standard-normal prior, per sample.
    kl_gaussian_free_bits      Per-dimension free-bits variant of the
                               KL: dims below the threshold contribute a
                               fixed cost, so the encoder is not pushed
                               to squash any single dim to zero
                               ([MVAE §6.3], Kingma et al., 2016).
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


def kl_gaussian_free_bits(mu, logvar, gamma: float):
    """Per-dimension free-bits KL ([MVAE §6.3]; Kingma et al., 2016).

    Floors each dimension's KL at `gamma` before summing across dims:

        KL_tilde = sum_d max(gamma, KL_d).

    Effect on training: dimensions with KL_d < gamma receive no
    gradient through the KL term, so the encoder is not pushed to
    squash them further. More robust than beta-annealing when the
    latent is small and a handful of dimensions carry all the
    posterior information.

    Args:
        mu, logvar: (B, d_z).
        gamma: per-dimension floor. Typical range [0.05, 0.5].
    Returns:
        Tensor of shape (B,), one free-bits KL per sample.
    """
    # KL_d = 1/2 * (mu_d^2 + sigma_d^2 - log sigma_d^2 - 1).
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
    return torch.clamp(kl_per_dim, min=gamma).sum(dim=1)


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


def delayed_warmup_schedule(epoch: int, delay_epochs: int,
                            warmup_epochs: int,
                            beta_min: float, beta_max: float) -> float:
    """Two-phase KL warmup: hold at `beta_min`, then linearly ramp.

    Phase 1 (epoch < delay_epochs): beta = beta_min. Reconstruction
    trains largely unregularised so the encoder / decoder reach a good
    fitting regime before any KL pressure kicks in.

    Phase 2 (delay_epochs <= epoch < delay_epochs + warmup_epochs):
    linear ramp from beta_min to beta_max.

    Phase 3 (epoch >= delay_epochs + warmup_epochs): beta = beta_max.

    Useful when a plain warmup starts KL pressure too early and the
    model can't recover from the initial poor reconstruction.
    """
    if epoch < delay_epochs:
        return beta_min
    if warmup_epochs <= 0:
        return beta_max
    progress = (epoch - delay_epochs) / warmup_epochs
    return beta_min + (beta_max - beta_min) * min(1.0, progress)
