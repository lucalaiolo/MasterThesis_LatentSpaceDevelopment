"""Disentangled RVQ-VAE model package ([paper Sec. 3.1.1])."""

from .gaitgen import DisentangledRVQVAE
from .layers import gradient_reverse, SelfAttnPool
from .rvq import ResidualVQ, VectorQuantizerEMA

__all__ = ["DisentangledRVQVAE", "build_model", "gradient_reverse",
           "SelfAttnPool", "ResidualVQ", "VectorQuantizerEMA"]


def build_model(config):
    """Build the disentangled RVQ-VAE from a :class:`GaitGenConfig`."""
    return DisentangledRVQVAE(config)
