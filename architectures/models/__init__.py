"""The two model families and a factory to build either from a config."""

from .conv_vae import ConvVAE
from .transformer_vae import TransformerVAE

__all__ = ["ConvVAE", "TransformerVAE", "build_model"]


def build_model(config):
    """Build the model the config asks for."""
    inpainting = config.recipe == 3
    if config.architecture == "conv":
        return ConvVAE(
            T=config.clip_length,
            J=config.n_joints,
            d_z=config.latent_dim,
            base_channels=config.conv_base_channels,
            kernels=config.conv_kernel_sizes,
            strides=config.conv_strides,
            inpainting=inpainting,
        )
    if config.architecture == "transformer":
        return TransformerVAE(
            T=config.clip_length,
            J=config.n_joints,
            d_z=config.latent_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
            inpainting=inpainting,
        )
    raise ValueError(f"unknown architecture: {config.architecture!r}")
