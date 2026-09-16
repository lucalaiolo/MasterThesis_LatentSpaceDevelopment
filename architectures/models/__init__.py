"""The two model families and a factory to build either from a config."""

from .conv_vae import ConvVAE
from .transformer_vae import TransformerVAE
from .spatiotemporal_vae import SpatioTemporalTransformerVAE
from .temporal_conv_vae import TemporalConvVAE
from .temporal_transformer_vae import TemporalTransformerVAE

__all__ = ["ConvVAE", "TransformerVAE", "SpatioTemporalTransformerVAE",
           "TemporalConvVAE", "TemporalTransformerVAE", "build_model"]


def build_model(config):
    """Build the model the config asks for."""
    inpainting = config.recipe == 3
    n_dims = getattr(config, "n_dims", 3)
    if config.architecture == "temporal_transformer":
        # The per-window latent is identical for both attention patterns; only
        # the encoder/decoder trunk differs. "temporal" (frame tokens) is the
        # default; "factorized" uses divided space-time attention.
        return TemporalTransformerVAE(
            T=config.clip_length,
            J=config.n_joints,
            d_z=config.latent_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
            inpainting=inpainting,
            n_dims=n_dims,
            downsample=getattr(config, "temporal_downsample", 4),
            attention=getattr(config, "transformer_attention", "temporal"),
        )
    if config.architecture in ("conv", "temporal_conv"):
        cls = TemporalConvVAE if config.architecture == "temporal_conv" else ConvVAE
        return cls(
            T=config.clip_length,
            J=config.n_joints,
            d_z=config.latent_dim,
            base_channels=config.conv_base_channels,
            kernels=config.conv_kernel_sizes,
            strides=config.conv_strides,
            inpainting=inpainting,
            n_dims=n_dims,
        )
    if config.architecture == "transformer":
        common = dict(
            T=config.clip_length,
            J=config.n_joints,
            d_z=config.latent_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
            inpainting=inpainting,
            n_dims=n_dims,
        )
        if getattr(config, "transformer_attention", "temporal") == "factorized":
            # SpatioTemporalTransformerVAE shares one n_layers across both
            # stacks (see TrainingConfig.validate — the per-side override
            # is rejected for this attention mode), so it takes no
            # n_enc_layers / n_dec_layers kwargs.
            return SpatioTemporalTransformerVAE(**common)
        return TransformerVAE(
            **common,
            n_enc_layers=getattr(config, "n_enc_layers", None),
            n_dec_layers=getattr(config, "n_dec_layers", None),
        )
    raise ValueError(f"unknown architecture: {config.architecture!r}")
