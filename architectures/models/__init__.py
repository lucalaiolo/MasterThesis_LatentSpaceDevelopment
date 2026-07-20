"""The two model families and a factory to build either from a config."""

from .conv_vae import ConvVAE
from .transformer_vae import TransformerVAE
from .spatiotemporal_vae import SpatioTemporalTransformerVAE
from .anchored_vae import AnchoredSpatioTemporalVAE, AnchoredTemporalVAE
from .temporal_conv_vae import TemporalConvVAE
from .temporal_transformer_vae import TemporalTransformerVAE
from .gaussian_mixture import GaussianMixturePrior

__all__ = ["ConvVAE", "TransformerVAE", "SpatioTemporalTransformerVAE",
           "AnchoredSpatioTemporalVAE", "AnchoredTemporalVAE",
           "TemporalConvVAE", "TemporalTransformerVAE", "GaussianMixturePrior",
           "build_model", "build_mixture"]


def build_model(config):
    """Build the model the config asks for.

    Conditioning (``n_cond > 0``) turns the model into a CVAE / GM-CVAE.
    The mixture prior itself is a separate object — see ``build_mixture``.
    """
    inpainting = config.recipe == 3
    n_dims = getattr(config, "n_dims", 3)
    if config.architecture == "temporal_transformer":
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
            n_cond=config.n_cond,
            cond_dim=config.cond_dim,
            cond_dropout=config.cond_dropout,
            n_dims=n_dims,
            downsample=getattr(config, "temporal_downsample", 4),
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
            n_cond=config.n_cond,
            cond_dim=config.cond_dim,
            cond_dropout=config.cond_dropout,
            n_dims=n_dims,
        )
    if config.architecture == "transformer":
        attention = getattr(config, "transformer_attention", "temporal")
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
            n_cond=config.n_cond,
            cond_dim=config.cond_dim,
            cond_dropout=config.cond_dropout,
            n_dims=n_dims,
        )
        if getattr(config, "anchored_residual", False):
            # Residual + FiLM model, on either backbone (orthogonal to the
            # attention pattern). Shares one n_layers (per-side rejected in
            # validate) and takes the torso-scale joint indices.
            anchor_cls = (AnchoredSpatioTemporalVAE if attention == "factorized"
                          else AnchoredTemporalVAE)
            return anchor_cls(
                **common,
                shoulder_joints=getattr(config, "anchor_shoulder_joints", None),
                hip_joints=getattr(config, "anchor_hip_joints", None),
                scale_eps=getattr(config, "anchor_scale_eps", 1e-3),
            )
        if attention == "factorized":
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


def build_mixture(config):
    """Build the Gaussian-mixture prior for a GM run, or None.

    Returns ``None`` when ``config.n_components == 0`` (standard N(0, I)
    prior). Otherwise a ``GaussianMixturePrior`` with K components over the
    latent dimension, seeded from ``config.seed`` for reproducibility.

    **Deprecated path** ([post-hoc plan §0]): a mixture is only built when
    ``config.allow_deprecated_gmvae`` is set. The GM-VAE / GM-CVAE were
    removed from the active pipeline (component collapse); the phenotype
    claim now runs post hoc on the plain VAE / CVAE latents. Kept callable
    for the record, guarded so no default run reaches it by accident.
    """
    if config.n_components == 0:
        return None
    if not getattr(config, "allow_deprecated_gmvae", False):
        raise DeprecationWarning(
            "build_mixture: GM-VAE / GM-CVAE is a deprecated path "
            "([post-hoc plan §0]). Set allow_deprecated_gmvae=True to opt in."
        )
    return GaussianMixturePrior(
        n_components=config.n_components,
        d_z=config.latent_dim,
        var_floor=config.gm_var_floor,
        init_spread=config.gm_init_spread,
        seed=config.seed,
        trainable=(config.gm_train == "gradient"),
    )
