"""Analytical parameter counts.

Reproduces the numbers in [ARCH §6.1]. Useful for two things: matching
the design note before training, and sizing a model to a target budget
by scanning C or d_model.

All counts include bias terms and exclude LayerNorm parameters, which
the design note rounds away (under 1k in total for either model at the
default sizes).
"""

from __future__ import annotations

from .config import TrainingConfig


def conv_param_count(config: TrainingConfig) -> dict[str, int]:
    """Convolutional model parameter counts by component ([ARCH §6.1])."""
    T = config.clip_length
    J = config.n_joints
    D = getattr(config, "n_dims", 3)
    d_z = config.latent_dim
    C = config.conv_base_channels
    k = config.conv_kernel_sizes
    d = config.downsample_factor()
    T_bot = T // d

    def conv1d(c_in, c_out, kernel):
        return kernel * c_in * c_out + c_out

    def linear(inp, out):
        return inp * out + out

    parts = {
        "encoder_block_1": conv1d((D + 1) * J, C, k[0]),
        "encoder_block_2": conv1d(C, 2 * C, k[1]),
        "encoder_block_3": conv1d(2 * C, 2 * C, k[2]),
        "bottleneck_heads": 2 * linear(2 * C * T_bot, d_z),
        "decoder_lift": linear(d_z, 2 * C * T_bot),
        "decoder_block_3": conv1d(2 * C, 2 * C, k[2]),
        "decoder_block_2": conv1d(2 * C, C, k[1]),
        "decoder_output_full": conv1d(C, D * J, k[0]),
    }
    # Recipe 3 adds a second, mask-conditioned output head ([MVAE §5.1]).
    if config.recipe == 3:
        parts["decoder_output_inp"] = conv1d(C + J, D * J, k[0])
    parts["total"] = sum(parts.values())
    return parts


def transformer_param_count(config: TrainingConfig) -> dict[str, int]:
    """Transformer model parameter counts by component ([ARCH §6.1])."""
    J = config.n_joints
    D = getattr(config, "n_dims", 3)
    d_z = config.latent_dim
    dm = config.d_model
    L = config.n_layers
    ffn = config.ffn_ratio * dm

    def linear(inp, out):
        return inp * out + out

    # A pre-norm block: q, k, v, and output projections, then a
    # two-layer feedforward. PyTorch's TransformerEncoderLayer packs Q,
    # K, V into one linear (3 dm dm + 3 dm) plus the output linear
    # (dm dm + dm); the feedforward is (dm ffn + ffn) then (ffn dm + dm).
    def attention_block():
        return 3 * dm * dm + 3 * dm + dm * dm + dm

    def ffn_block():
        return dm * ffn + ffn + ffn * dm + dm

    def transformer_block():
        return attention_block() + ffn_block()

    parts = {
        "encoder_token_embed": linear((D + 1) * J, dm),
        "encoder_class_token": dm,
        "encoder_stack": L * transformer_block(),
        "bottleneck_heads": 2 * linear(dm, d_z),
        "decoder_query_lift": linear(d_z, dm),
        "decoder_stack": L * transformer_block(),
        "decoder_output_full": linear(dm, D * J),
    }
    # Recipe 3 adds a second, mask-conditioned output head ([MVAE §5.1]).
    if config.recipe == 3:
        parts["decoder_output_inp"] = linear(dm + J, D * J)
    parts["total"] = sum(parts.values())
    return parts


def summarise(config: TrainingConfig) -> dict[str, int]:
    """One entry point: return the counts for whichever architecture is set."""
    if config.architecture == "conv":
        return conv_param_count(config)
    if config.architecture == "transformer":
        return transformer_param_count(config)
    raise ValueError(f"unknown architecture: {config.architecture!r}")
