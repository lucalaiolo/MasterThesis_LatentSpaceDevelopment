"""Analytical parameter counts.

Reproduces the numbers in [ARCH §6.1]. Useful for two things: matching
the design note before training, and sizing a model to a target budget
by scanning C or d_model.

All counts include bias terms and exclude LayerNorm parameters (the
per-block norms plus the transformer's terminal pre-norm LayerNorm),
which the design note rounds away — a small, depth-linear remainder that
the smoke test reports as the actual-vs-analytical delta.
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
    L_enc = config.encoder_layers()
    L_dec = config.decoder_layers()
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
        "encoder_stack": L_enc * transformer_block(),
        "bottleneck_heads": 2 * linear(dm, d_z),
        "decoder_query_lift": linear(d_z, dm),
        "decoder_stack": L_dec * transformer_block(),
        "decoder_output_full": linear(dm, D * J),
    }
    # Recipe 3 adds a second, mask-conditioned output head ([MVAE §5.1]).
    if config.recipe == 3:
        parts["decoder_output_inp"] = linear(dm + J, D * J)
    parts["total"] = sum(parts.values())
    return parts


def spatiotemporal_param_count(config: TrainingConfig) -> dict[str, int]:
    """Factorised space-time transformer parameter counts by component.

    One token per (joint, frame): the token embed is D+1 -> d_model, a learned
    joint (spatial) embedding of J*d_model is added on each of the encoder and
    decoder, and every factorised block holds *two* transformer sub-layers
    (spatial + temporal), so the stacks cost ``2 * n_layers`` blocks. Like the
    frame-token count, this excludes LayerNorm scales/biases and assumes
    ``n_cond == 0``.
    """
    J = config.n_joints
    D = getattr(config, "n_dims", 3)
    d_z = config.latent_dim
    dm = config.d_model
    L = config.n_layers
    ffn = config.ffn_ratio * dm

    def linear(inp, out):
        return inp * out + out

    def attention_block():
        return 3 * dm * dm + 3 * dm + dm * dm + dm

    def ffn_block():
        return dm * ffn + ffn + ffn * dm + dm

    def transformer_block():
        return attention_block() + ffn_block()

    parts = {
        "encoder_token_embed": linear(D + 1, dm),
        "encoder_joint_pos": J * dm,
        "encoder_stack": L * 2 * transformer_block(),   # spatial + temporal
        # Bottleneck reads every joint (pool over time, keep joints).
        "bottleneck_heads": 2 * linear(J * dm, d_z),
        "decoder_query_lift": linear(d_z, dm),
        "decoder_joint_pos": J * dm,
        "decoder_stack": L * 2 * transformer_block(),
        "decoder_output_full": linear(dm, D),
    }
    if config.recipe == 3:
        parts["decoder_output_inp"] = linear(dm + 1, D)
    parts["total"] = sum(parts.values())
    return parts


def anchored_param_count(config: TrainingConfig) -> dict[str, int]:
    """Anchored residual space-time transformer parameter counts.

    The factorised backbone (``2 * n_layers`` sub-layer blocks per stack) plus:
    a shared conditioning MLP ``(D*J + 1) -> d_model -> d_model``; a learned
    mask token and one FiLM site in the encoder; a FiLM site before each of the
    two sub-attentions of every decoder block. Each FiLM is a
    ``d_model -> 2*d_model`` linear. The residual token embed is ``D -> d_model``
    and the output head is ``d_model -> D``. Excludes LayerNorm scales/biases
    and assumes ``n_cond == 0`` (the model has no cohort path).
    """
    J = config.n_joints
    D = getattr(config, "n_dims", 3)
    d_z = config.latent_dim
    dm = config.d_model
    L = config.n_layers
    ffn = config.ffn_ratio * dm

    def linear(inp, out):
        return inp * out + out

    def transformer_block():
        attn = 3 * dm * dm + 3 * dm + dm * dm + dm
        ff = dm * ffn + ffn + ffn * dm + dm
        return attn + ff

    film = linear(dm, 2 * dm)

    parts = {
        "cond_mlp": linear(D * J + 1, dm) + linear(dm, dm),
        "encoder_token_embed": linear(D, dm),
        "encoder_mask_token": dm,
        "encoder_joint_pos": J * dm,
        "encoder_film": film,                           # single encoder site
        "encoder_stack": L * 2 * transformer_block(),
        "bottleneck_heads": 2 * linear(dm, d_z),
        "decoder_query_lift": linear(d_z, dm),
        "decoder_joint_pos": J * dm,
        "decoder_stack": L * 2 * transformer_block(),
        "decoder_film": L * 2 * film,                   # one per sub-attention
        "decoder_output_full": linear(dm, D),
    }
    if config.recipe == 3:
        parts["decoder_output_inp"] = linear(dm + 1, D)
    parts["total"] = sum(parts.values())
    return parts


def anchored_temporal_param_count(config: TrainingConfig) -> dict[str, int]:
    """Anchored frame-token transformer parameter counts.

    The frame-token backbone (``n_layers`` single-axis blocks per stack, a class
    token, terminal norms) plus the anchor machinery: a conditioning MLP
    ``(D*J + 1) -> d_model -> d_model``; one encoder FiLM site; a FiLM site
    before every decoder layer. The residual frame token embeds ``(D+1)*J`` and
    the output head is ``d_model -> D*J``. Excludes LayerNorm scales/biases and
    assumes ``n_cond == 0``.
    """
    J = config.n_joints
    D = getattr(config, "n_dims", 3)
    d_z = config.latent_dim
    dm = config.d_model
    L = config.n_layers
    ffn = config.ffn_ratio * dm

    def linear(inp, out):
        return inp * out + out

    def transformer_block():
        attn = 3 * dm * dm + 3 * dm + dm * dm + dm
        ff = dm * ffn + ffn + ffn * dm + dm
        return attn + ff

    film = linear(dm, 2 * dm)

    parts = {
        "cond_mlp": linear(D * J + 1, dm) + linear(dm, dm),
        "encoder_token_embed": linear((D + 1) * J, dm),
        "encoder_class_token": dm,
        "encoder_film": film,                           # single encoder site
        "encoder_stack": L * transformer_block(),
        "bottleneck_heads": 2 * linear(dm, d_z),
        "decoder_query_lift": linear(d_z, dm),
        "decoder_stack": L * transformer_block(),
        "decoder_film": L * film,                       # one per decoder layer
        "decoder_output_full": linear(dm, D * J),
    }
    if config.recipe == 3:
        parts["decoder_output_inp"] = linear(dm + J, D * J)
    parts["total"] = sum(parts.values())
    return parts


def summarise(config: TrainingConfig) -> dict[str, int]:
    """One entry point: return the counts for whichever architecture is set."""
    if config.architecture == "conv":
        return conv_param_count(config)
    if config.architecture == "transformer":
        attention = getattr(config, "transformer_attention", "temporal")
        if getattr(config, "anchored_residual", False):
            return (anchored_param_count(config) if attention == "factorized"
                    else anchored_temporal_param_count(config))
        if attention == "factorized":
            return spatiotemporal_param_count(config)
        return transformer_param_count(config)
    raise ValueError(f"unknown architecture: {config.architecture!r}")
