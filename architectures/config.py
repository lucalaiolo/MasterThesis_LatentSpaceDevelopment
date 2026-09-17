"""Training configuration for the masked neonate-motion VAE.

One dataclass holds every knob. Defaults follow the architectures note
[ARCH §6.1, §6.2]: convolutional model with base channel count 64 and
kernels 5, 3, 3, or transformer with model width 96 and three layers of
four heads. The recipe selector picks one of the three training regimes
of [MVAE §3-5].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainingConfig:
    """All the settings a training run needs.

    Attributes:
        clip_length: T, the number of frames per clip.
        n_joints: J, the joint count of the skeleton. Generic.
        n_dims: D, the coordinate dimension per joint. 3 (default) for 3D
            motion capture; 2 for image-plane keypoints (e.g. OpenPose /
            COCO 2D pose). Every model, loss, and parameter count is generic
            in D, so a 2D dataset only needs ``n_dims=2`` — no other change.
        fps: recording rate, used only for schedule reporting.
        architecture: which model to build. "conv" or "transformer" compress
            the clip to one latent vector; "temporal_conv" (conv backbone) and
            "temporal_transformer" (attention backbone) keep a latent **per
            time-window** (the T2M-GPT / MotionGPT style) so the reconstruction
            can move instead of collapsing to a static mean pose
            ([ARCH §4.5-4.6]). For the temporal models the effective latent is
            ``latent_dim * (clip_length / downsample)`` — ``latent_dim`` is the
            per-window width. "temporal_conv" sets the down-sampling from the
            conv strides; "temporal_transformer" from ``temporal_downsample``.
        temporal_downsample: window length ``l`` for "temporal_transformer" —
            the T frames are grouped into ``T/l`` latent windows.
            ``clip_length`` must divide it. Ignored by the other architectures
            (the conv models get their down-sampling from ``conv_strides``).
        latent_dim: d_z, the latent width. For the temporal models this is the
            width of *each* window's latent, not the whole clip's.
        conv_base_channels: C in [ARCH §3].
        conv_kernel_sizes: the three encoder kernels.
        conv_strides: the three encoder strides; the product sets the
            downsampling factor. With (1, 2, 2) the encoder halves T twice.
        d_model: the transformer model width.
        n_heads: attention head count.
        n_layers: transformer block count used for *both* the encoder and
            the decoder unless overridden per side (below). Free to set
            deeper than the [ARCH §6.1] default of 3 — the pre-norm stacks
            carry a terminal LayerNorm so depth stays well-conditioned.
        n_enc_layers: optional encoder-only block count. ``None`` (default)
            falls back to ``n_layers``; set it to make the encoder deeper
            (or shallower) than the decoder.
        n_dec_layers: optional decoder-only block count. ``None`` (default)
            falls back to ``n_layers``.
        ffn_ratio: feedforward inner width as a multiple of d_model.
        dropout: applied after attention and after the feedforward.
        transformer_attention: how the transformer attends ([ARCH §4]).
            "temporal" (default) — the frame-token model: each frame is one
            token (its J joints flattened) and attention runs over time only.
            "factorized" — one token per (joint, frame) with alternating
            **spatial** attention (across joints within a frame) and
            **temporal** attention (across frames within a joint), the
            divided space-time / PoseFormer construction. Ignored when
            ``architecture == "conv"``.
        batch_size: B in [MVAE §6.4].
        n_epochs: total training epochs.
        learning_rate: peak learning rate.
        weight_decay: AdamW weight decay.
        beta_max: the top of the KL weight schedule (only used when
            `beta_mode="warmup"`).
        warmup_epochs: linear warmup from 0 to beta_max over this many
            epochs, then hold at beta_max (only used when
            `beta_mode="warmup"`).
        beta_mode: how the KL weight is chosen at each training step.
            "warmup" (default) — linear ramp from 0 to `beta_max` over
            `warmup_epochs`, matching [MVAE §6.2].
            "delayed_warmup" — hold beta at `beta_min` for
            `delay_epochs`, then linearly ramp to `beta_max` over
            `warmup_epochs`. Lets the AE settle into a good
            reconstruction regime before KL pressure kicks in.
            "computed" — Asperti-Trentin 2020 recipe: track gamma_sq
            as the running minimum of batch MSE and set the effective
            KL weight to 2 * gamma_sq. Keeps the ratio between
            reconstruction and KL constant across training, so latents
            get activated one at a time as reconstruction improves.
            `beta_max` and `warmup_epochs` are ignored in this mode.
        beta_min: floor for the KL weight during phase 1 of
            `delayed_warmup`. 0.0 (default) disables KL entirely during
            pre-training; small positive values keep some regularisation.
        delay_epochs: number of epochs to hold beta at `beta_min` before
            the ramp in `delayed_warmup` mode.
        free_bits: when > 0, replaces the vanilla KL with the free-bits
            KL of Kingma et al., 2016 ([MVAE §6.3]). Each latent
            dimension gets a per-sample floor of `free_bits` nats, so
            dims below the floor stop receiving gradient through the KL
            term. Typical range [0.05, 0.5]. Zero (default) keeps the
            vanilla KL. Compatible with either `beta_mode`.
        checkpoint_metric: which validation quantity picks the ``best.pt``
            checkpoint. Default ``"loss"``
        recipe: one of 1, 2, or 3, matching [MVAE §3-5].
        lambda_aux: weight on the auxiliary reconstruction term. Recipe 2
            weights the masked-pass reconstruction ([MVAE §4.2]);
            Recipe 3 weights the hidden-only inpainting head ([MVAE §5.2]).
            Ignored for Recipe 1.
        lambda_velocity: weight on the optional velocity (temporal-
            difference) reconstruction term. 0.0 (default) disables it.
            When > 0 the loss adds ``lambda_velocity *
            reconstruction_velocity_mse(x_hat, x)`` on the full-clip
            reconstruction, so the generated motion — not just the per-
            frame pose — is scored against the target. Cheap regulariser
            against temporal jitter; applies to every recipe.
        mask_policy: one of "none", "uniform", "top_k_speed",
            "softmax_speed", "per_frame_speed", "limb". See
            `mask_policies.py` for the definitions ([MVAE §2]).
        mask_rho: target hidden fraction. Used by every policy except
            "none" and "limb".
        mask_softmax_temperature: softmax temperature for the
            "softmax_speed" policy. tau -> 0 recovers "top_k_speed";
            tau -> infinity recovers "uniform".
        mask_limb_names: limb names to pick from for the "limb" policy.
        mask_limb_speed_weighted: sample limbs with probability
            proportional to their mean speed instead of uniformly.
        device: "cuda" or "cpu".
        seed: seed for every stochastic step of training.
        log_every: log the running loss every this many steps.
        save_every: checkpoint every this many epochs; 0 disables.
        out_dir: directory for checkpoints and logs.
    """

    # Data.
    clip_length: int = 32
    n_joints: int = 22
    n_dims: int = 3
    fps: int = 25

    # Model.
    architecture: Literal["conv", "transformer", "temporal_conv",
                          "temporal_transformer"] = "conv"
    latent_dim: int = 32

    conv_base_channels: int = 64
    conv_kernel_sizes: tuple[int, int, int] = (5, 3, 3)
    conv_strides: tuple[int, int, int] = (1, 2, 2)

    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    n_enc_layers: int | None = None
    n_dec_layers: int | None = None
    ffn_ratio: int = 4
    dropout: float = 0.1
    temporal_downsample: int = 4
    transformer_attention: Literal["temporal", "factorized"] = "temporal"

    # Training.
    batch_size: int = 64
    n_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    beta_max: float = 1.0
    warmup_epochs: int = 10
    beta_mode: Literal["warmup", "delayed_warmup", "computed"] = "warmup"
    beta_min: float = 0.0
    delay_epochs: int = 20
    free_bits: float = 0.0
    checkpoint_metric: Literal["rec_full", "elbo", "loss"] = "loss"

    # Recipe.
    recipe: Literal[1, 2, 3] = 1
    lambda_aux: float = 1.0
    lambda_velocity: float = 0.0

    # Masking.
    mask_policy: Literal["none", "uniform", "top_k_speed",
                         "softmax_speed", "per_frame_speed",
                         "limb"] = "uniform"
    mask_rho: float = 0.3
    mask_softmax_temperature: float = 1.0
    mask_limb_names: list[str] = field(default_factory=list)
    mask_limb_speed_weighted: bool = False

    # Runtime.
    device: str = "cuda"
    seed: int = 0
    log_every: int = 50
    save_every: int = 10
    out_dir: str = "checkpoints"

    def downsample_factor(self) -> int:
        """Product of the three encoder strides. The encoder divides T by this."""
        f = 1
        for s in self.conv_strides:
            f *= s
        return f

    def encoder_layers(self) -> int:
        """Transformer encoder depth, resolving the per-side override."""
        return self.n_layers if self.n_enc_layers is None else self.n_enc_layers

    def decoder_layers(self) -> int:
        """Transformer decoder depth, resolving the per-side override."""
        return self.n_layers if self.n_dec_layers is None else self.n_dec_layers

    def validate(self) -> None:
        """Cheap checks that catch bad settings before the run starts."""
        if self.clip_length % self.downsample_factor() != 0:
            raise ValueError(
                f"clip_length ({self.clip_length}) must divide by the product "
                f"of conv strides ({self.downsample_factor()})."
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must divide by n_heads "
                f"({self.n_heads})."
            )
        if (self.architecture == "temporal_transformer"
                and self.clip_length % self.temporal_downsample != 0):
            raise ValueError(
                f"clip_length ({self.clip_length}) must divide by "
                f"temporal_downsample ({self.temporal_downsample})."
            )
        if self.encoder_layers() < 1 or self.decoder_layers() < 1:
            raise ValueError(
                f"transformer depth must be >= 1 per side (got encoder="
                f"{self.encoder_layers()}, decoder={self.decoder_layers()}; "
                f"n_layers={self.n_layers}, n_enc_layers={self.n_enc_layers}, "
                f"n_dec_layers={self.n_dec_layers})."
            )
        if (self.transformer_attention == "factorized"
                and (self.n_enc_layers is not None
                     or self.n_dec_layers is not None)):
            raise ValueError(
                "n_enc_layers / n_dec_layers have no effect with "
                "transformer_attention='factorized': this backbone shares one "
                "n_layers across both stacks. Set n_layers instead, or use "
                "transformer_attention='temporal' for per-side depth."
            )
        if self.n_dims < 1:
            raise ValueError(
                f"n_dims ({self.n_dims}) must be >= 1 (2 for 2D keypoints, "
                f"3 for 3D motion capture)."
            )
        if self.recipe in (2, 3) and self.mask_policy == "none":
            raise ValueError(
                f"Recipe {self.recipe} needs a mask policy; set 'uniform' or 'limb'. "
                f"Recipe 2's auxiliary pass and Recipe 3's inpainting head "
                f"both require joints to be hidden."
            )
        if self.lambda_velocity < 0:
            raise ValueError(
                f"lambda_velocity ({self.lambda_velocity}) must be >= 0 "
                f"(0 disables the velocity term)."
            )
