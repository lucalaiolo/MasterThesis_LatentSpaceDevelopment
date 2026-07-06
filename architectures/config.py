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
        fps: recording rate, used only for schedule reporting.
        architecture: which model to build, "conv" or "transformer".
        latent_dim: d_z, the latent width.
        conv_base_channels: C in [ARCH §3].
        conv_kernel_sizes: the three encoder kernels.
        conv_strides: the three encoder strides; the product sets the
            downsampling factor. With (1, 2, 2) the encoder halves T twice.
        d_model: the transformer model width.
        n_heads: attention head count.
        n_layers: transformer block count in each of the encoder and decoder.
        ffn_ratio: feedforward inner width as a multiple of d_model.
        dropout: applied after attention and after the feedforward.
        batch_size: B in [MVAE §6.4].
        n_epochs: total training epochs.
        learning_rate: peak learning rate.
        weight_decay: AdamW weight decay.
        beta_max: the top of the KL weight schedule.
        warmup_epochs: linear warmup from 0 to beta_max over this many
            epochs, then hold at beta_max.
        recipe: one of 1, 2, or 3, matching [MVAE §3-5].
        lambda_aux: weight on the auxiliary reconstruction term. Recipe 2
            weights the masked-pass reconstruction ([MVAE §4.2]);
            Recipe 3 weights the hidden-only inpainting head ([MVAE §5.2]).
            Ignored for Recipe 1.
        mask_policy: "none", "uniform", or "limb".
        mask_uniform_rho: fraction of joints hidden per frame under the
            uniform policy.
        mask_limb_names: limb names to pick from for the limb policy.
        device: "cuda" or "cpu".
        seed: seed for every stochastic step of training.
        log_every: log the running loss every this many steps.
        save_every: checkpoint every this many epochs; 0 disables.
        out_dir: directory for checkpoints and logs.
    """

    # Data.
    clip_length: int = 32
    n_joints: int = 22
    fps: int = 25

    # Model.
    architecture: Literal["conv", "transformer"] = "conv"
    latent_dim: int = 32

    conv_base_channels: int = 64
    conv_kernel_sizes: tuple[int, int, int] = (5, 3, 3)
    conv_strides: tuple[int, int, int] = (1, 2, 2)

    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    ffn_ratio: int = 4
    dropout: float = 0.1

    # Training.
    batch_size: int = 64
    n_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    beta_max: float = 1.0
    warmup_epochs: int = 10

    # Recipe.
    recipe: Literal[1, 2, 3] = 1
    lambda_aux: float = 1.0

    # Masking.
    mask_policy: Literal["none", "uniform", "limb"] = "uniform"
    mask_uniform_rho: float = 0.3
    mask_limb_names: list[str] = field(default_factory=list)

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
        if self.recipe in (2, 3) and self.mask_policy == "none":
            raise ValueError(
                f"Recipe {self.recipe} needs a mask policy; set 'uniform' or 'limb'. "
                f"Recipe 2's auxiliary pass and Recipe 3's inpainting head "
                f"both require joints to be hidden."
            )
