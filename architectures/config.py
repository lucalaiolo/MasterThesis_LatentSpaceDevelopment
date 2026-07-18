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
        n_cond: number of conditioning categories for the CVAE arm
            ([CARE-PD §6]). 0 (default) disables conditioning and gives a
            plain VAE. Set to the cohort count to condition on cohort. The
            data iterator must then yield a per-clip integer id in
            ``[0, n_cond)``.
        cond_dim: width of the learned conditioning embedding e(c)
            ([CARE-PD §6], d_c in [4, 16]). Ignored when ``n_cond == 0``.
        cond_dropout: conditioning-dropout rate ([CARE-PD §6, §10]). With
            this probability the decoder's e(c) is replaced by zeros during
            training so the decoder cannot degenerate into one sub-decoder
            per cohort and must keep using z. Encoder always sees clean
            e(c) so it can *stop* routing cohort into z. Ignored when
            ``n_cond == 0``.
        site_adv_lambda_max: strength ceiling of the gradient-reversal site
            adversary ([Phase 2c]). 0 (default) disables it. When > 0, a
            :class:`SiteAdversary` is trained to predict cohort from the
            posterior mean and its gradient is reversed into the encoder, so
            the encoder is driven to make cohort *un*predictable from the
            latent — the explicit invariance term that conditioning alone
            never provided. Needs ``cohort_per_video`` (the cohort labels)
            at train time, but does **not** feed cohort into the networks:
            set ``n_cond=0`` for the pure adversarial VAE ("goodbye to the
            CVAE" — the encoder is z=f(x) with no cohort leak channel), or
            keep ``n_cond>0`` to combine conditioning with the adversary.
        site_adv_warmup_epochs: epochs over which the reversal strength
            ramps linearly from 0 to ``site_adv_lambda_max``. Starting at
            full strength destabilises training; ramping lets the adversary
            learn cohort first, then the encoder unlearn it. Ignored when
            ``site_adv_lambda_max == 0``.
        site_adv_hidden: hidden width of the adversary MLP (two hidden
            layers, matching the evaluation site probe). Ignored when
            ``site_adv_lambda_max == 0``.
        n_components: K, the number of Gaussian-mixture prior components
            ([CARE-PD §7.3]). 0 (default) keeps the standard N(0, I) prior.
            When > 0 the run trains a GM-VAE (or GM-CVAE if ``n_cond > 0``).
        gm_train: how the mixture parameters are fit ([CARE-PD §7.3]).
            "gradient" (default) — the regular / VaDE regime: the
            component means, log-variances, and weights are trainable
            parameters optimised jointly with the ELBO by the same
            optimiser as the networks. Robust and the recommended default.
            "em" — the EM-inspired block-coordinate scheme of
            [GM-VAE §3.3]: the parameters are refreshed by closed-form EM
            M-steps over the epoch's cached latents while the networks are
            frozen. More faithful to the physics paper but less stable;
            prone to component collapse. Ignored when ``n_components == 0``.
        gm_beta_z: weight on the mixture KL E_q(y)[KL(q(z|x) || p(z|y))]
            (the plan's beta_z, [CARE-PD §7.3]). Ignored when
            ``n_components == 0``.
        gm_beta_y: weight on the categorical KL KL(q(y|x) || p(y))
            (the plan's beta_y). Ignored when ``n_components == 0``.
        gm_aux_beta: weight of the auxiliary KL(q(z|x) || N(0, I)) term for
            GM runs. 0 (default) removes it — in a GM-VAE the mixture *is*
            the prior, so the N(0,I) term is redundant. A small positive
            value re-adds it as the safety tether of [GM-VAE §3.3] (useful
            mainly with ``gm_train="em"``). Note this replaces the beta
            schedule for GM runs: ``beta_max`` and ``beta_mode`` no longer
            weight any prior term when ``n_components > 0``, though
            ``delay_epochs`` / ``warmup_epochs`` still shape the mixture-KL
            ramp via ``gm_kl_warmup``.
        gm_em_steps: number of EM iterations run over the cached epoch
            latents to refresh the mixture parameters after each gradient
            epoch ([GM-VAE Alg. 1], the N_EM inner loop). Ignored when
            ``n_components == 0``.
        gm_var_floor: lower clamp on the per-component variances during the
            EM M-step, guarding against a component collapsing onto a
            single point. Ignored when ``n_components == 0``.
        gm_init_spread: standard deviation used to scatter the initial
            component means, so components start distinguishable rather
            than all at the origin. Ignored when ``n_components == 0``.
        gm_entropy_weight: initial weight of an entropy bonus on the soft
            assignments q(y|x), decayed linearly to zero over
            ``gm_entropy_epochs`` ([CARE-PD §10], component-collapse
            mitigation). Rewards near-uniform assignments early so no
            component dies before the latent has organised. Ignored when
            ``n_components == 0``.
        gm_entropy_epochs: number of epochs over which the entropy bonus
            decays to zero. Ignored when ``n_components == 0``.
        gm_kl_warmup: when True (default), the mixture KL terms (gm_beta_z,
            gm_beta_y) are ramped by the *same* [0, 1] warm-up shape as the
            beta schedule — held at 0 during a ``delayed_warmup`` delay,
            then ramped over ``warmup_epochs``. This lets the "learn to use
            the latent first, apply KL pressure later" recipe cover the
            mixture terms too, not just the N(0, I) regulariser, and echoes
            the brief pre-training phase of [GM-VAE §6]. Set False to hold
            the mixture KL at full strength from epoch 0. Ignored when
            ``n_components == 0``.
        allow_deprecated_gmvae: **DEPRECATED PATH GUARD** ([post-hoc plan
            §0]). The mixture-prior models (GM-VAE / GM-CVAE) were removed
            from the active pipeline: they suffer component collapse when
            the latent is not cleanly multimodal, and the phenotype claim
            is now made post hoc on the plain VAE / CVAE latents instead.
            The source is kept for the record but ``train`` refuses to build
            a mixture (``n_components > 0``) unless this flag is explicitly
            set to True. Leave it False in every default run. Ignored when
            ``n_components == 0``.
        beta_max / warmup_epochs (GM runs): for a GM run the beta schedule
            drives the auxiliary KL(q(z|x) || N(0, I)) regulariser that
            [GM-VAE §3.3, Alg. 1] adds on top of the mixture terms to keep
            the manifold well-conditioned. Keep it small (e.g. 1e-2) or
            zero it out; the mixture KL does the main regularising.
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
    beta_mode: Literal["warmup", "delayed_warmup", "computed"] = "warmup"
    beta_min: float = 0.0
    delay_epochs: int = 20
    free_bits: float = 0.0
    checkpoint_metric: Literal["rec_full", "elbo", "loss"] = "loss"

    # Recipe.
    recipe: Literal[1, 2, 3] = 1
    lambda_aux: float = 1.0

    # Conditioning (CVAE / GM-CVAE arm, [CARE-PD §6]).
    n_cond: int = 0
    cond_dim: int = 8
    cond_dropout: float = 0.15

    # Site adversary (gradient-reversal cohort-invariance, [Phase 2c]).
    site_adv_lambda_max: float = 0.0
    site_adv_warmup_epochs: int = 30
    site_adv_hidden: int = 128

    # Gaussian-mixture prior (GM-VAE / GM-CVAE arm, [CARE-PD §7.3],
    # trained with the EM scheme of [GM-VAE §3.3]).
    n_components: int = 0
    gm_train: Literal["gradient", "em"] = "gradient"
    gm_beta_z: float = 1.0
    gm_beta_y: float = 1.0
    gm_aux_beta: float = 0.0
    gm_em_steps: int = 1
    gm_var_floor: float = 1e-4
    gm_init_spread: float = 1.0
    gm_entropy_weight: float = 0.0
    gm_entropy_epochs: int = 5
    gm_kl_warmup: bool = True
    # Deprecated-path guard ([post-hoc plan §0]): the mixture-prior models
    # are off the active path; training one requires opting in explicitly.
    allow_deprecated_gmvae: bool = False

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
        if self.n_cond < 0:
            raise ValueError(f"n_cond ({self.n_cond}) must be >= 0.")
        if self.n_cond > 0 and self.cond_dim <= 0:
            raise ValueError(
                f"cond_dim ({self.cond_dim}) must be positive when "
                f"conditioning is enabled (n_cond={self.n_cond})."
            )
        if not 0.0 <= self.cond_dropout < 1.0:
            raise ValueError(
                f"cond_dropout ({self.cond_dropout}) must be in [0, 1)."
            )
        if self.site_adv_lambda_max < 0:
            raise ValueError(
                f"site_adv_lambda_max ({self.site_adv_lambda_max}) must be "
                f">= 0 (0 disables the site adversary)."
            )
        if self.site_adv_lambda_max > 0 and self.site_adv_hidden <= 0:
            raise ValueError(
                f"site_adv_hidden ({self.site_adv_hidden}) must be positive "
                f"when the site adversary is enabled."
            )
        if self.n_components < 0:
            raise ValueError(
                f"n_components ({self.n_components}) must be >= 0."
            )
        if self.n_components == 1:
            raise ValueError(
                "n_components == 1 is a plain VAE with a shifted prior; set "
                "0 for the N(0, I) prior or >= 2 for a real mixture."
            )
        if self.n_components > 0 and not self.allow_deprecated_gmvae:
            raise DeprecationWarning(
                "GM-VAE / GM-CVAE (n_components > 0) is a DEPRECATED path "
                "([post-hoc plan §0]): the mixture prior collapses when the "
                "latent is not cleanly multimodal, and phenotype structure "
                "is now recovered post hoc on the plain VAE / CVAE latents "
                "(see vae_analysis.posthoc). The source is kept for the "
                "record only. To run it anyway — against the plan — set "
                "TrainingConfig(allow_deprecated_gmvae=True) explicitly."
            )
        if self.n_components > 0 and self.gm_em_steps < 1:
            raise ValueError(
                f"gm_em_steps ({self.gm_em_steps}) must be >= 1 for a GM run."
            )
