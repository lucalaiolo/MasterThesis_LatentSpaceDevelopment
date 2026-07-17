"""Configuration for the neonate GAITGen (disentangled RVQ-VAE).

One dataclass holds every knob. Defaults follow the GAITGen paper
(Sec. 3.1.1, Eqs. 3-8, and the two-stage schedule) with the neonate
adaptations from the build plan:

- input is **2D** keypoints, so there is no SO(3) rotation term (the paper's
  geodesic loss Eq. 4 is dropped; only the L1 position loss Eq. 3 remains);
- the conditioning label is generic (``label_type`` selects ordinal vs
  nominal), not hard-coded to CARE-PD severity.

Nothing here assumes a particular joint count ``J`` — set it (and the
optional ``bone_pairs``) once the real dataset shape is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GaitGenConfig:
    """Every setting a neonate-GAITGen run needs.

    Attributes:
        n_joints: J, the number of 2D keypoints. Set to the real value.
        in_channels: coordinates per joint (2 for 2D). The encoder input
            width is ``n_joints * in_channels`` after flattening.
        clip_length: T, frames per window (60, [plan §2.4]).
        stride: hop between windows (50% overlap → ``clip_length // 2``).
        fps: native frame rate, for reporting only (not resampled).
        root_joint: index of the pelvis / torso-midpoint used for
            root-centring. If < 0, the per-frame mean of all joints is used.
        torso_joints: ``(top, bottom)`` keypoint indices whose distance is
            the torso length used for scale normalisation (shoulder-mid to
            hip-mid). If either is < 0, the per-clip mean bone length is used.

        label_type: ``"ordinal"`` (severity-like) or ``"nominal"`` (binary /
            multi-class). Only downstream analyses that assume an ordering
            read this ([plan §4]); the architecture is agnostic.
        n_classes: C, number of pathology classes for ``c_p``.
        healthy_id: the class id treated as "healthy" for the latent-dropout
            strategy (``ADD-H_zeroout``, [plan §3.5], paper Table 5). When a
            sample's ``c_p == healthy_id`` its ``q_p`` is zeroed so the motion
            latent must reconstruct healthy motion alone.
        n_nuisance: number of optional nuisance categories for the secondary
            motion adversary ([plan §4]). 0 disables it.

        d_motion / d_pathology: D_m, D_p, the latent (and codebook) widths
            (64 each, paper). Kept equal so ``q_m + alpha * q_p`` is defined.
        cond_dim: width of the pathology-condition embedding e(c_p) (8).
        hidden_channels: conv width inside the encoders/decoder (512).
        downsample: temporal downsampling factor of the encoders (4). T must
            be divisible by it.
        n_rvq_layers: N, residual-quantisation layers (6).
        codebook_motion / codebook_pathology: #cb_m, #cb_p (512 / 128 — the
            pathology codebook is deliberately smaller to constrain capacity).
        ema_decay: EMA decay for the codebook updates.
        quant_dropout: probability of truncating the residual layers during
            training (RVQ quantisation dropout, 0.2).
        codebook_reset_threshold: reset a code whose usage falls below this
            fraction (dead-code revival, paper's reset strategy).
        alpha: interference weight in ``x_hat = D(q_m + alpha * q_p)`` (1.0
            for training; a scalar knob for analysis).

        lambda_rec / lambda_cls / lambda_adv / lambda_emb: loss weights
            (1, 0.01, 0.01, 0.02, paper Eq. 7). Do not tune first pass.
        bone_weight: optional bone-length consistency term ([plan §3.4], 0
            disables; 0.1 suggested). Uses ``bone_pairs``.
        bone_pairs: connected keypoint index pairs for the bone term
            (shoulder-elbow, elbow-wrist, hip-knee, knee-ankle, mirrored).

        stage1_epochs / stage2_epochs: 200 / 300 (paper).
        lr_stage1: E_m + D pretraining lr (2e-6).
        lr_motion_stage2 / lr_pathology / lr_classifier: stage-2 lrs
            (2e-7 / 2e-6 / 2e-7). E_m is fine-tuned slower to retain motion.
        grl_lambda_max / grl_warmup_epochs: adversarial gradient-reversal
            strength and its ramp within stage 2 (start at 0 to stabilise).
        batch_size: 512 (paper).
        weight_decay / seed / device / out_dir / log_every: run controls.
    """

    # ---- Data ----
    n_joints: int = 17
    in_channels: int = 2
    clip_length: int = 60
    stride: int = 30
    fps: float = 30.0
    root_joint: int = 0
    torso_joints: tuple[int, int] = (0, 1)

    # ---- Conditioning ----
    label_type: Literal["ordinal", "nominal"] = "nominal"
    n_classes: int = 2
    healthy_id: int = 0
    n_nuisance: int = 0

    # ---- Model ----
    d_motion: int = 64
    d_pathology: int = 64
    cond_dim: int = 8
    hidden_channels: int = 512
    downsample: int = 4
    n_rvq_layers: int = 6
    codebook_motion: int = 512
    codebook_pathology: int = 128
    ema_decay: float = 0.99
    quant_dropout: float = 0.2
    codebook_reset_threshold: float = 1.0
    alpha: float = 1.0

    # ---- Losses ----
    lambda_rec: float = 1.0
    lambda_cls: float = 0.01
    lambda_adv: float = 0.01
    lambda_emb: float = 0.02
    bone_weight: float = 0.0
    bone_pairs: list[tuple[int, int]] = field(default_factory=list)

    # ---- Training ----
    stage1_epochs: int = 200
    stage2_epochs: int = 300
    lr_stage1: float = 2e-6
    lr_motion_stage2: float = 2e-7
    lr_pathology: float = 2e-6
    lr_classifier: float = 2e-7
    grl_lambda_max: float = 1.0
    grl_warmup_epochs: int = 30
    healthy_zeroout: bool = True
    batch_size: int = 512
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cuda"
    log_every: int = 50
    out_dir: str = "runs/gaitgen_neonate"

    @property
    def input_dim(self) -> int:
        """Flattened per-frame input width, ``J * in_channels``."""
        return self.n_joints * self.in_channels

    @property
    def t_latent(self) -> int:
        """Downsampled temporal length T'."""
        return self.clip_length // self.downsample

    def validate(self) -> None:
        if self.clip_length % self.downsample != 0:
            raise ValueError(
                f"clip_length ({self.clip_length}) must divide by downsample "
                f"({self.downsample})."
            )
        if self.d_motion != self.d_pathology:
            raise ValueError(
                "d_motion must equal d_pathology so q_m + alpha*q_p is "
                f"well-defined (got {self.d_motion}, {self.d_pathology})."
            )
        if self.n_classes < 2:
            raise ValueError(f"n_classes ({self.n_classes}) must be >= 2.")
        if not 0 <= self.healthy_id < self.n_classes:
            raise ValueError(
                f"healthy_id ({self.healthy_id}) must be in [0, n_classes)."
            )
        if self.bone_weight > 0 and not self.bone_pairs:
            raise ValueError(
                "bone_weight > 0 needs bone_pairs (connected keypoint index "
                "pairs). Set them for the dataset's skeleton."
            )
