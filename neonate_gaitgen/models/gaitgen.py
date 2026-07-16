"""Disentangled RVQ-VAE assembly ([paper Sec. 3.1.1], neonate 2D variant).

Wires the two encoders, two residual quantizers, decoder, and the pathology
/ adversarial classifiers into one module, with a two-stage-aware loss.

2D adaptation: the reconstruction is **only** the L1 position loss (paper
Eq. 3). The SO(3) geodesic rotation loss (paper Eq. 4) is intentionally
**dropped** — our representation is 2D keypoint positions, with no rotation
manifold. See ``L_pos`` below.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .rvq import ResidualVQ
from .networks import (MotionEncoder, PathologyEncoder, Decoder,
                       PathologyClassifier, AdversarialClassifier)


class DisentangledRVQVAE(nn.Module):
    """Motion/pathology disentangled RVQ-VAE for 2D neonate keypoints."""

    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        c = config
        self.motion_encoder = MotionEncoder(
            c.input_dim, c.hidden_channels, c.d_motion, c.downsample)
        self.pathology_encoder = PathologyEncoder(
            c.input_dim, c.n_classes, c.cond_dim, c.hidden_channels,
            c.d_pathology, c.downsample)
        self.rvq_motion = ResidualVQ(
            c.n_rvq_layers, c.codebook_motion, c.d_motion, decay=c.ema_decay,
            quant_dropout=c.quant_dropout, reset_threshold=c.codebook_reset_threshold)
        self.rvq_pathology = ResidualVQ(
            c.n_rvq_layers, c.codebook_pathology, c.d_pathology,
            decay=c.ema_decay, quant_dropout=c.quant_dropout,
            reset_threshold=c.codebook_reset_threshold)
        self.decoder = Decoder(
            c.input_dim, c.hidden_channels, c.d_motion, c.downsample)
        self.pathology_clf = PathologyClassifier(c.d_pathology, c.n_classes)
        self.adversary = AdversarialClassifier(c.d_motion, c.n_classes)
        self.nuisance_adversary = (
            AdversarialClassifier(c.d_motion, c.n_nuisance)
            if c.n_nuisance > 0 else None)

        # Bone pairs as a buffer for the optional anatomical term.
        if c.bone_weight > 0 and c.bone_pairs:
            self.register_buffer(
                "bone_index", torch.tensor(c.bone_pairs, dtype=torch.long))
        else:
            self.bone_index = None

    # ---- Reconstruction terms -------------------------------------------
    def _l_pos(self, x, x_hat):
        """L1 position loss (paper Eq. 3; no rotation components in 2D)."""
        return (x - x_hat).abs().mean()

    def _bone_loss(self, x, x_hat):
        """Optional bone-length consistency ([plan §3.4])."""
        if self.bone_index is None:
            return x.new_zeros(())
        B, T, _ = x.shape
        J = self.config.n_joints
        xj = x.reshape(B, T, J, 2)
        xh = x_hat.reshape(B, T, J, 2)
        a, b = self.bone_index[:, 0], self.bone_index[:, 1]
        d_true = (xj[:, :, a] - xj[:, :, b]).norm(dim=-1)
        d_pred = (xh[:, :, a] - xh[:, :, b]).norm(dim=-1)
        return (d_true - d_pred).abs().mean()

    def _mpjpe(self, x, x_hat):
        """Mean per-joint position error in **normalised units** ([plan §5])."""
        B, T, _ = x.shape
        J = self.config.n_joints
        e = (x.reshape(B, T, J, 2) - x_hat.reshape(B, T, J, 2)).norm(dim=-1)
        return e.mean()

    # ---- Encoding / decoding (analysis-facing) --------------------------
    def encode(self, x, c_p, with_pathology: bool = True):
        """Return quantized latents and codes (uses RVQ dropout if training)."""
        out = {}
        z_m = self.motion_encoder(x)
        q_m, idx_m, commit_m, usage_m = self.rvq_motion(z_m)
        out.update(q_m=q_m, idx_m=idx_m, commit_m=commit_m, usage_m=usage_m)
        if with_pathology:
            z_p = self.pathology_encoder(x, c_p)
            q_p, idx_p, commit_p, usage_p = self.rvq_pathology(z_p)
            out.update(q_p=q_p, idx_p=idx_p, commit_p=commit_p, usage_p=usage_p)
        return out

    def decode(self, q_m=None, q_p=None, alpha: float | None = None):
        """x_hat = D(q_m + alpha*q_p). Either latent may be omitted (zeroed)."""
        alpha = self.config.alpha if alpha is None else alpha
        if q_m is None:
            q_m = torch.zeros_like(q_p)
        code = q_m if q_p is None else q_m + alpha * q_p
        return self.decoder(code)

    @torch.no_grad()
    def encode_latents(self, x, c_p):
        """Deterministic full-depth latents for analysis (no RVQ dropout)."""
        was = self.training
        self.eval()
        z_m = self.motion_encoder(x)
        q_m, idx_m = self.rvq_motion.quantize_from_z(z_m)
        z_p = self.pathology_encoder(x, c_p)
        q_p, idx_p = self.rvq_pathology.quantize_from_z(z_p)
        self.train(was)
        return {"q_m": q_m, "q_p": q_p, "idx_m": idx_m, "idx_p": idx_p}

    # ---- Two-stage loss --------------------------------------------------
    def compute_loss(self, x, c_p, c_nuis=None, stage: int = 2,
                     grl_lambda: float = 0.0, alpha: float | None = None,
                     healthy_zeroout: bool = True) -> tuple:
        """Loss + scalar parts for one batch, per stage ([paper Eqs. 3-8]).

        Stage 1: E_m + decoder on reconstruction only (q_p zeroed). Stage 2:
        joint training with both classifiers, commitment on both codebooks,
        the adversarial term, and healthy latent dropout.
        """
        cfg = self.config
        alpha = cfg.alpha if alpha is None else alpha

        z_m = self.motion_encoder(x)
        q_m, _, commit_m, usage_m = self.rvq_motion(z_m)

        if stage == 1:
            x_hat = self.decoder(q_m)                       # q_p = 0
            rec = self._l_pos(x, x_hat)
            bone = self._bone_loss(x, x_hat)
            loss = cfg.lambda_rec * (rec + cfg.bone_weight * bone) \
                + cfg.lambda_emb * commit_m
            parts = {"rec": rec, "bone": bone, "commit": commit_m,
                     "cls": x.new_zeros(()), "adv": x.new_zeros(()),
                     "cls_acc": x.new_zeros(()), "adv_acc": x.new_zeros(()),
                     "mpjpe": self._mpjpe(x, x_hat),
                     "usage_m": float(sum(usage_m) / len(usage_m)),
                     "usage_p": 0.0}
            return loss, parts

        # ---- Stage 2 ----
        z_p = self.pathology_encoder(x, c_p)
        q_p, _, commit_p, usage_p = self.rvq_pathology(z_p)

        if healthy_zeroout and cfg.healthy_zeroout:
            keep = (c_p != cfg.healthy_id).to(q_p.dtype)    # (B,)
            q_p = q_p * keep.view(-1, 1, 1)                 # zero q_p for healthy

        x_hat = self.decode(q_m, q_p, alpha)
        rec = self._l_pos(x, x_hat)
        bone = self._bone_loss(x, x_hat)

        logits_p = self.pathology_clf(q_p)
        logits_adv = self.adversary(q_m, grl_lambda)
        l_cls = F.cross_entropy(logits_p, c_p)
        l_adv = F.cross_entropy(logits_adv, c_p)
        commit = commit_m + commit_p

        loss = (cfg.lambda_rec * (rec + cfg.bone_weight * bone)
                + cfg.lambda_cls * l_cls
                + cfg.lambda_adv * l_adv
                + cfg.lambda_emb * commit)

        if self.nuisance_adversary is not None and c_nuis is not None \
                and bool((c_nuis >= 0).any()):
            valid = c_nuis >= 0
            if valid.any():
                logits_n = self.nuisance_adversary(q_m[valid], grl_lambda)
                loss = loss + cfg.lambda_adv * F.cross_entropy(
                    logits_n, c_nuis[valid])

        parts = {
            "rec": rec, "bone": bone, "commit": commit,
            "cls": l_cls, "adv": l_adv,
            "cls_acc": (logits_p.argmax(1) == c_p).float().mean(),
            "adv_acc": (logits_adv.argmax(1) == c_p).float().mean(),
            "mpjpe": self._mpjpe(x, x_hat),
            "usage_m": float(sum(usage_m) / len(usage_m)),
            "usage_p": float(sum(usage_p) / len(usage_p)),
        }
        return loss, parts
