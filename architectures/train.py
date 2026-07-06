"""Training loop for the masked neonate-motion VAE.

`train(config, videos, limbs=None)` runs a full training run: it slices
videos into clips, splits by time within each video, builds the model
and optimiser, and runs the recipe-appropriate loop for `n_epochs`
epochs.

The three recipes ([MVAE §3-5]) share the encoder and decoder and differ
in how many forward passes the batch does and how the loss is composed:

    Recipe 1  one pass with the masked clip; MSE on the full clip,
              plus KL. [MVAE §3]
    Recipe 2  two passes: primary with the clean clip contributes MSE
              + KL, auxiliary with the masked clip contributes MSE
              only. [MVAE §4]
    Recipe 3  one pass with the masked clip; dual decoder heads —
              full-clip MSE from the full head, hidden-only MSE from
              the mask-conditioned inpainting head, plus KL. [MVAE §5]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import TrainingConfig
from .data import build_clips, make_loader, train_val_split
from .losses import (kl_gaussian, reconstruction_mse,
                     reconstruction_mse_hidden, beta_schedule)
from .mask_policies import build_policy
from .models import build_model


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("Training needs PyTorch.") from e


def train(config: TrainingConfig,
          videos: list[np.ndarray],
          limbs: dict[str, list[int]] | None = None,
          stride: int | None = None) -> dict:
    """Run a full training loop.

    Args:
        config: a validated TrainingConfig.
        videos: list of videos, each shape (F_v, J, 3).
        limbs: joint-index lists per limb name, for the limb policy.
        stride: hop between clip starts. Defaults to `clip_length // 2`.
    Returns:
        Dict with the trained model, the loss history, and the path of
        the last checkpoint written.
    """
    torch = _torch()

    config.validate()
    if stride is None:
        stride = config.clip_length // 2

    # ---- Reproducibility ----------------------------------------------
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ---- Data ---------------------------------------------------------
    clips, video_id, time_index = build_clips(videos, config.clip_length, stride)
    train_mask, val_mask = train_val_split(clips, video_id)
    print(f"[data] {len(clips)} clips, {train_mask.sum()} train, {val_mask.sum()} val")

    policy = build_policy(config, limbs=limbs)
    train_loader = make_loader(clips[train_mask], policy,
                               config.batch_size, shuffle=True,
                               seed=config.seed)
    val_loader = make_loader(clips[val_mask], policy,
                             config.batch_size, shuffle=False,
                             seed=config.seed + 1)

    # ---- Model and optimiser ------------------------------------------
    device = torch.device(config.device if torch.cuda.is_available()
                          or config.device == "cpu" else "cpu")
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {config.architecture} VAE, {n_params:,} parameters")

    opt = torch.optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    # ---- Output directory ---------------------------------------------
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {"train": [], "val": []}
    best_val = float("inf")
    last_ckpt = None

    # ---- Loop ---------------------------------------------------------
    for epoch in range(config.n_epochs):
        beta = beta_schedule(epoch, config.warmup_epochs, config.beta_max)
        t0 = time.time()

        model.train()
        train_stats = _run_epoch(model, train_loader, config, beta, opt,
                                 device, train=True)
        model.eval()
        with torch.no_grad():
            val_stats = _run_epoch(model, val_loader, config, beta, None,
                                   device, train=False)

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        dt = time.time() - t0
        print(f"[epoch {epoch:3d}] beta={beta:.3f} "
              f"train_loss={train_stats['loss']:.4f} "
              f"val_loss={val_stats['loss']:.4f}  ({dt:.1f} s)")

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            last_ckpt = out / "best.pt"
            torch.save({"model": model.state_dict(),
                        "config": config.__dict__,
                        "epoch": epoch}, last_ckpt)

        if config.save_every and epoch and epoch % config.save_every == 0:
            ck = out / f"epoch_{epoch:04d}.pt"
            torch.save({"model": model.state_dict(),
                        "config": config.__dict__,
                        "epoch": epoch}, ck)
            last_ckpt = ck

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return {"model": model, "history": history, "checkpoint": last_ckpt}


def _run_epoch(model, loader, config, beta, opt, device, train: bool) -> dict:
    """Run one training or validation epoch and return the mean losses.

    `rec_full` is the full-clip reconstruction (primary MSE for Recipes
    1, 2, 3). `rec_aux` is the auxiliary MSE — Recipe 2's masked-pass
    reconstruction, or Recipe 3's hidden-only inpainting MSE. Recipe 1
    leaves `rec_aux` at zero.
    """
    torch = _torch()
    totals = {"loss": 0.0, "rec_full": 0.0, "rec_aux": 0.0, "kl": 0.0}
    n_batches = 0

    for step, (X, M) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        M = M.to(device, non_blocking=True)

        loss, parts = _step_loss(model, X, M, config, beta)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        totals["loss"] += float(loss)
        totals["rec_full"] += float(parts["rec_full"])
        totals["rec_aux"] += float(parts["rec_aux"])
        totals["kl"] += float(parts["kl"])
        n_batches += 1

        if train and config.log_every and step % config.log_every == 0:
            print(f"    step {step:4d}  loss={float(loss):.4f}  "
                  f"rec_full={float(parts['rec_full']):.4f}  "
                  f"rec_aux={float(parts['rec_aux']):.4f}  "
                  f"kl={float(parts['kl']):.3f}")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def _step_loss(model, X, M, config, beta):
    """Compute the loss for one batch under the configured recipe.

    Recipe 1 ([MVAE §3.6]):
        L = MSE(X, X_hat_masked_in) + beta * KL.

    Recipe 2 ([MVAE §4.2]):
        primary pass with an all-ones mask contributes MSE + KL.
        auxiliary pass with the drawn mask contributes MSE only.
        L = MSE_primary + lambda * MSE_aux + beta * KL_primary.

    Recipe 3 ([MVAE §5.2]):
        one masked-input pass with two decoder heads.
        L = MSE(X, X_hat_full) + lambda * MSE_hidden(X, X_hat_inp, M)
            + beta * KL.
    """
    torch = _torch()
    zero = torch.zeros((), device=X.device)

    if config.recipe == 1:
        X_hat, mu, logvar = model(X, M)
        rec = reconstruction_mse(X_hat, X)
        kl = kl_gaussian(mu, logvar).mean()
        loss = rec + beta * kl
        return loss, {"rec_full": rec, "rec_aux": zero, "kl": kl}

    if config.recipe == 2:
        # Primary pass: clean clip in, full-clip MSE + KL.
        M_ones = torch.ones_like(M)
        X_hat_primary, mu, logvar = model(X, M_ones)
        rec_primary = reconstruction_mse(X_hat_primary, X)
        kl = kl_gaussian(mu, logvar).mean()

        # Auxiliary pass: masked clip in, full-clip MSE, no KL.
        X_hat_aux, _, _ = model(X, M)
        rec_aux = reconstruction_mse(X_hat_aux, X)

        loss = rec_primary + config.lambda_aux * rec_aux + beta * kl
        return loss, {"rec_full": rec_primary, "rec_aux": rec_aux, "kl": kl}

    if config.recipe == 3:
        # Single masked pass; two decoder heads.
        X_hat_full, X_hat_inp, mu, logvar = model(X, M)
        rec_full = reconstruction_mse(X_hat_full, X)
        rec_inp = reconstruction_mse_hidden(X_hat_inp, X, M)
        kl = kl_gaussian(mu, logvar).mean()
        loss = rec_full + config.lambda_aux * rec_inp + beta * kl
        return loss, {"rec_full": rec_full, "rec_aux": rec_inp, "kl": kl}

    raise ValueError(f"unknown recipe: {config.recipe!r}")
