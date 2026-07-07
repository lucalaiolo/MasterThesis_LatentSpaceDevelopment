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

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from .config import TrainingConfig
from .data import build_clips, make_loader, train_val_split
from .losses import (kl_gaussian, kl_gaussian_free_bits,
                     reconstruction_mse, reconstruction_mse_hidden,
                     beta_schedule, delayed_warmup_schedule)
from .mask_policies import build_policy
from .models import build_model


ALL_RECIPES: tuple[int, ...] = (1, 2, 3)
ALL_MASK_POLICIES: tuple[str, ...] = (
    "none", "uniform", "top_k_speed",
    "softmax_speed", "per_frame_speed", "limb",
)


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

    # State for the Asperti-Trentin "computed" KL-weight mode. Holds the
    # running minimum of batch MSE across training. Kept as a dict so
    # `_step_loss` can mutate it in place.
    kl_state: dict[str, float | None] = {"gamma_sq": None}

    # ---- Loop ---------------------------------------------------------
    for epoch in range(config.n_epochs):
        t0 = time.time()

        model.train()
        train_stats = _run_epoch(model, train_loader, config, epoch,
                                 kl_state, opt, device, train=True)
        model.eval()
        with torch.no_grad():
            val_stats = _run_epoch(model, val_loader, config, epoch,
                                   kl_state, None, device, train=False)

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        dt = time.time() - t0
        print(f"[epoch {epoch:3d}] "
              f"beta={train_stats['beta']:.4f} "
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

    # Best-effort summary plots. Skipped silently if matplotlib is missing.
    try:
        from .visualize import plot_training_summary
        written = plot_training_summary(
            history, out_dir=out / "plots", config=config,
            model=model, loader=val_loader, device=str(device),
        )
        print(f"[plots] wrote {len(written)} figure(s) to {out / 'plots'}")
    except ImportError as e:
        print(f"[plots] skipped: {e}")

    return {"model": model, "history": history, "checkpoint": last_ckpt}


def _run_epoch(model, loader, config, epoch, kl_state, opt, device,
               train: bool) -> dict:
    """Run one training or validation epoch and return the mean losses.

    `rec_full` is the full-clip reconstruction (primary MSE for Recipes
    1, 2, 3). `rec_aux` is the auxiliary MSE — Recipe 2's masked-pass
    reconstruction, or Recipe 3's hidden-only inpainting MSE. Recipe 1
    leaves `rec_aux` at zero. `beta` is the effective KL weight used
    on each step, averaged over the epoch — the same in warmup mode,
    but data-dependent in the computed mode.
    """
    totals = {"loss": 0.0, "rec_full": 0.0, "rec_aux": 0.0,
              "kl": 0.0, "beta": 0.0}
    n_batches = 0

    for step, (X, M) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        M = M.to(device, non_blocking=True)

        loss, parts = _step_loss(model, X, M, config, epoch,
                                 kl_state, update=train)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        totals["loss"] += float(loss)
        totals["rec_full"] += float(parts["rec_full"])
        totals["rec_aux"] += float(parts["rec_aux"])
        totals["kl"] += float(parts["kl"])
        totals["beta"] += float(parts["beta"])
        n_batches += 1

        if train and config.log_every and step % config.log_every == 0:
            print(f"    step {step:4d}  loss={float(loss):.4f}  "
                  f"rec_full={float(parts['rec_full']):.4f}  "
                  f"rec_aux={float(parts['rec_aux']):.4f}  "
                  f"kl={float(parts['kl']):.3f}  "
                  f"beta={float(parts['beta']):.4f}")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def _kl_term(mu, logvar, config):
    """KL contribution for one forward pass, respecting `config.free_bits`.

    When `config.free_bits > 0` the per-dimension free-bits variant
    ([MVAE §6.3]) is used, so dims with KL_d < gamma stop receiving
    gradient. Otherwise the vanilla KL is returned.
    """
    if config.free_bits > 0:
        return kl_gaussian_free_bits(mu, logvar, config.free_bits).mean()
    return kl_gaussian(mu, logvar).mean()


def _resolve_beta(config, epoch: int, kl_state: dict, rec_full,
                  update: bool) -> float:
    """Pick the effective KL weight for this batch.

    `warmup` mode: linear ramp from 0 to `beta_max` over `warmup_epochs`
    ([MVAE §6.2]).

    `delayed_warmup` mode: hold beta at `beta_min` for `delay_epochs`,
    then linearly ramp to `beta_max` over `warmup_epochs`. Reconstruction
    trains lightly-regularised first, then KL kicks in.

    `computed` mode: Asperti-Trentin 2020. Track gamma_sq as the
    running minimum of the training batch MSE and return `2 * gamma_sq`.
    On the very first batch (before any update) fall back to 1.0.

    `update=True` allows the running minimum to move; `update=False`
    (validation) freezes it at the current value so val loss stays
    comparable to train loss within the same epoch.
    """
    if config.beta_mode == "computed":
        if update:
            g2_new = float(rec_full.detach())
            g2_prev = kl_state.get("gamma_sq")
            if g2_prev is None or g2_new < g2_prev:
                kl_state["gamma_sq"] = g2_new
        g2 = kl_state.get("gamma_sq")
        return 2.0 * g2 if g2 is not None else 1.0
    if config.beta_mode == "delayed_warmup":
        return delayed_warmup_schedule(
            epoch, config.delay_epochs, config.warmup_epochs,
            config.beta_min, config.beta_max,
        )
    return beta_schedule(epoch, config.warmup_epochs, config.beta_max)


def _step_loss(model, X, M, config, epoch: int, kl_state: dict,
               update: bool = True):
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

    All three routes go through `_kl_term` (vanilla or free-bits KL)
    and `_resolve_beta` (linear warmup or Asperti-Trentin computed).
    """
    torch = _torch()
    zero = torch.zeros((), device=X.device)

    if config.recipe == 1:
        X_hat, mu, logvar = model(X, M)
        rec_full = reconstruction_mse(X_hat, X)
        rec_aux = zero
        kl = _kl_term(mu, logvar, config)

    elif config.recipe == 2:
        # Primary pass: clean clip in, full-clip MSE + KL.
        M_ones = torch.ones_like(M)
        X_hat_primary, mu, logvar = model(X, M_ones)
        rec_full = reconstruction_mse(X_hat_primary, X)
        kl = _kl_term(mu, logvar, config)

        # Auxiliary pass: masked clip in, full-clip MSE, no KL.
        X_hat_aux, _, _ = model(X, M)
        rec_aux = reconstruction_mse(X_hat_aux, X)

    elif config.recipe == 3:
        # Single masked pass; two decoder heads.
        X_hat_full, X_hat_inp, mu, logvar = model(X, M)
        rec_full = reconstruction_mse(X_hat_full, X)
        rec_aux = reconstruction_mse_hidden(X_hat_inp, X, M)
        kl = _kl_term(mu, logvar, config)

    else:
        raise ValueError(f"unknown recipe: {config.recipe!r}")

    beta = _resolve_beta(config, epoch, kl_state, rec_full, update=update)
    loss = rec_full + config.lambda_aux * rec_aux + beta * kl \
        if config.recipe in (2, 3) else rec_full + beta * kl

    beta_tensor = torch.as_tensor(beta, device=X.device, dtype=rec_full.dtype)
    return loss, {"rec_full": rec_full, "rec_aux": rec_aux,
                  "kl": kl, "beta": beta_tensor}


def train_sweep(base_config: TrainingConfig,
                videos: list[np.ndarray],
                limbs: dict[str, list[int]] | None = None,
                stride: int | None = None,
                recipes: tuple[int, ...] = ALL_RECIPES,
                mask_policies: tuple[str, ...] = ALL_MASK_POLICIES,
                ) -> dict[tuple[int, str], dict]:
    """Run `train` across every valid (recipe, mask_policy) combination.

    Shared knobs — architecture, latent width, batch size, epochs, beta
    schedule, seed — come from `base_config`. For each combo we clone
    the config with `recipe` and `mask_policy` overridden, and route the
    run's outputs to a subdirectory `<base out_dir>/recipe{N}_{policy}`
    so nothing collides.

    Skipped combos: Recipes 2 and 3 with `mask_policy="none"` (rejected
    by `TrainingConfig.validate`), and `"limb"` when no `limbs` map was
    passed.

    Args:
        base_config: template config; `recipe`, `mask_policy`, and
            `out_dir` are overridden per run.
        videos: forwarded to `train`.
        limbs: joint-index lists per limb name; required for the "limb"
            policy, ignored otherwise.
        stride: forwarded to `train`.
        recipes: which recipes to sweep. Defaults to (1, 2, 3).
        mask_policies: which policies to sweep. Defaults to all six.
    Returns:
        Dict keyed by (recipe, mask_policy) with each run's `train`
        return value.
    """
    base_out = Path(base_config.out_dir)
    results: dict[tuple[int, str], dict] = {}

    for recipe in recipes:
        for policy in mask_policies:
            if recipe in (2, 3) and policy == "none":
                print(f"[sweep] skip recipe={recipe} policy={policy!r}: "
                      "recipes 2 and 3 need a mask policy.")
                continue
            if policy == "limb" and not limbs:
                print(f"[sweep] skip recipe={recipe} policy={policy!r}: "
                      "no `limbs` map provided.")
                continue

            sub = base_out / f"recipe{recipe}_{policy}"
            cfg = dataclasses.replace(
                base_config,
                recipe=recipe,
                mask_policy=policy,
                out_dir=str(sub),
            )
            print(f"\n[sweep] === recipe={recipe} policy={policy!r} "
                  f"-> {sub} ===")
            results[(recipe, policy)] = train(
                cfg, videos, limbs=limbs, stride=stride,
            )

    return results
