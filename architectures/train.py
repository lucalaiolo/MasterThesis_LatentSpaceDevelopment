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
                     reconstruction_velocity_mse,
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
          stride: int | None = None,
          init_state: dict | None = None,
          val_fraction: float | None = None) -> dict:
    """Run a full training loop.

    Args:
        config: a validated TrainingConfig.
        videos: list of videos, each shape (F_v, J, 3).
        limbs: joint-index lists per limb name, for the limb policy.
        stride: hop between clip starts. Defaults to `clip_length // 2`.
        val_fraction: fraction of *videos* held out for the monitoring /
            checkpoint-selection split. ``None`` (default) uses the standard
            0.15 video-wise hold-out. ``0.0`` trains on **all** clips — no
            validation curve and no val-picked ``best.pt`` (the returned model
            is the final epoch and a ``final.pt`` is written). Use ``0.0`` for a
            committed full-data pretrain after the config is already selected.
        init_state: optional model ``state_dict`` to warm-start from before the
            loop (fine-tuning). Must match the config's architecture / J / D.
            Loaded with ``strict=True`` so a shape mismatch fails loudly rather
            than silently training from scratch. Pair with a lowered
            ``learning_rate`` to fine-tune a pretrained checkpoint on new data.
    Returns:
        Dict with the trained model, the loss history, and the path of the
        last checkpoint written.
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
    if val_fraction is not None and val_fraction == 0.0:
        # Committed full-data run: every clip trains, nothing held out.
        train_mask = np.ones(len(clips), dtype=bool)
        val_mask = np.zeros(len(clips), dtype=bool)
    elif val_fraction is None:
        train_mask, val_mask = train_val_split(clips, video_id)
    else:
        train_mask, val_mask = train_val_split(clips, video_id,
                                               val_fraction=val_fraction)
    has_val = bool(val_mask.any())
    print(f"[data] {len(clips)} clips, {train_mask.sum()} train, {val_mask.sum()} val"
          + ("  (training on ALL data; no val split)" if not has_val else ""))

    policy = build_policy(config, limbs=limbs)
    train_loader = make_loader(clips[train_mask], policy, config.batch_size,
                               shuffle=True, seed=config.seed)
    val_loader = (make_loader(clips[val_mask], policy, config.batch_size,
                              shuffle=False, seed=config.seed + 1)
                  if has_val else None)

    # ---- Model and optimiser ------------------------------------------
    device = torch.device(config.device if torch.cuda.is_available()
                          or config.device == "cpu" else "cpu")
    model = build_model(config).to(device)
    if init_state is not None:
        # Warm-start (fine-tuning): load the pretrained weights before the
        # optimiser is built, so its state matches the loaded parameters.
        model.load_state_dict(init_state)
        print("[model] warm-started from a pretrained state_dict (fine-tuning)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] VAE ({config.architecture}), {n_params:,} parameters")

    opt = torch.optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    # ---- Output directory ---------------------------------------------
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {"train": [], "val": []}
    best_val = float("inf")
    best_ckpt = None
    best_epoch = -1
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

        if has_val:
            model.eval()
            with torch.no_grad():
                val_stats = _run_epoch(model, val_loader, config, epoch,
                                       kl_state, None, device, train=False)
        else:
            val_stats = {}                       # no held-out data this run

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        # Pick best.pt on a KL-schedule-independent score so annealing does
        # not lock the checkpoint onto the untrained epoch-0 model. With no val
        # split there is nothing to select on — the final epoch is the model.
        saved_best = False
        if has_val:
            val_score = _val_selection_score(config, val_stats)
            saved_best = val_score < best_val
            if saved_best:
                best_val = val_score
                best_epoch = epoch
                best_ckpt = out / "best.pt"
                _save_ckpt(best_ckpt, model, config, epoch)
                last_ckpt = best_ckpt

        dt = time.time() - t0
        extra = ""
        # Train/val pairs in the log, or train-only when there is no val split.
        def _tv(key):
            t = train_stats[key]
            return f"{t:.4f}/{val_stats[key]:.4f}" if has_val else f"{t:.4f}"
        if getattr(config, "lambda_velocity", 0.0) > 0:
            extra += f" vel={_tv('rec_vel')}"
        print(f"[epoch {epoch:3d}] "
              f"beta={train_stats['beta']:.2e} "
              f"loss={_tv('loss')} "
              f"rec={_tv('rec_full')}"
              f"{extra}  ({dt:.1f} s)"
              f"{'  *best' if saved_best else ''}")

        if config.save_every and epoch and epoch % config.save_every == 0:
            ck = out / f"epoch_{epoch:04d}.pt"
            _save_ckpt(ck, model, config, epoch)
            last_ckpt = ck

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if best_ckpt is not None:
        metric = getattr(config, "checkpoint_metric", "rec_full")
        print(f"[ckpt] best.pt = epoch {best_epoch} "
              f"(val {metric}={best_val:.4f}) -> {best_ckpt}")
    elif not has_val:
        # No val split: there is no best.pt, so persist the final-epoch model.
        best_ckpt = out / "final.pt"
        _save_ckpt(best_ckpt, model, config, config.n_epochs - 1)
        last_ckpt = best_ckpt
        print(f"[ckpt] final.pt = last epoch -> {best_ckpt}")

    # Best-effort summary plots. Never let a plotting problem sink a run that
    # already trained and checkpointed: skip on a missing matplotlib, and
    # skip (with a traceback) on any other plotting error rather than raising.
    try:
        from .visualize import plot_training_summary
        written = plot_training_summary(
            history, out_dir=out / "plots", config=config,
            model=model, loader=(val_loader if has_val else train_loader),
            device=str(device),
        )
        print(f"[plots] wrote {len(written)} figure(s) to {out / 'plots'}")
    except ImportError as e:
        print(f"[plots] skipped: {e}")
    except Exception as e:  # noqa: BLE001 - plotting is non-essential
        import traceback
        print(f"[plots] skipped (plotting error): {e}")
        traceback.print_exc()

    return {"model": model, "history": history,
            "checkpoint": best_ckpt or last_ckpt, "best_epoch": best_epoch}


def _val_selection_score(config, val_stats) -> float:
    """Validation score for picking ``best.pt`` ([config.checkpoint_metric]).

    Selecting on the *scheduled* total loss is biased by KL annealing: at
    epoch 0 beta ~ 0, so the total loss is smallest there and ``best.pt``
    locks onto the untrained, latent-collapsed model. The default
    ``"rec_full"`` selects on reconstruction, which is beta-independent and
    monotone enough to never pick the untrained epoch; ``"elbo"`` uses the
    objective at the ceiling beta so KL is weighted identically at every
    epoch; ``"loss"`` is the legacy (biased) behaviour.
    """
    metric = getattr(config, "checkpoint_metric", "rec_full")
    if metric == "loss":
        return float(val_stats["loss"])
    if metric == "elbo":
        beta_ceiling = 0.0 if config.beta_mode == "computed" else config.beta_max
        score = float(val_stats["rec_full"]) + beta_ceiling * float(val_stats["kl"])
        if config.recipe in (2, 3):
            score += config.lambda_aux * float(val_stats["rec_aux"])
        if getattr(config, "lambda_velocity", 0.0) > 0:
            score += config.lambda_velocity * float(val_stats.get("rec_vel", 0.0))
        return score
    return float(val_stats["rec_full"])          # "rec_full" (default)


def _save_ckpt(path, model, config, epoch):
    """Write the model state, the config that built it, and the epoch."""
    torch = _torch()
    torch.save({"model": model.state_dict(),
                "config": config.__dict__,
                "epoch": epoch}, path)


def _run_epoch(model, loader, config, epoch, kl_state, opt, device,
               train: bool):
    """Run one epoch and return the mean of each loss component.

    `rec_full` is the full-clip reconstruction (primary MSE for Recipes
    1, 2, 3). `rec_aux` is the auxiliary MSE — Recipe 2's masked-pass
    reconstruction, or Recipe 3's hidden-only inpainting MSE. Recipe 1
    leaves `rec_aux` at zero. `kl` is the KL against the N(0, I) prior and
    `beta` its effective weight, averaged over the epoch.
    """
    totals = {"loss": 0.0, "rec_full": 0.0, "rec_aux": 0.0, "rec_vel": 0.0,
              "kl": 0.0, "beta": 0.0}
    n_batches = 0

    for step, batch in enumerate(loader):
        X, M = (t.to(device, non_blocking=True) for t in batch)

        loss, parts = _step_loss(model, X, M, config, epoch, kl_state,
                                 update=train)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        totals["loss"] += float(loss.detach())
        for key in ("rec_full", "rec_aux", "rec_vel", "kl", "beta"):
            totals[key] += float(parts[key])
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

    When ``config.lambda_velocity > 0`` an extra term
    ``lambda_velocity * MSE(velocity(X_hat_full), velocity(X))`` is added
    to every recipe, scoring the reconstruction's frame-to-frame motion
    (temporal-smoothness regulariser); it is 0 otherwise.

    All routes go through `_kl_term` (vanilla or free-bits KL) and
    `_resolve_beta` (linear warmup or Asperti-Trentin computed).
    """
    torch = _torch()
    zero = torch.zeros((), device=X.device)

    if config.recipe == 1:
        X_hat, mu, logvar = model(X, M)
        x_hat_full = X_hat
        rec_full = reconstruction_mse(X_hat, X)
        rec_aux = zero
        kl = _kl_term(mu, logvar, config)

    elif config.recipe == 2:
        # Primary pass: clean clip in, full-clip MSE + KL.
        M_ones = torch.ones_like(M)
        X_hat_primary, mu, logvar = model(X, M_ones)
        x_hat_full = X_hat_primary
        rec_full = reconstruction_mse(X_hat_primary, X)
        kl = _kl_term(mu, logvar, config)

        # Auxiliary pass: masked clip in, full-clip MSE, no KL.
        X_hat_aux, _, _ = model(X, M)
        rec_aux = reconstruction_mse(X_hat_aux, X)

    elif config.recipe == 3:
        # Single masked pass; two decoder heads.
        X_hat_full, X_hat_inp, mu, logvar = model(X, M)
        x_hat_full = X_hat_full
        rec_full = reconstruction_mse(X_hat_full, X)
        rec_aux = reconstruction_mse_hidden(X_hat_inp, X, M)
        kl = _kl_term(mu, logvar, config)

    else:
        raise ValueError(f"unknown recipe: {config.recipe!r}")

    # Optional velocity term: match the reconstruction's frame-to-frame
    # motion to the target's, scored on the full-clip head for every recipe.
    lambda_vel = getattr(config, "lambda_velocity", 0.0)
    rec_vel = (reconstruction_velocity_mse(x_hat_full, X)
               if lambda_vel > 0 else zero)

    loss = rec_full
    if config.recipe in (2, 3):
        loss = loss + config.lambda_aux * rec_aux
    if lambda_vel > 0:
        loss = loss + lambda_vel * rec_vel

    beta = _resolve_beta(config, epoch, kl_state, rec_full, update=update)
    loss = loss + beta * kl

    beta_tensor = torch.as_tensor(beta, device=X.device, dtype=rec_full.dtype)
    return loss, {"rec_full": rec_full, "rec_aux": rec_aux,
                  "rec_vel": rec_vel, "kl": kl, "beta": beta_tensor}


def train_sweep(base_config: TrainingConfig,
                videos: list[np.ndarray],
                limbs: dict[str, list[int]] | None = None,
                stride: int | None = None,
                recipes: tuple[int, ...] = ALL_RECIPES,
                mask_policies: tuple[str, ...] = ALL_MASK_POLICIES,
                n_layers_grid: tuple[int, ...] | None = None,
                ) -> dict[tuple, dict]:
    """Run `train` across every valid (recipe, mask_policy) combination.

    Shared knobs — architecture, latent width, batch size, epochs, beta
    schedule, seed — come from `base_config`. For each combo we clone
    the config with `recipe` and `mask_policy` overridden, and route the
    run's outputs to a subdirectory `<base out_dir>/recipe{N}_{policy}`
    so nothing collides.

    Transformer **depth** can be swept as an extra axis with
    ``n_layers_grid``: pass e.g. ``(3, 6, 12)`` to train every
    (depth, recipe, policy) combination. Each depth overrides
    ``base_config.n_layers`` (both encoder and decoder), its runs land in
    ``<base out_dir>/L{depth}/recipe{N}_{policy}``, and the result keys
    grow to ``(n_layers, recipe, mask_policy)``. Left as ``None`` (the
    default) the depth stays whatever ``base_config`` carries and the keys
    remain the original ``(recipe, mask_policy)`` — a no-op for callers
    that only want to sweep recipe and policy. (For a conv backbone,
    depth is fixed by the three-block design, so a grid is ignored beyond
    its effect on the config.)

    Skipped combos: Recipes 2 and 3 with `mask_policy="none"` (rejected
    by `TrainingConfig.validate`), and `"limb"` when no `limbs` map was
    passed.

    Args:
        base_config: template config; `recipe`, `mask_policy`, `n_layers`
            (when a grid is given), and `out_dir` are overridden per run.
        videos: forwarded to `train`.
        limbs: joint-index lists per limb name; required for the "limb"
            policy, ignored otherwise.
        stride: forwarded to `train`.
        recipes: which recipes to sweep. Defaults to (1, 2, 3).
        mask_policies: which policies to sweep. Defaults to all six.
        n_layers_grid: optional transformer depths to sweep. ``None``
            keeps ``base_config``'s depth and the two-field result keys.
    Returns:
        Dict keyed by (recipe, mask_policy), or by (n_layers, recipe,
        mask_policy) when ``n_layers_grid`` is given, with each run's
        `train` return value.
    """
    base_out = Path(base_config.out_dir)
    results: dict[tuple, dict] = {}
    sweep_depth = n_layers_grid is not None
    depths = list(n_layers_grid) if sweep_depth else [None]

    for depth in depths:
        for recipe in recipes:
            for policy in mask_policies:
                tag = f"depth={depth} " if sweep_depth else ""
                if recipe in (2, 3) and policy == "none":
                    print(f"[sweep] skip {tag}recipe={recipe} policy={policy!r}: "
                          "recipes 2 and 3 need a mask policy.")
                    continue
                if policy == "limb" and not limbs:
                    print(f"[sweep] skip {tag}recipe={recipe} policy={policy!r}: "
                          "no `limbs` map provided.")
                    continue

                overrides = dict(recipe=recipe, mask_policy=policy)
                if sweep_depth:
                    overrides["n_layers"] = depth
                    sub = base_out / f"L{depth}" / f"recipe{recipe}_{policy}"
                    key: tuple = (depth, recipe, policy)
                else:
                    sub = base_out / f"recipe{recipe}_{policy}"
                    key = (recipe, policy)
                cfg = dataclasses.replace(base_config, out_dir=str(sub),
                                          **overrides)
                print(f"\n[sweep] === {tag}recipe={recipe} policy={policy!r} "
                      f"-> {sub} ===")
                results[key] = train(cfg, videos, limbs=limbs,
                                     stride=stride)

    return results


def model_selection(base_config: TrainingConfig,
                    videos: list[np.ndarray],
                    recipes: tuple[int, ...] = ALL_RECIPES,
                    mask_policies: tuple[str, ...] = ALL_MASK_POLICIES,
                    limbs: dict[str, list[int]] | None = None,
                    stride: int | None = None,
                    metric: str = "mpjpe_all",
                    eval_seed: int = 0,
                    n_layers_grid: tuple[int, ...] | None = None) -> dict:
    """Sweep (recipe × mask policy) and pick the best by held-out reconstruction.

    The masking recipe and policy are means, not ends — one default pair
    carries the model progression, and it is chosen by reconstruction on the
    held-out split rather than guessed. Trains one run per valid
    (recipe, mask_policy) combination via :func:`train_sweep`, scores every
    run on the *same* time-based validation split with
    :func:`architectures.evaluate.evaluate`, and returns the winner.

    ``metric="mpjpe_all"`` (default) selects on the unmasked-input
    reconstruction error, which is recipe/policy-comparable; pass
    ``"mpjpe_inpainted"`` to select on inpainting (hidden-joint) error
    instead. Lower is better either way.

    Args:
        base_config: template config; ``recipe`` / ``mask_policy`` /
            ``out_dir`` are overridden per run. ``architecture`` (the
            backbone) is held fixed — sweep it by calling this once per
            backbone if you want that axis too.
        videos: forwarded to :func:`train_sweep`.
        recipes, mask_policies: the grid. Defaults to all three recipes and
            all six policies (``limb`` is skipped unless ``limbs`` is given,
            and recipes 2/3 skip ``"none"``).
        limbs: joint-index lists per limb name, needed only for the ``limb``
            policy.
        stride: clip stride; defaults to ``clip_length // 2`` (as in ``train``).
        metric: which held-out MPJPE to rank on.
        eval_seed: seeds the evaluation mask draws.
        n_layers_grid: optional transformer depths to add as a selection
            axis (forwarded to :func:`train_sweep`). When given, the table
            and ranking range over ``(n_layers, recipe, policy)`` and
            ``best_config`` also pins the winning depth; ``None`` keeps the
            original ``(recipe, policy)`` selection.
    Returns:
        Dict with ``best`` (the winning key — ``(recipe, policy)``, or
        ``(n_layers, recipe, policy)`` when a depth grid is swept),
        ``best_config`` (a :class:`TrainingConfig` with those fields set —
        feed it to the final progression), ``best_run`` (that run's
        :func:`train` output, incl. the checkpoint path), ``table`` (every
        combination's MPJPE dict), and ``runs`` (the raw
        :func:`train_sweep` output).
    """
    torch = _torch()
    from .evaluate import evaluate

    if stride is None:
        stride = base_config.clip_length // 2

    runs = train_sweep(base_config, videos, limbs=limbs, stride=stride,
                       recipes=recipes, mask_policies=mask_policies,
                       n_layers_grid=n_layers_grid)

    # Rebuild the exact video-wise val split train() used internally, so the
    # selection metric is measured on subjects no run trained on.
    clips, video_id, _ = build_clips(videos, base_config.clip_length, stride)
    _, val_mask = train_val_split(clips, video_id)
    val_clips = clips[val_mask]

    device = torch.device(base_config.device if torch.cuda.is_available()
                          or base_config.device == "cpu" else "cpu")

    def _unpack_key(k):
        """(recipe, policy[, depth]) from either key shape."""
        if len(k) == 3:
            depth, recipe, policy = k
            return recipe, policy, depth
        recipe, policy = k
        return recipe, policy, None

    table: dict[tuple, dict] = {}
    for key, run in runs.items():
        recipe, policy, depth = _unpack_key(key)
        overrides = dict(recipe=recipe, mask_policy=policy)
        if depth is not None:
            overrides["n_layers"] = depth
        cfg = dataclasses.replace(base_config, **overrides)
        pol = build_policy(cfg, limbs=limbs)
        table[key] = evaluate(
            run["model"], val_clips, pol, batch_size=base_config.batch_size,
            device=str(device), seed=eval_seed, recipe=recipe)

    if not table:
        raise ValueError(
            "no (recipe, mask_policy) combinations were trained; check "
            "`recipes` / `mask_policies` and the `limbs` map."
        )

    ranked = sorted(table, key=lambda k: table[k][metric])
    best = ranked[0]
    best_recipe, best_policy, best_depth = _unpack_key(best)
    best_overrides = dict(recipe=best_recipe, mask_policy=best_policy)
    if best_depth is not None:
        best_overrides["n_layers"] = best_depth
    best_config = dataclasses.replace(base_config, **best_overrides)
    print(f"\n[model-selection] ranking by {metric} (lower is better):")
    for k in ranked:
        recipe, policy, depth = _unpack_key(k)
        star = "  <- best" if k == best else ""
        depth_tag = f"L={depth} " if depth is not None else ""
        print(f"    {depth_tag}recipe={recipe} policy={policy:<14} "
              f"{metric}={table[k][metric]:.4f} mm"
              f"  (inpaint={table[k]['mpjpe_inpainted']:.4f}){star}")

    return {"best": best, "best_config": best_config,
            "best_run": runs[best], "table": table, "runs": runs}
