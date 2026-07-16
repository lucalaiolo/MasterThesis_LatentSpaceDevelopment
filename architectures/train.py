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
from .models import build_model, build_mixture


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
          cohort_per_video: np.ndarray | list[int] | None = None) -> dict:
    """Run a full training loop.

    Args:
        config: a validated TrainingConfig.
        videos: list of videos, each shape (F_v, J, 3).
        limbs: joint-index lists per limb name, for the limb policy.
        stride: hop between clip starts. Defaults to `clip_length // 2`.
        cohort_per_video: optional length-``len(videos)`` array of integer
            conditioning ids (e.g. cohort index, [CARE-PD §6]). Required
            when ``config.n_cond > 0``; the per-video id is broadcast to
            every clip cut from that video and fed to the model as ``c``.
    Returns:
        Dict with the trained model, the loss history, the fitted mixture
        prior (or None), and the path of the last checkpoint written.
    """
    torch = _torch()

    config.validate()
    if stride is None:
        stride = config.clip_length // 2
    if config.n_cond > 0 and cohort_per_video is None:
        raise ValueError(
            "config.n_cond > 0 but no cohort_per_video was passed; the "
            "CVAE / GM-CVAE needs a conditioning id per video."
        )

    # ---- Reproducibility ----------------------------------------------
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ---- Data ---------------------------------------------------------
    clips, video_id, time_index = build_clips(videos, config.clip_length, stride)
    train_mask, val_mask = train_val_split(clips, video_id)
    print(f"[data] {len(clips)} clips, {train_mask.sum()} train, {val_mask.sum()} val")

    # Broadcast the per-video conditioning id to a per-clip id.
    clip_cohort = None
    if cohort_per_video is not None:
        cpv = np.asarray(cohort_per_video, dtype=np.int64)
        if len(cpv) != len(videos):
            raise ValueError(
                f"cohort_per_video length ({len(cpv)}) must match the "
                f"video count ({len(videos)})."
            )
        clip_cohort = cpv[video_id]

    policy = build_policy(config, limbs=limbs)
    train_loader = make_loader(
        clips[train_mask], policy, config.batch_size, shuffle=True,
        seed=config.seed,
        cohort=None if clip_cohort is None else clip_cohort[train_mask])
    val_loader = make_loader(
        clips[val_mask], policy, config.batch_size, shuffle=False,
        seed=config.seed + 1,
        cohort=None if clip_cohort is None else clip_cohort[val_mask])

    # ---- Model, mixture, optimiser ------------------------------------
    device = torch.device(config.device if torch.cuda.is_available()
                          or config.device == "cpu" else "cpu")
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    kind = _model_kind(config)
    print(f"[model] {kind} ({config.architecture}), {n_params:,} parameters")

    mixture = build_mixture(config)
    opt_params = list(model.parameters())
    if mixture is not None:
        mixture = mixture.to(device)
        if mixture.trainable:
            # Regular / VaDE regime: the mixture parameters are optimised
            # jointly with the networks by the same optimiser.
            opt_params += list(mixture.parameters())
            print(f"[mixture] {config.n_components} components, "
                  f"gradient-trained (regular GM-VAE)")
        else:
            print(f"[mixture] {config.n_components} components, "
                  f"EM {config.gm_em_steps} step(s)/epoch")

    opt = torch.optim.AdamW(opt_params,
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    # In the gradient regime, seed the mixture on the pre-trained
    # autoencoder's latents once, right before the KL warm-up begins — a
    # VaDE run clusters far better from a data-driven GMM init than from
    # noise. Chosen as the last delay epoch (or epoch 0 without a delay).
    gm_init_epoch = -1
    if mixture is not None and mixture.trainable:
        gm_init_epoch = (max(0, config.delay_epochs - 1)
                         if config.beta_mode == "delayed_warmup" else 0)
    gm_seeded = False

    # ---- Output directory ---------------------------------------------
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {"train": [], "val": []}
    if mixture is not None:
        history["gm_occupancy"] = []
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
        train_stats, cached = _run_epoch(model, train_loader, config, epoch,
                                         kl_state, opt, device, train=True,
                                         mixture=mixture)
        # Mixture bookkeeping after the gradient epoch.
        if mixture is not None and cached is not None:
            mu_all, logvar_all = cached
            if mixture.trainable:
                # Gradient regime: parameters already moved via the
                # optimiser. One-time data-driven seeding at gm_init_epoch,
                # then just record occupancy from the current mixture.
                if not gm_seeded and epoch >= gm_init_epoch:
                    mixture.init_from_latents(mu_all)
                    gm_seeded = True
                with torch.no_grad():
                    rho = mixture.responsibilities(mu_all).mean(dim=0)
            else:
                # EM regime: closed-form M-step with the networks frozen.
                rho = mixture.em_update(mu_all, logvar_all,
                                        n_steps=config.gm_em_steps)
            history["gm_occupancy"].append([round(float(r), 5) for r in rho])

        model.eval()
        with torch.no_grad():
            val_stats, _ = _run_epoch(model, val_loader, config, epoch,
                                      kl_state, None, device, train=False,
                                      mixture=mixture)

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        # Pick best.pt on a KL-schedule-independent score so annealing does
        # not lock the checkpoint onto the untrained epoch-0 model.
        val_score = _val_selection_score(config, val_stats)
        saved_best = val_score < best_val
        if saved_best:
            best_val = val_score
            best_epoch = epoch
            best_ckpt = out / "best.pt"
            _save_ckpt(best_ckpt, model, mixture, config, epoch)
            last_ckpt = best_ckpt

        dt = time.time() - t0
        extra = ""
        if mixture is not None:
            occ = history["gm_occupancy"][-1]
            extra = (f" klz={train_stats['kl_z']:.3f} kly={train_stats['kl_y']:.3f}"
                     f" occ=[{', '.join(f'{o:.2f}' for o in occ)}]")
        print(f"[epoch {epoch:3d}] "
              f"beta={train_stats['beta']:.2e} "
              f"loss={train_stats['loss']:.4f}/{val_stats['loss']:.4f} "
              f"rec={train_stats['rec_full']:.4f}/{val_stats['rec_full']:.4f}"
              f"{extra}  ({dt:.1f} s)"
              f"{'  *best' if saved_best else ''}")

        if config.save_every and epoch and epoch % config.save_every == 0:
            ck = out / f"epoch_{epoch:04d}.pt"
            _save_ckpt(ck, model, mixture, config, epoch)
            last_ckpt = ck

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if best_ckpt is not None:
        metric = getattr(config, "checkpoint_metric", "rec_full")
        print(f"[ckpt] best.pt = epoch {best_epoch} "
              f"(val {metric}={best_val:.4f}) -> {best_ckpt}")

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

    return {"model": model, "mixture": mixture,
            "history": history, "checkpoint": best_ckpt or last_ckpt,
            "best_epoch": best_epoch}


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
        if config.n_components > 0:
            score += (config.gm_beta_z * float(val_stats["kl_z"])
                      + config.gm_beta_y * float(val_stats["kl_y"]))
        return score
    return float(val_stats["rec_full"])          # "rec_full" (default)


def _model_kind(config) -> str:
    """Human-readable name of the model class the config selects."""
    gm = config.n_components > 0
    cond = config.n_cond > 0
    if gm and cond:
        return f"GM-CVAE (K={config.n_components})"
    if gm:
        return f"GM-VAE (K={config.n_components})"
    if cond:
        return "CVAE"
    return "VAE"


def _save_ckpt(path, model, mixture, config, epoch):
    """Checkpoint the model, and the mixture parameters when present."""
    torch = _torch()
    blob = {"model": model.state_dict(),
            "config": config.__dict__,
            "epoch": epoch}
    if mixture is not None:
        blob["mixture"] = mixture.state_dict()
    torch.save(blob, path)


def _kl_warmup_factor(config, epoch: int) -> float:
    """A [0, 1] ramp for the mixture KL, matching the beta schedule shape.

    Reproduces the "learn first, apply KL later" recipe for the GM terms:
    0 during a ``delayed_warmup`` delay, then a linear ramp to 1 over
    ``warmup_epochs`` (or a plain 0->1 ramp in ``warmup`` mode). Returns 1
    when warm-up is disabled or in ``computed`` mode. Unlike ``beta`` this
    is a pure fraction, so ``gm_beta_z`` / ``gm_beta_y`` reach their full
    configured strength rather than being scaled by ``beta_max``.
    """
    if not config.gm_kl_warmup:
        return 1.0
    if config.beta_mode == "delayed_warmup":
        if epoch < config.delay_epochs:
            return 0.0
        if config.warmup_epochs <= 0:
            return 1.0
        return min(1.0, (epoch - config.delay_epochs) / config.warmup_epochs)
    if config.beta_mode == "warmup":
        if config.warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / config.warmup_epochs)
    return 1.0


def _entropy_weight(config, epoch: int) -> float:
    """Decaying weight of the assignment-entropy bonus ([CARE-PD §10]).

    Linearly anneals from ``gm_entropy_weight`` to 0 across
    ``gm_entropy_epochs`` epochs, then stays at 0. Rewards near-uniform
    q(y|x) early so components do not die before the latent organises.
    """
    if config.gm_entropy_weight <= 0 or config.gm_entropy_epochs <= 0:
        return 0.0
    frac = max(0.0, 1.0 - epoch / config.gm_entropy_epochs)
    return config.gm_entropy_weight * frac


def _run_epoch(model, loader, config, epoch, kl_state, opt, device,
               train: bool, mixture=None):
    """Run one epoch and return (mean losses, cached latents or None).

    `rec_full` is the full-clip reconstruction (primary MSE for Recipes
    1, 2, 3). `rec_aux` is the auxiliary MSE — Recipe 2's masked-pass
    reconstruction, or Recipe 3's hidden-only inpainting MSE. Recipe 1
    leaves `rec_aux` at zero. `kl` is the N(0, I) KL (the sole prior term
    for a plain VAE, and the auxiliary regulariser for a GM run). `kl_z`
    and `kl_y` are the mixture terms, zero without a mixture. `beta` is
    the effective N(0, I) KL weight, averaged over the epoch.

    For a GM training epoch the posterior means and log-variances are
    cached and returned as ``(mu_all, logvar_all)`` so the caller can run
    the EM M-step; ``None`` otherwise.
    """
    totals = {"loss": 0.0, "rec_full": 0.0, "rec_aux": 0.0,
              "kl": 0.0, "kl_z": 0.0, "kl_y": 0.0, "entropy": 0.0,
              "beta": 0.0}
    n_batches = 0
    ent_w = _entropy_weight(config, epoch)
    cache_mu, cache_logvar = [], []
    caching = train and mixture is not None

    for step, batch in enumerate(loader):
        X, M, c = _unpack_batch(batch, device)

        loss, parts = _step_loss(model, X, M, config, epoch, kl_state,
                                 update=train, mixture=mixture, c=c,
                                 entropy_weight=ent_w)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        if caching:
            cache_mu.append(parts["mu"].detach())
            cache_logvar.append(parts["logvar"].detach())

        totals["loss"] += float(loss)
        for key in ("rec_full", "rec_aux", "kl", "kl_z", "kl_y",
                    "entropy", "beta"):
            totals[key] += float(parts[key])
        n_batches += 1

        if train and config.log_every and step % config.log_every == 0:
            print(f"    step {step:4d}  loss={float(loss):.4f}  "
                  f"rec_full={float(parts['rec_full']):.4f}  "
                  f"rec_aux={float(parts['rec_aux']):.4f}  "
                  f"kl={float(parts['kl']):.3f}  "
                  f"kl_z={float(parts['kl_z']):.3f}  "
                  f"kl_y={float(parts['kl_y']):.3f}  "
                  f"beta={float(parts['beta']):.4f}")

    means = {k: v / max(n_batches, 1) for k, v in totals.items()}
    cached = None
    if caching and cache_mu:
        cached = (_torch().cat(cache_mu, dim=0),
                  _torch().cat(cache_logvar, dim=0))
    return means, cached


def _unpack_batch(batch, device):
    """Split a loader batch into (X, M, c), moving tensors to the device.

    Batches are ``(X, M)`` for a plain / GM-VAE run and ``(X, M, c)`` when
    the loader carries conditioning ids. ``c`` is ``None`` in the first
    case so the models fall back to their unconditional path.
    """
    if len(batch) == 3:
        X, M, c = batch
        c = c.to(device, non_blocking=True)
    else:
        X, M = batch
        c = None
    X = X.to(device, non_blocking=True)
    M = M.to(device, non_blocking=True)
    return X, M, c


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
               update: bool = True, mixture=None, c=None,
               entropy_weight: float = 0.0):
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

    When a ``mixture`` is supplied the prior term changes ([CARE-PD §7.3],
    [GM-VAE §3.3]). The standard-normal KL becomes an auxiliary
    regulariser (weight ``beta``), and two mixture terms are added:

        + gm_beta_z * E_q(y)[ KL(q(z|x) || p(z|y)) ]
        + gm_beta_y * KL(q(y|x) || p(y))
        - entropy_weight * H(q(y|x)).

    The responsibilities q(y|x) are the exact posterior p(c|z) under the
    current (EM-frozen) mixture, evaluated at the posterior mean ``mu``.
    Gradients flow into the encoder through ``mu``/``logvar`` and through
    the responsibilities; the mixture parameters are updated separately by
    EM in the M-step, so this step is pure network optimisation.

    ``c`` is the per-clip conditioning id for the CVAE / GM-CVAE arm, or
    None for the unconditional models. All routes go through `_kl_term`
    (vanilla or free-bits KL) and `_resolve_beta` (linear warmup or
    Asperti-Trentin computed).
    """
    torch = _torch()
    zero = torch.zeros((), device=X.device)

    if config.recipe == 1:
        X_hat, mu, logvar = model(X, M, c)
        rec_full = reconstruction_mse(X_hat, X)
        rec_aux = zero
        kl = _kl_term(mu, logvar, config)

    elif config.recipe == 2:
        # Primary pass: clean clip in, full-clip MSE + KL.
        M_ones = torch.ones_like(M)
        X_hat_primary, mu, logvar = model(X, M_ones, c)
        rec_full = reconstruction_mse(X_hat_primary, X)
        kl = _kl_term(mu, logvar, config)

        # Auxiliary pass: masked clip in, full-clip MSE, no KL.
        X_hat_aux, _, _ = model(X, M, c)
        rec_aux = reconstruction_mse(X_hat_aux, X)

    elif config.recipe == 3:
        # Single masked pass; two decoder heads.
        X_hat_full, X_hat_inp, mu, logvar = model(X, M, c)
        rec_full = reconstruction_mse(X_hat_full, X)
        rec_aux = reconstruction_mse_hidden(X_hat_inp, X, M)
        kl = _kl_term(mu, logvar, config)

    else:
        raise ValueError(f"unknown recipe: {config.recipe!r}")

    beta = _resolve_beta(config, epoch, kl_state, rec_full, update=update)
    loss = rec_full + beta * kl
    if config.recipe in (2, 3):
        loss = loss + config.lambda_aux * rec_aux

    # ---- Mixture-prior terms ([CARE-PD §7.3], [GM-VAE §3.3]) ----------
    kl_z = zero
    kl_y = zero
    entropy = zero
    if mixture is not None:
        resp = mixture.responsibilities(mu)
        kl_z = mixture.kl_z_given_y(mu, logvar, resp).mean()
        kl_y = mixture.kl_y(resp).mean()
        entropy = mixture.assignment_entropy(resp).mean()
        # Ramp the mixture KL by the same warm-up shape as beta, so the
        # "learn first" delay covers the mixture terms too ([GM-VAE §6]).
        kl_ramp = _kl_warmup_factor(config, epoch)
        loss = loss + kl_ramp * (config.gm_beta_z * kl_z
                                 + config.gm_beta_y * kl_y)
        if entropy_weight > 0:
            loss = loss - entropy_weight * entropy

    beta_tensor = torch.as_tensor(beta, device=X.device, dtype=rec_full.dtype)
    return loss, {"rec_full": rec_full, "rec_aux": rec_aux,
                  "kl": kl, "kl_z": kl_z, "kl_y": kl_y,
                  "entropy": entropy, "beta": beta_tensor,
                  "mu": mu, "logvar": logvar}


def train_sweep(base_config: TrainingConfig,
                videos: list[np.ndarray],
                limbs: dict[str, list[int]] | None = None,
                stride: int | None = None,
                recipes: tuple[int, ...] = ALL_RECIPES,
                mask_policies: tuple[str, ...] = ALL_MASK_POLICIES,
                cohort_per_video: np.ndarray | list[int] | None = None,
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
                cohort_per_video=cohort_per_video,
            )

    return results


def model_selection(base_config: TrainingConfig,
                    videos: list[np.ndarray],
                    cohort_per_video: np.ndarray | list[int] | None = None,
                    recipes: tuple[int, ...] = ALL_RECIPES,
                    mask_policies: tuple[str, ...] = ALL_MASK_POLICIES,
                    limbs: dict[str, list[int]] | None = None,
                    stride: int | None = None,
                    metric: str = "mpjpe_all",
                    eval_seed: int = 0) -> dict:
    """Sweep (recipe × mask policy) and pick the best by held-out reconstruction.

    Model selection ([CARE-PD §4.9]): the masking recipe and policy are
    means, not ends — one default pair carries the model progression, and it
    is chosen by reconstruction on the held-out split rather than guessed.
    Trains one run per valid (recipe, mask_policy) combination via
    :func:`train_sweep` — conditioned on cohort when ``cohort_per_video`` is
    given, i.e. a **CVAE** sweep — then scores every run on the *same*
    time-based validation split with the cohort-aware
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
        cohort_per_video: per-video conditioning ids; pass
            ``bundle.cohort_ids`` to select the CVAE, or None for a plain VAE.
        recipes, mask_policies: the grid. Defaults to all three recipes and
            all six policies (``limb`` is skipped unless ``limbs`` is given,
            and recipes 2/3 skip ``"none"``).
        limbs: joint-index lists per limb name, needed only for the ``limb``
            policy.
        stride: clip stride; defaults to ``clip_length // 2`` (as in ``train``).
        metric: which held-out MPJPE to rank on.
        eval_seed: seeds the evaluation mask draws.
    Returns:
        Dict with ``best`` (the winning ``(recipe, policy)``), ``best_config``
        (a :class:`TrainingConfig` with those two fields set — feed it to the
        final progression), ``best_run`` (that run's :func:`train` output,
        incl. the checkpoint path), ``table`` (every combination's MPJPE
        dict), and ``runs`` (the raw :func:`train_sweep` output).
    """
    torch = _torch()
    from .evaluate import evaluate

    if stride is None:
        stride = base_config.clip_length // 2

    runs = train_sweep(base_config, videos, limbs=limbs, stride=stride,
                       recipes=recipes, mask_policies=mask_policies,
                       cohort_per_video=cohort_per_video)

    # Rebuild the exact time-based val split train() used internally, so the
    # selection metric is measured on data no run trained on.
    clips, video_id, _ = build_clips(videos, base_config.clip_length, stride)
    _, val_mask = train_val_split(clips, video_id)
    val_clips = clips[val_mask]
    val_cohort = None
    if cohort_per_video is not None:
        cpv = np.asarray(cohort_per_video, dtype=np.int64)
        val_cohort = cpv[video_id][val_mask]

    device = torch.device(base_config.device if torch.cuda.is_available()
                          or base_config.device == "cpu" else "cpu")

    table: dict[tuple[int, str], dict] = {}
    for (recipe, policy), run in runs.items():
        cfg = dataclasses.replace(base_config, recipe=recipe, mask_policy=policy)
        pol = build_policy(cfg, limbs=limbs)
        table[(recipe, policy)] = evaluate(
            run["model"], val_clips, pol, batch_size=base_config.batch_size,
            device=str(device), seed=eval_seed, recipe=recipe,
            cohort=val_cohort)

    if not table:
        raise ValueError(
            "no (recipe, mask_policy) combinations were trained; check "
            "`recipes` / `mask_policies` and the `limbs` map."
        )

    ranked = sorted(table, key=lambda k: table[k][metric])
    best = ranked[0]
    best_config = dataclasses.replace(base_config, recipe=best[0],
                                      mask_policy=best[1])
    print(f"\n[model-selection] ranking by {metric} (lower is better):")
    for k in ranked:
        star = "  <- best" if k == best else ""
        print(f"    recipe={k[0]} policy={k[1]:<14} "
              f"{metric}={table[k][metric]:.4f} mm"
              f"  (inpaint={table[k]['mpjpe_inpainted']:.4f}){star}")

    return {"best": best, "best_config": best_config,
            "best_run": runs[best], "table": table, "runs": runs}
