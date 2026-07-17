"""Two-stage training for the disentangled RVQ-VAE ([paper Sec. 3.1.1, §3.5]).

Stage 1 pretrains the motion encoder + decoder on reconstruction alone
(``q_p`` zeroed); Stage 2 jointly trains the pathology encoder, both
classifiers, all losses, with a reduced motion-encoder learning rate, the
adversarial gradient-reversal ramp, and healthy latent dropout.

Per-loss, codebook-usage, and classifier-accuracy curves are logged every
epoch. The train/val split is by **subject**, never by clip ([plan §5]).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import GaitGenConfig
from .models import build_model
from .preprocess import WindowedData, subject_split, make_loader


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("neonate_gaitgen training needs PyTorch.") from e


def _grl_lambda(config, epoch_in_stage2: int) -> float:
    """Linear ramp of the adversarial reversal strength within stage 2."""
    if config.grl_lambda_max <= 0:
        return 0.0
    if config.grl_warmup_epochs <= 0:
        return config.grl_lambda_max
    return config.grl_lambda_max * min(1.0, epoch_in_stage2 / config.grl_warmup_epochs)


def _param_groups(model, config, stage: int):
    """Learning-rate groups for the given stage ([paper §3.5])."""
    if stage == 1:
        params = (list(model.motion_encoder.parameters())
                  + list(model.rvq_motion.parameters())
                  + list(model.decoder.parameters()))
        return [{"params": params, "lr": config.lr_stage1}]
    # Stage 2: reduced E_m lr; E_p + RVQ_p at the pathology lr; classifiers slow.
    groups = [
        {"params": list(model.motion_encoder.parameters())
         + list(model.rvq_motion.parameters()), "lr": config.lr_motion_stage2},
        {"params": list(model.pathology_encoder.parameters())
         + list(model.rvq_pathology.parameters())
         + list(model.decoder.parameters()), "lr": config.lr_pathology},
        {"params": list(model.pathology_clf.parameters())
         + list(model.adversary.parameters()), "lr": config.lr_classifier},
    ]
    if model.nuisance_adversary is not None:
        groups.append({"params": list(model.nuisance_adversary.parameters()),
                       "lr": config.lr_classifier})
    return groups


def _run_epoch(model, loader, config, stage, grl_lambda, opt, device, train):
    torch = _torch()
    keys = ("rec", "bone", "commit", "cls", "adv", "cls_acc", "adv_acc",
            "mpjpe", "usage_m", "usage_p")
    totals = {k: 0.0 for k in keys}
    totals["loss"] = 0.0
    n = 0
    model.train(train)
    for batch in loader:
        x, c_p, c_nuis, _subj = batch
        x = x.to(device)
        c_p = c_p.to(device)
        c_nuis = c_nuis.to(device)
        loss, parts = model.compute_loss(
            x, c_p, c_nuis=c_nuis, stage=stage, grl_lambda=grl_lambda,
            healthy_zeroout=config.healthy_zeroout)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        totals["loss"] += float(loss)
        for k in keys:
            totals[k] += float(parts[k])
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def train(config: GaitGenConfig, data: WindowedData,
          val_fraction: float = 0.2) -> dict:
    """Run the full two-stage training ([paper §3.5]).

    Args:
        config: a :class:`GaitGenConfig`.
        data: :class:`WindowedData` (windowed clips + subject/label arrays).
        val_fraction: fraction of **subjects** held out for validation.
    Returns:
        Dict with the trained ``model``, ``history``, and the ``checkpoint``
        path (``best.pt``, selected on validation reconstruction).
    """
    torch = _torch()
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available()
                          or config.device == "cpu" else "cpu")
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] disentangled RVQ-VAE, {n_params:,} params, "
          f"T'={config.t_latent}, D_m=D_p={config.d_motion}")

    train_idx, val_idx = subject_split(data, val_fraction, seed=config.seed)
    train_idx = np.where(train_idx)[0]
    val_idx = np.where(val_idx)[0]
    print(f"[data] {data.n} clips, {len(train_idx)} train / {len(val_idx)} val "
          f"(subject split), {len(set(data.subject))} subjects")

    train_loader = make_loader(data, train_idx, config.batch_size,
                               shuffle=True, seed=config.seed)
    val_loader = make_loader(data, val_idx, config.batch_size,
                             shuffle=False, seed=config.seed + 1)

    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history: dict[str, list] = {"stage1": [], "stage2": []}
    best_val, best_ckpt = float("inf"), None

    def save_ckpt(path, epoch, stage):
        torch.save({"model": model.state_dict(), "config": config.__dict__,
                    "epoch": epoch, "stage": stage}, path)

    # ---- Stage 1: reconstruction pretraining ----
    print(f"\n[stage 1] {config.stage1_epochs} epochs — E_m + decoder, recon only")
    opt = torch.optim.Adam(_param_groups(model, config, stage=1),
                           weight_decay=config.weight_decay)
    for epoch in range(config.stage1_epochs):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, config, 1, 0.0, opt, device, True)
        with torch.no_grad():
            va = _run_epoch(model, val_loader, config, 1, 0.0, None, device, False)
        history["stage1"].append({"train": tr, "val": va})
        if va["mpjpe"] < best_val:
            best_val, best_ckpt = va["mpjpe"], out / "best.pt"
            save_ckpt(best_ckpt, epoch, 1)
        if epoch % max(1, config.stage1_epochs // 20) == 0 \
                or epoch == config.stage1_epochs - 1:
            print(f"  [s1 {epoch:3d}] rec={tr['rec']:.4f} "
                  f"mpjpe={tr['mpjpe']:.4f}/{va['mpjpe']:.4f} "
                  f"cb_m={tr['usage_m']:.2f} ({time.time()-t0:.1f}s)")

    # ---- Stage 2: joint disentangled training ----
    print(f"\n[stage 2] {config.stage2_epochs} epochs — joint, classifiers, GRL")
    opt = torch.optim.Adam(_param_groups(model, config, stage=2),
                           weight_decay=config.weight_decay)
    for epoch in range(config.stage2_epochs):
        t0 = time.time()
        lam = _grl_lambda(config, epoch)
        tr = _run_epoch(model, train_loader, config, 2, lam, opt, device, True)
        with torch.no_grad():
            va = _run_epoch(model, val_loader, config, 2, lam, None, device, False)
        history["stage2"].append({"train": tr, "val": va, "grl_lambda": lam})
        if va["mpjpe"] < best_val:
            best_val, best_ckpt = va["mpjpe"], out / "best.pt"
            save_ckpt(best_ckpt, epoch, 2)
        if epoch % max(1, config.stage2_epochs // 20) == 0 \
                or epoch == config.stage2_epochs - 1:
            print(f"  [s2 {epoch:3d}] rec={tr['rec']:.4f} "
                  f"mpjpe={va['mpjpe']:.4f} cls_acc={tr['cls_acc']:.2f} "
                  f"adv_acc={tr['adv_acc']:.2f} (chance {1/config.n_classes:.2f}) "
                  f"cb_m={tr['usage_m']:.2f} cb_p={tr['usage_p']:.2f} "
                  f"lam={lam:.2f} ({time.time()-t0:.1f}s)")

    save_ckpt(out / "final.pt", config.stage2_epochs - 1, 2)
    with open(out / "history.json", "w") as f:
        json.dump(_to_python(history), f, indent=2)
    print(f"\n[done] best.pt (val mpjpe={best_val:.4f}) -> {best_ckpt}")
    return {"model": model, "history": history,
            "checkpoint": best_ckpt or (out / "final.pt")}


def _to_python(obj):
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def load_checkpoint(path, device: str = "cpu"):
    """Rebuild a trained model from a checkpoint ([train] writes these)."""
    torch = _torch()
    ckpt = torch.load(path, map_location=device)
    config = GaitGenConfig(**ckpt["config"])
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config
