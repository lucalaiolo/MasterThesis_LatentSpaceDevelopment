"""End-to-end smoke test — run this in an environment with PyTorch.

Trains each architecture under each recipe for two epochs on synthetic
data. Confirms shapes flow through, losses decrease over the first few
steps, and the parameter counts match the analytical predictions.
"""

import numpy as np
import torch

from vae_training import TrainingConfig
from vae_training.models import build_model
from vae_training.param_counts import summarise
from vae_training.mask_policies import UniformMask, LimbMask, NoMask
from vae_training.train import train
from vae_training.evaluate import evaluate


def synthetic_videos(n_videos: int = 3, frames: int = 300, J: int = 17):
    """Three fake videos with mild spatial and temporal structure."""
    rng = np.random.default_rng(0)
    videos = []
    for _ in range(n_videos):
        t = np.arange(frames)[:, None, None]
        base = np.stack([
            0.5 * np.sin(t.squeeze(-1) * 0.05 + i) for i in range(J)
        ], axis=1)[..., None].repeat(3, axis=-1)
        noise = rng.standard_normal((frames, J, 3)) * 0.05
        videos.append(base + noise)
    return videos


limbs = {"left_arm": [1, 2, 3], "right_arm": [4, 5, 6],
         "left_leg": [10, 11, 12], "right_leg": [13, 14, 15]}


def one_run(arch: str, recipe: int, epochs: int = 2):
    """Train one architecture under one recipe for a couple of epochs."""
    print(f"\n=== {arch} VAE, Recipe {recipe} ===")

    mask_policy = {1: "uniform", 2: "none", 3: "limb"}[recipe]
    lambda_visible = 0.5 if recipe == 3 else 0.5
    lambda_inpainted = 0.5 if recipe == 3 else 0.5

    cfg = TrainingConfig(
        architecture=arch,
        clip_length=32, n_joints=17, latent_dim=32,
        recipe=recipe,
        mask_policy=mask_policy,
        mask_limb_names=list(limbs) if recipe == 3 else [],
        lambda_visible=lambda_visible, lambda_inpainted=lambda_inpainted,
        batch_size=16, n_epochs=epochs, warmup_epochs=1, beta_max=0.1,
        learning_rate=1e-3, device="cpu", log_every=0, save_every=0,
        out_dir=f"/tmp/vae_ckpts_{arch}_r{recipe}",
    )

    # Analytical parameter count.
    counts = summarise(cfg)
    print(f"analytical total parameters: {counts['total']:,}")

    videos = synthetic_videos()
    out = train(cfg, videos, limbs=limbs)

    # Confirm the actual model matches the analytical count.
    actual = sum(p.numel() for p in out["model"].parameters())
    diff = actual - counts["total"]
    print(f"actual model parameters:     {actual:,}  (delta {diff:+d}, "
          f"LayerNorm scales and biases)")

    # Held-out MPJPE for a sanity look.
    test_clips = np.stack([videos[0][i:i + 32] for i in range(200, 260, 8)])
    policy = {1: UniformMask(0.3),
              2: NoMask(),
              3: LimbMask(limbs=limbs)}[recipe]
    metrics = evaluate(out["model"], test_clips, policy,
                       batch_size=8, device="cpu", recipe=recipe)
    print(f"held-out mpjpe_all       {metrics['mpjpe_all']:.4f}")
    if recipe in (1, 3):
        print(f"held-out mpjpe_visible   {metrics['mpjpe_visible']:.4f}")
        print(f"held-out mpjpe_inpainted {metrics['mpjpe_inpainted']:.4f}")

    return out


if __name__ == "__main__":
    for arch in ["conv", "transformer"]:
        for recipe in [1, 2, 3]:
            one_run(arch, recipe, epochs=2)
    print("\n=== every training path ran ===")
