# vae_training

Training code for the masked neonate-motion VAE. Two architectures from
the design note ([ARCH §3, §4]) and the three recipes from the masked
VAE note ([MVAE §3-5]). Everything is generic in the joint count J.

## Install

```
pip install numpy torch
```

## Wire in your data

Your dataset comes in as a list of NumPy arrays, one per video, each of
shape `(F_v, J, 3)`: F_v frames of J joints in 3D. The training loop
slices them into overlapping clips of length T at a stride you choose.

```python
from vae_training import TrainingConfig
from vae_training.train import train

config = TrainingConfig(
    architecture="conv",       # or "transformer"
    clip_length=32,
    n_joints=J,                # your J
    latent_dim=32,
    recipe=1,                  # 1, 2, or 3
    mask_policy="uniform",     # "none", "uniform", or "limb"
    mask_uniform_rho=0.3,
    batch_size=64,
    n_epochs=100,
    device="cuda",
)

# For the limb policy, describe the joint groups:
limbs = {"left_arm": [1, 2, 3], "right_arm": [4, 5, 6],
         "left_leg": [10, 11, 12], "right_leg": [13, 14, 15]}

out = train(config, videos, limbs=limbs)
model = out["model"]
```

`smoke_test.py` runs every combination of architecture and recipe on
synthetic data for two epochs; read it as a full worked example.

## The three recipes ([MVAE §3-5])

| Recipe | Encoder input | Decoder input | Loss |
|:---:|:---|:---|:---|
| 1 | masked   | z          | MSE on all joints |
| 2 | unmasked | z          | MSE on all joints |
| 3 | masked   | z and mask | weighted split MSE on visible and hidden |

The configuration enforces the pairing between recipe and mask policy;
Recipe 2 refuses a non-empty mask, Recipes 1 and 3 refuse `mask_policy="none"`.

## Files

| File | What it does |
|:---|:---|
| `config.py`         | TrainingConfig dataclass |
| `mask_policies.py`  | NoMask, UniformMask, LimbMask ([MVAE §2]) |
| `data.py`           | video slicing, DataLoader, time-based train/val split |
| `losses.py`         | KL, MSE, split MSE, beta schedule |
| `models/common.py`  | LayerNorm across channels, sinusoidal PE, reparameterisation |
| `models/conv_vae.py`         | 1D temporal convolutional VAE ([ARCH §3]) |
| `models/transformer_vae.py`  | frame-token transformer VAE ([ARCH §4.1, §4.2]) |
| `train.py`          | end-to-end loop, per-epoch validation, checkpoints |
| `evaluate.py`       | MPJPE reconstruction, MPJPE inpainting ([MVAE §7]) |
| `param_counts.py`   | analytical parameter counts, no torch needed |

## Parameter budgets

Both models are small by design. At `clip_length=32`, `n_joints=22`,
`latent_dim=32`:

- Convolutional model: 296,706 parameters, matching the "≈ 297k"
  in [ARCH §6.1] to the last hundred.
- Transformer model: 693,154 parameters, matching the "≈ 689k" in
  [ARCH §6.1] up to LayerNorm scales and biases the note rounds away.

`test_no_torch.py` verifies each per-component figure against the design
note. Change `n_joints` in the config and the counts scale cleanly.

## Choosing a recipe

The design note argues Recipe 1 first, with the convolutional model,
because it has the fewest failure modes and the fastest iteration
([ARCH §5]). Move to Recipe 3 once you have Recipe 1 trained end to end
and its MPJPE numbers on record.

## Two small warnings

`ClipDataset` redraws masks per access, so a training epoch sees fresh
masks even on the same clip. That is the point ([MVAE §6.4]), but it
means the validation loss varies from run to run unless you fix the mask
seed. `make_loader` takes a `seed` argument for that.

The transformer's parameter count reported by the analytical helper
excludes LayerNorm scales and biases. The actual model has about 4,000
more parameters than the helper reports. That is not a bug; it is what
the design note rounds away.
