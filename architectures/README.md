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
    mask_policy="uniform",     # "none", "uniform", "top_k_speed",
                               # "softmax_speed", "per_frame_speed", "limb"
    mask_rho=0.3,              # target hidden fraction (ignored by "none" / "limb")
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

| Recipe | Forward passes | Reconstruction terms | Where the KL comes from |
|:---:|:---|:---|:---|
| 1 | one masked pass                 | MSE(X, X̂) on all joints                                                        | the same masked pass |
| 2 | one unmasked + one masked pass  | MSE(X, X̂_primary) + λ·MSE(X, X̂_aux)                                          | the unmasked (primary) pass only |
| 3 | one masked pass, two heads      | MSE(X, X̂_full) from the full head + λ·MSE_hidden(X, X̂_inp, M) from the inp head | the masked pass |

Recipes 2 and 3 require a masking policy (Recipe 2's auxiliary pass
needs hidden joints, Recipe 3's inpainting head has nothing to score on
otherwise). Recipe 1 accepts `mask_policy="none"` as a plain-VAE
ablation ([MVAE §8]).

## Files

| File | What it does |
|:---|:---|
| `config.py`         | TrainingConfig dataclass |
| `mask_policies.py`  | NoMask, UniformMask, TopKSpeedMask, SoftmaxSpeedMask, PerFrameSpeedMask, LimbMask ([MVAE §2]) |
| `data.py`           | video slicing, DataLoader, time-based train/val split |
| `losses.py`         | KL (vanilla and free-bits), full-clip MSE, hidden-only MSE, beta schedule |
| `models/common.py`  | LayerNorm across channels, sinusoidal PE, reparameterisation |
| `models/conv_vae.py`         | 1D temporal convolutional VAE ([ARCH §3]) |
| `models/transformer_vae.py`  | frame-token transformer VAE ([ARCH §4.1, §4.2]) |
| `train.py`          | end-to-end loop, per-epoch validation, checkpoints |
| `evaluate.py`       | MPJPE reconstruction, MPJPE inpainting ([MVAE §7]) |
| `visualize.py`      | loss curves, latent diagnostics, pose reconstructions, mask previews |
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

## Visualising a run

`train(...)` writes `history.json` and, if matplotlib is installed, a
directory of PNGs to `<out_dir>/plots/`:

| PNG | What it shows |
|:---|:---|
| `loss_curves.png`         | train/val curves for total loss, KL, full-clip MSE, auxiliary MSE |
| `beta_schedule.png`       | the KL-weight warmup ([MVAE §6.2]) |
| `latent_kl_per_dim.png`   | per-dim KL bar chart — spots posterior collapse ([MVAE §6.5]) |
| `active_units.png`        | Var(E[z_d ∣ X]) per dim, active-unit count highlighted |
| `latent_pca.png`          | 2-D PCA of posterior means over the val set |
| `reconstruction_frames.png` | ground-truth vs predicted pose at three frames |
| `joint0_trajectory.png`   | x/y/z of joint 0 over time, true vs predicted |
| `mpjpe_per_joint.png`     | per-joint MPJPE, split into visible and hidden |
| `mpjpe_per_frame.png`     | MPJPE aggregated across joints, one point per frame |
| `mask_examples.png`       | heatmaps of the first few masks in a batch |

For ad-hoc plots the `visualize` module also exposes `plot_latent_traversal`
(sweep a single latent dimension and decode) and `collect_latent_stats`
(returns the numpy arrays behind the diagnostics).

## Fighting posterior collapse

If `active_units.png` shows only a handful of dimensions with
Var(E[z_d | X]) above the threshold, the encoder has collapsed most
of the latent to the prior. Three knobs to fight this:

- **β-annealing** (`beta_mode="warmup"`, the default): reduce
  `beta_max` (try 0.1 – 0.5) or extend `warmup_epochs` so the KL term
  doesn't crush the posterior in the first epochs. Cheap and often
  enough ([MVAE §6.2]).
- **Delayed warmup** (`beta_mode="delayed_warmup"`): hold β at
  `beta_min` for `delay_epochs` before starting the ramp to `beta_max`
  over `warmup_epochs`. Reconstruction trains lightly-regularised
  first, and KL only starts to matter once the AE is already good at
  reconstructing. Useful when plain warmup starts KL pressure before
  the model has enough capacity to survive it.
- **Free-bits**: set `free_bits > 0` (typical range 0.05 – 0.5). Each
  latent dim gets that many nats "for free" before the KL term starts
  charging, so the encoder has no incentive to squash any single dim
  to zero. More surgical than β-annealing when the latent is small
  and only a few dims carry all the information ([MVAE §6.3]).
  Compatible with either β-mode.
- **Auto-computed β** (`beta_mode="computed"`): the Asperti-Trentin
  (2020) recipe. Track `gamma_sq` as the running minimum of batch
  MSE across training and set the KL weight to `2 * gamma_sq` at every
  step. Effect: β starts high (matching the initially large
  reconstruction error), keeping latent variables in "limbo" close to
  the prior, and *drops* as reconstruction improves. Individual dims
  get activated one at a time as the decoder starts to need them.
  `beta_max` and `warmup_epochs` are ignored in this mode; look at
  `beta_trajectory.png` for the curve that actually ran.

## CARE-PD GM-CVAE extension

The stack now carries the CARE-PD Parkinsonian-gait plan on top of the
neonate recipes: cohort conditioning, a Gaussian-mixture prior, the
CARE-PD data adapter, and the evaluation battery. All of it is additive —
with `n_cond=0` and `n_components=0` the config is the original plain VAE
and the parameter counts are unchanged.

### The four models ([CARE-PD §7])

Two orthogonal switches on `TrainingConfig` select the model class:

| `n_cond` | `n_components` | Model | What it adds |
|:---:|:---:|:---|:---|
| 0 | 0 | VAE | reconstruction floor, N(0, I) prior |
| >0 | 0 | CVAE | cohort embedding e(c) into encoder + decoder, conditioning dropout — strips the nuisance cohort axis |
| 0 | ≥2 | GM-VAE | K-component mixture prior, EM-trained |
| >0 | ≥2 | GM-CVAE | the target model: mixture prior **and** cohort conditioning |

```python
cfg = TrainingConfig(
    architecture="conv", clip_length=60, n_joints=17, latent_dim=32,
    n_cond=3, cond_dim=8, cond_dropout=0.15,        # CVAE / GM-CVAE
    n_components=5, gm_beta_z=1e-2, gm_beta_y=0.5,   # GM-VAE / GM-CVAE
    gm_train="gradient",             # regular VaDE regime (default); "em" also available
    gm_entropy_weight=1.0, gm_entropy_epochs=5,
    # No N(0,I) term for GM runs — the mixture is the prior. gm_aux_beta
    # defaults to 0; the beta schedule (beta_max) does not weight anything
    # here, though delay_epochs/warmup_epochs still shape the mixture ramp.
)
out = train(cfg, videos, cohort_per_video=cohort_ids)
model, mixture = out["model"], out["mixture"]
```

### How the GM prior is trained ([CARE-PD §7.3])

By default the mixture parameters `(π, μ_c, σ²_c)` are **gradient-trained
jointly with the ELBO** (the regular / VaDE regime), and the GMM is seeded
once from the pre-trained autoencoder's latents just before the KL warm-up
begins. There is **no N(0,I) prior term** — the mixture is the prior, so
`gm_aux_beta` defaults to 0. Set `gm_train="em"` for the EM-inspired
block-coordinate scheme of [GM-VAE §3.3] instead: each epoch does a
gradient pass with the mixture **frozen**, then an EM M-step over the
epoch's cached posterior means updates `(pi, mu, sigma^2)` in closed form
(`GaussianMixturePrior.em_update`); it can re-add the N(0,I) safety tether
via `gm_aux_beta`. The EM regime is more faithful to the paper but prone to
component collapse, which is why gradient is the default. The soft
assignment of a latent point
is the exact posterior `p(c | z)` under the current mixture — there is no
amortised `q(y|x)` head. Per-component occupancy is logged every epoch
(`history["gm_occupancy"]`) so component collapse ([CARE-PD §10]) is
visible from epoch 0; `gm_entropy_weight` adds a decaying entropy bonus to
counter it.

### Data adapter ([CARE-PD §8], `care_pd.py`)

Maps the CARE-PD `h36m/` release into the clip iterator. The release is
what `bash scripts/preprocess_smpl2h36m.sh` emits: one subdirectory per
cohort with `h36m_3d_world_*.npz` plus four camera-projected variants;
each `.npz` is a **flat** dict `{ "subject__walkid": (F, 17, 3) }`.
`load_cohorts` picks the world file (skipping `world2cam*` and
`world2cam2img*`) and splits the `subject__walkid` key so LOSO works.
Preprocessing runs per-frame root-centring (the plan §8 step);
`resample_fps` and `align_direction` are kept but early-return on the
already-canonical release. Windowing is delegated to `build_clips`.

Note **the h36m release carries no labels** — `smpl2h36m.py` exports only
pose arrays. Pass `source_dir=` (or `source_pkl=`) to `load_cohorts` to
attach UPDRS / medication / freezer / `other` from the sibling raw SMPL
`.pkl` (chumpy-free — only the label fields are read). `Walk` gains a
`walk_id` field to match against the source pickle. Training does not
need labels; only the §11 analysis does.

Use **`n_joints=17`** in `TrainingConfig` for the h36m release (the H36M
regressor is the 17-joint standard).

### Metrics ([CARE-PD §11], `metrics.py`)

Frozen-latent evaluators, model-agnostic: `site_probe` (§11.1, two-layer
MLP predicting cohort), `cluster_label_agreement` + `kmeans_labels` /
`hdbscan_labels` (§11.2, ARI/NMI vs UPDRS / freezer / medication),
`linear_probe` (§11.3, UPDRS R² and freezer/medication balanced
accuracy), and `occupancy` (§10). scikit-learn is imported lazily.

`gm_smoke_test.py` trains all four models on synthetic multi-cohort data
and runs the whole battery — read it as a worked example.

## Two small warnings

`ClipDataset` redraws masks per access, so a training epoch sees fresh
masks even on the same clip. That is the point ([MVAE §6.4]), but it
means the validation loss varies from run to run unless you fix the mask
seed. `make_loader` takes a `seed` argument for that.

The transformer's parameter count reported by the analytical helper
excludes LayerNorm scales and biases. The actual model has about 4,000
more parameters than the helper reports. That is not a bug; it is what
the design note rounds away.
