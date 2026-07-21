# Temporal-Latent Motion VAE — Handoff / New-Chat Primer

> Purpose: a self-contained briefing so a fresh session can pick up this thesis
> work without re-deriving context. It states **what the model is now**, **why
> we moved to it**, and **which analyses to run on it** (with the subtleties the
> temporal latent introduces).

---

## 1. Project in one paragraph

We are learning a latent space for **human motion** from **2D pose sequences**
(YouTube videos, COCO-18 keypoints, `(F, J=18, D=2)` per video; the pipeline is
generic and also runs on CARE-PD 3D H36M). The model is a **VAE** trained on
short clips (`T` frames). The goal is a latent that is (a) faithful — the
reconstruction actually *moves* like the input — and (b) *analyzable*: its
geometry, intrinsic dimension, dynamics, and cluster structure should carry
interpretable motion information (eventual phenotype-level questions).

## 2. The core lesson: single-global-latent collapse → temporal latent

The earlier models (`conv`, `transformer`, and the `factorized`/`anchored`
variants) compressed a whole clip to **one latent vector** `z ∈ ℝ^{d_z}`. Every
one of them **collapsed the reconstruction to a static, temporally-averaged
pose** — the decoder ignored the (small) motion signal because outputting the
mean pose already minimizes the MSE. This was robust to `β=0`, huge velocity
loss (`λ_vel=1000`), LR warmup, and more epochs. We root-caused it: a single
global latent has no *per-time* capacity, so the ELBO's easiest optimum is "emit
the mean." The **anchored / FiLM** attempt (condition on mean pose + scale,
decode a residual) made it worse in the diagnostic sense — it drove the residual
`r̂ → 0` because the anchor *is* the mean.

**The fix (implemented, verified moving): a latent _sequence_, not a vector.**
Downsample time by a factor `l` and keep a `d_z`-wide latent **per window**. This
is the continuous (non-VQ) analogue of the temporally-downsampled latent used by
T2M-GPT / MotionGPT / PRISM. Because a distinct latent drives each window, the
decoded clip *cannot* collapse to one pose.

### 2.1 The two temporal architectures (both live, both verified)

| `ARCHITECTURE`        | backbone                        | file |
|-----------------------|---------------------------------|------|
| `temporal_conv`       | 1-D conv trunk (per-window head)| `architectures/models/temporal_conv_vae.py` |
| `temporal_transformer`| frame-token pre-norm transformer| `architectures/models/temporal_transformer_vae.py` |

`temporal_conv` is the **recommended default** — at 60 epochs its reconstruction
temporal std reaches ~94% of the target (it moves); it is small and fast.
`temporal_transformer` also moves and is the attention counterpart.

### 2.2 Latent dimension formula (important — `d_z` is *per window*)

```
d_total = (T / l) × d_z          # windows × width-per-window
```

- `T` = `CLIP_LENGTH`, `d_z` = `LATENT_DIM` (per window, **not** whole-clip).
- `temporal_conv`:        `l = ∏ CONV_STRIDES`
- `temporal_transformer`: `l = TEMPORAL_DOWNSAMPLE`
- **Hard constraint:** `T` must be divisible by `l` (enforced in `validate()`).
- Legacy `conv` / `transformer`: `d_total = d_z` (whole clip).

Current defaults: `T=48`, strides `(1,2,2)` → `l=4` → `48/4 = 6` windows × `d_z=8`
= **48 total latent units** (a `(6, 8)` latent sequence, flattened to 48 so the
standard ELBO/KL/β machinery and the analysis toolkit run unchanged).
`model.pool_global(z)` mean-pools the windows to one `(d_z,)` per-clip descriptor
for phenotype-level statistics.

## 3. Training knobs that matter

- **β (KL) schedule is fundamental** — `beta_mode` (`warmup`/`delayed_warmup`/
  `computed`), `beta_max`, `warmup_epochs`, `delay_epochs`, `free_bits`. Collapse
  vs. blur is tuned here. Watch final KL nats (→0 ⇒ posterior collapse).
- **Velocity loss** — `lambda_velocity` adds an MSE on frame-differences
  (temporal smoothness); helped, but was *not* what fixed motion (the temporal
  latent was).
- **Recipe / masking** — `recipe` (1/2/3), `mask_policy`, `mask_rho`. Recipe 3 is
  masked-inpainting.
- **Train/val split is now video-wise** (`architectures/data.py`,
  `train_val_split`): whole videos are held out (~15%), subjects disjoint — the
  honest split for a generalisation claim. Seed-deterministic; single-video
  datasets fall back to a within-video time cut.

The single Colab cell (`scratchpad/colab_cell.py`) exposes **all** of the above,
switches cleanly between the four architectures, and runs: train → β diagnostics
→ encode → motion readout → Jacobian panels → reconstruction video → traversal.

## 4. Analysis toolkit (`vae_analysis/`) and how the temporal latent changes it

Everything runs through `ArchitecturesAdapter` + `encode_dataset`. The key
subtlety: the latent is now a **flattened `(6×8)` sequence**, so per-*dim*
analyses still work, but "one vector per clip" summaries should use
`model.pool_global`, and *dynamics* should exploit the within-clip window axis.

### 4.1 Decoder geometry — the Jacobian analyses (`decoder_geometry.py`)
- `decoder_jacobian` (via `jacrev`), `sensitivity_maps` (joint×latent,
  time×latent), `metric_spectrum` / `pullback_metric` (G = JᵀJ, condition
  number), `geodesic`, `path_curvature`, `measured_traversal`.
- **Temporal reading:** each latent dim is now one `(window, channel)`. The
  `time × latent` sensitivity map should show each dim lighting up **its own
  window** — a direct visual check that the latent is temporally localized (the
  whole point of the redesign). A dim that influences all frames equally would
  signal the window structure isn't being used.

### 4.2 Encoder geometry (`encoder_geometry.py`)
- `encoder_jacobian`, `encoder_sensitivity_map` (latent×joint read map),
  `precision_spectrum` (posterior precision per dim → **live/active dims**),
  `read_write_mismatch`.
- **Temporal reading:** `precision_spectrum` now reports live dims out of
  `d_total = 48`. Expect several live per window; all-dead windows mean that part
  of the clip carries no information.

### 4.3 Prior fit / latent shape (`posterior_geometry.py`)
- `mmd_prior_test` (is q(z) ≈ N(0,I)? — **loses power in high-D**, so read it
  alongside per-dim precision, not alone), `intrinsic_dimension_twonn` (TwoNN
  intrinsic dim), `cluster_structure` (GMM).
- **Temporal caveat:** with 48 flattened dims the aggregate posterior is
  higher-D; TwoNN on the **pooled** per-clip latent (`pool_global`) is the more
  interpretable intrinsic-dimension number for phenotype questions.

### 4.4 Dynamics — the temporal payoff (`dynamics.py`)
- `encode_video` → per-clip latent **trajectory** `(K, d_z)`; `hmm_states`
  (`covariance_type="diag"`, hardened for wide latents; raises on too-short
  trajectories), `change_points`, `ou_process` (Ornstein-Uhlenbeck fit),
  `sliding_windows`.
- **This is where the temporal latent pays off.** Recommended next step (offered,
  not yet built): an `encode_window_sequence` helper that returns the *within-
  clip* `(T/l, d_z)` window sequence so HMM / change-points / OU run over motion
  *inside* a clip, not just across clips. The building blocks exist.

### 4.5 Two-sample & structure (`two_sample.py`, `information.py`, `disentanglement.py`, `screening.py`)
- `classifier_two_sample` (**C2ST** — the high-D-robust alternative to MMD),
  `persistent_homology`.
- `tc_decomposition` (total-correlation), `active_units`, `rate_distortion_curve`.
- `mig`, `dci`, `sap`, `selectivity` (disentanglement — needs factors of
  variation to be meaningful).
- `typicality_score`, `screening_auc`, `attention_entropy`, `selected_frames`
  (masking/screening arm).

## 5. Recommended next steps (priority order)

1. **`encode_window_sequence` + route dynamics through it.** Unlock HMM /
   change-points / OU on the *within-clip* window sequence — the natural home for
   the new temporal latent. (§4.4)
2. **Hybrid global+temporal head for phenotype clustering.** Add a pooled global
   latent alongside the per-window sequence so clustering/cluster_structure has a
   clean per-subject vector while motion stays in the windows. (`pool_global`
   exists; a dedicated trained head would be cleaner than mean-pooling.)
3. **β / `d_z` / `l` sweep on the temporal model.** The collapse-vs-blur frontier
   moved; re-tune with the video-wise split (results are now generalisation
   numbers). Watch final KL nats + motion std together.
4. **Confirm the PRISM citation** before it goes in the thesis (arxiv PDF 403'd
   during research — the reference string is unverified).

## 6. Repo pointers

- Models: `architectures/models/{temporal_conv_vae,temporal_transformer_vae}.py`;
  factory `architectures/models/__init__.py` (`build_model`).
- Config + validation: `architectures/config.py` (`TrainingConfig`,
  `temporal_downsample`, per-side depth resolvers, `validate()`).
- Training loop / sweep: `architectures/train.py` (`train`, `train_sweep`,
  `model_selection`; velocity term in `_step_loss`).
- Data / split: `architectures/data.py` (`build_clips`, `train_val_split` —
  now video-wise).
- Losses: `architectures/losses.py` (`reconstruction_velocity_mse`).
- Analysis: `vae_analysis/` (see §4); adapter
  `vae_analysis/architectures_adapter.py`.
- Runnable cell: `scratchpad/colab_cell.py` (all knobs, 4 architectures).
- Working branch: `claude/vae-architecture-deep-models-s4hkqa`.

## 7. One-line status

The motion-collapse problem is **solved** by the temporal-latent design
(`temporal_conv` default, `temporal_transformer` available); the split is now
**video-wise**; the analysis toolkit runs on the temporal latent with the
dynamics-over-windows extension as the top next step.
