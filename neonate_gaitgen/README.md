# neonate_gaitgen

Phase A of the neonate GAITGen build plan: GAITGen's motion/pathology
**disentangled Residual VQ-VAE** (paper Sec. 3.1.1) adapted to **2D neonate
keypoints**. Two-dimensional positions, so the SO(3) geodesic rotation loss
(paper Eq. 4) is **dropped** — only the L1 position loss (Eq. 3) remains.
The conditioning label is generic (`label_type ∈ {ordinal, nominal}`), not
hard-coded to CARE-PD severity.

> The GAITGen reference repo (`vadeli/GAITGen`) is "code coming soon", so
> the model is **clean-room from the paper**. `references/` is gitignored.

## Install

```
pip install numpy scipy scikit-learn matplotlib torch
pip install umap-learn hmmlearn        # optional: UMAP panels, token HMM
```

## Run (once the dataset arrives)

```python
from neonate_gaitgen import GaitGenConfig, train
from neonate_gaitgen.preprocess import Sequence, build_windowed_data
from neonate_gaitgen.analysis import run_analysis

# 1) wrap each clip: pose (F, J, 2), c_p int label, subject id
seqs = [Sequence(pose=arr, c_p=label, subject=sid) for ...]

cfg  = GaitGenConfig(n_joints=J, n_classes=2, label_type="nominal",
                     healthy_id=0)            # set J / labels to the real data
data = build_windowed_data(seqs, cfg)         # root-centre, scale, window
out  = train(cfg, data)                       # two-stage (paper §3.5)
run_analysis(out["model"], data, cfg)         # -> outputs/gaitgen_neonate/
```

`smoke_test.py` runs the whole Phase-A path on synthetic 2D gait (no real
data / no GPU needed) — read it as a worked example.

## Layout

| Module | Plan / paper | What it does |
|:---|:---|:---|
| `config.py`        | §3, §4 | `GaitGenConfig` — J, `label_type`, dims, codebooks, loss weights (Eq. 7), two-stage schedule, healthy latent-dropout |
| `preprocess.py`    | §2 | root-centre, torso-scale, **no mirroring**, 60-frame 50%-overlap windows, `(T, J*2)`; `Sequence`/`WindowedData`, subject split |
| `models/rvq.py`    | Eqs. 1-2, 8 | `ResidualVQ` — EMA codebooks, dead-code reset, quant-dropout |
| `models/networks.py` | Sec. 3.1.1 | conv-ResNet motion (uncond.) + pathology (c_p-cond.) encoders, mirror decoder, pathology + adversarial classifiers |
| `models/gaitgen.py`| Eqs. 3-8 | `DisentangledRVQVAE` — two-stage loss, `decode(q_m,q_p,α)`, `encode_latents` |
| `train.py`         | §3.5 | Stage 1 (E_m+D recon) → Stage 2 (joint, classifiers, GRL ramp, healthy zero-out) |
| `analysis/`        | §6-§8 | PORE/PMPG/DS, probes, HSIC, paired UMAP, clustering, codebook, token HMM, `run_analysis`, `summary.md` |
| `analysis/baselines.py` | §7 | PCA, plain continuous VAE, shared `evaluate_latent` |

## What's validated

The synthetic smoke test (`python -m neonate_gaitgen.smoke_test`) reaches:
stage-1 reconstruction falling, **PMPG ≈ +0.6** (q_p→c_p ≈ 1.0 vs q_m→c_p ≈
0.4 at chance 0.33), and **PORE > 0** — i.e. pathology lands in `q_p` and is
absent from `q_m`, the disentanglement claim, in the right direction.

## Not built (deliberately)

- **Phase B** — Mask + Residual Transformers for conditional generation and
  Mix-and-Match (paper Sec. 3.1.2-3). Only after Phase A is reported.
- **§6.10 Mix-and-Match** and the **§6.8 per-code stick-figure animations**
  are qualitative figures; `codebook.decode_single_code` provides the decode
  hook, but rendering is left to the caller.
- **GAITGen_w/o-dis** baseline: train the main model with
  `lambda_cls = lambda_adv = 0` and evaluate its combined latent through
  `analysis.baselines.evaluate_latent`.

## Units & rigor guardrails ([plan §9])

- Reconstruction error is reported in **normalised units** (torso-scaled),
  never relabelled as mm.
- No SO(3) rotation loss (2D data) — stated in `models/gaitgen.py`.
- Every split is **by subject**, never by clip.
