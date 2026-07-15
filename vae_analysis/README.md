# vae_analysis

A toolkit for the latent-space analyses in the two design notes for the
masked neonate-motion VAE. Every function is generic in the joint count
J, the clip length T, and the latent width d_z. Nothing assumes J = 22.

## Install the dependencies

Core (needed): `numpy`, `scipy`, `scikit-learn`.

Model-dependent (needed for the Jacobian and attention tools): `torch`.

Optional (each guarded, with a clear message if absent):
`ripser` and `persim` for persistent homology, `hmmlearn` for the hidden
Markov model, `ruptures` for change-point detection.

```
pip install numpy scipy scikit-learn torch
pip install ripser persim hmmlearn ruptures   # optional
```

## Wire in your model

Wrap your trained network once, following `adapter_example.py`:

```python
from adapter_example import TorchVAEAdapter, example_skeleton
model = TorchVAEAdapter(net)          # your trained torch VAE
skel = example_skeleton()             # replace with your joint layout
```

Then encode a dataset and run any analysis:

```python
from vae_analysis import encode_dataset
from vae_analysis import posterior_geometry as pg, symmetry as sym

latent = encode_dataset(model, X, M, video_id=vid, time_index=t0)
print(pg.mmd_prior_test(latent))                  # aggregate posterior vs prior
eq = sym.fit_equivariance(model, X, M, skel)      # laterality subspace
```

`smoke_test.py` runs every core path against a fake model and synthetic
data; read it as a full worked example.

## Map to the design notes

| Module | Notes | What it does |
|:---|:---|:---|
| `posterior_geometry` | I §3 | MMD vs prior, intrinsic dimension, clusters |
| `decoder_geometry`   | I §4, II §17 | Jacobian maps, traversals, pullback metric, geodesics |
| `encoder_geometry`   | II §14 | encoder Jacobian, precision spectrum, read-write mismatch |
| `features`           | I §5 | kinematic features, ridge regression, canonical correlation |
| `masking`            | I §6 | mask-jitter, latent recovery, split MPJPE |
| `dynamics`           | I §7, II §22 | sliding windows, change points, hidden Markov model, Ornstein-Uhlenbeck |
| `generation`         | I §8 | bone-length variation, Frechet distance, interpolation curvature |
| `information`        | I §9, II §19 | total-correlation split, active units, rate-distortion |
| `symmetry`           | II §15 | flip operator, equivariance fit, asymmetry score |
| `disentanglement`    | II §16 | MIG, DCI, SAP, selectivity control |
| `two_sample`         | II §18, §20 | persistent homology, classifier two-sample test |
| `screening`          | II §21, §23 | density typicality, attention entropy |
| `honesty`            | I §12 | block bootstrap, permutation test |

## Post-hoc CARE-PD analysis (`posthoc/`)

The `posthoc` subpackage implements the CARE-PD post-hoc structure plan.
The mixture-prior models (GM-VAE / GM-CVAE) are **off the pipeline**
(component collapse); the phenotype claim is made post hoc on the two core
latents — the plain **VAE** (baseline) and the **CVAE** (target) — and
scored against the clinical labels. One entry point runs the whole battery
and writes to `outputs/posthoc/`:

```python
from architectures.care_pd import load_cohorts, build_bundle, TIER1_COHORTS
from vae_analysis.posthoc import run_posthoc

walks  = load_cohorts("data/h36m", TIER1_COHORTS, source_dir="data/smpl")
bundle = build_bundle(walks)
out = run_posthoc(vae_checkpoint="runs/vae/best.pt",
                  cvae_checkpoint="runs/cvae/best.pt",
                  bundle=bundle)          # -> outputs/posthoc/{*.png, *.csv, results.json, summary.md}
```

| Module | Plan § | What it does |
|:---|:---|:---|
| `data`       | §1, §4.1 | cohort-aware encoding to μ, clip/label alignment, outer-loop trajectories |
| `palette`    | §6 | fixed cohort / state / model colours, sequential/diverging maps, `save_fig` ≥150 dpi |
| `clustering` | §1, §2, §2.1 | BIC vs K, k-means / GMM / HDBSCAN, reseed/subsample/bootstrap/cross-method stability, consensus matrix |
| `agreement`  | §2.2, §2.3, §3 | ARI/NMI vs UPDRS/freezer/med/cohort, PCA+UMAP panels, subject composition, within-severity substructure |
| `temporal`   | §4 | Gaussian HMM regimes (BIC, dwell, transitions, label usage) + PELT change points validated on E-LC FoG |
| `probes`     | §5 | subject-split ridge/logistic phenotype probes + the site probe, VAE vs CVAE bar chart |
| `report`     | §7 | `summary.md` with a plain-language verdict per section |
| `driver`     | — | `run_posthoc` orchestration → `outputs/posthoc/` |

`posthoc/smoke_test.py` runs the whole thing on synthetic multi-cohort data
with fake encoders (no torch needed) — read it as a worked example.
Optional deps `umap-learn`, `hmmlearn`, `ruptures` unlock the UMAP panels,
the HMM, and PELT; each is guarded and skips cleanly when absent.

## Two cautions carried over from the notes

The screening score (§21) means "unlike the training set", not
"abnormal". With a handful of infants the training set is not the
population, so treat it as a research signal.

Between-video claims need `honesty`. Frame-level and clip-level
statistics stand on their own; a difference between two infants does not,
until the sample grows.
```
