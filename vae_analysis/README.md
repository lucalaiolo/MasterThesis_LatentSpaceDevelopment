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

## Two cautions carried over from the notes

The screening score (§21) means "unlike the training set", not
"abnormal". With a handful of infants the training set is not the
population, so treat it as a research signal.

Between-video claims need `honesty`. Frame-level and clip-level
statistics stand on their own; a difference between two infants does not,
until the sample grows.
```
