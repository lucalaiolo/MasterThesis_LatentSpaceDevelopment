# Deprecated GM-VAE / GM-CVAE checkpoints

This directory is the archive for any **GM-VAE / GM-CVAE** checkpoints that
were trained before the mixture-prior models were removed from the active
pipeline ([post-hoc analysis plan §0]).

## Why they are here and not in the pipeline

The mixture-prior models suffer **component collapse** in this setting:
when the latent is not cleanly multimodal, a mixture prior has nothing to
latch onto and dumps its mass onto one or two components. The thesis claim
is representational and does not require the clustering to be baked into
the prior:

- **Nuisance removal** is measured by the site probe (plain VAE vs CVAE).
- **Phenotype recovery** is measured by clustering the plain VAE / CVAE
  latent **post hoc** and scoring agreement against clinical labels
  (see `vae_analysis/posthoc/`).

## Policy

- The GM-VAE / GM-CVAE **source is kept** (`architectures/models/gaussian_mixture.py`),
  but marked deprecated and guarded: training one requires
  `TrainingConfig(allow_deprecated_gmvae=True)`, which no default run sets.
- Any GM checkpoints that already exist should be **moved here** for the
  record. **Do not use them downstream** — the post-hoc analysis runs on
  the plain VAE and CVAE checkpoints only.

Drop the archived `*.pt` files (and their `history.json` / `config`)
alongside this README.
