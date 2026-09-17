# Latent-Space Development for Infant Motion

A latent representation of human motion learned from 2D keypoint sequences, used
as the substrate for an interpretable analysis of infant **general movements**.
The clinical target is the **fidgety movements** of Prechtl's General Movements
Assessment: the small, continual movements of neck, trunk and limbs normally
present at roughly 9–20 weeks post-term, whose *absence* is an early marker of
neurological risk.

The full method — every model, loss, selection rule, statistical test and figure,
with the literature it rests on — is in **[`docs/THESIS_COMPANION.md`](docs/THESIS_COMPANION.md)**.

## The pipeline

1. **Pretrain** a temporal-latent VAE on a large corpus of 2D poses from web
   video (`architectures/`, driven by `youtube_motion/`).
2. **Fine-tune** it on the clinical cohort — RVI-38, 38 recordings of 38 infants
   aged 12–21 weeks, each carrying one GMA label (32 normal / 6 abnormal).
3. **Fit a temporal state model** — an HMM or AR-HMM — over the per-recording
   latent trajectory (`vae_analysis/hmm_pipeline.py`, `arhmm.py`).
4. **Read out the clinical endpoints** and test them against the label
   (`rvi38_methods/`).

Stage 4 produces four endpoints, each reported on its own against an exact
permutation null: fluency `Φ`, Kemeny constant `K`, WCLR-PP inter-limb
synchrony `mean F`, and `FF` from FidgetyFind — a published detector this project
did not design, computed from the keypoints alone as an external yardstick.

The run ends on the **abnormality index** (`rvi38_methods/a11_index.py`), the one
place the four are combined. Each is standardised against the *normal*
recordings alone, signed by its pathological pole, and reduced to the largest of
them, `T(x) = maxⱼ dⱼ z(x)ⱼ`; a recording is flagged above the Bonferroni normal
quantile `τ` with `Pr(Z ≤ τ) = 1 − α/p` (`α = 0.05`, `p = 4`, so `τ ≈ 2.24`). The
maximum, because the assessment needs only one abnormal pattern to fire. No
parameter in it is fitted against the label.

## Layout

| Directory | What it holds |
|:---|:---|
| `architectures/` | the VAE itself — config, masking policies, conv and transformer backbones (whole-clip and temporal-latent), training loop, losses, evaluation, parameter counts |
| `youtube_motion/` | the pretraining corpus: long-format keypoint loader, BODY-15 / COCO-18 skeletons, the configuration sweep and its ranked report |
| `vae_analysis/` | the latent-geometry toolkit used for model selection, plus the HMM / AR-HMM interpretability layer (`hmm_pipeline`, `arhmm`, `hmm_report`) |
| `rvi38_methods/` | the clinical analysis: the four endpoints, the inference layer, the abnormality index, the figures, and the end-to-end runner |
| `docs/` | the technical companion and the three reference notes it points at |
| `tests/` | cross-cutting checks (subject-wise CV partitioning) |

Each package carries its own README with the detail.

## Install

```
pip install numpy scipy scikit-learn pandas matplotlib joblib
pip install torch                     # training and any Jacobian-based analysis
pip install hmmlearn                  # the Gaussian HMM
pip install ripser persim ruptures    # optional: homology, change points
```

`ssm` (the AR-HMM) installs from source and is fussy about its NumPy pin; see
`vae_analysis/arhmm.py` for what it needs. Every optional dependency is guarded
and skips with a clear message when absent.

## Run

**Sweep the VAE configurations** on the pretraining corpus, ranked by held-out
reconstruction:

```
python -m youtube_motion.driver --csv keypoints.csv \
    --out checkpoints/youtube_motion
```

**Fit the state model and render every figure** from a frozen checkpoint:

```python
from vae_analysis.architectures_adapter import ArchitecturesAdapter
from vae_analysis.hmm_report import run_hmm_report

res = run_hmm_report(ArchitecturesAdapter(net), videos,
                     bones=bones, limbs=limbs, clip_len=64,
                     model="arhmm", stream="pose",
                     labels=labels, out_dir="hmm_out")
```

**Run the clinical analysis** end to end:

```
cd rvi38_methods
python run_analysis.py --csv rvi38_analysis.csv \
    --model "K=11=arhmm_k11.pkl" \
    --model "K=14=arhmm_k14.pkl" \
    --labels RVI_38_labels.mat --outdir rvi38_out
```

`--fast` cuts every resampling count ~20× for a smoke run. Every package also
ships a smoke test that runs its full path on synthetic data — read those as
worked examples.

## How to read the results

The cohort is 38 recordings with 6 positives. Every clinical result is
exploratory and every interval is wide; window and clip counts are estimation
resources, not sample size. The endpoints are reported uncorrected for one
another, so a small *p* on one of several is a lead, not a result. The honesty
ledger in the companion (§11) states each limitation the thesis must carry,
including what FidgetyFind's adaptation to a keypoint-only cohort costs.
