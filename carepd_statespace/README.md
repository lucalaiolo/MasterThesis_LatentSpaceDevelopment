# carepd_statespace

Reproduces the ARHMM movement-state pipeline of **Passmore et al. (2024,
*Sci Rep* 14:28598)** — originally supine-infant video — on the three
**CARE-PD Tier 1 mocap cohorts** (BMCLab, KUL-DT-T, E-LC), pooled, adapted
for gait. Follows the build guideline.

> The reference code (`garedaba/state-space`) is cloned to `references/`
> (gitignored) and its `process_data` / `run_svd` logic reused.

## ARHMM backend (choose one)

The ARHMM is **pluggable** (`backends.py`, `build_arhmm`). Pass `backend=` to
`run_pipeline`; the one used is recorded in `results.json` / `RESULTS.md`.

| `backend` | Library | Notes |
|:---|:---|:---|
| `"ssm"` | Linderman-lab `ssm` | **the paper's implementation** — `ssm.HMM(K, D, init_method="kmeans", observations="ar", observation_kwargs={"lags": L}, method="stochastic_em", transitions="standard")`, exactly as `ARHMM.ipynb`. Fussy install: `pip install "numpy<1.24" cython` then `pip install git+https://github.com/lindermanlab/ssm`. |
| `"dynamax"` | `dynamax` (JAX) | maintained fallback; `LinearAutoregressiveHMM`. `pip install jax dynamax`. Variable-length walks are right-padded, so prefer `ssm` for a strict reproduction. |
| `"numpy"` | built-in | **default**, always available; self-contained EM ARHMM. |
| `"auto"` | — | prefers `ssm` → `dynamax` → `numpy`. |

```python
run_pipeline(data, backend="ssm")     # follow the paper exactly
```

`available_backends()` reports which import in the current environment.
Parameter averaging over refits (§4.4) is done for the NumPy backend; for
`ssm`/`dynamax` the best-log-likelihood restart is used (still aligned).

## Install

```
pip install numpy scipy scikit-learn pandas matplotlib statsmodels
pip install hmmlearn                    # optional
```

## Run (once the CARE-PD h36m release is available)

The CARE-PD h36m release keeps **3D joints and labels in two separate
files** (this is the thing that trips people up): the per-cohort
`h36m_3d_world_*.npz` holds only the `(F, 17, 3)` joints keyed
`subject__walkid`, while the source SMPL `.pkl` (e.g. the Borealis `TWIKMK`
deposit) holds only SMPL params + clinical labels. `load_h36m_cohorts`
reads the joints from the `.npz` and joins the labels from the `.pkl` by
walk id — mirroring the release reader in `architectures/care_pd.py`.

```python
from carepd_statespace.carepd_adapter import load_h36m_cohorts, build_dataset
from carepd_statespace.driver import run_pipeline

walks = load_h36m_cohorts(
    "/content/drive/MyDrive/CARE-PD_h36m",             # <root>/<cohort>/h36m_3d_world_*.npz
    source_dir="/content/assets/datasets/TWIKMK")      # <source_dir>/<cohort>.pkl (labels)
data = build_dataset(walks, feature_set="B")     # HumanML3D decomposition (default)
out  = run_pipeline(data)                          # -> carepd_statespace/outputs/ + RESULTS.md
```

`load_h36m_cohorts` defaults to the three Tier-1 cohorts; pass `cohorts=` to
subset, `npz_paths=` / `pkl_paths=` to point at files individually, or omit
`source_dir` to run label-free (only the §6 clinical analysis needs labels).
`load_cohort_pkls(...)` remains for a hand-built pickle that *already*
carries joints under `joints_key` (or via a `pose_to_joints` regressor).

`smoke_test.py` runs the whole pipeline on synthetic 3-cohort gait with a
planted freezing signal (no data / no GPU needed) and reaches a **GO**
verdict — read it as a worked example.

## What each stage does (guideline §)

| Module | § | What it does |
|:---|:---|:---|
| `carepd_adapter.py` | §2 | loads H36M-17 joints (`.npz`) + labels (source `.pkl`) and joins them (`load_h36m_cohorts`), egocentric norm (root-centre + **per-frame heading**), Set A / **Set B** (+root vel/height), gentle band-pass + MAD + resample to 15 Hz, variable-length walks + `info` |
| `principal_movements.py` | §3 | balanced-subset RobustScaler+SVD basis, 90%-var + **TWO-NN** count, project + velocity, variance curve (Fig 1b), **site diagnostic** |
| `statespace.py` | §4 | NumPy **ARHMM** (EM, log-space FB, AR(L), Viterbi) + GMM + Gaussian HMM (`L=0`); Hungarian state alignment |
| `driver.py` | §4, §8 | subject-CV over K×L (Fig 1d), stable refit×25 with alignment+averaging, generative check (Fig 1e), go/no-go gate |
| `analysis.py` | §5-§7 | occupancy/**dwell**/**transition** metrics, subject aggregation, clinical **MixedLM** (FoG primary, medication paired, UPDRS), Bonferroni over K, state characterisation (Fig 2, Fig 3) |

## Key adaptations from the infant original

- 3D joints (17×3) **+ 4 root channels** (Set B), not 2D keypoints.
- **Per-frame** heading alignment (not per-walk — preserves KUL-DT-T turns).
- 15 Hz, AR lag grid to **8** (a step is ~7-8 frames at 15 Hz).
- CV split **by subject** (not by video) — many walks/subject, both med states.
- Clinical axis led by **FoG**; metric families are occupancy **+ dwell +
  transitions** (freezing ≈ a breakdown in state progression).

## The go/no-go gate (§8)

After Bonferroni, does **any** state statistic (occupancy, dwell, transition)
separate FoG / medication / UPDRS **beyond cohort** (cohort + sex are LME
covariates)? **GO** → states carry clinical signal, proceed to Tier 2. **NO-GO**
→ retry with longer lag, Set A vs B, or a warped ARHMM (Costacurta et al. 2022)
before abandoning. The verdict is written to `outputs/RESULTS.md`.

## Guideline gotchas, handled

- No `ssm` install pain — NumPy ARHMM.
- Refits are Hungarian-aligned before averaging (`statespace.align_states`).
- Per-**frame** heading (turns preserved via root angular velocity channel).
- CV **by subject**; short-walk cohorts **aggregated to subject** for stats.
- Band-pass keeps the cadence fundamental (high-pass only 0.01 Hz).
