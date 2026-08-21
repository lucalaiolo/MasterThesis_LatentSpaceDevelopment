# RVI-38: fluency, mixing and raw-kinematic analyses

Implementation of `METHODS_3.md` — fluency (A1) and mixing structure (A7) on the
RVI-38 cohort, plus the raw-kinematic constructs (WCLR-PP inter-limb
coordination for cramped-synchronised movements, and the per-state velocity
profiles), with the inference layer of METHODS §Inference.
Alongside them sits **FidgetyFind** (`a10_fidgetyfind.py`), the published
detector of the fidgety movements the GMA label is about: a construct nobody
here designed, computed from the keypoints alone, against which the
model-based ones can be read on the identical cohort.

The metastable decomposition (PCCA+, implied timescales, the kinematic
dendrogram and their agreement statistics) has been removed. The earlier
cosine-Gram co-movement construct has been superseded by WCLR-PP
(`a9_wclrpp.py`), a vector-valued directed measure that separates genuine
coupling from shared limb autocorrelation and preserves lead-lag phase.

## Module map (§12.2)

| Module | Responsibility |
|---|---|
| `load_models.py` | lazy stub package for `ssm`/`autograd`; recovers arrays from joblib dumps; normalises both dict layouts to one schema |
| `build_pose.py` | long CSV to per-video `(F,15,2)`; verifies frame contiguity and the constant-joint property |
| `a1_core.py` | window-to-frame geometry, state profiles, the direction-aware second-moment signature (`M`, its double-angle shape coordinate `u`) and the magnitude+shape similarity, fluency estimators |
| `fluency_curve.py` | exploratory temporal decomposition of `Φ`: the Gaussian-on-index curve `φ(t)` whose flat-kernel limit is `Φ` (reuses `S`, the visit sequence and the cached null) |
| `a1_stats.py` | the whole reported inference and nothing else: exact Mann-Whitney U, AUC, the stratified percentile bootstrap, and the Pearson/Spearman correlation table |
| `a57_graph.py` | jump chain, fundamental matrix, MFPT, Kemeny, shrinkage |
| `a8_movement.py` | raw kinematics: per-state velocity profiles |
| `a9_wclrpp.py` | WCLR-PP inter-limb coordination: vector-valued conditional limb regression, peak-picking, per-pair F/R2, circular-shift surrogate null |
| `a10_fidgetyfind.py` | FidgetyFind (Morais et al., 2023, as adapted in METHODS §FidgetyFind): small-amplitude displacement direction entropy per window on six limb chains, reduced to `FF`, `FF_hip`, `FF_dist` |
| `report.py` | one call that runs every construct, summarises it and collects the figures (`run_report`) |
| `figures.py` | figure panels, every annotation computed from the run |
| `run_analysis.py` | end-to-end runner |
| `test_methods.py` | 136 checks with a definite right answer (§12.4 style) |
| `make_synthetic.py` | synthetic cohort with planted structure, for smoke tests |

## Run

    pip install numpy scipy pandas joblib scikit-learn matplotlib
    python run_analysis.py --csv rvi38_analysis.csv \
        --model "K=11=arhmm_k11.pkl" \
        --model "K=14=arhmm_k14.pkl" \
        --labels RVI_38_labels.mat --outdir rvi38_out

`--fast` cuts every resampling count ~20x for a smoke run.

### Choosing the models

`--model` is repeatable and takes as many fitted models as you like, in any
mix: two AR-HMMs at different `K`, an AR-HMM and a Gaussian HMM, one model,
five. `NAME=PATH` names a model for the report; a bare path is named after the
file, which is enough to keep two fits of the same kind apart. The name may
itself contain `=` (`"K=11=fit.pkl"` works).

The **first** model loaded is the primary one — the clinical layer, the
correlation analysis and every figure are computed on it — unless `--primary
NAME` says otherwise. Every other model is a **replication**: its fluency and
its Kemeny constant are correlated (Spearman, across the whole cohort) against
the primary's, which asks whether the ordering of the infants survives
refitting the state model. Any number of replications is fine; they are all
reported.

`--arhmm PATH` and `--hmm PATH` remain as shorthands for
`--model "AR-HMM=PATH"` and `--model "Gaussian HMM=PATH"`; with no model flags
at all the two legacy default filenames are tried.

### One call for everything (`report.py`)

`run_report` runs the five constructs the results chapter reports — fluency,
the fluency curve, the Kemeny mixing time, WCLR-PP synchrony and FidgetyFind —
with the settings that produce the complete picture (the fluency curve and the
per-recording FidgetyFind panels are on), and returns the results, a headline
summary and every figure path:

```python
from report import run_report

out = run_report(csv="rvi38_analysis.csv",
                 models={"K=11": "arhmm_k11.pkl",     # first = primary
                         "K=14": "arhmm_k14.pkl"},    # the rest = replications
                 labels="RVI_38_labels.mat",
                 outdir="rvi38_out",
                 show="main")          # display the figures in a notebook

out["summary"]["fidgetyfind"]["group"]["auc"]
```

`models` takes one path, a list of paths, a list of `(name, path)` pairs or a
`{name: path}` dict; `primary="K=14"` picks the primary model when it is not
the first.

It writes `summary.md` and `summary.json` next to the run alongside everything
`run_analysis.py` writes. Keyword arguments pass through to the runner
(`fluency_omega=0.7` becomes `--fluency-omega 0.7`), `fast=True` is the smoke
run, and `synchrony=False` skips WCLR-PP, by far the slowest block. Paste
`colab_rvi38_report.py` as a single Colab cell to run the same thing there.

### Direction-aware fluency (`DIRECTION_AWARE_KINEMATIC_SIMILARITY.md`)

The similarity `S` behind the fluency measure `Φ` is built from a
**direction-aware** state signature. Each free joint's motion in a state is
summarised not by a single RMS speed but by the pooled second-moment matrix
`M[k,j] = mean(v vᵀ)`, whose trace is the same `a[k,j]²` as before and whose
trace-normalised part is the double-angle axis coordinate
`u = (ρ cos 2θ, ρ sin 2θ)` — anisotropy `ρ = ‖u‖ ∈ [0,1]` and principal axis
`θ = ∠u / 2` (mod π). `S` then combines a **magnitude** channel (the log-RMS-speed
correlation, unchanged) with a **shape** channel (the residual-axis cosine). This
is local to the signature: no model is refitted and one statistic enters where
one left.

* `--fluency-similarity {separated,concatenated,scalar}` — how the channels are
  combined (default `separated`). `separated` (preferred) is
  `S = ω·S_mag + (1−ω)·S_shape`; `concatenated` standardises and stacks all `3J`
  residuals into one correlation; `scalar` is the legacy magnitude-only
  similarity (`ω = 1`), i.e. the §11 sensitivity check against the
  direction-blind version.
* `--fluency-omega Ω` — magnitude weight in the separated form (default `0.5`,
  fixed a priori, in `[0,1]`). `Ω = 1` recovers the scalar measure, `Ω = 0` is
  shape-only.
* `--fluency-drop-state-term` — drop the per-state term from the shape
  residualisation (keep only the per-joint anatomical-axis removal). Off by
  default (state term kept, matching the magnitude channel); this is the one
  free modelling choice, reported both ways as a sensitivity check.

The run reports a **magnitude-versus-direction split**: `Φ` recomputed under
each channel and contrasted across groups, so the study can say whether fluency
rides on how much a joint moves or on the axis along which it moves.

### Temporal decomposition of `Φ` (`fluency_curve.py`)

`Φ` is one number per recording — the excess similarity of consecutive
movements, averaged over every visit transition. `--fluency-curve` unrolls that
same average along the transition index into a smoothed curve

    φ(t) = ( Σ_t' k_σ(t'−t) s_t' ) / ( Σ_t' k_σ(t'−t) ) − c ,   k_σ(u) = e^(−u²/2σ²)

with `s_t = S[q_t, q_{t+1}]` the per-transition kinematic similarity, `c` A1's
§7.2 occupancy-matched null mean, and the Gaussian kernel running over the
**transition index** (never over seconds), at `σ ∈ {3,5}` transitions. Nothing
new is estimated: with a flat kernel `φ(t)` collapses to the constant
`mean_t s_t − c = Φ` (Prop 1), and the run asserts that identity before
plotting. Everything is reused from A1 — the similarity `S` (built from the same
`--fluency-omega`), the run-length-compressed visit sequence `q`, and the cached
null `c` and scalar `Φ` — so the temporal view and the reported scalar can never
disagree. The wall-clock time `T_t` of each transition (derived from the window
geometry and visit dwell lengths) is the plot axis only.

This is deliberately **exploratory**: the reported scalar stays the
transition-averaged `Φ`; `σ` is fixed a priori and never tuned against the group
contrast; the CSV carries no test column; and there is no time-integral average
`(1/T)∫φ` (which equals `Φ + Cov_t(Δ,s)/mean(Δ)`, a duration-weighted quantity
that is *not* `Φ`). Outputs: one panel per recording at
`<outdir>/figures/fluency_curve/{id}.png` — the two `σ` curves against `T_t`,
horizontal lines at `0` (chance) and `Φ`, and a rug of the transitions below the
axis — plus `<outdir>/results/fluency_curve.csv` (`id, label, Φ, n_transitions`).

**Fluency only.** `--skip-raw-kinematics` runs the fluency (and mixing)
analysis and the whole §5–§11 clinical layer, but skips the raw-kinematic block
— WCLR-PP inter-limb coordination (the synchrony construct, and the slow part
of a run) and the per-state velocity profiles. Use it
for a fast fluency turnaround; the `Φ` result, its group contrast, the channel
split and the fluency figures are all unaffected. The WCLR-PP window and
peak-picking are exposed as `--wclr-w` (window frames, default 50 = 2 s),
`--wclr-tau-max` (max lag, default 13), `--wclr-c` (ΔR² cutoff, default 0.25) and
`--wclr-ell-min` (minimum coupled run, default 19) and `--wclr-dtau` (peak-lag
continuity tolerance, default 3 — consecutive windows chain into one coupled run
only while their peak lags stay within this many frames, so lowering it breaks a
run at every lag step and F falls); vary `w`, `tau_max` and `c`
over declared grids for the robustness pass. `--wclr-limb-signal` chooses which
joints define a limb's velocity — `distal` (default: elbow+wrist, knee+ankle,
the specified construct), `end_effector` (wrist/ankle alone) or `limb` (the
whole chain). Read `end_effector` as the robustness check against the default
rather than as an upgrade to it: the limb average buys far less noise
suppression than it appears to (see `LIMB_SIGNALS` in
`a9_wclrpp.py`), and `limb` additionally folds in shoulder and hip, whose
torso-normalised velocities are near-negations of each other across the
midline — ΔR² ignores the sign of a relation, so that reads as coupling in
exactly the homologous pairs that carry the cramped-synchronised signature.
`--stream
delta|pose|auto` selects the frame-attribution convention; `auto` infers it from
the stored `lengths`, since the delta trajectory is exactly one window per
subject shorter than the pose trajectory. Either model may be omitted.

`--state-names FILE` labels the states in the figures, as a JSON list or one
name per line, and must carry exactly `K` of them. Without it the states are
numbered. There is deliberately no fallback that invents names: a name like
"left leg" is a reading of one particular fit, identified by its stream, its
`K`, its seed and its data, so a table keyed on `K` alone attaches one fit's
reading to another's states. Because names are figure text that no statistic
reads, nothing downstream would disagree — the figures would simply be wrong
and the run would say nothing. The run logs which source it used.

### Inference (`a1_stats.py`)

Every clinical endpoint is **one scalar per recording**, contrasted between the
`n1 = 6` abnormal and `n0 = 32` normal recordings, and that contrast is the
whole of the reported inference:

* **Null.** `H0: F_X = F_Y`. Equality of the two distributions makes the labels
  exchangeable, which is what makes the permutation law below the true null law
  of `U`.
* **Statistic.** Pool the 38 values, rank them with mid-ranks for ties, and let
  `R1` be the abnormal group's summed rank. Then
  `U = R1 - n1(n1+1)/2 = ΣΣ [1{X>Y} + ½·1{X=Y}]` and
  `AUC = U / (n1 n0)`: the probability that a random abnormal recording exceeds
  a random normal one, equal to ½ under the null.
* **p-value.** Under the null the abnormal set is a uniform random six-subset,
  so the exact null law of `U` is its distribution over all
  `C(38,6) = 2,760,681` label assignments — enumerated exactly, by dynamic
  programming over the rank multiset, not sampled. The two-sided p is the
  exact-null probability of an AUC at least as far from ½ as the observed one.
* **Interval.** A stratified nonparametric bootstrap: `B = 10,000` times, each
  group is resampled independently, with replacement, at its original size, the
  AUC is recomputed, and the reported interval is the `[2.5, 97.5]` percentile
  pair of those replicates. It holds the 6/32 allocation fixed and cannot leave
  `[0, 1]`.

Endpoints: fluency `Φ`, the Kemeny constant `𝒦`, whole-body synchrony
(`mean F`) and the three FidgetyFind numbers `FF`, `FF_hip`, `FF_dist`, plus
each of the six WCLR-PP pairs and the two similarity channels of `Φ`. Every one
of them goes through exactly the procedure above and nothing else. **Nothing is
corrected for multiplicity**, **no nuisance is partialled out of a contrast**,
and **no endpoint has to clear an admission gate to be reported**.

The nuisances are reported as correlations instead: `correlation_analysis`
gives every endpoint's **Pearson and Spearman** correlation with occupancy
entropy `H = -Σ o_k log o_k`, mean dwell time, and log recording length,
written to `results['correlations']` and drawn as `correlations.png`. The label
enters no fit.

That is the entire inferential layer, and `a1_stats.py` now contains nothing
else. There is no multiplicity correction (no Holm, no Westfall-Young
maximum-statistic), no normal approximation to the AUC, no nuisance adjustment,
no BCa interval on a group median, no leave-one-out sweep, no split-half
reliability, no truncation control, no estimability gate and no pre-specified
gate table. All of those were removed rather than kept and marked.

### FidgetyFind: the literature's detector (`a10_fidgetyfind.py`)

> Morais, R., Le, V., Morgan, C., Spittle, A., Badawi, N., Valentine, J.,
> Hurrion, E. M., Dawson, P. A., Tran, T., Venkatesh, S. (2023). *Robust and
> Interpretable General Movement Assessment Using Fidgety Movement Detection.*
> IEEE JBHI 27(10), 5042–5053. Reference code:
> `github.com/RomeroBarata/fidgetyfind`.

Fidgety movements are *small* and *directionally variable*. FidgetyFind asks one
question of each limb: are its small movements varied in direction, or do they
all point the same way? Varied direction is fidgety movement; a single direction
is its absence. Near 1: the limb wandered in every direction, fidgety movement
is present — the normal pole. Near 0: one direction only, or too few small
movements to judge — absent fidgety movement, which is what the GMA label marks.
**The abnormal group is therefore expected below the normal one, and the
reported AUC below 0.5.**

This module implements the **adapted** construct of METHODS §FidgetyFind, which
is what the thesis specifies and what the code follows. The adaptation exists
because this cohort has no RGB data.

**Per frame.** For a moving joint `c` and its parent `b`, the displacement
`v_c(t) = x_c(t+1) − x_c(t)` gives

```
r(t)     = 100 ‖v_c(t)‖ / ℓ,          ℓ = ‖x_c − x_b‖   (constant: rigid skeleton)
α(t)     = ∠(u(t), v_c(t)) ∈ (−π, π], u(t) = x_c(t+1) − x_b(t+1)
q_b(t)   = 100 ‖x_b(t+1) − x_b(t)‖ / ‖x_Neck − x_MidHip‖
```

Dividing by the limb length removes body size and camera distance; measuring the
angle against the limb removes the infant's pose; `q_b` separates a limb being
transported by its parent from a limb fidgeting on a still one. The published
thresholds are read as **percentages** (hence the explicit 100), and the
published arccosine — whose range is `[0, π]` — is replaced by the **signed**
angle, which is what eight bins over `(−π, π]` are consistent with.

**Per window** of `L` frames (so `L − 1` displacements, the denominator of every
rate), with `B_i = {t : r_min ≤ r(t) ≤ r_max}`:

```
E_c(i) = NaN   if  |{t : g_c(t) ≤ τ_m1}| / (L−1) < τ_m     unassessable
E_c(i) = 0     if  |B_i| / (L−1) < ν                       assessable, nothing found
E_c(i) = Σ(−p log p) / log B   otherwise, over the directions {α(t) : t ∈ B_i}
```

with `(g_c, τ_m1, τ_m) = (r, τ_hip, 0.2)` on the hip chains, `(q_b, τ_hand, 0.3)`
on the hands and `(q_b, τ_foot, 0.1)` on the feet, and `B = 8` equal bins of
`(−π, π]`. The paper prints the distal inequality the other way round, which
would keep only windows dominated by large movements; that is treated as a
misprint and the proximal direction is applied to every chain.

**Per recording**, three levels, mirroring how a rater judges a recording —
fidgety movement counts if any joint on a side shows it, if it recurs through
the recording, and if it is present on both sides. With
`C_R = {R hip, R hand, R foot}` and `C_L` likewise:

```
s_σ(i) = max{ E_c(i) : c ∈ C_σ assessable }      window scorable iff one chain is
S_σ    = Q₉₀({ s_σ(i) })                          ~15 s of fidgety movement in 3 min
FF     = min(S_L, S_R)
```

A recording is scored only when **at least a quarter** of its windows on each
side are scorable; below that FidgetyFind declines and the recording is held as
`NaN` rather than forced to a value. The same three levels restricted to the two
hip chains give `FF_hip` (the part of the published method that needs no
adaptation) and restricted to the four limb chains give `FF_dist`. Those three
numbers are the endpoints; nothing else in the block is tested.

**Calibration.** The published band `[4.5, 8.0]` % of the parent limb per frame
is tuned to raw OpenPose output. These keypoints are smoothed and rigidified
upstream, so per-frame amplitudes are smaller — the published floor 4.5 sits near
this cohort's 85th percentile and almost nothing lands in the band. Every
threshold is rescaled by one factor: pool `r(t)` over all recordings and chains,
take `Q₇₅`, and set

```
ς = Q₇₅ / √(r_min r_max),   (r_min, r_max, τ_hip, τ_hand, τ_foot) ← ς (4.5, 8.0, 10.0, 1.0, 2.5)
```

The first three are fractions of the parent limb and the last two of the trunk,
so a single factor across all five assumes the two scales shifted together. On
the real cohort ς = 0.45, giving the band `[2.04, 3.63]`. Smaller amplitudes then
need a longer window to collect the same count: with the published `L = 50` and
`ν = 0.2` a window needs ten in-band frames, so

```
L = min{ L' ∈ {50, 70, 100, 150} : median over all length-L' windows |B_i(L')| ≥ 10 },  ν = 10/L
```

which gives `L = 100`, `ν = 0.1`; the stride is set to `0.4 L = 40` frames, the
published overlap ratio. Calibration is part of the construct here, not an
option — `FF.calibrate` runs on every call and its factor, percentile and window
grid are recorded in `results.json`.

**What the adaptation is.** The reference scores hands and feet from dense
optical flow over segmented hand/foot pixels, since OpenPose detects neither. We
have no video, so the wrist against the forearm and the ankle against the shank
are scored by the same keypoint-only estimator — the very axis and reference
length the flow path normalises by. The published distal gate is defined on the
distal joint's own displacement over the trunk; here it is the *parent* joint's,
so `τ_hand` and `τ_foot` are the published values applied to a different
quantity. All six chains use 8 bins: the reference's 16 distal bins count
thousands of flow vectors per frame, ours counts one displacement per frame. The
camera-motion gate needs the video and is not implemented — these are
fixed-camera cot recordings. There is **no internal smoothing** (the keypoints
arrive smoothed), **no frame-rate rescaling** (the specified `r(t)` is a plain
ratio of two lengths) and **no confidence gate** (the three branches above are
the whole gate list).

Read [`docs/FIDGETYFIND_FIDELITY.md`](../docs/FIDGETYFIND_FIDELITY.md) before
quoting any of these numbers as "FidgetyFind": it says which are the published
measurement (`FF_hip`), which are our substitution (`FF_dist`) and where the
rigid-skeleton input limits the construct.

`--skip-fidgetyfind` omits the block; `--ff-panels` additionally writes one
timeline panel per recording under `figures/fidgetyfind/`. Outputs:
`fidgetyfind_per_subject.csv` (the three endpoints, the per-side `S_σ` and
scorable fractions, and each chain's assessable fraction) and the three cohort
figures `fidgetyfind_subject`, `fidgetyfind_chains` and `fidgetyfind_windows`.

Outputs: `results.json`, `per_subject.csv`, `similarity_matrix.csv` (the
combined `S`) with `similarity_matrix_magnitude.csv` and
`similarity_matrix_shape.csv` for its two channels, `state_amplitude_profile.csv`,
`state_shape_profile.csv` (per state and free joint: anisotropy `ρ`, principal
axis `θ`, and the `(u₁,u₂)` coordinate), `run.log`, and one figure pair
(PNG + PDF) per panel, including `wclrpp_pairs` (the six per-limb-pair panels)
and `wclrpp_summary` (the per-infant aggregation).

Without the archive, exercise the pipeline on synthetic data:

    python make_synthetic.py --outdir synth
    python run_analysis.py --csv synth/synth_analysis.csv \
        --arhmm synth/synth_arhmm.pkl --hmm synth/synth_hmm.pkl \
        --outdir synth/out --fast

## Findings that changed the implementation

**The §7.2 permutation null is biased upward by its own diagonal.** §7.1
run-length compresses the path, so the observed visit sequence can never place a
state next to itself. A uniform permutation of that same multiset can, and does
— at 16.5% of adjacent pairs in a synthetic check. Every such pair contributes
`S_kk = 1`, the largest entry of `S`, so the null sits above the range the
observed statistic can occupy and `Phi` acquires a negative offset whose size
depends on the subject's occupancy concentration. Since removing the occupancy
confound "by construction" is exactly what §7.3 claims for this null, both
versions are computed: `excess` is §7.2 as written, `excess_offdiag` conditions
the null on adjacent entries differing, and `null_repeat_rate` reports how large
the effect is for each subject. Conclusions should agree across the two.

**§5.2 defines a set union, so overlapping spans must not be double-counted.**
Consecutive delta windows overlap by `l` frames. Concatenating the spans of a
state's windows counts every frame shared by two same-state windows twice, which
over-weights long dwells in the RMS. `state_frame_mask` accumulates a boolean
union instead.

**§3.4 only resolves the AR block convention when one ordering is clearly
better.** The real archive separates cleanly (0.031 against 0.365). A near-tie
means the innovation covariance cannot discriminate, so the run reports
`inconclusive` and fails the check rather than asserting an ordering — asserting
one is precisely the silent corruption §3.4 warns about.

**Exact enumeration is used for the group contrasts.** §10.1 notes the label
space is only `C(38,6) = 2,760,681` assignments. The rank-sum null is enumerated
in full by dynamic programming over the doubled mid-rank lattice, so the
headline p-values carry no Monte Carlo error. Verified against brute-force
enumeration and against `scipy`'s exact Mann-Whitney.

**Degenerate terciles are reported, not silently returned as NaN.** With few
distinct values in `S` a tercile can come out empty; §7.5 then falls back to the
extreme non-empty terciles and sets `degenerate_terciles`.

**Float boundary in the shrinkage gate.** `linspace(0, 0.95, 20)` lands on
`0.8999999999999999`, so a bare `alpha >= 0.9` reported a maximally shrunk fit
as informative. The comparison carries a tolerance.

## Differences from the supplied reference implementation

The reference (`A_analysis_code.zip`) covers A1 and A2 and provides A5/A7
primitives, but as delivered:

* `figs.py` / `figs57.py` load `S.npy`, `amp.npy`, `phi_exc.npz`, `memory.npz`,
  `a5_out.npz`, `a5_markov.npz`, `a7_kemeny.npz`, `a5_pcca.csv` and
  `mfpt_jump.npy` — **no script in the archive writes any of them**, so the
  figure scripts cannot run.
* Figure titles hard-code results (`ARI = 1.000`, `p = 5e-11`, `crispness 0.78`,
  and eleven state names). A stale number would survive any change upstream.
  Here every annotation is computed from the run, and state names come from a
  file the caller passes or not at all (see `--state-names`).
* `run_analysis.py` runs A1 and A2 only; A5 and A7 have no runner.
* Output paths are hard-coded to `/mnt/user-data/outputs/`.
* `choose_alpha` rebuilds the group matrix inside its inner loop, making the
  selection quadratic in the number of subjects for no change in result.
* Most of §10 is absent. That no longer matters: METHODS §Inference specifies
  the exact Mann-Whitney contrast, its bootstrap interval and the correlation
  table as the whole of the reported statistics, and this implementation carries
  exactly those.
