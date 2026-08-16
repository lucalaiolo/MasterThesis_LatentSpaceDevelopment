# RVI-38: fluency, mixing and raw-kinematic analyses

Implementation of `METHODS_3.md` — fluency (A1) and mixing structure (A7) on the
RVI-38 cohort, plus the raw-kinematic constructs (WCLR-PP inter-limb
coordination for cramped-synchronised movements, and the per-state velocity
profiles), with the §10 inference layer and the §11 pre-specified gates.

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
| `a1_stats.py` | exact/permutation Mann-Whitney, Holm, maxT, split-half, ICC, BCa, Freedman-Lane, Mantel, power |
| `a57_graph.py` | jump chain, fundamental matrix, MFPT, Kemeny, shrinkage, block bootstrap |
| `a8_movement.py` | raw kinematics: per-state velocity profiles |
| `a9_wclrpp.py` | WCLR-PP inter-limb coordination: vector-valued conditional limb regression, peak-picking, per-pair F/R2, label-permutation and circular-shift surrogate inference |
| `figures.py` | figure panels, every annotation computed from the run |
| `run_analysis.py` | end-to-end runner |
| `test_methods.py` | 59 checks with a definite right answer (§12.4 style) |
| `make_synthetic.py` | synthetic cohort with planted structure, for smoke tests |

## Run

    pip install numpy scipy pandas joblib scikit-learn matplotlib
    python run_analysis.py --csv rvi38_analysis.csv \
        --arhmm arhmm_rvi38_stream_delta.pkl \
        --hmm hmm_rvi38_stream_delta.pkl \
        --labels RVI_38_labels.mat --outdir rvi38_out

`--fast` cuts every resampling count ~20x for a smoke run.

### Direction-aware fluency (`DIRECTION_AWARE_KINEMATIC_SIMILARITY.md`)

The similarity `S` behind the fluency measure `Φ` is built from a
**direction-aware** state signature. Each free joint's motion in a state is
summarised not by a single RMS speed but by the pooled second-moment matrix
`M[k,j] = mean(v vᵀ)`, whose trace is the same `a[k,j]²` as before and whose
trace-normalised part is the double-angle axis coordinate
`u = (ρ cos 2θ, ρ sin 2θ)` — anisotropy `ρ = ‖u‖ ∈ [0,1]` and principal axis
`θ = ∠u / 2` (mod π). `S` then combines a **magnitude** channel (the log-RMS-speed
correlation, unchanged) with a **shape** channel (the residual-axis cosine). This
is local to the signature: no model is refitted, the confirmatory family stays
`{Φ, 𝒦}`, and the maxT correction is untouched — one statistic enters and one
leaves.

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
split, split-half reliability, the gates and the fluency figures are all
unaffected. The WCLR-PP window and
peak-picking are exposed as `--wclr-w` (window frames, default 50 = 2 s),
`--wclr-tau-max` (max lag, default 13), `--wclr-c` (ΔR² cutoff, default 0.25) and
`--wclr-ell-min` (minimum coupled run, default 19) and `--wclr-dtau` (peak-lag
continuity tolerance, default 1 — consecutive windows chain into one coupled run
only while their peak lags stay within this many frames, so raising it tolerates
more lag wander and F rises); vary `w`, `tau_max` and `c`
over declared grids for the robustness pass. `--wclr-limb-signal` chooses which
joints define a limb's velocity — `end_effector` (default: wrist/ankle alone,
the specified construct), `distal` (elbow+wrist, knee+ankle) or `limb` (the
whole chain). Use the averages as a robustness check, not as an upgrade: they
buy far less noise suppression than they appear to (see `LIMB_SIGNALS` in
`a9_wclrpp.py`), and `limb` additionally folds in shoulder and hip, whose
torso-normalised velocities are near-negations of each other across the
midline — ΔR² ignores the sign of a relation, so that reads as coupling in
exactly the homologous pairs that carry the cramped-synchronised signature.
`--stream
delta|pose|auto` selects the frame-attribution convention; `auto` infers it from
the stored `lengths`, since the delta trajectory is exactly one window per
subject shorter than the pose trajectory. Either model may be omitted.

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
  the gate PASS/FAIL list, and eleven state names). A stale number would survive
  any change upstream. Here every annotation is computed from the run.
* `run_analysis.py` runs A1 and A2 only; A5 and A7 have no runner.
* Output paths are hard-coded to `/mnt/user-data/outputs/`.
* `choose_alpha` rebuilds the group matrix inside its inner loop, making the
  selection quadratic in the number of subjects for no change in result.
* Most of §10 and all of §11 are absent: split-half/Spearman-Brown, ICC, BCa,
  maxT, Freedman-Lane, Mantel, AMI, minimum detectable effect, the duration and
  truncation controls, and the gates table.
