# RVI-38: fluency, mixing and raw-kinematic analyses

Implementation of `METHODS_3.md` — fluency (A1) and mixing structure (A7) on the
RVI-38 cohort, plus the raw-kinematic constructs (limb co-movement for
cramped-synchronised movements, and the fidgety band ratio), with the §10
inference layer and the §11 pre-specified gates.

The metastable decomposition (PCCA+, implied timescales, the kinematic
dendrogram and their agreement statistics) has been removed.

## Module map (§12.2)

| Module | Responsibility |
|---|---|
| `load_models.py` | lazy stub package for `ssm`/`autograd`; recovers arrays from joblib dumps; normalises both dict layouts to one schema |
| `build_pose.py` | long CSV to per-video `(F,15,2)`; verifies frame contiguity and the constant-joint property |
| `a1_core.py` | window-to-frame geometry, state profiles, double-centred similarity, fluency estimators |
| `a1_stats.py` | exact/permutation Mann-Whitney, Holm, maxT, split-half, ICC, BCa, Freedman-Lane, Mantel, power |
| `a57_graph.py` | jump chain, fundamental matrix, MFPT, Kemeny, shrinkage, block bootstrap |
| `a8_movement.py` | raw kinematics: limb co-movement (cramped-synchronised), fidgety band ratio, per-state velocity profiles |
| `figures.py` | figure panels, every annotation computed from the run |
| `run_analysis.py` | end-to-end runner |
| `test_methods.py` | 55 checks with a definite right answer (§12.4 style) |
| `make_synthetic.py` | synthetic cohort with planted structure, for smoke tests |

## Run

    pip install numpy scipy pandas joblib scikit-learn matplotlib
    python run_analysis.py --csv rvi38_analysis.csv \
        --arhmm arhmm_rvi38_stream_delta.pkl \
        --hmm hmm_rvi38_stream_delta.pkl \
        --labels RVI_38_labels.mat --outdir rvi38_out

`--fast` cuts every resampling count ~20x for a smoke run. `--stream
delta|pose|auto` selects the frame-attribution convention; `auto` infers it from
the stored `lengths`, since the delta trajectory is exactly one window per
subject shorter than the pose trajectory. Either model may be omitted.

Outputs: `results.json`, `per_subject.csv`, `similarity_matrix.csv`,
`state_amplitude_profile.csv`, `run.log`, and four figure pairs (PNG + PDF).

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
