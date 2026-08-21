# FidgetyFind in this repository: what is the published construct, and what is not

> **One-sentence answer.** `rvi38_methods/a10_fidgetyfind.py` implements the
> **adapted** FidgetyFind of METHODS §FidgetyFind, not the published method
> unaltered. The per-frame measurement, the small-movement band, the direction
> histogram and its normalised entropy are the published ones; the hands-and-feet
> path had to be re-derived because it needs video this cohort does not have; two
> readings of the paper are corrected (percentages, and a signed angle in place of
> an arccosine); the amplitude ladder is rescaled to this cohort by a single
> factor; and the reduction of per-window entropies to one number per recording
> follows the paper's aggregation, which the released code does not contain.
>
> If you quote a number from this pipeline, the safe phrasing is in
> [§8](#8-how-to-describe-this-in-the-thesis).

This document exists because "we ran FidgetyFind" is a claim about someone else's
method, and the only honest way to make it is to say exactly which parts are
theirs. Everything below is checkable: each claim names the reference file and
function, ours, and — where it is a numerical claim — the test in
`rvi38_methods/test_methods.py` that pins it ([§9](#9-how-to-check-every-claim-here)).

---

## 1. Provenance: what "official" means here

**The paper.**

> Morais, R., Le, V., Morgan, C., Spittle, A., Badawi, N., Valentine, J.,
> Hurrion, E. M., Dawson, P. A., Tran, T., & Venkatesh, S. (2023). *Robust and
> Interpretable General Movement Assessment Using Fidgety Movement Detection.*
> IEEE Journal of Biomedical and Health Informatics **27**(10), 5042–5053.
> DOI [10.1109/JBHI.2023.3298708](https://ieeexplore.ieee.org/document/10195984/)

**The code.** `https://github.com/RomeroBarata/fidgetyfind`, branch `main`,
commit `84e796f8f1a2fdce04a4d68a886c77b393631514`, fetched 2026-08-19. The files
consulted, with the first 16 hex digits of their SHA-256 as fetched:

| file | sha256 (16) | what it contributes |
|---|---|---|
| `fidgetyfind/proximal.py` | `181308707ae775b8` | the hips path: the estimator we reproduce |
| `fidgetyfind/distal.py` | `4ee9b1a24bfcf266` | the hands/feet path (optical flow) |
| `fidgetyfind/geometry.py` | `b6529052ff953b79` | `get_angle_between`, `get_reference_length` |
| `fidgetyfind/skeleton_smoothing.py` | `4cbffaed35da43fe` | the 5-frame score-weighted Gaussian |
| `fidgetyfind/skin_and_flow.py` | `2fce12011024e57b` | skin segmentation, Farnebäck flow, camera gate |
| `fidgetyfind/constants.py` | `895182e880f95220` | joint indices and score thresholds |
| `scripts/fidgetyfind-single-video.py` | `9f3cec46bb129a36` | **every numeric threshold**, as called |

**A caveat you should not skip.** The IEEE article itself is paywalled and was
not read in full. Where the released code and the thesis specification differ,
this implementation follows **the thesis specification** (METHODS §FidgetyFind),
which is the construct the study reports; the released code is used as the
reference for what the published estimator computes per frame and per window.

**Our implementation.** `rvi38_methods/a10_fidgetyfind.py`, with the analysis
wiring in `rvi38_methods/run_analysis.py::fidgetyfind` and figures in
`rvi38_methods/figures.py`.

---

## 2. The construct as implemented

FidgetyFind asks one question of each limb: are its small movements varied in
direction, or do they all point the same way? Varied direction is fidgety
movement; a single direction is its absence.

### 2.1 Per frame

For a moving joint `c` with parent joint `b` (knee against hip, wrist against
elbow, ankle against knee), every consecutive frame pair contributes the
displacement `v_c(t) = x_c(t+1) − x_c(t)`, and from it

```
r(t)   = 100 ‖v_c(t)‖ / ℓ,              ℓ = ‖x_c − x_b‖
α(t)   = ∠(u(t), v_c(t)) ∈ (−π, π],     u(t) = x_c(t+1) − x_b(t+1)
q_b(t) = 100 ‖x_b(t+1) − x_b(t)‖ / ‖x_Neck − x_MidHip‖
```

Dividing by the limb length removes body size and camera distance. Measuring the
angle against the limb removes the infant's pose. `q_b` — the parent joint's own
displacement, as a fraction of the trunk — separates a limb being *transported*
by its parent from a limb fidgeting on a still one.

`ℓ` is written without a time index in the specification and is justified there
by the rigid skeleton. The code reads it as `‖u(t)‖`, the length of the very axis
the angle is measured against; on this cohort that *is* the constant `ℓ`
([§6](#6-what-the-rigid-skeleton-costs-this-construct)), so no frame is
privileged. On non-rigid input the choice would matter, and it is stated here
rather than hidden.

Two readings of the paper are corrected, as METHODS records:

* the published thresholds take values between 1 and 10 against a ratio of two
  lengths, so they are read as **percentages** and the factor 100 is carried
  explicitly;
* the published angle is an **arccosine**, whose range is `[0, π]`, yet it is
  binned over `(−π, π]`. The **signed** angle is taken instead, which is what
  eight bins over `(−π, π]` are consistent with.

### 2.2 Per window

Over a window of `L` frames — which carries `L − 1` displacements, and `L − 1` is
the denominator of every rate — the in-band frames are
`B_i = {t : r_min ≤ r(t) ≤ r_max}`, and

```
E_c(i) = NaN                    if |{t : g_c(t) ≤ τ_m1}| / (L−1) < τ_m
E_c(i) = 0                      if |B_i| / (L−1) < ν
E_c(i) = Σ(−p_m log p_m)/log B  otherwise
```

over `B = 8` equal bins of `(−π, π]`, with the amplitude gate

| chain class | `g_c` | `τ_m1` | `τ_m` |
|---|---|---|---|
| hip (knee against hip) | `r` | `τ_hip` | 0.2 |
| hand (wrist against elbow) | `q_b` | `τ_hand` | 0.3 |
| foot (ankle against knee) | `q_b` | `τ_foot` | 0.1 |

The first branch marks the chain **unassessable** when too few frames are small
in amplitude, fidgety movement being small in scale. The second branch scores
**zero** when too few frames fall in the band — that zero is a *measurement*, not
a failure, and the distinction from `NaN` is load-bearing. Otherwise `E_c(i)` is
the normalised entropy of the in-band directions: near one when they pointed
everywhere, near zero when they pointed one way.

The paper prints the distal inequality in the opposite direction to the proximal
one, which would keep only windows dominated by *large* movements. That is
treated as a misprint and the proximal direction is applied to every chain.

### 2.3 Per recording

Three levels, mirroring how a rater judges a recording: fidgety movement counts
if any joint on a side shows it, if it recurs through the recording rather than
appearing only briefly, and if it is present on both sides. Group the six chains
by body side, `C_R = {R hip, R hand, R foot}` and `C_L` likewise, and

```
s_σ(i) = max{ E_c(i) : c ∈ C_σ, E_c(i) assessable }   window scorable iff one is
S_σ    = Q₉₀({ s_σ(i) : s_σ(i) defined })
FF     = min(S_L, S_R)
```

The ninetieth percentile encodes the clinical benchmark of roughly fifteen
seconds of fidgety movement in a three-minute recording. A recording is scored
only when **at least a quarter** of its windows on each side are scorable; below
that FidgetyFind declines to score it, and the recording is held as `NaN` rather
than forced to a value — it drops out of its contrast instead of entering it with
an invented number. The same three levels restricted to the two hip chains give
`FF_hip`, and restricted to the four limb chains give `FF_dist`.

High `FF` means fidgety movement is present, the normal pole, so the at-risk
group is expected **below** the normal one and the reported AUC below 0.5.
Nothing here is one-sided; the tests are two-sided as everywhere else.

### 2.4 Calibration

The published band `[4.5, 8.0]` % of the parent limb per frame is tuned to raw
OpenPose output. These keypoints are smoothed and rigidified upstream, so
per-frame amplitudes are smaller: the published floor 4.5 sits near this cohort's
**85th percentile**, and almost nothing lands in the band
([§5](#5-what-the-published-band-does-on-this-cohort)). Every threshold is
rescaled by one factor. Pool `r(t)` over all recordings and chains, let `Q₇₅` be
its 75th percentile, and set

```
ς = Q₇₅ / √(r_min r_max)
(r_min, r_max, τ_hip, τ_hand, τ_foot) ← ς (4.5, 8.0, 10.0, 1.0, 2.5)
```

The first three values are fractions of the parent limb and the last two are
fractions of the trunk, so a single factor across all five assumes the two scales
shifted together. On the real cohort `ς = 0.45`, giving the band `[2.04, 3.63]`.

Smaller amplitudes then need a longer window to collect the same count. The
published window is `L = 50` with stride 20 and an in-band floor `ν = 0.2`, so a
window needs at least ten in-band frames. The shortest `L` whose **median** window
reaches ten is chosen:

```
L = min{ L' ∈ {50, 70, 100, 150} : median over all length-L' windows |B_i(L')| ≥ 10 }
ν = 10 / L
```

which gives `L = 100`, `ν = 0.1`; the stride is `0.4 L = 40` frames, preserving
the published overlap ratio 20/50. `I(L')` runs over *all* length-`L'` window
positions, so the counts are taken at stride 1 by a cumulative sum, not at the
reporting stride.

Calibration is part of the construct here, not an option: `FF.calibrate` runs on
every call, and `ς`, the percentile, the window grid and the per-`L` medians are
recorded in `results.json`. **This is a re-calibration and must be labelled as
one** — see [§8](#8-how-to-describe-this-in-the-thesis).

---

## 3. Verdict table

Read the verdict column as: **Identical** — reproduces the released code;
**Corrected** — the specification reads the paper differently from the released
code and the specification wins; **Forced** — the released code cannot run on our
data and this is the substitute; **Chosen** — we could have matched the reference
and deliberately did not; **Specified** — from METHODS §FidgetyFind, with no
counterpart in the released code; **Upstream** — a property of this project's
data, not a decision about the construct.

| # | Element | Verdict | Where |
|---|---|---|---|
| 1 | Amplitude `r(t)` (÷ parent limb, ×100) | Identical | [§2.1](#21-per-frame) |
| 2 | Direction against the limb axis | Identical in reference frame; **Corrected** to a signed angle | [§2.1](#21-per-frame) |
| 3 | Thresholds read as percentages | Corrected | [§2.1](#21-per-frame) |
| 4 | Small-movement band `[r_min, r_max]`, rate rule → `0.0` not `NaN` | Identical in form | [§2.2](#22-per-window) |
| 5 | Histogram over `(−π, π]`, entropy ÷ `log B` | Identical (no epsilon; `0 log 0 = 0`) | [§2.2](#22-per-window) |
| 6 | Amplitude gate direction on the distal chains | Corrected (paper's inequality read as a misprint) | [§2.2](#22-per-window) |
| 7 | Distal gate on the **parent** joint's displacement | Specified — the published gate is on the distal joint's own | [§2.2](#22-per-window) |
| 8 | Joint indices | Identical (BODY-15 ≡ BODY-25 for 0–14) | [§4.4](#44-the-joint-indices) |
| 9 | **Hands and feet: dense optical flow over segmented skin** | **Forced** — no video | [§4.1](#41-the-handsfeet-path-the-big-one) |
| 10 | **Camera-motion gate on background flow** | **Forced** — omitted, no video | [§4.2](#42-the-camera-motion-gate-omitted) |
| 11 | Pose detector | Forced — not the authors' fine-tuned OpenPose | [§4.3](#43-the-pose-detector) |
| 12 | Histogram bins on hands/feet (16 → 8) | Chosen | [§4.5](#45-bins-on-the-distal-chains-16--8) |
| 13 | Distal angles limb-relative, not image-frame | Chosen | [§4.6](#46-the-angular-reference-frame-on-the-distal-chains) |
| 14 | Keypoint smoothing inside the construct | **Not implemented** — the keypoints arrive smoothed | [§4.7](#47-what-is-deliberately-absent) |
| 15 | Frame-rate rescaling by `fps / 30` | **Not implemented** — the specified `r(t)` is a plain ratio | [§4.7](#47-what-is-deliberately-absent) |
| 16 | Detection-confidence gate | **Not implemented** — the three branches are the whole gate list | [§4.7](#47-what-is-deliberately-absent) |
| 17 | Amplitude ladder rescaled by `ς`, window length from the in-band count | Specified | [§2.4](#24-calibration) |
| 18 | Three-level reduction, `Q₉₀`, `min` over sides, quarter-of-windows rule | Specified — the released code has no reduction | [§2.3](#23-per-recording) |
| 19 | Torso-normalised coordinates | Upstream | [§6](#6-what-the-rigid-skeleton-costs-this-construct) |
| 20 | Rigid-skeleton reconstruction | Upstream, and the most consequential one | [§6](#6-what-the-rigid-skeleton-costs-this-construct) |
| 21 | Gaps already interpolated before we see them | Upstream | [§4.7](#47-what-is-deliberately-absent) |
| 22 | 25 fps recordings | Upstream | [§4.7](#47-what-is-deliberately-absent) |

---

## 4. Differences, and why

### 4.1 The hands/feet path (the big one)

For each frame and each of the four distal limbs,
`skin_and_flow.py::get_flow_features`:

1. computes dense **Farnebäck optical flow** between consecutive video frames
   (`cv.calcOpticalFlowFarneback`, `pyr_scale 0.5`, 3 levels, `winsize 15`);
2. builds a quadrilateral **anatomical mask** for the hand or foot, placed by
   rotating the forearm/shank direction by ±120° and sized as a fraction of the
   trunk length (0.35 for hands, 0.6 for feet);
3. intersects it with a **skin mask** obtained by sampling the HSV colour in a
   small square at the wrist/ankle and thresholding `± [7, 60, 200]` around it,
   eroded and dilated with a 3×3 ellipse, then keeps the connected component
   containing the joint;
4. drops flow vectors indistinguishable from the **background flow vector**
   (computed outside the skeleton's bounding box), keeping pixels whose flow
   differs by more than `0.2 / 100 · trunk length`;
5. histograms the **image-frame orientations** of the surviving pixel flows whose
   magnitude exceeds 8 % of the parent limb's length, into 16 bins, accumulating
   over the window before taking the entropy.

Every one of steps 1–4 requires the video. This project's analysis consumes
`rvi38_analysis.csv`, a long-format keypoint table; there is no video in the
pipeline at all. Reproducing this path is not a matter of effort — the input does
not exist.

**What we do instead.** The wrist against the forearm and the ankle against the
shank are scored by the *same* keypoint-only estimator as the hips: exactly the
axis and reference length the flow path normalises by, and the reference's own
proximal path is that estimator at the knee.

**What it costs.** The flow path measures the hand or foot *itself* — thousands
of pixel flow vectors inside a segmented region, which is what makes 16 bins
sensible there. Ours measures the wrist or ankle **keypoint**, one displacement
per frame. A wrist that stays put while the fingers fidget scores zero here and
would not there. `FF_dist` is therefore **our substitution, not the published
distal measurement**, and it must be named as such.

The distal transport gate keeps the reference's intent — a window is voided when
the parent joint is carrying the limb — but METHODS defines it on the **parent**
joint's displacement over the trunk, where the published gate is on the distal
joint's own. `τ_hand = 1.0` and `τ_foot = 2.5` are therefore the published values
applied to a different quantity.

### 4.2 The camera-motion gate (omitted)

The reference rejects windows in which the *background* optical flow is large,
which needs the video. These are fixed-camera cot recordings, so the gate has
little to reject; it is nonetheless a gate the published method has and this one
does not.

### 4.3 The pose detector

The authors fine-tuned OpenPose on infant data. This cohort's keypoints come from
a different upstream pipeline. Every threshold in FidgetyFind is a calibration
against a detector's noise floor as much as against infant movement, which is a
large part of why [§2.4](#24-calibration) is needed at all.

### 4.4 The joint indices

BODY-15 and BODY-25 agree on indices 0–14, which is every joint this construct
touches. No adaptation.

### 4.5 Bins on the distal chains: 16 → 8

The reference uses 16 bins for the distal histograms because each frame
contributes thousands of flow vectors. Ours contributes one displacement per
frame, so all six chains use the proximal setting of 8 bins; 16 bins over the
in-band frames of one window would read sparsity as order.

### 4.6 The angular reference frame on the distal chains

The reference histograms distal flow orientations in the **image frame**. Ours
measures every chain's angle against its own limb axis, as the proximal path
does, which keeps the six chains on one convention and keeps the measure free of
how the infant is lying.

### 4.7 What is deliberately absent

Three things a reader familiar with the released code will look for and not find,
because METHODS §FidgetyFind does not specify them:

* **No smoothing inside the construct.** The reference applies a score-weighted
  5-frame Gaussian to the keypoints. Here the keypoints arrive already smoothed
  and rigidified from the upstream pipeline ([§6](#6-what-the-rigid-skeleton-costs-this-construct)),
  and the specification takes them as given. Smoothing them again would be a
  second pass, not a reproduction of the first.
* **No frame-rate rescaling.** The reference multiplies every magnitude by
  `fps / 30`, since the published thresholds are calibrated at 30 fps. The
  specified `r(t)` is a plain ratio of two lengths with no such factor, and the
  amplitude ladder is calibrated to this cohort's own distribution anyway
  ([§2.4](#24-calibration)), which absorbs a constant scale.
* **No detection-confidence gate.** The reference gates on OpenPose's per-joint
  score. This table carries a binary `observed` flag instead (interior gaps are
  linearly interpolated upstream, so the coordinates are always present), and the
  specification's three branches are the whole gate list. Nothing in
  `a10_fidgetyfind.py` reads the flag.

A frame whose coordinates are not finite yields `NaN` in `r`, `α` and `q`, and a
`NaN` satisfies no inequality: it is neither small in amplitude nor in band, so
it can only make a window *less* likely to be scored, never more.

---

## 5. What the published band does on this cohort

Everything above describes the construct as built. This section reports what the
**published, uncalibrated** band measures when pointed at the real cohort, which
is why [§2.4](#24-calibration) exists.

38 recordings, 145 721 frames, 25 fps. Per-frame amplitudes pooled over frames
(percent of the parent limb per frame):

| chain | p25 | median | p75 | p90 | share in `[4.5, 8.0]` |
|---|---|---|---|---|---|
| R hip | 0.48 | 1.13 | 2.66 | 5.53 | 8.2 % |
| L hip | 0.48 | 1.09 | 2.44 | 4.90 | 7.6 % |
| R hand | 0.64 | 1.41 | 3.12 | 6.06 | 9.9 % |
| L hand | 0.60 | 1.32 | 2.97 | 5.80 | 9.5 % |
| R foot | 0.56 | 1.20 | 2.60 | 5.18 | 8.0 % |
| L foot | 0.56 | 1.18 | 2.47 | 4.72 | 7.6 % |

`4.5` sits at the **84th–89th percentile** of this cohort's own amplitude
distribution: the band's *floor* is what most of the data fails to reach. The
rate rule needs 20 % of a window's frames inside `[4.5, 8.0]` and the cohort
delivers 2–6 %, so at the published band nearly every window takes the legitimate
score `0.0` and nearly every recording reduces to `0.0`.

**Verified against the authors' released code.** To rule out a re-implementation
bug, the authors' own `fidgetyfind/proximal.py` (commit `84e796f8…`) was run on
these keypoints with the published parameters. It agrees with the hip path
window-by-window to within `3.8 × 10⁻⁴` (an epsilon-convention difference) and
returns the same all-zero result. The zeros are FidgetyFind's own behaviour on
this data, not ours.

**Why the amplitudes sit low.** The upstream pipeline applies a temporal filter
and a rigid-skeleton fit, both of which remove exactly the high-frequency,
per-frame content the band measures — FidgetyFind's `[4.5, 8.0]` was calibrated
on raw, unsmoothed OpenPose output. (The reference's *own* 5-tap kernel was not
already applied: it has exact spectral nulls at 5.830 Hz and 10.167 Hz at 25 fps,
and Welch spectra over the 1 023 long continuously-observed segments put each
null bin at 0.91 and 1.10 relative to its shoulders — no notch — while applying
the kernel once drives them to 0.068 and 0.053. Whatever filter was used
upstream, it was not this one.)

This is what [§2.4](#24-calibration) rescales, and it is why the reported numbers
are a re-calibration rather than the published measurement.

---

## 6. What the rigid skeleton costs this construct

Every bone length in `rvi38_analysis.csv` is **constant to float32 precision
across every frame, every recording and every subject**, and bilaterally
symmetric:

| bone | length (torso units) | sd over all 145 721 frames |
|---|---|---|
| RHip→RKnee, LHip→LKnee | 0.45622 | 4.1 × 10⁻⁷ |
| RElbow→RWrist, LElbow→LWrist | 0.38926 | 4.1 × 10⁻⁷ |
| RKnee→RAnkle, LKnee→LAnkle | 0.57672 | 4.1 × 10⁻⁷ |
| Neck→MidHip | 1.00000 | 0 |

A per-frame similarity normalisation cannot manufacture that — it scales all
bones by one factor, so unequal raw bones would stay unequal. The table is a
**rigid-skeleton reconstruction on a single canonical body**, not raw detections.
This is what makes `ℓ` genuinely constant ([§2.1](#21-per-frame)), and it has two
consequences for this construct specifically.

**The limb normalisation is inert.** `‖x_c − x_b‖` is what makes FidgetyFind free
of body size and camera distance. Here it is a compile-time constant per chain.
The measure still has the right dimensions, but it carries no per-subject
adaptation and cannot reflect how long this infant's thigh actually is.

**One of the two channels that fill the direction histogram is identically
zero.** Write the moving joint's displacement as `v = Δx_c = Δx_b + Δu`, where
`u = x_c − x_b` is the bone. Constant `‖u‖` constrains **only** `Δu`:
`‖u+Δu‖ = ‖u‖` forces `u_{t+1} · Δu = ‖Δu‖²/2`, i.e. `Δu` is perpendicular to the
limb axis up to exactly `Δθ/2`. Measured here, `∠(u_{t+1}, Δu)` is 89.59°–89.78°
on every chain.

It does **not** follow that `v` is perpendicular to the limb, because `Δx_b` is
unconstrained. Whether the histogram collapses depends on how much the parent
translates relative to how much the bone turns, and that varies by chain — the
column to read is how far the *measured* displacement departs from perpendicular:

| chain | median ‖Δx_b‖ / ‖v‖ | `Δu`: dev. from ⊥ | `v`: dev. from ⊥ | in-band directions in the four bins straddling ±90° |
|---|---|---|---|---|
| hips (parent = RHip/LHip) | 0.29–0.30 | 0.4° | **7.6°** | 99.7 % |
| hands (parent = elbow) | 0.57–0.60 | 0.4° | **19–22°** | 88–90 % |
| feet (parent = knee) | 0.73–0.75 | 0.2° | **46°** | 41.6 % |

So the concentration at ±90° is not a geometric necessity — it is what happens
where the parent is nearly still. At the hips the parent is almost pinned (MidHip
is the normalisation's origin and MidHip→RHip is rigid), so the knee's
displacement is nearly pure thigh rotation and the bins around 0° and 180° —
motion *along* the limb — hold 0.2–0.3 %. Pooled hip entropy is 0.64, not 1.0,
before any biology enters. The feet are barely affected.

What rigidity removes is the **radial** component of the child's motion relative
to its parent. In real detector output that component is populated by keypoint
noise along the bone and by genuine out-of-plane limb motion, which a 2D
projection registers as foreshortening. Both fill the 0°/180° bins in the
authors' pixel data. Here the 2D bone length is constant to float32, so neither
is representable at all. That matters for this construct specifically, because
fidgety movements are small and three-dimensional. **On `FF_hip` — the chains
that are the unadapted published path — the dynamic range of the direction
histogram is roughly halved by a property of the input format.**

This is the most consequential fidelity issue in this document. It is not fixable
by re-tuning a threshold.

**The real fix, if the data exists.** Every problem in §5 and §6 comes from the
input representation, not from the code: what is needed is the keypoint table
**before** torso normalisation and **before** the rigid-skeleton fit, in pixels,
with the detector's per-joint confidence. `a10_fidgetyfind.py` consumes that with
no change — nothing in it assumes normalised input. If that table exists anywhere
in the upstream pipeline, this is where to spend the effort.

---

## 7. What the analysis does with the three numbers

`run_analysis.py::fidgetyfind` computes `FF`, `FF_hip` and `FF_dist`, one scalar
per recording each, and puts each through the contrast of METHODS §Inference and
nothing else: the exact Mann-Whitney U over all `C(38,6) = 2,760,681` label
assignments, the AUC as the effect size, and a stratified percentile bootstrap as
the interval. A recording the quarter-of-windows rule declined to score is `NaN`
and drops out of its contrast; the printed line and `summary.md` say how many
recordings entered.

Each of the three also appears in the correlation table against occupancy
entropy, mean dwell and log recording length. There is no per-chain endpoint, no
family-wise correction, no agreement statistic against the model-based
constructs, and no admission gate. Per-chain assessable fractions are reported as
descriptives — they are the honest limit on the construct and what the
quarter-of-windows rule reads — but nothing is tested on them.

Outputs: `fidgetyfind_per_subject.csv` (the three endpoints, each side's `S_σ`
and scorable fraction, each chain's assessable fraction) and the figures
`fidgetyfind_subject`, `fidgetyfind_chains`, `fidgetyfind_windows`, plus one
timeline panel per recording under `figures/fidgetyfind/` with `--ff-panels`.

---

## 8. How to describe this in the thesis

Wording that is defensible:

> Fidgety movement was additionally quantified with an adaptation of
> **FidgetyFind** (Morais et al., 2023). The hip chains reproduce the published
> skeleton-based estimator; because the published hand and foot path requires
> dense optical flow over segmented video, which this keypoint-only cohort does
> not provide, those chains were scored with the same estimator applied at the
> wrist and ankle, and the video-based camera-motion gate was omitted. The
> published amplitude thresholds are calibrated to raw detector output and do not
> transfer to these smoothed, rigidified keypoints, so all five were rescaled by
> a single factor fixed from the cohort's own amplitude distribution, and the
> window length was set from the in-band frame count that rescaling produces.

Wording to avoid:

* "We ran FidgetyFind" *without qualification* — four of six chains are adapted
  and every amplitude threshold is re-calibrated.
* "FidgetyFind detected …" for an `FF_dist` result — say "our keypoint-only
  adaptation of the FidgetyFind distal path".
* Quoting any of the three numbers as **the published measurement**. At the
  published band the construct does not fire on this cohort ([§5](#5-what-the-published-band-does-on-this-cohort));
  every setting that makes it fire is a re-calibration. `FF_hip` is the closest
  thing to an unadapted number, and even it is measured through a rigid skeleton
  that halves the histogram's dynamic range ([§6](#6-what-the-rigid-skeleton-costs-this-construct)).

If a reviewer asks "is this FidgetyFind?", the answer is: the measurement is, at
the hips exactly in form and at the wrists/ankles by a documented substitution;
the thresholds are re-calibrated to this cohort; the reduction follows the
paper's aggregation, which the released code does not implement.

Two further cautions:

* **`ς` and the window grid are fixed before the labels are seen**, and they must
  stay that way. Choosing the band after seeing a contrast, or dropping chains
  that look bad, invalidates the p-value the pipeline prints — which is
  uncorrected for any such search.
* The per-window entropies are a defensible primary object in their own right,
  and the released code stops there. `results['fidgetyfind']['window_entropy']`
  carries them per recording.

---

## 9. How to check every claim here

```bash
cd rvi38_methods
python test_methods.py          # 136 checks; the FidgetyFind ones are below
```

| claim | check |
|---|---|
| `r` is `100‖v_c‖/ℓ`; `α` is the **signed** angle in `(−π, π]`; `q` normalises by the trunk | `test_fidgetyfind_features` |
| no threshold and no feature carries a frame-rate factor ([§4.7](#47-what-is-deliberately-absent)) | `test_fidgetyfind_features` |
| entropy is 0 / `log2/log8` / 1 for one / two / uniform directions | `test_fidgetyfind_entropy` |
| the bins are half-open on the left: `−π` and `+π` share one bin; out-of-range angles fold in | `test_fidgetyfind_entropy` |
| rotation, scale and translation invariance ([§2.1](#21-per-frame)) | `test_fidgetyfind_invariance` |
| branch 1 (too few small frames → `NaN`), branch 2 (under `ν` → exactly `0.0`), branch 3 (spread over eight bins → 1) | `test_fidgetyfind_windows` |
| the rates divide by `L − 1` | `test_fidgetyfind_windows` |
| `s_σ` is the per-side max; `S_σ` is `Q₉₀`; `FF` is the smaller side | `test_fidgetyfind_reduction` |
| the quarter-of-windows rule declines a recording, and `≥` is the comparison | `test_fidgetyfind_reduction` |
| `FF_hip` / `FF_dist` restrict the same three levels to fewer chains | `test_fidgetyfind_reduction` |
| `ς = Q₇₅/√(r_min r_max)` scales all five thresholds by one factor | `test_fidgetyfind_calibration` |
| `L` is the shortest grid window reaching ten in-band frames; `ν L = 10`; stride `= 0.4 L` | `test_fidgetyfind_calibration` |
| the published band is all-zero on a small-amplitude cohort and the calibrated one is not | `test_fidgetyfind_calibration` |
| a planted fidgety signal is recovered with the right sign (AUC < 0.5) | `test_fidgetyfind_planted_cohort` |

The claims of §5 and §6, which need the real cohort, are reproduced by the
snippet below, run from `rvi38_methods/`.

```python
import numpy as np, build_pose, a10_fidgetyfind as FF
from scipy.signal import welch

vids, pose, obs, _ = build_pose.build("rvi38_analysis.csv", out=None)
P = [pose[v] for v in vids]

# §5 — the amplitude distribution, and where the published floor sits in it
M = np.concatenate([FF.motion_features(x)["r"] for x in P])
print("median amplitude per chain:", np.round(np.nanmedian(M, 0), 2))
print("percentile of 4.5:", [round(100 * float((c[np.isfinite(c)] < 4.5).mean()), 1)
                             for c in M.T])
print("share in the published band:",
      [round(100 * float(((c >= 4.5) & (c <= 8.0)).mean()), 1) for c in M.T])

# §2.4 — the calibration the pipeline actually applies
cal = FF.calibrate(P)
p = cal["params"]
print(f"varsigma {cal['scale']:.3f}  band [{p.r_min:.2f}, {p.r_max:.2f}]  "
      f"L {p.window}  nu {p.nu:.2f}  stride {p.stride}")
print("median in-band frames per window, by L:", cal["median_in_band"])

# §5 — the published band really is all-zero here
E = FF.fidgetyfind_dataset(P, FF.PUBLISHED)["E"]
print("max window entropy at the published band:",
      float(np.nanmax(np.concatenate([e.ravel() for e in E if e.size]))))

# §5 — was the reference kernel already applied? (exact nulls at 5.830/10.167 Hz)
k = np.exp(-0.5 * ((np.arange(5) - 2.0) / 2.0) ** 2); k /= k.sum()
segs = [pose[v][s:e, j, c].astype(float)
        for v in vids for j in build_pose.FREE
        for s, e in zip(*[np.where(np.diff(np.concatenate(
            ([0], (obs[v][:, j] > 0).astype(int), [0]))) == d)[0] for d in (1, -1)])
        if e - s >= 256 + len(k) for c in (0, 1)]
def notches(apply_k):
    Q = 0
    for x in segs:
        x = np.convolve(x, k, "valid") if apply_k else x
        f, q = welch(x - x.mean(), fs=25.0, nperseg=256, detrend="linear")
        Q = Q + q
    return [round(float(Q[i] / np.exp(np.mean(np.log(np.r_[Q[i-4:i-1], Q[i+2:i+5]])))), 3)
            for i in (int(np.argmin(abs(f - 5.8297))), int(np.argmin(abs(f - 10.1667))))]
print("null/shoulder as delivered:", notches(False))   # ~[0.92, 1.12]
print("null/shoulder +1 pass     :", notches(True))    # ~[0.07, 0.05]

# §6 — the skeleton is rigid
J = {n: i for i, n in enumerate(build_pose.JOINTS)}
for a, b in (("RHip", "RKnee"), ("RElbow", "RWrist"), ("RKnee", "RAnkle")):
    L = np.concatenate([np.linalg.norm(pose[v][:, J[b]] - pose[v][:, J[a]], axis=-1)
                        for v in vids])
    print(f"{a}->{b}: mean {L.mean():.5f}  sd {L.std():.1e}")

# §6 — and what that does, and does not, force about the direction histogram
def signed_angle(ax, w):
    return np.degrees(np.arctan2(ax[:, 0] * w[:, 1] - ax[:, 1] * w[:, 0],
                                 ax[:, 0] * w[:, 0] + ax[:, 1] * w[:, 1]))
for c, (b, cj) in FF.CHAINS.items():
    dxb, v, a_du, a_v = [], [], [], []
    for vid in vids:
        X = pose[vid].astype(float)
        db, dc = X[1:, b] - X[:-1, b], X[1:, cj] - X[:-1, cj]
        u1 = X[1:, cj] - X[1:, b]
        dxb.append(np.linalg.norm(db, axis=-1))
        v.append(np.linalg.norm(dc, axis=-1))
        # deviation from perpendicular; NOT median|angle|, which sits near 90
        # whatever the spread, because the distribution is symmetric about +/-90
        a_du.append(abs(abs(signed_angle(u1, dc - db)) - 90))   # the identity
        a_v.append(abs(abs(signed_angle(u1, dc)) - 90))         # what FF measures
    f = np.median(np.concatenate(dxb)) / np.median(np.concatenate(v))
    print(f"{c:8s} |dx_b|/|v| {f:.2f}   "
          f"du off perpendicular {np.median(np.concatenate(a_du)):.2f} deg   "
          f"v off perpendicular {np.median(np.concatenate(a_v)):.1f} deg")
```
