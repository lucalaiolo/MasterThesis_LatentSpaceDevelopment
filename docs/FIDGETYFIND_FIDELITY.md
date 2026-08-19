# FidgetyFind in this repository: what is the published construct, and what is not

> **One-sentence answer.** The per-frame measurement, the window gates, the
> band, the histogram entropy and the window tiling are the published ones,
> reproduced from the authors' released code and checked against it; the
> hands-and-feet path had to be re-derived because it needs video we do not
> have, four settings were chosen differently in ways that can change a number
> (§7.1, §7.2, §7.4, §7.7) and five in ways that cannot (§7.3, §7.5, §7.6,
> §7.8, §7.9), and the reduction of per-window entropies to one number per
> recording is **ours** — the released code does not contain one.
>
> If you quote a number from this pipeline, the safe phrasing is in
> [§10](#10-how-to-describe-this-in-the-thesis).

This document exists because "we ran FidgetyFind" is a claim about someone
else's method, and the only honest way to make it is to say exactly which parts
are theirs. Everything below is checkable: each claim names the reference file and
function, ours, and — where it is a numerical claim — the test in
`rvi38_methods/test_methods.py` that pins it (§11).

---

## 1. Provenance: what "official" means here

**The paper.**

> Morais, R., Le, V., Morgan, C., Spittle, A., Badawi, N., Valentine, J.,
> Hurrion, E. M., Dawson, P. A., Tran, T., & Venkatesh, S. (2023). *Robust and
> Interpretable General Movement Assessment Using Fidgety Movement Detection.*
> IEEE Journal of Biomedical and Health Informatics **27**(10), 5042–5053.
> DOI [10.1109/JBHI.2023.3298708](https://ieeexplore.ieee.org/document/10195984/)

**The code.** `https://github.com/RomeroBarata/fidgetyfind`, branch `main`,
commit `84e796f8f1a2fdce04a4d68a886c77b393631514`, fetched 2026-08-19. The
files consulted, with the first 16 hex digits of their SHA-256 as fetched:

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
not read in full. The reference for this implementation is therefore *the code
the authors released*, plus the paper's abstract and the repository README.
Where the article may describe something the code does not implement — and we
know of at least one such place, the reduction to a single score (§8) — we
followed the code, because the code is what we could verify. If you obtain the
article, the two places worth re-checking against it are §8 (the reduction) and
§7.2 (the distal band).

**Our implementation.** `rvi38_methods/a10_fidgetyfind.py`, with the analysis
wiring in `rvi38_methods/run_analysis.py::fidgetyfind` and figures in
`rvi38_methods/figures.py`.

---

## 2. The construct, in one page

For a *moving* joint `c` with *parent* joint `b` (knee against hip, wrist
against elbow, ankle against knee), every consecutive frame pair contributes:

- **amplitude**
  `r_t = 100 · ‖x_c(t+1) − x_c(t)‖ / ‖x_c(t) − x_b(t)‖ · (f / 30)`
  — the joint's displacement as a percentage of the parent limb's own length,
  which makes it free of camera distance and body size, rescaled to the 30 fps
  the published thresholds were calibrated at;
- **direction**
  `a_t = ∠( x_c(t+1) − x_b(t+1) , x_c(t+1) − x_c(t) ) ∈ [−π, π]`
  — the displacement measured against the limb's *own axis*, which makes it
  free of how the infant is lying under the camera;
- **confidence** `s_t = min` over both joints at both frames.

Within a window of 50 frames, stride 20, starting at frame 100:

1. **Void the window** (result `NaN`, "not assessable") if too many frames are
   low-confidence, or if too many frames move far above the band — a limb being
   *transported* is not a limb fidgeting.
2. Otherwise keep the frames whose `r_t` lies in the small-movement band
   `[4.5, 8.0]`. If fewer than 20 % of frames qualify, the window scores
   **`0.0`** — assessable, and nothing fidgety found. That zero is a
   measurement, not a failure, and the distinction from `NaN` is load-bearing.
3. Otherwise histogram the kept directions into 8 bins over `[−π, π]` and score
   the window with the Shannon entropy divided by `log 8`, so it lands in
   `[0, 1]`. Near 1: the limb went in every direction — fidgety movement is
   present, the normal pole. Near 0: one direction only — absent, the pole the
   GMA label marks.

---

## 3. Verdict table

Read the verdict column as: **Identical** — reproduces the released code;
**Forced** — the released code cannot run on our data and this is the
substitute; **Chosen** — we could have matched the reference and deliberately
did not; **Ours** — no counterpart exists in the released code; **Upstream** —
a property of this project's data, not a decision about the construct.

| # | Element | Verdict | Where |
|---|---|---|---|
| 1 | Amplitude `r_t` (÷ parent limb, ×100) | Identical | [§5.1](#51-the-per-frame-triple) |
| 2 | Direction `a_t` against the limb axis | Identical | [§5.1](#51-the-per-frame-triple) |
| 3 | Confidence `s_t` = min over 2 joints × 2 frames | Identical | [§5.1](#51-the-per-frame-triple) |
| 4 | fps rescaling to 30 fps | Identical at 25 fps; [dead zone removed](#73-the-fps-dead-zone) | [§7.3](#73-the-fps-dead-zone) |
| 5 | Window tiling (start 100, length 50, stride 20) | Identical (verified index-by-index) | [§5.2](#52-the-window-tiling) |
| 6 | Low-confidence gate (rate; threshold moot on binary confidence) | Identical | [§5.3](#53-the-window-gates) |
| 7 | Large-motion gate (hips) | Identical | [§5.3](#53-the-window-gates) |
| 8 | Small-movement band `[4.5, 8.0]` (hips) | Identical | [§5.3](#53-the-window-gates) |
| 9 | In-band rate rule → score `0.0`, not `NaN` | Identical | [§5.3](#53-the-window-gates) |
| 10 | Histogram, entropy, ÷ `log(bins)` | Identical up to `4×10⁻⁴` ([EPS](#75-the-epsilon-in-the-entropy)) | [§5.4](#54-the-entropy) |
| 11 | Joint indices | Identical (BODY-15 ≡ BODY-25 for 0–14) | [§5.5](#55-the-joint-indices) |
| 12 | Keypoint smoothing (5-frame Gaussian, σ = 2, confidence-weighted) | Identical in form; [weights kept non-negative](#76-the-smoothing-weights) | [§7.6](#76-the-smoothing-weights) |
| 13 | **Hands and feet: dense optical flow over segmented skin** | **Forced** — no video | [§6.1](#61-the-handsfeet-path-the-big-one) |
| 14 | Distal transport gate (parent motion 1.0 / 2.5, rates 0.3 / 0.1) | Forced — same thresholds, read at *our* parent joint | [§6.1](#61-the-handsfeet-path-the-big-one) |
| 15 | **Camera-motion gate on background flow** | **Forced** — omitted, no video | [§6.3](#63-the-camera-motion-gate-omitted) |
| 16 | Detection confidence source | Forced — `observed` flag, not OpenPose scores | [§6.4](#64-detection-confidence) |
| 17 | Pose detector | Forced — not the authors' fine-tuned OpenPose | [§6.5](#65-the-pose-detector) |
| 18 | Histogram bins on hands/feet (16 → 8) | Chosen | [§7.1](#71-bins-on-the-distal-chains-16--8) |
| 19 | Band on hands/feet (`≥ 0.08`, no upper → `[4.5, 8.0]`) | Chosen | [§7.2](#72-the-band-on-the-distal-chains) |
| 20 | In-band rate gate applied to hands/feet too | Chosen | [§7.4](#74-an-in-band-rate-gate-on-the-distal-chains) |
| 21 | Distal angles limb-relative, not image-frame | Chosen | [§7.7](#77-the-angular-reference-frame-on-the-distal-chains) |
| 22 | Window length kept at 50 *frames* (= 2.0 s, not 1.67 s) | Chosen | [§7.8](#78-window-length-in-frames-not-in-seconds) |
| 23 | Degenerate-limb guard, angle wrapping | Chosen (defensive; no effect on valid input) | [§7.9](#79-two-defensive-guards) |
| 24 | **Per-recording score, `θ = 0.5`, per-chain reporting** | **Ours** | [§8](#8-what-is-entirely-ours) |
| 25 | Torso-normalised coordinates | Upstream | [§9.1](#91-torso-normalised-coordinates) |
| 26 | Gaps already interpolated before we see them | Upstream | [§9.2](#92-interpolation-happens-upstream) |
| 27 | 25 fps recordings | Upstream | [§9.3](#93-25-fps) |

---

## 4. Why the hands/feet path could not be reproduced

This is the single largest difference, so it is worth being concrete about what
the reference actually does there. For each frame and each of the four distal
limbs, `skin_and_flow.py::get_flow_features`:

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
5. histograms the **image-frame orientations** of the surviving pixel flows
   whose magnitude exceeds 8 % of the parent limb's length, into 16 bins,
   accumulating over the window before taking the entropy.

Every one of steps 1–4 requires the video. This project's analysis consumes
`rvi38_analysis.csv`, a long-format keypoint table; there is no video in the
pipeline at all. Reproducing this path is not a matter of effort — the input
does not exist. §6.1 says what we do instead and what it costs.

---

## 5. What is identical

### 5.1 The per-frame triple

Reference, `proximal.py::get_proximal_motion_features`:

```python
score = np.min([score_t[limb[1]], score_t[limb[2]],
                score_tp1[limb[1]], score_tp1[limb[2]]])
v = skeleton_tp1[limb[2]] - skeleton_t[limb[2]]
ref_len = get_reference_length(skeleton_t, reference_joints_indices=[limb[2], limb[1]])
mag = 100 * np.linalg.norm(v) / ref_len
angle = get_angle_between(skeleton_tp1[limb[2]] - skeleton_tp1[limb[1]], v)
```

Ours, `a10_fidgetyfind.py::motion_features`:

```python
v    = x[1:, cj] - x[:-1, cj]                            # displacement of c
ref  = np.linalg.norm(x[:-1, cj] - x[:-1, b], axis=-1)   # parent limb at t
axis = x[1:, cj] - x[1:, b]                              # limb axis at t+1
s    = np.minimum(np.minimum(c[:-1, b], c[:-1, cj]),
                  np.minimum(c[1:, b],  c[1:, cj]))
m    = 100.0 * np.linalg.norm(v, axis=-1) / ref * params.rate_scale
dot  = axis[:, 0]*v[:, 0] + axis[:, 1]*v[:, 1]
det  = axis[:, 0]*v[:, 1] - axis[:, 1]*v[:, 0]
a    = np.arctan2(det, dot)
```

Same displacement, same reference length (the limb measured at frame `t`, as in
the reference — not `t+1`), same angle convention (`atan2(det, dot)` from
`geometry.py::get_angle_between`, with the limb axis taken at `t+1`, as in the
reference), same confidence rule. The only structural difference is that ours
is vectorised over frames instead of looping.

The reference's chains carry three indices (`[8, 9, 10]` = MidHip, RHip,
RKnee), the first of which — MidHip — is **never read** by
`get_proximal_motion_features`: only `limb[1]` (the parent) and `limb[2]` (the
moving joint) enter the computation. We store the pair `(parent, moving)` that
the estimator actually uses.

### 5.2 The window tiling

Reference (`proximal.py`): `range(start_frame, num_frames - window_length, window_stride)`
with `num_frames = motion_features.shape[0] + 1`.
Reference (`distal.py`): `range(start_frame, len(...) - window_length + 1, window_stride)`.
Ours (`window_starts`): `np.arange(start, n_feat - window + 1, stride)`.

The three expressions look different and are the same set. Verified
index-by-index over recordings of 500, 1234, 2001 and 7500 frames: identical to
both reference paths.

### 5.3 The window gates

| gate | reference (as called in `fidgetyfind-single-video.py`) | ours |
|---|---|---|
| low confidence | `mean(score < LIMB_SCORE_THRESH=0.1) > 0.1` (hips) | same, `params.lowconf_rate = 0.1` |
| large motion (hips) | `mean(r > 10.0) > 0.2` | same, `large_motion`, `large_motion_rate` |
| band (hips) | zero out `r < 4.5` or `r > 8.0` | `in_band = (m >= 4.5) & (m <= 8.0)` |
| in-band rate | `sum(r > 0)/50 < 0.2` → score `0.0` | `in_band.sum()/n < 0.2` → `E = 0.0` |
| unusable frames | `r`, `a` zeroed where `s < 0` | excluded by `sc > 0` in the mask |

Two encoding notes, neither a behavioural change:

- The reference marks unusable frames with **negative** scores (`-1` for
  interpolated, `-1000` for undetected) and excludes them with `s < 0`. Our
  confidence is `{0, 1}` from the `observed` flag, so the equivalent test is
  `sc > 0`. The same frames are excluded.
- The reference selects the surviving angles with the sentinel `a1 = a[a != 0.0]`
  (out-of-band angles having been set to exactly `0.0`), falling back to the
  full vector when the selection is empty. We carry the boolean mask instead.
  The two differ only if an in-band displacement has an angle of *exactly*
  `0.0` — a measure-zero event in which the mask is the correct behaviour and
  the sentinel drops a valid sample.

### 5.4 The entropy

Reference proximal:

```python
ha  = np.histogram(a, bins=num_hist_bins, range=(-np.pi, np.pi), density=True)
ha0 = ha[0] / ha[0].sum() + EPS                       # EPS = 1e-5
ea  = -(ha0 * np.log(np.abs(ha0))).sum()
# ... later: / np.log(num_hist_bins)
```

Reference distal:

```python
def entropy(x, eps=1e-7):
    return np.sum(-(x * np.log(x + eps)))
# ... / math.log(num_hist_bins)
```

Ours (`direction_entropy`) is **exactly** the distal form: counts normalised to
probabilities, `-(p · log(p + 1e-7))`, divided by `log(bins)`. Against the
proximal form the largest disagreement over 2000 random windows — uniform and
concentrated direction distributions, 10 to 50 samples each — was
`3.8 × 10⁻⁴`, on a scale where the reported entropies span 0 to 1. §7.5
explains why the two reference forms differ at all.

### 5.5 The joint indices

The reference indexes OpenPose BODY-25; this project uses BODY-15. For indices
0–14 the two layouts coincide exactly, so the constants transfer unchanged:

| index | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| joint | Neck | RSho | RElb | RWri | LSho | LElb | LWri | MidHip | RHip | RKnee | RAnk | LHip | LKnee | LAnk |

The reference's `PROXIMAL_LOWER_LIMBS_INDICES = [[8,9,10],[8,12,13]]`,
`HANDS_LIMBS = [[3,4,-1],[6,7,-1]]` and `FEET_LIMBS = [[10,11,-100],[13,14,-100]]`
therefore address the same joints in our data. (The `-1` / `-100` are the hand
and foot "joints" OpenPose does not detect, which is precisely why the distal
path needs pixels.)

---

## 6. Differences forced by the data

### 6.1 The hands/feet path (the big one)

**What the reference does:** §4 above — segment the hand/foot in the image,
take the optical flow of its pixels, histogram their orientations.

**What we do:** score the wrist and the ankle with *the same estimator the
reference uses at the knee*, one joint proximally — the wrist moving against
the forearm, the ankle moving against the shank.

**Why this substitute and not another.** The reference's own distal path
normalises flow magnitudes by `‖wrist − elbow‖` and gates on the motion of the
wrist; the forearm is already its unit of length and its frame of reference.
Our surrogate measures the same limb, at the same scale, against the same axis
— it simply observes the joint the detector gives us instead of the blob of
skin beyond it. And the estimator itself is not invented: it is the reference's
proximal estimator, applied to a different chain.

**What it costs, stated plainly.**

- *Resolution.* The reference gets thousands of flow vectors per frame from the
  hand; we get one displacement per frame from the wrist. A hand that rotates
  or flutters while the wrist stays put is invisible to us and visible to them.
  Fidgety movement of the *hand itself* is therefore under-measured; our hand
  and foot chains are closer to "distal limb" than to "hand" and "foot".
- *Direction of bias.* Under-measuring hand-only motion pushes our distal
  entropies **down**, i.e. toward the abnormal pole, in both groups. It is a
  loss of sensitivity, not a group-specific bias, unless hand-only fidgeting is
  itself group-specific — which, since absence of fidgety movement is the
  abnormal class, it plausibly is. Treat the distal chains as the weaker half
  of the evidence.
- *What is unaffected.* The hips are the reference's own path on the reference's
  own joints. They are reported separately everywhere — in the log, in
  `fidgetyfind_per_subject.csv` (`fidgetyfind_score_hips`), in `summary.md`
  ("hips only (the unadapted published path)") and in the figures — exactly so
  that a claim resting on the unadapted path can be told from one resting on
  the adapted one.

**Gates, and the one-joint shift in them.** The reference voids a distal window
when the *parent of the hand or foot blob* moves too much
(`large_parent_motion_threshold` = 1.0 for hands, 2.5 for feet, in percent of
the Neck–MidHip length per frame, tolerated in 30 % / 10 % of frames) — and
that parent is the **wrist** (`HANDS_LIMBS = [3, 4, -1]`, so `limb[1] = 4`) or
the **ankle**. In our substitution the wrist and the ankle are the *moving*
joints, so the same rule with the same thresholds is read one joint up: at the
**elbow** and the **knee**. Two consequences, both worth knowing:

- The elbow generally moves less than the wrist it carries, so the same numeric
  threshold fires **less often** for us: our distal gate is the more permissive
  of the two.
- The reference's distal path has no gate on the *moving* thing's own speed
  (its band has no upper edge), and we kept that structure. So where the
  reference would **void** a transported hand (`NaN`, not assessable), a
  transported wrist in our version usually falls out of the band instead and
  scores **`0.0`** ("assessable, nothing fidgety"). Same event, different
  bookkeeping: ours pulls the distal median down and leaves coverage high, the
  reference's leaves the median alone and lowers coverage. The `coverage_*`
  columns of `fidgetyfind_per_subject.csv` are where this is visible.

On the synthetic cohort the foot gate voids ~90 % of foot windows, which is the
gate working as intended: the planted knee fidget *is* transport for the
ankle.

### 6.2 Where the distal difference lands in the numbers

Because of §6.1 and §7.1–7.2, `median_entropy` for `R hand`, `L hand`,
`R foot`, `L foot` should be read as *our* measurement of the reference's
target, not as the reference's measurement. Only `R hip` and `L hip` are the
published quantity. This is why the pipeline reports:

- `fidgetyfind_score_hips` — the unadapted path, its own AUC, CI and p;
- `fidgetyfind_score` — the mean over all six chains;
- `fidgetyfind_score_distal` — the four adapted chains, so the two can be
  compared directly and any divergence is visible rather than averaged away.

### 6.3 The camera-motion gate (omitted)

The reference computes the mean optical-flow magnitude **outside** the
skeleton's bounding box each frame, and voids a window whose 75th percentile of
that background flow exceeds `0.80` (hips) or `1.75` (hands/feet) — a camera
that is being moved or bumped makes the window unscoreable. Both numbers are
pixel-flow quantities and both need the video.

We omit it. Two mitigations, neither a replacement:

- these are fixed-camera cot recordings, so the failure mode the gate exists
  for is rarer here than in the authors' setting;
- our coordinates are root-centred on MidHip (§9.1), so a uniform translation
  of the whole scene — the largest component of camera motion — cancels before
  the measurement.

What is *not* mitigated: camera **rotation** or **zoom**, which survive
root-centring and would inflate direction variability, i.e. push scores toward
the normal pole. `window_entropies` takes a `scoreable` argument for exactly
this: if a per-window mask is ever computed from the video, it can be passed in
without touching anything else.

### 6.4 Detection confidence

The reference gates on OpenPose's per-joint score, a continuous value in
`(0, 1]`, with thresholds 0.1 (hips) and 0.35 (hands/feet). Our table carries
an `observed` flag instead: `1` for a real detection, `0` for a frame filled by
the upstream interpolation. We use it as the confidence, so the threshold is
any value in `(0, 1)` and the reference's 0.1 and 0.35 are indistinguishable on
our data. The *rates* (10 % of frames for hips, 20 % for hands/feet) are the
reference's and are applied as such.

Consequence: our confidence gate is coarser. It cannot distinguish a
low-quality detection from a good one, only a real one from an interpolated
one. If the `observed` column is missing from a table, every frame counts as
detected and the confidence gates never fire at all — the run log says which of
the two situations you are in.

### 6.5 The pose detector

The authors use an OpenPose model fine-tuned on infants (their
`openpose_keras`, `parallel` branch). Our keypoints come from this project's
own pipeline. The band `[4.5, 8.0]` and the 5-frame smoothing were calibrated
against *their* detector's noise; a detector with more jitter pushes more
frames into the band from below and inflates entropy, one with less does the
reverse. Nothing in the code can correct for this, and it is the reason
`--ff-minr` / `--ff-maxr` / `--ff-no-smooth` exist as flags: the sensitivity of
the result to the band is checkable, and should be checked before the band is
quoted as validated on this cohort.

---

## 7. Deliberate choices

Each of these could have matched the reference and does not. The reasoning is
given so you can overrule it — all but §7.6–7.7 are exposed as parameters.

### 7.1 Bins on the distal chains: 16 → 8

The reference uses 16 bins for hands and feet and 8 for hips. That asymmetry
tracks the sample count: a distal window aggregates thousands of pixel flows,
a proximal window has at most 50 displacements. Our distal windows also have at
most 50, so 16 bins would spread ~10–40 in-band samples over 16 cells and read
sparsity as order — pushing entropy *down*, toward the abnormal pole,
artefactually. All six chains use the reference's proximal setting of 8.
Parameter: `--ff-bins`.

### 7.2 The band on the distal chains

The reference gates distal flow at `≥ 0.08` of the parent limb per frame with
**no upper edge** (`flow_rel_mag_low_threshold=0.08`,
`flow_rel_mag_high_threshold=None`). We apply the proximal band `[4.5, 8.0]`
instead.

The argument is that `0.08` sizes a different signal. It is a threshold on
*raw dense optical flow*, unsmoothed, on pixels; ours is a *temporally smoothed
keypoint displacement*, which is the signal `minr`/`maxr` were calibrated for.
Measured on this project's data, the 5-frame smoothing leaves the median
per-frame amplitude at **42–46 %** of its unsmoothed value (all six chains,
synthetic cohort), so applying an unsmoothed-signal threshold to a smoothed
signal excludes most genuine movement. Concretely: with `minr_distal = 8.0` and
no upper edge, all four distal chains score exactly `0.0` for **all 38**
recordings of the synthetic cohort, while the hips — which keep the proximal
band — separate the groups normally. The construct would have reported "no
fidgety movement anywhere in the hands or feet of any infant", which is an
artefact of the mismatched threshold and not a finding.

The reference's values are still reachable: `FFParams(minr_distal=8.0,
maxr_distal=float("inf"))` restores them exactly, and the defaults are `None`
meaning "use the proximal band". **This is the one deviation we would most like
to check against the full article** (§1).

### 7.3 The fps dead zone

The reference rescales per-frame magnitudes by `fps / 30` only when
`abs(fps - 30) > 1.0`; inside that band it applies no correction. We apply
`fps / 30` unconditionally. At our 25 fps the two are identical
(`0.8333`); they differ only for 29–31 fps material, where the reference
applies 1.0 and we apply the true ratio (at 29.97 fps: 1.0 vs 0.999). The
discontinuity in the reference looks like a guard against float noise; removing
it changes nothing measurable and makes the transform continuous.

### 7.4 An in-band rate gate on the distal chains

The reference applies the "too few in-band frames → score `0.0`" rule only in
the proximal path. Its distal path has no such rule and does not need one: with
thousands of pixels per frame the aggregated histogram is always well
populated. With one sample per frame it is not, and a window with three in-band
displacements would otherwise receive a confident-looking entropy computed from
three points. `in_range_rate = 0.2` therefore applies to all six chains.

### 7.5 The epsilon in the entropy

The reference's two paths use different conventions: proximal adds `EPS = 1e-5`
to *every bin* before the log (so the probabilities sum to `1 + 8·EPS` and
empty bins each contribute `−EPS·log EPS ≈ 1.15 × 10⁻⁴`); distal adds
`eps = 1e-7` *inside* the log, the standard guard, contributing nothing for
empty bins. We use the distal convention for all chains. Maximum disagreement
with the proximal convention over 2000 random windows: `3.8 × 10⁻⁴`. The
snippet in §11 recomputes it.

### 7.6 The smoothing weights

The reference smooths each joint and channel along time with a 5-frame Gaussian
(`σ = 2`) whose weights are the kernel **multiplied by the detection score**,
so an unreliable neighbour cannot drag the estimate:

```python
weight  = np.multiply(kernel, padded_score[t:t + window])
new_sig[t] = np.dot(padded_sig[t:t + window], weight) / np.sum(weight)
```

In that code the score of an interpolated frame is `-1` and of an undetected
frame `-1000`, so a neighbouring bad frame contributes a large **negative**
weight, and the weighted mean of an otherwise good frame can be dominated by
it. We take the stated intent — down-weight unreliable neighbours — and
implement it with non-negative weights: `weight = kernel × confidence` with
confidence in `{0, 1}`, falling back to the frame's own coordinates when the
entire neighbourhood is unobserved. This is a deviation from the code as
written; we consider the negative-weight behaviour a defect rather than a
specification, and say so here rather than reproducing it silently.

The smoothing itself is *not* optional in spirit and is on by default (and it
is what costs the 54–58 % of per-frame amplitude quoted in §7.2):
FidgetyFind's band starts a few percent of a limb length per frame, which is
the scale of keypoint jitter, and jitter is directionally uniform — unsmoothed,
it reads as fidgety movement in every infant. `--ff-no-smooth` exists as a
sensitivity check, not as an alternative setting.

### 7.7 The angular reference frame on the distal chains

Worth stating precisely because it is easy to miss in the reference. The
proximal path measures each displacement **against the limb axis**
(`get_angle_between(limb, v)`). The distal path measures pixel flow in the
**image frame** (`orientation = np.arctan2(gy, gx)`); it normalises flow
*magnitudes* by the parent limb but never rotates the flow into the limb's
frame.

Our distal surrogate uses the limb-relative convention, i.e. the proximal one,
for all six chains. Entropy is invariant to a constant rotation, so this
matters only when the limb rotates *within* a window; there, the limb-relative
convention removes the apparent direction change caused by the limb turning,
while the image-frame convention keeps it. Ours can therefore only report
*less* direction variability than the reference's convention would on the same
motion — a conservative difference, and consistent with what the reference
itself does at the hips.

### 7.8 Window length in frames, not in seconds

The reference's 50-frame window is 1.67 s at 30 fps; ours is 2.0 s at 25 fps.
We kept the frame count rather than the duration, which keeps the histogram's
sample budget identical (at most 50 displacements) and matches the rest of this
package, where WCLR-PP also uses a 50-frame (2 s) window. The alternative — 42
frames to preserve 1.67 s — costs 16 % of the samples. The same applies to
`start_frame = 100` (3.33 s there, 4.0 s here) and to `stride = 20`.
Parameters: `--ff-window`, `--ff-stride`, `--ff-start-frame`.

### 7.9 Two defensive guards

- **Degenerate limb.** The reference divides by `ref_len` with no guard; a
  coincident parent and moving joint would produce `inf`/`nan`. We treat
  `ref_len == 0` as unusable (confidence `0`), which routes it through the
  existing low-confidence gate. Likewise a parent magnitude that cannot be
  normalised (a degenerate torso) is recorded as `inf`, so the distal transport
  gate voids the window rather than silently scoring it.
- **Angle wrapping.** `np.histogram(..., range=(-π, π))` silently *discards*
  out-of-range values. Inputs from `arctan2` are always in range, so this never
  fires; we wrap into `[−π, π]` anyway so that an angle-convention mistake by a
  future caller becomes a correct result rather than a quietly wrong one.

Neither changes any result on valid input.

---

## 8. What is entirely ours

**The reduction to one number per recording.** The released code stops at
per-window entropies: `scripts/fidgetyfind-single-video.py` saves
`hips.npy`, `hands.npy`, `feet.npy` and ends. The paper's abstract describes
"a strategy to reduce those measurements to a single score … a direct
translation of the qualitative procedure domain experts use", but that strategy
is not in the repository and the article was not available to us (§1). Ours is
therefore ours, and is deliberately the plainest thing available:

- per chain: the **median** entropy over scoreable windows; the **fraction** of
  scoreable windows at or above `θ`; the **coverage** (share of windows
  assessable at all);
- per recording: the mean of each over the six chains, plus the hips-only and
  distal-only means (§6.2).

**`θ = 0.5`** is fixed a priori as halfway up the normalised entropy scale — a
window whose directions fill two of eight bins scores `log 2 / log 8 = 0.33`,
four of eight `0.67`. It is the only free number in the reduction, it is
exposed as `--ff-theta`, and it affects only the secondary "fidgety-window
rate", never the primary median-entropy score.

**Per-chain reporting and the group contrasts.** Reporting all six chains
whatever any one shows, the exact Mann–Whitney contrast per chain, the
Westfall–Young family-wise p beside it, and the Spearman agreement with Φ, the
Kemeny constant and whole-body coupling are this project's analysis design, not
the paper's.

**Naming.** "R hip / L hip / R hand / L hand / R foot / L foot" follows the
reference's own grouping (`hips`, `hands`, `feet`). Note that the "hip" chains
watch the **knee** move against the thigh; the name is the reference's, the
moving joint is documented in `CHAINS`.

---

## 9. Differences that come from this project's data

These are not decisions about the construct, but they change what it sees, so
they belong in any honest account.

### 9.1 Torso-normalised coordinates

Our keypoints are root-centred on MidHip and scaled so the Neck–MidHip distance
is 1, per frame. The reference works in raw pixels.

- **Harmless part.** Every quantity FidgetyFind computes is a *ratio* to a limb
  length or to the trunk, so a constant global scale cancels exactly. This is
  the property that lets the published thresholds transfer at all.
- **Not harmless part.** The normalisation is re-estimated **every frame**. If
  the Neck–MidHip distance breathes by a fraction `ε` between frames
  (out-of-plane torso rotation, detector noise), every joint acquires a
  spurious displacement pointing radially away from MidHip, of size
  `ε · ‖x_c‖`. As a fraction of the parent limb — the unit the band is measured
  in — that is `ε · ‖x_c‖ / ‖x_c − x_b‖`: about `1.1 ε` at the knee
  (`0.48 / 0.45`) but about `2.1 ε` at the wrist (`0.60 / 0.28`) — the forearm
  is short and far from the root. (Both ratios are read off the canonical
  resting pose in `make_synthetic.BASE`; they vary with posture.) At
  `ε ≈ 1 %` that is ~1 % of a limb length per frame at the knee and ~2 % at the
  wrist, against a band that starts at 4.5 %: not enough to put still frames
  into the band on its own, but enough to add to genuine motion, and its
  direction is radial rather than uniform, so it perturbs the direction
  histogram rather than merely blurring it. The effect is largest exactly where
  our distal substitution already has least resolution (§6.1).
- **Also a benefit.** Root-centring removes whole-scene translation, which is
  part of what the omitted camera gate (§6.3) was there for.

A run on raw pixel coordinates would be closer to the reference in this respect
and is possible — nothing in `a10_fidgetyfind.py` assumes normalised input.

### 9.2 Interpolation happens upstream

The reference interpolates gaps itself, inside `_smooth_skeletons`, with
`scipy.interpolate.interp1d` over frames with positive score, and only where
fewer than `window/3` of the 5-frame neighbourhood are bad. This project's
loader has already linearly interpolated interior gaps and held the ends flat
before `a10_fidgetyfind.py` sees anything. The `observed` flag marks those
frames, and our confidence gate excludes them, so the *gating* outcome matches;
what differs is that the reference would refuse to interpolate long gaps, while
upstream we interpolate them and then exclude them by flag. Net effect on the
measurement: none we can identify, since both paths end up not measuring those
frames.

### 9.3 25 fps

The reference's `STANDARD_FPS = 30`. Ours are 25 fps recordings. The per-frame
amplitude rescaling (§5.1, §7.3) handles the thresholds; the window geometry is
discussed in §7.8. Nothing else in the construct depends on frame rate.

---

## 10. How to describe this in the thesis

Wording that is defensible:

> Fidgety movement was additionally quantified with **FidgetyFind** (Morais et
> al., 2023), reimplemented from the authors' released code. The hip chains
> reproduce the published skeleton-based estimator; because the published
> hand and foot path requires dense optical flow over segmented video, which
> this keypoint-only cohort does not provide, those chains were scored with
> the same estimator applied at the wrist and ankle, and the video-based
> camera-motion gate was omitted. The reduction of per-window entropies to a
> per-recording score is our own, the released code providing none.

Wording to avoid:

- "We ran FidgetyFind" *without qualification* — four of six chains are adapted.
- "FidgetyFind detected …" for a distal-chain result — say "our skeleton-only
  adaptation of the FidgetyFind distal path".
- Quoting `fidgetyfind_score` as *the* published score — it is our reduction of
  the published per-window measure. `fidgetyfind_score_hips` is the closest
  thing to an unadapted number.

If a reviewer asks "is this FidgetyFind?", the answer is: the measurement is,
at the hips exactly and at the wrists/ankles by a documented substitution; the
score built on top of it is ours.

---

## 11. How to check every claim here

```bash
cd rvi38_methods
python test_methods.py          # 124 checks; the FidgetyFind ones are below
```

| claim | check |
|---|---|
| entropy is 0 / `log2/log8` / 1 for one / two / uniform directions | `test_fidgetyfind_entropy` |
| out-of-range angles are wrapped, not dropped (§7.9) | `test_fidgetyfind_entropy` |
| in-band random directions score ≈ 1; one axis scores `1/3` | `test_fidgetyfind_windows` |
| a still recording scores `0.0`, **not** `NaN` (§5.3, rule 2) | `test_fidgetyfind_windows` |
| movement far above the band voids the window (`NaN`) | `test_fidgetyfind_windows` |
| unobserved keypoints void the window (§6.4) | `test_fidgetyfind_windows` |
| rotation, scale and translation invariance (§2) | `test_fidgetyfind_invariance` |
| the same trajectory at 30 fps scores `30/25` larger (§5.1) | `test_fidgetyfind_invariance` |
| smoothing shrinks jitter; an unobserved frame carries no weight (§7.6) | `test_fidgetyfind_smoothing_and_reduction` |
| the reduction is median-per-chain, mean-over-chains (§8) | `test_fidgetyfind_smoothing_and_reduction` |
| a planted fidgety signal is recovered with the right sign | `test_fidgetyfind_planted_cohort` |

The claims in this document that the test suite does not cover are reproduced
by the snippet below, run from `rvi38_methods/`. It checks the window tiling
against **both** reference expressions, the entropy bound of §5.4/§7.5, the fps
dead zone of §7.3, the smoothing cost of §7.2/§7.6, and the "reference distal
band zeroes every distal chain" claim of §7.2.

```python
import numpy as np, build_pose, a10_fidgetyfind as FF

p = FF.FFParams()

# §5.2 — the tiling is index-identical to proximal.py and to distal.py
for F in (500, 1234, 2001, 7500):
    n = F - 1                                        # motion-feature rows
    ours = list(FF.window_starts(F, p))
    assert ours == list(range(p.start_frame, n + 1 - p.window, p.stride))
    assert ours == list(range(p.start_frame, n - p.window + 1, p.stride))

# §5.4 / §7.5 — our entropy is the reference's distal form exactly, and within
# 4e-4 of its proximal form (which adds EPS=1e-5 to every bin)
rng = np.random.default_rng(0)
def prox(a, bins=8, EPS=1e-5):
    h = np.histogram(a, bins=bins, range=(-np.pi, np.pi), density=True)[0]
    h = h / h.sum() + EPS
    return -(h * np.log(np.abs(h))).sum() / np.log(bins)
worst = max(abs(prox(a) - FF.direction_entropy(a, 8))
            for a in (rng.uniform(-np.pi, np.pi, rng.integers(10, 50))
                      for _ in range(2000)))
assert worst < 4e-4, worst

# §7.3 — the fps rescale, with the reference's |fps-30|<=1 dead zone removed
assert abs(FF.FFParams(fps=25.0).rate_scale - 25 / 30) < 1e-12
assert FF.FFParams(fps=30.0).rate_scale == 1.0

# §7.2 / §7.6 — smoothing cost, and what the reference's distal band would do.
# Any cohort CSV works; the synthetic one comes from make_synthetic.py.
vids, pose, obs, _ = build_pose.build("synth/synth_analysis.csv", out=None)
P, O = [pose[v] for v in vids], [obs[v] for v in vids]
sm  = FF.motion_features(P[0], O[0], FF.FFParams(smooth=True))["magnitude"]
raw = FF.motion_features(P[0], O[0], FF.FFParams(smooth=False))["magnitude"]
print("smoothed / raw median amplitude per chain:",
      np.round(np.median(sm, 0) / np.median(raw, 0), 2))        # ~0.42-0.46

ref_band = FF.FFParams(minr_distal=8.0, maxr_distal=float("inf"))
M = FF.fidgetyfind_dataset(P, O, ref_band)["median_entropy"]
print("recordings with any distal signal under the reference band:",
      int(np.nansum(M[:, 2:] > 0)), "of", M[:, 2:].size)          # 0 of 152
```

---

## 12. Restoring exact parity if video becomes available

In order of value:

1. **The distal path.** Port `skin_and_flow.py::get_flow_features` and
   `distal.py::get_distal_windows_entropy` and feed their per-window entropies
   in place of our hand/foot chains. Everything downstream — the reduction, the
   contrasts, the figures — is agnostic to where a chain's entropies came from.
2. **The camera-motion gate.** Compute `get_background_flow` and
   `get_scoreable_windows_wrt_flow` and pass the result as the `scoreable`
   argument of `window_entropies`; the hook is already there.
3. **Confidence.** Carry the detector's per-joint scores into the keypoint
   table instead of a binary `observed` flag, and the 0.1 / 0.35 thresholds
   become meaningful (§6.4).
4. **Raw pixel coordinates.** Run on un-normalised keypoints to remove the
   per-frame re-normalisation of §9.1.

With 1–4 done, the only remaining differences would be the deliberate ones of
§7 — each a one-line parameter change — and the reduction of §8, which has no
published counterpart to restore.
