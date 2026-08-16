# Latent-Space Development for Infant Motion: A Technical Companion

> Purpose. This document is a complete, self-contained record of the method,
> mathematics, implementation, and figures of the project, written to be handed
> to an assistant to draft the thesis. It is deliberately detailed: every model,
> loss, selection rule, statistical test, and plot is specified, with the
> literature it rests on cited inline. Where a citation could not be verified
> against the primary source it is flagged **[verify]** — do not present those as
> settled in the final bibliography without checking.

---

## 0. Problem and contributions

We learn a low-dimensional **latent representation of human motion** from 2D
keypoint sequences and use it as the substrate for an interpretable,
clinically-oriented analysis of infant **general movements**. The clinical target
is the **fidgety movements** of Prechtl's General Movements Assessment (GMA): the
small, continual, elegant movements of neck, trunk and limbs normally present at
roughly 9–20 weeks post-term, whose *absence* is an early marker of neurological
risk (Einspieler & Prechtl, 2005; Prechtl, 1990).

The pipeline has four stages: (i) train a variational autoencoder (VAE) on a
large 2D-pose pretraining corpus (YouTube), (ii) fine-tune it on the clinical
cohort (RVI-38), (iii) fit an interpretable temporal state model — a hidden
Markov model (HMM) or autoregressive HMM (AR-HMM) — over the learned latent
trajectory, and (iv) read out phenotype features and test them against the GMA
label.

Contributions, in the order the document develops them:
1. A **temporal-latent** VAE whose reconstruction *moves* (does not collapse to a
   static mean pose), with two backbones (convolutional and transformer) and an
   optional factorised space-time attention.
2. A principled **model-selection** protocol gated on non-collapse, a
   pretrain→fine-tune recipe, and a subject-wise data split.
3. An **HMM / AR-HMM interpretability layer** over the latent, with a
   full-covariance emission model justified by an affine-invariance argument, a
   seam-free stitching of clip-local latents into per-recording trajectories, and
   per-state dwell times read directly off the decoded path.
4. A set of **interpretability figures** and an **exploratory clinical test**,
   with the statistical honesty the small cohort demands.

---

## 1. Data and preprocessing

### 1.1 Two datasets

**Pretraining — YouTube 2D poses.** A large corpus of 2D keypoint sequences
extracted from web video, used only to learn a generic motion prior.

**Clinical — RVI-38.** 38 RGB recordings of 38 different infants aged 12–21 weeks,
collected during routine care at the Royal Victoria Infirmary, Newcastle, and
introduced with the Newcastle pose feature-fusion work (McCay et al. **[verify:
exact title/venue/year]**). Recorded top-down, 1920×1080 at 25 fps, durations
40 s–5 min (mean ≈ 3 min 36 s). Each recording carries a single GMA label:
presence (normal) vs. absence (abnormal/at-risk) of fidgety movements. The class
distribution is **32 normal / 6 abnormal**; the six positive subjects are
0005, 0009, 0010, 0011, 0018, 0019. The 12–21-week window coincides with the
fidgety period, so the recordings capture the movement the analysis targets, and
the label is the fidgety construct itself, not a proxy.

### 1.2 The BODY-15 skeleton

Keypoints are the OpenPose (Cao et al., 2017) **BODY-15** subset — 15 joints, the
torso and four limbs, dropping the four face keypoints and inserting a *MidHip*:

```
0 Nose   1 Neck   2 RShoulder 3 RElbow 4 RWrist 5 LShoulder 6 LElbow 7 LWrist
8 MidHip 9 RHip  10 RKnee    11 RAnkle 12 LHip  13 LKnee    14 LAnkle
```

Bones, limb groups (`right_arm=[2,3,4]`, `left_arm=[5,6,7]`, `right_leg=[9,10,11]`,
`left_leg=[12,13,14]`, `head=[0]`) and left/right pairs are defined against this
index order. The loader auto-detects the joint count from the data and resolves
the matching skeleton, so the same code path serves BODY-15 and the legacy
COCO-18 layout.

### 1.3 Normalisation

Coordinates are stored **root-centred on MidHip and torso-scaled** (Neck at
`(0,1)`, MidHip at `(0,0)`, unit Neck–MidHip length). Two consequences matter
downstream: (i) MidHip and Neck are **constant by construction** and carry no
variance — their reconstruction is trivially perfect and they contribute nothing
to the latent, so the effective free-joint count is 13; (ii) because the data is
already normalised, the training/analysis default is `preprocess="none"`; a
`center` / `center_scale` path is available for raw-pixel exports. Missing
detections are linearly interpolated over interior gaps and held flat at the
ends; a joint never seen in a clip is filled with the origin.

### 1.4 Clips

A recording of `F` frames is cut into overlapping windows of length `T` at stride
`s` (default `s = T/2`), each a tensor `(T, J, D)` with `J=15`, `D=2`. Each clip
records which video and start-frame it came from, so splits and analyses can be
kept subject-aware.

---

## 2. The variational autoencoder

### 2.1 ELBO and what variance is modelled

For an observation `x` (a clip), latent `z`, the VAE (Kingma & Welling, 2014;
Rezende et al., 2014) maximises the evidence lower bound

$$
\mathcal{L}(x) = \mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]
- \beta\, D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big),
$$

with a diagonal-Gaussian encoder `q_\phi(z|x)=\mathcal N(\mu_\phi(x),\operatorname{diag}\sigma^2_\phi(x))`,
standard normal prior `p(z)=\mathcal N(0,I)`, and the KL in closed form

$$
D_{\mathrm{KL}} = \tfrac12\sum_{d}\big(\mu_d^2 + \sigma_d^2 - \log\sigma_d^2 - 1\big).
$$

The **posterior variance** `σ²_φ` **is modelled** (the encoder outputs `logvar`).
The **decoder/observation variance is not**: we use a mean-squared reconstruction,
i.e. an isotropic fixed-unit-variance Gaussian likelihood
`p_θ(x|z)=\mathcal N(\hat x_\theta(z), I)`, for which `-\log p_θ(x|z)` reduces (up
to a constant) to `\|x-\hat x\|^2`. The weight `β` follows β-VAE (Higgins et al.,
2017); `β=1` is the exact ELBO.

### 2.2 Posterior collapse and the single-global-latent failure

A recurring failure (Bowman et al., 2016; Razavi et al., 2019) is **posterior
collapse**: the KL term drives `q≈p`, the latent stops carrying information, and
the decoder reproduces the data-independent mean. For motion we observed a
specific, robust variant: **a VAE that compresses a whole clip to one latent
vector reconstructs the temporally-averaged (static) pose** — regardless of
`β=0`, a large velocity penalty, LR warm-up, or more epochs. Diagnosis: a single
global latent has no per-time capacity, so the easiest optimum of the MSE is "emit
the mean". Conditioning on the mean pose and decoding a residual (a FiLM-style
anchor; Perez et al., 2018) made it worse — it drove the residual to zero, since
the anchor *is* the mean. This motivated the temporal-latent design (§2.4).

### 2.3 Backbones (whole-clip)

Two whole-clip encoders/decoders, both generic in the coordinate dimension:

- **Convolutional** (ConvVAE): a 1-D temporal CNN over frames, with kernels and
  strides `(5,3,3)/(1,2,2)`; the stride product sets the temporal downsample.
- **Transformer** (TransformerVAE): frame tokens (each frame's `J` joints + a
  mask channel projected to `d_model`), sinusoidal temporal positional encoding
  (Vaswani et al., 2017), pre-norm blocks with a terminal LayerNorm (to control
  residual-stream drift at depth), depth configurable per side.
- **Factorised space-time** (SpatioTemporalTransformerVAE): one token per
  *(joint, frame)*, alternating **spatial** attention across joints within a
  frame and **temporal** attention across frames within a joint — the divided
  attention of PoseFormer / ST-Transformer (Zheng et al., 2021). It separates the
  pose and motion inductive biases and is quadratic in `J` and `T` separately
  rather than in `J·T`.

All three, being whole-clip single-latent models, exhibit the collapse of §2.2
and are retained mainly as baselines.

### 2.4 Temporal-latent VAEs (the models that move)

The fix is a **latent sequence** rather than a latent vector: down-sample time by
a factor `l` and keep a `d_z`-wide latent at **each** of the `T/l` windows. This is
the continuous (non-VQ) analogue of the temporally-downsampled latent used by
motion generators — T2M-GPT (Zhang et al., 2023), MotionGPT (Jiang et al., 2023),
and PRISM (Ling et al. **[verify: arXiv id / authors]**) — but kept as a plain
Gaussian VAE. The per-window posterior is `(d_z, T/l)`, flattened to
`(B, d_z\cdot T/l)` so the standard ELBO/KL/β machinery and the whole analysis
toolkit run unchanged.

- **`temporal_conv`.** The conv trunk produces `(2C, T/l)`; per-window
  `1×1` convolutions read `μ`, `logvar`, and lift `z` back to `2C` before the
  decoder upsamples to `T`.
- **`temporal_transformer`.** Frame (or joint-frame) tokens run through the
  transformer; the `T` tokens are grouped into `T/l` windows and mean-pooled; a
  per-window linear head reads `d_z`. The decoder lifts each window latent,
  nearest-neighbour-upsamples to `l` frame queries, refines with transformer
  blocks, and emits `D·J` per frame. Two attention modes share this latent
  layout: **frame-token** (`temporal`) and **factorised** (`factorized`,
  reusing the space-time block of §2.3). Because the latent layout is identical
  across modes, all downstream analysis is attention-agnostic.

Because a distinct latent controls each window, the decoded clip cannot collapse
to one pose; the reconstruction reaches ≈94 % of the target temporal standard
deviation. A `pool_global` helper mean-pools the windows to one per-clip vector
for phenotype-level summaries.

### 2.5 Training recipes, masking, and auxiliary losses

Three **recipes** control the objective (a masked-VAE framing):
1. plain reconstruction;
2. plain + an **auxiliary masked reconstruction** (weight `λ_aux`), where a
   masking policy hides part of the input and the model must reconstruct the
   visible/hidden split;
3. **inpainting** — a mask-conditioned decoder head reconstructs hidden joints.

**Masking policies:** none, uniform, top-k-speed, softmax-speed, per-frame-speed,
and limb (hide one whole limb per clip). Speed-based policies score joints by
frame-to-frame speed so the mask preferentially hides moving joints.

An optional **velocity loss** adds the MSE of frame differences
`v_t = x_{t+1}-x_t`, i.e. `λ_vel\,\|\hat v - v\|^2`, a temporal-smoothness /
motion term. It helps but was *not* the cure for collapse (the temporal latent
was).

### 2.6 The KL schedule (β), and why it is fundamental

Collapse-vs-blur is governed by the KL weight schedule:
- **warmup**: `β(e)=β_{\max}\min(1, e/E_{\text{warm}})` — linear ramp; `E_{\text{warm}}=0` gives a constant `β_{\max}`;
- **delayed_warmup**: hold `β_{\min}` for `E_{\text{delay}}` epochs (train the
  autoencoder largely unregularised), then ramp — a cyclical/delayed-annealing
  idea (Fu et al., 2019);
- **computed**: the Asperti–Trentin (2020) rule that sets `β` from the running
  minimum reconstruction so the two terms stay balanced.
- **free bits** (Kingma et al., 2016): replace the KL by
  `\sum_d \max(\gamma, \mathrm{KL}_d)`, reserving a per-dimension floor `γ` of
  information that the KL cannot penalise away — a direct anti-collapse device.

**What β means for the downstream state model.** Because the emission model of
the HMM/AR-HMM is *full-covariance* and hence **affine-invariant** (§4.3,
Prop. 4.1), the *magnitude* of the KL — largely a latent-scale artefact — does not
matter to it. What matters is only that the latent (a) encodes motion and (b) is
not collapsed. So β is tuned to avoid collapse and, optionally, to prune noise
dimensions (a cleaner latent gives better-conditioned emission covariances and a
more honest intrinsic-dimension estimate), not to hit any KL target.

### 2.7 Checkpoint selection and the data split

`best.pt` is chosen on a KL-schedule-independent validation quantity — by default
**reconstruction MSE** (`checkpoint_metric="rec_full"`), which is `β`-independent
and monotone enough never to lock onto the untrained epoch-0 model; `"elbo"` (at
the ceiling β) and `"loss"` (legacy) are alternatives.

The **train/validation split is video-wise**: whole recordings are held out
(≈15 %), so subjects are disjoint between splits. A within-video time cut would
train and validate on the same subject and leak content across neighbouring
overlapping clips; the video-wise split is the honest choice for a generalisation
claim. It is seed-deterministic so every caller (training, sweep scoring, the
analysis pass) rebuilds the identical partition. A single-video dataset falls back
to a within-video time cut. Training can also be run with **no** held-out split
(`val_fraction=0.0`) for a committed full-data pretrain once the configuration is
fixed.

### 2.8 Pretrain → fine-tune

The selected configuration is pretrained on the full YouTube corpus, saved as
`{state, config}` (the config travels with the weights so architecture/capacity
can never drift on reload), then **warm-started** on RVI-38 at a ~10× lower
learning rate (`train(..., init_state=pretrained_state)`; strict load so a shape
mismatch fails loudly). Fine-tuning uses a constant β at the pretrain ceiling
(`warmup_epochs=0`) — re-running the warm-up from zero would let the converged
posterior drift for the whole ramp. All stages share one fixed geometry
(`T=64`, `l=4` → 16 windows, `d_z=8`; both datasets at 25 fps).

---

## 3. Model selection and the latent-geometry diagnostics

### 3.1 The sweep, gated on motion

Model selection sweeps the discrete axes — architecture ∈ {`temporal_conv`,
`temporal_transformer`}, attention ∈ {`temporal`, `factorized`}, recipe ∈ {1,2,3},
masking policy — at fixed capacity and a fixed β schedule, on the video-wise
split. Runs are ranked by **held-out masked reconstruction, gated on a
non-collapse criterion**: a model whose reconstruction temporal standard deviation
is below ~50 % of the target (i.e. it does not move) is disqualified regardless of
its MSE. This is the one deviation from "pick the lowest validation loss": two
models can have near-identical MSE while one is static and useless for the
dynamics analysis. Capacity (width, depth, channels) is fixed across the sweep and
tuned for the winner afterwards, to keep the grid tractable.

### 3.2 The latent-geometry toolkit (`vae_analysis/`)

A battery of diagnostics characterises the trained latent (used for validation and
figures, not all reported in the thesis):
- **Decoder geometry / Jacobian.** The decoder Jacobian `J=∂\hat x/∂z` (via
  `jacrev`) gives sensitivity maps (joint×latent, time×latent) and the **pullback
  metric** `G=J^\top J`, whose eigenvalue spectrum and condition number describe
  the local stretching of the latent manifold (Arvanitidis et al., 2018;
  Shao et al., 2018). For the temporal latent, the time×latent map should show
  each dimension acting on its own window — a direct check of temporal
  localisation.
- **Encoder geometry.** Encoder Jacobian, a latent×joint read map, and the
  **posterior-precision spectrum** (per-dimension `1/σ²`), giving the count of
  *live/active* latent dimensions.
- **Prior fit / shape.** An MMD two-sample test of `q(z)` vs `N(0,I)` (Gretton
  et al., 2012; note it loses power in high dimension, so it is read alongside
  per-dimension precision), the TwoNN intrinsic-dimension estimator (Facco et al.,
  2017), and Gaussian-mixture cluster structure.
- **Information / disentanglement.** Total-correlation decomposition, active-unit
  counts, rate–distortion curves; MIG/DCI/SAP where factors of variation exist.
- **Dynamics.** Per-video latent trajectories with change-point, HMM, and
  Ornstein–Uhlenbeck fits (the seed of §4).

---

## 4. The temporal state model: HMM

### 4.1 The Gaussian HMM

Fit over a per-recording latent-window sequence `Z=(z_1,\dots,z_W)`,
`z_w\in\mathbb R^{d}` (`d=d_z`), sampled at the window rate `f_{\text{win}}`. A
hidden state `s_w\in\{1,\dots,K\}` per window, with

$$
\pi_k=\Pr(s_1{=}k),\quad A_{jk}=\Pr(s_{w+1}{=}k\mid s_w{=}j),\quad
p(z_w\mid s_w{=}k)=\mathcal N(z_w;\mu_k,\Sigma_k),
$$

and the two conditional-independence assumptions (first-order Markov chain; each
observation depends only on its state). The joint factorises as

$$
p(Z,S\mid\theta)=\pi_{s_1}\mathcal N(z_1;\mu_{s_1},\Sigma_{s_1})
\prod_{w=2}^{W}A_{s_{w-1},s_w}\,\mathcal N(z_w;\mu_{s_w},\Sigma_{s_w}).
$$

With `N` independent recordings the objective is the **summed** log-likelihood
`\ell(\theta)=\sum_n\log p(Z^{(n)}\mid\theta)` — this is the content of the
`lengths` argument: concatenating recordings would insert a fictitious transition
across a boundary that does not exist. (Rabiner, 1989; Bishop, 2006, ch. 13.)

### 4.2 Inference and learning (from a package, not hand-rolled)

Learning is EM / **Baum–Welch** (Baum et al., 1970; Dempster et al., 1977) via
the forward–backward recursions; the posterior state and transition
responsibilities `γ_w(k)`, `ξ_w(j,k)` give closed-form M-step updates for `π`, `A`,
`μ_k`, `Σ_k`. Decoding for all dwell/transition statistics is the MAP path from
the **Viterbi** algorithm (Viterbi, 1967), *not* the marginal argmax, which
flickers between adjacent windows and would corrupt dwell times. Model selection
uses the Bayesian Information Criterion (Schwarz, 1978) alongside cross-validated
held-out likelihood. Implementation: `hmmlearn`'s `GaussianHMM`; EM is not
re-implemented.

### 4.3 The full-covariance emission decision (affine invariance)

Emissions are **full-covariance** with a ridge floor, chosen a priori on a
structural argument, not tuned.

> **Proposition 4.1 (affine invariance).** Under an invertible affine map
> `z\mapsto\tilde z=Az+b`, the full-covariance Gaussian-HMM fit is invariant: the
> maximiser transforms as `μ_k\mapsto A^{-1}(μ_k-b)`,
> `Σ_k\mapsto A^{-1}Σ_kA^{-\top}`; the responsibilities and the Viterbi path are
> unchanged; the log-likelihood shifts by the constant `-M\log|\det A|`
> (`M=\sum_nW_n`), which cancels in every model comparison.
>
> *Proof sketch.* A Gaussian density transforms as
> `\mathcal N(\tilde z;Aμ_k+b,AΣ_kA^\top)=|\det A|^{-1}\mathcal N(z;μ_k,Σ_k)`;
> substituting into the joint contributes one factor `|\det A|^{-1}` per emission,
> hence `|\det A|^{-M}` overall; `π,A_{jk}` are unaffected; responsibilities are
> ratios in which the constant cancels. ∎

Consequences: (i) whitening is a no-op under full covariance, so the covariance
and whitening questions are one question; (ii) the HMM is insensitive to the
latent's scale/rotation — much of what the KL magnitude reflects — so β need not
target a KL value. A **diagonal** emission model does not have this invariance:
"Σ diagonal" is not preserved by rotation, and by the law of total covariance the
pooled covariance `C=\Sigma_W+\Sigma_B` (within- plus between-state) is
diagonalised by PCA only when `[\Sigma_W,\Sigma_B]=0`, which does not hold
generically — so a diagonal model diagonalises the wrong object. At `d=8` a full
covariance is `d(d+1)/2=36` parameters per state, comfortably estimable given the
window budget (§4.5). Regularisation is a two-rung ladder: a ridge floor
(`min_covar`) for ill-conditioning, and a drop to **tied** covariance (one shared
full matrix; Gales, 1999) for a genuine occupancy shortfall; both are logged, and
the fallback is documented up to semi-tied (MLLT) and factor-analysed emissions
(Ghahramani & Hinton, 1996; Ledoit & Wolf, 2004) though not needed here.

### 4.4 Selection, restarts, and cost

EM is a *local* optimiser, so each fit uses several **k-means-seeded restarts**,
keeping the best training log-likelihood; this guards against degenerate optima
and stabilises the K comparison. `K` is chosen by **held-out log-likelihood per
window under the video-wise split** (BIC reported alongside): with 38 recordings a
single split is noisy, so 5-fold (or leave-one-video-out) is used. The number of
EM fits is `n_{\text{restarts}}\times(1+n_{\text{splits}})\times|K\text{-range}|`;
they are independent, so K-selection is parallelised across CPU cores
(`n_jobs`). Progress is logged per K. **Reading the score:** the selection score
is a (held-out) log-likelihood — higher wins; training likelihood is
monotonically non-decreasing in `K` (never select on it), whereas the held-out
score rises then plateaus/falls, its turning point identifying `K`.

### 4.5 Data budget (why 38 recordings suffice for the *fit*)

Total footage `≈38×216 s ≈ 8200 s`; at `l=4`, 25 fps the window budget is
`≈8200×6.25 ≈ 51{,}000` windows (`≈33{,}000` taking ~65 % as active). A rare state
at 5 % occupancy still sees thousands of windows against the 36-parameter emission
— full covariance is estimable. **Guardrail (pseudoreplication).** The window
count is an *estimation* resource, not a sample size: there are 38 independent
subjects, and every *inferential* claim uses `n=38`, never a per-window or
per-clip count.

---

## 5. From clip-local latents to per-recording trajectories

### 5.1 The seam problem

The VAE encodes only `T=64` frames, so a recording is a run of clips whose
`(16, d_z)` latent blocks must be joined. Each clip is an independent forward pass
with no cross-clip memory, so two independently-encoded clips can place the same
motion at different latent locations, and the last window of one clip need not
join the first window of the next. Non-overlapping clips therefore inject a
**periodic discontinuity every `T` frames** — a comb at `f_{\text{frame}}/T`
(≈0.39 Hz at 25 fps) with harmonics — which would create a fake 16-window
transition rhythm in the HMM and cap every dwell estimate at 16 windows.

### 5.2 Overlap-crop stitching

Encode clips at **50 % overlap** (stride `T/2`) and keep only each clip's
**central windows**, discarding the edge windows. Consecutive kept regions then
tile the recording with no gap and no overlap; every retained window carries
intra-clip context on both sides; the trajectory length equals non-overlapping
tiling at twice the encoding cost. The seam and its comb vanish. A **diagnostic**
computes the Welch power spectral density of the stitched trajectory and checks
that no line survives at `f_{\text{frame}}/T` or its harmonics; it gates the
per-recording `lengths` construction. (Because the analysis then runs on long
per-recording trajectories rather than 16-window clips, the periodogram resolution
improves from `Δf≈0.47 Hz` to `≈0.005 Hz`.)

### 5.3 Two emission streams: pose vs. delta

The stitched trajectory can feed the state model as the **pose** stream `x_w=z_w`
or the **delta** stream `x_w=\Delta z_w=z_{w+1}-z_w`. The delta is a discrete
first derivative — a **latent velocity** (`Δz_w≈Δt\,\dot z`). Under the pose
stream a state is a *kind of posture*; under the delta stream a state is a *kind
of movement* (a characteristic velocity), with a near-zero-`Δz` state encoding
*stillness*. Because the fidgety label is about movement *quality*, not posture,
the delta stream sits closer to the construct: absent-fidgety shows up as high
occupancy of the near-zero state, and the delta stream is the latent-space twin of
the raw-velocity spectral feature of §6, so agreement between them is
evidence the latent captured the clinical dynamics. Costs: delta states have no
single decodable pose (render the *integrated* motion instead), differencing
amplifies noise (`\operatorname{Var}(\Delta z)=2σ^2(1-ρ)`, so it wants a smoother
latent), and there is one fewer window per recording. The stream is selected by
held-out likelihood, never by the label.

---

## 6. State timescales: dwell time

### 6.1 Rates

`f_{\text{win}}=f_{\text{frame}}/l` (6.25 Hz at 25 fps, `l=4`), so one window is
0.16 s. Dwell times are quoted in windows and in seconds; the conversion is a
division by `f_{\text{win}}`.

### 6.2 Dwell time

In an HMM the dwell in state `k` is geometric with mean

$$
\bar\tau_k=\frac{1}{1-A_{kk}}\ \text{windows}
=\frac{1}{(1-A_{kk})f_{\text{win}}}\ \text{seconds}.
$$

This is the only timescale a first-order chain asserts about a state: **how long
the state is held once entered**. It follows from the transition matrix alone and
says nothing about when the state recurs or whether the underlying movement is
periodic.

> **This is deliberately where the timescale argument stops.** An earlier version
> of this work mapped `A_{kk}` to an oscillation frequency by treating each state
> as one half-cycle of a periodic movement, `f=f_{\text{win}}/2\bar\tau`, and
> flagged states whose implied frequency fell in a nominal 0.5–2 Hz band. That
> map is not identified by the fit: the HMM never estimates a period, the decoded
> states do not alternate in the regular two-state cycle the derivation assumes,
> and the band edges had no empirical support in this cohort. Both the frequency
> map and the band have been removed rather than re-tuned. A claim about
> recurrence needs a construct that measures recurrence.

### 6.3 Measured versus implied dwell

`state_dwell_times` returns both:

- the **measured** dwell — the mean run length of the Viterbi path, with runs
  counted inside a recording and never stitched across a recording boundary;
- the **implied** dwell `1/(1-A_{kk})` from the fitted transition matrix (a
  never-leaving state, `A_{kk}\ge1`, maps to `∞`).

They coincide when the geometric holding-time model fits, and diverge when the
decoded runs are more or less persistent than a first-order chain predicts. That
divergence is a real property of the data, so the figure plots both — bars for
measured, markers for implied — rather than reporting one number that hides it.

### 6.4 What the estimate needs

Dwell times are run-length statistics: they need long trajectories and they need
the seam gone. A three-minute recording at `f_{\text{win}}=6.25` Hz gives ~1,100
windows, enough for a stable mean run length at any state with non-trivial
occupancy; a single 16-window clip gives nothing. States with very high `A_{kk}`
(past ≈0.95) sit on the steep part of `1/(1-A_{kk})`, where a small change in the
fit moves the implied dwell by seconds — report those from the measured side.

---

## 7. The autoregressive HMM (AR-HMM)

An alternative emission model where each state is a **linear autoregressive**
Gaussian process — the behavioural-syllable model of MoSeq (Wiltschko et al.,
2015; Johnson et al., 2016):

$$
z_t = \sum_{p=1}^{P} A_k^{(p)} z_{t-p} + b_k + e,\qquad e\sim\mathcal N(0,\Sigma_k).
$$

A state is now a **dynamical regime** — how the latent *evolves* (decay,
oscillation, drift), captured by `A_k`, rather than where it sits. This matches
"kinds of movement" better than a static cluster and is especially natural on the
delta stream. The AR order `P` (`lags`) is selected jointly with `K` by held-out
predictive likelihood, so a higher order only wins if it generalises. EM is
performed by the `ssm` package (Linderman et al.; the SLDS/AR-HMM library), not
re-implemented. The AR-HMM returns the same `states`/`transition`/`occupancy`/
`dwell` interface as the Gaussian HMM, so the dwell-time summary, phenotype
features, and every figure run unchanged; only the decoded-state-appearance panel is
skipped (an AR state is dynamics, not a pose). It is slower (~5× per fit, and
unparallelised), so the practical recipe is to pick `K` on the fast Gaussian HMM
and then fit the AR-HMM at that `K` with a small lag sweep.

---

## 8. Interpretability figures — what they show and how they are produced

All state figures take the fitted model's `res` dict (`states`, `k`,
`transition`, `occupancy`, `f_win`, per-state parameters) and are produced by one
call, `run_hmm_report`, which also saves each figure. States are per-window
(Viterbi); "one dot per video" statistics use the per-recording blocks of the
Viterbi path.

1. **Transition matrix + stationary distribution.** `A` as a heat map; the
   stationary `ρ` as the left eigenvector of `A` for eigenvalue 1
   (`ρA=ρ`, normalised). The clinical signal for GMs lives in the transition
   structure and dwell times, not the state means.

2. **Per-subject occupancy and dwell heat maps.** From the Viterbi path, per
   recording: state occupancy `= \mathrm{bincount}(s)/W`, and mean dwell per state
   `=` mean within-recording run length `/ f_{\text{win}}` seconds (runs never
   counted across a recording boundary). Rows = subjects, columns = states. These
   two matrices are simultaneously the **interpretability figure** and the
   **phenotype feature matrix** (`[occupancy | dwell]`, `n_subjects × 2K`).

3. **Decoded state appearance** (pose stream, Gaussian HMM only). Build a constant
   latent block `Z_k=[μ_k,\dots,μ_k]`, push it through the frozen decoder, and
   render the mid-frame stick figure; the panel is titled with the state's mean
   dwell time. A state that decodes to nothing recognisable signals `K` too high.
   Skipped for delta and AR states (which are changes/dynamics, not a pose).

4. **State movement dynamics — quiver (Fig-3a analogue).** For each state, pool
   the **raw keypoint velocities** (`Δ`pose) of the frames assigned to that state
   (via the window→frame map of §8.6) and draw them as faint line segments
   emanating from an anchor pose, with a bold per-joint mean-velocity arrow. The
   anchor is the **per-state mean pose** for the pose stream (a state *is* a
   posture) and the **global mean pose** for the delta stream (a state is a
   movement, so a shared reference is correct — matching the paper's "centred on
   the group average position"). Velocities are clipped at a robust percentile and
   scaled skeleton-relative so panels are comparable. Panels are laid out with the
   last row centred.

5. **State movement dynamics — velocity boxplot (Fig-3b analogue).** For each
   recording and joint, mark the frames in that joint's **top-`ρ`** (default 10 %)
   by speed as "high-velocity"; for each `(state, body-group)`, the metric is the
   **percentage of that state's frames that are high-velocity, averaged over the
   group's joints** — one value per recording, drawn as grouped boxplots with a
   dot per recording. Body grouping is a knob: **regions** (head / arms / legs),
   **lateral** (left_arm / right_arm / left_leg / right_leg, coloured by side and
   hatched by limb) for left–right **asymmetry**, or **side** (whole left vs
   right). A quiet state reads low everywhere, a whole-body state high everywhere,
   a limb- or side-specific state shows one group elevated. Velocities are raw, so
   only the state *labels* come from the model and the figure is immune to any
   latent/stitching artefact.

6. **The window→frame map.** The frame span of every kept window is recomputed
   with the exact overlap-crop crop/keep logic, so each window's state maps to
   precisely the frames it was encoded from. The delta stream has one fewer state
   than windows, so each state is mapped to the first `W-1` window spans (the
   "from" window). This alignment underlies figures 4 and 5.

Supporting training-time figures: **β diagnostics** (β schedule, KL nats
train/val, reconstruction MSE), a **reconstruction video** (original vs. decoded
overlaid, FuncAnimation), a **latent traversal** (`z^*+α e_d` decoded across a
range of `α`), and the **Jacobian panels** of §3.2.

---

## 9. Phenotype features and clinical validation

### 9.1 Features

Per recording: the `K`-dim occupancy histogram and the `K`-dim mean-dwell vector —
assembled from the Viterbi path alone, never touching the label. Extra per-subject
covariates can be appended. Optionally clustered (TwoNN intrinsic dimension first;
if it exceeds the apparent cluster count the structure is treated as continuous
rather than partitioned; then a Gaussian mixture by BIC, silhouette reported
internally).

### 9.2 The clinical test (labels enter only here)

Labels are aligned to the kept-recording order by parsing subject IDs, with a
loud audit that the split is exactly 32/6 over 38 and that no positive subject was
dropped. Two families of contrast, each abnormal (`1`) vs. normal (`0`), both read
straight off the Viterbi path:
- **Per-state occupancy:** does the share of the recording spent in state `k`
  separate the groups? **Mann–Whitney U** (Mann & Whitney, 1947) with the AUC /
  rank-biserial effect size and an **exact** p-value (the exact rank-sum null;
  asymptotic only if ties force it).
- **Per-state mean dwell time:** does how long state `k` is held once entered
  separate them? Same test. A state that too few subjects in either group ever
  visit is reported as untestable rather than silently dropped.

That is `2K` tests over one dataset, so **Holm** step-down adjusted p-values are
reported next to the raw ones and every state is shown, not the best one. Each is
reported with **leave-one-subject-out** stability (the p and effect size
recomputed dropping each subject), and the plain caveat that with **6 positives**
the confidence intervals are wide and the analysis is exploratory — no classifier
accuracy is reported, and labels never enter any fit. **Polarity** (that `1`
encodes abnormal/absent-fidgety) is stated to be confirmed against the dataset's
own documentation, and the output prints the direction of every contrast. A null
result is reported as a finding, not hidden; so is a single state surviving Holm
out of `2K`, which is a lead rather than a result.

---

## 9bis. Limb co-movement: the cramped-synchronised construct

### 9bis.1 Why a separate construct is needed

**Cramped-synchronised** general movements are the abnormal form most strongly
associated with later cerebral palsy (Prechtl et al., 1997; Ferrari et al., 2002).
Their defining feature is **simultaneity**: limbs and trunk contract and relax
together instead of moving independently. This is a property of a *single
instant*, so it cannot be expressed by the ordering of the state sequence
(fluency) nor by its traversal (mixing) — both are properties of the *sequence*.
Clinical scoring further separates **consistent** from **intermittent**
cramped-synchronised movement, and only the consistent form carries the high risk,
so the construct must report a **distribution over the recording**, not a single
average.

The measure uses **raw keypoint displacements only**, so it is independent of the
encoder, the latent space, and the state model — a fully independent estimator
alongside the HMM's occupancy and dwell times (§6).

### 9bis.2 Signed, midline-reflected limb displacement

Speed discards direction, and direction is exactly what separates genuine
co-contraction from two limbs that merely move at the same instant, so the
**signed** displacement is kept. For the four limbs
`L ∈ {RA, LA, RL, LL}` given by the BODY-15 joint triples `{2,3,4}`, `{5,6,7}`,
`{9,10,11}`, `{12,13,14}`:

$$
\mathbf{w}^{(i)}_{L}(t) = \frac{1}{3}\sum_{j \in L} R_L
\bigl(x^{(i)}_{t+1,j} - x^{(i)}_{t,j}\bigr) \in \mathbb{R}^2 ,
\qquad
R_L = \begin{cases}\operatorname{diag}(-1, 1) & L \in \{\text{LA}, \text{LL}\},\\
I & L \in \{\text{RA}, \text{RL}\}.\end{cases}
$$

The reflection `R_L` changes coordinates **for each limb separately, before any
pair is formed**, so that the first axis points away from the body midline on that
limb's own side. Without it a bilaterally symmetric movement would register as
*opposition* rather than synchrony — verified numerically: a mirror-image
two-arm movement gives `D_{RA,LA} = +1` with the reflection and `−1` without.
The construct lives at **frame rate**: averaging signed displacements over time
would cancel the very oscillation it exists to detect.

### 9bis.3 Epochs and the co-movement matrix

Recording `i` is partitioned into `E_i` consecutive **non-overlapping epochs** of
`τ = 5` s, discarding any trailing remainder. At 25 fps an epoch holds 125 frames
— long enough that a single co-contraction does not fill an epoch, short enough to
stay within one bout of activity. Within epoch `e` the
frames are stacked into `W^{(i,e)}_L ∈ ℝ^{250}`, with energy
`a^{(i,e)}_L = \|W^{(i,e)}_L\|^2` and normalisation
`\widehat W = W/\|W\|`. The **co-movement matrix** is the cosine similarity

$$
D^{(i,e)}_{LL'}=\bigl\langle \widehat W^{(i,e)}_L,\widehat W^{(i,e)}_{L'}\bigr\rangle
=\frac{\sum_{t\in e}\bigl\langle \mathbf w^{(i)}_L(t),\mathbf w^{(i)}_{L'}(t)\bigr\rangle}
{\bigl(a^{(i,e)}_L a^{(i,e)}_{L'}\bigr)^{1/2}}\in[-1,1],
$$

a **Gram matrix of unit vectors**, hence positive semi-definite with unit diagonal
(verified: symmetry, unit diagonal, range, and non-negative eigenvalues). Its sign
is clinical: a **positive** entry means the two limbs move together and in the same
anatomical sense — the cramped-synchronised direction — and a **negative** entry
means they move in opposition, the reciprocal pattern of normal movement. A limb
with (near-)zero energy in an epoch has no direction, so entries involving it are
`NaN` rather than an arbitrary value.

### 9bis.4 The six pairs and their anatomical grouping

Each epoch yields the six off-diagonal entries, and the grouping is the
interpretive key:

| class | pairs | reading of a positive entry |
|---|---|---|
| homologous | RA–LA, RL–LL | both arms, or both legs, contracting together: the cramped-synchronised signature |
| same-side | RA–RL, LA–LL | arm and leg of one side moving as a unit |
| contralateral | RA–LL, LA–RL | diagonal coupling |

### 9bis.5 The figure

Six panels, one per pair (rows = anatomical class), each showing **every recording
as a strip of its epoch values**, coloured by diagnostic label, with a bar at the
per-recording median and a dashed zero line. **Within-infant spread separates
consistent from intermittent co-movement**, and between-infant differences are read
across strips, so consistency and magnitude appear in one display.

### 9bis.6 Inference

Group contrasts are made by **permuting the 38 diagnostic labels, never the
epochs** — epochs within a recording are not exchangeable, and the exchangeable
unit is the recording. The statistic for pair `p` is the per-recording median
`\widetilde D^{(i)}_p = \operatorname{med}_e D^{(i,e)}_p`, contrasted between
groups. The six pairs are tested **together under a maximum-statistic
(Westfall–Young) correction**: the null distribution is that of `\max_p |T_p|`
over label permutations, and each pair's corrected p-value is the tail of *that*
distribution at its observed statistic (so the corrected p is always ≥ the
uncorrected one). **The result is reported for all six pairs whatever any one
shows.** Because all six statistics live on the same `[-1,1]` scale, the maximum
is taken without standardisation. Validated on synthetic data: a planted
synchrony effect is recovered (`p_corr = 0.0025`) while a label-shuffled control is
not (`p_corr = 0.84`).



- **Repository layout.** `architectures/` (VAE models, config, training loop,
  losses, param counts); `youtube_motion/` (data loader, BODY-15 skeleton, sweep
  driver); `vae_analysis/` (`hmm_pipeline` — stitcher, seam diagnostic, HMM fit,
  dwell-time summary, movement-dynamics computation; `arhmm` — the ssm AR-HMM;
  `hmm_report` — the one-call report and all figures; plus the latent-geometry
  toolkit).
- **One-call report.** `run_hmm_report(adapter, videos, ..., model="hmm"|"arhmm",
  stream="pose"|"delta", velocity_grouping="regions"|"lateral"|"side")` runs
  stitch → seam check → fit → dwell times → phenotype features → all figures
  → optional clinical test, and saves a joblib bundle (`res`, `Z`, `lengths`, a
  version-proof `model_params` backup, and metadata) so a reload skips both the
  encode-stitch and the refit.
- **Reproducibility / dependencies.** `hmmlearn` (Gaussian HMM), `ssm`
  (AR-HMM; installed from source), `scipy`, `scikit-learn`; all fits are
  CPU-bound and seed-deterministic.

---

## 11. Honesty ledger (state these limitations in the thesis)

- **Small clinical cohort.** `n=38` with **6 positives**: every clinical result is
  exploratory, with wide CIs and reported LOO fragility. Window/clip counts are
  estimation resources, not sample size.
- **No rate or periodicity claim.** The state model supports dwell times and
  transition structure only. The earlier dwell→frequency map and its 0.5–2 Hz
  band are withdrawn (§6.2); nothing here measures recurrence.
- **Dwell resolution.** One window is 0.16 s at `l=4`, and states with
  `A_{kk}>0.95` sit where the implied dwell is very sensitive to the fit (§6.4).
- **Two constant joints** (MidHip, Neck) by normalisation — read the motion
  statistics and Jacobian rows with that in mind.
- **Unverified references:** the PRISM latent-motion work and the RVI-38
  introducing paper are cited at preparation-time confidence — verify before the
  bibliography. Verify also the page ranges of the older speech-recognition
  references (Rabiner; Gales).
- **Deprecated paths** not used in the final pipeline: the GM-VAE / GM-CVAE
  (component collapse) and the anchored/FiLM residual model (drove the residual to
  zero); retained in the code for the record.

---

## 12. References

*Verify entries marked **[verify]** against the primary source.*

**Variational autoencoders and training.**
- Kingma, D.P., Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.
- Rezende, D.J., Mohamed, S., Wierstra, D. (2014). Stochastic backpropagation and
  approximate inference in deep generative models. *ICML*.
- Higgins, I., et al. (2017). β-VAE: learning basic visual concepts with a
  constrained variational framework. *ICLR*.
- Bowman, S.R., et al. (2016). Generating sentences from a continuous space.
  *CoNLL*.
- Razavi, A., van den Oord, A., Poole, B., Vinyals, O. (2019). Preventing
  posterior collapse with δ-VAEs. *ICLR*.
- Kingma, D.P., et al. (2016). Improved variational inference with inverse
  autoregressive flow (free bits). *NeurIPS*.
- Fu, H., et al. (2019). Cyclical annealing schedule: a simple approach to
  mitigating KL vanishing. *NAACL*.
- Asperti, A., Trentin, M. (2020). Balancing reconstruction error and
  Kullback–Leibler divergence in variational autoencoders. *IEEE Access*.
- Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization
  (AdamW). *ICLR*.
- Perez, E., et al. (2018). FiLM: visual reasoning with a general conditioning
  layer. *AAAI*.

**Sequence models and motion.**
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Zheng, C., et al. (2021). 3D human pose estimation with spatial and temporal
  transformers (PoseFormer). *ICCV*.
- Zhang, J., et al. (2023). T2M-GPT: generating human motion from textual
  descriptions with discrete representations. *CVPR*.
- Jiang, B., et al. (2023). MotionGPT: human motion as a foreign language.
  *NeurIPS*.
- Ling, ... et al. PRISM: ... **[verify: authors, title, arXiv id, year]**.

**Hidden Markov models and covariance structure.**
- Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected
  applications in speech recognition. *Proc. IEEE* 77(2), 257–286. **[verify pages]**
- Baum, L.E., Petrie, T., Soules, G., Weiss, N. (1970). A maximization technique
  ... probabilistic functions of Markov chains. *Ann. Math. Stat.* 41(1), 164–171.
- Dempster, A.P., Laird, N.M., Rubin, D.B. (1977). Maximum likelihood from
  incomplete data via the EM algorithm. *JRSS B* 39(1), 1–38.
- Viterbi, A.J. (1967). Error bounds for convolutional codes and an asymptotically
  optimum decoding algorithm. *IEEE Trans. Inf. Theory* 13(2), 260–269.
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*, ch. 9, 13.
- Schwarz, G. (1978). Estimating the dimension of a model. *Ann. Stat.* 6(2), 461–464.
- Gales, M.J.F. (1999). Semi-tied covariance matrices for HMMs. *IEEE Trans.
  Speech Audio Process.* 7(3), 272–281. **[verify pages]**
- Ghahramani, Z., Hinton, G.E. (1996). The EM algorithm for mixtures of factor
  analysers. Tech. Rep. CRG-TR-96-1, Univ. Toronto.
- Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. *J. Multivar. Anal.* 88(2), 365–411.

**Autoregressive / behavioural state models.**
- Wiltschko, A.B., et al. (2015). Mapping sub-second structure in mouse behaviour.
  *Neuron* 88(6), 1121–1135.
- Johnson, M.J., Duvenaud, D., Wiltschko, A.B., Datta, S.R., Adams, R.P. (2016).
  Composing graphical models with neural networks for structured representations
  and fast inference (SLDS/AR-HMM). *NeurIPS*.
- Linderman, S., et al. `ssm`: Bayesian learning and inference for state space
  models. Software. **[verify citation form]**

**Latent geometry, prior tests, intrinsic dimension.**
- Arvanitidis, G., Hansen, L.K., Hauberg, S. (2018). Latent space oddity: on the
  curvature of deep generative models. *ICLR*.
- Shao, H., Kumar, A., Fletcher, P.T. (2018). The Riemannian geometry of deep
  generative models. *CVPR Workshops*.
- Gretton, A., et al. (2012). A kernel two-sample test (MMD). *JMLR* 13, 723–773.
- Facco, E., d'Errico, M., Rodriguez, A., Laio, A. (2017). Estimating the
  intrinsic dimension of datasets by a minimal neighborhood information (TwoNN).
  *Scientific Reports* 7, 12140.
- Lopez-Paz, D., Oquab, M. (2017). Revisiting classifier two-sample tests. *ICLR*.

**Spectral estimation and statistics.**
- Welch, P.D. (1967). The use of fast Fourier transform for the estimation of
  power spectra. *IEEE Trans. Audio Electroacoust.* 15(2), 70–73.
- Mann, H.B., Whitney, D.R. (1947). On a test of whether one of two random
  variables is stochastically larger than the other. *Ann. Math. Stat.* 18(1), 50–60.

**Pose estimation and the clinical construct/dataset.**
- Cao, Z., Simon, T., Wei, S.-E., Sheikh, Y. (2017). Realtime multi-person 2D pose
  estimation using part affinity fields (OpenPose). *CVPR*.
- Einspieler, C., Prechtl, H.F.R. (2005). Prechtl's assessment of general
  movements ... *Ment. Retard. Dev. Disabil. Res. Rev.* 11(1), 61–67.
- Prechtl, H.F.R. (1990). Qualitative changes of spontaneous movements ... *Early
  Hum. Dev.* 23(3), 151–158.
- Prechtl, H.F.R., Einspieler, C., Cioni, G., Bos, A.F., Ferrari, F., Sontheimer,
  D. (1997). An early marker for neurological deficits after perinatal brain
  lesions. *The Lancet* 349(9062), 1361–1363. **[verify]** *(cramped-synchronised
  GMs and cerebral palsy)*
- Ferrari, F., Cioni, G., Einspieler, C., et al. (2002). Cramped synchronized
  general movements in preterm infants as an early marker for cerebral palsy.
  *Archives of Pediatrics & Adolescent Medicine* 156(5), 460–467. **[verify]**
- Westfall, P.H., Young, S.S. (1993). *Resampling-Based Multiple Testing:
  Examples and Methods for p-Value Adjustment*. Wiley. *(maximum-statistic
  correction)*
- McCay, K.D., Ho, E.S.L., Shum, H.P.H., et al. Pose-based feature fusion for the
  early prediction of cerebral palsy in infants; introduction of the RVI-38
  dataset. **[verify: exact title, venue, year]**
