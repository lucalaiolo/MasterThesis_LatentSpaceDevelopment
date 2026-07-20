# HMM Analysis of the Temporal VAE Latent (RVI-38) — Claude Code Build Document

> Purpose: a self-contained specification for building the hidden-Markov-model
> interpretability layer on top of the temporal-latent motion VAE, run on the
> RVI-38 neonatal dataset. It states the mathematics the implementation must
> respect, the emission-covariance decision and its justification, the
> clip-stitching correction the fixed 64-frame VAE input forces, the
> fidgety-band frequency argument, and the guardrails that keep the statistics
> honest. Claude Code implements the pipeline autonomously against this
> document; the numbered guardrails in each section are binding.
>
> **Implementation:** `vae_analysis/hmm_pipeline.py` implements this spec against
> the frozen temporal VAE, model- and dataset-agnostic (any `temporal_conv` /
> `temporal_transformer` checkpoint + any video list). RVI-38 plugs in where the
> synthetic clips do in the headless tests.

---

## 0. Context and fixed facts

The VAE is unchanged. It consumes a clip of $T = 64$ frames of 2D pose,
$(F=64, J=18, D=2)$, and emits a latent block of $W = 16$ windows by $d = 16$
channels, $Z \in \mathbb{R}^{16 \times 16}$. The temporal downsample is therefore
$l = T / W = 4$, one window spanning four frames. We do not modify the VAE, the
clip length, or the latent geometry; this document builds analysis on top of the
frozen encoder and decoder.

The dataset for this thread is RVI-38, collected as part of routine clinical care
at the Royal Victoria Infirmary, Newcastle, and introduced with the Newcastle
pose feature-fusion work (McCay et al.; see references). It contains 38 RGB
sequences of different infants aged 12 to 21 weeks, recorded top-down on a
handheld Sony DSC-RX100 at 1920x1080 and 25 fps, with durations from 40 seconds
to five minutes (mean three minutes 36 seconds), each carrying a single General
Movements Assessment label: presence (normal) or absence (abnormal) of fidgety
movements. The 12-to-21-week window coincides with Prechtl's fidgety period
(roughly 9 to 20 weeks), so the recordings capture the movement the frequency
analysis targets, and the label is the fidgety construct itself rather than a
proxy.

Two frame rates appear below. The VAE was trained at the pipeline rate
$f_{\text{frame}}$ (30 fps in the current pipeline; use whatever the frozen model
was trained on). The raw-keypoint frequency analysis of Section 4 runs at the
native 25 fps, since that path never enters the VAE and interpolation should not
be introduced where it is avoidable. Set the window rate as

$$
f_{\text{win}} = \frac{f_{\text{frame}}}{l} \ \text{Hz},
$$

which is 7.5 Hz at 30 fps and 6.25 Hz at 25 fps, both with $l = 4$. Every
frequency claim below is parameterised in $f_{\text{win}}$; state the value used.

---

## 1. Data budget and why 38 videos is enough for the fit

Total footage averages $38 \times 216\ \text{s} \approx 8{,}200\ \text{s}$, so
RVI-38 trades subject count for video length and reaches roughly the footage of
the 85-video Chambers set. At $l = 4$ and 30 fps the window budget is
$\text{footage} \times f_{\text{win}} \approx 8{,}200 \times 7.5 \approx 61{,}000$
windows over full footage, or about 40,000 taking 65% as active (GMA recordings
are mostly awake by protocol). This is roughly 2,500 to 4,000 non-overlapping
64-frame clips.

The consequence for estimation: with $K = 6$ states and a genuinely rare state at
5% occupancy, the thinnest emission sees $0.05 \times 40{,}000 = 2{,}000$ windows
against the 136 parameters of a full $16 \times 16$ covariance, a ratio near
$15{:}1$. Even a 3%-occupancy state clears $8{:}1$. Full-covariance emissions are
comfortably estimable.

> **Guardrail 1.1 (pseudoreplication).** The window count is an estimation
> resource, not a sample size. There are 38 independent subjects. The windows
> stabilise the emission covariances; they do not supply 40,000 independent
> observations. For any inferential claim (a state associating with abnormality,
> a cluster separating outcomes) the effective sample size is $n = 38$. Never
> compute a p-value or a classifier accuracy that treats windows or clips as
> independent units.

---

## 2. Mathematical background

This section is reference. It fixes what the HMM is, so the implementation and the
diagnostics below have precise objects to name.

### 2.1 The Gaussian HMM

For one window sequence let $Z = (z_1, \dots, z_W)$ with $z_w \in \mathbb{R}^{d}$.
Introduce a hidden state $s_w \in \{1, \dots, K\}$ per window, with

$$
\pi_k = \Pr(s_1 = k), \qquad A_{jk} = \Pr(s_{w+1}=k \mid s_w = j), \qquad
p(z_w \mid s_w = k) = \mathcal{N}(z_w;\, \mu_k, \Sigma_k),
$$

where $\pi \in \Delta^{K-1}$ and each row of $A$ lies on the simplex. The model
rests on two conditional-independence assumptions: the state chain is first-order
Markov, $s_{w+1} \perp s_{1:w-1} \mid s_w$, and each observation depends only on
its own state, $z_w \perp \{z_{\neq w}, s_{\neq w}\} \mid s_w$. The joint density
factorises as

$$
p(Z, S \mid \theta) = \pi_{s_1}\, \mathcal{N}(z_1; \mu_{s_1}, \Sigma_{s_1})
\prod_{w=2}^{W} A_{s_{w-1}, s_w}\, \mathcal{N}(z_w; \mu_{s_w}, \Sigma_{s_w}),
\qquad \theta = (\pi, A, \{\mu_k, \Sigma_k\}_{k=1}^{K}).
$$

### 2.2 Likelihood over multiple sequences

With $N$ sequences $Z^{(1)}, \dots, Z^{(N)}$ independent given $\theta$, the
objective is

$$
\ell(\theta) = \sum_{n=1}^{N} \log p(Z^{(n)} \mid \theta), \qquad
p(Z^{(n)} \mid \theta) = \sum_{S} p(Z^{(n)}, S \mid \theta).
$$

Each sequence carries its own $s_1 \sim \pi$, so the log-likelihoods are summed
rather than the windows concatenated. This is the formal content of the
`lengths` argument: concatenation would insert a transition
$A_{s_{W}^{(n)},\, s_{1}^{(n+1)}}$ across a boundary that does not exist in the
data.

### 2.3 Forward-backward

Define the forward variable $\alpha_w(k) = p(z_{1:w}, s_w = k \mid \theta)$.

> **Lemma 2.1 (forward recursion).**
> $$\alpha_1(k) = \pi_k\, \mathcal{N}(z_1; \mu_k, \Sigma_k),$$
> $$\alpha_w(k) = \mathcal{N}(z_w; \mu_k, \Sigma_k)\sum_{j=1}^{K}
> \alpha_{w-1}(j)\, A_{jk}, \qquad w = 2, \dots, W.$$
>
> *Proof.* Marginalise the previous state and factor:
> $\alpha_w(k) = \sum_j p(z_w \mid s_w = k)\,\Pr(s_w = k \mid s_{w-1} = j)\,
> p(z_{1:w-1}, s_{w-1} = j)$, where the observation term detaches by
> $z_w \perp z_{1:w-1} \mid s_w$ and the transition by the Markov property.
> $\blacksquare$

Symmetrically $\beta_w(k) = p(z_{w+1:W} \mid s_w = k, \theta)$ satisfies
$\beta_W(k) = 1$ and $\beta_w(k) = \sum_j A_{kj}\,
\mathcal{N}(z_{w+1}; \mu_j, \Sigma_j)\,\beta_{w+1}(j)$, and the likelihood closes
as $p(Z \mid \theta) = \sum_k \alpha_W(k)$. The posterior responsibilities are

$$
\gamma_w(k) = \frac{\alpha_w(k)\beta_w(k)}{\sum_j \alpha_w(j)\beta_w(j)},
\qquad
\xi_w(j,k) = \frac{\alpha_w(j)\, A_{jk}\,
\mathcal{N}(z_{w+1}; \mu_k, \Sigma_k)\, \beta_{w+1}(k)}{p(Z \mid \theta)}.
$$

Run the recursions in log-space or with per-column scaling to avoid underflow.

### 2.4 EM (Baum-Welch)

Each EM iteration maximises a Jensen lower bound on $\ell$ that touches it at the
current iterate, so $\ell$ does not decrease and the sequence converges to a
stationary point. Summing responsibilities over sequences $n$ and windows $w$,
the M-step is

$$
\hat\pi_k = \frac{1}{N}\sum_{n}\gamma_1^{(n)}(k), \qquad
\hat A_{jk} = \frac{\sum_n\sum_{w=1}^{W_n-1}\xi_w^{(n)}(j,k)}
{\sum_n\sum_{w=1}^{W_n-1}\gamma_w^{(n)}(j)},
$$

$$
\hat\mu_k = \frac{\sum_n\sum_w \gamma_w^{(n)}(k)\, z_w^{(n)}}
{\sum_n\sum_w \gamma_w^{(n)}(k)}, \qquad
\hat\Sigma_k = \frac{\sum_n\sum_w \gamma_w^{(n)}(k)\,
\big(z_w^{(n)} - \hat\mu_k\big)\big(z_w^{(n)} - \hat\mu_k\big)^{\!\top}}
{\sum_n\sum_w \gamma_w^{(n)}(k)}.
$$

The last expression is the full-covariance estimate used here (Section 2.6). Local
optima are real; use several k-means-seeded restarts and keep the best training
$\ell$.

### 2.5 Decoding

Per-window labels for rendering and for dwell statistics come from the MAP path
$S^\star = \arg\max_S p(Z, S \mid \theta)$ by the Viterbi recursion, not from the
marginal $\arg\max_k \gamma_w(k)$. Viterbi respects $A$ and yields a coherent
trajectory; the marginal argmax can flicker between adjacent windows and would
corrupt dwell-time estimates.

> **Guardrail 2.2 (dwell statistics from Viterbi only).** All dwell-time and
> transition-count statistics derive from the Viterbi path. The marginal
> responsibilities are for soft diagnostics only.

### 2.6 Emission covariance: the decision

Fit `covariance_type="full"` with a ridge floor. The justification is structural,
not a tuning choice, and it should be reported as fixed a priori.

> **Proposition 2.3 (affine invariance of the full-covariance HMM).** Let
> $z \mapsto \tilde z = A z + b$ for invertible $A$. The full-covariance Gaussian
> HMM fit is invariant: the maximiser transforms as
> $\mu_k \mapsto A^{-1}(\mu_k - b)$ and
> $\Sigma_k \mapsto A^{-1}\Sigma_k A^{-\top}$, the responsibilities
> $\gamma, \xi$ and the Viterbi path are unchanged, and the log-likelihood shifts
> by the constant $-M \log\lvert\det A\rvert$, which cancels in every model
> comparison ($M = \sum_n W_n$).
>
> *Proof.* Under an invertible affine map a Gaussian density transforms as
> $\mathcal{N}(\tilde z; A\mu_k + b, A\Sigma_k A^\top) =
> \lvert\det A\rvert^{-1}\,\mathcal{N}(z; \mu_k, \Sigma_k)$. Substituting into the
> joint of Section 2.1, every emission contributes one factor
> $\lvert\det A\rvert^{-1}$, giving $\lvert\det A\rvert^{-M}$ overall; $\pi$ and
> $A_{jk}$ are unaffected. The responsibilities are ratios in which the constant
> cancels, so $\gamma, \xi$ and the argmax path are identical, and the
> log-likelihood changes by the additive constant only. $\blacksquare$

A whitening transform is one such $A$, so under full covariance whitening is a
no-op: same fit, rotated axes. This is why the covariance question and the
whitening question are one question. Under a **diagonal** emission model the
constraint "$\Sigma_k$ diagonal" is not preserved by rotation, so whitening
becomes a genuine modelling device, but it then diagonalises the wrong object.
Writing the pooled covariance by the law of total covariance,

$$
C = \underbrace{\sum_k \rho_k \Sigma_k}_{\Sigma_W}
\;+\; \underbrace{\sum_k \rho_k (\mu_k - \bar\mu)(\mu_k - \bar\mu)^\top}_{\Sigma_B},
\qquad \bar\mu = \sum_k \rho_k \mu_k,
$$

with $\rho_k$ the stationary occupancy, PCA diagonalises $C$ whereas a diagonal
emission model needs $\Sigma_W$ diagonalised, and these coincide only when
$[\Sigma_W, \Sigma_B] = 0$, which does not hold generically. Full covariance makes
the constraint rotation-invariant and removes the problem rather than patching it.
At $d = 16$ the 136 parameters per state are affordable given the budget of
Section 1, so there is no reason to accept the diagonal model's misspecification.

**Regularisation.** The failure mode of full covariance is a thin state whose
matrix goes ill-conditioned. Two levels of defence:

1. Ridge floor. Add $\varepsilon I$ to each covariance every M-step
   (hmmlearn `min_covar`). Tune $\varepsilon$ up from the $10^{-3}$ default while
   watching the logged per-state condition number.
2. Shrinkage, if a trigger fires. Replace the M-step estimate by
   $(1-\lambda_k)\hat\Sigma_k + \lambda_k \nu_k I$, shrinking low-occupancy
   states harder toward a scaled identity, with $\lambda_k$ and $\nu_k$ from the
   Ledoit-Wolf closed form. This is a small custom M-step over hmmlearn.

> **Guardrail 2.4 (covariance triggers, logged every fit).** Log, per state and
> per fit, the minimum occupancy $\rho_{\min} M$ and the covariance condition
> number. Enable shrinkage only when $\rho_{\min} M$ falls below roughly ten
> times 136 or the condition number exceeds a preset ceiling. Report the
> covariance family as fixed a priori on Proposition 2.3, not tuned.

**Fallback ladder**, in decreasing convenience, engaged only if a state genuinely
cannot support 136 parameters after regularisation:

- Tied covariance: one shared full matrix across states (hmmlearn
  `covariance_type="tied"`), 136 parameters total, states differing in means.
- Semi-tied covariance (MLLT): a shared global linear transform plus per-state
  diagonals, learned under the likelihood (Gales 1999). Not in hmmlearn.
- Factor-analysed emissions: per-state $\Sigma_k = \Lambda_k\Lambda_k^\top +
  \Psi_k$ with $\Lambda_k \in \mathbb{R}^{d \times q}$, $q \ll d$
  (Ghahramani and Hinton 1996). Custom.

At $d = 16$ with this budget the expectation is that the ladder stays at level
zero; it is documented as the reasoned answer to "why not something cheaper," not
as an anticipated need.

### 2.7 Model selection

With diagonal covariance the free-parameter count is
$p(K) = (K-1) + K(K-1) + Kd + Kd$; full covariance replaces the last $Kd$ with
$K\,d(d+1)/2$, so

$$
\text{BIC}(K) = -2\,\ell(\hat\theta_K) + p(K)\log M, \qquad M = \sum_n W_n.
$$

BIC's penalty assumes independent samples, and HMM windows are dependent, so the
effective sample size sits between $N$ and $M$ and BIC is a heuristic here. Select
$K$ by held-out log-likelihood on the video-wise split; show BIC alongside.

> **Guardrail 2.5 (selection and splitting).** Model selection uses held-out
> log-likelihood under a video-wise (subject-disjoint) split. With 38 videos a
> single 15% split leaves roughly six test videos and is noisy, so use
> leave-one-video-out or average over several seeded splits. Never select or
> validate on a clip-level split; a clip-level split leaks subjects across the
> boundary and is a hard failure.

---

## 3. The clip-stitching problem and the per-video trajectory

The VAE sees only 64 frames, so there is no primitive that encodes a whole video.
A recording becomes a run of consecutive 64-frame clips whose $(16, 16)$ blocks
are concatenated. The count is unchanged by this, but the continuity is not free.

**The seam.** Each clip is an independent forward pass with no cross-clip memory.
The pose stream abuts cleanly because the four-step normalisation runs per video,
but the encoder resets at every clip boundary, so two independently-encoded clips
can place the same motion at different latent locations, and window 16 of one clip
need not join smoothly to window 1 of the next. With non-overlapping 64-frame
clips this places a discontinuity every 64 frames, a periodic artifact at
$f_{\text{frame}}/64$ with harmonics. At 30 fps that comb sits at 0.47, 0.94,
1.41, and 1.88 Hz; at 25 fps at 0.39, 0.78, 1.17, 1.56, and 1.95 Hz. Both land
inside the 0.5-to-2 Hz fidgety band. Naive concatenation therefore injects fake
spectral power exactly where the band-ratio of Section 4 reads, and a fake
transition into the HMM exactly every 16 windows.

**The fix: overlap-tile with centre-crop.** Encode clips at stride 32 frames (50%
overlap) and from each clip keep only the central eight windows (indices 4 to 11),
discarding the four at each edge. Consecutive kept regions tile the recording with
no gap and no overlap, every retained window carries at least 16 frames of
intra-clip context on each side, and because half the windows are kept from twice
as many clips the trajectory length is identical to non-overlapping tiling, at
twice the encoding cost. The periodic seam and its comb disappear. Stride 48
keeping the central 12 windows is the cheaper, less smooth alternative.

**Diagnostic.** After stitching, plot the concatenated trajectory and its power
spectral density per active dimension and confirm no spike at $f_{\text{frame}}/64$
or its harmonics. This check gates the per-video `lengths` construction.

> **Guardrail 3.1 (no naive concatenation with per-video lengths).** Do not set
> per-video `lengths` on naively concatenated windows. Either use the
> overlap-crop stitcher above, or, if clips are tiled without overlap, exclude
> every clip-boundary transition from the transition-count sums of Section 2.4
> and from the dwell counts. Otherwise the HMM learns a spurious 16-window rhythm
> and every dwell-time estimate inherits it.

---

## 4. The fidgety-band frequency argument

### 4.1 Rates

From Section 0, $f_{\text{win}} = f_{\text{frame}}/l$, one window spanning
$l/f_{\text{frame}}$ seconds. The Nyquist ceiling of the window trajectory is
$f_{\text{win}}/2$, which is 3.75 Hz at 30 fps and 3.1 Hz at 25 fps, so the
0.5-to-2 Hz band lies in range at either rate.

### 4.2 Dwell time to frequency

In an HMM the time in state $k$ before leaving is geometric,
$\Pr(\text{dwell} = m) = A_{kk}^{m-1}(1 - A_{kk})$, with mean

$$
\bar\tau_k = \frac{1}{1 - A_{kk}} \ \text{windows} = \frac{\bar\tau_k}{f_{\text{win}}}
\ \text{seconds}.
$$

An oscillatory movement swings between two configurations, a flexed state and an
extended state, so a full period visits both and one cycle is two dwells, not
one. With symmetric half-cycle dwells of mean $\bar\tau$ windows the fundamental
frequency is

$$
f = \frac{f_{\text{win}}}{2\,\bar\tau} \quad\Longleftrightarrow\quad
\bar\tau = \frac{f_{\text{win}}}{2f}, \qquad
A_{kk} = 1 - \frac{2f}{f_{\text{win}}}.
$$

The factor of two is the correction: a periodic motion reaches its extremes at
rate $2f$, so state switches occur at $2f$ and dwell scales as $1/(2f)$.

### 4.3 The $A_{kk}$ map

Mapping the band edges and the midpoint through
$A_{kk} = 1 - 2f/f_{\text{win}}$:

| motion $f$ | $\bar\tau$, 30 fps (win) | $A_{kk}$, 30 fps | $\bar\tau$, 25 fps (win) | $A_{kk}$, 25 fps |
|---|---|---|---|---|
| 0.5 Hz | 7.50 | 0.87 | 6.25 | 0.84 |
| 1.0 Hz | 3.75 | 0.73 | 3.13 | 0.68 |
| 2.0 Hz | 1.88 | 0.47 | 1.56 | 0.36 |

A state whose fitted self-transition sits in the band's range (roughly 0.47 to
0.87 at 30 fps, 0.36 to 0.84 at 25 fps) has a dwell distribution consistent with
fidgety-band motion. Take the exact clinical band edges from Einspieler and
Prechtl, not from this table; the table supplies the map, not the cut-offs.

### 4.4 The band-power ratio, on the raw signal first

Rather than infer frequency only from dwell times, measure it directly. Compute
the power spectral density $P(f)$ of the motion signal by Welch's method and
define a fidgety-band ratio

$$
\text{FBR} = \frac{\int_{0.5}^{2} P(f)\,df}{\int_{0}^{f_{\text{win}}/2} P(f)\,df}.
$$

Compute the primary FBR on the **raw keypoint velocities**, which are continuous
across the whole recording with no encoder seams, at native 25 fps. Then, as
corroboration, compute band power on the seam-handled latent trajectory. The two
paths are independent: a seam artifact in the latent can never silently corrupt
the headline figure, because the headline figure comes from the raw signal.
Agreement between the two is the informative result, since it shows the latent
captured the clinically defined dynamics rather than the analysis having imposed
them. This is the same two-estimator logic used elsewhere in the thesis, placed
where it protects the primary number.

> **Guardrail 4.1 (decouple the clinical frequency from the latent).** The
> reported fidgety-band frequency feature is the raw-velocity FBR. The latent
> band power and the HMM dwell-frequency are corroborating estimators, not the
> headline. If they disagree, report the disagreement rather than selecting the
> favourable one.

### 4.5 Resolution

Representing a switch rate of $2f$ needs $f_{\text{win}} \gg 4f$. At the top of
the band, $f = 2$ Hz with $f_{\text{win}} = 7.5$ Hz gives a ratio of 1.9, which
is marginal, so dwell-based estimates near 2 Hz are weak and should be read
alongside the spectral FBR rather than alone. If fidgety fidelity at the top of
the band becomes critical, a future VAE at $l = 2$ ($f_{\text{win}} = 15$ Hz)
doubles the windows per cycle; note this as a design lever, do not retrain here.

The periodogram resolution of a single 16-window clip is
$\Delta f = f_{\text{win}}/W \approx 0.47$ Hz, far too coarse to resolve the band.
This is why the frequency analysis runs on the concatenated per-video trajectory,
where a three-minute recording gives on the order of 1,600 windows and
$\Delta f \approx 0.005$ Hz. Long trajectories are what make both the spectral and
the dwell estimators trustworthy, and RVI-38 supplies them where short clips did
not.

---

## 5. Implementation plan

Everything below runs on the frozen VAE through the existing
`ArchitecturesAdapter`. Use posterior means throughout.

> **Guardrail 5.0 (means, not samples).** Feed the encoder posterior mean $\mu$
> into every downstream step. Never feed sampled $z$; the sampling variance would
> enter the state and cluster geometry for no reason.

### 5.1 `encode_window_sequence(video) -> (M_v, d)`

Build the per-video trajectory by the overlap-crop stitcher of Section 3: encode
64-frame clips at stride 32, keep the central eight windows of each
$(16, 16)$ block, and concatenate in temporal order into $(M_v, 16)$ for the
video. Return, per video, the trajectory and its window count $M_v$. Assemble the
dataset trajectory as the vertical stack over videos with a `lengths` array
$[M_1, \dots, M_{38}]$. Run the Section 3 seam diagnostic and refuse to proceed if
a $f_{\text{frame}}/64$ spike survives.

Provide a switch to emit either the pose-state stream $z_w$ or the change stream
$\Delta z_w = z_{w+1} - z_w$. Emissions on $z_w$ describe kinds of pose; emissions
on $\Delta z_w$ describe kinds of change, which sits closer to the fidgety
construct at the cost of states that no longer decode to a single renderable pose.
Fit both and report which separates subjects better.

### 5.2 HMM fitting

Standardise each dimension to unit scale before fitting (for optimiser
conditioning and k-means seeding only; by Proposition 2.3 the full-covariance fit
is invariant to it). Fit `GaussianHMM(covariance_type="full")` with the
`lengths` array, the ridge floor and shrinkage triggers of Guardrail 2.4, and
several k-means-seeded restarts keeping the best training log-likelihood. Sweep
$K = 2, \dots, 10$ and select by Guardrail 2.5. Decode with Viterbi
(Guardrail 2.2).

### 5.3 Interpretability outputs

For each retained model produce, in order of value:

1. **Decoded state appearance.** For state $k$, construct a constant latent
   block $Z_k = [\mu_k, \dots, \mu_k] \in \mathbb{R}^{16 \times 16}$, decode
   through the frozen decoder to 64 frames, and render the pose sequence. A state
   that decodes to nothing recognisable is a sign $K$ is too high. For a
   difference-stream model, render the integrated trajectory rather than a single
   pose.
2. **Transition matrix $A$ and stationary distribution $\rho$**, with $\rho$ the
   left eigenvector of $A$ for eigenvalue one. The clinical signal for GMs lives
   in dwell times and transition structure, not in the state means.
3. **Per-video state-occupancy histogram and mean dwell time per state**, from
   the Viterbi path. These are the phenotype features of Section 5.4.
4. **Frequency labelling.** Map each state's fitted $A_{kk}$ through Section 4.3
   and flag the states whose implied frequency lies in band.

### 5.4 Phenotype clustering

Build one feature vector per video: the $K$-dimensional state-occupancy
histogram, the $K$-dimensional mean-dwell vector, and the scalar raw-velocity FBR.
Run TwoNN on the 38 vectors first; if the intrinsic dimension exceeds the number
of apparent clusters, treat the structure as continuous rather than forcing a
partition. Then cluster (GMM or k-means, with spectral clustering on an
energy-distance kernel as an alternative), reporting silhouette internally and
ARI or AMI against the fidgety label post hoc.

> **Guardrail 5.1 (small-$n$ honesty).** With $n = 38$ this analysis is
> exploratory. Report effect sizes, permutation-test significance, and
> leave-one-video-out stability. Do not report a classifier accuracy or claim
> discovered phenotypes from 38 vectors. Labels never enter the clustering fit.

### 5.5 Clinical validation

The label is fidgety presence itself, so it is the direct test of the frequency
work. Primary test: does the raw-velocity FBR separate normal from abnormal
videos (Mann-Whitney U, effect size, permutation null)? Secondary: does
fidgety-band-state occupancy separate them? The headline result is the agreement
of the three estimators, raw-velocity FBR, latent band power, and HMM
dwell-frequency, on which videos carry in-band motion.

> **Guardrail 5.2 (report the negative).** If the FBR and the fidgety-band states
> do not separate the two label groups, report it as a finding. The thesis frames
> what works and states the limits of what does not; a null clinical
> correspondence is a result, not a failure to be hidden.

---

## 6. Failure modes to guard against (summary)

- Sampled $z$ instead of posterior means in any downstream step (Guardrail 5.0).
- Marginal-argmax decoding instead of Viterbi for dwell statistics
  (Guardrail 2.2).
- Naive clip concatenation with per-video `lengths`, injecting a 16-window rhythm
  and a $f_{\text{frame}}/64$ spectral comb in the fidgety band (Guardrail 3.1).
- Clip-level train/test splits, leaking subjects (Guardrail 2.5).
- Treating window or clip counts as sample size; $n = 38$ for inference
  (Guardrail 1.1, 5.1).
- Reading the latent band power as the headline frequency figure instead of the
  raw-velocity FBR (Guardrail 4.1).
- Presenting the full-covariance choice as tuned rather than fixed a priori on
  Proposition 2.3 (Guardrail 2.4).
- Reducing dimensionality by top variance if any reduction is attempted; variance
  rank is not discriminability rank, and full covariance needs no reduction in
  the first place (Section 2.6).

---

## 7. References

Method and model.

- Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected
  applications in speech recognition. *Proceedings of the IEEE* 77(2), 257-286.
- Baum, L.E., Petrie, T., Soules, G., Weiss, N. (1970). A maximization technique
  occurring in the statistical analysis of probabilistic functions of Markov
  chains. *Annals of Mathematical Statistics* 41(1), 164-171.
- Dempster, A.P., Laird, N.M., Rubin, D.B. (1977). Maximum likelihood from
  incomplete data via the EM algorithm. *Journal of the Royal Statistical
  Society B* 39(1), 1-38.
- Viterbi, A.J. (1967). Error bounds for convolutional codes and an
  asymptotically optimum decoding algorithm. *IEEE Transactions on Information
  Theory* 13(2), 260-269.
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*, ch. 9 and 13.
  Springer.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*
  6(2), 461-464.

Covariance structure.

- Gales, M.J.F. (1999). Semi-tied covariance matrices for hidden Markov models.
  *IEEE Transactions on Speech and Audio Processing* 7(3), 272-281.
- Ghahramani, Z., Hinton, G.E. (1996). The EM algorithm for mixtures of factor
  analysers. Technical Report CRG-TR-96-1, University of Toronto.
- Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. *Journal of Multivariate Analysis* 88(2), 365-411.
- Davis, S.B., Mermelstein, P. (1980). Comparison of parametric representations
  for monosyllabic word recognition in continuously spoken sentences. *IEEE
  Transactions on Acoustics, Speech, and Signal Processing* 28(4), 357-366.

HMM over a learned latent, behavioural analogues.

- Johnson, M.J., Duvenaud, D., Wiltschko, A.B., Datta, S.R., Adams, R.P. (2016).
  Composing graphical models with neural networks for structured representations
  and fast inference. *Advances in Neural Information Processing Systems*.
- Wiltschko, A.B., Johnson, M.J., Iurilli, G., Peterson, R.E., Katon, J.M.,
  Pashkovski, S.L., Abraira, V.E., Adams, R.P., Datta, S.R. (2015). Mapping
  sub-second structure in mouse behaviour. *Neuron* 88(6), 1121-1135.

Spectral estimation.

- Welch, P.D. (1967). The use of fast Fourier transform for the estimation of
  power spectra: a method based on time averaging over short, modified
  periodograms. *IEEE Transactions on Audio and Electroacoustics* 15(2), 70-73.

Clinical construct and dataset.

- Einspieler, C., Prechtl, H.F.R. (2005). Prechtl's assessment of general
  movements: a diagnostic tool for the functional assessment of the young nervous
  system. *Mental Retardation and Developmental Disabilities Research Reviews*
  11(1), 61-67.
- Prechtl, H.F.R. (1990). Qualitative changes of spontaneous movements in fetus
  and preterm infant are a marker of neurological dysfunction. *Early Human
  Development* 23(3), 151-158.
- McCay, K.D., Ho, E.S.L., Shum, H.P.H., Fehringer, G., Marcroft, C., Embleton,
  N.D. Pose-based feature fusion and classification for the early prediction of
  cerebral palsy in infants, and the introduction of the RVI-38 dataset (Newcastle
  University). Verify the exact title, venue, and year against the published
  record before citing.

Software.

- hmmlearn (scikit-learn-contrib), `GaussianHMM`.

> Reference note: authors, years, titles, and venues are given at the confidence
> level reached in preparing this document. Verify the page ranges on the older
> speech-recognition references (Rabiner, Davis and Mermelstein, Gales) and the
> full bibliographic record of the RVI-38 introducing paper against the originals
> before they enter the thesis bibliography.
