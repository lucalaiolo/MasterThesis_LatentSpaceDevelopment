# Pose stream vs. delta stream for the HMM (`STREAM = "pose" | "delta"`)

> What the `stream` switch in `encode_window_sequence` / `stitch_dataset` does,
> the mathematics behind it, and why the **delta** stream may be closer to the
> fidgety construct than the pose stream. Build-doc reference: `HMM_RVI38_BUILD.md`
> §5.1 ("emit either the pose-state stream $z_w$ or the change stream $\Delta z_w$").

---

## 1. The two streams

The temporal VAE gives a per-window latent trajectory for a video,
$z_1, z_2, \dots, z_M$ with $z_w \in \mathbb{R}^{d}$ ($d = $ `LATENT_DIM`, e.g. 8),
sampled at the window rate $f_{\text{win}} = f_{\text{frame}}/l$ (6.25 Hz at 25 fps,
$l=4$). The HMM is fit on one of two derived sequences:

$$
\textbf{pose:}\quad x_w = z_w
\qquad\qquad
\textbf{delta:}\quad x_w = \Delta z_w = z_{w+1} - z_w .
$$

`stream="delta"` is literally `np.diff` of the (seam-free, overlap-crop) trajectory,
so it has $M-1$ points and is still sampled at $f_{\text{win}}$. Because the
difference is taken **after** the seam-free stitch, no artificial jump is
introduced at clip boundaries.

The delta is a **discrete first derivative**. With window spacing
$\Delta t = 1/f_{\text{win}}$,

$$
\Delta z_w = z_{w+1} - z_w \;=\; \Delta t \cdot \frac{z_{w+1}-z_w}{\Delta t}
\;\approx\; \Delta t \, \dot z(w),
$$

i.e. $\Delta z_w$ is a **latent velocity** (up to the constant $\Delta t$). The
pose stream models *where the latent is*; the delta stream models *how fast and
in which direction it is moving*.

---

## 2. What an HMM state means under each stream

The Gaussian HMM clusters the emissions $x_w$ into $K$ states with
$x_w \mid s_w{=}k \sim \mathcal{N}(\mu_k, \Sigma_k)$.

| | **pose stream** ($x_w = z_w$) | **delta stream** ($x_w = \Delta z_w$) |
|---|---|---|
| a state $k$ is… | a **region of latent space** = a *kind of posture / configuration* | a **characteristic velocity** = a *kind of movement* (direction + magnitude of change) |
| $\mu_k$ | a pose code (decodes to a renderable pose) | a change vector (a movement, not a pose) |
| $\|\mu_k\|\approx 0$, small $\Sigma_k$ | a *frequently-held pose* | **stillness** — little movement of any kind |
| large $\|\mu_k\|$ or large $\Sigma_k$ | a pose far from the mean | **vigorous / variable movement** |
| dwell time | how long a **posture** is held | how long a **movement style** persists |
| transition $A_{jk}$ | posture $j \to$ posture $k$ | movement-style $j \to$ movement-style $k$ |
| occupancy $\rho_k$ | fraction of time in posture $k$ | fraction of time moving in style $k$ |

The key shift: **pose-stream states are postural; delta-stream states are
kinematic.** Two infants can visit the same set of postures (same pose-stream
occupancy) yet differ entirely in *how* they move between them — and that "how"
is exactly what the delta stream captures.

---

## 3. Why delta is closer to the fidgety construct

Prechtl's **fidgety movements** are defined by the *quality of ongoing motion* —
small, continual, elegant, variable movements of neck/trunk/limbs — **not** by
which postures the infant adopts. The clinical label is *presence vs. absence of
this movement character*, not a posture inventory.

- **Absence of fidgety movements** (the abnormal / at-risk class) shows up as
  *reduced, monotonous, or cramped* motion. On the delta stream that is a latent
  velocity concentrated near $\mathbf{0}$ with low variability — i.e. high
  occupancy of a **near-zero delta ("still") state** and low occupancy of the
  active-movement states.
- **Presence of fidgety movements** shows up as latent velocity that is
  *continually non-zero and variable* — occupancy spread across several
  active-movement states, frequent transitions.

So the delta-stream phenotype features (occupancy of the near-zero state, dwell
in active states, transition rate) map **directly** onto the diagnostic axis. The
pose-stream features only capture it indirectly, through whatever postural
correlates the movement happens to have.

There is also a **two-estimator symmetry** worth stating in the thesis: the
headline clinical figure is the raw-keypoint **velocity** band-power ratio (FBR).
The delta-stream HMM is the *latent-space* counterpart of that same velocity
signal. Agreement between "raw-velocity FBR separates the groups" and
"delta-state occupancy separates the groups" is strong evidence that the latent
captured the clinically-defined *movement* dynamics — not a postural confound.

---

## 4. The frequency argument actually fits delta *better*

The dwell$\to$frequency map (§4.2) assumes an oscillation visits two extremes per
cycle, hence the factor of two:
$A_{kk} = 1 - 2f/f_{\text{win}}$. On the **delta** stream this is natural: a
periodic movement $z(t)$ has a velocity $\dot z(t)$ that points one way on the
flexion half-cycle and the opposite way on the extension half-cycle, so a full
cycle is two delta-states (a "$+\Delta z$" state and a "$-\Delta z$" state), each
dwelling $\approx \tfrac{1}{2f}$. The self-transition of a delta-state therefore
encodes the half-cycle dwell exactly as the frequency map assumes. The same
`label_state_frequencies` call applies unchanged (same $f_{\text{win}}$).

---

## 5. The costs (why it is not the automatic default)

Differencing is not free.

1. **States no longer decode to a single pose.** `decode_state_appearance`
   builds a constant block $[\mu_k,\dots,\mu_k]$ and decodes it; for the pose
   stream that is a renderable pose, but for the delta stream $\mu_k$ is a
   *change*. To visualise a delta-state you must **integrate**: start from the
   group-mean pose $\bar a$ and step $\hat z_t = \bar a + t\,\mu_k$ (or feed a
   ramped latent block), rendering the *movement* as a short animation rather
   than a still. The state-appearance panel and the Fig-3a "movement dynamics"
   figure need this adaptation for delta.

2. **Differencing amplifies noise.** For per-window encoder jitter of variance
   $\sigma^2$ and lag-1 correlation $\rho$,

   $$
   \operatorname{Var}(\Delta z_w) = 2\sigma^2(1-\rho).
   $$

   If consecutive windows are noisy and weakly correlated ($\rho$ small), the
   delta is dominated by noise and the HMM clusters jitter, not movement. Delta
   therefore **demands a smoother latent** — a temporally-coherent encoder and a
   little KL regularisation (moderate $\beta$) matter more here than for the pose
   stream. This is the practical link to the earlier $\beta$ discussion: for the
   delta stream, "not collapsed **and** not jittery" is the target.

3. **One fewer window per video** ($M-1$). Negligible.

4. **A non-stationarity upside.** Differencing removes slow drift / baseline
   shift (any integrated, $I(1)$-like component — residual normalisation drift,
   slow postural trend). If the pose trajectory drifts, the pose-stream HMM
   spends states modelling the drift; the delta stream is more stationary, so its
   states are about intrinsic dynamics. This is the classic reason to difference a
   trending series, and it often makes the delta-state structure cleaner.

---

## 6. How to choose (and the leakage rule)

Fit **both** streams and select the stream by **held-out log-likelihood** under
the video-wise split — the same unsupervised criterion used to pick $K$
(Guardrail 2.5). Do **not** pick the stream by how well it separates the clinical
labels: that is selection on the label and inflates the clinical result. Report
the clinical separation of the *chosen* stream post hoc, or, if you must compare
both streams on the clinical contrast, treat it as exploratory and correct for
the two looks.

Expected outcome, stated as a hypothesis to test rather than a foregone
conclusion: the delta stream should give a cleaner "stillness vs. movement"
phenotype and a stronger raw-velocity-FBR / latent-dynamics agreement, at the
cost of needing a smoother latent and a modified state-rendering panel.

---

## 7. The knob

Everything downstream (stitch $\to$ seam check $\to$ `fit_hmm` $\to$ frequency
labels $\to$ occupancy/dwell phenotype $\to$ clinical test) runs identically for
both streams; only the emitted sequence changes:

```python
STREAM = "delta"          # instead of "pose"
Z, lengths, vidid = H.stitch_dataset(adapter, videos,
                                     clip_len=cfg.clip_length,
                                     stride=cfg.clip_length // 2,
                                     stream=STREAM)
# fit_hmm / label_state_frequencies / occupancy / clinical test: unchanged.
# Only the state-*appearance* rendering must switch to integrated trajectories.
```

`f_win` is unchanged (the delta is sampled at the same window rate), so the
0.5–2 Hz band flags and the FBR comparison stay directly comparable across the
two streams.
