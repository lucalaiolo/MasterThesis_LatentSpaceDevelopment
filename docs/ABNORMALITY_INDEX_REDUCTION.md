# Reducing the endpoints to one index: why not PCA

The abnormality index needs one number per recording, built from the four
per-recording endpoints — fluency `Φ`, the Kemeny constant `𝒦`, WCLR-PP
synchrony `mean F` and FidgetyFind `FF` — so that a normative band can be cut
on it. The first implementation used `PC1` of the standardised matrix. This
note explains why that is the wrong reduction for this question, what the
**whitened a-priori direction** is instead, and what changing it does and does
not buy on the RVI-38 cohort.

Nothing here is fitted against the label. The label is used in exactly one
place, unchanged from before: choosing which 32 recordings define the normative
band.

---

## 1. The score, in one line

With `x` a recording's four (or three) standardised endpoints, `μ₀` and `Σ₀`
the mean and covariance of the **normal** recordings, and `d` the a-priori
pathological direction — `+1` where a high value is the abnormal one, `−1`
where a low one is, `0` to abstain — the index is

```
s(x) = dᵀ Σ₀⁻¹ (x − μ₀)
```

Read it as: *how far along the pathological direction does this recording sit,
once you account for how the endpoints normally move together?*

That is all of it. The normative band is then built on `s` exactly as before —
`μ₀ ± 2σ₀` of the healthy cohort's own `s` — so everything downstream of the
reduction is unchanged.

---

## 2. The problem it solves

On this cohort the four endpoints correlate like this:

|            |   `Φ`   |  `𝒦`   | `mean F` |  `FF`  |
|------------|--------:|--------:|---------:|-------:|
| `Φ`        | `+1.00` | `+0.77` |  `+0.32` | `−0.17`|
| `𝒦`        | `+0.77` | `+1.00` |  `+0.10` | `−0.41`|
| `mean F`   | `+0.32` | `+0.10` |  `+1.00` | `+0.35`|
| `FF`       | `−0.17` | `−0.41` |  `+0.35` | `+1.00`|

`Φ` and `𝒦` sit at **`+0.77`**. That is not a finding: both are read off the
same Viterbi path of the same fit, so they are two summaries of one object.

PCA maximises **variance**, and the largest variance direction in a matrix like
that is whatever the two correlated columns share. `PC1` came out at
`Φ +0.64, 𝒦 +0.67, mean F +0.16, FF −0.34` and kept 49.1 % of the standardised
variance: it is essentially *the HMM state-path axis*, with the two constructs
built independently of the model nearly ignored.

**Maximum variance is not maximum abnormality**, and on this cohort the two
directions genuinely differ. PCA has no way to know that `Φ` and `𝒦` agreeing
is a property of the estimator rather than evidence about the infant, so it
counts the same information twice and calls the result the principal component.

Whitening is precisely the correction. `Σ₀⁻¹` asks how surprising a *pattern*
is given how the endpoints normally co-vary, so a second measurement of
something already measured adds little, and an independent measurement adds a
lot.

---

## 3. Where the formula comes from

Model the normal recordings as `x ~ N(μ₀, Σ₀)`, and the pathological
alternative as the same distribution shifted along the a-priori direction,
`x ~ N(μ₀ + δd, Σ₀)` for some `δ > 0`. The log-likelihood ratio between the two
is

```
log [ N(x; μ₀ + δd, Σ₀) / N(x; μ₀, Σ₀) ]
      = δ · dᵀ Σ₀⁻¹ (x − μ₀)  −  ½ δ² · dᵀ Σ₀⁻¹ d
```

The second term does not depend on `x`. So for any shift size `δ`, the evidence
that a recording came from the shifted distribution rather than the healthy one
is a monotone function of `dᵀ Σ₀⁻¹ (x − μ₀)` and nothing else. The index is the
**Gaussian likelihood ratio for a shift along the stated pathological
direction**, and `δ` never has to be chosen — it only rescales the score, and
the normative band rescales with it.

This is the answer to "could we use a multivariate Gaussian surprise instead of
PCA?". It *is* a multivariate Gaussian surprise. It is the **directional** one.

---

## 4. What it is not

**Not Mahalanobis distance.** The omnidirectional surprise
`D² = (x − μ₀)ᵀ Σ₀⁻¹ (x − μ₀)` is the same Gaussian and the same `Σ₀⁻¹`, but it
squares the deviation and so is **direction-blind**: a recording that is
unusually *healthy* — very high `FF`, very low `Φ` — is exactly as surprising as
a pathological one. The GMA label carries a direction and `D²` discards it. On
this cohort `D²` scores **AUC 0.552**, which is chance, and worse than three of
the four endpoints taken one at a time. `s` is the projection of the same
whitened deviation onto the whitened pathological direction: writing
`Σ₀⁻¹ = LᵀL`, `D² = ‖L(x − μ₀)‖²` while `s = (Ld)ᵀ L(x − μ₀)`.

**Not Fisher's linear discriminant.** LDA is the same expression with `d`
replaced by `μ₁ − μ₀`, estimated from the labelled abnormal group. That is a
fitted direction, and at `n₁ = 6` it is fitted on almost nothing. Substituting
the *stated* poles keeps the form and removes the fit. On this cohort the two
land in the same place — the a-priori direction scores **AUC 0.786** against the
label-fitted LDA's **0.781**, on the same three endpoints and with the LDA one
false positive worse — so there is nothing left for a discriminant to find,
which is the strongest available argument for the normative framing.

**Not a tuned classifier.** `d` has no free parameters: it is `+1`, `−1` or `0`
per endpoint, read off what METHODS already states about each construct's
pathological pole. `Σ₀` and `μ₀` are estimated from the normal recordings only,
which is the same information the band already uses.

---

## 4a. The direction is a claim, and it carries the whole result

`PC1` uses `d` for one bit: which way round to point a component it found on
its own. The two-sided band then ignores even that. **The whitened score is
different in kind — it *is* `d`**, filtered through `Σ₀⁻¹`. Every pole in it is
a claim doing real work, and the result is only as good as those claims.

METHODS states a pathological pole for two of the four endpoints: high WCLR-PP
coupling is the cramped-synchronised pole, and high FidgetyFind is normal. It
states nothing for `Φ` or `𝒦`. Running the whitened score on the stated poles
alone — `d = (0, 0, −1)` on `Φ, 𝒦, FF` — gives

| poles | `d` on `Φ, 𝒦, FF` | AUC | p | sens | spec |
|---|---|---:|---:|---:|---:|
| **stated only** | `(0, 0, −1)` | **0.661** | 0.23 | 1/6 | 31/32 |
| plus argued poles | `(+1, +1, −1)` | **0.786** | 0.026 | 3/6 | 31/32 |

The entire advantage of the whitened reduction over `PC1` — 0.661 against
0.786, one flagged abnormal recording against three — comes from asserting that
**high `Φ` and high `𝒦` are the abnormal poles**. With only what METHODS
states, the whitened score is the *worst* option on the table, well behind
`PC1`'s 0.776.

The argument for those two poles is real and comes from the constructs, not the
data: fidgety movement is directionally variable, so its absence should make
consecutive movements more alike and raise `Φ`; and a restricted repertoire
should take longer to reach a randomly drawn state, raising `𝒦`. But both
predictions agree with the direction of effect observed on this cohort, and
they were written down after that direction was visible. That is the exact
shape of a post-hoc rationalisation, whether or not it is one here.

So the choice is a genuine fork, and it has to be made in the open:

* **`--index-poles stated`** (the default) is defensible without further
  argument, and under a directional reduction it is weak, because it reduces to
  FidgetyFind plus whatever the covariance borrows from the other two.
* **`--index-poles construct`** is the stronger readout and requires the thesis
  to state the `Φ` and `𝒦` predictions **as predictions**, argued from the
  constructs, and to acknowledge that they were not registered before the
  cohort was seen.

The run logs which set was used and prints this warning whenever a directional
reduction is combined with the argued poles. Under `pc1` the question is nearly
moot — the poles only fix a sign the two-sided band discards — which is one
real advantage `PC1` has here.

---

## 5. What the whitening actually does here

Dropping `mean F` (see §7) and standardising, the healthy covariance is

|          |   `Φ`   |  `𝒦`   |  `FF`   |
|----------|--------:|--------:|--------:|
| `Φ`      | `+0.908`| `+0.598`| `−0.051`|
| `𝒦`      | `+0.598`| `+0.764`| `−0.226`|
| `FF`     | `−0.051`| `−0.226`| `+0.733`|

and the three reductions weight the endpoints like this:

| endpoint | unit weights | `PC1` | **whitened `Σ₀⁻¹d`** |
|---|---:|---:|---:|
| `Φ`      | `+0.577` | `+0.619` | `+0.558` |
| `𝒦`      | `+0.577` | `+0.675` | `+0.193` |
| `FF`     | `−0.577` | `−0.402` | `−0.807` |

Read across the last two columns. Among the healthy, `𝒦` correlates with `Φ` at
`+0.60` and `FF` correlates with `Φ` at `−0.05`.

* **`PC1` puts its weight on `𝒦`** (`+0.675`, the largest of the three) and
  discounts `FF`. It rewards the redundancy.
* **Whitening nearly zeroes `𝒦`** (`+0.193`) and puts the weight on `FF`
  (`−0.807`). Once `Φ` is known, a high `𝒦` is only surprising to the extent it
  exceeds what `Φ` already predicts — which is not much. `FF` is almost
  independent of `Φ`, so an unusual `FF` is genuinely new information and is
  weighted accordingly.

The two reductions do **opposite** things with the same matrix. That is the
whole argument, and it is visible in three numbers.

---

## 6. What it buys, and what it does not

Every row below is leave-one-out for the 32 normal recordings' **band**; the
abnormal never enter a band, so they are already out-of-sample. Band at
`±2 SD`.

> **These rows hold the direction fixed.** `PC1`'s loadings and the whitened
> weights are themselves estimated from all 38 recordings, and §6a shows that
> once the direction is refit without the recording being scored, the ordering
> below changes and the whitened score loses its advantage. Read §6a before
> quoting anything from this table.

| reduction | endpoints | AUC | p | sens | spec |
|---|---|---:|---:|---:|---:|
| `PC1` (as first implemented) | all four | 0.766 | 0.041 | 2/6 | 30/32 |
| Mahalanobis `D²` | all four | 0.552 | 0.71 | 3/6 | 30/32 |
| `PC1` | `Φ, 𝒦, FF` | 0.776 | 0.033 | 3/6 | 30/32 |
| unit weights | `Φ, 𝒦, FF` | 0.771 | 0.037 | 3/6 | 29/32 |
| whitened, **stated poles only** | `Φ, 𝒦, FF` | 0.661 | 0.23 | 1/6 | 31/32 |
| **whitened, argued poles** (§4a) | `Φ, 𝒦, FF` | **0.786** | **0.026** | **3/6** | **31/32** |
| whitened, argued poles, `λ = 0.5` | `Φ, 𝒦, FF` | 0.776 | 0.033 | 3/6 | 31/32 |
| *Fisher LDA (uses the label)* | `Φ, 𝒦, FF` | *0.781* | *0.030* | *3/6* | *30/32* |

Held at a fixed direction, the whitened score with the argued poles is the best
of the label-free reductions and reaches the label-fitted bound — but see §6a,
which withdraws that conclusion once the direction is held out too. But be clear about the size of
the win: against `PC1` on the same three endpoints it is **one false
positive**, and the AUCs differ by 0.010 at `n = 38`. It also depends entirely
on §4a — with the stated poles alone it drops to 0.661 and 1/6. The honest
summary is that `PC1` and the whitened score are within noise of each other on
this cohort, and the case for the whitened one is that it is the better
*estimator* given `Φ`–`𝒦` at `+0.77`, not that it rescues the result.

---

## 6a. Refitting the direction too, and what survives

The covariance can be inverted here — six parameters from 32 recordings at
`p = 3`, condition number 7.5; ten and 9.0 at `p = 4` — but *invertible* is not
*estimated well enough to pay for itself*. A stratified bootstrap (2000 draws)
moves the fitted direction by:

| direction | median | 90th pct | max |
|---|---:|---:|---:|
| `PC1`, all four | 15.8° | **64.9°** | 89.8° |
| whitened, all four | 19.2° | 39.8° | 77.7° |
| whitened `λ = 0.5`, all four | 14.6° | 26.6° | 54.2° |
| `PC1`, `Φ, 𝒦, FF` | **9.5°** | 23.5° | 89.1° |
| whitened, `Φ, 𝒦, FF` | 22.9° | **55.1°** | 89.6° |
| whitened `λ = 0.5`, `Φ, 𝒦, FF` | 15.8° | 29.0° | 53.8° |

Neither direction is well determined, and `PC1` is not the safe alternative: it
is an eigenvector of the same covariance, so its stability is governed by the
eigengap, and on all four endpoints the top two eigenvalues are `2.02` and
`1.42` — a ratio of 1.42, which is barely identified. It sometimes flips
outright.

Refitting **everything** — the standardisation, the direction and the band —
without the recording being scored gives the honest readout:

| reduction | endpoints | LOO AUC | sens | spec |
|---|---|---:|---:|---:|
| **unit weights** | **all four** | **0.781** | 3/6 | 30/32 |
| `PC1` | `Φ, 𝒦, FF` | 0.786 | 3/6 | 30/32 |
| whitened `λ = 0.5` | `Φ, 𝒦, FF` | 0.776 | 3/6 | 29/32 |
| unit weights | `Φ, 𝒦, FF` | 0.771 | 3/6 | 29/32 |
| whitened `λ = 0` | `Φ, 𝒦, FF` | 0.766 | 2/6 | 30/32 |
| whitened `λ = 0.5` | all four | 0.766 | 2/6 | 30/32 |
| `PC1` | all four | 0.760 | 2/6 | 30/32 |
| whitened `λ = 0` | all four | 0.693 | 1/6 | 31/32 |

**The whitened score's 0.786 in §6 was in-sample for the direction.** Held out
properly it is 0.766, below `PC1` on the same endpoints. At `n₀ = 32` the
whitening does not pay for the covariance it has to estimate, and §5's argument
— that it is the better *estimator* given `Φ`–`𝒦` at `+0.77` — is correct in
principle and does not survive the sample size.

What wins is **unit weights**: the a-priori poles alone, no covariance, no
eigenvector, nothing estimated. Best out-of-sample readout of the eight, most
stable by construction, and it keeps `mean F`, so the feature-selection problem
of §7 does not arise either.

§4a survives all of this untouched. Unit weights on all four with the **stated**
poles alone give AUC 0.688 and 1/6. The poles carry the result under every
directional reduction — whitened, unit or otherwise — and remain the
load-bearing assumption.

---

**The ceiling is set by the constructs, not by the reduction.** On the whitened
score the six abnormal recordings sit at

```
−0.45   +0.05   +0.64   +2.06   +2.99   +3.21     SDs of the healthy cohort
```

with the most extreme healthy recording at `+2.26`. Three of the six are inside
the healthy body of the distribution on every one of these constructs, so
**3/6 is the ceiling for any monotone score built from these numbers at any
threshold**. No reduction reaches them. That is a finding about `Φ`, `𝒦` and
`FF` on this cohort and it belongs in the results as one, not as a sensitivity
figure that reads like a failure.

The band width is the other binding constraint. The abnormal group's mean is
`+1.41 SD` on the composite, so a `2 SD` cut mathematically retains only about
28 % of a group centred there:

| cut | flagged | sensitivity | specificity |
|---|---|---:|---:|
| `1.65 SD` | 5/38 | 3/6 | 30/32 |
| `2.00 SD` | 4/38 | 3/6 | 31/32 |
| `2.50 SD` | 2/38 | 2/6 | 32/32 |

---

## 7. Dropping `mean F`

WCLR-PP synchrony contributes nothing to any composite: its own contrast is
**AUC 0.464, p 0.80**, and the abnormal group sits `−0.10 SD` from the normal
one — that is, marginally *opposite* to the cramped-synchronised pole the
construct was built to detect. Including it costs sensitivity in every
reduction tested.

**Pre-specify the drop on the endpoint's own null contrast**, which is already
reported independently of any composite. Do not justify it by the composite
improving: at `n = 38` with six positives, selecting a feature set on the
composite's own AUC is selection on the outcome, and it should be called out as
such by any referee. The four-endpoint version stays in the results as the
pre-specified primary if that is what was registered; this is a documented
change of reduction, not a search over reductions.

---

## 8. Estimating `Σ₀` at `n₀ = 32`

`Σ₀` has `p(p+1)/2` free parameters: six at `p = 3`, ten at `p = 4`. That is
5.3 observations per parameter on three endpoints and 3.2 on four — estimable,
but thin. `Σ₀⁻¹` amplifies whatever error is in `Σ₀`, and the amplification is
worst exactly where the whitening does the most work — the near-collinear
`Φ`–`𝒦` block, where the inverse's off-diagonal reaches `−2.02` against a
diagonal of `+2.40`.

Two mitigations, in order of preference:

1. **Shrink `Σ₀` toward its diagonal**, `Σ̃₀ = (1−λ)Σ₀ + λ·diag(Σ₀)`. `λ = 0`
   is the whitened score, `λ = 1` is unit weights on the standardised
   endpoints, and the path between them is a continuum. On the four-endpoint
   version `λ = 0.5` was the best of everything tested (AUC 0.781, 3/6, 31/32),
   which is a fair sign that the raw inverse is over-fitting the healthy
   covariance. **`λ` must be fixed a priori** — `0.5` on the argument that
   nothing about `n₀ = 32` justifies trusting the off-diagonals fully — never
   tuned against the contrast.
2. **Report the sensitivity of the readout to `λ`** over `{0, 0.25, 0.5, 0.75,
   1}` alongside the primary, as a declared grid.

---

## 9. Running it

```
--index-reduction {pc1,whitened,unit}   default pc1
--index-poles     {stated,construct}    default stated
--index-shrinkage LAMBDA                default 0.0, in [0, 1]
--index-features  phi,kemeny,FF         default: all four
```

The readout reported above is
`--index-reduction whitened --index-poles construct --index-features phi,kemeny,FF`.
`report.run_report` takes the same as keywords, and `results.json` records all
four so a number can always be traced to the reduction that produced it.

---

## 10. What to state in the thesis

* The reduction is a **Gaussian likelihood ratio for a shift along a
  pre-specified direction**, not a fitted classifier. `d` comes from each
  construct's stated pathological pole; `μ₀` and `Σ₀` from the normal
  recordings; nothing is estimated against the outcome.
* The **specificity is in-sample** — the band is built from the normal cohort
  and applied to it — so the leave-one-out specificity is reported beside it.
  The sensitivity needs no such correction, since no abnormal recording enters
  the band.
* The standardisation and `Σ₀` see all `N` rows, which is **transductive but
  label-free**.
* Unlike the two-sided `PC1` band, this score **is** `d`, not merely oriented by
  it. That is what recovers the information `D²` throws away, and it is also
  what makes §4a load-bearing: state which pole set was used, and if it is
  `construct`, present the `Φ` and `𝒦` poles as predictions argued from the
  constructs while acknowledging they were not registered in advance.
* **Three of the six abnormal recordings are not separable from the normal
  cohort by these constructs at any threshold.** Report that as a property of
  the constructs.
