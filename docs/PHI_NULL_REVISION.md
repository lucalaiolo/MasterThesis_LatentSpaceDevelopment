# The fluency null: replacement text for §7.2

The reported `Phi` no longer reorders a subject's visits over all `n!`
permutations. It reorders them over the arrangements that never repeat a
state, which is the constraint the visit sequence itself carries. This file
carries the replacement for eq. (26) and the paragraph that justifies it,
written to drop into `\subsection{Fluency}` in place of the current
equation and the sentence beginning "The null is the substance of the
estimator".

Everything else in §Fluency is unchanged: the construct, the visit-sequence
compression, eq. (30)'s channel split, the fluency curve, Propositions 1
and 2, and the reported scalar all read the same.

---

## 1. Replacement for eq. (26)

Insert before the equation:

> Write $\mathfrak{A}(q^{(i)})$ for the set of orderings of the visits of
> $q^{(i)}$ in which no state occupies two consecutive positions. Because
> $q^{(i)}$ is run-length compressed, $q^{(i)} \in \mathfrak{A}(q^{(i)})$, so
> the set is never empty.

Then eq. (26) becomes

```latex
\begin{equation}
\label{eq:phi}
\Phi^{(i)} =
\underbrace{\frac{1}{n_i - 1}\sum_{t=1}^{n_i-1} S_{q_t,\, q_{t+1}}}_{\text{observed}}
\;-\;
\underbrace{\E_{\pi \sim \mathcal{U}(\mathfrak{A}(q^{(i)}))}
\Biggl[\frac{1}{n_i - 1}\sum_{t=1}^{n_i-1} S_{\pi(t),\, \pi(t+1)}\Biggr]}
_{\text{occupancy-matched null}},
\end{equation}
```

the null estimated from $2000$ draws from $\mathcal{U}(\mathfrak{A}(q^{(i)}))$.

Note the index change: $\pi$ is now an ordering, not a permutation of
positions, so the summand is $S_{\pi(t),\pi(t+1)}$ rather than
$S_{q_{\pi(t)},q_{\pi(t+1)}}$.

## 2. Replacement justification paragraph

Replacing "The null is the substance of the estimator: a uniform permutation
preserves the multiset of visited states exactly, so occupancy and every
occupancy-derived statistic are identical between the observed sequence and
every null draw. Equation~\eqref{eq:phi} therefore asks whether an infant's
movements follow one another more self-similarly than its own repertoire
composition would produce by chance."

> The null is the substance of the estimator, and the constraint on it is not
> cosmetic. Every ordering in $\mathfrak{A}(q^{(i)})$ carries the same multiset
> of visits as the observation, so occupancy and every occupancy-derived
> statistic are identical between the observed sequence and every null draw;
> and every draw lies in the space the observation lives in, so the comparison
> is against orderings the infant could actually have produced.
>
> Dropping the constraint and drawing from all of $\mathfrak{S}_{n_i}$ would
> preserve occupancy just as exactly and still fail. A uniform permutation
> places a state beside itself, which \eqref{eq:phi-visit} forbids the
> observation from ever doing, and each such pair contributes the diagonal
> entry $S_{kk} = 1$, the largest value $S$ takes. The null is therefore lifted
> above the range the observed statistic can occupy, by an amount governed by
> the rate at which those pairs arise. For a uniform permutation of the visits
> that rate is exactly
> $\sum_k n_k(n_k - 1) \big/ n_i(n_i - 1)$, with $n_k$ the number of visits to
> state $k$ --- the probability that two positions drawn without replacement
> hold the same state. It tends to $\sum_k o_k^2$ in the occupancy $o_k$, though
> the visit sequences here are short enough (tens of visits) that the
> finite-sample form is the one to quote. Either way the rate is a property of
> the infant, not of the ordering: an infant whose movements concentrate on a
> few states is penalised more than one whose repertoire is spread. The
> unconstrained null would thus subtract a term that varies across infants with
> exactly the quantity it was introduced to hold fixed, and $\Phi$ would confound
> sequencing with repertoire concentration --- the confound
> Section~\ref{sec:mixing} treats as a separate construct. Restricting to
> $\mathfrak{A}(q^{(i)})$ removes it, and \eqref{eq:phi} then asks what it
> claims to ask: whether an infant's movements follow one another more
> self-similarly than its own repertoire composition would produce by chance.

Cross-reference `\eqref{eq:phi-visit}` to whichever label carries
"$q_{t+1} \neq q_t$ throughout"; the current text states that inline without a
label, so one may need adding.

## 3. Optional sentence on how the null is drawn

Only if a methods reviewer would ask. Rejection sampling from
$\mathfrak{S}_{n_i}$ is not an option: its acceptance rate is
$\lvert\mathfrak{A}(q^{(i)})\rvert$ over the number of distinct permutations of
the visits, which is $0.13$ at thirty visits over eleven states, $3\times
10^{-5}$ at a hundred and twenty, and $10^{-10}$ at two hundred and fifty.

> Draws are taken by sequential importance sampling. Each ordering is built
> left to right, the next state drawn --- from those that remain, differ from
> the one just placed, and would not strand the remainder --- with probability
> proportional to how many copies are left, and carries the weight $1/q(w)$ for
> the proposal probability $q(w)$ of the ordering it produced. The draws are
> independent, and a weighted mean under those weights is consistent for the
> expectation under $\mathcal{U}(\mathfrak{A}(q^{(i)}))$. Because the mean of the
> unnormalised weights estimates $\lvert\mathfrak{A}(q^{(i)})\rvert$, which is
> separately computable in closed form by block inclusion--exclusion, the
> sampler is checked against an exactly known quantity rather than assumed
> correct. Draws are added until the effective sample size
> $(\sum_b w_b)^2 / \sum_b w_b^2$ reaches $2000$, and the smallest effective
> size attained across recordings is reported.

The estimator falls back to a Metropolis chain on $\mathfrak{A}(q^{(i)})$, whose
draws are unweighted and so cannot degenerate, if the importance weights
concentrate. That happens only for long visit sequences whose occupancy
concentrates on a few states --- the effective fraction falls to $0.15$ at two
hundred and fifty visits and $0.001$ at two thousand --- and none of this
cohort's recordings is expected to need it. Report whether any did.

The chain is retained in the code as an independent cross-check on the
importance sampler rather than as a second estimator: the two share no
machinery and agree on the null to within $0.006$ across all recordings.

## 4. What to report in Results

The correction is per-recording, so its spread is the reportable quantity, not
just its median. The run prints, and `summary.md` carries, the median offset
between the two nulls, its range across recordings, and the median rate of
adjacent-equal pairs in the unconstrained null. `per_subject.csv` carries
`phi_excess` and `phi_null` (reported) alongside `phi_excess_uniform` and
`phi_null_uniform` (the unconstrained versions) for every recording, plus
`phi_null_ess` and `phi_null_method`, which say how many draws effectively
counted and whether the chain fallback was needed.

Two numbers worth stating in Results, since they are what justify the change
having been made at all:

- the median offset and its range across the 38 recordings --- a wide range is
  the direct evidence that the unconstrained null was subtracting a
  subject-specific quantity;
- the correlation of `Phi` with occupancy entropy in
  Section~\ref{sec:controls}, under both nulls. Under the unconstrained null
  that correlation carries the artefact described above; it should fall under
  the reported one, and if it does not, that is itself worth saying.
