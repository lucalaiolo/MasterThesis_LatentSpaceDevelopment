"""Latent-space geometry: the three diagnostics the results section reports.

The results section *Latent-space geometry* reports exactly three probes of
[MATH §2-§6] for the **selected** model, all computed on the **posterior means
of the held-out clips** over the **flattened** latent of width ``d_z * T/l``
(128 for the selected temporal model: ``d_z = 8`` per window, ``T = 64``
frames, ``l = 4``, so ``16`` windows):

1. **Intrinsic dimension** — TwoNN [MATH §sec:twonn] (Facco et al., 2017).
   Lemma ``lem:mu`` makes ``-log(1 - F(mu))`` linear in ``log mu`` with slope
   ``d``; the estimate is the slope of the **least-squares line through the
   origin** fitted to the empirical point set. Figure: the log-log fit.
2. **Prior fit** — the kernel two-sample test [MATH §sec:mmd] (Gretton et al.,
   2012), unbiased ``MMD^2_u`` with a Gaussian kernel at the median-heuristic
   bandwidth, calibrated by a ``B``-fold permutation test. Read through
   Proposition ``prop:elbo_surgery``: the test speaks only to
   ``KL(q(z) || p(z))``, so it is reported alongside the per-dimension
   divergence ``eq:klclosed``, whose collapse reading is [MATH §sec:collapse].
3. **Latent interpolation** — the two-endpoint use of the decoder geometry
   [MATH §sec:geometry]. The straight latent segment ``z(t)`` is decoded at
   evenly spaced ``t`` (the morph figure), and the scalar summary is the
   curvature ratio ``rho = L[g(gamma*)] / L[g(z(.))]``: the decoded arc length
   ``eq:arclength`` of the discrete geodesic ``eq:geodesic`` (energy
   ``eq:discrete-energy``, minimised by :mod:`vae_analysis.geodesic_interpolation`)
   over that of the straight segment. ``rho <= 1`` up to discretisation, and
   ``rho ~ 1`` says the manifold is nearly flat between the two endpoints.

Deliberately **not** run here, because the section says they are not reported:
the sensitivity maps and the pullback-metric spectrum
(:mod:`vae_analysis.decoder_geometry`) and the cluster structure of the
aggregate posterior (:func:`vae_analysis.posterior_geometry.cluster_structure`).

Run it::

    python -m vae_analysis.latent_geometry \\
        --checkpoint runs/temporal_conv/best.pt \\
        --csv data/keypoints.csv \\
        --out-dir outputs/latent_geometry

or from Python::

    from vae_analysis.latent_geometry import run_latent_geometry
    out = run_latent_geometry("runs/temporal_conv/best.pt", videos)

Either way the run writes ``fig_twonn.pdf``, ``fig_interp.pdf`` (the two
figures the section leaves as placeholders), ``fig_kl_per_dim.pdf``,
``results.json``, ``values.tex``, and ``summary.md``. ``results.json``
carries a ``placeholders`` map from the section's ``[d\\_hat]``-style slots to
formatted values, and ``--fill-tex`` writes a copy of the section source with
those slots substituted.

References
    E. Facco, M. d'Errico, A. Rodriguez, A. Laio. Estimating the intrinsic
    dimension of datasets by a minimal neighborhood information. Sci. Rep. 2017.
    A. Gretton, K. Borgwardt, M. Rasch, B. Schölkopf, A. Smola. A kernel
    two-sample test. JMLR 2012.
    H. Shao, A. Kumar, P. T. Fletcher. The Riemannian geometry of deep
    generative models. CVPR-W 2018.
    G. Arvanitidis, L. K. Hansen, S. Hauberg. Latent space oddity. ICLR 2018.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ===========================================================================
# 1. Intrinsic dimension — TwoNN by the chapter's least-squares fit.
# ===========================================================================

def neighbour_ratios(points: np.ndarray) -> dict:
    """The TwoNN ratios ``mu_i = r_2(i) / r_1(i)`` of a point cloud.

    ``r_1(i) < r_2(i)`` are the distances from point ``i`` to its first two
    neighbours (itself excluded). Under Theorem ``thm:shell`` the ratio is
    Pareto with exponent the intrinsic dimension, independently of the local
    density — which is what makes the estimator local and curvature-robust.

    Points whose nearest neighbour sits at distance zero (exact duplicates)
    have an undefined ratio and are dropped, with the count reported so a
    duplicate-heavy set cannot pass unnoticed.

    Args:
        points: sample, shape ``(N, d)``.
    Returns:
        Dict with ``mu`` (the finite ratios), ``n_points``, and
        ``n_dropped_duplicates``.
    """
    from sklearn.neighbors import NearestNeighbors

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError(f"TwoNN needs at least 3 points, got {len(points)}.")
    nn = NearestNeighbors(n_neighbors=3).fit(points)
    dist, _ = nn.kneighbors(points)                  # (N, 3); col 0 is self
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = r1 > 0
    return {"mu": r2[ok] / r1[ok], "n_points": int(len(points)),
            "n_dropped_duplicates": int((~ok).sum())}


def twonn_least_squares(points: np.ndarray | None = None, *,
                        mu: np.ndarray | None = None,
                        discard_fraction: float = 0.1,
                        n_bootstrap: int = 200,
                        rng: np.random.Generator | None = None) -> dict:
    """TwoNN intrinsic dimension as the origin-constrained least-squares slope.

    The chapter's estimator, not the maximum-likelihood one: sort the ratios,
    build the empirical distribution ``F_emp(mu_(i)) = i / N``, place the
    points

        S = {(log mu_i, -log(1 - F_emp(mu_i)))}

    and take ``d_hat`` as the slope of the least-squares line **through the
    origin** through ``S`` — the linear form of the Pareto law ``eq:pareto``.

    The largest ``discard_fraction`` of the sorted ratios is dropped, following
    Facco et al. (2017): the top ratios are where cluster-hopping neighbours
    and the unbounded ``-log(1 - F)`` of the final order statistic live, and
    both bend the fit. ``discard_fraction = 0`` still drops the single
    non-finite point at ``F_emp = 1``.

    ``d_hat_mle`` is the maximum-likelihood cross-check, ``N / sum_i log mu_i``
    over **every** ratio. It is computed on the untrimmed sample on purpose:
    that closed form is the MLE of the untruncated Pareto law, so evaluating
    it on a sample whose top tail has been cut inflates it — by a factor of
    about 1.34 at ``discard_fraction = 0.1``, which is exactly what
    :func:`vae_analysis.posterior_geometry.intrinsic_dimension_twonn` does.
    On clean Pareto ratios the least-squares slope and this full-sample MLE
    agree to well under a percent, so a gap between them is a real warning
    that the single-exponent model is strained, not a difference of recipe.

    Two standard errors are reported. ``standard_error`` is the textbook
    least-squares one, ``sqrt( sum_i resid_i^2 / ((n-1) * sum_i x_i^2) )``;
    it is the ``+-`` the chapter quotes, and it is optimistic, because the
    ``y`` coordinates share one empirical distribution function and so are not
    independent. ``bootstrap_se`` resamples the ratios, rebuilds the
    distribution function, and refits, which prices that dependence in.

    Args:
        points: sample, shape ``(N, d)``. Give this or ``mu``.
        mu: pre-computed neighbour ratios, shape ``(N,)``.
        discard_fraction: top fraction of the sorted ratios to drop.
        n_bootstrap: bootstrap resamples for ``bootstrap_se``; 0 to skip.
        rng: seedable generator for the bootstrap.
    Returns:
        Dict with ``d_hat``, ``standard_error``, ``bootstrap_se``, ``n_used``,
        ``d_hat_mle``, ``r_squared`` (uncentred, the right one for a fit
        through the origin), the fitted coordinates ``log_mu`` / ``y`` for the
        figure, and the inputs ``mu``, ``discard_fraction``.
    """
    if (points is None) == (mu is None):
        raise ValueError("pass exactly one of `points` or `mu`.")
    dropped = 0
    if mu is None:
        r = neighbour_ratios(points)
        mu, dropped = r["mu"], r["n_dropped_duplicates"]
    mu = np.asarray(mu, dtype=np.float64)
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must lie in [0, 1).")

    x, y = _twonn_coordinates(mu, discard_fraction)
    d_hat, se, r2 = _fit_through_origin(x, y)
    log_sum = float(np.log(mu).sum())                # untrimmed: see the docstring

    boot = float("nan")
    if n_bootstrap:
        rng = np.random.default_rng() if rng is None else rng
        draws = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            resample = rng.choice(mu, size=len(mu), replace=True)
            bx, by = _twonn_coordinates(resample, discard_fraction)
            draws[b] = _fit_through_origin(bx, by)[0]
        boot = float(np.std(draws, ddof=1))

    return {
        "d_hat": float(d_hat),
        "standard_error": float(se),
        "bootstrap_se": boot,
        "n_used": int(len(x)),
        "n_ratios": int(len(mu)),
        "n_dropped_duplicates": int(dropped),
        "d_hat_mle": float(len(mu) / log_sum) if log_sum > 0 else float("nan"),
        "r_squared": float(r2),
        "log_mu": x,
        "y": y,
        "mu": mu,
        "discard_fraction": float(discard_fraction),
    }


def _twonn_coordinates(mu: np.ndarray, discard_fraction: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Sorted ``(log mu, -log(1 - F_emp(mu)))`` with the top tail dropped."""
    mu_sorted = np.sort(mu)
    n = len(mu_sorted)
    f_emp = np.arange(1, n + 1, dtype=np.float64) / n
    x = np.log(mu_sorted)
    with np.errstate(divide="ignore"):
        y = -np.log1p(-f_emp)                        # +inf at the last point
    keep = int(np.floor(n * (1.0 - discard_fraction)))
    keep = max(2, min(keep, n))
    x, y = x[:keep], y[:keep]
    finite = np.isfinite(x) & np.isfinite(y)         # guards F_emp = 1
    return x[finite], y[finite]


def _fit_through_origin(x: np.ndarray, y: np.ndarray
                        ) -> tuple[float, float, float]:
    """Least-squares slope through the origin, its standard error, and R^2."""
    sxx = float(np.dot(x, x))
    if sxx <= 0 or len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.dot(x, y) / sxx)
    resid = y - slope * x
    sigma2 = float(np.dot(resid, resid)) / max(len(x) - 1, 1)
    se = float(np.sqrt(sigma2 / sxx))
    syy = float(np.dot(y, y))                        # uncentred: no intercept
    r2 = float(1.0 - np.dot(resid, resid) / syy) if syy > 0 else float("nan")
    return slope, se, r2


# ===========================================================================
# 2. Prior fit — the kernel two-sample test with a permutation threshold.
# ===========================================================================

def _pooled_squared_distances(P: np.ndarray) -> np.ndarray:
    """All pairwise squared distances of ``P``, clipped at zero on the diagonal."""
    sq = np.einsum("ij,ij->i", P, P)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (P @ P.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def mmd_permutation_test(X: np.ndarray, Y: np.ndarray, *,
                         n_perm: int = 500,
                         bandwidth: float | None = None,
                         rng: np.random.Generator | None = None,
                         chunk: int = 64) -> dict:
    """Unbiased ``MMD^2_u`` with a Gaussian kernel and a permutation p-value.

    Implements ``eq:mmd_estimator`` and the permutation recipe that follows it:
    pool the two samples, repartition at random ``B`` times, and read
    ``p_hat = (1/B) sum_b 1(T_b >= T_obs)``. Under the null the pooled labels
    are exchangeable, so the permutation threshold is exact and no appeal to
    the (infinite chi-squared mixture) null distribution is needed.

    The kernel is ``k(x, y) = exp(-||x - y||^2 / (2 l^2))`` with ``l`` the
    median pairwise distance of the pooled sample (the median heuristic of the
    remark after Definition *characteristic kernel*), matching
    :func:`vae_analysis.posterior_geometry.mmd2_unbiased`.

    The pooled kernel matrix is formed **once** and every permutation is one
    matrix-vector product against a 0/1 indicator, which is what makes ``B``
    in the hundreds cheap: with ``a`` the indicator of the first block and
    ``K0`` the kernel with a zeroed diagonal,

        sum_{i != i' in A} k = a' K0 a,
        sum_{i in A, j in B} k = a' K0 1 - a' K0 a,
        sum_{j != j' in B} k = 1' K0 1 - 2 a' K0 1 + a' K0 a.

    Memory is the one cost: the pooled matrix is ``(m + n)^2`` doubles, so
    ``m = n = 2000`` needs about 128 MB. :func:`prior_fit` caps the sample for
    that reason.

    Args:
        X, Y: samples, shapes ``(m, d)`` and ``(n, d)``.
        n_perm: permutation count ``B``.
        bandwidth: kernel bandwidth ``l``; the median heuristic when None.
        rng: seedable generator for the permutations.
        chunk: permutations per matrix product (memory/speed trade).
    Returns:
        Dict with ``mmd2``, ``p_value`` (the chapter's ``(1/B) sum 1(.)``),
        ``p_value_add_one`` (the ``(count + 1)/(B + 1)`` variant, never zero),
        ``bandwidth``, ``n_perm``, ``m``, ``n``, and the permutation
        ``null_statistics``.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        raise ValueError("the unbiased estimator needs at least 2 points a side.")

    pooled = np.vstack([X, Y])
    d2 = _pooled_squared_distances(pooled)
    if bandwidth is None:
        iu = np.triu_indices(len(pooled), k=1)
        med = float(np.sqrt(np.median(d2[iu])))
        bandwidth = med if med > 0 else 1.0
    K0 = np.exp(-d2 / (2.0 * bandwidth * bandwidth))
    np.fill_diagonal(K0, 0.0)                        # the i != i' of eq:mmd_estimator
    del d2

    rowsum = K0.sum(axis=1)
    total = float(rowsum.sum())

    def statistic(a: np.ndarray) -> np.ndarray:
        """``T`` for one or many 0/1 indicators ``a`` of the first block."""
        s = K0 @ a                                   # (L,) or (L, chunk)
        saa = np.einsum("i...,i...->...", a, s)      # a' K0 a
        ar = a.T @ rowsum if a.ndim > 1 else float(a @ rowsum)
        sab = ar - saa                               # a' K0 b
        sbb = total - 2.0 * ar + saa                 # b' K0 b
        return (saa / (m * (m - 1)) + sbb / (n * (n - 1))
                - 2.0 * sab / (m * n))

    a_obs = np.zeros(m + n)
    a_obs[:m] = 1.0
    t_obs = float(statistic(a_obs))

    null = np.empty(n_perm)
    done = 0
    while done < n_perm:
        size = min(chunk, n_perm - done)
        A = np.zeros((m + n, size))
        for c in range(size):
            A[rng.permutation(m + n)[:m], c] = 1.0   # a uniform partition
        null[done:done + size] = statistic(A)
        done += size

    count = int(np.sum(null >= t_obs))
    return {"mmd2": t_obs,
            "p_value": count / n_perm,
            "p_value_add_one": (count + 1) / (n_perm + 1),
            "n_exceed": count,
            "bandwidth": float(bandwidth),
            "n_perm": int(n_perm),
            "m": int(m), "n": int(n),
            "null_statistics": null}


def prior_fit(mu: np.ndarray, *, n_perm: int = 500,
              max_points: int = 2000,
              rng: np.random.Generator | None = None) -> dict:
    """The section's prior-fit test: encoded clips against an equal prior sample.

    The chapter runs the test on the **posterior means** of the held-out clips
    against an equal-sized draw from ``N(0, I)``. Sampling the posteriors
    instead would test the aggregate posterior ``eq:aggregate`` including its
    per-clip noise; the means are the deterministic code the downstream state
    model actually consumes, so those are what is compared.

    Args:
        mu: posterior means, shape ``(N, d)``.
        n_perm: permutation count ``B`` (the section quotes 500).
        max_points: cap per side; the pooled kernel is ``(2 * points)^2``.
            A cap is a subsample, reported as ``n_used`` and ``subsampled``.
        rng: seedable generator.
    Returns:
        The dict of :func:`mmd_permutation_test`, plus ``n_used`` and
        ``subsampled``.
    """
    rng = np.random.default_rng() if rng is None else rng
    mu = np.asarray(mu, dtype=np.float64)
    n_used = min(len(mu), int(max_points))
    subsampled = n_used < len(mu)
    q = mu[rng.choice(len(mu), size=n_used, replace=False)] if subsampled else mu
    p = rng.standard_normal((n_used, mu.shape[1]))
    out = mmd_permutation_test(q, p, n_perm=n_perm, rng=rng)
    out["n_used"] = int(n_used)
    out["subsampled"] = bool(subsampled)
    return out


def per_dimension_kl(mu: np.ndarray, logvar: np.ndarray, *,
                     threshold: float = 0.01, fold=None) -> dict:
    """Closed-form divergence ``eq:klclosed``, per latent dimension.

    ``KL = (1/2) sum_d (mu_d^2 + sigma_d^2 - log sigma_d^2 - 1)``, averaged
    over clips and kept per dimension. This is the quantity the section
    reports *alongside* the two-sample verdict, because Proposition
    ``prop:elbo_surgery`` splits the divergence the objective controls into
    ``I_q(x; z)`` and ``KL(q(z) || p(z))`` and the kernel test sees only the
    second. A dimension whose divergence sits at zero is collapsed
    [MATH §sec:collapse]: ``mu_d = 0``, ``sigma_d^2 = 1``, no information
    carried.

    For a temporal latent the flat vector of ``d_z * n_windows`` dimensions is
    also folded to ``(n_windows, d_z)``, so an all-dead window — a stretch of
    the clip carrying nothing — is visible as an empty row. The fold is the
    model's own, passed in: the conv head flattens channel-major and the
    transformer head window-major, so reshaping by hand here would silently
    scramble one of the two.

    Args:
        mu, logvar: posterior parameters, shape ``(N, D)``.
        threshold: nats below which a dimension counts as collapsed.
        fold: optional callable mapping a ``(D,)`` vector to ``(n_windows,
            d_z)`` — e.g. the adapter's ``window_latents`` (see
            :func:`fold_with_adapter`). None leaves the fold out.
    Returns:
        Dict with ``kl_per_dim`` ``(D,)``, ``kl_total`` (mean per-clip nats),
        ``n_active``, ``n_dims``, ``threshold``, ``mu_rms``, ``sigma2_mean``,
        and, when ``fold`` is given, ``kl_by_window`` ``(n_windows, d_z)``,
        ``kl_per_window`` ``(n_windows,)``, and ``window_of_dim`` ``(D,)``
        naming each flattened dimension's window.
    """
    mu = np.asarray(mu, dtype=np.float64)
    logvar = np.asarray(logvar, dtype=np.float64)
    var = np.exp(logvar)
    kl = 0.5 * (mu ** 2 + var - logvar - 1.0)        # (N, D), eq:klclosed
    kl_per_dim = kl.mean(axis=0)

    out = {
        "kl_per_dim": kl_per_dim,
        "kl_total": float(kl_per_dim.sum()),
        "n_active": int((kl_per_dim > threshold).sum()),
        "n_dims": int(kl_per_dim.shape[0]),
        "threshold": float(threshold),
        "mu_rms": np.sqrt((mu ** 2).mean(axis=0)),
        "sigma2_mean": var.mean(axis=0),
    }
    if fold is not None:
        folded = np.asarray(fold(kl_per_dim))
        n_win, d_z = folded.shape
        if n_win * d_z != out["n_dims"]:
            raise ValueError(f"the fold returned {n_win} x {d_z} for a latent "
                             f"of width {out['n_dims']}.")
        # Send the dimension indices through the same fold, so the window a
        # flattened dimension belongs to is read off the model's layout too.
        placement = np.asarray(fold(np.arange(out["n_dims"], dtype=np.float64)))
        window_of_dim = np.empty(out["n_dims"], dtype=int)
        window_of_dim[placement.astype(int).ravel()] = np.repeat(
            np.arange(n_win), d_z)
        out["kl_by_window"] = folded
        out["kl_per_window"] = folded.sum(axis=1)
        out["window_of_dim"] = window_of_dim
    return out


def fold_with_adapter(adapter):
    """Return the ``(D,) -> (n_windows, d_z)`` fold of a temporal model.

    Delegates to ``adapter.window_latents``, which delegates to the model, so
    the conv/transformer difference in flatten order stays where it belongs.
    Returns None for a whole-clip (non-temporal) latent, which has no windows.
    """
    if not adapter.is_temporal():
        return None

    def fold(vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        return np.asarray(adapter.window_latents(v))[0]

    return fold


# ===========================================================================
# 3. Latent interpolation — the straight segment, the geodesic, and rho.
# ===========================================================================

def bone_length_report(clips: np.ndarray, bones) -> dict:
    """Bone-length behaviour along a decoded interpolation path.

    The section's visual claim is that intermediate decodings "hold bone
    lengths near constant and show no skeletal tearing". This is its number.
    Two quantities, both dimensionless:

    * ``cv_within`` — per interpolation step, the coefficient of variation of
      each bone length **across the frames of that decoded clip**. A rigid
      skeleton in motion keeps this small; tearing shows up as a spike.
    * ``drift`` — per step, the relative departure of each bone's mean length
      from the average of the two endpoint clips. This is the one that catches
      a midpoint decoding that shrinks or stretches the whole body, which the
      within-clip figure alone would miss.

    Args:
        clips: decoded path, shape ``(S, T, J, D)``, ``S`` interpolation steps.
        bones: iterable of ``(a, b)`` joint-index pairs.
    Returns:
        Dict with ``cv_within`` ``(S, n_bones)``, ``drift`` ``(S, n_bones)``,
        their means and maxima, and the step index where the drift peaks.
    """
    clips = np.asarray(clips, dtype=np.float64)
    idx = np.asarray(list(bones), dtype=int)
    if idx.size == 0:
        raise ValueError("no bones given; pass the skeleton's bone pairs.")
    a = clips[:, :, idx[:, 0], :]
    b = clips[:, :, idx[:, 1], :]
    length = np.linalg.norm(a - b, axis=-1)          # (S, T, n_bones)

    mean_len = length.mean(axis=1)                   # (S, n_bones)
    cv_within = length.std(axis=1) / (mean_len + 1e-12)
    ref = 0.5 * (mean_len[0] + mean_len[-1])         # the two real clips
    drift = np.abs(mean_len - ref) / (ref + 1e-12)
    return {
        "cv_within": cv_within,
        "cv_within_mean": float(cv_within.mean()),
        "cv_within_max": float(cv_within.max()),
        "drift": drift,
        "drift_mean": float(drift.mean()),
        "drift_max": float(drift.max()),
        "drift_max_step": int(np.argmax(drift.max(axis=1))),
        "mean_length": mean_len,
    }


def latent_interpolation(interp, x0: np.ndarray, x1: np.ndarray, *,
                         n_steps: int = 7, n_segments: int = 32,
                         method: str = "lbfgs", iters: int = 200,
                         bones=None) -> dict:
    """Decode the straight latent segment, and measure it against the geodesic.

    Encodes the two clips to their posterior means ``z_0``, ``z_1``, decodes
    ``z(t) = (1 - t) z_0 + t z_1`` at ``n_steps`` evenly spaced ``t`` for the
    morph figure, and computes the curvature ratio

        rho = L[g(gamma*)] / L[g(z(.))]

    on a **separate, finer** discretisation of ``n_segments`` segments — the
    same one for both curves, so the polyline's chord bias cancels and only
    the shape difference survives. ``gamma*`` is the discrete geodesic of
    :func:`vae_analysis.geodesic_interpolation.geodesic`, which minimises the
    energy ``eq:discrete-energy`` over the interior nodes by reverse-mode
    autodiff, starting from the straight segment.

    ``rho <= 1`` holds because ``gamma*`` minimises length; a value above 1
    (beyond rounding) means the energy descent has not converged, so the
    optimiser diagnostics come back with it: ``speed_cv`` (the coefficient of
    variation of the geodesic's decoded segment speeds, which a converged
    constant-speed minimiser drives toward 0) and the two energies.

    Args:
        interp: a :class:`~vae_analysis.geodesic_interpolation.VaeInterpolator`.
        x0, x1: the two held-out clips, each ``(T, J, D)``.
        n_steps: decoded points on the straight segment, for the figure.
        n_segments: segments of the two curves compared by ``rho``.
        method, iters: energy-descent settings (``"lbfgs"`` or ``"adam"``).
        bones: skeleton bone pairs; enables :func:`bone_length_report`.
    Returns:
        Dict with ``curvature_ratio``, the two ``arclength_*``, the two
        ``energy_*``, ``speed_cv``, ``t`` and ``straight_clips`` ``(n_steps,
        T, J, D)`` for the figure, ``geodesic_clips`` at the same ``t``,
        ``z0``/``z1`` as NumPy, ``latent_distance``, and ``bone_lengths`` when
        ``bones`` is given.
    """
    from .geodesic_interpolation import (curve_energy, decoded_arclength,
                                         geodesic, speed_profile, straight_path)

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2 (the two endpoints).")
    if n_segments < 2:
        raise ValueError("n_segments must be at least 2 for an interior node.")

    z0, z1 = interp.encode(x0), interp.encode(x1)

    # Figure resolution: the decoded straight segment at evenly spaced t.
    t = np.linspace(0.0, 1.0, n_steps)
    Z_fig = straight_path(z0, z1, T=n_steps - 1)
    straight_clips = interp.decode_clips(Z_fig)

    # Measurement resolution: geodesic vs straight, same node count.
    Z_geo = geodesic(interp.decode, z0, z1, T=n_segments,
                     method=method, iters=iters)
    Z_lin = straight_path(z0, z1, T=n_segments)
    len_geo = float(decoded_arclength(interp.decode, Z_geo))
    len_lin = float(decoded_arclength(interp.decode, Z_lin))
    _, speed_cv = speed_profile(interp.decode, Z_geo)

    # The geodesic sampled at the figure's t, for the optional comparison row.
    geo_fig = _resample_path(Z_geo, n_steps)

    out = {
        "curvature_ratio": len_geo / len_lin if len_lin > 0 else float("nan"),
        "arclength_geodesic": len_geo,
        "arclength_straight": len_lin,
        "energy_geodesic": float(curve_energy(interp.decode(Z_geo))),
        "energy_straight": float(curve_energy(interp.decode(Z_lin))),
        "speed_cv": float(speed_cv),
        "n_segments": int(n_segments),
        "t": t,
        "straight_clips": straight_clips,
        "geodesic_clips": interp.decode_clips(geo_fig),
        "z0": z0.detach().cpu().numpy(),
        "z1": z1.detach().cpu().numpy(),
        "latent_distance": float(np.linalg.norm(
            z0.detach().cpu().numpy() - z1.detach().cpu().numpy())),
    }
    if bones is not None:
        out["bone_lengths"] = bone_length_report(straight_clips, bones)
    return out


def _resample_path(Z, n_steps: int):
    """Pick ``n_steps`` evenly spaced nodes from a ``(P, d)`` path."""
    idx = np.linspace(0, Z.shape[0] - 1, n_steps).round().astype(int)
    return Z[idx]


def select_clip_pair(mu: np.ndarray, video_id: np.ndarray | None = None, *,
                     mode: str = "random", n_candidates: int = 4096,
                     rng: np.random.Generator | None = None
                     ) -> tuple[int, int]:
    """Choose the two held-out clips to interpolate between.

    Which pair is picked changes what the figure shows, so the choice is made
    explicit and recorded rather than left to whichever index came first.
    Pairs are drawn from **different videos** whenever labels allow it: two
    clips of the same recording overlap in content, and morphing between them
    would flatter the model.

    Args:
        mu: posterior means, shape ``(N, d)``.
        video_id: per-clip video labels, or None.
        mode: ``"random"`` (default; a seeded cross-video pair),
            ``"median"`` (the pair whose latent distance is nearest the median
            candidate distance — a typical transition, not a showcase), or
            ``"farthest"`` (the most demanding morph in the candidate set).
        n_candidates: candidate pairs sampled for ``median`` / ``farthest``.
        rng: seedable generator.
    Returns:
        ``(i, j)`` indices into ``mu``.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(mu)
    if n < 2:
        raise ValueError("need at least two clips to interpolate between.")
    # One video in the split means no cross-video pair exists; don't retry for one.
    cross_video = video_id is not None and len(np.unique(video_id)) > 1

    def draw() -> tuple[int, int]:
        for _ in range(64 if cross_video else 1):
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            if not cross_video or video_id[i] != video_id[j]:
                return int(i), int(j)
        i, j = rng.choice(n, size=2, replace=False)
        return int(i), int(j)

    if mode == "random":
        return draw()
    if mode not in ("median", "farthest"):
        raise ValueError(f"unknown mode {mode!r}; use random/median/farthest.")

    pairs = [draw() for _ in range(int(n_candidates))]
    dist = np.array([np.linalg.norm(mu[i] - mu[j]) for i, j in pairs])
    k = int(np.argmax(dist)) if mode == "farthest" \
        else int(np.argmin(np.abs(dist - np.median(dist))))
    return pairs[k]


# ===========================================================================
# Figures.
# ===========================================================================

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def plot_twonn(fit: dict, plt=None, *, max_points: int = 4000):
    """The section's TwoNN figure: the log-log fit and its slope.

    Empirical ratios in the coordinates ``(log mu, -log(1 - F_emp(mu)))``
    against the fitted line of slope ``d_hat`` through the origin. Under
    ``eq:pareto`` the points lie on that line, so systematic curvature away
    from it is the honest signal that the single-exponent Pareto model is
    strained — worth seeing next to the number it produces.

    Args:
        fit: the dict from :func:`twonn_least_squares`.
        plt: pyplot module; imported (Agg) when None.
        max_points: scatter at most this many points (thinned evenly).
    """
    plt = plt or _import_matplotlib()
    x, y = fit["log_mu"], fit["y"]
    if len(x) > max_points:                          # thin, keep the range
        sel = np.linspace(0, len(x) - 1, max_points).round().astype(int)
        xs, ys = x[sel], y[sel]
    else:
        xs, ys = x, y

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(xs, ys, s=6, alpha=0.35, color="steelblue",
               edgecolors="none", label="empirical", rasterized=True)
    grid = np.linspace(0.0, float(x.max()) * 1.02, 100)
    ax.plot(grid, fit["d_hat"] * grid, color="crimson", linewidth=1.8,
            label=rf"slope $\hat d$ = {fit['d_hat']:.2f}")
    ax.set_xlabel(r"$\log \mu$")
    ax.set_ylabel(r"$-\log\left(1 - F_{\mathrm{emp}}(\mu)\right)$")
    ax.set_title(rf"TwoNN: $\hat d$ = {fit['d_hat']:.2f} $\pm$ "
                 rf"{fit['standard_error']:.2f}  ($n$ = {fit['n_used']}, "
                 rf"$R^2$ = {fit['r_squared']:.3f})", fontsize=10)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def plot_interpolation(clips: np.ndarray, t: np.ndarray, bones=None, plt=None, *,
                       n_frame_rows: int = 4, invert_y: bool = False,
                       title: str = "", row_label: str = "frame",
                       point_size: float = 7.0, linewidth: float = 1.3):
    """The section's morph figure: decoded skeletons across the interpolation.

    One column per interpolation weight ``t`` — leftmost the source clip,
    rightmost the target — and one row per sampled frame **within** each
    decoded clip. The rows matter: every column is a whole ``(T, J, D)``
    motion, not a pose, so a single-row strip would hide exactly the
    collapse-to-a-static-pose failure the temporal latent exists to prevent.
    All panels share one bounding box and an equal aspect ratio, so lengths
    are comparable across the grid by eye.

    Args:
        clips: decoded path, shape ``(S, T, J, D)``. With ``D > 2`` the first
            two coordinates are plotted, i.e. a 3-D skeleton is shown in
            projection.
        t: the ``S`` interpolation weights.
        bones: ``(a, b)`` joint-index pairs; scatter-only when None.
        plt: pyplot module; imported (Agg) when None.
        n_frame_rows: frames sampled per decoded clip.
        invert_y: flip the vertical axis (for a y-down image-plane export).
        title: figure title.
        row_label: prefix for the row labels.
    """
    plt = plt or _import_matplotlib()
    clips = np.asarray(clips)
    S, T, J, D = clips.shape
    if D < 2:
        raise ValueError("need at least 2 coordinates per joint to plot.")
    rows = np.linspace(0, T - 1, min(n_frame_rows, T)).round().astype(int)

    pts = clips[..., :2].reshape(-1, 2)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    centre = 0.5 * (mins + maxs)
    half = 0.55 * float((maxs - mins).max()) or 1.0

    fig, axes = plt.subplots(len(rows), S,
                             figsize=(1.45 * S, 1.55 * len(rows)),
                             squeeze=False)
    for r, frame in enumerate(rows):
        for c in range(S):
            ax = axes[r][c]
            f = clips[c, frame, :, :2]
            # Endpoints are real clips; the interior is the model's invention.
            colour = "#1f77b4" if c in (0, S - 1) else "#d62728"
            if bones is not None:
                for a, b in bones:
                    ax.plot([f[a, 0], f[b, 0]], [f[a, 1], f[b, 1]],
                            color=colour, linewidth=linewidth,
                            solid_capstyle="round")
            ax.scatter(f[:, 0], f[:, 1], s=point_size, color=colour,
                       zorder=3, edgecolors="none")
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[1] - half, centre[1] + half)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(f"$t$ = {t[c]:.2f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{row_label} {frame}", fontsize=8)
            if invert_y:
                ax.invert_yaxis()

    # Every panel is a decoder output, endpoints included: the t = 0 and t = 1
    # columns are the *reconstructions* of the two held-out clips, not the
    # clips themselves, so the columns differ only in t.
    axes[0][0].annotate("source $x_0$ (decoded)", xy=(0.5, 1.30),
                        xycoords="axes fraction", ha="center", fontsize=8,
                        color="#1f77b4")
    axes[0][-1].annotate("target $x_1$ (decoded)", xy=(0.5, 1.30),
                         xycoords="axes fraction", ha="center", fontsize=8,
                         color="#1f77b4")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 0.97))
    return fig


def plot_kl_per_dim(kl: dict, plt=None, *, title: str = ""):
    """Per-dimension divergence ``eq:klclosed``, with the collapse floor marked.

    Bars over the flattened latent, coloured by window when the temporal
    layout is known, with the collapse threshold drawn as a line: everything
    at or below it is a dimension the encoder has handed back to the prior.
    """
    plt = plt or _import_matplotlib()
    per_dim = np.asarray(kl["kl_per_dim"])
    d = len(per_dim)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.055 * d), 3.4))

    if "window_of_dim" in kl:
        # Bars stay in flattened order; the colour says which window a dim is in.
        window_of = np.asarray(kl["window_of_dim"])
        n_win = int(window_of.max()) + 1
        cmap = plt.get_cmap("viridis")
        colours = [cmap(w / max(n_win - 1, 1)) for w in window_of]
    else:
        colours = "steelblue"

    ax.bar(np.arange(d), per_dim, color=colours, width=0.9)
    ax.axhline(kl["threshold"], color="crimson", linewidth=1.0, linestyle="--",
               label=f"collapse floor {kl['threshold']:g} nats")
    ax.set_xlabel("latent dimension (flattened)")
    ax.set_ylabel("KL (nats)")
    ax.set_title(title or (f"per-dimension KL: {kl['n_active']}/{kl['n_dims']} "
                           f"active, total {kl['kl_total']:.2f} nats"),
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ===========================================================================
# Reporting — numbers out in the shapes the chapter asks for.
# ===========================================================================

def _fmt_tex(value: float, sig: int = 3) -> str:
    """Format a number for LaTeX **math mode**, going to ``a\\times 10^{b}``
    when the plain decimal would be unreadable."""
    if value is None or not np.isfinite(value):
        return r"\mathrm{n/a}"
    av = abs(value)
    if av != 0 and (av < 1e-2 or av >= 1e4):
        mant, exp = f"{value:.{sig - 1}e}".split("e")
        return rf"{mant}\times 10^{{{int(exp)}}}"
    return f"{value:.{sig}g}"


def placeholder_values(results: dict) -> dict:
    """Map the section's ``[slot]`` placeholders to formatted strings.

    The section source carries ``\\texttt{[d\\_hat]}``-style slots with a
    trailing comment naming the result each one wants. This returns exactly
    those slots, so :func:`fill_placeholders` (or a careful copy-paste) can
    close them.

    The strings are for **math mode**: a squared discrepancy of ``0.0012``
    comes back as ``1.20\\times 10^{-3}``, which typesets only inside ``$...$``
    (every slot in the section already sits in math). That is also why
    :func:`fill_placeholders` drops the ``\\texttt`` wrapper rather than
    filling it — ``\\texttt`` switches to text mode, where ``\\times`` and
    ``^`` are errors, and a thesis would not set a measured number in
    typewriter face anyway.

    A p-value of zero is reported as ``< 1/B`` rather than ``0``: no finite
    permutation run can license an exact zero, and writing one into a thesis
    claims a precision the test does not have.
    """
    twonn = results["twonn"]
    mmd = results["prior_fit"]
    interp = results.get("interpolation") or {}

    if mmd["n_exceed"] == 0:
        p_txt = f"< {1.0 / mmd['n_perm']:.3g}"
    else:
        p_txt = _fmt_tex(mmd["p_value"])

    def two_dp(value: float) -> str:
        """Two decimals, unless that would round a real number to 0.00."""
        return (f"{value:.2f}" if abs(value) >= 0.01 or value == 0
                else _fmt_tex(value))

    return {
        "[d\\_hat]": two_dp(twonn["d_hat"]),
        "[se]": two_dp(twonn["standard_error"]),
        "[n\\_used]": f"{twonn['n_used']}",
        "[mmd2]": _fmt_tex(mmd["mmd2"]),
        "[p]": p_txt,
        "[rho]": (f"{interp['curvature_ratio']:.3f}"
                  if interp.get("curvature_ratio") is not None
                  else r"\mathrm{n/a}"),
    }


_TEX_MACROS = {
    "[d\\_hat]": "twonnDhat",
    "[se]": "twonnSe",
    "[n\\_used]": "twonnNused",
    "[mmd2]": "mmdTwo",
    "[p]": "mmdPvalue",
    "[rho]": "curvatureRatio",
}


def write_values_tex(results: dict, path: str | Path) -> Path:
    """Write ``\\newcommand`` macros for every number the section quotes.

    The macros expand to math-mode material (see :func:`placeholder_values`),
    so use them inside ``$...$``: ``$\\hat d = \\twonnDhat \\pm \\twonnSe$``.
    """
    values = placeholder_values(results)
    lines = ["% Generated by vae_analysis.latent_geometry — do not edit by hand.",
             "% Each macro fills the like-named [slot] of the "
             "latent-space-geometry section.",
             "% The macros expand to math-mode material: use them inside $...$.",
             ""]
    for slot, macro in _TEX_MACROS.items():
        lines.append(rf"\newcommand{{\{macro}}}{{{values[slot]}}}"
                     f"  % {slot}")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def fill_placeholders(tex_source: str, results: dict) -> str:
    """Substitute the section's ``[slot]`` placeholders with the run's values.

    Rewrites ``\\texttt{[d\\_hat]}`` to ``5.31`` — the whole ``\\texttt{...}``
    token, wrapper included, because the wrapper marks a slot to fill rather
    than a font choice, and because a value like ``1.20\\times 10^{-3}`` is a
    LaTeX error inside ``\\texttt`` (see :func:`placeholder_values`). Every
    other character of the source is left untouched, including the ``% from
    [twonn]: ...`` comments, which stay as the audit trail of where each
    number came from.

    One exception, for the one case where a plain substitution would produce
    nonsense: when no permutation reached the observed statistic the p-value
    is a bound, and ``$\\hat p = \\texttt{[p]}$`` would fill to
    ``$\\hat p = < 0.002$``. The preceding ``=`` is then swallowed too, giving
    ``$\\hat p < 0.002$``.
    """
    import re

    out = tex_source
    for slot, value in placeholder_values(results).items():
        if value.lstrip().startswith("<"):
            pattern = re.compile(r"=\s*" + re.escape(rf"\texttt{{{slot}}}"))
            out = pattern.sub(lambda _m, v=value.lstrip(): v, out)
        out = out.replace(rf"\texttt{{{slot}}}", value)
    return out


def _to_python(obj):
    """Recursively convert NumPy scalars/arrays to JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def write_summary(results: dict, path: str | Path) -> Path:
    """Write the human-readable ``summary.md``: the numbers and how to read them."""
    tw = results["twonn"]
    mm = results["prior_fit"]
    kl = results["kl"]
    ip = results.get("interpolation")
    meta = results["meta"]
    v = placeholder_values(results)

    reject = mm["p_value_add_one"] < 0.05
    # A p-value that no permutation reached is a bound, so it reads "p < x".
    p_text = (f"p {v['[p]']}" if v["[p]"].lstrip().startswith("<")
              else f"p = {v['[p]']}")
    lines = [
        "# Latent-space geometry",
        "",
        f"Checkpoint: `{meta['checkpoint']}`  ",
        f"Architecture: `{meta['architecture']}`, clip {meta['clip_length']} "
        f"frames, J = {meta['n_joints']}, D = {meta['n_dims']}  ",
        f"Latent: {meta['d_latent']} flattened"
        + (f" = {meta['n_windows']} windows x d_z {meta['d_z']}"
           if meta.get("n_windows") else "") + "  ",
        f"Clips: {meta['n_clips']} ({meta['split']} split, "
        f"{meta['n_videos']} videos), encoding mask `{meta['mask_policy']}`",
        "",
        "## Intrinsic dimension (TwoNN)",
        "",
        f"- `d_hat = {tw['d_hat']:.3f} +- {tw['standard_error']:.3f}` "
        f"(least squares through the origin) over {tw['n_used']} points",
        f"- bootstrap standard error `{tw['bootstrap_se']:.3f}`, "
        f"uncentred `R^2 = {tw['r_squared']:.4f}`"
        + ("" if not (tw["bootstrap_se"] > 2 * tw["standard_error"]) else
           " — the bootstrap error is more than twice the least-squares one, "
           "which is the least-squares error understating the uncertainty "
           "(the points share one empirical distribution function and are not "
           "independent). Quote the bootstrap figure if a single `+-` has to "
           "carry the claim"),
        f"- maximum-likelihood cross-check on the untrimmed ratios: "
        f"`d_hat_mle = {tw['d_hat_mle']:.3f}`"
        + ("" if abs(tw["d_hat_mle"] - tw["d_hat"]) <= 0.1 * tw["d_hat"] else
           " — **more than 10% from the least-squares slope**, so the "
           "single-exponent Pareto model is strained; read the fit figure "
           "before quoting either number"),
        f"- ambient width {meta['d_latent']}, so the estimate is "
        f"{tw['d_hat'] / meta['d_latent']:.1%} of it: "
        + ("a low-dimensional manifold inside the flattened latent — the "
           "continuous rather than partitioned regime."
           if tw["d_hat"] < 0.5 * meta["d_latent"] else
           "NOT far below the ambient width; the low-dimensional-manifold "
           "reading does not hold here."),
        "",
        "## Prior fit (kernel two-sample test)",
        "",
        f"- `MMD^2_u = {mm['mmd2']:.6g}`, permutation `{p_text}` "
        f"over B = {mm['n_perm']} (add-one `p = {mm['p_value_add_one']:.4g}`)",
        f"- bandwidth (median heuristic) `l = {mm['bandwidth']:.4g}`, "
        f"{mm['n_used']} points a side"
        + (" (subsampled from the held-out set)" if mm["subsampled"] else ""),
        f"- verdict at 5%: **{'reject' if reject else 'do not reject'}** "
        f"q(z) = p(z)"
        + ("" if reject or mm["n_used"] >= 5 * meta["d_latent"] else
           f" — but with {mm['n_used']} points in {meta['d_latent']} "
           "dimensions the kernel test has little power, so this is weak "
           "evidence of a match rather than evidence of one"),
        f"- per-dimension KL: total `{kl['kl_total']:.3f}` nats, "
        f"`{kl['n_active']}/{kl['n_dims']}` dimensions above "
        f"{kl['threshold']:g} nats",
        "",
        "  The test speaks only to `KL(q(z) || p(z))`, the second term of the "
        "divergence decomposition; a rejection localises the mismatch to the "
        "aggregate posterior and does not indict the per-clip posteriors the "
        "state model consumes. The latent here is used for downstream "
        "analysis, not for sampling from the prior.",
        "",
    ]

    if ip:
        rho = ip["curvature_ratio"]
        lines += [
            "## Latent interpolation",
            "",
            f"- clips {ip['index_a']} and {ip['index_b']} "
            f"(videos {ip['video_a']} / {ip['video_b']}, "
            f"selection `{ip['selection']}`), latent distance "
            f"`{ip['latent_distance']:.3f}`",
            f"- curvature ratio `rho = {rho:.4f}` over {ip['n_segments']} "
            f"segments (geodesic {ip['arclength_geodesic']:.4g} / straight "
            f"{ip['arclength_straight']:.4g})",
            f"- energy {ip['energy_straight']:.4g} -> "
            f"{ip['energy_geodesic']:.4g}; constant-speed CV "
            f"{ip['speed_cv']:.4f}",
            "- " + ("`rho` is at most 1 as it should be."
                    if rho <= 1.0 + 1e-6 else
                    "**`rho` exceeds 1**, which the geometry forbids: the "
                    "energy descent has not converged. Raise `--iters` or "
                    "switch `--method adam` before quoting it."),
            "- " + (f"`rho ~ 1` ({rho:.3f}): the manifold is nearly flat "
                    "between the endpoints, so the straight latent segment is "
                    "already close to the manifold's own shortest route."
                    if rho > 0.95 else
                    f"`rho = {rho:.3f}` is well below 1: the geodesic is "
                    "materially shorter, so the straight segment is not a "
                    "good proxy for the manifold's route here."),
        ]
        if "bone_lengths" in ip:
            bl = ip["bone_lengths"]
            lines += [
                f"- bone lengths along the morph: within-clip CV mean "
                f"{bl['cv_within_mean']:.4f} / max {bl['cv_within_max']:.4f}; "
                f"drift from the endpoint mean, max {bl['drift_max']:.2%} "
                f"(at step {bl['drift_max_step']})",
            ]
        lines.append("")

    lines += ["## Section placeholders", "",
              "Math-mode LaTeX, one per `[slot]` of the section source.", "",
              "| slot | value |", "|:---|:---|"]
    lines += [f"| `{k}` | {val} |" for k, val in v.items()]
    lines += ["", "Written by `vae_analysis.latent_geometry`."]

    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


# ===========================================================================
# Driver.
# ===========================================================================

def run_latent_geometry(checkpoint_path: str | Path,
                        videos: list[np.ndarray], *,
                        out_dir: str | Path | None = None,
                        device: str = "cpu",
                        split: str = "val",
                        stride: int | None = None,
                        mask_policy: str = "none",
                        limbs: dict[str, list[int]] | None = None,
                        bones=None,
                        n_perm: int = 500,
                        max_mmd_points: int = 2000,
                        discard_fraction: float = 0.1,
                        n_bootstrap: int = 200,
                        kl_threshold: float = 0.01,
                        include_interpolation: bool = True,
                        clip_a: int | None = None,
                        clip_b: int | None = None,
                        pair_selection: str = "random",
                        n_interp_steps: int = 7,
                        n_interp_frames: int = 4,
                        n_segments: int = 32,
                        geodesic_method: str = "lbfgs",
                        geodesic_iters: int = 200,
                        invert_y: bool = False,
                        fill_tex: str | Path | None = None,
                        seed: int = 0,
                        verbose: bool = True) -> dict:
    """Load a checkpoint, encode the held-out clips, run the three diagnostics.

    The split is rebuilt exactly as training built it
    (:func:`architectures.data.train_val_split`, video-wise and
    seed-deterministic), so ``split="val"`` is the same held-out set the
    checkpoint never saw.

    Encoding uses an **all-visible** mask by default (``mask_policy="none"``).
    That is the deliberate choice: the section's latent is the clean code of a
    clip, and the interpolation endpoints come from
    ``VaeInterpolator.encode``, which is all-visible too — mixing conventions
    would put the three diagnostics on two different latents. Pass the
    checkpoint's own policy name to reproduce the training-time distribution
    instead.

    Args:
        checkpoint_path: ``.pt`` written by ``architectures.train``.
        videos: the same list of per-video ``(F, J, D)`` arrays used to train.
        out_dir: destination; defaults to
            ``<checkpoint dir>/latent_geometry/``.
        device: ``"cpu"`` or ``"cuda"``.
        split: ``"val"`` (default, held-out), ``"train"``, or ``"all"``.
        stride: clip stride; defaults to ``clip_length // 2`` as in training.
        mask_policy: policy name for encoding; ``"none"`` = all visible.
        limbs: limb map, needed only for the ``"limb"`` policy.
        bones: skeleton bone pairs; resolved from the joint count when None.
        n_perm: permutations ``B`` for the kernel test.
        max_mmd_points: cap per side of the two-sample test.
        discard_fraction: TwoNN top-tail discard.
        n_bootstrap: bootstrap resamples for the TwoNN standard error.
        kl_threshold: nats below which a latent dimension counts as collapsed.
        include_interpolation: set False to skip the third diagnostic; the
            geodesic energy descent is far the slowest step of the run.
        clip_a, clip_b: explicit endpoint indices **into the selected split**;
            None picks by ``pair_selection``.
        pair_selection: ``"random"``, ``"median"``, or ``"farthest"``.
        n_interp_steps: decoded points on the straight segment (figure columns).
        n_interp_frames: frames drawn per decoded clip (figure rows).
        n_segments: segment count for the geodesic-vs-straight comparison.
        geodesic_method, geodesic_iters: energy-descent settings.
        invert_y: flip the skeleton figure vertically.
        fill_tex: optional path to the section's ``.tex``; a filled copy is
            written next to the other outputs.
        seed: seeds every stochastic step (mask draws, prior sample,
            permutations, bootstrap, pair choice).
        verbose: print progress.
    Returns:
        Dict with ``results`` (JSON-safe), ``written`` (paths), ``latent``
        (the :class:`~vae_analysis.interfaces.LatentSet`), ``config``, and
        ``model``.
    """
    from architectures.analyze import load_checkpoint
    from architectures.data import build_clips, train_val_split
    from architectures.mask_policies import build_policy
    from .architectures_adapter import ArchitecturesAdapter
    from .interfaces import encode_dataset

    import dataclasses

    plt = _import_matplotlib()
    rng = np.random.default_rng(seed)

    def log(msg: str):
        if verbose:
            print(f"[geometry] {msg}")

    # ---- Model + held-out clips -----------------------------------------
    model, config = load_checkpoint(checkpoint_path, device=device)
    if stride is None:
        stride = config.clip_length // 2
    clips, video_id, time_index = build_clips(videos, config.clip_length, stride)
    train_mask, val_mask = train_val_split(clips, video_id)
    if split == "val":
        sel = val_mask
    elif split == "train":
        sel = train_mask
    elif split == "all":
        sel = np.ones(len(clips), dtype=bool)
    else:
        raise ValueError(f"split must be 'val', 'train', or 'all', got {split!r}.")
    X = clips[sel].astype(np.float32)
    vid = video_id[sel]
    tix = time_index[sel]
    if len(X) < 3:
        raise ValueError(f"only {len(X)} clips in the {split!r} split; the "
                         "diagnostics need more.")

    policy = build_policy(dataclasses.replace(config, mask_policy=mask_policy),
                          limbs=limbs)
    M = np.stack([policy.sample(config.clip_length, config.n_joints, rng, X=X[b])
                  for b in range(len(X))]).astype(np.float32)

    adapter = ArchitecturesAdapter(model, device=device)
    latent = encode_dataset(adapter, X, M, video_id=vid, time_index=tix)
    n_windows = adapter.n_windows() if adapter.is_temporal() else None
    d_z = adapter.d_z if adapter.is_temporal() else latent.d_z
    log(f"encoded {latent.n} clips ({split!r} split, "
        f"{len(np.unique(vid))} videos), flattened latent d = {latent.d_z}"
        + (f" = {n_windows} windows x {d_z}" if n_windows else ""))

    if bones is None:
        try:
            from youtube_motion.skeleton import skeleton_for
            bones = skeleton_for(int(config.n_joints))["bones"]
        except Exception as exc:                     # noqa: BLE001
            log(f"no skeleton for J = {config.n_joints} ({exc}); "
                "the morph figure will be scatter-only.")
            bones = None

    out_dir = Path(out_dir) if out_dir is not None \
        else Path(checkpoint_path).parent / "latent_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, stem: str):
        for ext in ("pdf", "png"):
            p = out_dir / f"{stem}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            written.append(p)
        plt.close(fig)

    results: dict = {
        "meta": {
            "checkpoint": str(checkpoint_path),
            "architecture": str(config.architecture),
            "clip_length": int(config.clip_length),
            "n_joints": int(config.n_joints),
            "n_dims": int(config.n_dims),
            "stride": int(stride),
            "split": split,
            "mask_policy": mask_policy,
            "n_clips": int(latent.n),
            "n_videos": int(len(np.unique(vid))),
            "d_latent": int(latent.d_z),
            "n_windows": None if n_windows is None else int(n_windows),
            "d_z": int(d_z),
            "seed": int(seed),
            "device": device,
        }
    }

    # ---- 1. Intrinsic dimension -----------------------------------------
    log("TwoNN intrinsic dimension ...")
    twonn = twonn_least_squares(latent.mu, discard_fraction=discard_fraction,
                                n_bootstrap=n_bootstrap, rng=rng)
    results["twonn"] = {k: v for k, v in twonn.items()
                        if k not in ("log_mu", "y", "mu")}
    log(f"  d_hat = {twonn['d_hat']:.3f} +- {twonn['standard_error']:.3f} "
        f"over {twonn['n_used']} points (MLE {twonn['d_hat_mle']:.3f})")
    save(plot_twonn(twonn, plt), "fig_twonn")

    # ---- 2. Prior fit ----------------------------------------------------
    log(f"kernel two-sample test (B = {n_perm}) ...")
    mmd = prior_fit(latent.mu, n_perm=n_perm, max_points=max_mmd_points, rng=rng)
    results["prior_fit"] = {k: v for k, v in mmd.items()
                            if k != "null_statistics"}
    log(f"  MMD^2_u = {mmd['mmd2']:.6g}, p = {mmd['p_value']:.4g} "
        f"(bandwidth {mmd['bandwidth']:.4g})")

    kl = per_dimension_kl(latent.mu, latent.logvar, threshold=kl_threshold,
                          fold=fold_with_adapter(adapter))
    results["kl"] = _to_python(kl)
    log(f"  per-dimension KL total {kl['kl_total']:.3f} nats, "
        f"{kl['n_active']}/{kl['n_dims']} active")
    save(plot_kl_per_dim(kl, plt), "fig_kl_per_dim")

    # ---- 3. Latent interpolation ----------------------------------------
    if include_interpolation:
        log("latent interpolation + curvature ratio ...")
        from .geodesic_interpolation import make_interpolator

        interp = make_interpolator(model, config=config, device=device)
        if clip_a is None or clip_b is None:
            i, j = select_clip_pair(latent.mu, vid, mode=pair_selection, rng=rng)
        else:
            i, j = int(clip_a), int(clip_b)
        if not (0 <= i < len(X) and 0 <= j < len(X)):
            raise IndexError(f"clip indices ({i}, {j}) out of range for the "
                             f"{len(X)}-clip {split!r} split.")

        ip = latent_interpolation(interp, X[i], X[j], n_steps=n_interp_steps,
                                  n_segments=n_segments, method=geodesic_method,
                                  iters=geodesic_iters, bones=bones)
        ip.update({"index_a": int(i), "index_b": int(j),
                   "video_a": int(vid[i]), "video_b": int(vid[j]),
                   "time_a": int(tix[i]), "time_b": int(tix[j]),
                   "selection": ("explicit" if clip_a is not None
                                 else pair_selection)})
        results["interpolation"] = _to_python(
            {k: v for k, v in ip.items()
             if k not in ("straight_clips", "geodesic_clips", "t", "z0", "z1")})
        log(f"  clips {i} / {j} (videos {vid[i]} / {vid[j]}), "
            f"rho = {ip['curvature_ratio']:.4f}, speed CV {ip['speed_cv']:.4f}")
        if ip["curvature_ratio"] > 1.0 + 1e-6:
            log("  WARNING: rho > 1 is geometrically impossible — the energy "
                "descent has not converged. Raise --iters or use --method adam.")

        save(plot_interpolation(ip["straight_clips"], ip["t"], bones, plt,
                                n_frame_rows=n_interp_frames, invert_y=invert_y,
                                title="Latent interpolation (straight segment)"),
             "fig_interp")
        save(plot_interpolation(ip["geodesic_clips"], ip["t"], bones, plt,
                                n_frame_rows=n_interp_frames, invert_y=invert_y,
                                title=(f"Geodesic interpolation "
                                       f"($\\rho$ = {ip['curvature_ratio']:.3f})")),
             "fig_interp_geodesic")
    else:
        log("interpolation skipped (include_interpolation=False)")

    # ---- Outputs ---------------------------------------------------------
    results["placeholders"] = placeholder_values(results)
    payload = _to_python(results)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    written.append(out_dir / "results.json")
    written.append(write_values_tex(results, out_dir / "values.tex"))
    written.append(write_summary(results, out_dir / "summary.md"))

    if fill_tex is not None:
        src = Path(fill_tex).read_text()
        dst = out_dir / (Path(fill_tex).stem + "_filled.tex")
        dst.write_text(fill_placeholders(src, results))
        written.append(dst)
        log(f"filled section written to {dst}")

    log(f"wrote {len(written)} file(s) to {out_dir}")
    return {"results": payload, "written": written, "latent": latent,
            "config": config, "model": model}


# ===========================================================================
# CLI.
# ===========================================================================

def _parse_column_map(text: str | None) -> dict | None:
    """``"part_idx=bp,video=subject"`` -> ``{"part_idx": "bp", ...}``."""
    if not text:
        return None
    out = {}
    for piece in text.split(","):
        if "=" not in piece:
            raise ValueError(f"bad --csv-columns entry {piece!r}; use k=v.")
        k, v = piece.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_videos(csv: str | None = None, npz: str | None = None, *,
                preprocess: str = "none", max_videos: int | None = None,
                columns: dict | None = None) -> list[np.ndarray]:
    """Load per-video ``(F, J, D)`` arrays from a keypoint CSV or an ``.npz``.

    The CSV path is the repo's long-format loader
    (:func:`youtube_motion.data.load_youtube_csv`), which auto-detects the
    joint count and interpolates missing detections. Column names differing
    from the default export (``video, frame, part_idx, x, y``) are remapped
    with ``columns`` — the RVI-38 table, for instance, calls the joint index
    ``bp``.

    The ``.npz`` path reads the ``pose.npz`` cache written by
    ``rvi38_methods.build_pose`` (a ``videos`` key naming ``x_<id>`` arrays),
    and falls back to "every 3-D array in the archive, in file order".
    """
    if (csv is None) == (npz is None):
        raise ValueError("pass exactly one of --csv or --npz.")
    if csv is not None:
        from youtube_motion.data import load_youtube_csv
        bundle = load_youtube_csv(csv, preprocess=preprocess,
                                  max_videos=max_videos, columns=columns)
        print(f"[data] {bundle.summary()}")
        return bundle.videos

    with np.load(npz, allow_pickle=True) as d:
        if "videos" in d.files:
            names = [str(v) for v in d["videos"]]
            vids = [np.asarray(d[f"x_{n}"], dtype=np.float32) for n in names]
        else:
            vids = [np.asarray(d[k], dtype=np.float32) for k in d.files
                    if np.asarray(d[k]).ndim == 3]
    if not vids:
        raise ValueError(f"no per-video (F, J, D) arrays found in {npz}.")
    if max_videos:
        vids = vids[:max_videos]
    print(f"[data] {len(vids)} videos, "
          f"{sum(v.shape[0] for v in vids)} frames, shape {vids[0].shape[1:]}")
    return vids


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Latent-space geometry: TwoNN intrinsic dimension, the "
                    "kernel prior-fit test, and latent interpolation with the "
                    "curvature ratio, on one trained checkpoint.")
    ap.add_argument("--checkpoint", required=True,
                    help="`.pt` written by architectures.train")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="long-format keypoint CSV")
    src.add_argument("--npz", help="pose.npz cache of per-video arrays")
    ap.add_argument("--csv-columns", default=None,
                    help="column overrides, e.g. 'part_idx=bp,video=subject'")
    ap.add_argument("--preprocess", default="none",
                    choices=["none", "center", "center_scale"])
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--out-dir", default=None,
                    help="default: <checkpoint dir>/latent_geometry/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--mask-policy", default="none",
                    help="encoding mask policy; 'none' = every joint visible")
    ap.add_argument("--n-perm", type=int, default=500,
                    help="permutations B for the kernel test")
    ap.add_argument("--max-mmd-points", type=int, default=2000,
                    help="cap per side of the two-sample test")
    ap.add_argument("--discard-fraction", type=float, default=0.1,
                    help="TwoNN top-tail discard")
    ap.add_argument("--n-bootstrap", type=int, default=200,
                    help="bootstrap resamples for the TwoNN standard error")
    ap.add_argument("--kl-threshold", type=float, default=0.01,
                    help="nats below which a latent dimension is collapsed")
    ap.add_argument("--no-interpolation", action="store_true",
                    help="skip the third diagnostic (the slowest step)")
    ap.add_argument("--clip-a", type=int, default=None,
                    help="endpoint index into the selected split")
    ap.add_argument("--clip-b", type=int, default=None)
    ap.add_argument("--pair", default="random",
                    choices=["random", "median", "farthest"],
                    help="how the endpoint pair is chosen when not given")
    ap.add_argument("--interp-steps", type=int, default=7,
                    help="decoded points on the straight segment (figure columns)")
    ap.add_argument("--interp-frames", type=int, default=4,
                    help="frames drawn per decoded clip (figure rows)")
    ap.add_argument("--segments", type=int, default=32,
                    help="segments of the geodesic-vs-straight comparison")
    ap.add_argument("--method", default="lbfgs", choices=["lbfgs", "adam"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--invert-y", action="store_true",
                    help="flip the skeleton figure (y-down exports)")
    ap.add_argument("--fill-tex", default=None,
                    help="section .tex whose [slots] to fill from this run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    videos = load_videos(args.csv, args.npz, preprocess=args.preprocess,
                         max_videos=args.max_videos,
                         columns=_parse_column_map(args.csv_columns))
    run_latent_geometry(
        args.checkpoint, videos,
        out_dir=args.out_dir, device=args.device, split=args.split,
        stride=args.stride, mask_policy=args.mask_policy,
        n_perm=args.n_perm, max_mmd_points=args.max_mmd_points,
        discard_fraction=args.discard_fraction, n_bootstrap=args.n_bootstrap,
        kl_threshold=args.kl_threshold,
        include_interpolation=not args.no_interpolation,
        clip_a=args.clip_a, clip_b=args.clip_b, pair_selection=args.pair,
        n_interp_steps=args.interp_steps, n_interp_frames=args.interp_frames,
        n_segments=args.segments, geodesic_method=args.method,
        geodesic_iters=args.iters, invert_y=args.invert_y,
        fill_tex=args.fill_tex, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
