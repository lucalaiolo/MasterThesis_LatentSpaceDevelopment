"""Between-clip dynamics (Part I Section 7, Part II Section 22).

A per-clip encoder folds fast motion into one latent. Slow structure
across the recording comes from encoding a video in a sliding window and
reading the resulting coarse latent trajectory. Three models: change
points, a Gaussian hidden Markov model, and a continuous-time
Ornstein-Uhlenbeck process that is honest about the window overlap.
"""

from __future__ import annotations

import numpy as np


def sliding_windows(video: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Cut a video into overlapping clips.

    Args:
        video: one recording, shape (F, J, 3).
        window: clip length T.
        stride: hop between clip starts, s.
    Returns:
        Clips of shape (K, window, J, 3).
    """
    F = video.shape[0]
    starts = range(0, F - window + 1, stride)
    return np.stack([video[s:s + window] for s in starts])


def encode_video(model, video: np.ndarray, window: int, stride: int,
                 mask=None) -> np.ndarray:
    """Encode a video into its coarse latent trajectory.

    Returns:
        Posterior means, shape (K, d_z), at latent sample spacing `stride`.
    """
    clips = sliding_windows(video, window, stride)
    if mask is None:
        mask = np.ones(clips.shape[:3], np.float32)
    mu, _ = model.encode(clips, mask)
    return mu


def change_points(trajectory: np.ndarray, penalty: float = 10.0) -> dict:
    """Segment a latent trajectory into stable regimes (Section 7.1).

    Uses the ruptures library with a Gaussian mean-shift cost when it is
    present; otherwise a light built-in that splits where the running mean
    jumps. Overlap inflates change counts, so encode with non-overlapping
    windows for this one.

    Returns:
        Dict with the break indices and the segment count.
    """
    try:
        import ruptures as rpt
        algo = rpt.Pelt(model="normal").fit(trajectory)
        bkps = algo.predict(pen=penalty)
        return {"breaks": bkps, "n_segments": len(bkps)}
    except ImportError:
        # Fallback: cumulative-mean jump detector. Coarse but dependency-free.
        K = len(trajectory)
        cost = np.zeros(K)
        for k in range(1, K):
            left = trajectory[:k].mean(axis=0)
            right = trajectory[k:].mean(axis=0)
            cost[k] = np.linalg.norm(left - right)
        thresh = cost.mean() + cost.std()
        breaks = list(np.where(cost > thresh)[0]) + [K]
        return {"breaks": breaks, "n_segments": len(breaks),
                "note": "install ruptures for PELT; this is a fallback"}


def hmm_states(trajectory: np.ndarray, k_range=range(2, 9),
               stride_seconds: float = 1.0, seed: int = 0,
               covariance_type: str = "diag") -> dict:
    """Fit a Gaussian hidden Markov model and pick the state count by BIC.

    A hidden Markov model (HMM) treats the trajectory as jumps between a
    few hidden states, each a Gaussian in latent space. Reports the
    transition matrix, per-state means, occupancy, and mean dwell time in
    seconds.

    Args:
        trajectory: coarse latents, shape (K, d_z).
        k_range: candidate state counts.
        stride_seconds: real time between latent samples, for dwell times.
        covariance_type: per-state covariance model ("diag" by default).
            ``"full"`` costs ``d_z*(d_z+1)/2`` parameters per state, which
            overwhelms a short trajectory in a wide latent (a 32-dim full
            covariance is 528 numbers per state) and yields a degenerate,
            non-converging fit; ``"diag"`` costs only ``d_z`` per state and
            is the robust default for regime segmentation.
    Returns:
        Dict with the chosen model and its summaries.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("hmm_states needs hmmlearn; pip install hmmlearn.") from e

    n, f = len(trajectory), int(trajectory.shape[1])

    def _n_params(k: int) -> int:
        """Free scalar parameters of a k-state GaussianHMM at this cov type."""
        trans = k * (k - 1) + (k - 1)              # transmat + startprob
        means = k * f
        if covariance_type == "full":
            cov = k * f * (f + 1) // 2
        elif covariance_type == "tied":
            cov = f * (f + 1) // 2
        elif covariance_type == "spherical":
            cov = k
        else:                                       # "diag"
            cov = k * f
        return trans + means + cov

    # Cap the state count two ways. (1) Enough windows to occupy each state
    # (~5 per state): fewer, and a state goes unvisited, whose transition row
    # sums to zero — which hmmlearn rejects. (2) Few enough parameters not to
    # exceed the data (n*f scalar observations): more parameters than data
    # gives a singular covariance and an EM that will not converge. For a long
    # trajectory in a modest latent both caps leave the default range untouched.
    budget = n * f
    k_max = min(max(2, n // 5), n - 1)
    ks = [k for k in k_range if 2 <= k <= k_max and _n_params(k) <= budget]
    if not ks:
        raise ValueError(
            f"trajectory too short ({n} windows, dim {f}) to fit an HMM without "
            f"overfitting; encode a longer video, shorten the window/stride, or "
            f"reduce the latent dimension before the HMM.")

    # We sweep k and select by BIC, deliberately skipping fits that fail to
    # converge, so hmmlearn's per-candidate chatter is noise here. hmmlearn
    # emits it through both `warnings` (the degenerate-params note) and its
    # `logging` logger (the "not converging" / zero-transmat-row notes), so
    # silence both channels for the duration of the sweep.
    import logging
    import warnings
    hmm_logger = logging.getLogger("hmmlearn")
    prev_level = hmm_logger.level
    best, best_bic, best_k = None, np.inf, None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="hmmlearn")
        hmm_logger.setLevel(logging.ERROR)
        try:
            for k in ks:
                try:
                    hmm = GaussianHMM(n_components=k,
                                      covariance_type=covariance_type,
                                      n_iter=200, random_state=seed)
                    hmm.fit(trajectory)
                    ll = hmm.score(trajectory)
                except Exception:  # noqa: BLE001 - degenerate fit at this k; skip
                    continue
                if not np.isfinite(ll):
                    continue
                bic = -2 * ll + _n_params(k) * np.log(n)
                if bic < best_bic:
                    best, best_bic, best_k = hmm, bic, k
        finally:
            hmm_logger.setLevel(prev_level)
    if best is None:
        raise ValueError("no HMM converged for any candidate state count.")

    states = best.predict(trajectory)
    occ = np.array([np.mean(states == s) for s in range(best_k)])

    # Closed-form mean dwell 1/(1 - p_ii) blows up for a near-absorbing state
    # (p_ii -> 1), so it is clipped to the trajectory duration — a state cannot
    # dwell longer than the recording. The empirical dwell (mean observed
    # run-length per state) is the honest, always-bounded readout.
    total_seconds = n * stride_seconds
    dwell = np.minimum(stride_seconds / (1.0 - np.diag(best.transmat_) + 1e-12),
                       total_seconds)
    emp = np.full(best_k, np.nan)
    for s in range(best_k):
        runs, run = [], 0
        for st in states:
            if st == s:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
        if runs:
            emp[s] = float(np.mean(runs)) * stride_seconds
    return {"model": best, "k": best_k, "states": states,
            "transition": best.transmat_, "means": best.means_,
            "occupancy": occ, "dwell_seconds": dwell,
            "empirical_dwell_seconds": emp}


def ou_process(trajectory: np.ndarray, stride_seconds: float = 1.0) -> dict:
    """Fit an Ornstein-Uhlenbeck process to the outer trajectory (Section 22).

    The Ornstein-Uhlenbeck process is the simplest mean-reverting
    continuous process. Fitting its first-order autoregression and taking
    a matrix logarithm recovers the reversion matrix, whose eigenvalue
    reciprocals are the return timescales in seconds. This respects the
    irregular information content that window overlap creates better than
    a discrete state model.

    Returns:
        Dict with the reversion matrix, its eigenvalues, and the
        timescales in seconds.
    """
    from scipy.linalg import logm

    M_minus = trajectory[:-1].T           # (d_z, K-1)
    M_plus = trajectory[1:].T             # (d_z, K-1)
    # Least-squares transition A_delta = M_plus M_minus^T (M_minus M_minus^T)^-1
    cov = M_minus @ M_minus.T
    A_delta = M_plus @ M_minus.T @ np.linalg.pinv(cov)
    Theta = -np.real(logm(A_delta)) / stride_seconds
    eig = np.linalg.eigvals(Theta)
    rates = np.real(eig)
    timescales = 1.0 / np.clip(rates, 1e-6, None)
    return {"reversion": Theta, "eigenvalues": eig,
            "timescales_seconds": np.sort(timescales)}
