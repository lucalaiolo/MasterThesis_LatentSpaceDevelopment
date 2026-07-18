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
               stride_seconds: float = 1.0, seed: int = 0) -> dict:
    """Fit a Gaussian hidden Markov model and pick the state count by BIC.

    A hidden Markov model (HMM) treats the trajectory as jumps between a
    few hidden states, each a Gaussian in latent space. Reports the
    transition matrix, per-state means, occupancy, and mean dwell time in
    seconds.

    Args:
        trajectory: coarse latents, shape (K, d_z).
        k_range: candidate state counts.
        stride_seconds: real time between latent samples, for dwell times.
    Returns:
        Dict with the chosen model and its summaries.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("hmm_states needs hmmlearn; pip install hmmlearn.") from e

    # Cap the state count by the trajectory length: fitting more states than
    # the data can occupy leaves a state unvisited, whose transition row sums
    # to zero — which hmmlearn rejects ("transmat_ rows must sum to 1"). Keep
    # at least ~5 windows per state. For a long trajectory this leaves the
    # default range (2..8) untouched.
    n = len(trajectory)
    k_max = min(max(2, n // 5), n - 1)
    ks = [k for k in k_range if 2 <= k <= k_max]
    if not ks:
        raise ValueError(
            f"trajectory too short ({n} windows) to fit an HMM; need at least "
            f"~10 windows (encode a longer video or shorten the window/stride).")

    best, best_bic, best_k = None, np.inf, None
    for k in ks:
        try:
            hmm = GaussianHMM(n_components=k, covariance_type="full",
                              n_iter=200, random_state=seed)
            hmm.fit(trajectory)
            ll = hmm.score(trajectory)
        except Exception:  # noqa: BLE001 - a degenerate fit at this k; skip it
            continue
        n_params = k * trajectory.shape[1] + k * trajectory.shape[1] + k * k
        bic = -2 * ll + n_params * np.log(n)
        if bic < best_bic:
            best, best_bic, best_k = hmm, bic, k
    if best is None:
        raise ValueError("no HMM converged for any candidate state count.")

    states = best.predict(trajectory)
    occ = np.array([np.mean(states == s) for s in range(best_k)])
    dwell = stride_seconds / (1.0 - np.diag(best.transmat_) + 1e-12)
    return {"model": best, "k": best_k, "states": states,
            "transition": best.transmat_, "means": best.means_,
            "occupancy": occ, "dwell_seconds": dwell}


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
