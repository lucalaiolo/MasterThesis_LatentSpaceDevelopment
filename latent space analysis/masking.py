"""Mask-invariance and robustness (Part I Section 6).

The encoder sees the mask, so changing the mask at a fixed clip moves the
posterior. An encoder that learned pose structure moves little. These
tools measure that movement, the graceful degradation under heavier
masking, and, for the inpainting recipe, the split of error between
visible and hidden joints.
"""

from __future__ import annotations

import numpy as np


def _uniform_mask(shape, rho, rng):
    """Draw a uniform mask that hides a fraction rho of joints per frame."""
    keep = rng.random(shape) > rho
    return keep.astype(np.float32)


def _limb_mask(T, J, limb_joints, rng):
    """Hide a whole limb for the clip."""
    M = np.ones((T, J), np.float32)
    M[:, limb_joints] = 0.0
    return M


def mask_jitter(model, clips: np.ndarray, mask_sampler, k: int = 16,
                rng: np.random.Generator | None = None) -> dict:
    """Dispersion of a clip's latent under repeated mask draws (Section 6.1).

    Encodes each clip k times with fresh masks and measures how far the
    posterior means scatter, then reports that scatter relative to the
    between-clip variance. A ratio near one means the mask draw, not the
    pose, sets the latent; below about 0.1 is the target.

    Args:
        model: a VAEModel.
        clips: shape (A, T, J, 3).
        mask_sampler: callable (T, J, rng) -> mask (T, J).
        k: mask draws per clip.
    Returns:
        Dict with the per-clip dispersion, the between-clip variance, and
        their ratio.
    """
    rng = np.random.default_rng() if rng is None else rng
    A, T, J, _ = clips.shape
    means = np.zeros((A, model.encode(clips[:1],
                     np.ones((1, T, J)))[0].shape[1]))
    disp = np.zeros(A)
    for a in range(A):
        stack = np.repeat(clips[a][None], k, axis=0)
        masks = np.stack([mask_sampler(T, J, rng) for _ in range(k)])
        mu, _ = model.encode(stack, masks)
        bar = mu.mean(axis=0)
        means[a] = bar
        disp[a] = np.mean(np.sum((mu - bar) ** 2, axis=1))
    between = float(np.mean(np.var(means, axis=0)))
    return {"dispersion": disp, "between_var": between,
            "ratio": float(disp.mean() / (between + 1e-12))}


def latent_recovery(model, clips: np.ndarray, skeleton,
                    fractions=(0.1, 0.3, 0.5, 0.7),
                    rng: np.random.Generator | None = None) -> dict:
    """Latent drift as masking grows heavier (Section 6.2).

    Encodes each clip with the empty mask, then with uniform masks of
    rising severity and with per-limb masks. Reports the mean latent
    distance from the empty-mask latent for each condition. A steep rise
    past half-masking says the encoder leans on visible joints.

    Args:
        model: a VAEModel.
        clips: shape (A, T, J, 3).
        skeleton: a Skeleton; its limbs drive the limb-mask conditions.
        fractions: uniform mask severities to test.
    Returns:
        Dict of condition name to mean recovery error.
    """
    rng = np.random.default_rng() if rng is None else rng
    A, T, J, _ = clips.shape
    full = np.ones((A, T, J), np.float32)
    mu_full, _ = model.encode(clips, full)

    out = {}
    for rho in fractions:
        masks = np.stack([_uniform_mask((T, J), rho, rng) for _ in range(A)])
        mu, _ = model.encode(clips, masks)
        out[f"uniform_{rho}"] = float(np.mean(np.linalg.norm(mu - mu_full, axis=1)))
    for name, joints in skeleton.limbs.items():
        masks = np.stack([_limb_mask(T, J, joints, rng) for _ in range(A)])
        mu, _ = model.encode(clips, masks)
        out[f"limb_{name}"] = float(np.mean(np.linalg.norm(mu - mu_full, axis=1)))
    return out


def split_mpjpe(X_true: np.ndarray, X_pred: np.ndarray, M: np.ndarray) -> dict:
    """Mean per-joint position error split by visibility (Section 6.3).

    The mean per-joint position error (MPJPE) is the average joint
    distance between truth and reconstruction. Recipe 3's inpainting head
    should be judged on the hidden joints; the ratio of hidden to visible
    error is the fair test.

    Args:
        X_true, X_pred: clips, shape (N, T, J, 3).
        M: masks, shape (N, T, J), 1 for visible.
    Returns:
        Dict with visible error, hidden (inpainted) error, and the ratio.
    """
    err = np.linalg.norm(X_pred - X_true, axis=-1)  # (N, T, J)
    vis = M > 0.5
    hid = ~vis
    e_vis = err[vis].mean() if vis.any() else np.nan
    e_hid = err[hid].mean() if hid.any() else np.nan
    return {"mpjpe_visible": float(e_vis), "mpjpe_inpainted": float(e_hid),
            "ratio": float(e_hid / (e_vis + 1e-12))}


# Mask samplers ready to pass to `mask_jitter`.
def uniform_sampler(rho: float):
    """Return a sampler that hides a fraction rho of joints per frame."""
    def sampler(T, J, rng):
        return _uniform_mask((T, J), rho, rng)
    return sampler


def limb_sampler(skeleton):
    """Return a sampler that hides one random named limb per clip."""
    names = list(skeleton.limbs)

    def sampler(T, J, rng):
        name = names[rng.integers(len(names))]
        return _limb_mask(T, J, skeleton.limbs[name], rng)
    return sampler
