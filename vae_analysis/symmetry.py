"""Symmetry and equivariance (Part II Section 15).

The infant body has a left-right symmetry, and asymmetry of movement is a
clinical marker. If a linear map on the latent reproduces a body mirror,
that map is an involution, and its negative-eigenvalue subspace is the
laterality code. Projecting onto it gives an asymmetry score with no
asymmetry labels.
"""

from __future__ import annotations

import numpy as np


def flip_clips(X: np.ndarray, skeleton) -> np.ndarray:
    """Mirror a batch of clips left to right.

    Swaps left and right joints and negates the lateral coordinate.

    Args:
        X: clips, shape (N, T, J, 3).
        skeleton: a Skeleton with left_right pairs and lateral_axis.
    Returns:
        Mirrored clips, same shape.
    """
    P = skeleton.flip_permutation()               # (J, J)
    Xf = np.einsum("jk,ntkc->ntjc", P, X)         # swap joints
    Xf = Xf.copy()
    Xf[..., skeleton.lateral_axis] *= -1.0        # mirror the axis
    return Xf


def flip_masks(M: np.ndarray, skeleton) -> np.ndarray:
    """Apply the left-right joint swap to masks."""
    P = skeleton.flip_permutation()
    return np.einsum("jk,ntk->ntj", P, M)


def fit_equivariance(model, X: np.ndarray, M: np.ndarray, skeleton) -> dict:
    """Fit the linear map that reproduces the mirror on the latent (Section 15.2).

    Encodes the clips and their mirrors, then fits mu(mirror) ~ A mu + b by
    least squares. The variance explained says how linearly the latent
    codes laterality.

    Args:
        model: a VAEModel.
        X: clips, shape (N, T, J, 3).
        M: masks, shape (N, T, J).
        skeleton: a Skeleton.
    Returns:
        Dict with A, b, and the fraction of variance explained.
    """
    mu, _ = model.encode(X, M)
    Xf, Mf = flip_clips(X, skeleton), flip_masks(M, skeleton)
    mu_f, _ = model.encode(Xf, Mf)

    # Solve [mu | 1] W = mu_f in least squares; W holds A^T stacked with b.
    ones = np.ones((len(mu), 1))
    design = np.hstack([mu, ones])
    W, *_ = np.linalg.lstsq(design, mu_f, rcond=None)
    A = W[:-1].T
    b = W[-1]

    pred = mu @ A.T + b
    ss_res = np.sum((mu_f - pred) ** 2)
    ss_tot = np.sum((mu_f - mu_f.mean(0)) ** 2) + 1e-12
    return {"A": A, "b": b, "variance_explained": float(1 - ss_res / ss_tot)}


def laterality_subspace(A: np.ndarray, tol: float = 0.0) -> dict:
    """Split the latent into mirror-symmetric and mirror-antisymmetric parts.

    The fit map is close to an involution, so its eigenvalues cluster at
    plus and minus one. The minus-one eigenvectors span the laterality
    code (Section 15.3).

    Args:
        A: the fitted linear map, shape (d_z, d_z).
        tol: eigenvalues below this real part count as the minus side.
    Returns:
        Dict with the eigenvalues, the projector onto the antisymmetric
        subspace, and that subspace's dimension.
    """
    w, V = np.linalg.eig(A)
    neg = np.real(w) < tol
    basis = np.real(V[:, neg])
    # Orthonormalise the antisymmetric basis for a clean projector.
    if basis.shape[1] > 0:
        Q, _ = np.linalg.qr(basis)
        proj = Q @ Q.T
    else:
        proj = np.zeros_like(A)
    return {"eigenvalues": w, "projector": proj,
            "antisymmetric_dim": int(neg.sum())}


def asymmetry_score(latent, projector: np.ndarray) -> np.ndarray:
    """Per-clip asymmetry as the latent's length in the laterality subspace.

    Args:
        latent: a LatentSet.
        projector: the antisymmetric projector from `laterality_subspace`.
    Returns:
        Score per clip, shape (N,).
    """
    proj = latent.mu @ projector.T
    return np.linalg.norm(proj, axis=1)
