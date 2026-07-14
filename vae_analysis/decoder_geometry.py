"""Decoder-side geometry (Part I Section 4, Part II Section 17).

The decoder maps one latent to a whole clip, so its derivative tells you
which latent moves which joint and when. The metric it induces on the
latent gives the honest notion of distance, and geodesics under that
metric give the honest traversal.

The Jacobian tools need torch and a model that implements `decode_torch`.
The traversal-measurement tools work on decoded arrays and need only
NumPy.
"""

from __future__ import annotations

import numpy as np


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("Decoder Jacobian tools need PyTorch. Install it "
                          "or use the array-based traversal tools instead.") from e


def decoder_jacobian(model, z: np.ndarray):
    """Decoder Jacobian at one latent.

    Args:
        model: a VAEModel with a differentiable `decode_torch`.
        z: latent, shape (d_z,).
    Returns:
        Jacobian array of shape (T, J, 3, d_z).
    """
    torch = _require_torch()
    from torch.func import jacrev

    z_t = torch.as_tensor(z, dtype=torch.float32)
    J = jacrev(model.decode_torch)(z_t)  # (T, J, 3, d_z), on model.device
    return np.asarray(J.detach().cpu())


def sensitivity_maps(model, anchors: np.ndarray) -> dict:
    """Average decoder sensitivity over a set of anchor latents (Section 4.1).

    Returns two maps. The joint-by-latent map answers "which latent moves
    which joint"; the time-by-latent map answers "when in the clip does a
    latent act".

    Args:
        model: a VAEModel.
        anchors: latents to average over, shape (A, d_z).
    Returns:
        Dict with `joint_latent` (J, d_z) and `time_latent` (T, d_z).
    """
    S_jl, S_tl = None, None
    for z in anchors:
        Jc = decoder_jacobian(model, z)          # (T, J, 3, d_z)
        sq = Jc ** 2
        jl = np.sqrt(sq.sum(axis=(0, 2)) / Jc.shape[0])   # (J, d_z)
        tl = np.sqrt(sq.sum(axis=(1, 2)) / Jc.shape[1])   # (T, d_z)
        S_jl = jl if S_jl is None else S_jl + jl
        S_tl = tl if S_tl is None else S_tl + tl
    return {"joint_latent": S_jl / len(anchors),
            "time_latent": S_tl / len(anchors)}


def measured_traversal(model, z_star: np.ndarray, skeleton,
                       steps=(-3, -2, -1, 0, 1, 2, 3)) -> dict:
    """Decode single-dimension traversals and measure their effect (Section 4.2).

    For each latent dimension and each step, decode z_star + step * e_i and
    record per-joint displacement, a laterality index, and bone stretch.

    Args:
        model: a VAEModel with a batch `decode`.
        z_star: reference latent, shape (d_z,).
        skeleton: a Skeleton for bones and left-right pairs.
        steps: the multiples of the unit vector to walk.
    Returns:
        Dict with `displacement` (d_z, n_steps, J), `laterality`
        (d_z, n_steps, n_pairs), and `bone_stretch` (d_z, n_steps, n_bones).
    """
    d_z = len(z_star)
    steps = np.asarray(steps, dtype=float)
    zero_idx = int(np.argmin(np.abs(steps)))

    base = model.decode(z_star[None])[0]  # (T, J, 3)
    bones = skeleton.bone_index()

    disp = np.zeros((d_z, len(steps), skeleton.n_joints))
    lat = np.zeros((d_z, len(steps), len(skeleton.left_right)))
    stretch = np.zeros((d_z, len(steps), len(bones)))

    for i in range(d_z):
        batch = np.array([z_star + a * np.eye(d_z)[i] for a in steps])
        dec = model.decode(batch)  # (n_steps, T, J, 3)
        for s in range(len(steps)):
            d = np.linalg.norm(dec[s] - base, axis=-1).mean(axis=0)  # (J,)
            disp[i, s] = d
            for p, (jl, jr) in enumerate(skeleton.left_right):
                lat[i, s, p] = d[jl] - d[jr]
            for b, (ja, jb) in enumerate(bones):
                len_s = np.linalg.norm(dec[s, :, ja] - dec[s, :, jb], axis=-1)
                len_0 = np.linalg.norm(base[:, ja] - base[:, jb], axis=-1)
                stretch[i, s, b] = (len_s - len_0).mean()
    return {"displacement": disp, "laterality": lat, "bone_stretch": stretch,
            "steps": steps, "zero_step": zero_idx}


def pullback_metric(model, z: np.ndarray) -> np.ndarray:
    """Pullback metric G(z) = J^T J at one latent (Section 4.3).

    The eigenvalues of G say how strongly each latent direction moves the
    decoded clip. A large condition number means Euclidean latent distance
    is a poor proxy for pose distance.

    Returns:
        The (d_z, d_z) metric matrix.
    """
    Jc = decoder_jacobian(model, z)               # (T, J, 3, d_z)
    Jm = Jc.reshape(-1, Jc.shape[-1])             # (T*J*3, d_z)
    return Jm.T @ Jm


def metric_spectrum(model, anchors: np.ndarray) -> dict:
    """Eigenvalue spectrum and condition number of the pullback metric.

    Averages over anchor latents. A mean condition number above about 10
    argues for the geodesic traversal over the straight line.
    """
    eigs, conds = [], []
    for z in anchors:
        G = pullback_metric(model, z)
        w = np.linalg.eigvalsh(G)
        w = np.clip(w, 0, None)
        eigs.append(w)
        pos = w[w > 0]
        conds.append(pos.max() / pos.min() if len(pos) else np.inf)
    return {"eigenvalues": np.array(eigs), "condition": np.array(conds),
            "mean_condition": float(np.mean(conds))}


def geodesic(model, z_a: np.ndarray, z_b: np.ndarray, n_points: int = 16,
             n_iter: int = 200, step: float = 1e-2, eps: float = 1e-3) -> np.ndarray:
    """Approximate the metric geodesic between two latents (Section 17).

    Minimises the discrete path energy sum_k (gamma_{k+1} - gamma_k)^T
    G(mid) (gamma_{k+1} - gamma_k) by gradient descent on the interior
    control points, with the metric estimated at each segment midpoint.
    Endpoints stay fixed. This is the relaxation route; it needs only the
    metric, not its derivatives.

    Args:
        model: a VAEModel.
        z_a, z_b: endpoints, shape (d_z,).
        n_points: control points including the two endpoints.
        n_iter: descent steps.
        step: descent step size.
        eps: unused placeholder for a metric-derivative variant.
    Returns:
        The path, shape (n_points, d_z).
    """
    ts = np.linspace(0, 1, n_points)[:, None]
    path = (1 - ts) * z_a[None] + ts * z_b[None]

    def energy_grad(path):
        grad = np.zeros_like(path)
        for k in range(len(path) - 1):
            mid = 0.5 * (path[k] + path[k + 1])
            G = pullback_metric(model, mid)
            diff = path[k + 1] - path[k]
            g = 2.0 * (G @ diff)
            grad[k + 1] += g
            grad[k] -= g
        grad[0] = 0.0
        grad[-1] = 0.0
        return grad

    for _ in range(n_iter):
        path = path - step * energy_grad(path)
    return path


def path_curvature(model, path: np.ndarray) -> float:
    """Frobenius curvature of a decoded path (used to compare traversals).

    Decodes each latent on the path and sums the norm of the discrete
    second difference of the pose sequence. Lower means smoother.
    """
    dec = model.decode(path)  # (P, T, J, 3)
    second = dec[2:] - 2 * dec[1:-1] + dec[:-2]
    return float(np.sqrt((second ** 2).sum(axis=(1, 2, 3))).sum())
