"""Geodesic interpolation between two clips in the VAE latent space.

Method: **discrete geodesic-energy minimisation** (Shao, Kumar & Fletcher,
*The Riemannian Geometry of Deep Generative Models*, CVPR-W 2018, Eqs. 4-5).
The geodesic is computed from the decoder's **first derivative only** — we never
integrate the continuous geodesic ODE (that needs the Christoffel symbols, hence
derivatives of the metric ``G = Jᵀ J``, hence second derivatives of the decoder).

The interpolant between two clips ``x0, x1`` is the minimum-energy curve joining
their latent codes ``z0 = h(x0), z1 = h(x1)`` under the metric that the decoder
pulls back from pose space. Decoding the nodes of that curve gives a sequence of
intermediate clips that stay on the manifold of decodable motion, unlike the
decoded straight latent segment (the baseline, :func:`straight_path`).

Discrete energy, evaluated **in pose space with no metric ever formed**
(Shao Eq. 4; the sum runs to ``T-1``)::

    E(z_1, ..., z_{T-1}) = (T / 2) * sum_{i=0}^{T-1} || g(z_{i+1}) - g(z_i) ||^2

with ``1/dt = T`` the factor that makes the minimiser constant-speed. Reverse-mode
autodiff of the scalar ``E`` w.r.t. the interior nodes returns Shao Eq. 5 exactly,
because a backward pass computes the vector-Jacobian product ``Jᵢᵀ v`` directly —
so the whole gradient is one ``E.backward()`` over a batched decode. The closed
form (:func:`energy_grad_closed_form`) is kept only for the gradient-check test.

Assumptions wired against this repo's ``architectures`` VAE (see
:class:`VaeInterpolator`):

1. **Mean head only.** ``g_theta == mu_theta`` (``decode_full``). Under the fixed
   decoder variance used in this thesis the expected pullback metric reduces to
   ``Ḡ = J_mu^T J_mu`` (Arvanitidis et al. 2018), so the variance head never
   enters. The decoder is never sampled.
2. **Deterministic forward.** ``model.eval()``; decoder parameters frozen. The
   decoded clip is a pure function of ``z``.
3. **Output units.** The Euclidean norm inherits the decoder's output units. For
   the YouTube / COCO-18 export those are already torso-normalised keypoint
   coordinates in the canonical frame (``preprocess="none"``), so the
   decode-to-keypoints map ``Phi`` is the identity. Pass ``coord_transform`` to
   compose a non-trivial ``Phi``; an affine ``Phi`` only rescales the metric.
4. **Optional joint weighting.** Pass ``weight`` (a fixed diagonal ``W``,
   broadcastable to ``(T, J, D)``) to emphasise e.g. wrists/ankles; this replaces
   ``||.||`` by ``||W.||`` in the energy. Default: no weighting (``W = I``).

References
    H. Shao, A. Kumar, P. T. Fletcher. The Riemannian Geometry of Deep
    Generative Models. CVPR-W 2018 (arXiv:1711.08014). Method = Section 3.1,
    Eqs. 4-5, Algorithm 1.
    G. Arvanitidis, L. K. Hansen, S. Hauberg. Latent Space Oddity. ICLR 2018
    (arXiv:1710.11379). Pullback metric, energy vs length, expected metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "Geodesic interpolation needs PyTorch. `pip install torch`."
        ) from e


# ===========================================================================
# Energy and arclength — pose space, no metric formed (Shao Eq. 4).
# ===========================================================================

def curve_energy(decoded_nodes):
    """Discrete curve energy of a decoded path (Shao Eq. 4).

    Args:
        decoded_nodes: ``(T+1, D)`` decoded nodes ``g(z_0), ..., g(z_T)`` in pose
            space (flattened). Must carry grad w.r.t. the interior nodes for the
            autodiff gradient.
    Returns:
        The scalar energy ``(T/2) * sum_i ||g(z_{i+1}) - g(z_i)||^2``.
    """
    T = decoded_nodes.shape[0] - 1
    diffs = decoded_nodes[1:] - decoded_nodes[:-1]          # (T, D)
    return 0.5 * T * diffs.pow(2).sum(dim=1).sum()          # 1/dt = T


def decoded_arclength(decode: Callable, Z):
    """Sum of Euclidean segment lengths of the decoded curve ``g(Z)`` in pose space.

    Args:
        decode: adapter mapping ``(M, d_z) -> (M, D)`` (flattened, deterministic).
        Z: ``(T+1, d_z)`` latent nodes.
    Returns:
        Scalar tensor, the decoded polyline length.
    """
    Gn = decode(Z)                                         # (T+1, D)
    return (Gn[1:] - Gn[:-1]).norm(dim=1).sum()


# ===========================================================================
# Geodesic — energy descent on the interior nodes (Shao Alg. 1).
# ===========================================================================

def geodesic(decode: Callable, z0, z1, T: int = 32,
             method: str = "lbfgs", iters: int = 200, lr: float | None = None,
             tol: float = 1e-6, history: list | None = None):
    """Discrete geodesic between ``z0`` and ``z1`` under the decoder pullback metric.

    Minimises the discrete curve energy (:func:`curve_energy`) over the ``T-1``
    interior latent nodes by gradient descent, holding the endpoints fixed and
    initialising at the straight latent segment. The gradient is obtained by
    reverse-mode autodiff of the scalar energy — one ``backward()`` per step,
    first decoder derivatives only.

    Args:
        decode: adapter mapping ``(M, d_z) -> (M, D)`` (flattened, differentiable,
            deterministic, mean head).
        z0, z1: ``(d_z,)`` endpoint latent codes (fixed).
        T: number of segments (``T+1`` nodes). Default 32; range 16-64.
        method: ``"lbfgs"`` (default; strong-Wolfe line search) or ``"adam"``.
        iters: iteration cap (L-BFGS ``max_iter``, or Adam step count).
        lr: step size. ``None`` -> 1.0 for L-BFGS, 1e-2 for Adam.
        tol: gradient-norm / energy-change tolerance.
        history: optional list; if given, the energy at each objective evaluation
            is appended (for a convergence plot / the monotone-decrease test).
    Returns:
        ``(T+1, d_z)`` latent nodes of the geodesic, endpoints included.
    """
    torch = _require_torch()
    z0 = z0 if torch.is_tensor(z0) else torch.as_tensor(z0, dtype=torch.float32)
    z1 = z1 if torch.is_tensor(z1) else torch.as_tensor(z1, dtype=torch.float32)
    device, dtype = z0.device, z0.dtype
    z0d, z1d = z0.detach().view(1, -1), z1.detach().view(1, -1)

    if T < 2:  # no interior nodes to optimise
        return torch.cat([z0d, z1d], dim=0)

    # Straight latent segment as initialisation (also the baseline, Section 7).
    ts = torch.linspace(0, 1, T + 1, device=device, dtype=dtype)[1:-1].unsqueeze(1)
    Z_int = ((1 - ts) * z0d + ts * z1d).clone().requires_grad_(True)   # (T-1, d_z)

    def closure():
        opt.zero_grad()
        Z = torch.cat([z0d, Z_int, z1d], dim=0)            # (T+1, d_z)
        E = curve_energy(decode(Z))
        E.backward()
        if history is not None:
            history.append(float(E.detach()))
        return E

    if method == "lbfgs":
        opt = torch.optim.LBFGS([Z_int], lr=(lr or 1.0), max_iter=iters,
                                tolerance_grad=tol, tolerance_change=tol * 1e-2,
                                line_search_fn="strong_wolfe")
        opt.step(closure)
    elif method == "adam":
        opt = torch.optim.Adam([Z_int], lr=(lr or 1e-2))
        prev = float("inf")
        for _ in range(iters):
            E = float(opt.step(closure).detach())
            if abs(prev - E) < tol:
                break
            prev = E
    else:
        raise ValueError(f"unknown method {method!r}; use 'lbfgs' or 'adam'.")

    with torch.no_grad():
        return torch.cat([z0d, Z_int.detach(), z1d], dim=0)


def straight_path(z0, z1, T: int = 32):
    """The naive baseline: the straight latent segment ``z_i^lin`` (Section 7).

    ``z_i = (1 - i/T) z0 + (i/T) z1`` for ``i = 0, ..., T``. This is exactly the
    initialisation :func:`geodesic` starts from.
    Returns ``(T+1, d_z)`` latent nodes.
    """
    torch = _require_torch()
    z0 = z0 if torch.is_tensor(z0) else torch.as_tensor(z0, dtype=torch.float32)
    z1 = z1 if torch.is_tensor(z1) else torch.as_tensor(z1, dtype=torch.float32)
    ts = torch.linspace(0, 1, T + 1, device=z0.device, dtype=z0.dtype).unsqueeze(1)
    return (1 - ts) * z0.view(1, -1) + ts * z1.view(1, -1)


def interpolate_clips(decode: Callable, encode: Callable, x0, x1, T: int = 32, **kw):
    """End-to-end: two clips in, ``(T+1)`` decoded intermediate clips out, plus the path.

    Args:
        decode: adapter ``(M, d_z) -> (M, D)``.
        encode: adapter mapping one clip to its posterior mean ``(d_z,)``.
        x0, x1: the two endpoint clips.
        T: number of segments.
        **kw: forwarded to :func:`geodesic`.
    Returns:
        ``(clips, Z)`` with ``clips = decode(Z)`` shape ``(T+1, D)`` (flattened)
        and ``Z`` the ``(T+1, d_z)`` geodesic nodes.
    """
    z0, z1 = encode(x0), encode(x1)                        # posterior means
    Z = geodesic(decode, z0, z1, T=T, **kw)               # (T+1, d_z)
    clips = decode(Z)                                      # (T+1, D)
    return clips, Z


# ===========================================================================
# Diagnostics (Section 6). The closed form is for the gradient-check test only.
# ===========================================================================

def build_jac_single(decode: Callable):
    """Return ``jac_single(z) -> (D, d_z)`` via forward-mode autodiff.

    Forward mode is the right tool here because ``D >> d_z`` (a whole clip vs a
    small latent). Used to materialise ``J`` for the closed-form gradient check
    and for metric-spectrum diagnostics — **not** needed for the energy gradient.
    """
    torch = _require_torch()

    def g_single(z):
        return decode(z.unsqueeze(0)).reshape(-1)         # (d_z,) -> (D,)

    return torch.func.jacfwd(g_single)


def energy_grad_closed_form(decode: Callable, jac_single: Callable, Z):
    """Shao Eq. 5, explicit: a decoded acceleration pulled back by one ``Jᵀ``.

    ``grad_{z_i} E = -(1/dt) Jᵢᵀ (g(z_{i+1}) - 2 g(z_i) + g(z_{i-1}))``, with
    ``1/dt = T``. For the gradient-check test only; production uses autodiff.

    Args:
        decode: adapter ``(M, d_z) -> (M, D)``.
        jac_single: ``z -> (D, d_z)`` (e.g. from :func:`build_jac_single`).
        Z: ``(T+1, d_z)`` nodes.
    Returns:
        ``(T-1, d_z)`` gradient w.r.t. the interior nodes ``i = 1, ..., T-1``.
    """
    torch = _require_torch()
    T = Z.shape[0] - 1
    Gn = decode(Z)                                        # (T+1, D)
    accel = Gn[2:] - 2 * Gn[1:-1] + Gn[:-2]               # (T-1, D), interior i
    # J_i^T @ accel: jac_single(z) is (D, d_z), so transpose to pull back the
    # (D,) pose-space acceleration to a (d_z,) latent gradient.
    grads = [-T * (jac_single(Z[i]).T @ accel[i - 1]) for i in range(1, T)]
    return torch.stack(grads)                             # (T-1, d_z)


def curvature_ratio(decode: Callable, z0, z1, T: int = 32, **kw) -> float:
    """``rho = L_geodesic / L_straight`` in pose space, in ``[0, 1]`` up to
    discretisation error. Near 1 => the manifold is nearly flat between the
    endpoints (Shao found little curvature for VAE image manifolds); the geodesic
    then mainly improves the on-manifold plausibility of the decoded transition
    rather than the distance. ``T`` is held fixed across both curves so the chord
    bias cancels.
    """
    Zgeo = geodesic(decode, z0, z1, T=T, **kw)
    Zlin = straight_path(z0, z1, T=T)
    return float(decoded_arclength(decode, Zgeo) / decoded_arclength(decode, Zlin))


def speed_profile(decode: Callable, Z):
    """Per-segment decoded speeds and their coefficient of variation.

    A constant-speed parametrisation (the signature of a converged energy
    minimiser) has ``CV -> 0``; a large CV means under-converged or ``T`` too
    small.
    Returns ``(segments, cv)`` with ``segments`` a NumPy array of length ``T``.
    """
    Gn = decode(Z)
    seg = (Gn[1:] - Gn[:-1]).norm(dim=1)                  # (T,)
    cv = float(seg.std(unbiased=False) / (seg.mean() + 1e-12))
    return seg.detach().cpu().numpy(), cv


# ===========================================================================
# Repo wiring — the decode/encode adapter over an `architectures` VAE.
# ===========================================================================

@dataclass
class VaeInterpolator:
    """Adapter that wires a trained ``architectures`` VAE to the geodesic tools.

    Provides the spec's ``decode`` / ``encode`` contract (mean head, deterministic,
    frozen, ``eval``) plus convenience methods that return the geodesic and the
    straight-line interpolant decoded to clips for visualisation.

    Build it with :func:`make_interpolator` (from a model) or
    :func:`load_interpolator` (from a checkpoint path).

    Attributes:
        model: the wrapped VAE, already in eval mode with parameters frozen.
        device, dtype: torch device / dtype for adapter tensors.
        T_frames: clip length ``T`` in frames (e.g. 64).
        n_joints, n_dims: ``J`` and ``D`` (2 for image-plane keypoints).
        d_latent: flattened latent width ``d_z * n_windows`` for a temporal model.
        weight: optional fixed diagonal ``W`` (torch tensor broadcastable to
            ``(T, J, D)``); applied only to the *energy*, never to displayed clips.
        coord_transform: optional ``Phi`` mapping ``(M, T, J, D) -> (M, T, J, D')``;
            applied to both the energy and the displayed clips.
    """

    model: object
    device: str
    T_frames: int
    n_joints: int
    n_dims: int
    d_latent: int
    weight: object | None = None
    coord_transform: Callable | None = None
    dtype: object = None

    # ---- tensor plumbing -------------------------------------------------
    def _as_tensor(self, a):
        torch = _require_torch()
        if torch.is_tensor(a):
            return a.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(a), device=self.device, dtype=self.dtype)

    def _apply_weight(self, x):
        """Multiply decoded clips ``(M, T, J, D)`` by the diagonal ``W``."""
        w = self.weight
        if w.ndim == 1 and w.shape[0] == self.n_joints:
            w = w.view(1, 1, self.n_joints, 1)             # per-joint weighting
        return x * w

    # ---- the spec's contract --------------------------------------------
    def decode(self, Z):
        """``(M, d_latent) -> (M, D_flat)`` decoded clips, flattened.

        Differentiable w.r.t. ``Z``, deterministic (mean head), decoder frozen in
        eval mode. Applies ``Phi`` and ``W`` per points 3-4 of the assumptions.
        This is the function the energy and its autodiff gradient are built on.
        """
        Z = self._as_tensor(Z)
        x = self.model.decode_full(Z)                      # (M, T, J, D), mean head
        if self.coord_transform is not None:
            x = self.coord_transform(x)
        if self.weight is not None:
            x = self._apply_weight(x)
        return x.reshape(x.shape[0], -1)                   # (M, D_flat)

    def encode(self, x):
        """One clip ``(T, J, D)`` (or ``(1, T, J, D)``) -> posterior mean ``(d_latent,)``.

        Uses an all-visible mask and returns the detached posterior mean ``mu``
        (the endpoints are fixed, so no grad is needed here).
        """
        torch = _require_torch()
        xt = self._as_tensor(x)
        if xt.ndim == 3:
            xt = xt.unsqueeze(0)                            # (1, T, J, D)
        with torch.no_grad():
            mask = torch.ones(xt.shape[0], self.T_frames, self.n_joints,
                              device=self.device, dtype=self.dtype)
            mu, _ = self.model.encode(xt, mask)
        return mu[0].detach()                              # (d_latent,)

    def decode_clips(self, Z):
        """``(M, d_latent) -> (M, T, J, D)`` NumPy clips for visualisation.

        No grad, no weighting ``W`` (that would distort the skeleton); ``Phi`` is
        applied so the displayed coordinates are the keypoints the metric sees.
        """
        torch = _require_torch()
        with torch.no_grad():
            x = self.model.decode_full(self._as_tensor(Z))  # (M, T, J, D)
            if self.coord_transform is not None:
                x = self.coord_transform(x)
        return x.detach().cpu().numpy()

    # ---- convenience -----------------------------------------------------
    def geodesic_path(self, z0, z1, T: int = 32, **kw):
        """The ``(T+1, d_latent)`` geodesic nodes between two latent codes."""
        return geodesic(self.decode, z0, z1, T=T, **kw)

    def straight_path(self, z0, z1, T: int = 32):
        """The ``(T+1, d_latent)`` straight-latent baseline nodes."""
        return straight_path(z0, z1, T=T)

    def interpolate(self, x0, x1, T: int = 32, method: str = "lbfgs",
                    iters: int = 200, report: bool = True, **kw) -> dict:
        """Two clips in; geodesic + straight paths, decoded clips, diagnostics out.

        Returns a dict with:
            ``z0, z1``           endpoint posterior means (torch, ``(d_latent,)``).
            ``Zgeo, Zlin``       geodesic / straight latent nodes ``(T+1, d_latent)``.
            ``geodesic_clips``   ``(T+1, T_frames, J, D)`` NumPy, decoded geodesic.
            ``straight_clips``   ``(T+1, T_frames, J, D)`` NumPy, decoded straight.
            ``x0_recon, x1_recon`` the decoded endpoints (== first/last node).
            ``curvature_ratio``  ``rho = L_geo / L_straight`` (report only).
            ``speed_cv``         constant-speed check (CV of geodesic segments).
            ``energy_geodesic``, ``energy_straight`` the discrete energies.
        """
        z0, z1 = self.encode(x0), self.encode(x1)
        Zgeo = self.geodesic_path(z0, z1, T=T, method=method, iters=iters, **kw)
        Zlin = self.straight_path(z0, z1, T=T)
        geo_clips = self.decode_clips(Zgeo)
        lin_clips = self.decode_clips(Zlin)
        out = {
            "z0": z0, "z1": z1, "Zgeo": Zgeo, "Zlin": Zlin,
            "geodesic_clips": geo_clips, "straight_clips": lin_clips,
            "x0_recon": geo_clips[0], "x1_recon": geo_clips[-1],
        }
        if report:
            seg, cv = speed_profile(self.decode, Zgeo)
            out["speed_cv"] = cv
            out["energy_geodesic"] = float(curve_energy(self.decode(Zgeo)))
            out["energy_straight"] = float(curve_energy(self.decode(Zlin)))
            Lg = float(decoded_arclength(self.decode, Zgeo))
            Ll = float(decoded_arclength(self.decode, Zlin))
            out["curvature_ratio"] = Lg / Ll if Ll > 0 else float("nan")
            out["arclength_geodesic"] = Lg
            out["arclength_straight"] = Ll
        return out


def make_interpolator(model, config=None, device: str = "cpu",
                      weight=None, coord_transform: Callable | None = None,
                      dtype=None) -> VaeInterpolator:
    """Freeze a trained VAE and wrap it as a :class:`VaeInterpolator`.

    Sets ``model.eval()`` and ``requires_grad_(False)`` on every parameter, so
    only the interior latent nodes are ever optimised (autograd still flows to
    them through the frozen decoder graph). Clip dimensions come from ``config``
    when given, else from the model's own attributes; the flattened latent width
    is probed from a dummy encode.

    Args:
        model: an ``architectures`` VAE exposing ``encode`` / ``decode_full``.
        config: optional ``TrainingConfig`` (for ``clip_length, n_joints, n_dims``).
        device: ``"cpu"`` or ``"cuda"``.
        weight: optional ``W`` (array/tensor broadcastable to ``(T, J, D)``).
        coord_transform: optional ``Phi`` on decoded clips ``(M, T, J, D)``.
        dtype: torch dtype for adapter tensors (default ``float32``).
    """
    torch = _require_torch()
    dtype = dtype or torch.float32
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if config is not None:
        T_frames = int(config.clip_length)
        n_joints = int(config.n_joints)
        n_dims = int(config.n_dims)
    else:
        T_frames = int(getattr(model, "T"))
        n_joints = int(getattr(model, "J"))
        n_dims = int(getattr(model, "n_dims"))

    with torch.no_grad():
        probe = torch.zeros(1, T_frames, n_joints, n_dims,
                            device=device, dtype=dtype)
        pmask = torch.ones(1, T_frames, n_joints, device=device, dtype=dtype)
        mu, _ = model.encode(probe, pmask)
    d_latent = int(mu.shape[1])

    w = None if weight is None else torch.as_tensor(
        np.asarray(weight), device=device, dtype=dtype)

    return VaeInterpolator(model=model, device=device, T_frames=T_frames,
                           n_joints=n_joints, n_dims=n_dims, d_latent=d_latent,
                           weight=w, coord_transform=coord_transform, dtype=dtype)


def load_interpolator(checkpoint_path, device: str = "cpu", **kw) -> VaeInterpolator:
    """Load an ``architectures`` checkpoint and wrap it (convenience for Colab).

    Imports :func:`architectures.analyze.load_checkpoint` lazily so this module
    keeps loading without the training package on the path.
    """
    from architectures.analyze import load_checkpoint
    model, config = load_checkpoint(checkpoint_path, device=device)
    return make_interpolator(model, config=config, device=device, **kw)


# ===========================================================================
# Visualisation — synchronised skeleton videos.
# ===========================================================================

def build_morph(node_clips: np.ndarray, n_show: int | None = 9):
    """Concatenate a subset of path nodes' clips into one morph animation.

    Each geodesic node decodes to a whole ``(T_frames, J, D)`` motion; playing a
    handful of evenly-spaced nodes back to back gives a video that morphs from the
    source motion (node 0) to the target motion (node T).

    Args:
        node_clips: ``(P, T_frames, J, D)`` decoded path (P = T+1 nodes).
        n_show: how many evenly-spaced nodes to show. ``None`` shows all P.
    Returns:
        ``(morph, alphas, idx)`` with ``morph`` shape ``(n_show*T_frames, J, D)``,
        ``alphas`` the per-display-frame path parameter in ``[0, 1]``, and ``idx``
        the selected node indices.
    """
    node_clips = np.asarray(node_clips)
    P, F = node_clips.shape[0], node_clips.shape[1]
    if n_show is None or n_show >= P:
        idx = np.arange(P)
    else:
        idx = np.unique(np.linspace(0, P - 1, n_show).round().astype(int))
    sel = node_clips[idx]                                   # (k, F, J, D)
    morph = sel.reshape(-1, *node_clips.shape[2:])          # (k*F, J, D)
    denom = max(P - 1, 1)
    alphas = np.repeat(idx / denom, F)
    return morph, alphas, idx


def animate_skeleton_row(panels, edges=None, fps: int = 25,
                         figsize=None, point_size: float = 8.0,
                         linewidth: float = 1.4, colors=None,
                         invert_y: bool = False, suptitle: str = ""):
    """Animate several skeleton clips side by side, sharing one bounding box.

    Args:
        panels: list of ``(title, clip)`` with ``clip`` shape ``(F_i, J, D)``.
            Panels may differ in length ``F_i``; each loops over its own frames.
        edges: skeleton connectivity as ``(a, b)`` index pairs (e.g.
            ``skeleton_for(J)["bones"]``). Scatter-only when ``None``.
        fps: playback rate (sets the frame interval).
        figsize: overridden default of ``(4.2*n_panels, 4.4)``.
        point_size, linewidth: marker / bone styling.
        colors: per-panel colour list; defaults cycle a fixed palette.
        invert_y: flip the y-axis. Default ``False`` (this repo's YouTube export
            is already y-up, like the existing ``animate_pose_comparison``); set
            ``True`` only for a y-down image-plane export that renders upside down.
        suptitle: optional figure title.
    Returns:
        ``(fig, anim)``. In Colab display with
        ``IPython.display.HTML(anim.to_jshtml())``; ``plt.close(fig)`` first to
        suppress the static preview. Save with
        ``anim.save('out.mp4', writer='ffmpeg', fps=fps)``.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as e:  # pragma: no cover
        raise ImportError("Skeleton animation needs matplotlib.") from e

    titles = [t for t, _ in panels]
    clips = [np.asarray(c) for _, c in panels]
    n = len(clips)
    D = clips[0].shape[-1]
    threeD = D >= 3
    lengths = [c.shape[0] for c in clips]
    n_frames = max(lengths)
    if colors is None:
        base = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        colors = [base[i % len(base)] for i in range(n)]

    # Shared bounding box, equal aspect, over every point of every panel.
    all_pts = np.concatenate([c.reshape(-1, D) for c in clips], axis=0)
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.55 * float((maxs - mins).max())

    if figsize is None:
        figsize = (4.2 * n, 4.4)
    fig = plt.figure(figsize=figsize)
    axes, scatters, lines_per_ax = [], [], []
    for i, (title, clip, color) in enumerate(zip(titles, clips, colors)):
        ax = (fig.add_subplot(1, n, i + 1, projection="3d") if threeD
              else fig.add_subplot(1, n, i + 1))
        f0 = clip[0]
        if threeD:
            scat = ax.scatter(f0[:, 0], f0[:, 1], f0[:, 2], s=point_size, color=color)
        else:
            scat = ax.scatter(f0[:, 0], f0[:, 1], s=point_size, color=color)
        scatters.append(scat)
        lines = []
        if edges is not None:
            for a, b in edges:
                if threeD:
                    (ln,) = ax.plot([f0[a, 0], f0[b, 0]], [f0[a, 1], f0[b, 1]],
                                    [f0[a, 2], f0[b, 2]], color=color, linewidth=linewidth)
                else:
                    (ln,) = ax.plot([f0[a, 0], f0[b, 0]], [f0[a, 1], f0[b, 1]],
                                    color=color, linewidth=linewidth)
                lines.append(ln)
        lines_per_ax.append(lines)

        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if threeD:
            ax.set_zlim(center[2] - half, center[2] + half)
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:  # pragma: no cover
                pass
        else:
            ax.set_aspect("equal")
            if invert_y:
                ax.invert_yaxis()
        axes.append(ax)

    caption = fig.suptitle(suptitle, fontsize=11)

    def update(t):
        for i in range(n):
            clip = clips[i]
            f = clip[t % lengths[i]]
            if threeD:
                scatters[i]._offsets3d = (f[:, 0], f[:, 1], f[:, 2])
            else:
                scatters[i].set_offsets(np.c_[f[:, 0], f[:, 1]])
            if edges is not None:
                for (a, b), ln in zip(edges, lines_per_ax[i]):
                    if threeD:
                        ln.set_data_3d([f[a, 0], f[b, 0]], [f[a, 1], f[b, 1]],
                                       [f[a, 2], f[b, 2]])
                    else:
                        ln.set_data([f[a, 0], f[b, 0]], [f[a, 1], f[b, 1]])
        frac = t / max(n_frames - 1, 1)
        caption.set_text((suptitle + "  " if suptitle else "")
                         + f"frame {t + 1}/{n_frames}   (path α ≈ {frac:.2f})")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000.0 / max(fps, 1), blit=False)
    return fig, anim
