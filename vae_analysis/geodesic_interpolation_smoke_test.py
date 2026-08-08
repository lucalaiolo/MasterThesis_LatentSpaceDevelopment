"""Testing checklist for the geodesic interpolation (spec Section 10).

Runs the six checks against a **real** ``TemporalConvVAE`` (random weights,
64-frame clips, temporal downsample 4, COCO-18 2D) so the adapter wiring is
exercised end to end, not just the maths:

    1. Gradient check   autodiff == closed form (Shao Eq. 5) == central finite diff
    2. Endpoints fixed  Z[0] == z0 and Z[-1] == z1 exactly
    3. Energy           decreases monotonically; E(geo) <= E(straight)
    4. Length           arclength(geo) <= arclength(straight); curvature_ratio <= 1
    5. Constant speed   segment-speed CV small at convergence, and below straight
    6. Determinism      two identical runs give identical Z

Run:  python -m vae_analysis.geodesic_interpolation_smoke_test
"""

from __future__ import annotations

import numpy as np
import torch

from architectures.config import TrainingConfig
from architectures.models import build_model
from vae_analysis.geodesic_interpolation import (
    build_jac_single, curve_energy, decoded_arclength, energy_grad_closed_form,
    geodesic, make_interpolator, speed_profile, straight_path,
)


def _build_interp(dtype=torch.float64, seed=0):
    """A small temporal_conv VAE (T=64, l=4, J=18, D=2), frozen, wrapped."""
    torch.manual_seed(seed)
    cfg = TrainingConfig(
        architecture="temporal_conv", clip_length=64, n_joints=18, n_dims=2,
        latent_dim=4, conv_base_channels=16, conv_strides=(1, 2, 2),
        recipe=1, mask_policy="none", device="cpu",
    )
    cfg.validate()
    model = build_model(cfg).to(dtype)
    interp = make_interpolator(model, config=cfg, device="cpu", dtype=dtype)
    return interp, cfg


def _two_latents(interp, seed=1):
    """Encode two random clips -> two endpoint posterior means."""
    rng = np.random.default_rng(seed)
    shape = (interp.T_frames, interp.n_joints, interp.n_dims)
    x0 = rng.standard_normal(shape).astype(np.float64)
    x1 = rng.standard_normal(shape).astype(np.float64)
    return interp.encode(x0), interp.encode(x1)


def test_gradient_check():
    """1. autodiff grad == closed form (Eq. 5) == central finite differences."""
    interp, _ = _build_interp(dtype=torch.float64)
    z0, z1 = _two_latents(interp)
    Tsg = 6

    # Random interior nodes (perturb the straight segment) so the grad is non-trivial.
    ts = torch.linspace(0, 1, Tsg + 1, dtype=torch.float64)[1:-1].unsqueeze(1)
    Z_int0 = (1 - ts) * z0.view(1, -1) + ts * z1.view(1, -1)
    Z_int0 = Z_int0 + 0.1 * torch.randn_like(Z_int0)
    Z_int = Z_int0.clone().requires_grad_(True)
    z0d, z1d = z0.view(1, -1), z1.view(1, -1)

    Z = torch.cat([z0d, Z_int, z1d], dim=0)
    E = curve_energy(interp.decode(Z))
    (g_auto,) = torch.autograd.grad(E, Z_int)

    jac_single = build_jac_single(interp.decode)
    g_closed = energy_grad_closed_form(interp.decode, jac_single, Z.detach())

    max_abs = (g_auto - g_closed).abs().max().item()
    assert max_abs < 1e-4, f"autodiff vs closed form max|Δ| = {max_abs:.2e}"

    # Central finite differences on a handful of coordinates.
    def energy_at(Zint):
        Zc = torch.cat([z0d, Zint, z1d], dim=0)
        return curve_energy(interp.decode(Zc)).item()

    eps = 1e-6
    rng = np.random.default_rng(0)
    coords = [(int(rng.integers(0, Tsg - 1)), int(rng.integers(0, interp.d_latent)))
              for _ in range(6)]
    max_fd = 0.0
    for (i, j) in coords:
        Zp = Z_int0.clone(); Zp[i, j] += eps
        Zm = Z_int0.clone(); Zm[i, j] -= eps
        fd = (energy_at(Zp) - energy_at(Zm)) / (2 * eps)
        max_fd = max(max_fd, abs(fd - g_auto[i, j].item()))
    assert max_fd < 1e-4, f"autodiff vs finite-diff max|Δ| = {max_fd:.2e}"
    print(f"  [1] gradient check OK  (closed-form Δ={max_abs:.2e}, fd Δ={max_fd:.2e})")


def test_endpoints_fixed():
    """2. The geodesic never moves the endpoints."""
    interp, _ = _build_interp()
    z0, z1 = _two_latents(interp)
    Z = geodesic(interp.decode, z0, z1, T=12, method="lbfgs", iters=50)
    assert torch.equal(Z[0], z0.to(Z.dtype)), "z0 moved"
    assert torch.equal(Z[-1], z1.to(Z.dtype)), "zT moved"
    print("  [2] endpoints fixed OK")


def test_energy_decreases():
    """3. Energy decreases monotonically and ends below the straight line.

    Per-iteration monotonicity is a property of the *descent scheme*, not of any
    optimiser: Adam's momentum can overshoot. We verify it the faithful way, with
    Armijo-backtracking gradient descent (monotone by construction), then confirm
    the production L-BFGS geodesic lands at or below the straight-line energy.
    """
    interp, _ = _build_interp()
    z0, z1 = _two_latents(interp)
    T = 12
    z0d, z1d = z0.view(1, -1), z1.view(1, -1)

    ts = torch.linspace(0, 1, T + 1, dtype=z0.dtype)[1:-1].unsqueeze(1)
    Z_int = ((1 - ts) * z0d + ts * z1d).clone().requires_grad_(True)

    def energy(Zint):
        return curve_energy(interp.decode(torch.cat([z0d, Zint, z1d], dim=0)))

    hist = [float(energy(Z_int).detach())]
    step = 1e-2
    for _ in range(120):
        E = energy(Z_int)
        (g,) = torch.autograd.grad(E, Z_int)
        E0, s = float(E.detach()), step
        for _ in range(30):                     # Armijo backtracking
            cand = (Z_int - s * g).detach()
            if float(energy(cand).detach()) <= E0:
                break
            s *= 0.5
        Z_int = cand.requires_grad_(True)
        hist.append(float(energy(Z_int).detach()))
    worst = max((hist[k + 1] - hist[k]) for k in range(len(hist) - 1))
    assert worst <= 1e-9 * max(hist), f"energy rose by {worst:.2e}"
    assert hist[-1] < hist[0], "energy did not decrease"

    Zgeo = geodesic(interp.decode, z0, z1, T=T, method="lbfgs", iters=200)
    Zlin = straight_path(z0, z1, T=T)
    E_geo = float(curve_energy(interp.decode(Zgeo)))
    E_lin = float(curve_energy(interp.decode(Zlin)))
    assert E_geo <= E_lin + 1e-9, f"E(geo)={E_geo:.4f} > E(straight)={E_lin:.4f}"
    print(f"  [3] energy monotone + decreases OK  "
          f"(E straight={E_lin:.4f} -> geodesic={E_geo:.4f})")


def test_length_non_increase():
    """4. Decoded arclength does not increase; curvature ratio <= 1 (+ tol)."""
    interp, _ = _build_interp()
    z0, z1 = _two_latents(interp)
    T = 16
    Zgeo = geodesic(interp.decode, z0, z1, T=T, method="lbfgs", iters=200)
    Zlin = straight_path(z0, z1, T=T)
    Lg = float(decoded_arclength(interp.decode, Zgeo))
    Ll = float(decoded_arclength(interp.decode, Zlin))
    assert Lg <= Ll * (1 + 1e-3), f"L(geo)={Lg:.4f} > L(straight)={Ll:.4f}"
    print(f"  [4] length non-increase OK  (ρ = L_geo/L_straight = {Lg / Ll:.4f})")


def test_constant_speed():
    """5. Segment-speed CV is small at convergence and below the straight line."""
    interp, _ = _build_interp()
    z0, z1 = _two_latents(interp)
    T = 16
    Zgeo = geodesic(interp.decode, z0, z1, T=T, method="lbfgs", iters=300)
    Zlin = straight_path(z0, z1, T=T)
    _, cv_geo = speed_profile(interp.decode, Zgeo)
    _, cv_lin = speed_profile(interp.decode, Zlin)
    assert cv_geo <= cv_lin + 1e-9, f"geodesic CV {cv_geo:.3f} > straight CV {cv_lin:.3f}"
    assert cv_geo < 0.05, f"geodesic speed CV {cv_geo:.3f} not < 0.05 (under-converged?)"
    print(f"  [5] constant speed OK  (CV straight={cv_lin:.3f} -> geodesic={cv_geo:.3f})")


def test_determinism():
    """6. Two runs with identical inputs give identical nodes."""
    interp, _ = _build_interp()
    z0, z1 = _two_latents(interp)
    Za = geodesic(interp.decode, z0, z1, T=12, method="lbfgs", iters=100)
    Zb = geodesic(interp.decode, z0, z1, T=12, method="lbfgs", iters=100)
    assert torch.equal(Za, Zb), "non-deterministic geodesic"
    print("  [6] determinism OK")


def main():
    print("Geodesic interpolation — Section 10 checklist "
          "(real TemporalConvVAE, T=64, l=4, J=18, D=2)")
    test_gradient_check()
    test_endpoints_fixed()
    test_energy_decreases()
    test_length_non_increase()
    test_constant_speed()
    test_determinism()
    print("All geodesic-interpolation checks passed.")


if __name__ == "__main__":
    main()
