"""Disentanglement metrics ([plan §6.1], paper Eq. 11 + PMPG + DS).

- PORE (Pathology-Only Reconstruction Error): how much motion leaks into the
  pathology latent, from the MPJPE gap between decoding ``q_p`` alone and
  decoding ``q_m + q_p`` (paper Eq. 11). Higher = better.
- PMPG (Pathology-Motion Predictive Gap): probe accuracy for ``c_p`` from
  ``q_p`` minus from ``q_m``. Higher = pathology is in ``q_p``, not ``q_m``.
- DS (Disentanglement Score): geometric mean of PORE and PMPG.

Reconstruction errors are in **normalised units** ([plan §5], not mm).
"""

from __future__ import annotations

import numpy as np

from .probes import probe


def reconstruction_mpjpe(model, data, device: str = "cpu",
                         batch: int = 256) -> float:
    """Overall reconstruction MPJPE over ``data`` (normalised units)."""
    import torch
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, data.n, batch):
            x = torch.from_numpy(data.x[i:i + batch]).float().to(device)
            c = torch.from_numpy(data.c_p[i:i + batch]).long().to(device)
            lat = model.encode_latents(x, c)
            x_hat = model.decode(lat["q_m"], lat["q_p"])
            tot += float(model._mpjpe(x, x_hat)) * len(x)
            n += len(x)
    return tot / max(n, 1)


def pore(model, data, device: str = "cpu", batch: int = 256,
         eps: float = 1e-8) -> dict:
    """Pathology-Only Reconstruction Error ([paper Eq. 11]).

    ``e_p`` = MPJPE of ``D(q_p)``; ``e_pm`` = MPJPE of ``D(q_m + q_p)``.
    ``PORE = (e_p - e_pm) / (e_pm + eps)``.
    """
    import torch
    model.eval()
    ep_tot, epm_tot, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, data.n, batch):
            x = torch.from_numpy(data.x[i:i + batch]).float().to(device)
            c = torch.from_numpy(data.c_p[i:i + batch]).long().to(device)
            lat = model.encode_latents(x, c)
            xp = model.decode(q_m=None, q_p=lat["q_p"])        # pathology only
            xpm = model.decode(lat["q_m"], lat["q_p"])          # both
            ep_tot += float(model._mpjpe(x, xp)) * len(x)
            epm_tot += float(model._mpjpe(x, xpm)) * len(x)
            n += len(x)
    e_p, e_pm = ep_tot / max(n, 1), epm_tot / max(n, 1)
    return {"e_p": e_p, "e_pm": e_pm, "pore": (e_p - e_pm) / (e_pm + eps)}


def pmpg(latents, seed: int = 0) -> dict:
    """Pathology-Motion Predictive Gap: acc(q_p→c_p) − acc(q_m→c_p) ([§6.1])."""
    a_qp = probe(latents.q_p, latents.c_p, latents.subject, seed=seed)
    a_qm = probe(latents.q_m, latents.c_p, latents.subject, seed=seed)
    gap = a_qp["balanced_acc"] - a_qm["balanced_acc"]
    return {"acc_qp": a_qp["balanced_acc"], "acc_qm": a_qm["balanced_acc"],
            "pmpg": gap, "chance": a_qp["chance"]}


def disentanglement_score(pore_val: float, pmpg_val: float) -> float:
    """Geometric mean of PORE and PMPG ([plan §6.1]); nan if either < 0."""
    if pore_val < 0 or pmpg_val < 0:
        return float("nan")
    return float(np.sqrt(pore_val * pmpg_val))


def disentanglement_table(model, latents, data, device: str = "cpu") -> dict:
    """PORE, PMPG, DS + reconstruction MPJPE for one model ([plan §6.1])."""
    por = pore(model, data, device)
    pmp = pmpg(latents)
    return {
        "recon_mpjpe": reconstruction_mpjpe(model, data, device),
        "pore": por["pore"], "e_p": por["e_p"], "e_pm": por["e_pm"],
        "pmpg": pmp["pmpg"], "acc_qp": pmp["acc_qp"], "acc_qm": pmp["acc_qm"],
        "ds": disentanglement_score(por["pore"], pmp["pmpg"]),
    }
