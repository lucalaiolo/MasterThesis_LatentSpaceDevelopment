"""Encode a windowed dataset into disentangled latents for analysis ([plan §6]).

Produces one mean-pooled ``q_m`` and ``q_p`` per clip (the pooling choice the
plan asks to state in captions), plus the raw RVQ token sequences for the
codebook and temporal analyses, aligned to the clip's class label and
subject.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaitGenLatents:
    """Per-clip disentangled latents + labels ([plan §6])."""
    q_m: np.ndarray        # (N, D_m) mean-pooled over time
    q_p: np.ndarray        # (N, D_p) mean-pooled over time
    idx_m: np.ndarray      # (N, T', L) motion token ids
    idx_p: np.ndarray      # (N, T', L) pathology token ids
    c_p: np.ndarray        # (N,) primary label
    c_nuis: np.ndarray     # (N,) nuisance label (-1 if unused)
    subject: np.ndarray    # (N,) subject id

    @property
    def n(self) -> int:
        return len(self.c_p)

    def has_nuisance(self) -> bool:
        return bool((self.c_nuis >= 0).any())


def encode_latents(model, data, device: str = "cpu", batch: int = 256
                   ) -> GaitGenLatents:
    """Run the frozen model over ``data`` and pack :class:`GaitGenLatents`."""
    import torch
    model.eval()
    qm, qp, im, ip = [], [], [], []
    with torch.no_grad():
        for i in range(0, data.n, batch):
            x = torch.from_numpy(data.x[i:i + batch]).float().to(device)
            c = torch.from_numpy(data.c_p[i:i + batch]).long().to(device)
            lat = model.encode_latents(x, c)
            qm.append(lat["q_m"].mean(dim=1).cpu().numpy())    # (b, D_m)
            qp.append(lat["q_p"].mean(dim=1).cpu().numpy())
            im.append(lat["idx_m"].cpu().numpy())
            ip.append(lat["idx_p"].cpu().numpy())
    return GaitGenLatents(
        q_m=np.concatenate(qm), q_p=np.concatenate(qp),
        idx_m=np.concatenate(im), idx_p=np.concatenate(ip),
        c_p=np.asarray(data.c_p), c_nuis=np.asarray(data.c_nuis),
        subject=np.asarray(data.subject))
