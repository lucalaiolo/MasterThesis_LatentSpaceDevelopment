"""RVQ codebook analysis ([plan §6.8]).

Per-layer code-usage frequencies and, for the pathology codebook's base
layer, how class-specific each code entry is (co-occurrence of ``c_p`` among
the clips that use it — a near-one-hot distribution means the entry is
class-specific). The per-code decode-and-animate (stick figures) is left to
``decode_single_code`` for the caller to render; the analytical summaries
below are what the report needs.
"""

from __future__ import annotations

import numpy as np

from . import palette as pal


def usage_per_layer(idx: np.ndarray, n_codes: int) -> np.ndarray:
    """Fraction of distinct codes used per RVQ layer. idx: (N, T', L)."""
    L = idx.shape[-1]
    return np.array([len(np.unique(idx[:, :, l])) / n_codes for l in range(L)])


def code_class_cooccurrence(idx: np.ndarray, c_p: np.ndarray, layer: int = 0
                            ) -> dict:
    """Per base-layer code, the distribution of ``c_p`` among its uses ([§6.8]).

    Returns ``{"matrix": (n_used, n_classes), "codes", "purity"}`` where
    purity is the max class fraction per code (1 = perfectly class-specific).
    """
    tokens = idx[:, :, layer]                     # (N, T')
    classes = np.repeat(np.asarray(c_p)[:, None], tokens.shape[1], axis=1)
    tok = tokens.reshape(-1)
    cls = classes.reshape(-1)
    n_classes = int(cls.max()) + 1
    used = np.array(sorted(np.unique(tok)))
    M = np.zeros((len(used), n_classes))
    for i, code in enumerate(used):
        vals, cnts = np.unique(cls[tok == code], return_counts=True)
        M[i, vals] = cnts
    row = M.sum(1, keepdims=True)
    dist = M / np.clip(row, 1, None)
    return {"matrix": dist, "codes": used, "purity": dist.max(1),
            "weight": row.ravel() / row.sum()}


def decode_single_code(model, codebook: str, layer: int, code_id: int,
                       device: str = "cpu"):
    """Decode a clip using only one code entry at one layer (others zero).

    The discrete analogue of a latent traversal ([plan §6.8]); returns a
    ``(T, J, 2)`` pose the caller can render as a stick-figure animation.
    """
    import torch
    cfg = model.config
    rvq = model.rvq_pathology if codebook == "p" else model.rvq_motion
    d = cfg.d_pathology if codebook == "p" else cfg.d_motion
    e = rvq.layers[layer].codebook[code_id]                 # (D,)
    q = e.view(1, 1, d).expand(1, cfg.t_latent, d).to(device)
    with torch.no_grad():
        if codebook == "p":
            x_hat = model.decode(q_m=None, q_p=q)
        else:
            x_hat = model.decode(q_m=q, q_p=None)
    return x_hat[0].cpu().numpy().reshape(cfg.clip_length, cfg.n_joints, 2)


def plot_usage(idx_m, idx_p, n_codes_m, n_codes_p, out_dir,
               name: str = "GAITGen"):
    """Bar plot of per-layer code usage for both codebooks ([plan §6.8])."""
    plt = pal.import_matplotlib()
    um = usage_per_layer(idx_m, n_codes_m)
    up = usage_per_layer(idx_p, n_codes_p)
    L = len(um)
    fig, ax = plt.subplots(figsize=(1.1 * L + 2, 4))
    w = 0.4
    ax.bar(np.arange(L) - w / 2, um, w, color=pal.LATENT_COLORS["q_m"],
           label="motion codebook")
    ax.bar(np.arange(L) + w / 2, up, w, color=pal.LATENT_COLORS["q_p"],
           label="pathology codebook")
    ax.axhline(0.2, ls=":", color="black", label="20% floor (reset if below)")
    ax.set_xlabel("RVQ layer")
    ax.set_ylabel("fraction of codes used")
    ax.set_ylim(0, 1)
    ax.set_title(f"{name} — codebook usage per layer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, f"codebook_usage_{name}.png")
