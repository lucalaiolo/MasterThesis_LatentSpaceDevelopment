"""Plotting helpers for the masked-VAE training run.

Three groups of plots:

    Training curves    losses, KL, and reconstruction errors per epoch.
                       Read from the history dict the training loop
                       returns (or from `history.json` on disk).

    Latent diagnostics per-dimension KL, active-unit count ([MVAE §6.5]),
                       latent-traversal grids, and a 2D PCA of the
                       posterior means ([MVAE §7.3]). Take a trained
                       model plus a batch of clips.

    Reconstructions    pose overlays, per-frame and per-joint MPJPE,
                       joint-coordinate trajectories over time, and a
                       heatmap of a batch of masks.

Every function returns a matplotlib Figure so callers can display it
inline or `savefig` it. `plot_training_summary(history, out_dir)` is
the entry point the training loop calls automatically at the end of a
run — it writes a directory of PNGs covering the curves and, when a
model+loader are passed, the latent diagnostics too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 3D projection
        return plt
    except ImportError as e:
        raise ImportError(
            "Plotting needs matplotlib. `pip install matplotlib`."
        ) from e


def _import_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("Latent and reconstruction plots need PyTorch.") from e


# ============================================================================
# 1. Training curves
# ============================================================================


def _stack_history(history: dict) -> dict[str, dict[str, np.ndarray]]:
    """Turn `history["train"]` (list of dicts) into a dict of arrays."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val"):
        rows = history.get(split, [])
        if not rows:
            out[split] = {}
            continue
        keys = rows[0].keys()
        out[split] = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}
    return out


def plot_loss_curves(history: dict, log_y: bool = True):
    """Total loss, KL, and the reconstruction terms over epochs.

    Four subplots: total loss, KL, `rec_full` (full-clip MSE), `rec_aux`
    (Recipe 2's masked-pass MSE or Recipe 3's hidden-only MSE). Recipe 1
    leaves `rec_aux` at zero, so that panel stays flat.
    """
    plt = _import_matplotlib()
    stacked = _stack_history(history)
    epochs = np.arange(len(stacked["train"].get("loss", [])))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("loss", "Total loss", axes[0, 0]),
        ("kl", "KL divergence", axes[0, 1]),
        ("rec_full", "Full-clip MSE", axes[1, 0]),
        ("rec_aux", "Auxiliary MSE (Recipes 2, 3)", axes[1, 1]),
    ]
    for key, title, ax in panels:
        for split, style in (("train", "-"), ("val", "--")):
            ys = stacked[split].get(key)
            if ys is None or len(ys) == 0:
                continue
            ax.plot(epochs, ys, style, label=split, linewidth=1.6)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        if log_y and key in ("loss", "rec_full", "rec_aux"):
            ax.set_yscale("log")
    for ax in axes[1, :]:
        ax.set_xlabel("epoch")
    fig.suptitle("Training curves")
    fig.tight_layout()
    return fig


def plot_beta_schedule(warmup_epochs: int, beta_max: float, n_epochs: int):
    """The linear KL-weight warmup used in [MVAE §6.2]."""
    plt = _import_matplotlib()
    epochs = np.arange(n_epochs)
    betas = np.minimum(1.0, epochs / max(warmup_epochs, 1)) * beta_max
    if warmup_epochs <= 0:
        betas[:] = beta_max
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(epochs, betas, "-", linewidth=2)
    ax.axhline(beta_max, linestyle=":", color="0.5",
               label=f"beta_max = {beta_max}")
    ax.axvline(warmup_epochs, linestyle=":", color="0.5")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\beta$")
    ax.set_title("KL weight schedule")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================================
# 2. Latent diagnostics
# ============================================================================


def collect_latent_stats(model, loader, device: str = "cpu",
                         max_batches: int | None = None) -> dict:
    """Gather posterior means, log-variances, and per-dim KL over a loader.

    Iterates the loader, encodes each batch, and stacks the posterior
    parameters. Returns:

        mus, logvars   (N, d_z) numpy arrays.
        kl_per_dim     (d_z,) mean KL contribution of each latent dim.
        var_of_mean    (d_z,) variance of E[z_d | X] across clips —
                       drives the active-unit count ([MVAE §6.5]).
        active_units   (d_z,) bool, True where var_of_mean > 1e-2.
    """
    torch = _import_torch()
    model.eval()
    mus, logvars = [], []
    with torch.no_grad():
        for i, (X, M) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            X = X.to(device)
            M = M.to(device)
            mu, logvar = model.encode(X, M)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    if not mus:
        return {"mus": np.zeros((0, 0)), "logvars": np.zeros((0, 0)),
                "kl_per_dim": np.zeros(0), "var_of_mean": np.zeros(0),
                "active_units": np.zeros(0, dtype=bool)}
    mus = np.concatenate(mus, axis=0)
    logvars = np.concatenate(logvars, axis=0)

    # Per-dim KL to N(0, 1), then averaged over clips.
    kl_per_sample_dim = 0.5 * (mus ** 2 + np.exp(logvars) - logvars - 1)
    kl_per_dim = kl_per_sample_dim.mean(axis=0)

    var_of_mean = mus.var(axis=0)
    active_units = var_of_mean > 1e-2

    return {"mus": mus, "logvars": logvars,
            "kl_per_dim": kl_per_dim, "var_of_mean": var_of_mean,
            "active_units": active_units}


def plot_latent_kl_per_dim(stats: dict):
    """Bar chart of per-dimension KL — spots posterior collapse.

    A bimodal shape (a few tall bars beside many near-zero) is
    partial collapse. All-zero KL says the encoder has been squashed
    to the prior ([MVAE §6.5]).
    """
    plt = _import_matplotlib()
    kl = stats["kl_per_dim"]
    order = np.argsort(kl)[::-1]
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(kl)), 3.6))
    ax.bar(np.arange(len(kl)), kl[order], color="steelblue")
    ax.set_xlabel("latent dimension (sorted by KL)")
    ax.set_ylabel(r"mean $\mathrm{KL}(q_\phi\,\Vert\,p)$ per dim")
    ax.set_title(f"Per-dimension KL — active units {int(stats['active_units'].sum())} "
                 f"/ {len(kl)}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_active_units(stats: dict, threshold: float = 1e-2):
    """Bar chart of Var_X(E[z_d | X]).

    Units above the threshold count as active (Burda et al., 2016).
    A single-digit active count on a 32-dim latent is a red flag.
    """
    plt = _import_matplotlib()
    v = stats["var_of_mean"]
    order = np.argsort(v)[::-1]
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(v)), 3.6))
    colors = ["seagreen" if v[k] > threshold else "lightgray" for k in order]
    ax.bar(np.arange(len(v)), v[order], color=colors)
    ax.axhline(threshold, linestyle=":", color="firebrick",
               label=f"threshold = {threshold}")
    ax.set_xlabel("latent dimension (sorted by variance)")
    ax.set_ylabel(r"$\mathrm{Var}_X(\mathbb{E}[z_d\mid X])$")
    ax.set_yscale("log")
    ax.set_title(f"Active units — {int((v > threshold).sum())} / {len(v)} above threshold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    return fig


def plot_latent_pca(stats: dict, colors: np.ndarray | None = None,
                    label: str = ""):
    """2-D PCA scatter of posterior means.

    Useful sanity check for latent structure. Optionally colour by any
    per-clip scalar (e.g. video id, clip start time, a lateralisation
    score).
    """
    plt = _import_matplotlib()
    mus = stats["mus"]
    if mus.shape[0] < 2 or mus.shape[1] < 2:
        raise ValueError("Need at least 2 samples and 2 latent dims for PCA.")
    X = mus - mus.mean(axis=0, keepdims=True)
    # SVD PCA — no sklearn dependency.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    scores = X @ Vt[:2].T
    explained = (S[:2] ** 2) / (S ** 2).sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(scores[:, 0], scores[:, 1],
                    c=colors if colors is not None else "steelblue",
                    s=10, alpha=0.7, cmap="viridis")
    if colors is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(label)
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f} %)")
    ax.set_title("Posterior means, first two PCs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_latent_traversal(model, ref_clip: np.ndarray, ref_mask: np.ndarray,
                          dims: Sequence[int], alphas: Sequence[float],
                          device: str = "cpu", joint_idx: int = 0):
    """Sweep one latent coord at a time and plot the decoded trajectory.

    Rows are latent dimensions, columns are alpha offsets. Each cell
    plots x/y/z of joint `joint_idx` over time. Monotone, meaningful
    changes across a row read as an interpretable latent axis
    ([MVAE §7.3]).
    """
    plt = _import_matplotlib()
    torch = _import_torch()
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(ref_clip[None].astype(np.float32)).to(device)
        M = torch.from_numpy(ref_mask[None].astype(np.float32)).to(device)
        mu0, _ = model.encode(X, M)
        T, J, _ = ref_clip.shape

        fig, axes = plt.subplots(len(dims), len(alphas),
                                 figsize=(1.8 * len(alphas), 1.6 * len(dims)),
                                 sharex=True, sharey=True, squeeze=False)
        for r, d in enumerate(dims):
            for c, a in enumerate(alphas):
                z = mu0.clone()
                z[0, d] = z[0, d] + a
                X_hat = _decode_full(model, z, M)
                traj = X_hat[0, :, joint_idx].cpu().numpy()  # (T, 3)
                ax = axes[r, c]
                ax.plot(np.arange(T), traj[:, 0], "-", linewidth=1, label="x")
                ax.plot(np.arange(T), traj[:, 1], "-", linewidth=1, label="y")
                ax.plot(np.arange(T), traj[:, 2], "-", linewidth=1, label="z")
                ax.set_title(rf"$z_{{{d}}}$ += {a:+.1f}", fontsize=8)
                ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Latent traversal on joint {joint_idx}")
        fig.tight_layout()
    return fig


def _decode_full(model, z, M):
    """Call the full-clip decoder head, whatever the recipe."""
    if hasattr(model, "decode_full"):
        return model.decode_full(z)
    # Legacy path.
    return model.decode(z, M)


# ============================================================================
# 3. Reconstruction and MPJPE plots
# ============================================================================


def compute_predictions(model, X: np.ndarray, M: np.ndarray,
                        device: str = "cpu", head: str = "auto") -> np.ndarray:
    """Run the model and return the reconstruction as a numpy array.

    `head`: "full" always uses the full-clip head; "inp" uses the
    inpainting head if the model has one; "auto" picks the inpainting
    head when the model is Recipe-3 and the mask has any hidden entries,
    else the full head.
    """
    torch = _import_torch()
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X.astype(np.float32)).to(device)
        Mt = torch.from_numpy(M.astype(np.float32)).to(device)
        mu, logvar = model.encode(Xt, Mt)
        # Use the posterior mean at eval, not a sample ([MVAE §3.7]).
        want_inp = (head == "inp") or (
            head == "auto" and hasattr(model, "decode_inp")
            and (M < 0.5).any()
        )
        if want_inp and hasattr(model, "decode_inp"):
            X_hat = model.decode_inp(mu, Mt)
        else:
            X_hat = _decode_full(model, mu, Mt)
        return X_hat.cpu().numpy()


def plot_pose_frame(x_frame: np.ndarray, ax=None, edges: Iterable[tuple[int, int]] | None = None,
                    color: str = "steelblue", label: str = ""):
    """Scatter one frame's joints in 3D, optionally with skeleton edges."""
    plt = _import_matplotlib()
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_frame[:, 0], x_frame[:, 1], x_frame[:, 2],
               s=20, color=color, label=label)
    if edges is not None:
        for a, b in edges:
            ax.plot([x_frame[a, 0], x_frame[b, 0]],
                    [x_frame[a, 1], x_frame[b, 1]],
                    [x_frame[a, 2], x_frame[b, 2]], color=color, linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return ax


def plot_pose_comparison(x_true: np.ndarray, x_pred: np.ndarray,
                         frames: Sequence[int] | None = None,
                         edges: Iterable[tuple[int, int]] | None = None):
    """Grid of ground-truth vs reconstructed poses at selected frames.

    `x_true`, `x_pred` are single clips of shape (T, J, 3). Default
    frames are the first, middle, and last.
    """
    plt = _import_matplotlib()
    T = x_true.shape[0]
    if frames is None:
        frames = [0, T // 2, T - 1]

    fig = plt.figure(figsize=(3.6 * len(frames), 3.6))
    for i, t in enumerate(frames):
        ax = fig.add_subplot(1, len(frames), i + 1, projection="3d")
        plot_pose_frame(x_true[t], ax=ax, edges=edges,
                        color="black", label="true")
        plot_pose_frame(x_pred[t], ax=ax, edges=edges,
                        color="tomato", label="pred")
        ax.set_title(f"frame {t}")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Pose reconstruction")
    fig.tight_layout()
    return fig


def plot_joint_trajectory(x_true: np.ndarray, x_pred: np.ndarray,
                          joint_idx: int = 0):
    """Coordinate trajectories for one joint over time — true vs predicted.

    Three lines each for true and predicted (x, y, z). `x_true` and
    `x_pred` are single clips of shape (T, J, 3).
    """
    plt = _import_matplotlib()
    T = x_true.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5), sharex=True)
    labels = ["x", "y", "z"]
    for c in range(3):
        axes[c].plot(np.arange(T), x_true[:, joint_idx, c], "-",
                     color="black", linewidth=1.4, label="true")
        axes[c].plot(np.arange(T), x_pred[:, joint_idx, c], "--",
                     color="tomato", linewidth=1.4, label="pred")
        axes[c].set_ylabel(labels[c])
        axes[c].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"Joint {joint_idx} trajectory")
    fig.tight_layout()
    return fig


def _per_joint_err(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """Euclidean distance per (clip, frame, joint)."""
    return np.linalg.norm(x_pred - x_true, axis=-1)


def plot_mpjpe_per_joint(x_true: np.ndarray, x_pred: np.ndarray,
                         joint_names: Sequence[str] | None = None,
                         M: np.ndarray | None = None):
    """Mean per-joint position error across a batch, one bar per joint.

    Pass `M` to split the bars into visible and hidden components; the
    hidden count is the number of positions the model had to fill in.
    """
    plt = _import_matplotlib()
    err = _per_joint_err(x_true, x_pred)              # (N, T, J)
    if M is None:
        per_joint = err.mean(axis=(0, 1))
        fig, ax = plt.subplots(figsize=(max(6, 0.35 * per_joint.size), 3.6))
        ax.bar(np.arange(per_joint.size), per_joint, color="steelblue")
    else:
        vis_err = np.where(M > 0.5, err, np.nan)
        inp_err = np.where(M < 0.5, err, np.nan)
        per_vis = np.nanmean(vis_err, axis=(0, 1))
        per_inp = np.nanmean(inp_err, axis=(0, 1))
        fig, ax = plt.subplots(figsize=(max(7, 0.4 * per_vis.size), 3.8))
        idx = np.arange(per_vis.size)
        w = 0.4
        ax.bar(idx - w / 2, per_vis, w, label="visible", color="steelblue")
        ax.bar(idx + w / 2, per_inp, w, label="hidden (inpainted)", color="tomato")
        ax.legend()

    ax.set_ylabel("MPJPE")
    ax.set_xlabel("joint")
    if joint_names is not None:
        ax.set_xticks(np.arange(len(joint_names)))
        ax.set_xticklabels(joint_names, rotation=60, ha="right", fontsize=8)
    ax.set_title("Per-joint MPJPE")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_mpjpe_per_frame(x_true: np.ndarray, x_pred: np.ndarray):
    """MPJPE aggregated over joints and clips as a function of frame index."""
    plt = _import_matplotlib()
    err = _per_joint_err(x_true, x_pred)              # (N, T, J)
    per_frame = err.mean(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(np.arange(per_frame.size), per_frame, "-", linewidth=1.6)
    ax.set_xlabel("frame")
    ax.set_ylabel("MPJPE")
    ax.set_title("MPJPE by frame")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mask_heatmap(M: np.ndarray, n_show: int = 8):
    """Heatmaps of the first `n_show` masks in a batch.

    `M` is (B, T, J). Rows are frames, columns are joints, dark cells
    are hidden. Useful for eyeballing whether the masking policy is
    doing what you asked.
    """
    plt = _import_matplotlib()
    B = M.shape[0]
    n = min(n_show, B)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(M[i], aspect="auto", cmap="Greys_r", vmin=0, vmax=1,
                       interpolation="nearest")
        axes[i].set_title(f"clip {i}", fontsize=9)
        axes[i].set_xlabel("joint")
    axes[0].set_ylabel("frame")
    fig.suptitle("Masks (dark = hidden)")
    fig.tight_layout()
    return fig


# ============================================================================
# 4. Composite training summary
# ============================================================================


def plot_training_summary(history: dict, out_dir: str | Path,
                          config=None, model=None, loader=None,
                          device: str = "cpu",
                          skeleton_edges: Iterable[tuple[int, int]] | None = None
                          ) -> list[Path]:
    """Write a directory of PNGs summarising a completed run.

    Always writes: `loss_curves.png`, `beta_schedule.png` (when `config`
    is passed). When `model` and `loader` are passed, also writes latent
    diagnostics and a batch reconstruction. Returns the list of files
    written.
    """
    plt = _import_matplotlib()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(fig, name: str):
        p = out_dir / name
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    fig = plot_loss_curves(history)
    _save(fig, "loss_curves.png")

    if config is not None:
        n_epochs = len(history.get("train", [])) or getattr(config, "n_epochs", 1)
        fig = plot_beta_schedule(config.warmup_epochs, config.beta_max, n_epochs)
        _save(fig, "beta_schedule.png")

    if model is not None and loader is not None:
        stats = collect_latent_stats(model, loader, device=device, max_batches=32)
        _save(plot_latent_kl_per_dim(stats), "latent_kl_per_dim.png")
        _save(plot_active_units(stats), "active_units.png")
        if stats["mus"].shape[0] >= 2 and stats["mus"].shape[1] >= 2:
            _save(plot_latent_pca(stats), "latent_pca.png")

        # One reconstruction / mask preview from the first batch.
        for X, M in loader:
            X_np = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
            M_np = M.numpy() if hasattr(M, "numpy") else np.asarray(M)
            X_hat = compute_predictions(model, X_np, M_np, device=device)
            _save(plot_pose_comparison(X_np[0], X_hat[0], edges=skeleton_edges),
                  "reconstruction_frames.png")
            _save(plot_joint_trajectory(X_np[0], X_hat[0], joint_idx=0),
                  "joint0_trajectory.png")
            _save(plot_mpjpe_per_joint(X_np, X_hat, M=M_np),
                  "mpjpe_per_joint.png")
            _save(plot_mpjpe_per_frame(X_np, X_hat),
                  "mpjpe_per_frame.png")
            _save(plot_mask_heatmap(M_np), "mask_examples.png")
            break

    return written
