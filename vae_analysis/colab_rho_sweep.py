# =====================================================================
# Hunting for curvature: sweep the curvature ratio over many clip pairs
# ---------------------------------------------------------------------
# The section reports rho for one pair of held-out clips. One pair cannot say
# whether rho ~ 1 is a property of the latent or of the pair that happened to
# be picked — and if the latent is curved anywhere, this is how to find where.
#
# TWO CELLS. Paste part 1 in one cell and part 2 in another: part 1 does the
# expensive sweep once, then you change RANK in part 2 and re-run only that
# cell to look at each candidate.
#
# The screen is conservative in the right direction: the energy descent starts
# at the straight segment and only shortens the curve, so an under-converged
# run OVERESTIMATES rho. A pair that screens low really is low — part 2
# re-measures the one you choose at full resolution before you quote it.
# =====================================================================

# =============== PART 1 — sweep (the slow cell) ======================

# ---- Repo + Drive (uncomment on a fresh Colab runtime) ----
# !git clone -b claude/latent-space-geometry-sofepz \
#     https://github.com/lucalaiolo/MasterThesis_LatentSpaceDevelopment.git
# %cd MasterThesis_LatentSpaceDevelopment
# from google.colab import drive; drive.mount('/content/drive')

# ============================ CONFIG =================================
CHECKPOINT_PATH = "/content/runs/temporal_conv/best.pt"
DATA_CSV        = "/content/keypoints.csv"   # long-format keypoint CSV
DATA_NPZ        = None                       # or a pose.npz cache instead
CSV_COLUMNS     = None                       # e.g. {"part_idx": "bp"} for RVI-38
OUT_DIR         = "/content/outputs/latent_geometry"

SPLIT           = "val"       # the held-out clips
N_PAIRS         = 48          # pairs to screen (each costs one energy descent)
STRATEGY        = "spread"    # "spread" covers the latent-distance range (start
                              # here); "far" hunts the most distant pairs, where
                              # curvature has the most room to accumulate;
                              # "random" gives the unbiased distribution of rho
SCREEN_SEGMENTS = 16          # cheap settings for the screen ...
SCREEN_ITERS    = 60
REPORT_SEGMENTS = 32          # ... full settings for the pair you pick
REPORT_ITERS    = 200
SHOW_N          = 12          # rows of the ranked table
SEED            = 0
# ====================================================================

import numpy as np
import torch

from architectures.analyze import load_checkpoint
from architectures.data import build_clips, train_val_split
from architectures.mask_policies import build_policy
from vae_analysis.architectures_adapter import ArchitecturesAdapter
from vae_analysis.geodesic_interpolation import make_interpolator
from vae_analysis.interfaces import encode_dataset
from vae_analysis.latent_geometry import (load_videos, plot_rho_sweep,
                                          sweep_curvature_ratio)

device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)

# ---- 1. Model + the held-out clips (the same split training held out) ----
videos = load_videos(csv=DATA_CSV if DATA_NPZ is None else None, npz=DATA_NPZ,
                     columns=CSV_COLUMNS)
model, config = load_checkpoint(CHECKPOINT_PATH, device=device)
clips, video_id, time_index = build_clips(videos, config.clip_length,
                                          config.clip_length // 2)
train_mask, val_mask = train_val_split(clips, video_id)
sel = {"val": val_mask, "train": train_mask,
       "all": np.ones(len(clips), bool)}[SPLIT]
X, vid, tix = clips[sel].astype(np.float32), video_id[sel], time_index[sel]

# All-visible masks: the clean code, matching what the interpolator encodes.
import dataclasses

policy = build_policy(dataclasses.replace(config, mask_policy="none"))
M = np.stack([policy.sample(config.clip_length, config.n_joints, rng, X=x)
              for x in X]).astype(np.float32)
adapter = ArchitecturesAdapter(model, device=device)
latent = encode_dataset(adapter, X, M, video_id=vid, time_index=tix)
interp = make_interpolator(model, config=config, device=device)
print(f"{latent.n} clips in the {SPLIT!r} split from "
      f"{len(np.unique(vid))} videos, latent d = {latent.d_z}")

# ---- 2. Screen the pairs ----
sweep = sweep_curvature_ratio(
    interp, X, vid, mu=latent.mu, n_pairs=N_PAIRS, strategy=STRATEGY,
    n_segments=SCREEN_SEGMENTS, iters=SCREEN_ITERS, rng=rng)

rho = sweep["rho"]
print(f"\nrho over {sweep['n_pairs']} pairs: min {rho.min():.4f}, "
      f"median {np.median(rho):.4f}, max {rho.max():.4f}")
print(f"{int((rho < 0.99).sum())} pairs below 0.99, "
      f"{int((rho < 0.95).sum())} below 0.95\n")
print(f"{'rank':>4}  {'clip a':>7} {'clip b':>7}  {'vid a':>5} {'vid b':>5}"
      f"  {'rho':>7}  {'|z1-z0|':>8}  {'speed CV':>8}")
for k, r in enumerate(sweep["records"][:SHOW_N]):
    print(f"{k:>4}  {r['index_a']:>7} {r['index_b']:>7}  "
          f"{r['video_a']:>5} {r['video_b']:>5}  "
          f"{r['curvature_ratio']:>7.4f}  {r['latent_distance']:>8.3f}  "
          f"{r['speed_cv']:>8.4f}")

from pathlib import Path

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
fig = plot_rho_sweep(sweep)
fig.savefig(f"{OUT_DIR}/fig_rho_sweep.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig_rho_sweep.pdf", bbox_inches="tight")
try:
    from IPython.display import Image, display
    display(Image(filename=f"{OUT_DIR}/fig_rho_sweep.png"))
except ImportError:
    pass
print(f"\nsweep figure -> {OUT_DIR}/fig_rho_sweep.pdf")
print("Now set RANK in part 2 and re-run that cell to inspect a pair.")


# =============== PART 2 — inspect one pair (the fast cell) ===========
# Change RANK (0 = lowest rho in the table above) and re-run THIS cell only.
# Or set PAIR to explicit clip indices to override the ranking.

RANK = 0
PAIR = None          # e.g. (12, 87) to look at a specific pair
INTERP_STEPS = 7     # figure columns
INTERP_FRAMES = 4    # figure rows
INVERT_Y = False

from youtube_motion.skeleton import skeleton_for
from vae_analysis.latent_geometry import latent_interpolation, plot_interpolation

rec = sweep["records"][RANK]
i, j = (rec["index_a"], rec["index_b"]) if PAIR is None else PAIR
bones = skeleton_for(int(config.n_joints))["bones"]

# Re-measure at full resolution: the screen only ever overestimates rho, so
# the number to quote is this one, not the table's.
ip = latent_interpolation(interp, X[i], X[j], n_steps=INTERP_STEPS,
                          n_segments=REPORT_SEGMENTS, iters=REPORT_ITERS,
                          bones=bones)
print(f"clips {i} / {j}  (videos {vid[i]} / {vid[j]}, "
      f"start frames {tix[i]} / {tix[j]})")
print(f"  rho          {ip['curvature_ratio']:.4f}   "
      f"(screen said {rec['curvature_ratio']:.4f})")
print(f"  arc length   geodesic {ip['arclength_geodesic']:.4f}  "
      f"straight {ip['arclength_straight']:.4f}")
print(f"  energy       {ip['energy_straight']:.4f} -> {ip['energy_geodesic']:.4f}"
      f"   constant-speed CV {ip['speed_cv']:.4f}")
bl = ip["bone_lengths"]
print(f"  bone lengths within-clip CV mean {bl['cv_within_mean']:.4f}, "
      f"drift from the endpoints max {bl['drift_max']:.2%}")
if ip["curvature_ratio"] > 1.0 + 1e-6:
    print("  WARNING: rho > 1 is impossible — raise REPORT_ITERS.")

for clips_to_show, t_vals, name in (
        (ip["straight_clips"], ip["t"], "straight segment"),
        (ip["geodesic_clips"], ip["t_geodesic"], "geodesic")):
    f = plot_interpolation(clips_to_show, t_vals, bones,
                           n_frame_rows=INTERP_FRAMES, invert_y=INVERT_Y,
                           title=f"rank {RANK}: clips {i}/{j} — {name} "
                                 f"($\\rho$ = {ip['curvature_ratio']:.3f})")
    stem = f"{OUT_DIR}/fig_pair_{i}_{j}_{name.split()[0]}"
    f.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    f.savefig(f"{stem}.pdf", bbox_inches="tight")
    try:
        from IPython.display import Image, display
        display(Image(filename=f"{stem}.png"))
    except ImportError:
        print(f"  wrote {stem}.pdf")

# Once you have the pair you want in the thesis, lock it in with
#   --clip-a i --clip-b j   (CLI)  or  clip_a=i, clip_b=j  (run_latent_geometry)
# so the reported figures and rho come from that exact pair.
