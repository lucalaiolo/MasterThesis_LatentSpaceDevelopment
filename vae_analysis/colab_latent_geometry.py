# =====================================================================
# Latent-space geometry — the three diagnostics the results chapter reports
# ---------------------------------------------------------------------
# TwoNN intrinsic dimension, the kernel prior-fit test, and latent interpolation
# with the curvature ratio, on the posterior means of the **held-out** clips
# over the flattened latent (d_z * T/l = 128 for the selected temporal model).
#
# This is the single Colab cell. Edit the CONFIG block, run it, and it prints
# every number the section quotes, displays the two figures, and leaves
# results.json / values.tex / summary.md in OUT_DIR.
# =====================================================================

# ---- 0. Repo + dependencies (uncomment on a fresh Colab runtime) ----
# !git clone -b claude/latent-space-geometry-sofepz \
#     https://github.com/lucalaiolo/MasterThesis_LatentSpaceDevelopment.git
# %cd MasterThesis_LatentSpaceDevelopment
# (Colab already ships torch, numpy, scikit-learn and matplotlib.)
#
# Checkpoint or data on Drive? Mount it first:
# from google.colab import drive; drive.mount('/content/drive')

# ============================ CONFIG =================================
CHECKPOINT_PATH = "/content/runs/temporal_conv/best.pt"  # the selected model (.pt)
DATA_CSV        = "/content/keypoints.csv"   # long-format keypoint CSV
DATA_NPZ        = None                       # or a pose.npz cache instead of the CSV
CSV_COLUMNS     = None                       # e.g. {"part_idx": "bp"} for the RVI-38 table
OUT_DIR         = "/content/outputs/latent_geometry"

SPLIT           = "val"      # "val" = the held-out clips the model never saw
MASK_POLICY     = "none"     # "none" = every joint visible (the clean code)
N_PERM          = 500        # permutations B for the kernel test (the section quotes 500)
MAX_MMD_POINTS  = 2000       # cap per side; the pooled kernel is (2*points)^2
PAIR            = "random"   # endpoint pair: "random", "median", or "farthest"
CLIP_A          = None       # or explicit indices into the selected split
CLIP_B          = None
SEGMENTS        = 32         # nodes of the geodesic-vs-straight comparison
ITERS           = 200        # energy-descent iteration cap
INTERP_STEPS    = 7          # decoded points on the straight segment (figure columns)
INTERP_FRAMES   = 4          # frames drawn per decoded clip (figure rows)
INVERT_Y        = False      # True only if the skeletons render upside down
FILL_TEX        = None       # path to the section .tex to fill, e.g. "/content/geometry.tex"
SEED            = 0
# ====================================================================

import torch

from vae_analysis.latent_geometry import load_videos, run_latent_geometry

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. Load the recordings the model was trained on ----
videos = load_videos(csv=DATA_CSV if DATA_NPZ is None else None, npz=DATA_NPZ,
                     columns=CSV_COLUMNS)

# ---- 2. Run the three diagnostics ----
# The split is rebuilt exactly as training built it (video-wise, seed-deterministic),
# so SPLIT="val" is the same held-out set the checkpoint never saw.
out = run_latent_geometry(
    CHECKPOINT_PATH, videos, out_dir=OUT_DIR, device=device,
    split=SPLIT, mask_policy=MASK_POLICY,
    n_perm=N_PERM, max_mmd_points=MAX_MMD_POINTS,
    clip_a=CLIP_A, clip_b=CLIP_B, pair_selection=PAIR,
    n_interp_steps=INTERP_STEPS, n_interp_frames=INTERP_FRAMES,
    n_segments=SEGMENTS, geodesic_iters=ITERS, invert_y=INVERT_Y,
    fill_tex=FILL_TEX, seed=SEED,
)
res = out["results"]

# ---- 3. The numbers the section quotes ----
tw, mm, kl = res["twonn"], res["prior_fit"], res["kl"]
ip = res.get("interpolation", {})
slots = res["placeholders"]

print(f"\nintrinsic dimension   d_hat = {tw['d_hat']:.3f} +- {tw['standard_error']:.3f}"
      f"  (bootstrap +- {tw['bootstrap_se']:.3f}) over {tw['n_used']} points")
print(f"                      ambient width {res['meta']['d_latent']}, "
      f"MLE cross-check {tw['d_hat_mle']:.3f}")
p_txt = (f"p {slots['[p]']}" if slots["[p]"].lstrip().startswith("<")
         else f"p = {slots['[p]']}")   # no permutation reached it => a bound
print(f"prior fit             MMD^2_u = {mm['mmd2']:.6g}, {p_txt} "
      f"(B = {mm['n_perm']}, bandwidth {mm['bandwidth']:.3g})")
print(f"                      per-dimension KL {kl['kl_total']:.2f} nats, "
      f"{kl['n_active']}/{kl['n_dims']} dimensions active")
if ip:
    print(f"interpolation         rho = {ip['curvature_ratio']:.4f} "
          f"(clips {ip['index_a']} / {ip['index_b']}, videos "
          f"{ip['video_a']} / {ip['video_b']}), speed CV {ip['speed_cv']:.4f}")
print("\nsection placeholders:")
for slot, value in slots.items():
    print(f"  {slot:<12} {value}")

# ---- 4. The figures, inline ----
from pathlib import Path

try:
    from IPython.display import Image, Markdown, display

    for name, caption in (("fig_twonn", "TwoNN fit"),
                          ("fig_interp", "latent interpolation (straight segment)"),
                          ("fig_interp_geodesic", "geodesic interpolation"),
                          ("fig_kl_per_dim", "per-dimension KL")):
        png = Path(OUT_DIR) / f"{name}.png"
        if png.exists():
            display(Markdown(f"**{caption}** — `{name}.pdf` is the one to \\includegraphics"))
            display(Image(filename=str(png)))

    display(Markdown((Path(OUT_DIR) / "summary.md").read_text()))
except ImportError:                                   # not in a notebook
    print(f"\nfigures and summary written to {OUT_DIR}")

# ---- 5. (optional) Pull the outputs down ----
# from google.colab import files
# files.download(f"{OUT_DIR}/fig_twonn.pdf")
# files.download(f"{OUT_DIR}/fig_interp.pdf")
# files.download(f"{OUT_DIR}/values.tex")
