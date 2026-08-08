# =====================================================================
# Geodesic interpolation between two clips in the VAE latent space
# ---------------------------------------------------------------------
# Shao, Kumar & Fletcher (CVPR-W 2018) discrete geodesic-energy minimisation:
# the interpolant is the minimum-energy curve joining the two clips' latent
# codes under the metric the decoder pulls back from pose space. Decoding its
# nodes gives intermediate clips that stay on the manifold of decodable motion,
# unlike the decoded straight latent segment (the naive baseline).
#
# This is the single Colab cell. Edit the CONFIG block, run it, and it prints
# the geometry diagnostics and displays the videos.  It works for the temporal
# VAEs (temporal_conv / temporal_transformer): 64-frame clips, temporal
# downsample 4  ->  a (16-window x d_z) latent, flattened.
# =====================================================================

# ---- 0. Repo + dependencies (uncomment on a fresh Colab runtime) ----
# !git clone -b claude/vae-geodesic-interpolation-sxh9ct \
#     https://github.com/lucalaiolo/MasterThesis_LatentSpaceDevelopment.git
# %cd MasterThesis_LatentSpaceDevelopment
# !pip -q install torch numpy matplotlib

# ============================ CONFIG =================================
CHECKPOINT_PATH = "/content/runs/temporal_conv/best.pt"  # your trained checkpoint (.pt)

# How to obtain the two clips x0, x1:
USE_CSV       = True                                     # True: slice them from a CSV
DATA_CSV      = "/content/keypoints.csv"                 # YouTube long-format keypoint CSV
CLIP_A_INDEX  = 0                                        # source clip x0 (index into the sliced clips)
CLIP_B_INDEX  = 25                                       # target clip x1
# If USE_CSV = False, set x0, x1 yourself further down (two (T, J, D) arrays).

T_SEGMENTS    = 32       # geodesic nodes = T_SEGMENTS + 1  (spec default; range 16-64)
METHOD        = "lbfgs"  # "lbfgs" (default) or "adam" if L-BFGS is unstable
ITERS         = 200      # optimiser iteration cap
N_SHOW        = 9        # intermediate motions shown in the morph video (subsampled path nodes)
FPS           = 25       # playback rate
INVERT_Y      = False    # set True only if the skeleton renders upside down
SHOW_BASELINE = True     # also show the straight-line-vs-geodesic comparison (spec Section 7)
# ====================================================================

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from architectures.analyze import load_checkpoint
from youtube_motion.skeleton import skeleton_for
from vae_analysis.geodesic_interpolation import (
    make_interpolator, build_morph, animate_skeleton_row)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. Load the trained VAE and freeze it into the interpolator ----
model, config = load_checkpoint(CHECKPOINT_PATH, device=device)
interp = make_interpolator(model, config=config, device=device)   # eval, mean head, frozen
bones = skeleton_for(interp.n_joints)["bones"]
print(f"model: {config.architecture} | clip {interp.T_frames} frames, "
      f"J={interp.n_joints}, D={interp.n_dims} | latent d={interp.d_latent}")

# ---- 2. Get the two endpoint clips x0, x1 ----
if USE_CSV:
    from youtube_motion.data import load_youtube_csv
    from architectures.data import build_clips
    bundle = load_youtube_csv(DATA_CSV)               # preprocess="none": already normalised
    clips, video_id, _ = build_clips(bundle.videos, interp.T_frames,
                                     stride=interp.T_frames // 2)
    print(f"{len(clips)} clips available from {bundle.n_videos} videos "
          f"(pick CLIP_A_INDEX / CLIP_B_INDEX in [0, {len(clips) - 1}])")
    x0, x1 = clips[CLIP_A_INDEX], clips[CLIP_B_INDEX]
else:
    # --- Plug two clips of shape (T_frames, J, D) directly ---
    x0 = ...   # e.g. my_clips[7]
    x1 = ...   # e.g. my_clips[42]

x0 = np.asarray(x0, dtype=np.float32)
x1 = np.asarray(x1, dtype=np.float32)
assert x0.shape == (interp.T_frames, interp.n_joints, interp.n_dims), x0.shape
assert x1.shape == (interp.T_frames, interp.n_joints, interp.n_dims), x1.shape

# ---- 3. Geodesic + straight-line interpolation, with diagnostics ----
res = interp.interpolate(x0, x1, T=T_SEGMENTS, method=METHOD, iters=ITERS)
print(f"\ncurvature ratio   rho = L_geodesic / L_straight = {res['curvature_ratio']:.4f}"
      f"   (<= 1; near 1 => nearly flat between the endpoints)")
print(f"constant-speed CV (geodesic segments)           = {res['speed_cv']:.4f}"
      f"   (-> 0 at convergence)")
print(f"discrete energy   straight {res['energy_straight']:.2f}  ->  "
      f"geodesic {res['energy_geodesic']:.2f}")

# ---- 4. Render the 3 clips as video: source | geodesic interp | target ----
mpl.rcParams["animation.embed_limit"] = 200   # MB, for the jshtml fallback


def show(anim, fps=FPS):
    """Prefer a compact mp4 (ffmpeg, present in Colab); fall back to JS frames."""
    try:
        html = anim.to_html5_video()
    except Exception:
        html = anim.to_jshtml(fps=fps)
    display(HTML(html))


geo_morph, alphas, node_idx = build_morph(res["geodesic_clips"], n_show=N_SHOW)
panels = [("source  x0", res["x0_recon"]),           # decoded z0 (loops)
          ("geodesic interpolation", geo_morph),      # morph through the geodesic nodes
          ("target  x1", res["x1_recon"])]            # decoded z1 (loops)
fig, anim = animate_skeleton_row(panels, edges=bones, fps=FPS, invert_y=INVERT_Y,
                                 suptitle="Geodesic interpolation")
plt.close(fig)
print(f"\nmorph plays geodesic nodes {list(node_idx)} of 0..{T_SEGMENTS} "
      f"(each a {interp.T_frames}-frame motion); endpoints loop for reference.")
show(anim)

# ---- 5. (optional) Baseline comparison: straight-line vs geodesic (spec Sec. 7) ----
if SHOW_BASELINE:
    str_morph, _, _ = build_morph(res["straight_clips"], n_show=N_SHOW)
    geo_morph2, _, _ = build_morph(res["geodesic_clips"], n_show=N_SHOW)
    panels2 = [("straight-line latent interp (naive)", str_morph),
               ("geodesic interp (on-manifold)", geo_morph2)]
    fig2, anim2 = animate_skeleton_row(panels2, edges=bones, fps=FPS, invert_y=INVERT_Y,
                                       suptitle="Naive vs geodesic interpolation")
    plt.close(fig2)
    show(anim2)
