# youtube_motion

Training + sweep code for the **YouTube 2D-keypoint** motion dataset. It reuses
the whole `architectures` VAE stack — the two backbones ([ARCH §3, §4]), the
three masked-VAE recipes ([MVAE §3-5]), the six masking policies ([MVAE §2]),
and the held-out MPJPE evaluators ([MVAE §7]) — and only switches the coordinate
dimension to **2D** (`n_dims=2`) and the skeleton to **COCO-18**.

Nothing about the models is re-implemented here. `architectures` was made
generic in the coordinate dimension `D`, so the exact same conv / transformer
VAEs run on image-plane keypoints; this folder adds the dataset adapter, the
skeleton, and the driver that sweeps configurations and reports each model's
performance.

## Install

```
pip install numpy torch          # matplotlib optional, for the training plots
```

`data` and `skeleton` need only numpy; training needs torch.

## The dataset

The export is **long** — one row per joint per frame — with the columns shown
in the dataset preview:

```
video_number,video,bp,frame,x,y,fps,pixel_x,pixel_y,time,part_idx
0,video_000000,RShoulder,0,-0.4135,1.3078,29.97,1280,720,0.0,2.0
0,video_000000,RElbow,0,-0.2862,0.7706,29.97,1280,720,0.0,3.0
...
```

`part_idx` runs 0..17 in the OpenPose **COCO-18** layout (Nose, Neck, R/L
Shoulder-Elbow-Wrist, R/L Hip-Knee-Ankle, R/L Eye, R/L Ear — see
`skeleton.py`). `x`/`y` are the image-plane coordinates (already
torso-normalised in this export); `pixel_x`/`pixel_y` are the frame size.

`load_youtube_csv` pivots this into the shape the training loop consumes
everywhere else — a list of per-video arrays, each `(F_v, J, 2)`:

```python
from youtube_motion.data import load_youtube_csv

bundle = load_youtube_csv("keypoints.csv")   # -> YoutubeMotionBundle
print(bundle.summary())
# 1234 videos, 456789 frames total, J=18, D=2, len[min/med/max]=..., fps~29.97
```

Dropped detections (occluded joints, missing frames) arrive as gaps and are
**linearly interpolated over time** per joint; a joint never seen in a clip is
filled with 0. Frame indices are shifted so each video starts at 0, preserving
gaps (and timing).

Coordinates are used **as stored** by default (`preprocess="none"`). If your
export holds raw pixels instead of normalised coordinates, pass
`preprocess="center"` (root-centre on the Neck) or `preprocess="center_scale"`
(root-centre + divide by the median torso length).

## Sweep every configuration

The point of this folder: train **both backbones** across **all three recipes**
and **every masking policy**, and report the held-out MPJPE of each.

Command line:

```
python -m youtube_motion.driver --csv keypoints.csv \
    --out checkpoints/youtube_motion --epochs 100 --device cuda
```

From Python:

```python
from youtube_motion.data import load_youtube_csv
from youtube_motion.driver import run_sweep

bundle = load_youtube_csv("keypoints.csv")
result = run_sweep(bundle, out_dir="checkpoints/youtube_motion",
                   n_epochs=100, device="cuda")

print(result["best"])            # winning (architecture, recipe, mask_policy)
print(result["results_md"])      # path to the ranked markdown table
```

Each valid `(architecture × recipe × mask_policy)` combination is trained with
`architectures.train`, then scored on the **same** time-based validation split
`train` held out internally (so no run is scored on its own training data). The
grid is:

- **architectures**: `conv`, `transformer`
- **recipes**: `1`, `2`, `3`
- **mask policies**: `none`, `uniform`, `top_k_speed`, `softmax_speed`,
  `per_frame_speed`, `limb`

Invalid combinations are skipped automatically: recipes 2 and 3 require a
masking policy (not `none`); `limb` uses the COCO-18 limb groups the bundle
carries. That leaves 16 runs per backbone (6 + 5 + 5), 32 in total by default.
Restrict any axis to iterate faster:

```python
run_sweep(bundle, architectures=("conv",), recipes=(1, 3),
          mask_policies=("uniform", "limb"), n_epochs=50)
```

### Output

`run_sweep` writes two files under `out_dir` and returns them in the result:

| File | What it holds |
|:---|:---|
| `results.json` | machine-readable: every run's params, best epoch, final losses, all three MPJPE variants, checkpoint path |
| `results.md`   | a ranked markdown table (best-first) with a highlighted winner |

plus a printed ranking. Per-run checkpoints and `history.json` land in
`<out_dir>/<arch>/recipe{N}_{policy}/` exactly as `architectures.train` writes
them. A run that errors (e.g. CUDA OOM) is recorded with `status="error"` and
the sweep continues.

Ranking is by `mpjpe_all` (reconstruction) by default; pass
`metric="mpjpe_inpainted"` to rank by hidden-joint inpainting instead.

## The three recipes and the masking policies

Same as the 3D pipeline — see `architectures/README.md` for the full account.
In one line each:

| Recipe | Passes | Reconstruction | KL from |
|:---:|:---|:---|:---|
| 1 | one masked pass | MSE on all joints | the masked pass |
| 2 | unmasked + masked | MSE(primary) + λ·MSE(aux) | the unmasked pass |
| 3 | one masked pass, two heads | full-clip MSE + λ·hidden-only MSE | the masked pass |

Masking policies ([MVAE §2]): `uniform` (i.i.d. per joint/frame), the three
speed-based policies (`top_k_speed`, `softmax_speed`, `per_frame_speed`), `limb`
(hide one whole limb), and `none` (plain-VAE ablation, Recipe 1 only).

## Files

| File | What it does |
|:---|:---|
| `skeleton.py` | COCO-18 joint names, limb groups, bones, root / torso joints |
| `data.py`     | long-CSV → list of `(F, 18, 2)` videos; missing-joint interpolation; preprocessing; `YoutubeMotionBundle` |
| `driver.py`   | `run_sweep` (arch × recipe × policy), `build_base_config`, CLI, ranked `results.{json,md}` |
| `smoke_test.py` | synthetic-data end-to-end run of the whole sweep (worked example) |
| `test_no_torch.py` | numpy-only checks: skeleton, adapter, preprocessing, 2D param counts |

## Sanity check

With torch installed:

```
python -m youtube_motion.smoke_test        # synthetic 2D, small sweep, 2 epochs each
```

Without torch (adapter + counts only):

```
python -m youtube_motion.test_no_torch
```

## Parameter budgets (2D, J=18, T=32, d_z=32)

The 2D models are a touch smaller than their 3D counterparts — one fewer
coordinate channel on the input and output heads:

- Convolutional: **276,196** parameters (Recipe 1/2; Recipe 3 adds the
  inpainting head).
- Transformer: **686,980** parameters.

`test_no_torch.py` recomputes these analytically (no torch needed) and checks
the encoder input width is `(D+1)·J`.

## What comes next

The sweep produces the trained checkpoints and the performance table; the
downstream latent-space analysis (the "then we think about the analysis" step)
can read those checkpoints with `architectures.analyze.load_checkpoint`, which
rebuilds the 2D model straight from the saved config.
