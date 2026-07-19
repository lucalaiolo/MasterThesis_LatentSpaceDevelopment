"""YouTube 2D-keypoint motion VAE experiments.

Trains the two ``architectures`` VAE backbones — the temporal-convolutional
model ([ARCH §3]) and the frame-token transformer ([ARCH §4]) — over the three
masked-VAE recipes ([MVAE §3-5]) and the six masking policies ([MVAE §2]) on a
2D (image-plane) keypoint dataset in the OpenPose COCO-18 layout.

Nothing here re-implements the models: the whole training / evaluation stack is
imported from ``architectures``, which is now generic in the coordinate
dimension (``n_dims``). This package only supplies the dataset adapter
(``data``), the skeleton definition (``skeleton``), and the sweep driver that
reports each model's held-out MPJPE (``driver``).

    from youtube_motion.data import load_youtube_csv
    from youtube_motion.driver import run_sweep

    bundle = load_youtube_csv("keypoints.csv")     # -> (F, 18, 2) per video
    result = run_sweep(bundle, n_epochs=100, device="cuda")
    print(result["best"])                          # winning (arch, recipe, policy)

``data`` and ``skeleton`` import with only numpy; ``run_sweep`` /
``build_base_config`` pull in ``architectures`` (and thus PyTorch) lazily, so a
torch-free environment can still ``from youtube_motion import load_youtube_csv``.
"""

from .data import (YoutubeMotionBundle, interpolate_missing, load_youtube_csv,
                   preprocess_video, root_center, torso_scale)
from .skeleton import (COCO18_BONES, COCO18_KEYPOINT_NAMES, COCO18_LIMBS,
                       COCO18_NAME_TO_IDX, N_DIMS, N_JOINTS, coco18_limbs)

__all__ = [
    # data
    "load_youtube_csv",
    "YoutubeMotionBundle",
    "preprocess_video",
    "root_center",
    "torso_scale",
    "interpolate_missing",
    # skeleton
    "COCO18_KEYPOINT_NAMES",
    "COCO18_NAME_TO_IDX",
    "COCO18_LIMBS",
    "COCO18_BONES",
    "coco18_limbs",
    "N_JOINTS",
    "N_DIMS",
    # driver (lazy — pulls in architectures / torch):
    "run_sweep",
    "build_base_config",
    "ALL_ARCHITECTURES",
    # analysis (lazy — pulls in vae_analysis / torch):
    "analyze_checkpoint",
    "analyze_best",
    "analyze_sweep",
    "compare_models",
    "coco18_skeleton",
]

_LAZY_DRIVER = {"run_sweep", "build_base_config", "ALL_ARCHITECTURES"}
_LAZY_ANALYSIS = {"analyze_checkpoint", "analyze_best", "analyze_sweep",
                  "compare_models", "coco18_skeleton"}


def __getattr__(name):
    if name in _LAZY_DRIVER:
        from . import driver
        return getattr(driver, name)
    if name in _LAZY_ANALYSIS:
        from . import analysis
        return getattr(analysis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
