"""Neonate GAITGen: disentangled Residual VQ-VAE for 2D neonate keypoints.

Phase A of the build plan — the motion/pathology disentangled RVQ-VAE
(paper Sec. 3.1.1), adapted to 2D keypoints (no SO(3) rotation loss) with a
generic discrete conditioning label. The Mask / Residual Transformers for
conditional generation (paper Sec. 3.1.2-3) are Phase B and are not built.

    from neonate_gaitgen import GaitGenConfig, train
    from neonate_gaitgen.preprocess import Sequence, build_windowed_data

    cfg  = GaitGenConfig(n_joints=17, n_classes=2, label_type="nominal")
    data = build_windowed_data(sequences, cfg)   # sequences: list[Sequence]
    out  = train(cfg, data)
"""

from __future__ import annotations

from .config import GaitGenConfig

__all__ = ["GaitGenConfig", "build_model", "train", "load_checkpoint",
           "preprocess", "models", "analysis"]

_LAZY = {"preprocess", "models", "analysis"}


def __getattr__(name):
    if name == "build_model":
        from .models import build_model
        return build_model
    if name in ("train", "load_checkpoint"):
        from . import train as _t
        return getattr(_t, name)
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
