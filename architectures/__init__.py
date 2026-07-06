"""Public API for the masked neonate-motion VAE training package.

`TrainingConfig` and the mask policies import cleanly with only numpy.
The model classes and `build_model` pull in PyTorch — they are exposed
here via a lazy `__getattr__` so a torch-free environment can still
`from architectures import TrainingConfig, UniformMask, ...`.
"""

from .config import TrainingConfig
from .mask_policies import (
    MaskPolicy,
    NoMask,
    UniformMask,
    TopKSpeedMask,
    SoftmaxSpeedMask,
    PerFrameSpeedMask,
    LimbMask,
    build_policy,
)

__all__ = [
    "TrainingConfig",
    "MaskPolicy",
    "NoMask",
    "UniformMask",
    "TopKSpeedMask",
    "SoftmaxSpeedMask",
    "PerFrameSpeedMask",
    "LimbMask",
    "build_policy",
    # Lazy — only imported when accessed:
    "ConvVAE",
    "TransformerVAE",
    "build_model",
]


_LAZY = {"ConvVAE", "TransformerVAE", "build_model"}


def __getattr__(name):
    if name in _LAZY:
        from . import models
        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
