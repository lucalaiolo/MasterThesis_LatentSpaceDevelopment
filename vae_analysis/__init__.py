"""vae_analysis — latent-space diagnostics for the masked-motion VAE.

Top-level exports match what smoke_test.py imports; the analysis
submodules stay unimported until you name them, so a
`from vae_analysis import Skeleton` does not pull in scikit-learn.
"""

from __future__ import annotations

from .interfaces import Skeleton, LatentSet, VAEModel, encode_dataset

__all__ = [
    "Skeleton", "LatentSet", "VAEModel", "encode_dataset",
    # Submodules — imported on access, not at package load.
    "posterior_geometry", "decoder_geometry", "encoder_geometry",
    "features", "masking", "dynamics", "generation", "information",
    "symmetry", "disentanglement", "two_sample", "screening", "honesty",
]


_LAZY = {"posterior_geometry", "decoder_geometry", "encoder_geometry",
         "features", "masking", "dynamics", "generation", "information",
         "symmetry", "disentanglement", "two_sample", "screening", "honesty"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
