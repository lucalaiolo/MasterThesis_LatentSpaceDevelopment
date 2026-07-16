"""Post-training analyses for the neonate GAITGen RVQ-VAE ([plan §6]).

Entry point ``run_analysis(model, data, config)`` writes the whole battery
to ``outputs/gaitgen_neonate/``. Submodules import on access so importing
the package does not pull in umap / hmmlearn / matplotlib.
"""

from __future__ import annotations

__all__ = ["run_analysis", "encode", "metrics", "probes", "umap_panels",
           "independence", "clustering", "codebook", "temporal", "palette",
           "report", "driver"]

_LAZY = {"encode", "metrics", "probes", "umap_panels", "independence",
         "clustering", "codebook", "temporal", "palette", "report", "driver"}


def __getattr__(name):
    if name == "run_analysis":
        from .driver import run_analysis
        return run_analysis
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
