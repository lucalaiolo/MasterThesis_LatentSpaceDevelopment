"""Post-hoc phenotype-structure analysis for CARE-PD ([post-hoc plan]).

The mixture-prior models are gone ([post-hoc plan §0]); the phenotype claim
is now made post hoc on the two core latents — the plain VAE (baseline) and
the CVAE (target). This subpackage encodes those latents and runs the
structure battery, writing to ``outputs/posthoc/``:

    §1   BIC sanity check (is the latent worth clustering?)
    §2   post-hoc clustering (k-means, GMM, HDBSCAN) + stability + agreement
    §3   within-severity substructure
    §4   HMM regimes + PELT change points vs annotated FoG
    §5   continuous linear probes (subject-split)
    §7   summary.md

Entry point::

    from vae_analysis.posthoc import run_posthoc
    run_posthoc(vae_checkpoint=..., cvae_checkpoint=..., bundle=...)

Submodules stay unimported until named, so importing the package does not
pull in scikit-learn / matplotlib.
"""

from __future__ import annotations

__all__ = [
    "run_posthoc", "build_posthoc_data", "load_encoder",
    "PosthocData", "WalkMeta", "TorchCohortEncoder",
    # submodules, imported on access:
    "data", "palette", "clustering", "agreement", "temporal", "probes",
    "report", "driver",
]

_LAZY_SUBMODULES = {"data", "palette", "clustering", "agreement", "temporal",
                    "probes", "report", "driver"}


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name == "run_posthoc":
        from .driver import run_posthoc
        return run_posthoc
    if name in ("build_posthoc_data", "load_encoder", "PosthocData",
                "WalkMeta", "TorchCohortEncoder"):
        from . import data as _d
        return getattr(_d, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
