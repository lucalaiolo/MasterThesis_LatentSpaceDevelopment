"""CARE-PD Tier-1 ARHMM state-space pipeline ([guideline]).

Reproduces Passmore et al. (2024) movement-state modelling on the three
Tier-1 mocap cohorts (BMCLab, KUL-DT-T, E-LC), pooled, adapted for gait:
3D joints + retained global motion, per-frame heading normalisation, a wider
AR lag, subject-level CV, and a clinical analysis led by freezing of gait.

    from carepd_statespace.carepd_adapter import load_h36m_cohorts, build_dataset
    from carepd_statespace.driver import run_pipeline

    # h36m 3D joints (.npz) live apart from the clinical labels (source .pkl);
    # load_h36m_cohorts reads the joints and joins the labels by walk id.
    walks = load_h36m_cohorts("<CARE-PD_h36m root>", source_dir="<cohort .pkl dir>")
    data  = build_dataset(walks, feature_set="B")     # HumanML3D decomposition
    out   = run_pipeline(data)                          # -> outputs/ + RESULTS.md
"""

from __future__ import annotations

__all__ = ["carepd_adapter", "principal_movements", "statespace", "analysis",
           "driver", "palette", "run_pipeline"]

_LAZY = {"carepd_adapter", "principal_movements", "statespace", "analysis",
         "driver", "palette"}


def __getattr__(name):
    if name == "run_pipeline":
        from .driver import run_pipeline
        return run_pipeline
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
