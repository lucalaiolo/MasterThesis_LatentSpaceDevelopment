"""Load the joblib-dumped fitted models without ``ssm`` installed (METHODS §3.2).

``ssm`` is needed only to reconstruct the model *object*; every quantity the
analysis uses (transition matrix, Viterbi path, AR parameters) is a plain numpy
array stored alongside it. A ``MetaPathFinder`` therefore manufactures stub
modules for any import beginning ``ssm`` or ``autograd``, so the unpickler
resolves the class names and joblib recovers the arrays. The model object
becomes an inert placeholder and is never called.

:func:`normalise` flattens the two dict layouts this project produces into one
schema, so downstream code never has to guess where a key lives:

* the METHODS §3.1 dumps, with the arrays under a ``res`` sub-dict and
  ``vidid`` / ``lengths`` at the top level;
* the flat ``joblib`` blobs written by the Stage-4 notebook, which carry
  ``res``, ``Z``, ``lengths``, ``vidid`` side by side.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types

import numpy as np

STUBBED: set[str] = set()


class _Stub:
    """Accepts any construction/reconstruction protocol pickle may use."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(
            state if isinstance(state, dict) else {"_state": state})

    def append(self, x):
        self.__dict__.setdefault("_items", []).append(x)

    def extend(self, xs):
        self.__dict__.setdefault("_items", []).extend(xs)

    def __setitem__(self, k, v):
        self.__dict__.setdefault("_map", {})[k] = v

    def __repr__(self):
        return f"<Stub {getattr(type(self), '_cls', '?')}>"


def _make_module(fullname: str) -> types.ModuleType:
    m = types.ModuleType(fullname)
    m.__path__ = []                                  # mark as a package

    def __getattr__(name, _fn=fullname):
        STUBBED.add(f"{_fn}.{name}")
        cls = type(name, (_Stub,), {"_cls": f"{_fn}.{name}", "__module__": _fn})
        setattr(sys.modules[_fn], name, cls)         # cache so identity is stable
        return cls

    m.__getattr__ = __getattr__
    return m


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    ROOTS = ("ssm", "autograd")

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _make_module(spec.name)

    def exec_module(self, module):
        pass


def install_stubs() -> None:
    """Idempotently install the stub finder ahead of the real import system."""
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())


install_stubs()

import joblib  # noqa: E402  — must follow install_stubs()


def load(path: str):
    """Raw ``joblib.load``; ``ssm``/``autograd`` classes resolve to stubs."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# schema normalisation
# ---------------------------------------------------------------------------
_ARRAY_KEYS = (
    "states", "transition", "occupancy", "lengths", "vidid", "Z",
    "means", "covars", "dwell_windows", "dwell_seconds",
    "ar_As", "ar_bs", "ar_Sigmas", "stationary",
)


def _search(blob: dict, key: str):
    """Find ``key`` at the top level or one level down (e.g. inside ``res``)."""
    if key in blob:
        return blob[key]
    for v in blob.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return None


def normalise(blob: dict, name: str = "model") -> dict:
    """Flatten a loaded blob to a single schema.

    Returns a dict with whichever of :data:`_ARRAY_KEYS` were present (as numpy
    arrays), plus ``k`` (number of states), ``n_subjects``, ``name``, and
    ``raw`` (the original blob, so nothing is lost).
    """
    if not isinstance(blob, dict):
        raise TypeError(f"{name}: expected a dict at the top level, "
                        f"got {type(blob).__name__}")
    out: dict = {"name": name, "raw": blob}
    for key in _ARRAY_KEYS:
        v = _search(blob, key)
        if v is not None:
            out[key] = np.asarray(v)

    if "states" not in out:
        raise KeyError(
            f"{name}: no 'states' (Viterbi path) found at the top level or one "
            f"level down. Present keys: {sorted(blob)}")
    if "transition" not in out:
        raise KeyError(f"{name}: no 'transition' matrix found.")

    st = out["states"].astype(int)
    out["states"] = st
    out["k"] = int(st.max()) + 1
    A = out["transition"]
    if A.shape != (out["k"], out["k"]):
        raise ValueError(
            f"{name}: transition is {A.shape} but the Viterbi path uses "
            f"{out['k']} states.")

    # vidid may be missing when only `lengths` was stored: rebuild it.
    if "vidid" not in out and "lengths" in out:
        out["vidid"] = np.repeat(np.arange(len(out["lengths"])),
                                out["lengths"].astype(int))
    if "vidid" in out:
        out["vidid"] = out["vidid"].astype(int)
        out["n_subjects"] = int(out["vidid"].max()) + 1
    elif "lengths" in out:
        out["n_subjects"] = int(len(out["lengths"]))
    else:
        raise KeyError(f"{name}: neither 'vidid' nor 'lengths' found; cannot "
                       f"attribute windows to subjects.")

    if "lengths" in out:
        out["lengths"] = out["lengths"].astype(int)
        if int(out["lengths"].sum()) != len(st):
            raise ValueError(
                f"{name}: lengths sum to {int(out['lengths'].sum())} but the "
                f"Viterbi path has {len(st)} windows.")
    return out


def load_normalised(path: str, name: str | None = None) -> dict:
    return normalise(load(path), name or path)


def describe(d, indent: int = 0, maxdepth: int = 3) -> None:
    """Print the object graph of a loaded blob, arrays shown by shape/dtype."""
    pad = "  " * indent
    if not isinstance(d, dict):
        print(f"{pad}{type(d).__name__}")
        return
    for k, v in d.items():
        key = str(k)                    # keys may be tuples, e.g. (K, n_lags)
        if isinstance(v, dict) and indent < maxdepth:
            print(f"{pad}{key}:  dict({len(v)})")
            describe(v, indent + 1, maxdepth)
        elif isinstance(v, np.ndarray):
            extra = f"  = {np.round(v, 4)}" if v.size <= 12 and v.ndim <= 1 else ""
            print(f"{pad}{key:26s} ndarray {str(v.shape):16s} "
                  f"{str(v.dtype):8s}{extra}")
        elif isinstance(v, (list, tuple)):
            print(f"{pad}{key:26s} {type(v).__name__}({len(v)})")
        elif isinstance(v, (int, float, str, bool, type(None))):
            print(f"{pad}{key:26s} {type(v).__name__:8s} = {v}")
        else:
            print(f"{pad}{key:26s} {type(v).__name__}")
