"""The abnormality index: PC1 of the four endpoints, against a normative band.

Every construct upstream ends in one scalar per recording, and each is
contrasted with the group on its own. This module is the **composite readout**:
the four endpoints are put in one ``N x 4`` matrix, standardised, reduced by PCA
to the first principal component -- a continuous *abnormality index* carrying
the largest single share of the variance the four share -- and turned into a
binary flag by a **normative reference range** rather than by a fitted
classifier.

Why a reference range and not a classifier. With ``n1 = 6`` positives, any
model with free parameters fitted against the label -- logistic regression, a
tree, an SVM, a tuned cut-off -- has more freedom than the positives can
constrain, and its in-sample accuracy says almost nothing. The normative
approach fits nothing to the outcome: the healthy cohort defines a range, and a
recording is flagged when it falls outside it. That is a screening instrument,
"this infant is unlike healthy development", not a prediction of the label.

**The procedure**, in the order it runs:

1. **Matrix.** ``X`` is ``N x p``, one row per recording, one column per
   endpoint (:data:`FEATURES`: fluency ``Phi``, Kemeny ``K``, synchrony
   ``mean F``, FidgetyFind ``FF``).
2. **Standardise.** ``Z = (X - mean) / sd`` column-wise over the whole cohort,
   labels not consulted. The four endpoints live on incomparable scales, so PCA
   on the raw matrix would be PCA on whichever column happens to have the
   largest variance.
3. **PCA.** The SVD of the centred ``Z``; ``PC1 = Z w1`` with ``w1`` the
   leading right singular vector. Its explained-variance ratio says how much of
   the four-endpoint structure one number keeps.
4. **Orient.** A principal component's sign is arbitrary. It is fixed here by
   the *stated* pathological directions of the endpoints (:data:`DIRECTIONS`):
   ``sign(w1 . d)``, so a larger index means more abnormal. Only the two
   endpoints whose pole METHODS actually states get a vote -- high WCLR-PP
   coupling is the cramped-synchronised pole, high FidgetyFind is normal --
   and the other two abstain. **No label enters this**, and the two-sided
   readout below does not depend on it at all.
5. **Normative band.** ``mu0`` and ``sd0`` are the mean and sample SD of the
   index over the ``label == 0`` recordings alone. The band is
   ``mu0 +- n_sd * sd0`` (``n_sd = 2``).
6. **Flag.** ``1`` when the index falls strictly outside the band, ``0`` inside.
   Two-sided by default -- "unlike the healthy distribution", in either
   direction -- which is also what makes the readout invariant to step 4.

**What the numbers mean, and what they do not.** The band is built from the
healthy recordings and then applied to them, so the **specificity is in-sample**
and is a description of the fit, not an estimate of out-of-sample specificity.
:func:`abnormality_index` therefore also reports a leave-one-out specificity,
where each healthy recording is scored against a band computed from the other
31. The sensitivity needs no such correction: no abnormal recording enters the
band, so it is already out-of-sample. The standardisation and the PCA see all
``N`` rows, which is transductive but label-free.

Nothing here is corrected for multiplicity, in keeping with the rest of the
reported inference: the index is one more endpoint, contrasted by the same
exact Mann-Whitney permutation null as the four it is built from, and the 2x2
table carries Fisher's exact p.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from scipy import stats

import a1_stats as ST


# ---------------------------------------------------------------------------
# the four endpoints
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Feature:
    """One endpoint of the pipeline, and where it lives in a results object.

    ``direction`` is the *stated* pathological pole: ``+1`` when a larger value
    is the abnormal one, ``-1`` when a smaller one is, and ``0`` when METHODS
    states no direction for it -- in which case the feature abstains from
    orienting PC1 (it still enters the matrix, the standardisation and the
    component like any other).
    """

    key: str
    label: str
    direction: int
    block: str          # results key holding it ("" = the primary model)
    path: tuple         # keys to walk from that block to the vector


FEATURES: tuple[Feature, ...] = (
    Feature("phi", "fluency Phi", 0, "", ("phi", "excess")),
    Feature("kemeny", "Kemeny (jumps)", 0, "", ("kemeny_per_subject",)),
    Feature("mean_F", "synchrony (mean F)", +1, "wclrpp", ("mean_F",)),
    Feature("FF", "FidgetyFind FF", -1, "fidgetyfind", ("FF",)),
)

#: The stated pathological pole of each endpoint, as used to orient PC1.
#: ``mean F``: high inter-limb coupling is the cramped-synchronised pole.
#: ``FF``: higher is normal, so the abnormal group is expected below.
#: ``Phi`` and ``Kemeny`` have no stated pole and abstain.
DIRECTIONS: dict[str, int] = {f.key: f.direction for f in FEATURES}


# ---------------------------------------------------------------------------
# the N x p matrix
# ---------------------------------------------------------------------------
def _walk(obj, path):
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def _as_float(v, n=None):
    """A results vector as float, with ``results.json``'s ``None`` back to NaN.

    ``run_analysis._json_safe`` writes every non-finite float as JSON ``null``,
    so a matrix rebuilt from ``results.json`` must put the NaNs back or a
    declined FidgetyFind score would arrive as ``None`` and poison the column.
    """
    if v is None:
        return None
    arr = np.array([np.nan if x is None else x for x in np.asarray(v).ravel()],
                   dtype=float)
    return arr if (n is None or arr.size == n) else None


def load_results(source):
    """A results object from a dict, a ``results.json`` path, or an outdir."""
    if isinstance(source, dict):
        return source
    path = str(source)
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    with open(path) as fh:
        return json.load(fh)


def feature_matrix(source, features: tuple[Feature, ...] = FEATURES) -> dict:
    """The ``N x p`` endpoint matrix out of a run, ready for :func:`abnormality_index`.

    ``source`` is a results dict, the path of a ``results.json``, or the output
    directory holding one -- so a finished run can be re-read later without
    recomputing anything.

    Returns ``X`` (``N x p``, NaN where an endpoint declined to score), the
    ``labels``, the ``videos``, the ``features`` that were found and the
    ``missing`` ones (a skipped block, say). Endpoints are taken in the order of
    ``features``; one that is absent is dropped from the matrix and named in
    ``missing`` rather than filled in.
    """
    results = load_results(source)
    primary = results.get("primary")
    labels = np.asarray(results.get("labels", []), int)
    videos = list(results.get("video_names", []))
    n = len(labels) or len(videos)

    cols, kept, missing = [], [], []
    for f in features:
        block = results.get(primary, {}) if f.block == "" else \
            results.get(f.block) or {}
        v = _as_float(_walk(block, f.path), n)
        if v is None:
            missing.append(f.label)
            continue
        cols.append(v)
        kept.append(f)
    X = np.column_stack(cols) if cols else np.empty((n, 0))
    return {"X": X, "labels": labels, "videos": videos,
            "features": tuple(kept), "names": [f.label for f in kept],
            "keys": [f.key for f in kept],
            "directions": np.array([f.direction for f in kept], float),
            "missing": missing, "primary": primary, "n": n}


def save_feature_matrix(fm: dict, path: str) -> str:
    """Cache the matrix so a later session can score it without a rerun."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez(path, X=np.asarray(fm["X"], float),
             labels=np.asarray(fm["labels"], int),
             videos=np.asarray(fm["videos"], dtype=object),
             names=np.asarray(fm["names"], dtype=object),
             keys=np.asarray(fm["keys"], dtype=object),
             directions=np.asarray(fm["directions"], float))
    return path


def load_feature_matrix(path: str) -> dict:
    """Read back what :func:`save_feature_matrix` wrote."""
    with np.load(path, allow_pickle=True) as z:
        fm = {"X": z["X"].astype(float),
              "labels": z["labels"].astype(int),
              "videos": [str(v) for v in z["videos"]],
              "names": [str(v) for v in z["names"]],
              "keys": [str(v) for v in z["keys"]],
              "directions": z["directions"].astype(float)}
    fm["n"] = len(fm["labels"])
    fm["missing"] = []
    fm["features"] = tuple(f for f in FEATURES if f.key in fm["keys"])
    return fm


# ---------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------
def _standardise(X, missing="impute"):
    """Column z-scores over the whole cohort; how NaNs are handled is declared.

    ``impute`` keeps every recording and sets a missing entry to its column
    mean, i.e. ``z = 0``: the value that adds no information and no leverage,
    which is the honest stand-in when one endpoint declined to score a
    recording the others did score. ``drop`` removes the row from the fit and
    from the readout entirely, leaving its index NaN.
    """
    X = np.asarray(X, float)
    ok_row = np.isfinite(X).all(1)
    fit = X if missing == "impute" else X[ok_row]
    mean = np.array([np.nanmean(c) if np.isfinite(c).any() else np.nan
                     for c in fit.T])
    sd = np.array([np.nanstd(c) if np.isfinite(c).any() else np.nan
                   for c in fit.T])
    bad = [j for j, s in enumerate(sd) if not np.isfinite(s) or s <= 0]
    if bad:
        raise ValueError(f"endpoint column(s) {bad} have no usable spread "
                         f"(all missing, or constant across the cohort), so "
                         f"they cannot be standardised")
    imputed = ~np.isfinite(X)
    Z = np.where(imputed, 0.0, (X - mean) / sd)
    scored = np.ones(len(X), bool) if missing == "impute" else ok_row
    return Z, mean, sd, scored, imputed


def _orient(w, directions):
    """Fix PC1's arbitrary sign from the stated pathological poles, not the label.

    ``sign(w . d)`` points the component at the abnormal pole of the endpoints
    that have one. If the two abstain and the other two cancel exactly, the
    largest loading is made positive instead, which is at least deterministic;
    the returned ``method`` says which happened.
    """
    d = np.asarray(directions, float)
    vote = float(np.dot(w, d))
    if abs(vote) > 1e-12:
        return (1.0 if vote > 0 else -1.0), vote, "stated pathological poles"
    j = int(np.argmax(np.abs(w)))
    return (1.0 if w[j] >= 0 else -1.0), vote, "largest loading made positive"


def _confusion(flag, labels):
    y = np.asarray(labels, int)
    f = np.asarray(flag, int)
    tp = int(np.sum((f == 1) & (y == 1)))
    fp = int(np.sum((f == 1) & (y == 0)))
    fn = int(np.sum((f == 0) & (y == 1)))
    tn = int(np.sum((f == 0) & (y == 0)))
    def _rate(a, b):
        return float(a / b) if b else float("nan")
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "sensitivity": _rate(tp, tp + fn),
           "specificity": _rate(tn, tn + fp),
           "ppv": _rate(tp, tp + fp), "npv": _rate(tn, tn + fn),
           "accuracy": _rate(tp + tn, tp + tn + fp + fn),
           "n_flagged": tp + fp, "n": tp + tn + fp + fn}
    out["balanced_accuracy"] = float(
        np.mean([out["sensitivity"], out["specificity"]]))
    out["youden_j"] = out["sensitivity"] + out["specificity"] - 1.0
    odds, p = stats.fisher_exact([[tp, fp], [fn, tn]])
    out["fisher_p"] = float(p)
    out["odds_ratio"] = float(odds)
    return out


def abnormality_index(X, labels, directions=None, names=None, n_sd: float = 2.0,
                      side: str = "two", missing: str = "impute",
                      boot: int = 10_000) -> dict:
    """PC1 of the standardised endpoints, cut at ``n_sd`` SD of the healthy cohort.

    ``X`` is ``N x p`` (rows = recordings, columns = endpoints), ``labels`` is
    ``1`` for abnormal and ``0`` for normal. ``directions`` gives each column's
    stated pathological pole (``+1`` / ``-1`` / ``0`` to abstain) and is used
    only to orient PC1, never the label.

    ``side`` is ``"two"`` (the reported readout: outside the band in either
    direction), ``"upper"`` or ``"lower"``. ``missing`` is ``"impute"`` or
    ``"drop"``; see :func:`_standardise`.

    Returns the index, the band, the flag and the readout. The AUC of the
    continuous index is the same exact Mann-Whitney contrast every other
    endpoint gets; the 2x2 table of the binary flag carries Fisher's exact p.
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(labels, int)
    n, p = X.shape
    if n != len(y):
        raise ValueError(f"X has {n} rows but there are {len(y)} labels")
    if p < 2:
        raise ValueError(f"the index needs at least two endpoints, got {p}")
    if side not in ("two", "upper", "lower"):
        raise ValueError(f"side must be 'two', 'upper' or 'lower', got {side!r}")
    if missing not in ("impute", "drop"):
        raise ValueError(f"missing must be 'impute' or 'drop', got {missing!r}")
    if directions is None:
        directions = np.zeros(p)
    names = list(names) if names is not None else [f"x{j}" for j in range(p)]

    Z, mean, sd, scored, imputed = _standardise(X, missing)
    Zc = Z - Z[scored].mean(0)
    _, S, Vt = np.linalg.svd(Zc[scored], full_matrices=False)
    ev = S ** 2 / max(len(np.flatnonzero(scored)) - 1, 1)
    tot = float((S ** 2).sum())
    ratio = S ** 2 / tot if tot > 0 else np.full(len(S), np.nan)

    w = Vt[0]
    sign, vote, how = _orient(w, directions)
    w = w * sign
    pc1 = np.where(scored, Zc @ w, np.nan)

    # normative band: the healthy cohort alone, which is the whole of what the
    # label is used for here.
    healthy = (y == 0) & scored
    n0 = int(healthy.sum())
    if n0 < 2:
        raise ValueError(f"the normative band needs at least two normal "
                         f"recordings, got {n0}")
    mu0 = float(np.mean(pc1[healthy]))
    sd0 = float(np.std(pc1[healthy], ddof=1))
    if not np.isfinite(sd0) or sd0 <= 0:
        raise ValueError("the normal recordings' index has zero spread, so no "
                         "normative band can be built from it")
    lo, hi = mu0 - n_sd * sd0, mu0 + n_sd * sd0
    dev = (pc1 - mu0) / sd0

    def _flag(v, m, s):
        h, l = m + n_sd * s, m - n_sd * s
        if side == "upper":
            return v > h
        if side == "lower":
            return v < l
        return (v > h) | (v < l)

    flag = np.where(scored, _flag(pc1, mu0, sd0), np.nan)
    fin = scored & np.isfinite(flag)
    readout = _confusion(flag[fin].astype(int), y[fin])

    # Leave-one-out: each healthy recording scored against a band built from
    # the other n0-1, which is what its specificity would be if it had not
    # helped define the band. The abnormal recordings never enter the band, so
    # their sensitivity is already out-of-sample and is repeated unchanged.
    loo = {}
    if n0 >= 3:
        idx = np.flatnonzero(healthy)
        f_loo = np.zeros(len(idx), bool)
        for i, k in enumerate(idx):
            rest = pc1[np.setdiff1d(idx, [k])]
            f_loo[i] = bool(_flag(pc1[k], float(np.mean(rest)),
                                  float(np.std(rest, ddof=1))))
        loo = {"n": int(len(idx)), "n_flagged": int(f_loo.sum()),
               "specificity": float(1.0 - f_loo.mean()),
               "note": ("each normal recording scored against a band built "
                        "from the other normal recordings; the sensitivity "
                        "needs no such correction, as no abnormal recording "
                        "enters the band")}

    group = ST.mannwhitney(pc1[(y == 1) & scored], pc1[healthy], boot=boot)

    return {
        "pc1": pc1, "deviation": dev, "flag": flag, "scored": scored,
        "z": Z, "mean": mean, "sd": sd, "imputed": imputed,
        "n_imputed": int(imputed.sum()), "n_scored": int(scored.sum()),
        "n": n, "names": names, "keys": list(names),
        "loadings": w, "explained_variance": ev,
        "explained_ratio": ratio, "pc1_explained": float(ratio[0]),
        "orientation": {"sign": float(sign), "vote": vote, "method": how,
                        "directions": np.asarray(directions, float).tolist()},
        "band": {"mu0": mu0, "sd0": sd0, "n_sd": float(n_sd), "lo": lo,
                 "hi": hi, "side": side, "n_normal": n0},
        "readout": readout, "loo": loo, "group": group,
        "missing_policy": missing,
        "note": ("PC1 of the standardised endpoints, cut at "
                 f"{n_sd:g} SD of the normal cohort's own distribution; "
                 "no parameter is fitted against the label"),
    }


def index_frame(fm: dict, ix: dict):
    """The per-recording table: raw endpoints, z-scores, index, band and flag."""
    import pandas as pd
    X, Z = np.asarray(fm["X"], float), np.asarray(ix["z"], float)
    rows = {"subject": np.arange(1, fm["n"] + 1), "video": fm["videos"],
            "label": np.asarray(fm["labels"], int)}
    for j, k in enumerate(fm["keys"]):
        rows[k] = X[:, j]
    for j, k in enumerate(fm["keys"]):
        rows[f"z_{k}"] = np.where(ix["imputed"][:, j], np.nan, Z[:, j])
    rows["pc1"] = ix["pc1"]
    rows["normative_z"] = ix["deviation"]
    rows["flag"] = ix["flag"]
    rows["correct"] = np.where(np.isfinite(ix["flag"]),
                               ix["flag"] == np.asarray(fm["labels"], float),
                               np.nan)
    return pd.DataFrame(rows)


def describe(fm: dict, ix: dict) -> str:
    """The readout as lines of text, for the run log and for a notebook."""
    r, b, L = ix["readout"], ix["band"], []
    L.append(f"  matrix: {ix['n']} recordings x {len(fm['keys'])} endpoints "
             f"({', '.join(fm['names'])})")
    if fm.get("missing"):
        L.append(f"     not available, so not in the index: "
                 f"{', '.join(fm['missing'])}")
    if ix["n_imputed"]:
        L.append(f"     {ix['n_imputed']} endpoint value(s) did not score and "
                 f"were set to their column mean (z = 0)"
                 if ix["missing_policy"] == "impute" else
                 f"     {ix['n'] - ix['n_scored']} recording(s) dropped for a "
                 f"missing endpoint")
    L.append(f"  PC1 keeps {ix['pc1_explained']:.1%} of the standardised "
             f"variance"
             + (f" (PC2 {ix['explained_ratio'][1]:.1%})"
                if len(ix["explained_ratio"]) > 1 else ""))
    L.append("     loadings: " + ",  ".join(
        f"{nm} {wj:+.3f}" for nm, wj in zip(fm["names"], ix["loadings"])))
    L.append(f"     sign fixed by {ix['orientation']['method']} "
             f"(higher index = more abnormal); the two-sided readout does not "
             f"depend on it")
    L.append(f"  normative band from the {b['n_normal']} normal recordings: "
             f"mean {b['mu0']:+.3f}, SD {b['sd0']:.3f}, so "
             f"{b['n_sd']:g} SD gives [{b['lo']:+.3f}, {b['hi']:+.3f}] "
             f"({b['side']}-sided)")
    L.append(f"  flagged {r['n_flagged']}/{r['n']} recordings: "
             f"sensitivity {r['sensitivity']:.3f} ({r['tp']}/{r['tp'] + r['fn']}"
             f"), specificity {r['specificity']:.3f} "
             f"({r['tn']}/{r['tn'] + r['fp']})")
    L.append(f"     PPV {r['ppv']:.3f}, NPV {r['npv']:.3f}, balanced accuracy "
             f"{r['balanced_accuracy']:.3f}, Youden J {r['youden_j']:+.3f}; "
             f"Fisher exact p = {r['fisher_p']:.4g}")
    if ix.get("loo"):
        lo = ix["loo"]
        L.append(f"     leave-one-out specificity {lo['specificity']:.3f} "
                 f"({lo['n'] - lo['n_flagged']}/{lo['n']}): the band is built "
                 f"from the normal cohort and then applied to it, so the "
                 f"in-sample specificity above is a description of the fit")
    g = ix["group"]
    L.append(f"  the index as a continuous endpoint: AUC {g['auc']:.3f} "
             f"[{g.get('auc_lo', float('nan')):.3f}, "
             f"{g.get('auc_hi', float('nan')):.3f}], p = {g['p']:.4g} "
             f"({g['method']})")
    return "\n".join(L)
