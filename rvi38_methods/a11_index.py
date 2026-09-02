"""The abnormality index: the largest signed deviation across the four endpoints.

Read alone, each of the four endpoints is null. This module is the **composite
readout**: the four constructs are put in one ``N x 4`` matrix, standardised
against the normal recordings alone, signed by each construct's stated
pathological pole, and reduced to one scalar per recording by taking the
**maximum**.

**Why the maximum, and not an average or a component.** The four constructs
span both periods of the assessment. Fluency, mixing and synchrony read the
writhing period, from term to about two months, whose abnormal forms are a poor
repertoire and cramped-synchronised movement; FidgetyFind reads the fidgety
period, from about two to five months. The pipeline is a general representation
of infant motion, not a detector for one pattern -- a poor repertoire early and
absent fidgety movement later both reflect reduced movement variety, so an
abnormal recording may deviate on any of the four constructs, not only the one
its label names.

The assessment itself needs only one abnormal pattern: a clinician calls a
recording abnormal on a poor repertoire, *or* on cramped-synchronised movement,
*or* on absent fidgety movement, and the other patterns need not appear. A score
that follows the assessment must therefore fire when any single construct
signals abnormality, not only when several agree. The maximum reads the one most
abnormal construct and ignores the rest, which is what the assessment does. An
average or a principal component would instead ask the constructs to agree, and
would dilute a single-axis deviation against three quiet ones.

**The procedure**, in the order it runs:

1. **Matrix.** ``X`` is ``N x p``, one row per recording, one column per
   endpoint (:data:`FEATURES`: fluency ``Phi``, Kemeny ``K``, synchrony
   ``mean F``, FidgetyFind ``FF``).
2. **Standardise against the normal cohort.** Writing ``N`` for the normal
   recordings, ``z(x)_j = (x_j - mu_j) / s_j`` with ``mu_j`` and ``s_j`` the
   mean and sample SD of endpoint ``j`` **over the normal recordings alone**.
   The four endpoints live on incomparable scales, so the raw columns cannot be
   compared, let alone maximised over.
3. **Sign by the pathological pole.** Each construct carries a pole fixed by its
   definition, not by the label. Ordering the endpoints
   ``(Phi, K, mean F, FF)``, :data:`DIRECTIONS` is ``d = (+1, +1, +1, -1)``:
   higher ``Phi`` (more similar consecutive movements), higher ``K`` (slower
   mixing) and higher ``mean F`` (more inter-limb coupling) are abnormal, and
   lower ``FF`` (less direction variety) is abnormal. The signed deviation
   ``d_j z(x)_j`` is large when construct ``j`` points toward its abnormal pole.
4. **Index.** ``T(x) = max_j d_j z(x)_j`` -- the number of standard deviations
   by which the *most extreme* construct sits toward its abnormal pole. The
   construct attaining the maximum is recorded as the recording's **driver**, so
   the axis of deviation stays visible.
5. **Flag.** ``y(x) = 1[T(x) > tau]`` with ``Pr(Z <= tau) = 1 - alpha/p``,
   ``Z ~ Normal(0, 1)``. The ``alpha/p`` level is a **Bonferroni** correction for
   taking the maximum over ``p`` constructs: each recording gets ``p`` chances to
   clear the cut. With ``alpha = 0.05`` and ``p = 4`` this gives
   ``tau ~ 2.24``.

**What the numbers mean, and what they do not.** ``mu_j`` and ``s_j`` are
estimated from the normal recordings and the index is then applied to them, so
the **specificity is in-sample** and is a description of the fit, not an
estimate of out-of-sample specificity. :func:`abnormality_index` therefore also
reports a leave-one-out specificity, where each normal recording is scored
against a reference computed from the other ``n0 - 1``. The sensitivity needs no
such correction: no abnormal recording enters the reference, so it is already
out-of-sample.

Two limits govern the reading, and both are reported. First, the maximum buys
sensitivity to a single-axis deviation at the price of false positives -- each
recording has ``p`` chances to clear the cut, which the ``alpha/p`` level holds
in check but does not remove. Second, an abnormal recording that lies inside the
normal range on *every* construct cannot be reached by any threshold on this
statistic; :func:`abnormality_index` counts those separately as the **ceiling**,
so a missed recording is not read as a threshold artefact.

Nothing here is fitted against the label: the standardisation sees only which
recordings are normal, the signs come from the constructs' definitions, and the
threshold is a normal quantile. The index is contrasted by the same exact
Mann-Whitney permutation null as the four endpoints it is built from, and the
2x2 table carries Fisher's exact p.
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

    ``direction`` is the pathological pole: ``+1`` when a larger value is the
    abnormal one, ``-1`` when a smaller one is. It is a property of the
    construct's definition, never of the label, and every construct entering the
    index must state one -- a maximum over signed deviations has no meaning for
    a column whose abnormal direction is undeclared.

    ``period`` is the assessment period the construct reads: ``"writhing"``
    (term to about two months; poor repertoire and cramped-synchronised
    movement) or ``"fidgety"`` (about two to five months; absent fidgety
    movement).
    """

    key: str
    label: str
    direction: int
    period: str
    block: str          # results key holding it ("" = the primary model)
    path: tuple         # keys to walk from that block to the vector


FEATURES: tuple[Feature, ...] = (
    Feature("phi", "fluency Phi", +1, "writhing", "", ("phi", "excess")),
    Feature("kemeny", "Kemeny (jumps)", +1, "writhing", "",
            ("kemeny_per_subject",)),
    Feature("mean_F", "synchrony (mean F)", +1, "writhing", "wclrpp",
            ("mean_F",)),
    Feature("FF", "FidgetyFind FF", -1, "fidgety", "fidgetyfind", ("FF",)),
)

#: The pathological pole of each endpoint, as used to sign its deviation.
#: ``Phi``: more similar consecutive movements is the poor-repertoire pole.
#: ``Kemeny``: slower mixing is the poor-repertoire pole.
#: ``mean F``: high inter-limb coupling is the cramped-synchronised pole.
#: ``FF``: higher is normal, so the abnormal recordings are expected below.
DIRECTIONS: dict[str, int] = {f.key: f.direction for f in FEATURES}

#: Which assessment period each endpoint reads.
PERIODS: dict[str, str] = {f.key: f.period for f in FEATURES}

#: Default family-wise level the threshold is set at, before the Bonferroni
#: division by the number of constructs the maximum ranges over.
ALPHA: float = 0.05


def threshold(alpha: float = ALPHA, n_constructs: int = 4) -> float:
    """``tau`` with ``Pr(Z <= tau) = 1 - alpha / n_constructs``, ``Z`` standard normal.

    The Bonferroni correction for taking a maximum over ``n_constructs``
    deviations: under the null each is standard normal, so the chance that *any*
    of them clears ``tau`` is at most ``alpha``. ``alpha = 0.05`` over four
    constructs gives ``tau = 2.2414``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if n_constructs < 1:
        raise ValueError(f"there must be at least one construct, got "
                         f"{n_constructs}")
    return float(stats.norm.ppf(1.0 - alpha / n_constructs))


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
            "periods": [f.period for f in kept],
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
    """Read back what :func:`save_feature_matrix` wrote.

    The poles are refreshed from :data:`DIRECTIONS` for every key the current
    :data:`FEATURES` knows, so a matrix cached by an earlier version scores
    against today's declared poles rather than against whatever was written into
    the file. The matrix itself is data and is read back unchanged.
    """
    with np.load(path, allow_pickle=True) as z:
        fm = {"X": z["X"].astype(float),
              "labels": z["labels"].astype(int),
              "videos": [str(v) for v in z["videos"]],
              "names": [str(v) for v in z["names"]],
              "keys": [str(v) for v in z["keys"]],
              "directions": z["directions"].astype(float)}
    fm["directions"] = np.array(
        [DIRECTIONS.get(k, d) for k, d in zip(fm["keys"], fm["directions"])],
        float)
    fm["periods"] = [PERIODS.get(k, "") for k in fm["keys"]]
    fm["n"] = len(fm["labels"])
    fm["missing"] = []
    fm["features"] = tuple(f for f in FEATURES if f.key in fm["keys"])
    return fm


# ---------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------
def _standardise(X, ref, missing="impute"):
    """Column z-scores against the reference rows; how NaNs are handled is declared.

    ``ref`` is the boolean mask of the recordings that define the reference --
    the normal cohort. ``mu_j`` and ``s_j`` are that subset's mean and *sample*
    SD (``ddof = 1``, the reference being a sample rather than the population),
    so a reference recording's z-scores have mean 0 and SD 1 by construction and
    an abnormal recording is read in units of the normal spread.

    ``impute`` keeps every recording and sets a missing entry to ``z = 0``, the
    reference mean: the value that adds no information and no leverage, which is
    the honest stand-in when one endpoint declined to score a recording the
    others did score. Under the maximum it also means a declined construct never
    drives the flag. ``drop`` removes the row from the reference and from the
    readout entirely, leaving its index NaN.
    """
    X = np.asarray(X, float)
    ref = np.asarray(ref, bool)
    ok_row = np.isfinite(X).all(1)
    use = ref if missing == "impute" else (ref & ok_row)
    fit = X[use]
    if len(fit) < 2:
        raise ValueError(f"the reference needs at least two normal recordings, "
                         f"got {len(fit)}")
    mean = np.array([np.nanmean(c) if np.isfinite(c).any() else np.nan
                     for c in fit.T])
    sd = np.array([np.nanstd(c, ddof=1) if np.isfinite(c).sum() > 1 else np.nan
                   for c in fit.T])
    bad = [j for j, s in enumerate(sd) if not np.isfinite(s) or s <= 0]
    if bad:
        raise ValueError(f"endpoint column(s) {bad} have no usable spread over "
                         f"the normal recordings (all missing, or constant), "
                         f"so they cannot be standardised")
    imputed = ~np.isfinite(X)
    Z = np.where(imputed, 0.0, (X - mean) / sd)
    scored = np.ones(len(X), bool) if missing == "impute" else ok_row
    return Z, mean, sd, scored, imputed


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


def _driver_counts(driver, keys, names, periods, mask):
    """Which construct attains the maximum, over the recordings in ``mask``."""
    by_construct, by_period = {}, {}
    for j, k in enumerate(keys):
        n = int(np.sum(mask & (driver == j)))
        by_construct[k] = {"name": names[j], "period": periods[j], "n": n}
        pd_ = periods[j] or "unstated"
        by_period[pd_] = by_period.get(pd_, 0) + n
    return {"by_construct": by_construct, "by_period": by_period}


def abnormality_index(X, labels, directions=None, names=None, keys=None,
                      periods=None, alpha: float = ALPHA,
                      tau: float | None = None, missing: str = "impute",
                      boot: int = 10_000) -> dict:
    """``T(x) = max_j d_j z(x)_j``, cut at the Bonferroni normal quantile.

    ``X`` is ``N x p`` (rows = recordings, columns = endpoints), ``labels`` is
    ``1`` for abnormal and ``0`` for normal. ``directions`` gives each column's
    pathological pole (``+1`` when large is abnormal, ``-1`` when small is);
    every column must state one, since a maximum over signed deviations has no
    meaning for an undeclared direction.

    The columns are standardised against the **normal recordings alone**, signed
    by ``directions``, and reduced by the maximum. A recording is flagged when
    that maximum clears ``tau``, which defaults to the ``1 - alpha/p`` normal
    quantile -- the Bonferroni correction for the ``p`` chances the maximum
    gives each recording. Pass ``tau`` to set the cut directly instead.

    ``missing`` is ``"impute"`` or ``"drop"``; see :func:`_standardise`.

    Returns the index, the threshold, the flag, the driving construct per
    recording and the readout. The AUC of the continuous index is the same exact
    Mann-Whitney contrast every other endpoint gets; the 2x2 table of the binary
    flag carries Fisher's exact p.
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(labels, int)
    n, p = X.shape
    if n != len(y):
        raise ValueError(f"X has {n} rows but there are {len(y)} labels")
    if p < 2:
        raise ValueError(f"the index needs at least two endpoints, got {p}")
    if missing not in ("impute", "drop"):
        raise ValueError(f"missing must be 'impute' or 'drop', got {missing!r}")
    if directions is None:
        raise ValueError("every endpoint must state its pathological pole; "
                         "pass directions of +1 (large is abnormal) or -1 "
                         "(small is abnormal), one per column")
    d = np.asarray(directions, float).ravel()
    if d.size != p:
        raise ValueError(f"directions has {d.size} entries but X has {p} "
                         f"columns")
    if not np.all(np.isin(d, (-1.0, 1.0))):
        raise ValueError(f"every direction must be +1 or -1; got "
                         f"{d.tolist()}. A construct with no stated "
                         f"pathological pole cannot enter a maximum over "
                         f"signed deviations")
    names = list(names) if names is not None else [f"x{j}" for j in range(p)]
    keys = list(keys) if keys is not None else list(names)
    periods = list(periods) if periods is not None else [""] * p

    normal = (y == 0)
    Z, mean, sd, scored, imputed = _standardise(X, normal, missing)

    # the signed deviation of every construct, and the largest of them
    signed = Z * d
    T = np.where(scored, signed.max(1), np.nan)
    driver = np.where(scored, signed.argmax(1), -1)

    tau = threshold(alpha, p) if tau is None else float(tau)
    flag = np.where(scored, T > tau, np.nan)
    fin = scored & np.isfinite(flag)
    readout = _confusion(flag[fin].astype(int), y[fin])

    ref = normal & scored
    n0 = int(ref.sum())

    # Leave-one-out: each normal recording standardised against the other
    # n0 - 1, which is what its specificity would be if it had not helped set
    # mu and s. The abnormal recordings never enter the reference, so their
    # sensitivity is already out-of-sample and needs no such correction.
    loo = {}
    if n0 >= 3:
        idx = np.flatnonzero(ref)
        T_loo = np.empty(len(idx))
        for i, k in enumerate(idx):
            rest = np.zeros(n, bool)
            rest[np.setdiff1d(idx, [k])] = True
            Zk, *_ = _standardise(X, rest, missing)
            T_loo[i] = float((Zk[k] * d).max())
        f_loo = T_loo > tau
        loo = {"n": int(len(idx)), "n_flagged": int(f_loo.sum()),
               "specificity": float(1.0 - f_loo.mean()),
               "index": T_loo,
               "note": ("each normal recording standardised against the other "
                        "normal recordings; the sensitivity needs no such "
                        "correction, as no abnormal recording enters the "
                        "reference")}

    # The ceiling: an abnormal recording whose index does not clear the largest
    # index among the normal recordings cannot be separated by ANY threshold on
    # this statistic without flagging every normal one. Counting these keeps a
    # missed recording from being read as an artefact of where tau was put.
    ab = (y == 1) & scored
    T_ref_max = float(np.max(T[ref])) if n0 else float("nan")
    reachable = ab & (T > T_ref_max)
    inside = ab & ~reachable
    ceiling = {
        "n_abnormal": int(ab.sum()),
        "max_normal_index": T_ref_max,
        "n_reachable": int(reachable.sum()),
        "n_inside_normal_range": int(inside.sum()),
        "max_sensitivity_at_full_specificity":
            float(reachable.sum() / ab.sum()) if ab.sum() else float("nan"),
        "note": ("an abnormal recording whose index sits at or below the "
                 "largest index among the normal recordings lies inside the "
                 "normal range on every construct; no threshold on this "
                 "statistic reaches it, so it is a ceiling rather than a "
                 "threshold artefact"),
    }

    # The price of the maximum, made checkable rather than asserted: the same
    # statistic cut at a plain 2 SD, which is what tau would be without the
    # Bonferroni division. It flags a superset, and how much larger a superset
    # is the size of the correction on this cohort.
    plain = 2.0
    plain_flag = scored & (T > plain)
    plain_cut = {"tau": plain,
                 "n_flagged": int(plain_flag.sum()),
                 "n_normal_flagged": int((plain_flag & ref).sum()),
                 "n_abnormal_flagged": int((plain_flag & ab).sum()),
                 "note": ("the same index cut at a plain 2 SD instead of the "
                          "Bonferroni tau; it flags a superset, and the "
                          "difference is what the alpha/p level holds in "
                          "check")}

    drivers = {
        "all": _driver_counts(driver, keys, names, periods, scored),
        "flagged": _driver_counts(driver, keys, names, periods,
                                  fin & (flag == 1)),
        "flagged_abnormal": _driver_counts(driver, keys, names, periods,
                                           fin & (flag == 1) & (y == 1)),
        "flagged_normal": _driver_counts(driver, keys, names, periods,
                                         fin & (flag == 1) & (y == 0)),
    }

    group = ST.mannwhitney(T[ab], T[ref], boot=boot)

    return {
        "index": T, "T": T, "signed": signed, "flag": flag, "scored": scored,
        "driver": driver,
        "driver_key": [keys[j] if j >= 0 else "" for j in driver],
        "driver_name": [names[j] if j >= 0 else "" for j in driver],
        "driver_period": [periods[j] if j >= 0 else "" for j in driver],
        "z": Z, "mean": mean, "sd": sd, "imputed": imputed,
        "n_imputed": int(imputed.sum()), "n_scored": int(scored.sum()),
        "n": n, "names": names, "keys": keys, "periods": periods,
        "directions": d.tolist(),
        "threshold": {"tau": tau, "alpha": float(alpha),
                      "n_constructs": int(p),
                      "per_construct_alpha": float(alpha) / p,
                      "method": ("Bonferroni over the constructs the maximum "
                                 "ranges over: Pr(Z <= tau) = 1 - alpha/p")},
        "reference": {"n_normal": n0, "mean": mean.tolist(), "sd": sd.tolist(),
                      "ddof": 1,
                      "note": ("mu and s are the normal recordings' mean and "
                               "sample SD, so the index is in units of the "
                               "normal spread")},
        "readout": readout, "loo": loo, "ceiling": ceiling, "drivers": drivers,
        "plain_cut": plain_cut,
        "group": group, "missing_policy": missing,
        "note": ("the largest signed deviation across the standardised "
                 f"endpoints, flagged above tau = {tau:.4g} "
                 f"(Bonferroni, alpha = {alpha:g} over {p} constructs); "
                 "no parameter is fitted against the label"),
    }


def index_frame(fm: dict, ix: dict):
    """The per-recording table: raw endpoints, signed z-scores, index and flag."""
    import pandas as pd
    X, Z = np.asarray(fm["X"], float), np.asarray(ix["z"], float)
    S = np.asarray(ix["signed"], float)
    rows = {"subject": np.arange(1, fm["n"] + 1), "video": fm["videos"],
            "label": np.asarray(fm["labels"], int)}
    for j, k in enumerate(fm["keys"]):
        rows[k] = X[:, j]
    for j, k in enumerate(fm["keys"]):
        rows[f"z_{k}"] = np.where(ix["imputed"][:, j], np.nan, Z[:, j])
    for j, k in enumerate(fm["keys"]):
        rows[f"signed_z_{k}"] = np.where(ix["imputed"][:, j], np.nan, S[:, j])
    rows["index"] = ix["index"]
    rows["driver"] = ix["driver_key"]
    rows["driver_period"] = ix["driver_period"]
    rows["tau"] = np.full(fm["n"], ix["threshold"]["tau"])
    rows["flag"] = ix["flag"]
    rows["correct"] = np.where(np.isfinite(ix["flag"]),
                               ix["flag"] == np.asarray(fm["labels"], float),
                               np.nan)
    return pd.DataFrame(rows)


def describe(fm: dict, ix: dict) -> str:
    """The readout as lines of text, for the run log and for a notebook."""
    r, th, L = ix["readout"], ix["threshold"], []
    L.append(f"  matrix: {ix['n']} recordings x {len(fm['keys'])} endpoints "
             f"({', '.join(fm['names'])})")
    if fm.get("missing"):
        L.append(f"     not available, so not in the index: "
                 f"{', '.join(fm['missing'])}")
    if ix["n_imputed"]:
        L.append(f"     {ix['n_imputed']} endpoint value(s) did not score and "
                 f"were set to the normal mean (z = 0)"
                 if ix["missing_policy"] == "impute" else
                 f"     {ix['n'] - ix['n_scored']} recording(s) dropped for a "
                 f"missing endpoint")
    L.append(f"  standardised against the {ix['reference']['n_normal']} normal "
             f"recordings (their mean and sample SD, per endpoint)")
    L.append("     pathological poles: " + ",  ".join(
        f"{nm} {'+' if dj > 0 else '-'}"
        for nm, dj in zip(fm["names"], ix["directions"]))
        + "   (+ = higher is abnormal)")
    L.append(f"  index T(x) = max_j d_j z_j, flagged above tau = "
             f"{th['tau']:.4f} "
             f"(alpha = {th['alpha']:g} / {th['n_constructs']} constructs, "
             f"Bonferroni)")
    L.append(f"  flagged {r['n_flagged']}/{r['n']} recordings: "
             f"sensitivity {r['sensitivity']:.3f} ({r['tp']}/{r['tp'] + r['fn']}"
             f"), specificity {r['specificity']:.3f} "
             f"({r['tn']}/{r['tn'] + r['fp']})")
    L.append(f"     PPV {r['ppv']:.3f}, NPV {r['npv']:.3f}, balanced accuracy "
             f"{r['balanced_accuracy']:.3f}, Youden J {r['youden_j']:+.3f}; "
             f"Fisher exact p = {r['fisher_p']:.4g}")
    drv = ix["drivers"]["flagged"]
    driving = ",  ".join(f"{v['name']} {v['n']}"
                         for v in drv["by_construct"].values() if v["n"])
    if driving:
        per = ",  ".join(f"{k} {v}" for k, v in drv["by_period"].items() if v)
        L.append(f"     the construct attaining the maximum, per flag: "
                 f"{driving}   (by period: {per})")
    if ix.get("loo"):
        lo = ix["loo"]
        L.append(f"     leave-one-out specificity {lo['specificity']:.3f} "
                 f"({lo['n'] - lo['n_flagged']}/{lo['n']}): mu and s are "
                 f"estimated on the normal cohort and then applied to it, so "
                 f"the in-sample specificity above is a description of the fit")
    pc = ix.get("plain_cut")
    if pc and th["tau"] > pc["tau"]:
        L.append(f"     the price of the maximum: a plain "
                 f"{pc['tau']:g} SD cut on the same index would flag "
                 f"{pc['n_normal_flagged']} of the "
                 f"{ix['reference']['n_normal']} normal recordings against "
                 f"{r['fp']} here, which is the size of the alpha/"
                 f"{th['n_constructs']} correction on this cohort")
    c = ix["ceiling"]
    if c["n_abnormal"]:
        L.append(f"     ceiling: {c['n_inside_normal_range']}/"
                 f"{c['n_abnormal']} abnormal recording(s) sit at or below the "
                 f"largest normal index ({c['max_normal_index']:+.3f}), so no "
                 f"threshold on this statistic reaches them; the most any cut "
                 f"could get at full specificity is {c['n_reachable']}/"
                 f"{c['n_abnormal']}")
    g = ix["group"]
    L.append(f"  the index as a continuous endpoint: AUC {g['auc']:.3f} "
             f"[{g.get('auc_lo', float('nan')):.3f}, "
             f"{g.get('auc_hi', float('nan')):.3f}], p = {g['p']:.4g} "
             f"({g['method']})")
    return "\n".join(L)
