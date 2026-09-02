# =====================================================================
# RVI-38 — the abnormality index, in one cell
# ---------------------------------------------------------------------
# The final readout. The four pipeline endpoints (fluency Phi, the Kemeny
# mixing time, WCLR-PP synchrony, FidgetyFind) are standardised against
# the NORMAL recordings alone, signed by each construct's pathological
# pole, and reduced to the largest of them:
#
#     T(x) = max_j d_j z(x)_j,      d = (+1, +1, +1, -1)
#
# A recording is flagged when T(x) clears tau, the 1 - alpha/4 normal
# quantile — a Bonferroni correction for the four chances the maximum
# gives each recording. With alpha = 0.05 that is tau = 2.241.
#
# The maximum, and not a mean or a component, because the assessment
# needs only ONE abnormal pattern: a clinician calls a recording abnormal
# on a poor repertoire, OR on cramped-synchronised movement, OR on absent
# fidgety movement, and the other patterns need not appear. The score has
# to fire when any single construct signals abnormality.
#
# This cell does NOT rerun the pipeline. It reads a finished run's
# results.json, caches the 38x4 matrix as feature_matrix.npz, and scores
# it in about a second — so alpha, tau and the missing-value policy can
# be changed and rescored as often as you like.
#
# Run `colab_rvi38_report.py` first (or `run_analysis.py`), then this.
# =====================================================================

# ============================ CONFIG =================================
REPO    = "/content/MasterThesis_LatentSpaceDevelopment"   # clone location
OUT_DIR = "/content/rvi38_out"      # the finished run (holds results.json)

ALPHA   = 0.05       # family-wise level behind the threshold. The cut is the
                     # 1 - ALPHA/p normal quantile, p being the number of
                     # constructs the maximum ranges over. 0.05 is reported.
TAU     = None       # set the cut directly, in SDs of the normal cohort,
                     # overriding ALPHA. None = derive it from ALPHA.
MISSING = "impute"   # "impute" = an endpoint that declined to score a recording
                     #            is set to the normal mean (z = 0: no
                     #            information, no leverage, and never the
                     #            construct that drives the flag), all 38 kept
                     # "drop"   = that recording is left unscored
BOOT    = 10_000     # bootstrap resamples behind the AUC interval (0 = skip)
SHOW    = True       # draw the figure inline
# ====================================================================

import os
import sys

sys.path.insert(0, os.path.join(REPO, "rvi38_methods"))

import numpy as np
import a11_index as IX

# ---- 1. the 38x4 matrix, cached so nothing upstream has to run again ----
CACHE = os.path.join(OUT_DIR, "feature_matrix.npz")
if os.path.exists(CACHE):
    fm = IX.load_feature_matrix(CACHE)
    print(f"matrix from the cache {CACHE}")
else:
    fm = IX.feature_matrix(OUT_DIR)      # reads OUT_DIR/results.json
    IX.save_feature_matrix(fm, CACHE)
    print(f"matrix built from {os.path.join(OUT_DIR, 'results.json')} "
          f"and cached at {CACHE}")

print(f"  {fm['X'].shape[0]} recordings x {fm['X'].shape[1]} endpoints: "
      f"{', '.join(fm['names'])}")
print("  pathological poles: "
      + ", ".join(f"{n} {'+' if d > 0 else '-'}"
                  for n, d in zip(fm["names"], fm["directions"]))
      + "   (+ = higher is abnormal)")
if fm["missing"]:
    print(f"  not in this run, so not in the index: {', '.join(fm['missing'])}")
print(f"  labels: {int(np.sum(fm['labels']))} abnormal, "
      f"{int(np.sum(1 - fm['labels']))} normal")

# ---- 2. score it ----------------------------------------------------
ix = IX.abnormality_index(fm["X"], fm["labels"], directions=fm["directions"],
                          names=fm["names"], keys=fm["keys"],
                          periods=fm["periods"], alpha=ALPHA, tau=TAU,
                          missing=MISSING, boot=BOOT)
print()
print(IX.describe(fm, ix))

# ---- 3. the per-recording table -------------------------------------
df = IX.index_frame(fm, ix)
csv = os.path.join(OUT_DIR, "abnormality_index.csv")
df.to_csv(csv, index=False)
print(f"\nwrote {csv}")
try:                                    # a notebook renders this properly
    from IPython.display import display
    display(df)
except ImportError:
    print(df.to_string(index=False))

# ---- 4. the figure --------------------------------------------------
if SHOW:
    import figures
    png = figures.fig_abnormality_index(
        None, {"labels": fm["labels"],
               "abnormality_index": dict(ix, feature_names=fm["names"])},
        os.path.join(OUT_DIR, "figures"))
    print(f"wrote {png}")
    try:
        from IPython.display import Image, display
        display(Image(filename=png, width=900))
    except ImportError:
        pass

# ---- what to quote --------------------------------------------------
# `ix` carries everything the write-up needs:
#   ix["index"]         the continuous index T(x), one number per recording
#   ix["signed"]        the four signed deviations d_j z_j it maximises over
#   ix["driver_key"]    which construct attained that maximum, per recording
#   ix["driver_period"] the assessment period that construct reads
#   ix["flag"]          the binary readout, 1 = T(x) > tau
#   ix["threshold"]     tau, alpha, and the Bonferroni divisor
#   ix["reference"]     the normal cohort's mu and s, per endpoint
#   ix["readout"]       sensitivity, specificity, PPV, NPV, Fisher's exact p
#   ix["drivers"]       the driving construct counted over the flags, and the
#                       same counts grouped by assessment period
#   ix["ceiling"]       how many abnormal recordings NO threshold can reach
#   ix["plain_cut"]     the same index at a plain 2 SD, so the price of the
#                       Bonferroni correction is a number, not an assertion
#   ix["loo"]           leave-one-out specificity
#   ix["group"]         the index as a continuous endpoint: exact Mann-Whitney
#
# Two limits govern the reading, and both are in `ix`:
#   * the maximum buys sensitivity to a single-axis deviation at the price of
#     false positives — each recording gets four chances to clear the cut, and
#     the alpha/4 level holds that in check without removing it.
#     ix["plain_cut"] says how many normal recordings a plain 2 SD cut on the
#     same index would flag, which is that correction's size on this cohort.
#   * an abnormal recording inside the normal range on every construct cannot
#     be reached by ANY threshold on this statistic. ix["ceiling"] counts them,
#     so a miss is not read as an artefact of where tau was put.
#
# Read the specificity as in-sample: mu and s are estimated on the normal
# recordings and the index is then applied to them, which is what ix["loo"]
# corrects. The sensitivity needs no correction — no abnormal recording enters
# the reference. Nothing anywhere in the index is fitted against the label.
#
# To rescore at a different cut without touching the cache:
#   ix3 = IX.abnormality_index(fm["X"], fm["labels"],
#                              directions=fm["directions"], names=fm["names"],
#                              keys=fm["keys"], periods=fm["periods"],
#                              tau=2.0, boot=0)
#   print(IX.describe(fm, ix3))
