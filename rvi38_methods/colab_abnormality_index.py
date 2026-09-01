# =====================================================================
# RVI-38 — the abnormality index, in one cell
# ---------------------------------------------------------------------
# The final readout: the four pipeline endpoints (fluency Phi, the Kemeny
# mixing time, WCLR-PP synchrony, FidgetyFind) standardised, reduced to
# PC1 as a continuous *abnormality index*, and cut at two standard
# deviations of the NORMAL cohort's own distribution — outside the band
# is flagged 1, inside 0.
#
# This cell does NOT rerun the pipeline. It reads a finished run's
# results.json, caches the 38x4 matrix as feature_matrix.npz, and scores
# it in about a second — so the band width, the side and the missing-value
# policy can be changed and rescored as often as you like.
#
# Run `colab_rvi38_report.py` first (or `run_analysis.py`), then this.
# =====================================================================

# ============================ CONFIG =================================
REPO    = "/content/MasterThesis_LatentSpaceDevelopment"   # clone location
OUT_DIR = "/content/rvi38_out"      # the finished run (holds results.json)

N_SD    = 2.0        # half-width of the normative band, in SDs of the normal
                     # recordings' index. 2.0 is the reported readout.
SIDE    = "two"      # "two"   = outside the healthy range either way (reported;
                     #           the only choice that ignores PC1's arbitrary sign)
                     # "upper" = only the abnormal-pole tail
                     # "lower" = only the other tail
MISSING = "impute"   # "impute" = an endpoint that declined to score a recording
                     #            is set to its column mean (z = 0: no
                     #            information, no leverage), all 38 kept
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
if fm["missing"]:
    print(f"  not in this run, so not in the index: {', '.join(fm['missing'])}")
print(f"  labels: {int(np.sum(fm['labels']))} abnormal, "
      f"{int(np.sum(1 - fm['labels']))} normal")

# ---- 2. score it ----------------------------------------------------
ix = IX.abnormality_index(fm["X"], fm["labels"], directions=fm["directions"],
                          names=fm["names"], n_sd=N_SD, side=SIDE,
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
#   ix["pc1"]        the continuous index, one number per recording
#   ix["deviation"]  that index in SDs of the normal cohort (the cut is +-N_SD)
#   ix["flag"]       the binary readout, 1 = outside the healthy range
#   ix["loadings"],  ix["pc1_explained"]   what PC1 is, and how much it keeps
#   ix["band"]       mu0, sd0 and the interval they define
#   ix["readout"]    sensitivity, specificity, PPV, NPV, Fisher's exact p
#   ix["loo"]        leave-one-out specificity
#   ix["group"]      the index as a continuous endpoint: exact Mann-Whitney AUC
#
# Read the specificity as in-sample: the band is built from the normal
# recordings and then applied to them, which is what ix["loo"] corrects. The
# sensitivity needs no correction — no abnormal recording enters the band.
# Nothing anywhere in the index is fitted against the label.
#
# To rescore at a different band width without touching the cache:
#   ix3 = IX.abnormality_index(fm["X"], fm["labels"],
#                              directions=fm["directions"], names=fm["names"],
#                              n_sd=3.0, boot=0)
#   print(IX.describe(fm, ix3))
