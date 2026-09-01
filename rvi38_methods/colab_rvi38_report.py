# =====================================================================
# RVI-38 — fluency, fluency curve, Kemeny mixing time, WCLR-PP synchrony,
# FidgetyFind and the abnormality index, in one cell
# ---------------------------------------------------------------------
# Edit the CONFIG block and run. It prints every headline number, writes
# results.json / summary.md / the per-subject CSVs / every figure into
# OUT_DIR, and displays the cohort figures inline.
#
# Runtime: ~25-40 min for the full run on a Colab CPU (WCLR-PP is most of
# it). SYNCHRONY = False cuts it to a few minutes. FAST = True cuts every
# resampling count ~20x — use it to check the paths, not to report numbers.
# (The endpoint p-values are exact enumerations either way; FAST only coarsens
# the resampled quantities, the bootstrap interval among them.)
# =====================================================================

# ---- 0. Repo + dependencies (uncomment on a fresh Colab runtime) ----
# !git clone -b claude/fidgetyfind-movement-detection-tv4zqv \
#     https://github.com/lucalaiolo/MasterThesis_LatentSpaceDevelopment.git
# !pip -q install numpy scipy pandas joblib scikit-learn matplotlib
#
# Data on Drive? Mount it first:
# from google.colab import drive; drive.mount('/content/drive')

# ============================ CONFIG =================================
REPO      = "/content/MasterThesis_LatentSpaceDevelopment"   # clone location
CSV       = "/content/rvi38_analysis.csv"                    # long keypoint table
LABELS    = "/content/RVI_38_labels.mat"                     # or None
OUT_DIR   = "/content/rvi38_out"

# Which fitted models to analyse. Any number, any mix of kinds — two AR-HMMs,
# an AR-HMM and a Gaussian HMM, one model, five. The FIRST is the primary one
# (clinical layer, correlation analysis and every figure are computed on it);
# every other is a replication, its fluency and Kemeny constant correlated
# against the primary's. Write it as a dict to name them, or as a plain list of
# paths to have them named after the files.
MODELS    = {
    "AR-HMM K=11": "/content/arhmm_rvi38_k11.pkl",
    "AR-HMM K=14": "/content/arhmm_rvi38_k14.pkl",
}
PRIMARY   = None      # e.g. "AR-HMM K=14" to make that one primary instead

FAST      = False     # True = smoke run (coarse p-values, ~20x fewer draws)
SYNCHRONY = True      # WCLR-PP inter-limb coordination; the slow block
FIDGETY   = True      # FidgetyFind, the literature's detector
INDEX     = True      # the abnormality index: PC1 of the four endpoints, cut
                      # at the normal cohort's own +/-2 SD band (the screening
                      # readout). Costs a second; needs >= 2 of the endpoints.
SHOW      = "main"    # "main" | "all" (adds the per-recording panels) | False
FPS       = 25.0
STREAM    = "auto"    # "delta" | "pose" | "auto" (infer from stored lengths)

# Optional overrides, passed straight to run_analysis (None = published default)
FLUENCY_OMEGA = None  # magnitude weight in S = w*S_mag + (1-w)*S_shape (0.5)
WCLR_LIMB     = None  # "end_effector" | "distal" | "limb"
# ====================================================================

import os
import sys

sys.path.insert(0, os.path.join(REPO, "rvi38_methods"))

from report import run_report

overrides = {k: v for k, v in (("fluency_omega", FLUENCY_OMEGA),
                               ("wclr_limb_signal", WCLR_LIMB)) if v is not None}

out = run_report(
    csv=CSV, models=MODELS, primary=PRIMARY, labels=LABELS, outdir=OUT_DIR,
    fast=FAST, fps=FPS, stream=STREAM,
    synchrony=SYNCHRONY,        # WCLR-PP
    fidgetyfind=FIDGETY,        # FidgetyFind
    abnormality_index=INDEX,    # the composite readout
    fluency_curve=True,         # one panel per recording
    fidgetyfind_panels=True,    # one FidgetyFind timeline per recording
    show=SHOW,
    **overrides)

# Everything is in `out`:
#   out["results"]  full results object (also OUT_DIR/results.json)
#   out["summary"]  headline numbers per construct (also OUT_DIR/summary.json)
#   out["figures"]  {"fluency": [...], "fluency_curve": [...], "kemeny": [...],
#                    "synchrony": [...], "fidgetyfind": [...],
#                    "abnormality_index": [...], "clinical": [...]}
#   out["markdown"] the summary you just read (also OUT_DIR/summary.md)
# Every endpoint: AUC with its stratified-bootstrap interval and the exact
# two-sided p over all C(38,6) = 2,760,681 label assignments. Uncorrected.
s = out["summary"]
def line(name, g):
    print("%-12s AUC %.3f [%.3f, %.3f]  p %.4g"
          % (name, g["auc"], g["auc_ci"][0], g["auc_ci"][1], g["p"]))

print()
line("fluency", s["fluency"]["group"])
line("kemeny", s["kemeny"]["group"])
if not s["synchrony"].get("skipped"):
    line("synchrony", s["synchrony"]["whole_body"])
if not s["fidgetyfind"].get("skipped"):
    for key, nm in (("FF", "FidgetyFind"), ("FF_hip", "  ... hips"),
                    ("FF_dist", "  ... limbs")):
        line(nm, s["fidgetyfind"]["endpoints"][key])
    print("             (below 0.5 is the expected direction for FidgetyFind)")

# The screening readout: PC1 of the four endpoints against the normal cohort's
# own band. No parameter in it is fitted against the label; the band is built
# from the normal recordings and then applied to them, so the specificity is
# in-sample (a leave-one-out one is reported beside it) while the sensitivity
# is not, since no abnormal recording enters the band.
ai = s.get("abnormality_index", {})
if not ai.get("skipped"):
    r, b = ai["readout"], ai["band"]
    print("\nabnormality index  PC1 of %d endpoints, %.0f%% of their variance"
          % (len(ai["features"]), 100 * ai["pc1_explained"]))
    print("             band [%.3f, %.3f] = %.3f +- %g x %.3f (n=%d normal)"
          % (b["lo"], b["hi"], b["mu0"], b["n_sd"], b["sd0"], b["n_normal"]))
    print("             flagged %d/%d: sensitivity %.3f (%d/%d), specificity "
          "%.3f (%d/%d), Fisher p %.4g"
          % (r["n_flagged"], r["n"], r["sensitivity"], r["tp"],
             r["tp"] + r["fn"], r["specificity"], r["tn"], r["tn"] + r["fp"],
             r["fisher_p"]))
    print("             leave-one-out specificity %.3f"
          % ai["loo_specificity"])
    line("  ... as AUC", ai["group"])

# To rescore the index at a different band width in a second, without rerunning
# anything, paste `colab_abnormality_index.py` — it reads OUT_DIR/results.json,
# caches the 38x4 endpoint matrix and scores it on its own.

# To re-display one construct's figures later, without rerunning anything:
#   from report import show_figures
#   show_figures(out["figures"], "fidgetyfind")
