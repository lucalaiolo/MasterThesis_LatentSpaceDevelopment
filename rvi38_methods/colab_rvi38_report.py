# =====================================================================
# RVI-38 — fluency, fluency curve, Kemeny mixing time, WCLR-PP synchrony
# and FidgetyFind, in one cell
# ---------------------------------------------------------------------
# Edit the CONFIG block and run. It prints every headline number, writes
# results.json / summary.md / the per-subject CSVs / every figure into
# OUT_DIR, and displays the cohort figures inline.
#
# Runtime: ~25-40 min for the full run on a Colab CPU (WCLR-PP is most of
# it). SYNCHRONY = False cuts it to a few minutes. FAST = True cuts every
# resampling count ~20x — use it to check the paths, not to report numbers.
# =====================================================================

# ---- 0. Repo + dependencies (uncomment on a fresh Colab runtime) ----
# !git clone -b claude/fidgetyfind-analysis-function-pe8tu0 \
#     https://github.com/lucalaiolo/MasterThesis_LatentSpaceDevelopment.git
# !pip -q install numpy scipy pandas joblib scikit-learn matplotlib
#
# Data on Drive? Mount it first:
# from google.colab import drive; drive.mount('/content/drive')

# ============================ CONFIG =================================
REPO      = "/content/MasterThesis_LatentSpaceDevelopment"   # clone location
CSV       = "/content/rvi38_analysis.csv"                    # long keypoint table
ARHMM     = "/content/arhmm_rvi38_stream_delta.pkl"          # primary model
HMM       = "/content/hmm_rvi38_stream_delta.pkl"            # replication model
LABELS    = "/content/RVI_38_labels.mat"                     # or None
OUT_DIR   = "/content/rvi38_out"

FAST      = False     # True = smoke run (coarse p-values, ~20x fewer draws)
SYNCHRONY = True      # WCLR-PP inter-limb coordination; the slow block
FIDGETY   = True      # FidgetyFind, the literature's detector
SHOW      = "main"    # "main" | "all" (adds the per-recording panels) | False
FPS       = 25.0
STREAM    = "auto"    # "delta" | "pose" | "auto" (infer from stored lengths)

# Optional overrides, passed straight to run_analysis (None = published default)
FLUENCY_OMEGA = None  # magnitude weight in S = w*S_mag + (1-w)*S_shape (0.5)
FF_THETA      = None  # entropy above which a window counts as fidgety (0.5)
WCLR_LIMB     = None  # "end_effector" | "distal" | "limb"
# ====================================================================

import os
import sys

sys.path.insert(0, os.path.join(REPO, "rvi38_methods"))

from report import run_report

overrides = {k: v for k, v in (("fluency_omega", FLUENCY_OMEGA),
                               ("ff_theta", FF_THETA),
                               ("wclr_limb_signal", WCLR_LIMB)) if v is not None}

out = run_report(
    csv=CSV, arhmm=ARHMM, hmm=HMM, labels=LABELS, outdir=OUT_DIR,
    fast=FAST, fps=FPS, stream=STREAM,
    synchrony=SYNCHRONY,        # WCLR-PP
    fidgetyfind=FIDGETY,        # FidgetyFind
    fluency_curve=True,         # one panel per recording
    fidgetyfind_panels=True,    # one FidgetyFind timeline per recording
    show=SHOW,
    **overrides)

# Everything is in `out`:
#   out["results"]  full results object (also OUT_DIR/results.json)
#   out["summary"]  headline numbers per construct (also OUT_DIR/summary.json)
#   out["figures"]  {"fluency": [...], "fluency_curve": [...], "kemeny": [...],
#                    "synchrony": [...], "fidgetyfind": [...], "clinical": [...]}
#   out["markdown"] the summary you just read (also OUT_DIR/summary.md)
s = out["summary"]
print("\nfluency      AUC %.3f  p %.4g" % (s["fluency"]["group"]["auc"],
                                           s["fluency"]["group"]["p"]))
print("kemeny       AUC %.3f  p %.4g" % (s["kemeny"]["group"]["auc"],
                                         s["kemeny"]["group"]["p"]))
if not s["synchrony"].get("skipped"):
    print("synchrony    AUC %.3f  p %.4g" % (s["synchrony"]["whole_body"]["auc"],
                                             s["synchrony"]["whole_body"]["p"]))
if not s["fidgetyfind"].get("skipped"):
    print("FidgetyFind  AUC %.3f  p %.4g  (below 0.5 is the expected direction)"
          % (s["fidgetyfind"]["group"]["auc"], s["fidgetyfind"]["group"]["p"]))

# To re-display one construct's figures later, without rerunning anything:
#   from report import show_figures
#   show_figures(out["figures"], "fidgetyfind")
