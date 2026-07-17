"""End-to-end smoke test for the CARE-PD state-space pipeline.

Builds synthetic 3-cohort gait (cyclic joint motion + global translation)
with a planted **freezing** signal — FoG walks spend stretches in a
low-amplitude, near-stationary regime — then runs the whole pipeline
(principal movements → CV model selection → ARHMM → occupancy/dwell/
transitions → clinical LME → figures → go/no-go). Proves every path flows
and every artefact is written; the exact verdict needs the real data.

Run:  python -m carepd_statespace.smoke_test
"""

from __future__ import annotations

import tempfile

import numpy as np

from carepd_statespace.carepd_adapter import Walk, build_dataset
from carepd_statespace.driver import run_pipeline

J = 17
# Joints that swing during gait (arms + legs), amplitude reduced in freezing.
SWING = {1: 0.8, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.0, 6: 1.2,      # legs
         11: 0.6, 12: 0.9, 13: 1.1, 14: 0.6, 15: 0.9, 16: 1.1}  # arms


def _gait_walk(rng, F, fps, cadence, fog, cohort_bias):
    """Cyclic gait joints (F,17,3) + forward trans; freeze episodes if fog."""
    t = np.arange(F) / fps
    phase = 2 * np.pi * cadence * t
    joints = np.zeros((F, J, 3))
    base = rng.standard_normal((J, 3)) * 0.05 + cohort_bias      # rig offset
    amp = np.ones(F)
    speed = np.ones(F)
    if fog:
        # a couple of freezing episodes: amplitude and speed collapse
        for _ in range(rng.integers(1, 3)):
            s = rng.integers(0, max(F - F // 5, 1))
            e = min(s + rng.integers(F // 8, F // 4), F)
            amp[s:e] *= 0.15
            speed[s:e] *= 0.1
    for j in range(J):
        sw = SWING.get(j, 0.1)
        joints[:, j, 0] = base[j, 0] + amp * sw * 0.3 * np.sin(phase + j)
        joints[:, j, 1] = base[j, 1] + amp * sw * 0.1 * np.cos(2 * phase)  # vertical bob
        joints[:, j, 2] = base[j, 2] + amp * sw * 0.2 * np.cos(phase + j)
    # hips left/right so heading is defined
    joints[:, 1, 2] += 0.15
    joints[:, 4, 2] -= 0.15
    joints += rng.standard_normal((F, J, 3)) * 0.01
    trans = np.zeros((F, 3))
    trans[:, 0] = np.cumsum(speed) / fps * 1.0                   # forward path
    trans[:, 1] = 0.9                                             # height
    return joints + trans[:, None, :] * np.array([0, 0, 0]), trans


def synthetic_walks(seed=0):
    rng = np.random.default_rng(seed)
    specs = [("BMCLab", 6, 150.0, 4, (0, 3.5)),      # short walks
             ("KUL-DT-T", 6, 100.0, 4, (0, 5.0)),
             ("E-LC", 5, 120.0, 2, (0, 20.0))]       # long walks
    walks = []
    for cohort, n_subj, fps, n_walk, (lo, hi) in specs:
        for s in range(n_subj):
            fog_subj = int(rng.random() < 0.5)
            bias = rng.standard_normal(3) * 0.1
            sex = rng.choice(["M", "M", "M", "F"])   # >75% male
            for w in range(n_walk):
                dur = rng.uniform(lo, hi)
                F = max(int(dur * fps), 40)
                cad = rng.uniform(1.2, 1.8)
                joints, trans = _gait_walk(rng, F, fps, cad, fog_subj, bias)
                med = rng.choice(["on", "off"]) if cohort in ("BMCLab", "E-LC") else np.nan
                updrs = rng.integers(0, 4) if cohort == "BMCLab" else np.nan
                walks.append(Walk(joints=joints, cohort=cohort,
                                  subject_id=f"{cohort}_s{s}", walk_id=f"w{w}",
                                  fps=fps, fog=fog_subj, medication=med,
                                  updrs_gait=float(updrs) if updrs == updrs else np.nan,
                                  sex=sex, trans=trans))
    return walks


def main():
    walks = synthetic_walks()
    data = build_dataset(walks, feature_set="B", dst_fps=15.0)
    print(f"[smoke] {data.n_walks} walks, d={data.d}, "
          f"{data.info.subject_id.nunique()} subjects, "
          f"cohorts={sorted(data.info.cohort.unique())}, "
          f"fog+={int(data.info.fog.sum())}")

    with tempfile.TemporaryDirectory() as td:
        out = run_pipeline(data, out_dir=td, K_grid=(3, 5), L_grid=(1, 2),
                           n_restarts=3, n_folds=3, seed=0, verbose=True)
        import pathlib
        figs = sorted(p.name for p in (pathlib.Path(td) / "figures").glob("*.png"))
        tabs = sorted(p.name for p in (pathlib.Path(td) / "tables").glob("*"))
        print(f"\n[smoke] figures: {figs}")
        print(f"[smoke] tables: {tabs}")
        for need in ("fig1b_variance.png", "fig1e_generative.png",
                     "fig2a_sequences.png", "fig3b_regions.png"):
            assert need in figs, f"missing figure {need}"
        assert (pathlib.Path(td) / "RESULTS.md").exists()
        v = out["results"]["verdict"]
        print(f"[smoke] selected K={out['results']['K']} L={out['results']['L']}, "
              f"PMs={out['results']['n_pm']}")
        print(f"[smoke] VERDICT: {v['gate']}  ({len(v['significant'])} sig stats)")
        print(f"[smoke] cohort clf acc={out['results']['site']['cohort_clf_acc']:.2f}")

    print("\n=== CARE-PD state-space pipeline ran end-to-end ===")


if __name__ == "__main__":
    main()
