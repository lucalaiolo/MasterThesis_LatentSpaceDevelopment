"""End-to-end smoke test for the post-hoc analysis ([post-hoc plan]).

Builds a synthetic multi-cohort bundle with a planted phenotype signal, a
cohort nuisance axis, and annotated FoG events on the E-LC cohort, plus two
NumPy "encoders" standing in for a trained plain VAE (leaks cohort) and
CVAE (cohort removed). Then runs the whole ``run_posthoc`` battery and
checks every expected artefact was written.

The point is to prove every code path flows and every figure / table is
produced — not to demonstrate the scientific result, which needs the real
data and real checkpoints. Run with

    python -m vae_analysis.posthoc.smoke_test

Needs numpy, scipy, scikit-learn, matplotlib. Exercises the optional
umap-learn / ruptures / hmmlearn paths when they are installed and skips
them cleanly otherwise.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from architectures.care_pd import Walk, build_bundle
from vae_analysis.posthoc import run_posthoc
from vae_analysis.posthoc.data import build_posthoc_data


# ---- Synthetic data with planted structure --------------------------------

J = 17
FPS = 30.0
D_Z = 16

# Which planted channels the FakeEncoder reads out of a clip.
PHENO_JOINT = 0   # amplitude of joint 0, x -> phenotype
COHORT_JOINT = 1  # mean of joint 1, x -> cohort (nuisance)
SUBJ_JOINT = 2    # mean of joint 2, x -> subject


def _make_walk(rng, cohort, cohort_id, subject, subject_code, phenotype,
               n_frames, fog_intervals):
    """One synthetic walk whose pose encodes phenotype / cohort / subject.

    A "freeze" bump is written into the phenotype channel during any FoG
    interval, so the outer-loop trajectory steps at the FoG boundaries and
    PELT has something to detect.
    """
    pose = rng.standard_normal((n_frames, J, 3)).astype(np.float32) * 0.05
    pose[:, PHENO_JOINT, 0] += 2.0 * phenotype               # 0 or 2
    pose[:, COHORT_JOINT, 0] += float(cohort_id)             # nuisance axis
    pose[:, SUBJ_JOINT, 0] += 0.3 * subject_code
    for (s, e) in fog_intervals:
        f0, f1 = int(s * FPS), int(e * FPS)
        pose[f0:f1, PHENO_JOINT, 0] += 3.0                   # freeze bump
    labels: dict = {}
    if cohort == "BMCLab":
        labels["updrs_gait"] = int(phenotype * 2 + rng.integers(0, 2))  # 0..3
        labels["med"] = "ON" if phenotype == 0 else "OFF"
    if cohort in ("KUL-DT-T", "E-LC"):
        labels["freezer"] = "freezer" if phenotype == 1 else "non-freezer"
    if cohort == "E-LC":
        labels["med"] = "OFF" if phenotype == 1 else "ON"
        labels["fog_intervals"] = [list(iv) for iv in fog_intervals]
    return Walk(pose=pose, cohort=cohort, subject=subject,
                walk_id=f"{subject}_w{rng.integers(0, 999)}", fps=FPS,
                labels=labels)


def synthetic_bundle(seed=0):
    rng = np.random.default_rng(seed)
    cohorts = ("BMCLab", "KUL-DT-T", "E-LC")
    walks = []
    for ci, cohort in enumerate(cohorts):
        for s in range(6):                       # 6 subjects / cohort
            subject = f"{cohort}_s{s}"
            pheno = s % 2                         # subject-level phenotype
            for _ in range(3):                    # 3 walks / subject
                n_frames = int(rng.integers(360, 560))
                fog = []
                if cohort == "E-LC" and pheno == 1:
                    # two freezing events inside the walk
                    dur_s = n_frames / FPS
                    fog = [(0.20 * dur_s, 0.30 * dur_s),
                           (0.55 * dur_s, 0.68 * dur_s)]
                walks.append(_make_walk(rng, cohort, ci, subject, s, pheno,
                                        n_frames, fog))
    return build_bundle(walks, cohorts=cohorts)


class FakeEncoder:
    """NumPy stand-in for a trained VAE / CVAE.

    Reads the planted phenotype / cohort / subject channels out of a clip.
    When ``conditioned`` (the CVAE), the cohort id ``c`` is subtracted from
    the cohort dimension, so the site probe on the CVAE latent falls to
    chance while the phenotype dimension is untouched — the invariance story
    in miniature.
    """

    def __init__(self, conditioned: bool, seed: int = 0):
        self.conditioned = conditioned
        self.rng = np.random.default_rng(seed)

    def encode_mu(self, X, M, c):
        X = np.asarray(X, dtype=np.float64)
        B = len(X)
        pheno = X[:, :, PHENO_JOINT, 0].mean(axis=1)
        coh = X[:, :, COHORT_JOINT, 0].mean(axis=1)
        subj = X[:, :, SUBJ_JOINT, 0].mean(axis=1)
        mu = np.zeros((B, D_Z), dtype=np.float64)
        mu[:, 0] = pheno
        mu[:, 1] = coh - (np.asarray(c, float) if (self.conditioned
                                                   and c is not None) else 0.0)
        mu[:, 2] = subj
        mu[:, 3] = 0.6 * pheno
        mu[:, 4:] = 0.02 * self.rng.standard_normal((B, D_Z - 4))
        return mu


def main():
    bundle = synthetic_bundle()
    encoders = {"VAE": FakeEncoder(conditioned=False, seed=1),
                "CVAE": FakeEncoder(conditioned=True, seed=2)}
    data = build_posthoc_data(encoders, bundle, clip_length=60, stride=30)
    print(f"[smoke] built PosthocData: {data.n} clips, d_z={data.d_z}, "
          f"models={data.models}, primary={data.primary}")
    print(f"[smoke] labels present: "
          f"{[k for k in data.labels if data.has_label(k)]}")
    n_fog = sum(len(m.fog_intervals) for m in data.traj_meta)
    print(f"[smoke] {n_fog} planted FoG intervals across E-LC walks")

    with tempfile.TemporaryDirectory() as td:
        out = run_posthoc(data=data, out_dir=td, n_seeds=6, seed=0,
                          verbose=True)
        written = {Path(p).name for p in out["written"]}
        print(f"\n[smoke] wrote {len(written)} files")

        expected = {
            "bic_vs_k.png", "stability_ari_distributions.png",
            "consensus_matrix_CVAE.png", "consensus_matrix_VAE.png",
            "pca_panels_CVAE.png", "pca_panels_VAE.png",
            "subject_composition_heatmap_CVAE.png",
            "cluster_labels_CVAE.csv", "cluster_labels_VAE.csv",
            "probes.png", "results.json", "summary.md",
        }
        missing = expected - written
        assert not missing, f"missing expected outputs: {missing}"

        # Optional-dependency artefacts: assert only if the dep is present.
        try:
            import umap  # noqa: F401
            assert "umap_panels_CVAE.png" in written, "UMAP panel missing"
            print("[smoke] UMAP panels present")
        except ImportError:
            print("[smoke] umap-learn absent — UMAP panels skipped (ok)")
        try:
            import hmmlearn  # noqa: F401
            assert out["results"]["hmm"] is not None, "HMM result missing"
            assert "hmm_transition_CVAE.png" in written
            print(f"[smoke] HMM fit: K={out['results']['hmm']['k']} states")
        except ImportError:
            print("[smoke] hmmlearn absent — HMM skipped (ok)")

        pelt = out["results"].get("pelt")
        if pelt is not None:
            print(f"[smoke] PELT vs FoG: penalty={pelt['penalty']}, "
                  f"recall={pelt['recall']}")
            assert "pelt_precision_recall.png" in written
        else:
            print("[smoke] PELT-FoG returned None (check FoG parsing)")

        # The invariance sanity check: CVAE site probe should sit below VAE.
        pv = out["results"]["probes"]
        print(f"[smoke] site probe  VAE={pv['VAE']['site_acc']:.2f}  "
              f"CVAE={pv['CVAE']['site_acc']:.2f}  "
              f"(chance≈{pv['CVAE']['site_chance']:.2f})")

        # Read summary back so we know it rendered.
        summary = (Path(td) / "summary.md").read_text()
        assert "post-hoc structure analysis" in summary

    print("\n=== every post-hoc analysis path ran ===")


if __name__ == "__main__":
    main()
