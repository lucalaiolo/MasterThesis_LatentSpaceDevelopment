"""Tests for the CARE-PD h36m loader ([guideline §1, §2.1]).

Exercises :func:`carepd_statespace.carepd_adapter.load_h36m_cohorts` against
a synthetic mini h36m release that mirrors the real on-disk layout — 3D
joints in a per-cohort ``.npz`` and clinical labels in a *separate* source
``.pkl`` — the split that trips up the naive "joints live in the pkl"
assumption. NumPy-only (plus the sibling ``architectures.care_pd`` reader);
the ``build_dataset`` leg is skipped when SciPy/Pandas are unavailable.

Run with ``python -m carepd_statespace.test_loader``.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np

from carepd_statespace.carepd_adapter import (
    load_h36m_cohorts, load_cohort_pkls, TIER1_COHORTS,
)

WORLD_NAME = "h36m_3d_world_floorXZZplus_30f_or_longer.npz"
DECOY_NAME = "h36m_3d_world2cam2img_backright_30f_or_longer.npz"


def _write_release(root: Path, source: Path):
    """Write a mini CARE-PD h36m release + source pkls under two dirs.

    ``root/<cohort>/h36m_3d_world_*.npz`` holds flat ``{subj__walk: (F,17,3)}``
    joints (BMCLab/KUL as multi-key ``savez``, E-LC as a single object array),
    each cohort dir also gets a 2D ``world2cam2img`` decoy. ``source/<cohort>.pkl``
    holds the nested ``{subj: {walk: {SMPL + labels}}}`` dict.
    """
    rng = np.random.default_rng(0)
    # cohort -> (n_subj, n_walk, flavour, has_fog, has_med, has_updrs)
    spec = {
        "BMCLab":   (2, 2, "multikey", False, True,  True),
        "KUL-DT-T": (2, 1, "multikey", True,  False, False),
        "E-LC":     (1, 2, "objarray", True,  True,  False),
    }
    counts = {}
    for cohort, (nsub, nwalk, flavour, has_fog, has_med, has_updrs) in spec.items():
        cdir = root / cohort
        cdir.mkdir(parents=True)
        joints_map, blob = {}, {}
        for s in range(nsub):
            subj = f"{cohort}_S{s}"
            fog_subj = int(rng.random() < 0.5) if has_fog else None
            sex = str(rng.choice(["M", "F"]))
            walks = {}
            for w in range(nwalk):
                F = int(rng.integers(50, 90))
                joints = rng.standard_normal((F, 17, 3)).astype(np.float32)
                joints[:, 0, 2] += 0.01 * np.arange(F)     # pelvis +Z travel
                joints[:, 1, 0] += 0.12                     # R hip
                joints[:, 4, 0] -= 0.12                     # L hip
                joints_map[f"{subj}__walk{w}"] = joints
                rec = {"pose": rng.standard_normal((F, 72)).astype(np.float32),
                       "betas": np.zeros(10, np.float32),
                       "trans": rng.standard_normal((F, 3)).astype(np.float32),
                       "gender": sex}
                if has_updrs:
                    rec["UPDRS_GAIT"] = int(rng.integers(0, 4))
                if has_med:
                    rec["medication"] = "ON" if w == 0 else "OFF"
                if has_fog:
                    rec["other"] = {"freezer": fog_subj}
                walks[f"walk{w}"] = rec
            blob[subj] = walks
        if flavour == "multikey":
            np.savez(cdir / WORLD_NAME, **joints_map)
        else:
            np.savez(cdir / WORLD_NAME, data=np.array(joints_map, dtype=object))
        # 2D decoy that the world glob must skip.
        np.savez(cdir / DECOY_NAME,
                 **{k: np.zeros((5, 17, 2), np.float32)
                    for k in list(joints_map)[:1]})
        with open(source / f"{cohort}.pkl", "wb") as f:
            pickle.dump(blob, f)
        counts[cohort] = len(joints_map)
    return counts


def test_load_h36m_joins_npz_and_pkl():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "CARE-PD_h36m"
        source = Path(d) / "TWIKMK"
        source.mkdir(parents=True)
        counts = _write_release(root, source)

        walks = load_h36m_cohorts(root, source_dir=source)
        assert len(walks) == sum(counts.values()), (len(walks), counts)

        by = {c: [w for w in walks if w.cohort == c] for c in TIER1_COHORTS}
        # Joints from the .npz: (F, 17, 3); trans dropped; fps standardised.
        for w in walks:
            assert w.joints.ndim == 3 and w.joints.shape[1:] == (17, 3), w.joints.shape
            assert w.joints.shape[-1] == 3, "2D decoy must not leak in"
            assert w.trans is None, "trans must be None on h36m walks"
            assert w.fps == 30.0
            # subject__walk key was split correctly.
            assert w.subject_id.startswith(w.cohort + "_S"), w.subject_id
            assert w.walk_id.startswith("walk"), w.walk_id

        # Labels joined from the source .pkl, per-cohort coverage.
        assert all(np.isfinite(w.updrs_gait) for w in by["BMCLab"])
        assert all(w.medication in ("on", "off") for w in by["BMCLab"])
        assert all(w.fog in (0, 1) for w in by["KUL-DT-T"])         # nested "other"
        assert all(not np.isfinite(w.updrs_gait) for w in by["KUL-DT-T"])
        assert all(w.fog in (0, 1) for w in by["E-LC"])
        assert all(w.medication in ("on", "off") for w in by["E-LC"])
        assert all(w.sex in ("M", "F") for w in walks)              # from "gender"
    print("ok  load_h36m_cohorts (npz joints + pkl labels joined by walk id)")


def test_load_h36m_without_labels():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "CARE-PD_h36m"
        source = Path(d) / "TWIKMK"
        source.mkdir(parents=True)
        counts = _write_release(root, source)
        # No source_dir -> joints load, labels are all missing.
        walks = load_h36m_cohorts(root)
        assert len(walks) == sum(counts.values())
        assert all(w.trans is None for w in walks)
        assert all((isinstance(w.fog, float) and np.isnan(w.fog)) for w in walks)
        assert all((isinstance(w.sex, float) and np.isnan(w.sex)) for w in walks)
    print("ok  load_h36m_cohorts (joints without labels when source_dir omitted)")


def test_explicit_paths_and_subset():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "CARE-PD_h36m"
        source = Path(d) / "TWIKMK"
        source.mkdir(parents=True)
        _write_release(root, source)
        walks = load_h36m_cohorts(
            cohorts=("BMCLab",),
            npz_paths={"BMCLab": root / "BMCLab" / WORLD_NAME},
            pkl_paths={"BMCLab": source / "BMCLab.pkl"})
        assert walks and all(w.cohort == "BMCLab" for w in walks)
        assert all(np.isfinite(w.updrs_gait) for w in walks)
    print("ok  load_h36m_cohorts (explicit npz_paths / pkl_paths, cohort subset)")


def test_missing_cohort_dir_errors():
    with tempfile.TemporaryDirectory() as d:
        try:
            load_h36m_cohorts(Path(d), cohorts=("BMCLab",))
        except FileNotFoundError as e:
            assert "BMCLab" in str(e)
            print("ok  load_h36m_cohorts (clear FileNotFoundError on missing cohort)")
            return
    raise AssertionError("expected FileNotFoundError for a missing cohort dir")


def test_old_pkl_loader_points_to_h36m():
    """A source-style pkl (labels, no joints) must fail loudly, not silently."""
    with tempfile.TemporaryDirectory() as d:
        source = Path(d) / "TWIKMK"
        source.mkdir(parents=True)
        _write_release(Path(d) / "CARE-PD_h36m", source)
        try:
            load_cohort_pkls({"BMCLab": str(source / "BMCLab.pkl")},
                             joints_key="joints3d")
        except ValueError as e:
            assert "load_h36m_cohorts" in str(e), str(e)
            print("ok  load_cohort_pkls (steers to load_h36m_cohorts on a "
                  "joint-less pkl)")
            return
    raise AssertionError("expected ValueError steering to load_h36m_cohorts")


def test_build_dataset_from_loaded_walks():
    try:
        import scipy  # noqa: F401
        import pandas  # noqa: F401
    except Exception:
        print("skip build_dataset (SciPy/Pandas unavailable)")
        return
    from carepd_statespace.carepd_adapter import build_dataset
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "CARE-PD_h36m"
        source = Path(d) / "TWIKMK"
        source.mkdir(parents=True)
        _write_release(root, source)
        walks = load_h36m_cohorts(root, source_dir=source)
        data = build_dataset(walks, feature_set="B", dst_fps=15.0)
        assert data.d == 17 * 3 + 4, data.d           # Set B = 55
        for col in ("recording", "subject_id", "cohort", "fog", "medication",
                    "updrs_gait", "sex", "n_frames"):
            assert col in data.info.columns, col
        assert data.n_walks == len(walks)
        assert build_dataset(walks, feature_set="A", dst_fps=15.0).d == 17 * 3
    print("ok  build_dataset over loaded walks (Set B d=55, Set A d=51)")


def main():
    test_load_h36m_joins_npz_and_pkl()
    test_load_h36m_without_labels()
    test_explicit_paths_and_subset()
    test_missing_cohort_dir_errors()
    test_old_pkl_loader_points_to_h36m()
    test_build_dataset_from_loaded_walks()
    print("\n=== all CARE-PD h36m loader tests passed ===")


if __name__ == "__main__":
    main()
