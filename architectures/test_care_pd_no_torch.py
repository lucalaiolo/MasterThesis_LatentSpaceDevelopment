"""NumPy-only tests for the CARE-PD adapter preprocessing ([CARE-PD §8]).

Runs without PyTorch, mirroring ``test_no_torch.py``. Exercises the parts
of ``care_pd`` that must be numerically right — root-centring,
direction alignment, fps resampling, windowing — plus the bundle and
split helpers. Run with ``python -m architectures.test_care_pd_no_torch``.
"""

import tempfile
from pathlib import Path

import numpy as np

from architectures.care_pd import (
    root_center, align_direction, resample_fps, make_windows,
    preprocess_walk, cohort_index, tier_cohorts,
    Walk, build_bundle, subset,
    leave_one_subject_out, leave_one_cohort_out,
    load_cohort, load_cohorts, load_labels_pkl,
    H36M_N_JOINTS,
    TIER1_COHORTS, TIER2_COHORTS,
)


def test_root_center():
    rng = np.random.default_rng(0)
    pose = rng.standard_normal((10, 22, 3))
    out = root_center(pose, root=0)
    assert np.allclose(out[:, 0, :], 0.0), "root joint must sit at the origin"
    # Relative geometry is preserved (differences unchanged).
    assert np.allclose(out[:, 5] - out[:, 3], pose[:, 5] - pose[:, 3])
    print("ok  root_center")


def test_align_direction_sends_travel_to_plus_x():
    # A walk translating along the diagonal of the x-z plane (y up).
    F, J = 30, 22
    base = np.zeros((F, J, 3))
    t = np.linspace(0, 1, F)
    # Root (joint 0) moves along (1, 0, 1) direction.
    base[:, 0, 0] = t
    base[:, 0, 2] = t
    # Give other joints a fixed offset so rotation is observable.
    base[:, 1, :] = base[:, 0, :] + np.array([0.0, 1.0, 0.0])
    out = align_direction(base, up_axis=1, root=0)

    disp = out[-1, 0] - out[0, 0]
    assert disp[0] > 0, "net travel should point +x after alignment"
    assert abs(disp[2]) < 1e-6, "z-component of travel should vanish"
    # Rotation preserves lengths: joint1-root distance unchanged.
    d_before = np.linalg.norm(base[:, 1] - base[:, 0], axis=-1)
    d_after = np.linalg.norm(out[:, 1] - out[:, 0], axis=-1)
    assert np.allclose(d_before, d_after), "alignment must be a rigid rotation"
    print("ok  align_direction")


def test_align_direction_noop_when_standing():
    pose = np.tile(np.random.default_rng(1).standard_normal((1, 22, 3)), (20, 1, 1))
    out = align_direction(pose, up_axis=1, root=0)
    assert np.allclose(out, pose), "no net travel -> no rotation"
    print("ok  align_direction (standing no-op)")


def test_resample_fps():
    # 4 seconds at 60 fps -> 30 fps should roughly halve the frame count.
    F = 241  # (F-1)/60 = 4.0 s
    pose = np.zeros((F, 22, 3))
    pose[:, 0, 0] = np.linspace(0, 4, F)  # ramp = time in seconds
    out = resample_fps(pose, src_fps=60.0, dst_fps=30.0)
    assert out.shape[0] == 121, f"expected 121 frames, got {out.shape[0]}"
    # Endpoints preserved and the ramp stays linear in seconds.
    assert np.isclose(out[0, 0, 0], 0.0) and np.isclose(out[-1, 0, 0], 4.0)
    assert np.isclose(out[60, 0, 0], 2.0, atol=1e-6), "midpoint = 2 s"
    # No-op when the rates already match.
    same = resample_fps(pose, 30.0, 30.0)
    assert same.shape == pose.shape
    print("ok  resample_fps")


def test_make_windows():
    pose = np.arange(150 * 22 * 3).reshape(150, 22, 3).astype(np.float32)
    w = make_windows(pose, clip_length=60, stride=30)
    # starts at 0,30,60,90 -> 90+60=150 fits; 120+60=180 does not.
    assert w.shape == (4, 60, 22, 3), w.shape
    assert np.array_equal(w[1, 0], pose[30]), "second window starts at frame 30"
    short = make_windows(pose[:40], clip_length=60, stride=30)
    assert short.shape == (0, 60, 22, 3), "walks shorter than T yield no windows"
    print("ok  make_windows")


def test_preprocess_walk_pipeline():
    rng = np.random.default_rng(2)
    pose = rng.standard_normal((100, 22, 3))
    pose[:, 0, 0] += np.linspace(0, 3, 100)  # travel in +x
    out = preprocess_walk(pose, src_fps=50.0, dst_fps=30.0)
    assert out.dtype == np.float32
    assert np.allclose(out[:, 0], 0.0, atol=1e-5), "root centred after pipeline"
    # 50 -> 30 fps on (100-1)/50 = 1.98 s -> round(1.98*30)+1 = 60 frames.
    assert out.shape[0] == 60, out.shape
    print("ok  preprocess_walk")


def test_cohort_index_and_tiers():
    idx = cohort_index(TIER1_COHORTS)
    assert idx == {"BMCLab": 0, "KUL-DT-T": 1, "E-LC": 2}
    assert tier_cohorts(1, 2) == TIER1_COHORTS + TIER2_COHORTS
    print("ok  cohort_index / tiers")


def _toy_walks():
    rng = np.random.default_rng(3)
    walks = []
    spec = [("BMCLab", "s1"), ("BMCLab", "s2"), ("KUL-DT-T", "s3"),
            ("E-LC", "s4"), ("BMCLab", "s1")]  # s1 has two walks
    for i, (coh, subj) in enumerate(spec):
        pose = rng.standard_normal((70, 22, 3)).astype(np.float32)
        walks.append(Walk(pose=pose, cohort=coh, subject=subj, fps=30.0,
                          labels={"updrs_gait": i % 3}))
    return walks


def test_build_bundle():
    b = build_bundle(_toy_walks())
    assert b.cohorts == ("BMCLab", "KUL-DT-T", "E-LC")
    assert b.n_cond == 3
    assert list(b.cohort_ids) == [0, 0, 1, 2, 0]
    assert len(b.videos) == 5 and b.videos[0].shape == (70, 22, 3)
    print("ok  build_bundle")


def test_splits():
    b = build_bundle(_toy_walks())
    # LOSO within BMCLab: subjects s1 (2 walks) and s2 (1 walk).
    folds = list(leave_one_subject_out(b, cohort="BMCLab"))
    subs = {name for name, _, _ in folds}
    assert subs == {"s1", "s2"}
    for name, tr, te in folds:
        assert set(tr).isdisjoint(set(te)), "train/test must be disjoint"
        # Test walks all belong to the held-out subject and to BMCLab.
        for i in te:
            assert b.subjects[i] == name and b.cohort_names[i] == "BMCLab"
    # LODO: each cohort held out once.
    lodo = {name: (tr, te) for name, tr, te in leave_one_cohort_out(b)}
    assert set(lodo) == {"BMCLab", "KUL-DT-T", "E-LC"}
    tr, te = lodo["BMCLab"]
    assert all(b.cohort_names[i] == "BMCLab" for i in te)
    assert all(b.cohort_names[i] != "BMCLab" for i in tr)
    # subset preserves the vocabulary.
    s = subset(b, te)
    assert s.cohorts == b.cohorts and len(s.videos) == len(te)
    print("ok  splits (LOSO / LODO / subset)")


def _fake_care_pd_blob(seed=7):
    """Build the nested {subject: {walk: record}} dict of the real release."""
    rng = np.random.default_rng(seed)
    blob = {}
    for subj in ("SUBJ_A", "SUBJ_B"):
        walks = {}
        for w in range(2):
            F = 80
            pose = rng.standard_normal((F, 22, 3)).astype(np.float32)
            pose[:, 0, 0] += np.linspace(0, 2, F)  # travel in +x
            walks[f"walk_{w}"] = {
                "pose": pose,
                "trans": rng.standard_normal((F, 3)).astype(np.float32),
                "beta": np.zeros(10, dtype=np.float32),
                "fps": 30,
                "UPDRS_GAIT": (w % 4),
                "medication": "ON" if w == 0 else "OFF",
                "other": None,
            }
        blob[subj] = walks
    return blob


def test_load_npz_nested_structure():
    # Save the nested dict as CARE-PD-style .npz and read it back.
    blob = _fake_care_pd_blob()
    with tempfile.TemporaryDirectory() as d:
        cohort_dir = Path(d) / "BMCLab"
        cohort_dir.mkdir()
        npz = cohort_dir / "h36m_3d_world_floorXZZplus_30f_or_longer.npz"
        # Also drop the camera-projected sibling the loader must skip.
        cam = cohort_dir / "h36m_3d_world2cam2img_sideright_30f_or_longer.npz"
        np.savez(npz, data=np.array(blob, dtype=object))
        np.savez(cam, data=np.array({"X": np.zeros((1, 22, 3))}, dtype=object))

        walks = load_cohort(npz, "BMCLab")
        assert len(walks) == 4, f"2 subjects x 2 walks, got {len(walks)}"
        # Subject id comes from the OUTER key.
        assert {w.subject for w in walks} == {"SUBJ_A", "SUBJ_B"}
        # Labels parsed with standardised aliases + raw passthrough.
        w0 = next(w for w in walks if w.labels.get("medication") == "ON")
        assert w0.labels["med"] == "ON"
        assert "updrs_gait" in w0.labels and "UPDRS_GAIT" in w0.labels
        # Preprocessing applied: root centred.
        assert np.allclose(walks[0].pose[:, 0], 0.0, atol=1e-5)

        # load_cohorts picks the world variant, not world2cam2img.
        walks2 = load_cohorts(d, ["BMCLab"])
        assert len(walks2) == 4
        b = build_bundle(walks2, cohorts=("BMCLab",))
        folds = list(leave_one_subject_out(b, cohort="BMCLab"))
        assert {name for name, _, _ in folds} == {"SUBJ_A", "SUBJ_B"}
    print("ok  load_cohort (nested .npz, outer-key subject, variant select)")


def _fake_h36m_release_dir(tmp: Path):
    """Write a fake ``h36m/<Cohort>/h36m_3d_world_*.npz`` mirroring the real
    output of ``smpl2h36m.py``: a flat ``{subject__walkid: (F,17,3)}`` npz.
    Also drops the two decoy world2cam / world2cam2img siblings so we can
    check that only the world variant is picked up.
    """
    rng = np.random.default_rng(11)
    cohort_dir = tmp / "h36m" / "BMCLab"
    cohort_dir.mkdir(parents=True)
    walks = {}
    for subj in ("SUBJ_A", "SUBJ_B"):
        for w in range(2):
            F = 80
            pose = rng.standard_normal((F, 17, 3)).astype(np.float32)
            pose[:, 0, 0] += np.linspace(0, 2, F)      # +X travel
            walks[f"{subj}__walk_{w}"] = pose
    np.savez(cohort_dir / "h36m_3d_world_floorXZZplus_30f_or_longer.npz",
             **walks)
    # Decoys that must NOT be picked up by the world glob.
    np.savez(cohort_dir / "h36m_3d_world2cam_sideright_...npz",
             SUBJ_A__walk_0=np.zeros((10, 17, 3), np.float32))
    np.savez(cohort_dir / "h36m_3d_world2cam2img_backright_...npz",
             SUBJ_A__walk_0=np.zeros((10, 17, 2), np.float32))
    return cohort_dir.parent


def _fake_source_pkl(path: Path):
    """Write a nested ``{subject: {walk: {pose, UPDRS_GAIT, medication, ...}}}``
    pickle mirroring the raw SMPL cohort files (labels only — no chumpy)."""
    import pickle
    rng = np.random.default_rng(12)
    blob = {}
    for si, subj in enumerate(("SUBJ_A", "SUBJ_B")):
        walks = {}
        for w in range(2):
            walks[f"walk_{w}"] = {
                "pose":  rng.standard_normal((80, 72)).astype(np.float32),
                "trans": rng.standard_normal((80, 3)).astype(np.float32),
                "beta":  np.zeros(10, np.float32),
                "fps": 30,
                "UPDRS_GAIT": (si + w) % 4,
                "medication": "ON" if w == 0 else "OFF",
                "other": None,
            }
        blob[subj] = walks
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def test_load_h36m_release_shape():
    with tempfile.TemporaryDirectory() as d:
        h36m_root = _fake_h36m_release_dir(Path(d))
        walks = load_cohorts(h36m_root, ["BMCLab"])

        # 2 subjects x 2 walks; joint count matches the real regressor.
        assert len(walks) == 4
        assert all(w.pose.shape[1:] == (H36M_N_JOINTS, 3) for w in walks), (
            "expected 17-joint H36M output"
        )
        assert H36M_N_JOINTS == 17, "H36M_N_JOINTS must be 17 for smpl2h36m.py"

        # Subject id came off the "subject__walkid" split.
        subs  = {w.subject  for w in walks}
        wids  = {w.walk_id  for w in walks}
        assert subs == {"SUBJ_A", "SUBJ_B"}
        assert wids == {"walk_0", "walk_1"}

        # Preprocess default runs; root-centring means pelvis at origin.
        assert np.allclose(walks[0].pose[:, 0], 0.0, atol=1e-5)

        # No labels yet — the .npz never had them.
        assert all(w.labels == {} for w in walks)
    print("ok  load_cohorts (flat h36m npz, 17 joints, subject split)")


def test_source_pkl_label_attach():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        h36m_root = _fake_h36m_release_dir(d)
        source_dir = d / "src"; source_dir.mkdir()
        _fake_source_pkl(source_dir / "BMCLab.pkl")

        # 1) explicit source_pkl kwarg on load_cohort.
        walks = load_cohort(
            h36m_root / "BMCLab" / "h36m_3d_world_floorXZZplus_30f_or_longer.npz",
            "BMCLab", source_pkl=source_dir / "BMCLab.pkl",
        )
        assert all("updrs_gait" in w.labels and "medication" in w.labels
                   for w in walks), "labels should be attached from source .pkl"
        # 2) auto-discovery via source_dir on load_cohorts.
        walks2 = load_cohorts(h36m_root, ["BMCLab"], source_dir=source_dir)
        assert all(w.labels for w in walks2)
        # 3) load_labels_pkl is double-keyed for robustness.
        lookup = load_labels_pkl(source_dir / "BMCLab.pkl")
        assert "SUBJ_A__walk_0" in lookup and "walk_0" in lookup
    print("ok  labels attached from source .pkl (double-keyed lookup)")


def main():
    test_root_center()
    test_align_direction_sends_travel_to_plus_x()
    test_align_direction_noop_when_standing()
    test_resample_fps()
    test_make_windows()
    test_preprocess_walk_pipeline()
    test_cohort_index_and_tiers()
    test_build_bundle()
    test_splits()
    test_load_npz_nested_structure()
    test_load_h36m_release_shape()
    test_source_pkl_label_attach()
    print("\n=== all CARE-PD preprocessing tests passed ===")


if __name__ == "__main__":
    main()
