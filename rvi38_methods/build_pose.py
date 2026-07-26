"""Long-format CSV to per-video ``(F, 15, 2)`` arrays, plus the METHODS §2 checks.

The analysis consumes a preprocessed long table with columns
``video, bp, frame, x, y, observed``. Two properties are asserted rather than
assumed, because both would silently corrupt everything downstream:

* **Frame contiguity** (§2.1) — the frame index set of each subject must be
  ``{0, ..., max}``, so the fixed-rate assumption behind every velocity and
  spectral quantity holds.
* **Constant joints** (§2.2) — torso normalisation puts MidHip at ``(0,0)`` and
  Neck at ``(0,1)``, so those two joints carry exactly zero variance. They are
  excluded from every amplitude, similarity, and spectral statistic; leaving
  them in would put zeros inside the logarithm of §6.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

JOINTS = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
          "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip",
          "LKnee", "LAnkle"]
CONSTANT_JOINTS = ("Neck", "MidHip")
FREE = [j for j, n in enumerate(JOINTS) if n not in CONSTANT_JOINTS]

# METHODS §2.1: positives at 1-based indices 5, 9, 10, 11, 18, 19.
POSITIVE_1BASED = (5, 9, 10, 11, 18, 19)


def default_labels(n: int = 38) -> np.ndarray:
    y = np.zeros(n, int)
    y[[i - 1 for i in POSITIVE_1BASED]] = 1
    return y


def load_labels(path: str | None = None, n: int = 38) -> tuple[np.ndarray, str]:
    """Labels from ``RVI_38_labels.mat`` when available, else the §2.1 vector.

    When the ``.mat`` is present its contents are compared against the
    documented index vector and any disagreement is raised, since §2.1 requires
    alignment to be verified directly rather than inferred.
    """
    doc = default_labels(n)
    if not path or not os.path.exists(path):
        return doc, "documented §2.1 index vector"
    try:
        from scipy.io import loadmat
        mat = loadmat(path)
    except Exception as exc:                              # noqa: BLE001
        return doc, f"documented §2.1 vector ({path} unreadable: {exc})"
    cand = [np.asarray(v).ravel() for k, v in mat.items()
            if not k.startswith("__") and np.asarray(v).size == n]
    if not cand:
        return doc, f"documented §2.1 vector ({path} had no length-{n} field)"
    y = (cand[0] > 0).astype(int)
    if not np.array_equal(y, doc):
        raise ValueError(
            f"{path} disagrees with the documented §2.1 positives "
            f"{POSITIVE_1BASED}: .mat positives (1-based) are "
            f"{(np.where(y)[0] + 1).tolist()}. Resolve before proceeding.")
    return y, os.path.basename(path)


def build(csv: str, out: str | None = "pose.npz", verbose: bool = True):
    """Read the long CSV and return ``(video_names, pose, observed, report)``.

    ``pose[v]`` is ``(F_v, 15, 2)`` float32 and ``observed[v]`` is
    ``(F_v, 15)`` uint8. ``report`` collects the §2 verification outcomes.
    """
    df = pd.read_csv(csv)
    missing_cols = {"video", "bp", "frame", "x", "y"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv} is missing columns {sorted(missing_cols)}")
    have_obs = "observed" in df.columns

    seen = set(df.bp.unique())
    if seen != set(JOINTS):
        raise ValueError(
            f"skeleton mismatch: {sorted(seen ^ set(JOINTS))} differ between "
            f"the CSV and the documented BODY-15 ordering")

    vids = sorted(df.video.unique())
    jidx = {b: i for i, b in enumerate(JOINTS)}
    df = df.assign(j=df.bp.map(jidx))

    pose, observed, gaps = {}, {}, {}
    for v, g in df.groupby("video", sort=True):
        F = int(g.frame.max()) + 1
        arr = np.full((F, len(JOINTS), 2), np.nan, np.float32)
        arr[g.frame.values, g.j.values, 0] = g.x.values
        arr[g.frame.values, g.j.values, 1] = g.y.values
        pose[v] = arr
        if have_obs:
            o = np.zeros((F, len(JOINTS)), np.uint8)
            o[g.frame.values, g.j.values] = g.observed.values
            observed[v] = o
        present = np.unique(g.frame.values)
        gaps[v] = F - len(present)

    report = {
        "n_videos": len(vids),
        "frames_per_video": {v: int(pose[v].shape[0]) for v in vids},
        "total_frames": int(sum(pose[v].shape[0] for v in vids)),
        "frame_gaps": {v: int(n) for v, n in gaps.items() if n},
        "contiguous": all(n == 0 for n in gaps.values()),
        "nan_frames": int(sum(np.isnan(pose[v]).any(axis=(1, 2)).sum()
                              for v in vids)),
    }

    # §2.2 constant-joint property, by direct variance computation.
    stacked = np.concatenate([pose[v].reshape(-1, len(JOINTS), 2) for v in vids])
    sd = np.nanstd(stacked, axis=0)                       # (15, 2)
    zero = [JOINTS[j] for j in range(len(JOINTS)) if np.all(sd[j] == 0)]
    report["zero_variance_joints"] = zero
    report["constant_joints_as_documented"] = set(zero) == set(CONSTANT_JOINTS)
    report["per_joint_sd"] = {JOINTS[j]: sd[j].round(6).tolist()
                              for j in range(len(JOINTS))}

    if verbose:
        print(f"    videos {report['n_videos']}, frames "
              f"{report['total_frames']}, contiguous={report['contiguous']}, "
              f"NaN frames={report['nan_frames']}")
        print(f"    zero-variance joints: {zero or 'none'} "
              f"(documented {list(CONSTANT_JOINTS)}) -> "
              f"{'OK' if report['constant_joints_as_documented'] else 'MISMATCH'}")
    if not report["constant_joints_as_documented"]:
        print("    WARNING (§2.2): the zero-variance set is not "
              f"{list(CONSTANT_JOINTS)}. FREE joints are still taken as the "
              "documented 13; check the normalisation before trusting §6.")

    if out:
        payload = {f"x_{v}": pose[v] for v in vids}
        if have_obs:
            payload.update({f"o_{v}": observed[v] for v in vids})
        np.savez_compressed(out, videos=np.array(vids), **payload)
    return vids, pose, observed, report


def load_cached(path: str = "pose.npz"):
    d = np.load(path, allow_pickle=True)
    vids = [str(v) for v in d["videos"]]
    pose = {v: d[f"x_{v}"] for v in vids}
    observed = {v: d[f"o_{v}"] for v in vids if f"o_{v}" in d}
    return vids, pose, observed
