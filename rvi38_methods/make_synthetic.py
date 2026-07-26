"""Generate a synthetic cohort with the RVI-38 structure, for smoke tests.

Produces a long-format CSV and two model dumps in the schema of METHODS §3.1,
so ``run_analysis.py`` can be exercised end to end without the real archive.
The planted structure is deliberate: states fall into blocks that are both
kinematically alike and dynamically connected, so A5 should recover a
partition, and the fluency of §7 should be positive.

    python make_synthetic.py --outdir synth --n-subjects 38 --k 11
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from a1_core import Geometry
from build_pose import JOINTS

# Canonical BODY-15 layout in torso-normalised coordinates: MidHip at (0,0),
# Neck at (0,1).
BASE = np.array([
    [0.00, 1.35],   # Nose
    [0.00, 1.00],   # Neck
    [-0.22, 1.00],  # RShoulder
    [-0.34, 0.72],  # RElbow
    [-0.40, 0.45],  # RWrist
    [0.22, 1.00],   # LShoulder
    [0.34, 0.72],   # LElbow
    [0.40, 0.45],   # LWrist
    [0.00, 0.00],   # MidHip
    [-0.14, 0.00],  # RHip
    [-0.17, -0.45], # RKnee
    [-0.19, -0.88], # RAnkle
    [0.14, 0.00],   # LHip
    [0.17, -0.45],  # LKnee
    [0.19, -0.88],  # LAnkle
])

GROUPS = {
    "head": [0], "R arm": [2, 3, 4], "L arm": [5, 6, 7],
    "R leg": [9, 10, 11], "L leg": [12, 13, 14],
}


def state_profiles(K, rng, bid=None):
    """Per-state per-joint motion amplitude.

    When ``bid`` (the transition-block id of each state) is supplied, states in
    the same dynamical block are given the same body-group signature. That is
    the structure A5 exists to detect: the kinematic clustering and the
    transition-derived clustering should then agree, so a synthetic run
    exercises the whole §8 chain rather than only its plumbing.
    """
    prof = np.full((K, 15), 0.0015)
    names = list(GROUPS)
    for k in range(K):
        g = names[(bid[k] if bid is not None else k) % len(names)]
        prof[k, GROUPS[g]] *= 4.0 + 1.5 * rng.random()
        if k % 3 == 2:                               # a milder sibling
            prof[k, GROUPS[g]] *= 0.55
    prof *= np.array([1.4, 1.0, 1.0, 1.6, 2.2, 1.0, 1.6, 2.2, 1.0,
                      1.0, 1.5, 2.0, 1.0, 1.5, 2.0])   # distal joints faster
    return prof


def block_transition(K, rng, n_blocks=4, stay=0.90, within=0.085):
    """Sticky chain whose jumps prefer states in the same block."""
    blocks = np.array_split(np.arange(K), n_blocks)
    bid = np.zeros(K, int)
    for b, idx in enumerate(blocks):
        bid[idx] = b
    A = np.full((K, K), (1 - stay - within) / max(K - 1, 1))
    for k in range(K):
        same = np.flatnonzero((bid == bid[k]) & (np.arange(K) != k))
        if len(same):
            A[k, same] += within / len(same)
        A[k, k] = stay
    A += rng.random((K, K)) * 1e-3
    return A / A.sum(1, keepdims=True), bid


def simulate(n_sub=38, K=11, seed=0, fmin=1600, fmax=2400, geom=None):
    geom = geom or Geometry()
    rng = np.random.default_rng(seed)
    A, bid = block_transition(K, rng)
    prof = state_profiles(K, rng, bid)
    cdf = A.cumsum(1)

    frames = rng.integers(fmin, fmax, n_sub)
    frames = (frames // geom.stride) * geom.stride     # tidy tiling
    rows, states, lengths, vidid = [], [], [], []

    for i in range(n_sub):
        F = int(frames[i])
        n_delta = geom.n_delta_windows(F)
        s = np.empty(n_delta, int)
        s[0] = rng.integers(K)
        u = rng.random(n_delta)
        for t in range(1, n_delta):
            s[t] = np.searchsorted(cdf[s[t - 1]], u[t])

        # per-frame amplitude from the covering window (§4.2 attribution)
        win_of_frame = np.clip((np.arange(F) - geom.f0) // geom.l, 0,
                               n_delta - 1)
        amp = prof[s[win_of_frame]]                    # (F, 15)

        x = np.tile(BASE, (F, 1, 1))
        jit = np.zeros((15, 2))
        a = 0.72                                       # OU retention
        for t in range(F):
            jit = a * jit + amp[t][:, None] * rng.normal(size=(15, 2))
            x[t] += jit
        x[:, 1] = BASE[1]                              # Neck exactly (0,1)
        x[:, 8] = BASE[8]                              # MidHip exactly (0,0)

        vid = f"synth_{i + 1:04d}"
        fr = np.repeat(np.arange(F), 15)
        jj = np.tile(np.arange(15), F)
        rows.append(pd.DataFrame({
            "video_number": i + 1, "video": vid,
            "bp": [JOINTS[j] for j in jj], "frame": fr,
            "x": x[fr, jj, 0], "y": x[fr, jj, 1],
            "fps": geom.fps, "observed": 1}))
        states.append(s)
        lengths.append(n_delta)
        vidid.append(np.full(n_delta, i))

    df = pd.concat(rows, ignore_index=True)
    states = np.concatenate(states)
    vidid = np.concatenate(vidid)
    lengths = np.array(lengths)

    # Empirical transition matrix over the pooled window path, within subjects.
    C = np.zeros((K, K))
    off = 0
    for L in lengths:
        seg = states[off:off + L]
        np.add.at(C, (seg[:-1], seg[1:]), 1.0)
        off += L
    P = C / C.sum(1, keepdims=True)
    return df, states, vidid, lengths, P, prof, bid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="synth")
    ap.add_argument("--n-subjects", type=int, default=38)
    ap.add_argument("--k", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fmin", type=int, default=1600)
    ap.add_argument("--fmax", type=int, default=2400)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df, states, vidid, lengths, P, prof, bid = simulate(
        a.n_subjects, a.k, a.seed, a.fmin, a.fmax)
    csv = os.path.join(a.outdir, "synth_analysis.csv")
    df.to_csv(csv, index=False)

    import joblib
    occ = np.bincount(states, minlength=a.k) / len(states)
    rng = np.random.default_rng(a.seed)
    d_z = 8
    blob = {
        "res": {"states": states, "transition": P, "occupancy": occ,
                "means": rng.normal(size=(a.k, d_z)),
                "covars": np.tile(np.eye(d_z), (a.k, 1, 1))},
        "lengths": lengths, "vidid": vidid,
        "Z": rng.normal(size=(len(states), d_z)),
        "ar_As": rng.normal(0, .1, (a.k, d_z, 2 * d_z)),
        "ar_bs": rng.normal(0, .1, (a.k, d_z)),
        "ar_Sigmas": np.tile(np.eye(d_z), (a.k, 1, 1)),
    }
    joblib.dump(blob, os.path.join(a.outdir, "synth_arhmm.pkl"), compress=3)

    gauss = {"res": {k: v for k, v in blob["res"].items()},
             "lengths": lengths, "vidid": vidid, "Z": blob["Z"]}
    joblib.dump(gauss, os.path.join(a.outdir, "synth_hmm.pkl"), compress=3)

    print(f"wrote {csv}  ({len(df):,} rows, {a.n_subjects} subjects, "
          f"{len(states):,} windows, K={a.k})")
    print(f"planted blocks: {bid.tolist()}")
    print(f"windows/subject: {lengths.min()}..{lengths.max()}")


if __name__ == "__main__":
    main()
