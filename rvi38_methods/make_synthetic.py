"""Generate a synthetic cohort with the RVI-38 structure, for smoke tests.

Produces a long-format CSV and two model dumps in the schema of METHODS §3.1,
so ``run_analysis.py`` can be exercised end to end without the real archive.
The planted structure is deliberate: states fall into blocks that are both
kinematically alike and dynamically connected, so the fluency of §7 should
be positive.

    python make_synthetic.py --outdir synth --n-subjects 38 --k 11
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from a1_core import Geometry
from build_pose import JOINTS, default_labels

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
    the structure the fluency analysis exists to detect: the kinematic clustering and the
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


# Chains FidgetyFind scores: moving joint -> parent joint. The planted fidgety
# layer moves exactly these, in the amplitude band the detector looks in.
FIDGETY_CHAINS = {10: 9, 13: 12,        # knees against the hips
                  4: 3, 7: 6,           # wrists against the forearms
                  11: 10, 14: 13}       # ankles against the shanks


def fidgety_layer(F, rng, present: bool, amp=(0.085, 0.115), hold: int = 4,
                  active: float = 0.85, absent_active: float = 0.12,
                  cone=np.deg2rad(20.0), radius: float = 0.30):
    """Per-frame offsets that carry (or withhold) a fidgety movement signal.

    The construct FidgetyFind measures is *small movement that keeps changing
    direction*, so that is exactly what is planted: steps whose length is a
    fixed fraction of the parent limb (``amp``, in units of that limb) and
    whose direction is redrawn every ``hold`` frames.

    ``hold`` is not decoration. A direction redrawn every single frame is white
    noise, and the confidence-weighted 5-frame smoothing the detector specifies
    removes it almost entirely -- which is precisely what that smoothing is
    for. Real fidgety movement is coherent over a tenth of a second or so, and
    a planted signal has to be too, or the synthetic cohort tests the smoother
    rather than the detector. ``amp`` is likewise set above the detector's
    band, because smoothing costs roughly half the per-frame amplitude.

    ``present=True`` (the normal group) steps on ``active`` of the direction
    blocks, each in a uniformly random direction. ``present=False`` (the
    abnormal group) steps on only ``absent_active`` of them and always along
    one axis, alternating sign: too few small movements to judge, and no
    direction variety in the ones there are. Excursions are kept within
    ``radius`` limb-lengths of the resting position by flipping a step that
    would leave it, so what is planted is a fidget and not a drift.

    Returns ``(F, 2)`` offsets in units of the parent limb's length.
    """
    lo, hi = amp
    off = np.zeros((F, 2))
    pos = np.zeros(2)
    axis = rng.uniform(-np.pi, np.pi)
    sign = 1.0
    step = np.zeros(2)
    for t in range(F):
        if t % max(int(hold), 1) == 0:
            if rng.random() < (active if present else absent_active):
                r = rng.uniform(lo, hi)
                if present:
                    a = rng.uniform(-np.pi, np.pi)
                else:
                    a = axis + cone * rng.uniform(-1, 1) + (0.0 if sign > 0
                                                            else np.pi)
                    sign = -sign
                step = r * np.array([np.cos(a), np.sin(a)])
            else:
                step = np.zeros(2)
        if step.any():
            if np.linalg.norm(pos + step) > radius:
                step = -step
            pos = pos + step
        off[t] = pos
    return off


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


def simulate(n_sub=38, K=11, seed=0, fmin=1600, fmax=2400, geom=None,
             fidgety=True):
    geom = geom or Geometry()
    rng = np.random.default_rng(seed)
    A, bid = block_transition(K, rng)
    prof = state_profiles(K, rng, bid)
    cdf = A.cumsum(1)

    # Anisotropic shape structure for the direction-aware similarity (§7): each
    # dynamical block gets its own principal axis, and the active joints of a
    # state move mostly along it. The per-joint transform ``T`` satisfies
    # ``trace(T Tᵀ) = 2``, the same total speed variance as isotropic noise, so
    # the magnitude channel ``a`` is unchanged in expectation and only the axis
    # is planted -- exactly the redirection the shape channel exists to detect.
    names = list(GROUPS)
    n_blocks = int(bid.max()) + 1
    aniso = 0.85                                       # -> anisotropy rho = 0.85
    sx, sy = np.sqrt(1 + aniso), np.sqrt(1 - aniso)
    Tshape = np.tile(np.eye(2), (K, 15, 1, 1))
    for k in range(K):
        ang = np.pi * bid[k] / max(n_blocks, 1)        # block axis in [0, pi)
        R = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang), np.cos(ang)]])
        for j in GROUPS[names[bid[k] % len(names)]]:
            Tshape[k, j] = R @ np.diag([sx, sy])

    labels_synth = default_labels(n_sub) if n_sub >= 19 else np.zeros(n_sub, int)

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

        state_f = s[win_of_frame]                      # state at each frame
        x = np.tile(BASE, (F, 1, 1))
        jit = np.zeros((15, 2))
        a = 0.72                                       # OU retention
        for t in range(F):
            noise = np.einsum("jab,jb->ja", Tshape[state_f[t]],
                              rng.normal(size=(15, 2)))
            jit = a * jit + amp[t][:, None] * noise
            x[t] += jit
        x[:, 1] = BASE[1]                              # Neck exactly (0,1)
        x[:, 8] = BASE[8]                              # MidHip exactly (0,0)

        # Fidgety layer: the signal FidgetyFind looks for, planted on the six
        # chains it scores and keyed to the same labels the clinical layer uses.
        # It is additive and joint-local, so the state structure planted above
        # (which lives in the *amplitude* profile of whole body groups) is
        # untouched.
        if fidgety:
            present = not bool(labels_synth[i])
            for moving, parent in FIDGETY_CHAINS.items():
                ref = float(np.linalg.norm(BASE[moving] - BASE[parent]))
                x[:, moving] += ref * fidgety_layer(F, rng, present)

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
    ap.add_argument("--no-fidgety", action="store_true",
                    help="omit the planted fidgety layer (the small, "
                         "direction-varying movement FidgetyFind detects). "
                         "Without it every window scores zero, which is the "
                         "detector working, not failing.")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df, states, vidid, lengths, P, prof, bid = simulate(
        a.n_subjects, a.k, a.seed, a.fmin, a.fmax,
        fidgety=not a.no_fidgety)
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
    fidgety = ("planted (absent in the labelled positives)"
               if not a.no_fidgety else "omitted")
    print(f"fidgety layer: {fidgety}")
    print(f"windows/subject: {lengths.min()}..{lengths.max()}")


if __name__ == "__main__":
    main()
