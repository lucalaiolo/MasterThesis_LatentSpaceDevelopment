"""Smoke test of the analysis toolkit with a fake model and synthetic data.

Exercises every path that needs only NumPy, SciPy, and scikit-learn.
Torch paths (encoder/decoder Jacobians) and optional-package paths
(persistent homology, hidden Markov model, PELT) are checked for import
and interface only.
"""

import numpy as np

from vae_analysis import Skeleton, LatentSet, encode_dataset
from vae_analysis import (posterior_geometry as pg, features as ft,
                          masking as mk, generation as gen, information as inf,
                          symmetry as sym, disentanglement as dis,
                          two_sample as ts, screening as scr, honesty as hon,
                          dynamics as dyn)

rng = np.random.default_rng(0)

# ---- Synthetic setup: generic J, not 22. ----
J, T, D_Z, N = 17, 32, 16, 400
skel = Skeleton(
    n_joints=J,
    bones=[(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8)],
    left_right=[(1, 4), (2, 5), (3, 6), (9, 12), (10, 13), (11, 14)],
    lateral_axis=0,
    limbs={"left_arm": [1, 2, 3], "right_arm": [4, 5, 6],
           "left_leg": [9, 10, 11], "right_leg": [12, 13, 14]},
)


class FakeModel:
    """A linear encoder/decoder standing in for a trained VAE."""

    def __init__(self, J, T, d_z, seed=1):
        r = np.random.default_rng(seed)
        self.J, self.T, self.d_z = J, T, d_z
        self.W_enc = r.standard_normal((d_z, T * J * 3)) * 0.05
        self.W_dec = r.standard_normal((T * J * 3, d_z)) * 0.05

    def encode(self, X, M):
        n = len(X)
        feat = (X * M[..., None]).reshape(n, -1)
        mu = feat @ self.W_enc.T
        logvar = np.full((n, self.d_z), -1.0)
        return mu, logvar

    def decode(self, z):
        n = len(z)
        return (z @ self.W_dec.T).reshape(n, self.T, self.J, 3)


model = FakeModel(J, T, D_Z)

# Synthetic clips with a mild left-right structure and two videos.
X = rng.standard_normal((N, T, J, 3)) * 0.3
X[:, :, :, 0] += 0.5  # shift lateral axis so the flip is non-trivial
M = (rng.random((N, T, J)) > 0.2).astype(np.float32)
video_id = (np.arange(N) < N // 2).astype(int)
time_index = np.concatenate([np.arange(N // 2), np.arange(N - N // 2)]) * (T // 2)

latent = encode_dataset(model, X, M, video_id=video_id, time_index=time_index)
latent.sample(rng)
print("encoded:", latent.n, "clips, d_z =", latent.d_z)

results = {}

# ---- Part I Section 3: posterior geometry ----
results["mmd_prior"] = pg.mmd_prior_test(latent, n_samples=300, n_perm=50, rng=rng)
results["intrinsic_dim"] = pg.intrinsic_dimension_twonn(latent.mu)["d_hat"]
results["clusters_k"] = pg.cluster_structure(latent, k_range=range(2, 6))["k"]

# ---- Part I Section 5: features ----
feats, names = ft.kinematic_features(X, skel)
results["n_features"] = len(names)
results["regression_mean_r2"] = ft.feature_regression(latent, feats, names)["_mean"]
results["cca"] = ft.canonical_correlation(latent, feats, n_components=3)["correlations"]

# ---- Part I Section 6: masking ----
results["mask_jitter"] = mk.mask_jitter(model, X[:40],
                                        mk.uniform_sampler(0.3), k=8, rng=rng)["ratio"]
results["latent_recovery"] = mk.latent_recovery(model, X[:40], skel, rng=rng)
Xhat = model.decode(latent.z)
results["split_mpjpe"] = mk.split_mpjpe(X, Xhat, M)["ratio"]

# ---- Part I Section 8: generation ----
gen_clips = model.decode(latent.prior_like(100, rng))
results["bone_plausibility"] = gen.bone_plausibility(gen_clips, X, skel)["ratio"].mean()
results["frechet"] = gen.frechet_distance(feats, ft.kinematic_features(gen_clips, skel)[0])
results["interp_curv"] = gen.interpolation_curvature(model, latent, n_pairs=20, rng=rng)

# ---- Part I Section 9 / Part II Section 19: information ----
results["tc"] = inf.tc_decomposition(latent, batch=256, rng=rng)
results["active_units"] = inf.active_units(latent)["n_active"]
rd = inf.rate_distortion_curve([
    {"beta": 1e-3, "rate": 40.0, "distortion": 2.0},
    {"beta": 1e-2, "rate": 22.0, "distortion": 2.4},
    {"beta": 1e-1, "rate": 8.0, "distortion": 3.6},
    {"beta": 1.0, "rate": 1.5, "distortion": 7.0},
])
results["rd_knee_beta"] = rd["knee_beta"]

# ---- Part II Section 15: symmetry ----
eq = sym.fit_equivariance(model, X, M, skel)
results["equivariance_r2"] = eq["variance_explained"]
lat_sub = sym.laterality_subspace(eq["A"])
results["antisym_dim"] = lat_sub["antisymmetric_dim"]
results["asymmetry_score_shape"] = sym.asymmetry_score(latent, lat_sub["projector"]).shape

# ---- Part II Section 16: disentanglement ----
results["mig"] = dis.mig(latent, feats)["mig"]
results["dci"] = {k: v for k, v in dis.dci(latent, feats).items()
                  if k in ("disentanglement", "completeness")}
results["sap"] = dis.sap(latent, feats)["sap"]
states = video_id  # stand-in state labels for the selectivity control
results["selectivity"] = dis.selectivity(latent, feats, states,
                                         score_fn=dis.mig, rng=rng)["selectivity"]

# ---- Part II Section 20: classifier two-sample ----
results["c2st"] = ts.classifier_two_sample(latent, n_samples=600, rng=rng)["accuracy"]

# ---- Part II Section 21: screening ----
dens = scr.fit_density(latent, method="gmm", n_components=4)
scores = scr.typicality_score(dens, latent)
results["typicality_shape"] = scores.shape
results["screening_auc"] = scr.screening_auc(scores, video_id)

# ---- Part II Section 23: attention ----
fake_attn = rng.dirichlet(np.ones(T), size=(N, 4))  # (N, H, T)
results["attn_entropy"] = scr.attention_entropy(fake_attn)["mean_entropy"].shape
motion = np.linalg.norm(np.diff(X, axis=1), axis=-1).mean(axis=-1)  # (N, T-1)
motion = np.concatenate([motion, motion[:, -1:]], axis=1)          # pad to T
results["selected_frames"] = len(scr.selected_frames(fake_attn, motion)["focused_heads"])

# ---- Part I Section 12: honesty ----
blocks = hon.time_blocks(latent, block_seconds=5.0, fps=25.0)
asym = sym.asymmetry_score(latent, lat_sub["projector"])
results["bootstrap"] = hon.block_bootstrap(asym, blocks, n_boot=100, rng=rng)
results["permutation"] = hon.permutation_between_videos(asym, video_id, blocks,
                                                        n_perm=100, rng=rng)

# ---- Part I Section 7 / Part II Section 22: dynamics (fallback paths) ----
video = rng.standard_normal((600, J, 3)) * 0.3
traj = dyn.encode_video(model, video, window=T, stride=T // 2)
results["traj_shape"] = traj.shape
results["change_points"] = dyn.change_points(traj)["n_segments"]
results["ou_timescales_shape"] = dyn.ou_process(traj, stride_seconds=0.64)["timescales_seconds"].shape

print("\n=== all paths ran ===")
for k, v in results.items():
    print(f"{k}: {v}")
