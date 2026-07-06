"""Smoke test of every part that does not need PyTorch.

Runs the config validation, the three mask policies, the data slicer,
the train/val split, and checks the analytical parameter counts against
the numbers in [ARCH §6.1].
"""

import numpy as np

from vae_training import (TrainingConfig, NoMask, UniformMask, LimbMask,
                          TopKSpeedMask, SoftmaxSpeedMask, PerFrameSpeedMask,
                          build_policy)
from vae_training.data import slice_video, build_clips, train_val_split
from vae_training.param_counts import conv_param_count, transformer_param_count


rng = np.random.default_rng(0)


# ---- Config validation ------------------------------------------------
cfg = TrainingConfig(architecture="conv", clip_length=32, n_joints=17,
                     recipe=1, mask_policy="uniform")
cfg.validate()
print(f"config OK, downsample factor = {cfg.downsample_factor()}")

try:
    bad = TrainingConfig(clip_length=30, conv_strides=(1, 2, 2), recipe=1)
    bad.validate()
    print("ERROR: expected a ValueError on non-divisible clip length")
except ValueError as e:
    print(f"clean error on bad clip length: {str(e)[:60]}...")

try:
    bad = TrainingConfig(recipe=2, mask_policy="none")
    bad.validate()
    print("ERROR: expected a ValueError on Recipe 2 without a mask policy")
except ValueError as e:
    print(f"clean error on Recipe 2 without a mask policy: {str(e)[:60]}...")

try:
    bad = TrainingConfig(recipe=3, mask_policy="none")
    bad.validate()
    print("ERROR: expected a ValueError on Recipe 3 without a mask policy")
except ValueError as e:
    print(f"clean error on Recipe 3 without a mask policy: {str(e)[:60]}...")


# ---- Mask policies ----------------------------------------------------
T, J = 32, 17
none = NoMask()
uni = UniformMask(rho=0.3)
limbs = {"left_arm": [1, 2, 3], "right_arm": [4, 5, 6],
         "left_leg": [10, 11, 12], "right_leg": [13, 14, 15]}
lim = LimbMask(limbs=limbs)

M_none = none.sample(T, J, rng)
M_uni = uni.sample(T, J, rng)
M_lim = lim.sample(T, J, rng)
print(f"none: all ones? {np.all(M_none == 1)}")
print(f"uniform: fraction hidden = {(1 - M_uni.mean()):.3f} (target 0.30)")
print(f"limb: one limb hidden, {(1 - M_lim).sum()} zero entries")


# ---- Speed-based mask policies ([MVAE §2.3-2.5]) ---------------------
# Give each joint its own oscillation frequency so speeds vary and the
# top-k policies pick something informative.
freqs = rng.uniform(0.02, 0.15, size=J)
tvec = np.arange(T)[:, None]
X_fake = np.stack([
    np.sin(tvec.squeeze(-1) * freqs[j] + j) for j in range(J)
], axis=1)[..., None].repeat(3, axis=-1)             # (T, J, 3)

topk = TopKSpeedMask(rho=0.3)
soft = SoftmaxSpeedMask(rho=0.3, temperature=1.0)
pf   = PerFrameSpeedMask(rho=0.3)

M_topk = topk.sample(T, J, rng, X=X_fake)
M_soft = soft.sample(T, J, rng, X=X_fake)
M_pf   = pf.sample(T, J, rng, X=X_fake)
k_target = int(np.floor(0.3 * J))
print(f"top_k_speed: {int((1 - M_topk[0]).sum())} joints hidden per frame "
      f"(target {k_target}); constant across frames? "
      f"{np.all(M_topk == M_topk[0])}")
print(f"softmax_speed: {int((1 - M_soft[0]).sum())} joints hidden per frame "
      f"(target {k_target})")
per_frame_counts = (1 - M_pf).sum(axis=1)
print(f"per_frame_speed: exactly {k_target} hidden per frame? "
      f"{np.all(per_frame_counts == k_target)}; "
      f"varies over time? {(M_pf != M_pf[0]).any()}")

# Speed-weighted limb picks the fastest limb more often; check that the
# distribution isn't uniform when the speeds are.
limbs_uneven = {"slow": [0, 1], "fast": [2, 3]}
X_uneven = np.zeros((T, J, 3), dtype=np.float32)
X_uneven[:, 2:4, 0] = np.linspace(0, 10, T)[:, None]   # only "fast" moves
lim_w = LimbMask(limbs=limbs_uneven, speed_weighted=True)
picks = [int((1 - lim_w.sample(T, J, rng, X=X_uneven)).sum() > 0
             and (1 - lim_w.sample(T, J, rng, X=X_uneven))[0, 2] == 1)
         for _ in range(200)]
print(f"speed-weighted limb: fast limb picked {sum(picks)}/200 times "
      f"(uniform would give ~100)")


# ---- Batch mask draw --------------------------------------------------
B = 8
Mb = uni.sample_batch(B, T, J, rng)
assert Mb.shape == (B, T, J)
print(f"batch mask draw shape: {Mb.shape}")


# ---- Policy factory ---------------------------------------------------
p1 = build_policy(TrainingConfig(recipe=1, mask_policy="uniform"))
p2 = build_policy(TrainingConfig(recipe=2, mask_policy="none"))
p3 = build_policy(TrainingConfig(recipe=3, mask_policy="limb",
                                 mask_limb_names=["left_arm", "right_arm"]),
                  limbs=limbs)
p_topk = build_policy(TrainingConfig(recipe=1, mask_policy="top_k_speed"))
p_soft = build_policy(TrainingConfig(recipe=1, mask_policy="softmax_speed",
                                     mask_softmax_temperature=0.5))
p_pf   = build_policy(TrainingConfig(recipe=1, mask_policy="per_frame_speed"))
print(f"factory: {type(p1).__name__}, {type(p2).__name__}, {type(p3).__name__}, "
      f"{type(p_topk).__name__}, {type(p_soft).__name__}, {type(p_pf).__name__}")


# ---- Video slicing ----------------------------------------------------
video = rng.standard_normal((200, J, 3))
clips = slice_video(video, T=32, stride=16)
print(f"slice_video: {clips.shape}, expected (11, 32, {J}, 3)")

videos = [rng.standard_normal((200, J, 3)) for _ in range(3)]
clips, vid, t0 = build_clips(videos, T=32, stride=16)
print(f"build_clips: {clips.shape}, {len(np.unique(vid))} videos, "
      f"first time_index values {t0[:5]}")

train, val = train_val_split(clips, vid, val_fraction=0.15)
print(f"split: {train.sum()} train / {val.sum()} val")


# ---- Parameter counts against ARCH §6.1 ------------------------------
# Design-note values: T = 32, J = 22, d_z = 32, conv C = 64.
cfg_conv = TrainingConfig(architecture="conv", clip_length=32, n_joints=22,
                          latent_dim=32, conv_base_channels=64,
                          recipe=1, mask_policy="uniform")
counts = conv_param_count(cfg_conv)
expected = {"encoder_block_1": 28224, "encoder_block_2": 24704,
            "encoder_block_3": 49280, "bottleneck_heads": 65600,
            "decoder_lift": 33792, "decoder_block_3": 49280,
            "decoder_block_2": 24640, "decoder_output": 21186}
print("\nconv parameter counts vs ARCH §6.1:")
for k, exp in expected.items():
    got = counts[k]
    ok = "OK" if got == exp else "MISMATCH"
    print(f"  {k:22s} expected {exp:>7d}  got {got:>7d}  {ok}")
print(f"  {'total':22s} expected {'≈ 297k':>7s}  got {counts['total']:>7d}")

# Transformer defaults from ARCH §6.1: d_model = 96, L = 3, H = 4, ffn 4x.
cfg_tx = TrainingConfig(architecture="transformer", clip_length=32,
                        n_joints=22, latent_dim=32, d_model=96,
                        n_layers=3, n_heads=4, ffn_ratio=4,
                        recipe=1, mask_policy="uniform")
tx = transformer_param_count(cfg_tx)
print("\ntransformer parameter counts (target ≈ 689k in ARCH §6.1):")
for k, v in tx.items():
    print(f"  {k:22s} {v:>7d}")


# ---- J-generic check --------------------------------------------------
print("\nvarying J (should scale cleanly):")
for J in (13, 22, 30, 44):
    c = TrainingConfig(architecture="conv", clip_length=32, n_joints=J,
                       recipe=1, mask_policy="uniform")
    print(f"  J = {J:2d}  conv total = {conv_param_count(c)['total']:>7d}")

print("\n=== every non-torch path ran ===")
