"""End-to-end smoke test for the neonate GAITGen RVQ-VAE (Phase A).

Builds synthetic 2D-keypoint gait sequences with a planted pathology signal
(reduced arm swing at higher class, like the paper's Fig. 4), runs both
training stages briefly with test-scale learning rates, and checks the
disentanglement mechanism moves in the right direction:

  * Stage 1 reconstruction error falls.
  * Stage 2 pathology classifier accuracy on q_p is high.
  * Pathology is more decodable from q_p than from q_m (PMPG > 0) — the
    adversary is pushing pathology out of the motion latent.
  * PORE > 0 (q_p alone reconstructs worse than q_m + q_p).

Needs numpy + torch. Run with:  python -m neonate_gaitgen.smoke_test
"""

from __future__ import annotations

import numpy as np

from neonate_gaitgen.config import GaitGenConfig
from neonate_gaitgen.preprocess import Sequence, build_windowed_data
from neonate_gaitgen.train import train


J = 17
ARM_JOINTS = [5, 6, 7, 8]


def synthetic_sequences(n_subjects=12, seqs_per=5, F=170, C=3, seed=0):
    """Gait sequences whose class reduces arm-swing amplitude (the pathology)."""
    rng = np.random.default_rng(seed)
    seqs = []
    for subj in range(n_subjects):
        phase = rng.uniform(0, 2 * np.pi)          # subject-specific gait phase
        cadence = rng.uniform(0.18, 0.24)
        for _ in range(seqs_per):
            cp = int(rng.integers(0, C))
            arm_amp = 1.0 - 0.3 * cp               # higher class -> less arm swing
            t = np.arange(F)
            pose = np.zeros((F, J, 2), dtype=np.float64)
            for j in range(J):
                pose[:, j, 0] = 0.1 * j + 0.20 * np.sin(cadence * t + phase + 0.3 * j)
                pose[:, j, 1] = 0.05 * j + 0.10 * np.cos(cadence * t + phase)
            for j in ARM_JOINTS:                   # arms carry the pathology signal
                pose[:, j, 0] += arm_amp * 0.6 * np.sin(cadence * t + phase)
            pose += rng.standard_normal((F, J, 2)) * 0.02
            seqs.append(Sequence(pose=pose, c_p=cp, subject=f"s{subj}"))
    return seqs


def _encode_all(model, data, device):
    import torch
    model.eval()
    qm, qp, cps = [], [], []
    with torch.no_grad():
        for i in range(0, data.n, 256):
            x = torch.from_numpy(data.x[i:i + 256]).float().to(device)
            c = torch.from_numpy(data.c_p[i:i + 256]).long().to(device)
            lat = model.encode_latents(x, c)
            qm.append(lat["q_m"].mean(1).cpu().numpy())   # mean-pool over time
            qp.append(lat["q_p"].mean(1).cpu().numpy())
            cps.append(data.c_p[i:i + 256])
    return np.concatenate(qm), np.concatenate(qp), np.concatenate(cps)


def _probe_acc(z, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import balanced_accuracy_score
    pred = cross_val_predict(LogisticRegression(max_iter=500), z, y, cv=3)
    return balanced_accuracy_score(y, pred)


def main():
    seqs = synthetic_sequences()
    cfg = GaitGenConfig(
        n_joints=J, in_channels=2, clip_length=60, stride=30,
        n_classes=3, label_type="ordinal", healthy_id=0,
        d_motion=32, d_pathology=32, cond_dim=8, hidden_channels=128,
        n_rvq_layers=4, codebook_motion=128, codebook_pathology=32,
        # test-scale schedule + LRs (paper's 2e-6 needs the full long run):
        stage1_epochs=20, stage2_epochs=40, lr_stage1=1e-3,
        lr_motion_stage2=1e-4, lr_pathology=1e-3, lr_classifier=1e-3,
        grl_lambda_max=1.0, grl_warmup_epochs=10,
        batch_size=64, device="cpu", out_dir="/tmp/gaitgen_smoke")
    data = build_windowed_data(seqs, cfg)
    print(f"[smoke] {data.n} clips, {len(set(data.subject))} subjects, "
          f"classes={sorted(set(data.c_p))}")

    out = train(cfg, data)
    model = out["model"]
    hist = out["history"]

    # Stage 1 reconstruction should improve.
    s1_first = hist["stage1"][0]["val"]["mpjpe"]
    s1_last = hist["stage1"][-1]["val"]["mpjpe"]
    print(f"[smoke] stage-1 val mpjpe {s1_first:.4f} -> {s1_last:.4f}")
    assert s1_last < s1_first, "stage-1 reconstruction did not improve"

    # Disentanglement direction: pathology more decodable from q_p than q_m.
    import torch
    device = next(model.parameters()).device
    qm, qp, cps = _encode_all(model, data, device)
    acc_qp = _probe_acc(qp, cps)
    acc_qm = _probe_acc(qm, cps)
    chance = 1.0 / cfg.n_classes
    print(f"[smoke] probe c_p:  q_p={acc_qp:.3f}  q_m={acc_qm:.3f}  "
          f"chance={chance:.3f}  (PMPG={acc_qp - acc_qm:+.3f})")
    assert acc_qp > chance + 0.1, "q_p does not carry pathology"
    assert acc_qp > acc_qm, "pathology not more decodable from q_p than q_m (PMPG<=0)"

    # PORE direction: q_p alone reconstructs worse than q_m + q_p.
    with torch.no_grad():
        x = torch.from_numpy(data.x[:256]).float().to(device)
        c = torch.from_numpy(data.c_p[:256]).long().to(device)
        lat = model.encode_latents(x, c)
        e_p = model._mpjpe(x, model.decode(q_m=None, q_p=lat["q_p"])).item()
        e_pm = model._mpjpe(x, model.decode(lat["q_m"], lat["q_p"])).item()
    pore = (e_p - e_pm) / (e_pm + 1e-8)
    print(f"[smoke] PORE: e_p={e_p:.4f} e_pm={e_pm:.4f} -> PORE={pore:+.3f}")
    assert e_p > e_pm, "pathology-only recon not worse than full recon (PORE<=0)"

    print("\n=== neonate GAITGen Phase-A path ran; disentanglement direction OK ===")


if __name__ == "__main__":
    main()
