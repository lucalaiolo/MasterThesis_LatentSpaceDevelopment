"""End-to-end smoke test for the CARE-PD GM-CVAE stack ([CARE-PD §7, §11]).

Needs PyTorch and scikit-learn. Builds a small synthetic multi-cohort
motion set with a planted phenotype signal, trains each of the four models
in the plan — VAE, CVAE(cohort), GM-VAE, GM-CVAE — for a few epochs, then
encodes a frozen latent and runs the evaluation battery (site probe,
cluster–label agreement, linear probe, occupancy).

The point is to prove every code path flows and every metric returns a
number, not to demonstrate the scientific result (that needs the real
data and a real training budget). Run with

    python -m architectures.gm_smoke_test
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from architectures.config import TrainingConfig
from architectures.train import train
from architectures import metrics
from architectures.models import GaussianMixturePrior  # noqa: F401


def synthetic_cohorts(n_cohorts: int = 3, subjects_per: int = 4,
                      walks_per: int = 3, frames: int = 200, J: int = 22,
                      seed: int = 0):
    """Synthetic walks with a cohort nuisance axis and a phenotype signal.

    Each walk carries (a) a cohort-specific offset/scale — the nuisance a
    CVAE should strip — and (b) a binary "phenotype" that shifts the gait
    frequency — the structure a GM model should recover. Returns the
    pieces ``train`` and the metrics expect.
    """
    rng = np.random.default_rng(seed)
    videos, cohort_ids, subjects, phenotype = [], [], [], []
    for coh in range(n_cohorts):
        cohort_bias = rng.standard_normal((J, 3)) * 0.5      # nuisance
        cohort_scale = 1.0 + 0.3 * coh
        for subj in range(subjects_per):
            sid = f"c{coh}_s{subj}"
            for _ in range(walks_per):
                pheno = rng.integers(0, 2)                   # 0 / 1
                freq = 0.05 if pheno == 0 else 0.12          # gait signal
                t = np.arange(frames) * freq
                base = np.stack([np.sin(t + i) for i in range(J)], axis=1)
                base = base[..., None].repeat(3, axis=-1)
                walk = cohort_scale * base + cohort_bias[None]
                walk += rng.standard_normal((frames, J, 3)) * 0.05
                videos.append(walk.astype(np.float32))
                cohort_ids.append(coh)
                subjects.append(sid)
                phenotype.append(int(pheno))
    return (videos, np.asarray(cohort_ids), subjects,
            np.asarray(phenotype), n_cohorts)


def encode_latents(result, videos, cohort_ids, clip_length=60, stride=30):
    """Encode one posterior mean per walk with the trained model.

    Uses the first window of each walk so the returned arrays line up with
    the per-walk cohort / phenotype labels for the metrics.
    """
    import torch
    model = result["model"].eval()
    mixture = result.get("mixture")
    device = next(model.parameters()).device
    mus, resp = [], []
    with torch.no_grad():
        for v, c in zip(videos, cohort_ids):
            clip = torch.from_numpy(v[:clip_length][None]).to(device)
            mask = torch.ones(1, clip_length, v.shape[1], device=device)
            cc = torch.tensor([int(c)], device=device)
            mu, logvar = model.encode(clip, mask, cc)
            mus.append(mu[0].cpu().numpy())
            if mixture is not None:
                resp.append(mixture.responsibilities(mu)[0].cpu().numpy())
    Z = np.stack(mus)
    R = np.stack(resp) if resp else None
    return Z, R


def run_model(kind, videos, cohort_ids, subjects, phenotype, n_cohorts,
              epochs=4):
    print(f"\n=== {kind} ===")
    common = dict(architecture="conv", clip_length=60, n_joints=22,
                  latent_dim=16, batch_size=32, n_epochs=epochs,
                  warmup_epochs=2, beta_max=0.02, learning_rate=1e-3,
                  device="cpu", log_every=0, save_every=0,
                  out_dir=f"/tmp/gm_smoke_{kind}")
    if kind == "VAE":
        cfg = TrainingConfig(**common)
        cpv = None
    elif kind == "CVAE":
        cfg = TrainingConfig(**common, n_cond=n_cohorts, cond_dropout=0.15)
        cpv = cohort_ids
    elif kind == "GM-VAE":
        cfg = TrainingConfig(**common, n_components=4, gm_beta_z=0.3,
                             gm_beta_y=0.1, gm_entropy_weight=1.0,
                             gm_entropy_epochs=3)
        cpv = None
    elif kind == "GM-CVAE":
        cfg = TrainingConfig(**common, n_cond=n_cohorts, cond_dropout=0.15,
                             n_components=4, gm_beta_z=0.3, gm_beta_y=0.1,
                             gm_entropy_weight=1.0, gm_entropy_epochs=3)
        cpv = cohort_ids
    else:
        raise ValueError(kind)

    result = train(cfg, videos, stride=30, cohort_per_video=cpv)
    Z, R = encode_latents(result, videos, cohort_ids)

    # §11.1 site probe.
    probe = metrics.site_probe(Z, cohort_ids, seed=0)
    print(f"  site probe top1={probe['top1']:.3f} (chance {probe['chance']:.3f})")
    # §11.3 linear probe of the planted phenotype.
    lp = metrics.linear_probe(Z, phenotype, task="classification", seed=0)
    print(f"  phenotype linear probe {lp['metric']}={lp['score']:.3f}")
    # §11.2 post-hoc clustering vs phenotype.
    km = metrics.kmeans_labels(Z, k=2, seed=0)
    agree = metrics.cluster_label_agreement(km, phenotype)
    print(f"  kmeans vs phenotype ARI={agree['ari']:.3f} NMI={agree['nmi']:.3f}")
    # §11.2 native clusters + §10 occupancy for GM models.
    if R is not None:
        native = R.argmax(axis=1)
        nat = metrics.cluster_label_agreement(native, phenotype)
        print(f"  native  vs phenotype ARI={nat['ari']:.3f} NMI={nat['nmi']:.3f}")
        print(f"  occupancy {np.round(metrics.occupancy(R), 3)}")
    return result


def main():
    data = synthetic_cohorts()
    for kind in ["VAE", "CVAE", "GM-VAE", "GM-CVAE"]:
        run_model(kind, *data)
    print("\n=== every CARE-PD model path + metric ran ===")


if __name__ == "__main__":
    main()
