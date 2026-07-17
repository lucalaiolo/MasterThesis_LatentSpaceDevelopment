"""Baselines for the results chapter ([plan §7]).

All baselines produce a single per-clip latent that is evaluated with the
*same* subject-split probes (§6.3/§6.4) and clustering (§6.6/§6.7) as the
disentangled model, so the comparison is apples-to-apples:

- **PCA** at matched dimensionality (linear, no learning).
- **Plain VAE** — a compact continuous conv VAE, no disentanglement, no RVQ
  (probes on its posterior mean ``mu``).
- **GAITGen_w/o-dis** — the disentangled model trained with the classifier
  and adversarial losses switched off (``lambda_cls = lambda_adv = 0``),
  i.e. a conditional RVQ-VAE with no disentanglement pressure. Build it by
  training the main model on a config with those weights zeroed and pass its
  combined latent ``q_m + q_p`` here.

``evaluate_latent`` is the shared scorer.
"""

from __future__ import annotations

import numpy as np

from . import probes as prb, clustering as clu


# ---- Shared evaluation -----------------------------------------------------

def evaluate_latent(z: np.ndarray, c_p: np.ndarray, subjects: np.ndarray,
                    name: str, seed: int = 0) -> dict:
    """Probe + cluster a single latent against ``c_p`` ([plan §7])."""
    z = np.asarray(z, dtype=np.float64)
    pr = prb.probe(z, c_p, subjects, seed=seed)
    cl = clu.analyze(z, c_p, subjects, name=name, seed=seed)
    return {"name": name, "dim": int(z.shape[1]),
            "probe_cp": pr,
            "cluster_ari": cl["agreement"].get("kmeans", {}).get("ari"),
            "cluster_nmi": cl["agreement"].get("kmeans", {}).get("nmi"),
            "cluster_k": cl["k"],
            "subject_pure": cl["composition"]["pure_fraction"]}


# ---- PCA baseline ----------------------------------------------------------

def pca_latent(data, dim: int, pool: str = "mean") -> np.ndarray:
    """Matched-dimensionality PCA latent per clip ([plan §7]).

    ``pool="mean"`` averages each clip over time before PCA (so the latent is
    a per-clip pose-distribution summary); ``pool="flatten"`` uses the whole
    clip. Returns (N, dim).
    """
    from sklearn.decomposition import PCA
    x = np.asarray(data.x, dtype=np.float64)          # (N, T, J*2)
    feats = x.mean(axis=1) if pool == "mean" else x.reshape(len(x), -1)
    k = min(dim, feats.shape[1], len(feats))
    return PCA(n_components=k, random_state=0).fit_transform(feats)


# ---- Plain continuous VAE baseline ----------------------------------------

class PlainVAEBaseline:
    """A compact continuous conv-VAE: no disentanglement, no RVQ ([plan §7]).

    Reuses the encoder/decoder conv stacks so reconstruction capacity is
    comparable; the latent is a single continuous ``mu`` per clip.
    """

    def __init__(self, config, latent_dim: int | None = None):
        import torch
        from torch import nn
        from ..models.networks import _ConvStack, Decoder
        self.torch = torch
        self.cfg = config
        d = latent_dim or (config.d_motion + config.d_pathology)
        self.d = d

        class _VAE(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = _ConvStack(config.input_dim, config.hidden_channels,
                                   2 * d, config.downsample)   # mu + logvar
                s.dec = Decoder(config.input_dim, config.hidden_channels, d,
                                config.downsample)

            def forward(s, x):
                h = s.enc(x.transpose(1, 2)).transpose(1, 2)    # (B, T', 2d)
                mu, logvar = h[..., :d], h[..., d:]
                z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                x_hat = s.dec(z)
                return x_hat, mu, logvar

        self.net = _VAE()

    def train(self, data, epochs=50, lr=1e-3, beta=1e-2, batch=64,
              device="cpu", seed=0):
        torch = self.torch
        from ..preprocess import make_loader
        torch.manual_seed(seed)
        net = self.net.to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loader = make_loader(data, None, batch, shuffle=True, seed=seed)
        for ep in range(epochs):
            net.train()
            for x, _cp, _cn, _s in loader:
                x = x.to(device)
                x_hat, mu, logvar = net(x)
                rec = (x - x_hat).abs().mean()
                kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
                loss = rec + beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def encode(self, data, device="cpu", batch=256) -> np.ndarray:
        torch = self.torch
        self.net.eval().to(device)
        out = []
        with torch.no_grad():
            for i in range(0, data.n, batch):
                x = torch.from_numpy(data.x[i:i + batch]).float().to(device)
                h = self.net.enc(x.transpose(1, 2)).transpose(1, 2)
                out.append(h[..., :self.d].mean(1).cpu().numpy())   # mean-pooled mu
        return np.concatenate(out)
