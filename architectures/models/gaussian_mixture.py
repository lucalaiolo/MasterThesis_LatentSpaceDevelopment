"""Gaussian-mixture prior for the GM-VAE / GM-CVAE ([CARE-PD §7.3]).

.. deprecated:: post-hoc plan §0
    The GM-VAE / GM-CVAE are **off the active pipeline**. In this setting
    the mixture prior suffers component collapse: when the latent is not
    cleanly multimodal the prior has nothing to latch onto and dumps its
    mass onto one or two components. The thesis claim is representational
    and does not need clustering baked into the prior — nuisance removal is
    measured by the site probe (VAE vs CVAE) and phenotype recovery by
    clustering the plain VAE / CVAE latent **post hoc** (see
    ``vae_analysis.posthoc``). This module is retained for the record only;
    it is not imported by any default run, and ``build_mixture`` /
    ``train`` refuse to instantiate it unless
    ``TrainingConfig(allow_deprecated_gmvae=True)`` is set explicitly. Do
    not use it downstream.

The mixture replaces the standard N(0, I) prior with

    p(y) = Cat(pi),   p(z | y) = N(mu_y, diag(sigma_y^2)),

so the aggregate posterior is pushed to organise into K modes rather than
one blob. This module holds the mixture parameters and the closed-form
quantities the training loop needs; it is deliberately *not* an
``nn.Module`` full of learnable weights.

Training strategy — EM-inspired block-coordinate descent ([GM-VAE §3.3,
Alg. 1]). Unlike the amortised categorical-head formulation, the mixture
here has no dedicated ``q(y|x)`` network. Instead, following Fan et al.,
the soft assignment (responsibility) of a latent point is the exact
posterior under the current mixture,

    gamma_{n,c} = p(c | z_n) = softmax_c( log pi_c + log N(z_n | mu_c, sigma_c^2) ),

and the mixture parameters (pi, mu, sigma^2) are refreshed by EM moment
updates rather than gradient descent. One training epoch therefore does:

    1. gradient steps on the encoder/decoder with the mixture *frozen*
       (the "maximization" of the networks), and
    2. EM steps on (pi, mu, sigma^2) with the networks *frozen* (the
       "expectation" + moment update).

The two blocks stop competing, which is the stability argument of the
paper. The parameters live in buffers so they move with ``.to(device)``,
checkpoint with ``state_dict``, but never receive a gradient.

Notation matches the plan and the paper: ``mu, logvar`` are the encoder's
per-sample posterior parameters q(z|x) = N(mu, sigma^2); ``means``,
``logvars`` are the K component parameters; ``pi`` the mixture weights.
"""

from __future__ import annotations

import math

from .common import torch, nn


_LOG_2PI = math.log(2.0 * math.pi)


class GaussianMixturePrior(nn.Module):
    """A K-component diagonal-Gaussian prior over the latent space.

    Two training regimes, selected by ``trainable``:

    - ``trainable=True`` (default) — the **regular / VaDE** regime the
      plan's §7.3 describes: the mixture parameters are ``nn.Parameter``s
      optimised by gradient descent jointly with the ELBO. Robust, no EM
      alternation, and what to reach for when the EM scheme is unstable.
    - ``trainable=False`` — the **EM** regime of [GM-VAE Alg. 1]: the
      parameters are buffers set in closed form by :meth:`em_update`;
      gradients never touch them.

    State (parameters or buffers depending on the regime):

        means:   (K, d_z)   component means mu_c
        logvars: (K, d_z)   component log-variances log sigma_c^2
        weights (K,)        mixture weights pi, on the simplex. In the
                            trainable regime these are a softmax of a free
                            ``weight_logits`` parameter; in the EM regime a
                            ``pi`` buffer. Read both through :meth:`weights`.
    """

    def __init__(self, n_components: int, d_z: int,
                 var_floor: float = 1e-4, init_spread: float = 1.0,
                 seed: int | None = 0, trainable: bool = True):
        super().__init__()
        if n_components < 2:
            raise ValueError("GaussianMixturePrior needs at least 2 components.")
        self.K = n_components
        self.d_z = d_z
        self.var_floor = float(var_floor)
        self.trainable = trainable

        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        # Scatter the means so components start distinguishable; unit
        # component variances; uniform weights ([CARE-PD §10] — start pi
        # at 1/K to guard against early component collapse).
        means_init = init_spread * torch.randn(n_components, d_z, generator=gen)
        logvars_init = torch.zeros(n_components, d_z)
        if trainable:
            # Gradient-trained parameters; weights via unconstrained logits
            # so a softmax keeps them on the simplex under any gradient.
            self.means = nn.Parameter(means_init)
            self.logvars = nn.Parameter(logvars_init)
            self.weight_logits = nn.Parameter(torch.zeros(n_components))
        else:
            # EM-updated buffers.
            self.register_buffer("means", means_init)
            self.register_buffer("logvars", logvars_init)
            self.register_buffer("pi", torch.full((n_components,),
                                                  1.0 / n_components))

    def weights(self):
        """Mixture weights pi on the simplex, whichever regime is active."""
        if self.trainable:
            return torch.softmax(self.weight_logits, dim=0)
        return self.pi

    def component_logvars(self):
        """Component log-variances, floored so sigma_c^2 >= var_floor.

        Guards the gradient regime, where an unconstrained ``logvars``
        parameter could otherwise collapse a component's variance to zero
        and blow up the density. A no-op for the EM regime, whose M-step
        already clamps to the floor.
        """
        return self.logvars.clamp_min(math.log(self.var_floor))

    # ---- Responsibilities -------------------------------------------------
    def component_log_prob(self, z):
        """Per-component log density log N(z | mu_c, sigma_c^2).

        Args:
            z: (B, d_z) latent points (posterior samples or means).
        Returns:
            (B, K) log densities.
        """
        # (B, 1, d_z) against (1, K, d_z).
        z = z.unsqueeze(1)
        means = self.means.unsqueeze(0)
        logvars = self.component_logvars().unsqueeze(0)
        var = logvars.exp()
        # log N = -1/2 sum_j [ log(2pi) + logvar_j + (z_j - mu_j)^2 / var_j ].
        quad = (z - means).pow(2) / var
        log_prob = -0.5 * (_LOG_2PI + logvars + quad).sum(dim=-1)
        return log_prob                                   # (B, K)

    def responsibilities(self, z):
        """gamma_{n,c} = p(c | z_n) under the current mixture ([GM-VAE §3.3]).

        Args:
            z: (B, d_z).
        Returns:
            (B, K) soft assignments, rows sum to 1.
        """
        log_pi = torch.log(self.weights().clamp_min(1e-12)).unsqueeze(0)  # (1, K)
        logits = log_pi + self.component_log_prob(z)               # (B, K)
        return torch.softmax(logits, dim=-1)

    # ---- Closed-form KL terms of the ELBO --------------------------------
    def kl_z_given_y(self, mu, logvar, resp):
        """E_{q(y|x)}[ KL( q(z|x) || p(z|y) ) ], per sample ([CARE-PD §7.3]).

        For a diagonal Gaussian posterior and component,

            KL( N(mu, s^2) || N(mu_c, s_c^2) )
              = 1/2 sum_j [ log(s_c,j^2 / s_j^2)
                            + (s_j^2 + (mu_j - mu_c,j)^2) / s_c,j^2 - 1 ],

        then averaged over components with weights gamma. This is exactly
        the mixture cross-entropy piece of the paper's per-example ELBO
        (Appendix A), re-expressed as a weighted KL.

        Args:
            mu, logvar: (B, d_z) encoder posterior parameters.
            resp: (B, K) responsibilities gamma.
        Returns:
            (B,) per-sample expected KL.
        """
        var = logvar.exp().unsqueeze(1)                   # (B, 1, d_z)
        mu = mu.unsqueeze(1)                              # (B, 1, d_z)
        c_logvar = self.component_logvars().unsqueeze(0)  # (1, K, d_z)
        c_var = c_logvar.exp()
        c_means = self.means.unsqueeze(0)                 # (1, K, d_z)

        # KL per (sample, component), summed over latent dims.
        kl = 0.5 * (
            c_logvar - logvar.unsqueeze(1)
            + (var + (mu - c_means).pow(2)) / c_var
            - 1.0
        ).sum(dim=-1)                                     # (B, K)
        return (resp * kl).sum(dim=-1)                    # (B,)

    def kl_y(self, resp):
        """KL( q(y|x) || p(y) ) = sum_c gamma_c ( log gamma_c - log pi_c ).

        Args:
            resp: (B, K).
        Returns:
            (B,) per-sample categorical KL (>= 0).
        """
        log_pi = torch.log(self.weights().clamp_min(1e-12)).unsqueeze(0)
        log_resp = torch.log(resp.clamp_min(1e-12))
        return (resp * (log_resp - log_pi)).sum(dim=-1)

    @staticmethod
    def assignment_entropy(resp):
        """H(q(y|x)) per sample, used for the entropy-warmup bonus."""
        log_resp = torch.log(resp.clamp_min(1e-12))
        return -(resp * log_resp).sum(dim=-1)             # (B,)

    # ---- EM update of the mixture parameters ------------------------------
    @torch.no_grad()
    def em_update(self, mu, logvar, n_steps: int = 1):
        """Refresh (pi, mu, sigma^2) by EM over cached epoch latents.

        Implements the E-step (responsibilities) and M-step (moment
        updates) of [GM-VAE Alg. 1]. The variance M-step uses the encoder
        posterior mean *and* variance,

            sigma_c^2 = sum_n gamma_{n,c} [ (mu_n - mu_c)^2 + s_n^2 ]
                        / sum_n gamma_{n,c},

        so the component width absorbs the posterior spread, not just the
        scatter of the means. Responsibilities are computed from the
        posterior means ``mu`` (a stable, deterministic proxy for the
        sampled z of the paper).

        Args:
            mu:     (N, d_z) cached posterior means for the training set.
            logvar: (N, d_z) cached posterior log-variances.
            n_steps: number of EM iterations (the N_EM inner loop).
        Returns:
            (K,) occupancy rho_c = mean_n gamma_{n,c} after the last step.
        """
        if self.trainable:
            raise RuntimeError(
                "em_update is for the EM regime (trainable=False); in the "
                "gradient regime the mixture is optimised by the optimiser."
            )
        var = logvar.exp()                                # (N, d_z)
        rho = self.pi.clone()
        for _ in range(max(1, n_steps)):
            resp = self.responsibilities(mu)              # (N, K)
            nk = resp.sum(dim=0)                          # (K,)
            nk_safe = nk.clamp_min(1e-8)

            # Weighted first and second moments.
            s1 = resp.t() @ mu                            # (K, d_z)
            new_means = s1 / nk_safe.unsqueeze(1)

            # E[(mu - mu_c)^2 + s^2] = E[mu^2 + s^2] - mu_c^2.
            s2 = resp.t() @ (mu.pow(2) + var)             # (K, d_z)
            new_vars = s2 / nk_safe.unsqueeze(1) - new_means.pow(2)
            new_vars = new_vars.clamp_min(self.var_floor)

            new_pi = nk / nk.sum().clamp_min(1e-8)

            # Only move components that actually own mass; leave a
            # momentarily empty component where it is instead of sending
            # its mean to 0 / its variance to the floor ([CARE-PD §10]).
            owned = (nk > 1e-6).unsqueeze(1)
            self.means = torch.where(owned, new_means, self.means)
            self.logvars = torch.where(owned, new_vars.log(), self.logvars)
            self.pi = new_pi
            rho = new_pi
        return rho

    @torch.no_grad()
    def init_from_latents(self, mu):
        """Seed the component means by k-means++-style spread over ``mu``.

        A short warm start ([GM-VAE §6], "brief pre-training phase")
        places the initial means on actual data rather than random noise —
        which helps both regimes: EM convergence, and (crucially) the
        gradient regime, where a VaDE run that starts its GMM from the
        pre-trained autoencoder's latents clusters far better than one
        started from noise. Falls back silently to the current init if
        there are fewer points than components.

        Works for both regimes: assigns through ``.data`` for the
        gradient-trained parameters and directly for the EM buffers.

        Args:
            mu: (N, d_z) posterior means from a warm-up pass.
        """
        n = mu.shape[0]
        if n < self.K:
            return
        # Greedy farthest-point seeding (k-means++ without the sampling).
        idx = [int(torch.randint(0, n, (1,)).item())]
        for _ in range(1, self.K):
            chosen = mu[idx]                              # (m, d_z)
            d2 = torch.cdist(mu, chosen).pow(2).min(dim=1).values
            idx.append(int(torch.argmax(d2).item()))
        new_means = mu[idx].clone()
        if self.trainable:
            self.means.data.copy_(new_means)
            self.logvars.data.zero_()
            self.weight_logits.data.zero_()
        else:
            self.means = new_means
            self.logvars = torch.zeros_like(self.logvars)
            self.pi = torch.full_like(self.pi, 1.0 / self.K)
