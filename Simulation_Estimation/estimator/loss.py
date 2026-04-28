"""
Loss function: differentiable forward pass + MSE on log(OGS) + regularisation.

Observation model (log scale):
    log(OGS_hat) = log(ES) + log(Bias) + log(Emot)
    log(OGS_obs) ≈ log(OGS_hat) + eps_OGS,   eps_OGS ~ N(0, sigma_OGS^2)

Primary loss:
    L_obs = MSE(log(OGS_obs), log(OGS_hat))  over all valid (i, t, m)

Regularisation:
    L_reg = lambda-weighted L2 on all deviation parameters

Total:
    L = L_obs + L_reg

Forward pass details
--------------------
The forward pass unrolls the session dynamics sequentially (T steps) because
C_ik^{t+1} depends on gains at session t, which in turn depend on C_ik^t.

For each session t:
    1. Gather per-game estimated parameters for all (i, m) simultaneously
    2. Compute delta[i, m] = C_current[i, k[i,m]] - d[i, t, m]
    3. Compute ES, Bias, Emot → OGS_hat[i, t, m]
    4. Compute Z, Psi, E, F → pi_hat[i, t, m]
    5. Accumulate Pi_hat[i, k] = sum_m pi_hat per domain using scatter_add
    6. Update Q_current, C_current via dynamics equations

Observed A and V (A_obs, V_obs from dataset) are used directly for Emot —
consistent with the design assumption that arousal/valence are measurable
quantities (survey / FER), not reconstructed from latent u.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

import core
from simulator import SimulationDataset
from estimator.parameters import ModelParameters


# ---------------------------------------------------------------------------
# Precompute flat game indices from dataset
# ---------------------------------------------------------------------------

def _flat_indices(dataset: SimulationDataset, Nk: list[int]) -> Tensor:
    """
    Precompute flat game index for every (i, t, m) from dataset.game_j/k.

    flat_games[i, t, m] = cumsum(Nk)[:k[i,t,m]] + j[i,t,m]

    Returns shape [I, T, M], dtype long.
    """
    cumsum = torch.tensor(
        [sum(Nk[:k]) for k in range(len(Nk))],
        dtype=torch.long,
    )
    flat = cumsum[dataset.game_k] + dataset.game_j   # [I, T, M], long
    return flat


# ---------------------------------------------------------------------------
# Batched forward pass
# ---------------------------------------------------------------------------

def forward_pass(
    params:  ModelParameters,
    dataset: SimulationDataset,
) -> Tensor:
    """
    Run the differentiable forward pass and return predicted OGS, shape [I, T, M].

    All operations are differentiable w.r.t. params.

    The dynamics are unrolled sequentially over T sessions.
    Within each session, all I participants and M games are batched together.

    Parameters
    ----------
    params  : ModelParameters (nn.Module with all trainable parameters)
    dataset : SimulationDataset (observed data; A_obs, V_obs, d, n_before used)

    Returns
    -------
    ogs_hat : Tensor shape [I, T, M], predicted OGS (before noise)
    """
    I, T, M = dataset.ogs.shape
    K = params.K
    J = params.J

    # --- Pre-fetch constrained parameters ---
    zeta_jk  = params.zeta_jk   # [J]
    beta0_jk = params.beta0_jk  # [J]
    beta1_jk = params.beta1_jk  # [J]
    u_all    = params.u_ijk      # [I, J]
    A_star   = params.A_star     # [I]
    eta_k    = params.eta_k      # [K]
    Bk_max   = params.Bk_max     # [K]

    # --- Precompute flat game indices [I, T, M] ---
    flat_games = _flat_indices(dataset, params.gs.Nk)   # [I, T, M]

    # --- Index game params to [I, T, M] ---
    zeta_itm  = zeta_jk[ flat_games]    # [I, T, M]
    beta0_itm = beta0_jk[flat_games]    # [I, T, M]
    beta1_itm = beta1_jk[flat_games]    # [I, T, M]

    # --- Index individual-game utility to [I, T, M] ---
    i_idx  = torch.arange(I).view(I, 1, 1).expand(I, T, M)   # [I, T, M]
    u_itm  = u_all[i_idx, flat_games]                          # [I, T, M]

    # --- A_star expanded for Emot computation [I, T, M] ---
    A_star_itm = A_star.view(I, 1, 1).expand(I, T, M)        # [I, T, M]

    # --- Bk_max indexed per game's domain [I, T, M] ---
    Bk_max_itm = Bk_max[dataset.game_k]   # [I, T, M]

    # --- Static data slices ---
    d_all      = dataset.d        # [I, T, M]
    n_all      = dataset.n_before # [I, T, M]
    A_obs_all  = dataset.A_obs    # [I, T, M]
    V_obs_all  = dataset.V_obs    # [I, T, M]
    k_all      = dataset.game_k   # [I, T, M]  domain index per game

    # --- Fatigue position tensor: 1-indexed [M] ---
    m_1indexed = torch.arange(1, M + 1, dtype=torch.float32)  # [M]

    # --- Initialise dynamics state ---
    C_current = params.C_init.clone()    # [I, K]
    Q_current = torch.zeros(I, K)        # [I, K]

    ogs_hat_list = []

    for t in range(T):
        k_t       = k_all[:, t, :]       # [I, M]  domain index
        d_t       = d_all[:, t, :]       # [I, M]
        n_t       = n_all[:, t, :]       # [I, M]
        A_obs_t   = A_obs_all[:, t, :]   # [I, M]
        V_obs_t   = V_obs_all[:, t, :]   # [I, M]
        zeta_t    = zeta_itm[:, t, :]    # [I, M]
        beta0_t   = beta0_itm[:, t, :]   # [I, M]
        beta1_t   = beta1_itm[:, t, :]   # [I, M]
        u_t       = u_itm[:, t, :]       # [I, M]
        Bk_max_t  = Bk_max_itm[:, t, :] # [I, M]
        A_star_t  = A_star_itm[:, t, :]  # [I, M]

        # delta[i, m] = C_ik^t - d  (gather domain ability for each game)
        C_ik  = C_current.gather(1, k_t)   # [I, M]
        delta = C_ik - d_t                  # [I, M]

        # Expected Score
        ES = core.expected_score(delta, params.gamma0, params.gamma1)   # [I, M]

        # Practice Bias  (Bk_min fixed at 0)
        Bias = core.practice_bias(
            n_t, delta,
            beta0_t, beta1_t,
            torch.zeros_like(Bk_max_t),   # Bk_min = 0
            Bk_max_t,
            params.kappa_B, params.delta_B_star,
        )   # [I, M]

        # Emotional Regulator  (uses observed A, V directly)
        Emot = core.emotional_regulator(
            A_obs_t, V_obs_t, A_star_t,
            params.E_min, params.kappa_A, params.kappa_V,
            params.alpha_A, params.alpha_V,
        )   # [I, M]

        # Predicted OGS (before noise)
        OGS_hat_t = core.obs_game_score_mean(ES, Bias, Emot)   # [I, M]
        OGS_hat_t = torch.clamp(OGS_hat_t, min=1e-3, max=100.0)  # match simulator clipping
        ogs_hat_list.append(OGS_hat_t)

        # --- Dynamics: compute gain and propagate state ---
        Z   = core.game_effectiveness(n_t, zeta_t, params.zeta_min, params.lambda_Z)
        Psi = core.mismatch_effect(delta, params.delta_star, params.kappa_L, params.kappa_R)
        E   = core.game_engagement(u_t, params.u_min, params.alpha_u)

        # Fatigue: broadcast [M] → [I, M] using 1-indexed positions
        m_1indexed_t = m_1indexed.unsqueeze(0).expand(I, M)   # [I, M]
        F_t = core.fatigue_cost(
            m_1indexed_t, M, delta, params.delta_star,
            params.rho0_F, params.rho_h_F, params.rho_e_F, params.tau_F,
        )   # [I, M]

        pi_t = core.single_game_gain(Z, Psi, E, F_t)   # [I, M]

        # Sum pi per domain via scatter_add: Pi[i, k] += pi[i, m]
        Pi_t = torch.zeros(I, K)
        Pi_t.scatter_add_(1, k_t, pi_t)   # [I, K]

        # Dynamics update
        Q_new = core.update_cumulative_impact(Q_current, Pi_t, params.rho_Q)
        C_new = core.update_ability(C_current, Q_new, eta_k.unsqueeze(0))

        Q_current = Q_new
        C_current = C_new

    # Stack along time dimension: [I, T, M]
    ogs_hat = torch.stack(ogs_hat_list, dim=1)
    return ogs_hat


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(
    params:       ModelParameters,
    dataset:      SimulationDataset,
    reg_lambdas:  dict[str, float] | None = None,
    reg_scale:    float = 1.0,
    gamma_prior:  dict | None = None,
    eps:          float = 1e-3,
) -> tuple[Tensor, dict[str, float]]:
    """
    Compute the total loss: MSE on log(OGS) + L2 regularisation + gamma prior.

    The regularisation is normalised by N_obs = I*T*M so that it is
    on the same scale as the mean observation loss (MAP-consistent).
    Without this, the sum-based regularisation dominates the mean-based
    observation loss by ~60,000x, freezing all individual deviations at zero.

    Parameters
    ----------
    params      : ModelParameters (nn.Module)
    dataset     : SimulationDataset (observed data)
    reg_lambdas : override regularisation strengths (default: from params._lambdas)
    reg_scale   : additional multiplier on regularisation (1.0 = MAP-consistent)
    gamma_prior : optional dict with 'gamma0_anchor', 'gamma1_anchor', 'lambda_gamma'
                  for informative prior on ES scale parameters (un-normalised penalty)
    eps         : small floor added inside log to prevent log(0)

    Returns
    -------
    loss       : scalar Tensor (total loss, differentiable)
    loss_info  : dict with 'obs', 'reg', 'gamma', 'total' (Python floats for logging)
    """
    # --- Forward pass ---
    ogs_hat = forward_pass(params, dataset)   # [I, T, M]

    # --- Log-scale MSE ---
    log_obs = torch.log(dataset.ogs.clamp(min=eps))     # [I, T, M]
    log_hat = torch.log(ogs_hat.clamp(min=eps))          # [I, T, M]
    L_obs   = F.mse_loss(log_hat, log_obs)               # scalar

    # --- Regularisation (normalised by N_obs for MAP consistency) ---
    L_reg_raw = params.regularization_loss(reg_lambdas)  # scalar (sum over params)
    N_obs     = dataset.ogs.numel()                      # I * T * M
    L_reg     = reg_scale * L_reg_raw / N_obs            # normalised to per-obs scale

    # --- Gamma prior (informative, un-normalised) ---
    # Penalises drift of gamma0 from warm-started anchor value.
    # gamma0 warm-start is reliable (based on mean logit(OGS/Emot));
    # gamma1 is NOT anchored because the adaptive staircase leaves near-zero
    # d variance in early sessions, making gamma1 warm-start unreliable.
    # With lambda_gamma=0.5 and L_obs≈0.02:
    #   drift of 0.2 → penalty ≈ 0.02 (1× L_obs)
    #   drift of 0.5 → penalty ≈ 0.125 (strong deterrent)
    L_gamma = torch.tensor(0.0)
    if gamma_prior is not None:
        g0_anchor = gamma_prior['gamma0_anchor']
        lam_g0 = gamma_prior['lambda_gamma0']
        lam_g1 = gamma_prior.get('lambda_gamma1', 0.0)
        L_gamma = lam_g0 * (params.gamma0 - g0_anchor) ** 2
        if lam_g1 > 0:
            log_g1_anchor = math.log(max(gamma_prior['gamma1_anchor'], 1e-6))
            L_gamma = L_gamma + lam_g1 * (params._log_gamma1 - log_g1_anchor) ** 2

    total = L_obs + L_reg + L_gamma

    loss_info = {
        'obs':   float(L_obs.item()),
        'reg':   float(L_reg.item()),
        'gamma': float(L_gamma.item()) if isinstance(L_gamma, Tensor) else 0.0,
        'total': float(total.item()),
    }

    return total, loss_info
