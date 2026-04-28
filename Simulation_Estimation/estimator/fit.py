"""
fit.py — Main optimisation loop for the PCIP estimator.

Fits all model parameters jointly via gradient descent (Adam) on the
log-scale OGS reconstruction loss + L2 regularisation.

Key features:
- Warm-start: C_init from early-session OGS; A_star from mean A_obs
- Two-phase training: Phase 1 (global + C_init); Phase 2 (all params)
- Stage 2: u_ijk estimated by inverting the Valence equation
- Stage 3: Short re-optimisation after u_ijk update
- Regularisation normalised by N_obs for MAP consistency (see loss.py)

Usage
-----
    result = fit(
        dataset        = sim_dataset,
        game_structure = game_pool.to_structure(),
        cfg            = cfg,
        n_epochs       = 1500,
        verbose        = True,
    )
    params = result['params']               # ModelParameters (trained)
    losses = result['loss_history']         # list[dict] per epoch

The estimator never receives the GamePool or ParticipantPool directly —
only the SimulationDataset (observations) and GameStructure (structural info).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

import torch
import torch.optim as optim

import core
from core.kernels import softplus_beta
from games import GameStructure
from simulator import SimulationDataset
from estimator.parameters import ModelParameters
from estimator.loss import compute_loss, _flat_indices


# ---------------------------------------------------------------------------
# Warm-start helpers
# ---------------------------------------------------------------------------

def _warm_start_C_init(
    params:     ModelParameters,
    dataset:    SimulationDataset,
    cfg:        dict,
    n_sessions: int = 3,
    gamma0:     float | None = None,
    gamma1:     float | None = None,
) -> torch.Tensor:
    """
    Estimate per-participant C_init from early-session OGS by inverting ES.

    Uses the first `n_sessions` sessions.  For each (participant, domain),
    collects all (OGS, d) pairs and solves:
        C ≈ d + (logit(OGS/100) − gamma0) / gamma1

    If gamma0/gamma1 are provided, they override config defaults (use with
    warm-started gamma for better C estimates).

    Returns C_est of shape [I, K].
    """
    gi = cfg['global_init']
    if gamma0 is None:
        gamma0 = float(gi['gamma0'])
    if gamma1 is None:
        gamma1 = float(gi['gamma1'])

    I, T, M = dataset.ogs.shape
    K = params.K
    n_use = min(n_sessions, T)

    # Early-session data  [I, n_use*M]
    ogs_early = dataset.ogs[:, :n_use, :].reshape(I, -1)
    d_early   = dataset.d[:, :n_use, :].reshape(I, -1)
    k_early   = dataset.game_k[:, :n_use, :].reshape(I, -1)

    # Invert ES:  C ≈ d + (logit(OGS/100) − γ₀) / γ₁
    ogs_frac   = torch.clamp(ogs_early / 100.0, 0.05, 0.95)
    C_hat_flat = d_early + (torch.logit(ogs_frac) - gamma0) / gamma1

    # Average per (i, k) via scatter_add
    C_sum = torch.zeros(I, K)
    count = torch.zeros(I, K)
    C_sum.scatter_add_(1, k_early, C_hat_flat)
    count.scatter_add_(1, k_early, torch.ones_like(C_hat_flat))

    count = count.clamp(min=1)
    C_est = torch.clamp(C_sum / count, 0.5, 9.5)

    return C_est


def _warm_start_A_star(
    dataset: SimulationDataset,
) -> torch.Tensor:
    """
    Estimate per-participant A_star from observed arousal.

    Simple approach: A_star_i ≈ mean(A_obs_i) across all sessions and games.
    Returns A_est of shape [I].
    """
    A_mean = dataset.A_obs.mean(dim=(1, 2))   # [I]
    return torch.clamp(A_mean, 0.05, 0.95)


# ---------------------------------------------------------------------------
# Warm-start gamma from early-session data
# ---------------------------------------------------------------------------

def _warm_start_gamma(
    dataset:    SimulationDataset,
    cfg:        dict,
    n_sessions: int = 3,
) -> tuple[float, float]:
    """
    Estimate gamma0 and gamma1 from early-session data where Bias ≈ 1.

    Identification strategy:
    - γ₁ is identified from the within-(i,k) slope of
      logit(OGS / (100·Emot)) vs difficulty d.  This uses variation in d
      across games within the same individual-domain group (shared C_ik),
      so it does NOT depend on knowing C.
    - γ₀ is identified from the overall mean of logit(OGS / (100·Emot)),
      assuming mean(δ) ≈ 0 in early sessions (staircase tracks ability).

    Emot is approximated using observed A/V and config default parameters.
    Bias is assumed ≈ 1 (n ≤ 2 in sessions 1–3; max Bias effect < 10%).

    Returns (gamma0_est, gamma1_default).  gamma1 is returned as the config
    default because the adaptive staircase creates near-zero d variance in
    early sessions, making the within-group slope unreliable.
    """
    gi = cfg['global_init']
    I, T, M = dataset.ogs.shape
    K = int(dataset.game_k.max().item()) + 1
    n_use = min(n_sessions, T)

    with torch.no_grad():
        # Approximate Emot using config defaults and rough A_star
        E_min_v   = torch.tensor(float(gi['E_min']))
        kappa_A_v = torch.tensor(float(gi['kappa_A']))
        kappa_V_v = torch.tensor(float(gi['kappa_V']))
        alpha_A_v = torch.tensor(float(gi['alpha_A']))
        alpha_V_v = torch.tensor(float(gi['alpha_V']))
        A_star_rough = dataset.A_obs.mean(dim=(1, 2))  # [I]

        # Collect early-session data into flat tensors
        y_parts = []

        for t in range(n_use):
            A_obs_t = dataset.A_obs[:, t, :]   # [I, M]
            V_obs_t = dataset.V_obs[:, t, :]   # [I, M]
            ogs_t   = dataset.ogs[:, t, :]     # [I, M]

            Emot_t = core.emotional_regulator(
                A_obs_t, V_obs_t,
                A_star_rough.unsqueeze(1).expand_as(A_obs_t),
                E_min_v, kappa_A_v, kappa_V_v, alpha_A_v, alpha_V_v,
            )

            # y = logit(OGS / (100 · Emot))  ≈  γ₀ + γ₁·(C_ik - d)
            ratio = torch.clamp(ogs_t / (100.0 * Emot_t.clamp(min=0.1)),
                                0.02, 0.98)
            y_t = torch.logit(ratio)  # [I, M]
            y_parts.append(y_t.reshape(-1))

        y_f = torch.cat(y_parts)  # [N]

        # --- Estimate γ₀ from overall mean ---
        # mean(y) = γ₀ + γ₁·mean(δ);  mean(δ) ≈ 0 in early sessions
        # (staircase starts at d ≈ mu_C, so C - d ≈ 0 on average)
        gamma0_est = float(y_f.mean().item())
        gamma0_est = max(-2.0, min(5.0, gamma0_est))  # sanity clamp

        # --- γ₁ is NOT warm-started ---
        # The adaptive staircase keeps d ≈ C, producing near-zero d variance
        # in early sessions (session 1: d std=0.0; sessions 1-3: d std≈0.15).
        # The within-group slope of y vs d is unreliable at this noise level
        # and biased by staircase endogeneity.  We keep γ₁ at the config
        # default (equivalent to an IRT calibration prior).
        gamma1_default = float(gi['gamma1'])

    return gamma0_est, gamma1_default


# ---------------------------------------------------------------------------
# Stage 2a: Arousal inversion for A_star
# ---------------------------------------------------------------------------

def _invert_arousal_for_A_star(
    params:  ModelParameters,
    dataset: SimulationDataset,
) -> torch.Tensor:
    """
    Estimate A_star_i by inverting the Emot equation from OGS residuals.

    Strategy:
    1. Unroll dynamics to compute ES and Bias at each (i,t,m)
    2. Emot_obs = OGS / (ES·Bias) captures the emotional regulator + noise
    3. Remove valence contribution to isolate the arousal factor:
           y = log(Emot_res) + (α_V/κ_V)·(V−1)²  =  −(α_A/κ_A)·(A − A*)²
       where Emot_res = (Emot_obs − E_min) / (1 − E_min)
    4. For each individual, fit quadratic  y ~ a·A² + b·A + c
       A*_i = −b/(2a)  (peak of the downward parabola)

    Falls back to weighted-mean(A_obs) if the quadratic has no peak (a ≥ 0).

    Returns A_star_est of shape [I].
    """
    I, T, M = dataset.ogs.shape
    K = params.K
    J = params.J

    flat_games = _flat_indices(dataset, params.gs.Nk)

    with torch.no_grad():
        zeta_jk  = params.zeta_jk.detach()
        beta0_jk = params.beta0_jk.detach()
        beta1_jk = params.beta1_jk.detach()
        u_all    = params.u_ijk.detach()
        eta_k    = params.eta_k.detach()
        Bk_max   = params.Bk_max.detach()

        i_idx      = torch.arange(I).view(I, 1, 1).expand(I, T, M)
        m_1indexed = torch.arange(1, M + 1, dtype=torch.float32) \
                          .unsqueeze(0).expand(I, M)

        # --- Unroll dynamics to get ES and Bias at each (i,t,m) ---
        C_current = params.C_init.clone()
        Q_current = torch.zeros(I, K)

        ES_list   = []
        Bias_list = []

        for t in range(T):
            k_t  = dataset.game_k[:, t, :]
            d_t  = dataset.d[:, t, :]
            n_t  = dataset.n_before[:, t, :]
            fg_t = flat_games[:, t, :]

            C_ik  = C_current.gather(1, k_t)
            delta = C_ik - d_t

            # ES
            ES_t = core.expected_score(
                delta, params.gamma0.item(), params.gamma1.item(),
            )

            # Bias
            beta0_t  = beta0_jk[fg_t]
            beta1_t  = beta1_jk[fg_t]
            Bk_max_t = Bk_max[k_t]
            Bias_t = core.practice_bias(
                n_t, delta, beta0_t, beta1_t,
                torch.zeros_like(Bk_max_t), Bk_max_t,
                params.kappa_B.item(), params.delta_B_star.item(),
            )

            ES_list.append(ES_t)
            Bias_list.append(Bias_t)

            # Advance dynamics
            zeta_t = zeta_jk[fg_t]
            u_t    = u_all[i_idx[:, t, :], fg_t]

            Z   = core.game_effectiveness(
                n_t, zeta_t, params.zeta_min, params.lambda_Z,
            )
            Psi = core.mismatch_effect(
                delta, params.delta_star, params.kappa_L, params.kappa_R,
            )
            E   = core.game_engagement(u_t, params.u_min, params.alpha_u)
            F_t = core.fatigue_cost(
                m_1indexed, M, delta, params.delta_star,
                params.rho0_F, params.rho_h_F, params.rho_e_F, params.tau_F,
            )
            pi_t = core.single_game_gain(Z, Psi, E, F_t)

            Pi_t = torch.zeros(I, K)
            Pi_t.scatter_add_(1, k_t, pi_t)

            Q_current = core.update_cumulative_impact(
                Q_current, Pi_t, params.rho_Q,
            )
            C_current = core.update_ability(
                C_current, Q_current, eta_k.unsqueeze(0),
            )

        # Stack: [I, T, M]
        ES_all   = torch.stack(ES_list, dim=1)
        Bias_all = torch.stack(Bias_list, dim=1)

        # --- Compute Emot residual ---
        OGS     = dataset.ogs
        ES_Bias = (ES_all * Bias_all).clamp(min=1e-3)
        Emot_obs = OGS / ES_Bias                          # [I, T, M]

        # --- Get estimated Emot global params ---
        E_min   = params.E_min.item()
        kappa_A = params.kappa_A.item()
        kappa_V = params.kappa_V.item()
        alpha_A = params.alpha_A.item()
        alpha_V = params.alpha_V.item()

        # --- Isolate arousal factor ---
        # Emot_res = (Emot - E_min) / (1 - E_min) = AF · VF
        Emot_res = (Emot_obs - E_min) / (1.0 - E_min + 1e-8)
        Emot_res = Emot_res.clamp(min=1e-6, max=2.0)

        # Remove valence factor:
        #   log(VF) = -alpha_V * (V - 1)^2 / kappa_V
        #   y = log(Emot_res) - log(VF)
        #     = log(AF) = -alpha_A * (A - A*)^2 / kappa_A
        V_obs = dataset.V_obs
        y = torch.log(Emot_res) + alpha_V * (V_obs - 1.0) ** 2 / kappa_V

        # --- Batched quadratic fit: y ~ a·A² + b·A + c ---
        A_obs  = dataset.A_obs
        A_flat = A_obs.reshape(I, -1)    # [I, N]
        y_flat = y.reshape(I, -1)        # [I, N]
        y_flat = y_flat.clamp(-5.0, 5.0) # remove extreme outliers

        # Design matrix: [I, N, 3]
        X = torch.stack([
            A_flat ** 2,
            A_flat,
            torch.ones_like(A_flat),
        ], dim=2)

        # Normal equations: (X^T X) coefs = X^T y  (batched)
        XtX  = torch.bmm(X.transpose(1, 2), X)                        # [I,3,3]
        Xty  = torch.bmm(
            X.transpose(1, 2), y_flat.unsqueeze(2),
        ).squeeze(2)                                                    # [I,3]

        coefs = torch.linalg.solve(XtX, Xty)                           # [I,3]
        a = coefs[:, 0]   # should be < 0
        b = coefs[:, 1]

        # A* = -b / (2a)  where a < 0
        A_star_quad = -b / (2.0 * a + 1e-8)

        # Weighted-mean fallback for individuals with a >= 0
        weights     = torch.exp(y_flat - y_flat.max(dim=1, keepdim=True)[0])
        A_star_wmean = (weights * A_flat).sum(dim=1) / weights.sum(dim=1)

        valid      = a < -1e-6
        A_star_est = torch.where(valid, A_star_quad, A_star_wmean)

    return A_star_est.clamp(0.05, 0.95)


# ---------------------------------------------------------------------------
# Stage 2b: Valence inversion for u_ijk
# ---------------------------------------------------------------------------

def _invert_valence_for_u(
    params:  ModelParameters,
    dataset: SimulationDataset,
    cfg:     dict,
) -> torch.Tensor:
    """
    Estimate u_ijk by inverting the Valence equation using estimated C and observed V.

    V = tanh([gamma0_V + gamma_Vu*(2u-1) - gamma_Vh*h^2 - gamma_Ve*e^2] / tau_V)

    Solving for u:
      u = 0.5 + [atanh(V)*tau_V - gamma0_V + gamma_Vh*h^2 + gamma_Ve*e^2] / (2*gamma_Vu)

    Returns u_est of shape [I, J_total].
    """
    vp = cfg['predefined']['valence']
    gamma0_V = vp['gamma0_V']
    gamma_Vu = vp['gamma_Vu']
    gamma_Vh = vp['gamma_Vh']
    gamma_Ve = vp['gamma_Ve']
    tau_V    = vp['tau_V']
    beta_Vh  = vp['beta_Vh']
    beta_Ve  = vp['beta_Ve']

    I, T, M = dataset.ogs.shape
    J = params.J
    K = params.K

    flat_games = _flat_indices(dataset, params.gs.Nk)   # [I, T, M]
    i_idx = torch.arange(I).view(I, 1, 1).expand(I, T, M)
    m_1indexed = torch.arange(1, M + 1, dtype=torch.float32).unsqueeze(0).expand(I, M)

    # Accumulate u estimates per (i, flat_game)
    u_sum   = torch.zeros(I, J)
    u_count = torch.zeros(I, J)

    with torch.no_grad():
        C_current = params.C_init.clone()
        Q_current = torch.zeros(I, K)
        eta_k   = params.eta_k.detach()
        zeta_jk = params.zeta_jk.detach()
        u_all   = params.u_ijk.detach()

        for t in range(T):
            k_t  = dataset.game_k[:, t, :]      # [I, M]
            d_t  = dataset.d[:, t, :]            # [I, M]
            V_t  = dataset.V_obs[:, t, :]        # [I, M]
            n_t  = dataset.n_before[:, t, :]     # [I, M]
            fg_t = flat_games[:, t, :]           # [I, M]

            C_ik  = C_current.gather(1, k_t)     # [I, M]
            delta = C_ik - d_t                    # [I, M]

            # Invert Valence for u
            h = softplus_beta(-delta, beta_Vh)
            e = softplus_beta( delta, beta_Ve)
            V_clamped = V_t.clamp(-0.999, 0.999)

            u_hat = 0.5 + (
                torch.atanh(V_clamped) * tau_V
                - gamma0_V
                + gamma_Vh * h ** 2
                + gamma_Ve * e ** 2
            ) / (2.0 * gamma_Vu)
            u_hat = u_hat.clamp(0.01, 0.99)

            u_sum.scatter_add_(1, fg_t, u_hat)
            u_count.scatter_add_(1, fg_t, torch.ones_like(u_hat))

            # Advance dynamics for next session's delta
            zeta_t = zeta_jk[fg_t]
            u_t    = u_all[i_idx[:, t, :], fg_t]

            Z   = core.game_effectiveness(n_t, zeta_t, params.zeta_min, params.lambda_Z)
            Psi = core.mismatch_effect(delta, params.delta_star, params.kappa_L, params.kappa_R)
            E   = core.game_engagement(u_t, params.u_min, params.alpha_u)
            F_t = core.fatigue_cost(m_1indexed, M, delta, params.delta_star,
                                    params.rho0_F, params.rho_h_F,
                                    params.rho_e_F, params.tau_F)
            pi_t = core.single_game_gain(Z, Psi, E, F_t)

            Pi_t = torch.zeros(I, K)
            Pi_t.scatter_add_(1, k_t, pi_t)

            Q_current = core.update_cumulative_impact(Q_current, Pi_t, params.rho_Q)
            C_current = core.update_ability(C_current, Q_current, eta_k.unsqueeze(0))

    # Average and fill unplayed games with prior
    u_count = u_count.clamp(min=1)
    u_est = u_sum / u_count
    never_played = (u_count < 0.5)
    u_est[never_played] = params.u_ijk.detach()[never_played]

    return u_est.clamp(0.01, 0.99)


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit(
    dataset:        SimulationDataset,
    game_structure: GameStructure,
    cfg:            dict,
    n_epochs:       int   = 1500,
    lr:             float = 1e-2,
    lr_decay:       float = 0.998,
    patience:       int   = 150,
    min_delta:      float = 1e-6,
    log_every:      int   = 100,
    verbose:        bool  = True,
    seed:           Optional[int] = None,
    warm_start:     bool  = True,
    phase1_frac:    float = 0.3,
    reg_scale:      float = 1.0,
    stage2_valence: bool  = True,      # run valence inversion for u_ijk
    stage2_arousal: bool  = True,      # run arousal inversion for A_star
    stage3_epochs:  int   = 200,       # re-optimisation after Stage 2 (0 = skip)
    lambda_gamma:   float = 0.5,       # informative prior on gamma0 (0 = disable)
    lambda_gamma1:  float = 0.1,       # weak prior on gamma1 at config default
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
) -> dict:
    """
    Fit PCIP model parameters to observed SimulationDataset.

    Parameters
    ----------
    dataset        : SimulationDataset from Simulator.run()
    game_structure : GameStructure from GamePool.to_structure()
    cfg            : full YAML config dict
    n_epochs       : maximum number of gradient steps (Phase 1 + Phase 2)
    lr             : initial Adam learning rate
    lr_decay       : multiplicative LR scheduler factor (per epoch)
    patience       : epochs without improvement before stopping (0 = off)
    min_delta      : minimum loss improvement to reset patience counter
    log_every      : logging interval (epochs)
    verbose        : print progress
    seed           : optional seed for parameter initialisation
    warm_start     : initialise C_init, A_star, and gamma0 from data
    phase1_frac    : fraction of epochs for Phase 1 (global + C_init only)
    reg_scale      : additional multiplier on regularisation (1.0 = MAP-consistent)
    stage2_valence : run Stage 2 valence inversion for u_ijk
    stage2_arousal : run Stage 2 arousal inversion for A_star
    stage3_epochs  : epochs for Stage 3 re-optimisation after u update (0 = skip)
    lambda_gamma   : informative prior strength on gamma0 (anchored at warm-start
                     value; 0 = disable)
    lambda_gamma1  : informative prior strength on gamma1 (anchored at config default;
                     0 = disable).  Default 0: gamma1 is freely estimated because
                     the adaptive staircase prevents reliable gamma1 warm-start
    progress_callback : optional callable(step_so_far, total_steps_planned, loss_info)
                     called once per epoch in both the Phase 1+2 main loop and
                     Stage 3 re-optimisation. step_so_far is 1-indexed cumulative;
                     total_steps_planned = n_epochs + (stage3_epochs if Stage 2 ran).

    Returns
    -------
    result : dict with keys:
        'params'        : ModelParameters — trained parameter object
        'loss_history'  : list[dict]      — per-epoch {'obs','reg','total'}
        'best_loss'     : float           — best total loss achieved
        'n_epochs_run'  : int             — actual epochs completed
    """
    if seed is not None:
        torch.manual_seed(seed)

    I = dataset.ogs.shape[0]

    # --- Initialise parameters ---
    params = ModelParameters(
        I=I,
        game_structure=game_structure,
        init_cfg=cfg,
        sim_cfg=cfg,
    )

    # --- Warm-start gamma from early-session data ---
    gamma_prior = None
    if warm_start:
        g0_est, g1_est = _warm_start_gamma(dataset, cfg)
        with torch.no_grad():
            params._gamma0.copy_(torch.tensor(g0_est))
            params._log_gamma1.copy_(torch.log(torch.tensor(g1_est)))
        if verbose:
            gi = cfg['global_init']
            print(
                f"Warm-started gamma: "
                f"gamma0={g0_est:.3f} (cfg={gi['gamma0']}), "
                f"gamma1={g1_est:.3f} (cfg={gi['gamma1']})"
            )

    if lambda_gamma > 0 or lambda_gamma1 > 0:
        if warm_start:
            anchor_g0 = g0_est
        else:
            anchor_g0 = float(cfg['global_init']['gamma0'])
        anchor_g1 = float(cfg['global_init']['gamma1'])  # always config default
        gamma_prior = {
            'gamma0_anchor': anchor_g0,
            'gamma1_anchor': anchor_g1,
            'lambda_gamma0': lambda_gamma,
            'lambda_gamma1': lambda_gamma1,
        }
        if verbose:
            parts = []
            if lambda_gamma > 0:
                parts.append(f"gamma0: anchor={anchor_g0:.3f}, lam={lambda_gamma}")
            if lambda_gamma1 > 0:
                parts.append(f"gamma1: anchor={anchor_g1:.3f}, lam={lambda_gamma1}")
            print(f"Gamma prior: {'; '.join(parts)}")

    # --- Warm-start C_init from early-session data ---
    if warm_start:
        C_est = _warm_start_C_init(
            params, dataset, cfg,
            gamma0=g0_est, gamma1=g1_est,
        )
        with torch.no_grad():
            mu_C_new = C_est.mean(dim=0)
            params.mu_C.copy_(mu_C_new)
            params.delta_C.copy_(C_est - mu_C_new.unsqueeze(0))
        if verbose:
            print(
                f"Warm-started C_init: "
                f"mean={C_est.mean():.2f}  std={C_est.std():.2f}  "
                f"range=[{C_est.min():.2f}, {C_est.max():.2f}]"
            )

    # --- Warm-start A_star from observed arousal ---
    if warm_start:
        A_est = _warm_start_A_star(dataset)
        with torch.no_grad():
            mu_A_new = torch.logit(A_est.mean().clamp(0.05, 0.95))
            params.mu_A_logit.copy_(mu_A_new)
            params.delta_A_logit.copy_(torch.logit(A_est) - mu_A_new)
        if verbose:
            print(
                f"Warm-started A_star: "
                f"mean={A_est.mean():.3f}  std={A_est.std():.3f}  "
                f"range=[{A_est.min():.3f}, {A_est.max():.3f}]"
            )

    # --- Phase 1: freeze individual deviations (except C_init) ---
    phase1_epochs = int(n_epochs * phase1_frac)
    _phase2_names = {
        'delta_zeta', 'delta_log_beta0', 'delta_log_beta1',
        'delta_u_logit', 'delta_A_logit',
    }

    if phase1_epochs > 0:
        for name, p in params.named_parameters():
            if name in _phase2_names:
                p.requires_grad_(False)
        if verbose:
            n_active = sum(p.numel() for p in params.parameters() if p.requires_grad)
            n_frozen = sum(p.numel() for p in params.parameters() if not p.requires_grad)
            print(f"Phase 1 (epochs 1-{phase1_epochs}): "
                  f"{n_active:,} active, {n_frozen:,} frozen params")

    # --- Optimiser + scheduler ---
    optimizer = optim.Adam(
        [p for p in params.parameters() if p.requires_grad], lr=lr
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # --- Training loop (Phase 1 + Phase 2) ---
    loss_history: list[dict] = []
    best_loss    = float('inf')
    best_state   = None
    patience_ctr = 0

    # Total step budget for progress reporting (Stage 3 may add more)
    _stage3_planned = stage3_epochs if (stage2_valence or stage2_arousal) else 0
    _total_steps    = n_epochs + _stage3_planned

    for epoch in range(1, n_epochs + 1):

        # --- Phase transition ---
        if epoch == phase1_epochs + 1 and phase1_epochs > 0:
            for p in params.parameters():
                p.requires_grad_(True)
            optimizer = optim.Adam(params.parameters(), lr=lr * 0.3)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
            patience_ctr = 0
            if verbose:
                n_total = sum(p.numel() for p in params.parameters())
                print(f"\n--- Phase 2 (epoch {epoch}): "
                      f"all {n_total:,} params unfrozen, lr={lr*0.3:.1e} ---\n")

        optimizer.zero_grad()
        loss, loss_info = compute_loss(params, dataset, reg_scale=reg_scale,
                                       gamma_prior=gamma_prior)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss_info)

        # --- Early stopping ---
        if loss_info['total'] < best_loss - min_delta:
            best_loss  = loss_info['total']
            best_state = deepcopy(params.state_dict())
            patience_ctr = 0
        elif patience > 0:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        # --- Logging ---
        if verbose and epoch % log_every == 0:
            print(
                f"Epoch {epoch:4d}/{n_epochs}  "
                f"loss={loss_info['total']:.5f}  "
                f"obs={loss_info['obs']:.5f}  "
                f"reg={loss_info['reg']:.5f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # --- Progress callback ---
        if progress_callback is not None:
            progress_callback(epoch, _total_steps, loss_info)

    # Restore best parameters from Phase 1+2
    if best_state is not None:
        params.load_state_dict(best_state)

    # =======================================================================
    # STAGE 2a: Valence inversion for u_ijk
    # =======================================================================
    if stage2_valence:
        if verbose:
            print("\n--- Stage 2a: Estimating u_ijk via valence inversion ---")

        u_est = _invert_valence_for_u(params, dataset, cfg)
        with torch.no_grad():
            mu_u_new = torch.logit(u_est.clamp(0.01, 0.99)).mean()
            params.mu_u_logit.copy_(mu_u_new)
            params.delta_u_logit.copy_(
                torch.logit(u_est.clamp(0.01, 0.99)) - mu_u_new
            )
        if verbose:
            print(
                f"  u_ijk: mean={u_est.mean():.3f}  std={u_est.std():.3f}  "
                f"range=[{u_est.min():.3f}, {u_est.max():.3f}]"
            )

    # =======================================================================
    # STAGE 2b: Arousal inversion for A_star
    # =======================================================================
    if stage2_arousal:
        if verbose:
            print("\n--- Stage 2b: Estimating A_star via arousal inversion ---")

        A_star_est = _invert_arousal_for_A_star(params, dataset)
        with torch.no_grad():
            mu_A_new = torch.logit(A_star_est.clamp(0.05, 0.95)).mean()
            params.mu_A_logit.copy_(mu_A_new)
            params.delta_A_logit.copy_(
                torch.logit(A_star_est.clamp(0.05, 0.95)) - mu_A_new
            )
        if verbose:
            print(
                f"  A_star: mean={A_star_est.mean():.3f}  "
                f"std={A_star_est.std():.3f}  "
                f"range=[{A_star_est.min():.3f}, {A_star_est.max():.3f}]"
            )

    # =======================================================================
    # STAGE 3: Short re-optimisation after Stage 2 inversions
    # =======================================================================
    if (stage2_valence or stage2_arousal) and stage3_epochs > 0:
        stage3_lr = lr * 0.1
        if verbose:
            print(f"\n--- Stage 3: Re-optimisation "
                  f"({stage3_epochs} epochs, lr={stage3_lr:.1e}) ---\n")

        # Build relaxed reg lambdas for Stage 3:
        # - u: set to 0 (valence inversion gives direct data-driven estimates)
        # - A: set to 0 (arousal inversion gives direct data-driven estimates)
        reg_lambdas_s3 = dict(params._lambdas)
        reg_lambdas_s3['u'] = 0.0
        reg_lambdas_s3['A'] = 0.0
        if verbose:
            print(f"  Relaxed reg: u λ={reg_lambdas_s3['u']:.2f} "
                  f"(was {params._lambdas['u']:.2f}), "
                  f"A λ={reg_lambdas_s3['A']:.2f} "
                  f"(was {params._lambdas['A']:.2f})")

        optimizer3 = optim.Adam(params.parameters(), lr=stage3_lr)
        scheduler3 = optim.lr_scheduler.ExponentialLR(optimizer3, gamma=lr_decay)
        best_loss3  = float('inf')
        best_state3 = None

        for epoch in range(1, stage3_epochs + 1):
            optimizer3.zero_grad()
            loss, loss_info = compute_loss(
                params, dataset,
                reg_lambdas=reg_lambdas_s3,
                reg_scale=reg_scale,
                gamma_prior=gamma_prior,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=5.0)
            optimizer3.step()
            scheduler3.step()

            loss_history.append(loss_info)

            if loss_info['total'] < best_loss3:
                best_loss3  = loss_info['total']
                best_state3 = deepcopy(params.state_dict())

            if verbose and epoch % 50 == 0:
                print(
                    f"  Stage 3 epoch {epoch:3d}/{stage3_epochs}  "
                    f"loss={loss_info['total']:.5f}  "
                    f"obs={loss_info['obs']:.5f}  "
                    f"reg={loss_info['reg']:.5f}"
                )

            if progress_callback is not None:
                progress_callback(n_epochs + epoch, _total_steps, loss_info)

        if best_state3 is not None:
            params.load_state_dict(best_state3)
        if best_loss3 < best_loss:
            best_loss = best_loss3

    return {
        'params':       params,
        'loss_history': loss_history,
        'best_loss':    best_loss,
        'n_epochs_run': len(loss_history),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compare_to_truth(
    params:      ModelParameters,
    game_pool,
    participant_pool,
) -> dict:
    """
    Compare estimated parameters to ground-truth values (simulation study only).

    Returns a dict of RMSE and correlation for each parameter group.
    """
    import numpy as np

    results: dict = {}

    def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).pow(2).mean().sqrt().item())

    def corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a_np = a.detach().float().numpy().ravel()
        b_np = b.detach().float().numpy().ravel()
        if a_np.std() < 1e-8 or b_np.std() < 1e-8:
            return float('nan')
        return float(np.corrcoef(a_np, b_np)[0, 1])

    gp  = game_pool.parameter_tensors()
    est_zeta  = params.zeta_jk.detach()
    est_beta0 = params.beta0_jk.detach()
    est_beta1 = params.beta1_jk.detach()

    results['zeta_jk']  = {'rmse': rmse(est_zeta,  gp['zeta']),
                            'corr': corr(est_zeta,  gp['zeta'])}
    results['beta0_jk'] = {'rmse': rmse(est_beta0, gp['beta0']),
                            'corr': corr(est_beta0, gp['beta0'])}
    results['beta1_jk'] = {'rmse': rmse(est_beta1, gp['beta1']),
                            'corr': corr(est_beta1, gp['beta1'])}

    C_true = participant_pool.C_init_tensor()
    A_true = participant_pool.A_star_tensor()
    u_true = participant_pool.u_tensor()

    results['C_init'] = {'rmse': rmse(params.C_init.detach(), C_true),
                          'corr': corr(params.C_init.detach(), C_true)}
    results['A_star'] = {'rmse': rmse(params.A_star.detach(), A_true),
                          'corr': corr(params.A_star.detach(), A_true)}
    results['u_ijk']  = {'rmse': rmse(params.u_ijk.detach(), u_true),
                          'corr': corr(params.u_ijk.detach(), u_true)}

    return results


def print_comparison(comparison: dict) -> None:
    """Pretty-print compare_to_truth results."""
    print("Parameter recovery (RMSE | corr):")
    for name, v in comparison.items():
        print(f"  {name:12s}: RMSE={v['rmse']:.4f}  corr={v['corr']:.4f}")
