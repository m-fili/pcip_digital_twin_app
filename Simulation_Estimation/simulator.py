"""
Simulator module: forward simulation of the PCIP model.

Classes
-------
SimParams         — all model parameters needed for simulation
SimulationDataset — output container for one full simulation run
Simulator         — orchestrates participant × session × game forward pass

Usage
-----
    cfg    = yaml.safe_load(open("config/default_params.yaml"))
    params = SimParams.from_config(cfg)

    pool      = ParticipantPool.generate(...)
    game_pool = GamePool.generate(...)
    policy    = StaircasePolicy(I=pool.I, M=params.M, game_structure=game_pool.to_structure())

    sim     = Simulator(params)
    dataset = sim.run(pool, game_pool, policy, seed=0)
    null_ds = sim.run_null(pool)

Simulation output arrays (all float32 unless noted)
----------------------------------------------------
    ogs      [I, T, M]        — observed game scores
    d        [I, T, M]        — difficulties played
    A_obs    [I, T, M]        — observed arousal (computed, not surveyed here)
    V_obs    [I, T, M]        — observed valence (computed, not surveyed here)
    game_j   [I, T, M] long   — local game index
    game_k   [I, T, M] long   — domain index
    m_order  [I, T, M] long   — play order within session (0-indexed)
    n_before [I, T, M]        — prior play count for each (i,t,m) game
    C_true   [I, T+1, K]      — latent ability (ground truth)
    Q_true   [I, T, K]        — cumulative impact (ground truth)
    pi_true  [I, T, M]        — single-game gain (ground truth)
    Pi_true  [I, T, K]        — total session gain per domain (ground truth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

import core
from games import GamePool
from participants import ParticipantPool
from policy.base_policy import BasePolicy


# ---------------------------------------------------------------------------
# SimParams  — model parameters for simulation
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """
    All model parameters required to run a forward simulation.

    Split into four groups mirroring config/default_params.yaml:
        predefined  — arousal / valence model constants (not estimated)
        global_     — globally shared estimated parameters
        domain      — per-domain parameters (replicated as length-K lists)
        program     — simulation program structure (K, M, T, etc.)
    """

    # --- Program structure ---
    K:   int = 3      # number of cognitive domains
    Nk:  int = 5      # games per domain (uniform)
    M:   int = 5      # games per session
    T:   int = 20     # total sessions
    I:   int = 50     # number of participants (informational; pool supplies actual I)

    # --- Noise ---
    sigma_OGS: float = 0.10   # log-scale OGS noise std
    sigma_pi:  float = 0.02   # gain noise std

    # --- Arousal predefined ---
    A_min:    float = 0.2
    gamma0_A: float = 0.0
    gamma1_A: float = 2.0
    K0:       float = 2.0
    rho_u:    float = 1.0

    # --- Valence predefined ---
    gamma0_V: float = 0.5
    gamma_Vu: float = 1.0
    gamma_Vh: float = 0.5
    gamma_Ve: float = 0.09
    tau_V:    float = 1.0
    beta_Vh:  float = 5.0
    beta_Ve:  float = 5.0

    # --- Expected score ---
    gamma0: float = 0.0
    gamma1: float = 1.0

    # --- Emotional regulator ---
    E_min:   float = 0.3
    kappa_A: float = 0.5
    kappa_V: float = 0.5
    alpha_A: float = 1.0
    alpha_V: float = 1.0

    # --- Practice bias ---
    kappa_B:      float = 2.0
    delta_B_star: float = 0.0

    # --- Game effectiveness ---
    zeta_min: float = 0.1
    lambda_Z: float = 0.1

    # --- Mismatch effect ---
    kappa_L:    float = 2.0
    kappa_R:    float = 1.0
    delta_star: float = 0.0

    # --- Game engagement ---
    u_min:   float = 0.1
    alpha_u: float = 2.0

    # --- Fatigue cost ---
    rho0_F:  float = 0.05
    rho_h_F: float = 0.5
    rho_e_F: float = 0.5
    tau_F:   float = 1.0

    # --- Ability dynamics ---
    rho_Q: float = 0.8    # cumulative impact decay

    # --- Per-domain (length-K lists, set by from_config) ---
    eta_k:  list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    Bk_max: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.3])
    Bk_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict, K: Optional[int] = None) -> SimParams:
        """
        Build SimParams from the loaded YAML config dict.

        cfg : the full YAML dict (top-level keys: predefined, global_init,
              domain_init, simulation)
        K   : override for number of domains (default: cfg['simulation']['program']['K'])
        """
        prog = cfg['simulation']['program']
        k    = K if K is not None else prog['K']

        pre   = cfg['predefined']
        glob  = cfg['global_init']
        dom   = cfg['domain_init']
        noise = cfg['simulation']['noise']

        return cls(
            # program
            K=k,
            Nk=prog['Nk'],
            M=prog['M'],
            T=prog['T'],
            I=prog['I'],
            # noise
            sigma_OGS=noise['sigma_OGS'],
            sigma_pi=noise['sigma_pi'],
            # arousal
            A_min=pre['arousal']['A_min'],
            gamma0_A=pre['arousal']['gamma0_A'],
            gamma1_A=pre['arousal']['gamma1_A'],
            K0=pre['arousal']['K0'],
            rho_u=pre['arousal']['rho_u'],
            # valence
            gamma0_V=pre['valence']['gamma0_V'],
            gamma_Vu=pre['valence']['gamma_Vu'],
            gamma_Vh=pre['valence']['gamma_Vh'],
            gamma_Ve=pre['valence']['gamma_Ve'],
            tau_V=pre['valence']['tau_V'],
            beta_Vh=pre['valence']['beta_Vh'],
            beta_Ve=pre['valence']['beta_Ve'],
            # global
            gamma0=glob['gamma0'],
            gamma1=glob['gamma1'],
            E_min=glob['E_min'],
            kappa_A=glob['kappa_A'],
            kappa_V=glob['kappa_V'],
            alpha_A=glob['alpha_A'],
            alpha_V=glob['alpha_V'],
            kappa_B=glob['kappa_B'],
            delta_B_star=glob['delta_B_star'],
            zeta_min=glob['zeta_min'],
            lambda_Z=glob['lambda_Z'],
            kappa_L=glob['kappa_L'],
            kappa_R=glob['kappa_R'],
            delta_star=glob['delta_star'],
            u_min=glob['u_min'],
            alpha_u=glob['alpha_u'],
            rho0_F=glob['rho0_F'],
            rho_h_F=glob['rho_h_F'],
            rho_e_F=glob['rho_e_F'],
            tau_F=glob['tau_F'],
            rho_Q=glob['rho_Q'],
            # per-domain (replicate scalar to list of length K)
            eta_k =[dom['eta_k']]  * k,
            Bk_max=[dom['Bk_max']] * k,
            Bk_min=[dom['Bk_min']] * k,
        )


# ---------------------------------------------------------------------------
# SimulationDataset  — output container
# ---------------------------------------------------------------------------

@dataclass
class SimulationDataset:
    """
    All data produced by one simulation run.

    Observation arrays  (float32 unless noted as long)
    --------------------------------------------------
    ogs      [I, T, M]       — observed game scores
    d        [I, T, M]       — difficulty levels used
    A_obs    [I, T, M]       — arousal at each game play
    V_obs    [I, T, M]       — valence at each game play
    game_j   [I, T, M] long  — local game index j
    game_k   [I, T, M] long  — domain index k
    m_order  [I, T, M] long  — play order within session (0-indexed)
    n_before [I, T, M]       — prior play count before this play

    Ground-truth arrays
    -------------------
    C_true   [I, T+1, K]     — latent ability at start of each session
    Q_true   [I, T, K]       — cumulative impact after each session
    pi_true  [I, T, M]       — single-game cognitive gain (noiseless)
    Pi_true  [I, T, K]       — total session gain per domain (noiseless)
    """
    # Observation
    ogs:     Tensor   # [I, T, M]
    d:       Tensor   # [I, T, M]
    A_obs:   Tensor   # [I, T, M]
    V_obs:   Tensor   # [I, T, M]
    game_j:  Tensor   # [I, T, M] long
    game_k:  Tensor   # [I, T, M] long
    m_order: Tensor   # [I, T, M] long
    n_before:Tensor   # [I, T, M]

    # Ground truth
    C_true:  Tensor   # [I, T+1, K]
    Q_true:  Tensor   # [I, T, K]
    pi_true: Tensor   # [I, T, M]
    Pi_true: Tensor   # [I, T, K]

    def __repr__(self) -> str:
        I, T, M = self.ogs.shape
        K = self.C_true.shape[-1]
        return (
            f"SimulationDataset(I={I}, T={T}, M={M}, K={K})"
        )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """
    Orchestrates a full PCIP simulation over all participants and sessions.

    Parameters
    ----------
    params : SimParams — all model parameters (ground-truth values for simulator)

    Methods
    -------
    run(pool, game_pool, policy, seed)  → SimulationDataset
        Full intervention simulation with a policy.

    run_null(pool)                      → SimulationDataset
        No-intervention baseline: all gains are zero, C stays at C_init.
        Useful for computing the counterfactual trajectory.
    """

    def __init__(self, params: SimParams) -> None:
        self.p = params

    # ------------------------------------------------------------------
    # Main simulation run
    # ------------------------------------------------------------------

    def run(
        self,
        pool:      ParticipantPool,
        game_pool: GamePool,
        policy:    BasePolicy,
        seed:      int | None = None,
    ) -> SimulationDataset:
        """
        Simulate all I participants over T sessions using the given policy.

        Steps per (participant i, session t, game position m):
            1. Policy selects M (j, k, d) tuples for participant i
            2. For each game m:
                a. Compute δ = C_ik - d
                b. Compute A, V (predefined)
                c. Compute ES, Bias, Emot
                d. Sample OGS = ES · Bias · Emot · exp(ε_OGS)
                e. Compute Z, Ψ, E, F
                f. Compute π = Z · Ψ · E − F + ε_π
            3. Update policy with observed OGS
            4. Accumulate session gain Π per domain
            5. Update Q and C via dynamics
            6. Record session in participant

        Parameters
        ----------
        pool      : ParticipantPool (provides ground-truth C_init, A_star, u)
        game_pool : GamePool (provides ground-truth zeta, beta0, beta1)
        policy    : BasePolicy (provides game/difficulty selection + update)
        seed      : global RNG seed for noise sampling

        Returns
        -------
        SimulationDataset with all observations and ground-truth arrays
        """
        p    = self.p
        I    = pool.I
        T    = p.T
        M    = p.M
        K    = p.K

        # Reset participants and policy to initial state
        pool.reset_all()
        policy.reset()

        # RNG for noise
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        # Ground-truth game parameter tensors  [J_total]
        gp_tensors = game_pool.parameter_tensors()
        zeta_all   = gp_tensors['zeta']    # [J_total]
        beta0_all  = gp_tensors['beta0']   # [J_total]
        beta1_all  = gp_tensors['beta1']   # [J_total]

        # Pre-allocate output arrays
        ogs_out     = torch.zeros(I, T, M)
        d_out       = torch.zeros(I, T, M)
        A_out       = torch.zeros(I, T, M)
        V_out       = torch.zeros(I, T, M)
        game_j_out  = torch.zeros(I, T, M, dtype=torch.long)
        game_k_out  = torch.zeros(I, T, M, dtype=torch.long)
        m_order_out = torch.zeros(I, T, M, dtype=torch.long)
        n_before_out= torch.zeros(I, T, M)
        C_out       = torch.zeros(I, T + 1, K)
        Q_out       = torch.zeros(I, T, K)
        pi_out      = torch.zeros(I, T, M)
        Pi_out      = torch.zeros(I, T, K)

        # Per-domain tensors
        eta_k_t   = torch.tensor(p.eta_k,   dtype=torch.float32)   # [K]
        Bk_max_t  = torch.tensor(p.Bk_max,  dtype=torch.float32)   # [K]
        Bk_min_t  = torch.tensor(p.Bk_min,  dtype=torch.float32)   # [K]

        for i, participant in enumerate(pool):
            # Store C_init in output
            C_out[i, 0] = participant.C_current.clone()

            for t_idx in range(T):          # t_idx = 0 → session 1
                t_1based = t_idx + 1        # 1-indexed session number

                # --- Policy selects M games ---
                selection = policy.select(
                    i=i,
                    t=t_1based,
                    C_current=participant.C_current,
                    n_current=participant.n_current,
                )  # list of M (j, k, d)

                # Domain-level accumulators for Π (sum π per domain)
                Pi_domain = torch.zeros(K)

                session_results: list[tuple[int, int, float, float]] = []

                for m, (j, k, d) in enumerate(selection):
                    flat = game_pool.flat_idx(j, k)

                    # Retrieve ground-truth game params
                    zeta_jk  = float(zeta_all[flat].item())
                    beta0_jk = float(beta0_all[flat].item())
                    beta1_jk = float(beta1_all[flat].item())

                    # Individual params
                    C_k      = float(participant.C_current[k].item())
                    A_star_i = participant.A_star
                    u_ij     = float(participant.u_flat[flat].item())
                    n_ij     = float(participant.n_current[flat].item())

                    d_t     = torch.tensor(d)
                    C_k_t   = torch.tensor(C_k)
                    u_ij_t  = torch.tensor(u_ij)
                    n_ij_t  = torch.tensor(n_ij)

                    # delta = ability - difficulty
                    delta = core.ability_difficulty_gap(C_k_t, d_t)

                    # Predefined: Arousal and Valence
                    A = core.arousal(
                        delta, u_ij_t,
                        p.A_min, p.gamma0_A, p.gamma1_A, p.K0, p.rho_u,
                    )
                    V = core.valence(
                        u_ij_t, delta,
                        p.gamma0_V, p.gamma_Vu,
                        p.gamma_Vh, p.gamma_Ve, p.tau_V,
                        p.beta_Vh, p.beta_Ve,
                    )

                    # Observation equation components
                    A_star_t = torch.tensor(A_star_i)
                    ES   = core.expected_score(delta, p.gamma0, p.gamma1)
                    Bias = core.practice_bias(
                        n_ij_t, delta,
                        torch.tensor(beta0_jk), torch.tensor(beta1_jk),
                        Bk_min_t[k], Bk_max_t[k],
                        p.kappa_B, p.delta_B_star,
                    )
                    Emot = core.emotional_regulator(
                        A, V, A_star_t,
                        p.E_min, p.kappa_A, p.kappa_V, p.alpha_A, p.alpha_V,
                    )

                    OGS_mean = core.obs_game_score_mean(ES, Bias, Emot)

                    # Sample OGS with log-normal noise
                    eps_OGS = p.sigma_OGS * torch.randn(1, generator=rng).squeeze()
                    OGS_obs = OGS_mean * torch.exp(eps_OGS)
                    OGS_obs = torch.clamp(OGS_obs, 0.0, 100.0)

                    # Gain equation components
                    Z   = core.game_effectiveness(
                        n_ij_t, torch.tensor(zeta_jk), p.zeta_min, p.lambda_Z,
                    )
                    Psi = core.mismatch_effect(delta, p.delta_star, p.kappa_L, p.kappa_R)
                    E   = core.game_engagement(u_ij_t, p.u_min, p.alpha_u)
                    F   = core.fatigue_cost(
                        float(m + 1), float(M), delta, p.delta_star,
                        p.rho0_F, p.rho_h_F, p.rho_e_F, p.tau_F,
                    )

                    pi_noiseless = core.single_game_gain(Z, Psi, E, F)
                    eps_pi       = p.sigma_pi * torch.randn(1, generator=rng).squeeze()
                    pi_obs       = pi_noiseless + eps_pi

                    # Accumulate domain gain
                    Pi_domain[k] = Pi_domain[k] + pi_obs.item()

                    # Update play count for this game
                    participant.n_current[flat] += 1.0

                    # Store in output arrays
                    ogs_out[i, t_idx, m]      = OGS_obs.detach()
                    d_out[i, t_idx, m]        = d
                    A_out[i, t_idx, m]        = A.detach()
                    V_out[i, t_idx, m]        = V.detach()
                    game_j_out[i, t_idx, m]   = j
                    game_k_out[i, t_idx, m]   = k
                    m_order_out[i, t_idx, m]  = m
                    n_before_out[i, t_idx, m] = n_ij  # count before this play
                    pi_out[i, t_idx, m]       = pi_noiseless.detach()

                    session_results.append((j, k, d, float(OGS_obs.item())))

                # --- Dynamics update ---
                Q_new = core.update_cumulative_impact(
                    participant.Q_current, Pi_domain, p.rho_Q,
                )
                C_new = core.update_ability(
                    participant.C_current, Q_new,
                    eta_k_t,
                )

                # Record in participant (advances C_current, Q_current, n_current)
                participant.record_session(
                    t_idx=t_idx,
                    Q_new=Q_new,
                    C_new=C_new,
                    n_after=participant.n_current.clone(),
                )

                # Store ground-truth arrays
                Q_out[i, t_idx]     = Q_new.detach()
                C_out[i, t_idx + 1] = C_new.detach()
                Pi_out[i, t_idx]    = Pi_domain.detach()

                # --- Policy update ---
                policy.update(i=i, t=t_1based, results=session_results)

        return SimulationDataset(
            ogs=ogs_out,
            d=d_out,
            A_obs=A_out,
            V_obs=V_out,
            game_j=game_j_out,
            game_k=game_k_out,
            m_order=m_order_out,
            n_before=n_before_out,
            C_true=C_out,
            Q_true=Q_out,
            pi_true=pi_out,
            Pi_true=Pi_out,
        )

    # ------------------------------------------------------------------
    # Null (no-intervention) baseline
    # ------------------------------------------------------------------

    def run_null(self, pool: ParticipantPool) -> SimulationDataset:
        """
        Counterfactual no-intervention baseline.

        No games are played: C_ik^t = C_ik^1 for all t.  All gain/OGS
        arrays are zeros.  Useful for computing the intervention effect
        (C_intervention − C_null).

        Returns
        -------
        SimulationDataset with zero gain arrays and flat C_true
        """
        p  = self.p
        I  = pool.I
        T  = p.T
        M  = p.M
        K  = p.K

        pool.reset_all()

        C_out = torch.zeros(I, T + 1, K)
        for i, participant in enumerate(pool):
            for t in range(T + 1):
                C_out[i, t] = participant.C_init.clone()

        zeros_ITM = torch.zeros(I, T, M)
        zeros_ITK = torch.zeros(I, T, K)

        return SimulationDataset(
            ogs=zeros_ITM.clone(),
            d=zeros_ITM.clone(),
            A_obs=zeros_ITM.clone(),
            V_obs=zeros_ITM.clone(),
            game_j=torch.zeros(I, T, M, dtype=torch.long),
            game_k=torch.zeros(I, T, M, dtype=torch.long),
            m_order=torch.zeros(I, T, M, dtype=torch.long),
            n_before=zeros_ITM.clone(),
            C_true=C_out,
            Q_true=zeros_ITK.clone(),
            pi_true=zeros_ITM.clone(),
            Pi_true=zeros_ITK.clone(),
        )
