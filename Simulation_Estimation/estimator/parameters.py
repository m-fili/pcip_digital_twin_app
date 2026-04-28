"""
ModelParameters: all trainable parameters for the PCIP estimator.

Parameter taxonomy
------------------
Global (shared, 1 value each):
    gamma0, gamma1                 — expected score
    E_min, kappa_A, kappa_V,
    alpha_A, alpha_V               — emotional regulator
    kappa_B, delta_B_star          — practice bias
    zeta_min, lambda_Z             — game effectiveness
    kappa_L, kappa_R, delta_star   — mismatch effect
    u_min, alpha_u                 — game engagement
    rho0_F, rho_h_F, rho_e_F, tau_F — fatigue cost
    rho_Q                          — ability dynamics decay

Per-domain (K values each):
    eta_k   — learning rate per domain
    Bk_max  — max practice bias amplitude per domain
    (Bk_min is fixed at 0; can be un-fixed if needed)

Game-level hierarchical (J_total values each):
    zeta_jk   = sigmoid(mu_zeta_logit  + delta_zeta)
    beta0_jk  = exp(mu_log_beta0       + delta_log_beta0)
    beta1_jk  = exp(mu_log_beta1       + delta_log_beta1)

Participant-level hierarchical (I values each):
    C_ik^1  = clamp(mu_C  + delta_C,       0, 10)   shape [I, K]
    A_star_i = sigmoid(mu_A_logit + delta_A_logit)   shape [I]
    u_ijk    = sigmoid(mu_u_logit + delta_u_logit)   shape [I, J_total]

Reparameterization & regularisation
-------------------------------------
Hierarchical parameters are reparameterised as  μ + δ  where:
    - μ (population mean) is a free parameter
    - δ_i (individual deviation) is regularised with L2: λ·‖δ‖²
      (equivalent to a Gaussian prior δ ~ N(0, 1/(2λ)))

Regularisation strengths λ are computed from the simulation generating
sigmas as  λ = 1 / (2 · sigma²), so that the prior matches the
true generating distribution.

Parameter constraints & transforms
------------------------------------
    > 0  (positive)  : stored as log, retrieved with exp()
    (0,1) (probability): stored as logit, retrieved with sigmoid()
    unconstrained      : stored directly
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from games import GameStructure


class ModelParameters(nn.Module):
    """
    All trainable parameters for the PCIP estimator, as an nn.Module.

    Parameters
    ----------
    I              : number of participants
    game_structure : GameStructure (structural only, no ground-truth params)
    init_cfg       : dict with keys 'global_init' and 'domain_init'
                     (from config/default_params.yaml)
    sim_cfg        : dict with key 'simulation' (for hierarchical prior sigmas)
                     Pass None to use default regularisation lambdas.
    """

    def __init__(
        self,
        I:              int,
        game_structure: GameStructure,
        init_cfg:       dict,
        sim_cfg:        dict | None = None,
    ) -> None:
        super().__init__()

        self.I  = I
        self.K  = game_structure.K
        self.J  = game_structure.J_total
        self.gs = game_structure

        gi = init_cfg['global_init']
        di = init_cfg['domain_init']

        K = self.K
        J = self.J

        # ------------------------------------------------------------------
        # Global parameters  (raw / unconstrained storage)
        # ------------------------------------------------------------------

        # Expected score
        self._gamma0        = nn.Parameter(torch.tensor(float(gi['gamma0'])))
        self._log_gamma1    = nn.Parameter(torch.log(torch.tensor(float(gi['gamma1']))))

        # Emotional regulator
        self._logit_E_min   = nn.Parameter(torch.logit(torch.tensor(float(gi['E_min']))))
        self._log_kappa_A   = nn.Parameter(torch.log(torch.tensor(float(gi['kappa_A']))))
        self._log_kappa_V   = nn.Parameter(torch.log(torch.tensor(float(gi['kappa_V']))))
        self._log_alpha_A   = nn.Parameter(torch.log(torch.tensor(float(gi['alpha_A']))))
        self._log_alpha_V   = nn.Parameter(torch.log(torch.tensor(float(gi['alpha_V']))))

        # Practice bias
        self._log_kappa_B    = nn.Parameter(torch.log(torch.tensor(float(gi['kappa_B']))))
        self._delta_B_star   = nn.Parameter(torch.tensor(float(gi['delta_B_star'])))

        # Game effectiveness
        self._logit_zeta_min = nn.Parameter(torch.logit(torch.tensor(float(gi['zeta_min']))))
        self._log_lambda_Z   = nn.Parameter(torch.log(torch.tensor(float(gi['lambda_Z']))))

        # Mismatch effect
        self._log_kappa_L   = nn.Parameter(torch.log(torch.tensor(float(gi['kappa_L']))))
        self._log_kappa_R   = nn.Parameter(torch.log(torch.tensor(float(gi['kappa_R']))))
        self._delta_star    = nn.Parameter(torch.tensor(float(gi['delta_star'])))

        # Game engagement
        self._logit_u_min   = nn.Parameter(torch.logit(torch.tensor(float(gi['u_min']))))
        self._log_alpha_u   = nn.Parameter(torch.log(torch.tensor(float(gi['alpha_u']))))

        # Fatigue cost
        self._log_rho0_F    = nn.Parameter(torch.log(torch.tensor(float(gi['rho0_F']))))
        self._log_rho_h_F   = nn.Parameter(torch.log(torch.tensor(float(gi['rho_h_F']))))
        self._log_rho_e_F   = nn.Parameter(torch.log(torch.tensor(float(gi['rho_e_F']))))
        self._log_tau_F     = nn.Parameter(torch.log(torch.tensor(float(gi['tau_F']))))

        # Ability dynamics
        self._logit_rho_Q   = nn.Parameter(torch.logit(torch.tensor(float(gi['rho_Q']))))

        # ------------------------------------------------------------------
        # Per-domain parameters  [K]
        # ------------------------------------------------------------------

        self._log_eta_k    = nn.Parameter(
            torch.log(torch.full((K,), float(di['eta_k'])))
        )
        # Bk_max ∈ (0, 1) via sigmoid; Bk_min fixed at 0
        self._logit_Bk_max = nn.Parameter(
            torch.logit(torch.full((K,), float(di['Bk_max'])))
        )

        # ------------------------------------------------------------------
        # Game-level hierarchical parameters
        # Reparameterisation: raw_jk = mu + delta_jk
        # ------------------------------------------------------------------

        # zeta_jk = sigmoid(mu_zeta_logit_k[domain] + delta_zeta)   [J]
        self.mu_zeta_logit_k = nn.Parameter(torch.zeros(K))   # per-domain mean
        self.delta_zeta      = nn.Parameter(torch.zeros(J))

        # beta0_jk = exp(mu_log_beta0 + delta_log_beta0)  [J]
        self.mu_log_beta0    = nn.Parameter(torch.tensor(-1.6))
        self.delta_log_beta0 = nn.Parameter(torch.zeros(J))

        # beta1_jk = exp(mu_log_beta1 + delta_log_beta1)  [J]
        self.mu_log_beta1    = nn.Parameter(torch.tensor(0.0))
        self.delta_log_beta1 = nn.Parameter(torch.zeros(J))

        # ------------------------------------------------------------------
        # Participant-level hierarchical parameters
        # ------------------------------------------------------------------

        # C_ik^1 = clamp(mu_C + delta_C, 0, 10)   [I, K]
        self.mu_C    = nn.Parameter(torch.full((K,), 5.0))
        self.delta_C = nn.Parameter(torch.zeros(I, K))

        # A_star_i = sigmoid(mu_A_logit + delta_A_logit)  [I]
        self.mu_A_logit    = nn.Parameter(torch.logit(torch.tensor(0.7)))
        self.delta_A_logit = nn.Parameter(torch.zeros(I))

        # u_ijk = sigmoid(mu_u_logit + delta_u_logit)     [I, J]
        self.mu_u_logit    = nn.Parameter(torch.tensor(0.0))
        self.delta_u_logit = nn.Parameter(torch.zeros(I, J))

        # ------------------------------------------------------------------
        # Regularisation lambdas  (computed from sim sigmas if provided)
        # ------------------------------------------------------------------
        self._lambdas = self._build_lambdas(sim_cfg)

    # ------------------------------------------------------------------
    # Static helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lambdas(sim_cfg: dict | None) -> dict[str, float]:
        """Compute L2 regularisation strength λ = 1 / (2 σ²) from sim sigmas."""
        if sim_cfg is None:
            # Default: mild regularisation
            return {
                'zeta':  1.0,
                'beta0': 1.0,
                'beta1': 1.0,
                'C':     0.1,
                'A':     1.0,
                'u':     0.5,
            }
        gp   = sim_cfg['simulation']['game_pool']
        pop  = sim_cfg['simulation']['population']

        def lam(sigma: float) -> float:
            return 1.0 / (2.0 * sigma ** 2 + 1e-12)

        return {
            'zeta':  lam(gp['sigma_zeta_logit']),
            'beta0': lam(gp['sigma_log_beta0']),
            'beta1': lam(gp['sigma_log_beta1']),
            'C':     lam(pop['sigma_C']),
            'A':     lam(pop['sigma_A_star']),
            'u':     lam(pop['sigma_u_logit']),
        }

    # ------------------------------------------------------------------
    # Constrained parameter properties
    # ------------------------------------------------------------------

    # --- Global ---

    @property
    def gamma0(self) -> Tensor:
        return self._gamma0

    @property
    def gamma1(self) -> Tensor:
        return torch.exp(self._log_gamma1)

    @property
    def E_min(self) -> Tensor:
        return torch.sigmoid(self._logit_E_min)

    @property
    def kappa_A(self) -> Tensor:
        return torch.exp(self._log_kappa_A)

    @property
    def kappa_V(self) -> Tensor:
        return torch.exp(self._log_kappa_V)

    @property
    def alpha_A(self) -> Tensor:
        return torch.exp(self._log_alpha_A)

    @property
    def alpha_V(self) -> Tensor:
        return torch.exp(self._log_alpha_V)

    @property
    def kappa_B(self) -> Tensor:
        return torch.exp(self._log_kappa_B)

    @property
    def delta_B_star(self) -> Tensor:
        return self._delta_B_star

    @property
    def zeta_min(self) -> Tensor:
        return torch.sigmoid(self._logit_zeta_min)

    @property
    def lambda_Z(self) -> Tensor:
        return torch.exp(self._log_lambda_Z)

    @property
    def kappa_L(self) -> Tensor:
        return torch.exp(self._log_kappa_L)

    @property
    def kappa_R(self) -> Tensor:
        return torch.exp(self._log_kappa_R)

    @property
    def delta_star(self) -> Tensor:
        return self._delta_star

    @property
    def u_min(self) -> Tensor:
        return torch.sigmoid(self._logit_u_min)

    @property
    def alpha_u(self) -> Tensor:
        return torch.exp(self._log_alpha_u)

    @property
    def rho0_F(self) -> Tensor:
        return torch.exp(self._log_rho0_F)

    @property
    def rho_h_F(self) -> Tensor:
        return torch.exp(self._log_rho_h_F)

    @property
    def rho_e_F(self) -> Tensor:
        return torch.exp(self._log_rho_e_F)

    @property
    def tau_F(self) -> Tensor:
        return torch.exp(self._log_tau_F)

    @property
    def rho_Q(self) -> Tensor:
        return torch.sigmoid(self._logit_rho_Q)

    # --- Per-domain ---

    @property
    def eta_k(self) -> Tensor:
        """Learning rate per domain, shape [K], > 0."""
        return torch.exp(self._log_eta_k)

    @property
    def Bk_max(self) -> Tensor:
        """Max practice bias amplitude per domain, shape [K], ∈ (0, 1)."""
        return torch.sigmoid(self._logit_Bk_max)

    # --- Game hierarchical ---

    @property
    def zeta_jk(self) -> Tensor:
        """Estimated game effectiveness, shape [J_total], ∈ (0, 1)."""
        mu_per_game = self.mu_zeta_logit_k[self.gs.game_domain_idx]  # [J]
        return torch.sigmoid(mu_per_game + self.delta_zeta)

    @property
    def beta0_jk(self) -> Tensor:
        """Estimated saturation rate, shape [J_total], > 0."""
        return torch.exp(self.mu_log_beta0 + self.delta_log_beta0)

    @property
    def beta1_jk(self) -> Tensor:
        """Estimated saturation shape, shape [J_total], > 0."""
        return torch.exp(self.mu_log_beta1 + self.delta_log_beta1)

    # --- Participant hierarchical ---

    @property
    def C_init(self) -> Tensor:
        """Estimated initial ability, shape [I, K], clamped to [0, 10]."""
        return torch.clamp(self.mu_C.unsqueeze(0) + self.delta_C, 0.0, 10.0)

    @property
    def A_star(self) -> Tensor:
        """Estimated optimal arousal, shape [I], ∈ (0, 1)."""
        return torch.sigmoid(self.mu_A_logit + self.delta_A_logit)

    @property
    def u_ijk(self) -> Tensor:
        """Estimated individual-game utility, shape [I, J_total], ∈ (0, 1)."""
        return torch.sigmoid(self.mu_u_logit + self.delta_u_logit)

    # ------------------------------------------------------------------
    # Regularisation loss
    # ------------------------------------------------------------------

    def regularization_loss(
        self,
        lambdas: dict[str, float] | None = None,
    ) -> Tensor:
        """
        L2 penalty on all deviation parameters (Gaussian prior equivalent).

        λ · ‖δ‖²  summed over:  delta_zeta, delta_log_beta0, delta_log_beta1,
                                 delta_C, delta_A_logit, delta_u_logit

        Parameters
        ----------
        lambdas : override dict; defaults to self._lambdas (computed from sim sigmas)
        """
        lam = lambdas if lambdas is not None else self._lambdas
        reg = (
              lam['zeta']  * self.delta_zeta.pow(2).sum()
            + lam['beta0'] * self.delta_log_beta0.pow(2).sum()
            + lam['beta1'] * self.delta_log_beta1.pow(2).sum()
            + lam['C']     * self.delta_C.pow(2).sum()
            + lam['A']     * self.delta_A_logit.pow(2).sum()
            + lam['u']     * self.delta_u_logit.pow(2).sum()
        )
        return reg

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def param_summary(self) -> str:
        lines = [
            "--- Global ---",
            f"  gamma0={self.gamma0.item():.3f}  gamma1={self.gamma1.item():.3f}",
            f"  E_min={self.E_min.item():.3f}  kappa_A={self.kappa_A.item():.3f}  kappa_V={self.kappa_V.item():.3f}",
            f"  kappa_L={self.kappa_L.item():.3f}  kappa_R={self.kappa_R.item():.3f}  delta_star={self.delta_star.item():.3f}",
            f"  zeta_min={self.zeta_min.item():.3f}  lambda_Z={self.lambda_Z.item():.3f}",
            f"  rho_Q={self.rho_Q.item():.3f}",
            "--- Per-domain ---",
            f"  eta_k={self.eta_k.detach().tolist()}",
            f"  Bk_max={self.Bk_max.detach().tolist()}",
            "--- Game (mean over J_total) ---",
            f"  mu_zeta_k={[f'{v:.3f}' for v in self.mu_zeta_logit_k.detach().tolist()]}",
            f"  zeta_jk:  mean={self.zeta_jk.mean().item():.3f}  std={self.zeta_jk.std().item():.3f}",
            f"  beta0_jk: mean={self.beta0_jk.mean().item():.3f}  std={self.beta0_jk.std().item():.3f}",
            f"  beta1_jk: mean={self.beta1_jk.mean().item():.3f}  std={self.beta1_jk.std().item():.3f}",
            "--- Participant (mean over I) ---",
            f"  C_init:  mean={self.C_init.mean().item():.3f}  std={self.C_init.std().item():.3f}",
            f"  A_star:  mean={self.A_star.mean().item():.3f}  std={self.A_star.std().item():.3f}",
            f"  u_ijk:   mean={self.u_ijk.mean().item():.3f}  std={self.u_ijk.std().item():.3f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"ModelParameters(I={self.I}, K={self.K}, J={self.J}, "
            f"n_params={n_params})"
        )
