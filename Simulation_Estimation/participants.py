"""
Participants module: individual entities and the participant pool.

Classes:
    Participant     — one individual with ground-truth parameters + state trajectory
    ParticipantPool — pool of I participants, generated hierarchically

Design notes
------------
- Participant stores ground-truth latent parameters (C_init, A_star, u_flat).
  These are never exposed to the estimator.
- Participant tracks the full state trajectory (C, Q, n over all sessions)
  for post-simulation analysis and comparison with the no-intervention baseline.
- ParticipantPool generates parameters hierarchically from population-level
  hyperparameters (same structure mirrored in estimator/parameters.py).
- All tensors are float32.  Integer counts (n) are stored as float for
  seamless compatibility with PyTorch arithmetic.

Trajectory indexing  (0-based array index → model session)
-----------------------------------------------------------
    C_traj[0]      = C_ik^1        ability before session 1  (= C_init)
    C_traj[t]      = C_ik^{t+1}    ability after session t   (t = 1..T)
    Q_traj[t-1]    = Q_ik^t        cumulative impact after session t
    n_traj[0]      = zeros         play counts before session 1
    n_traj[t]      = n_ijk^{t+1}   play counts after session t
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Participant  (single entity)
# ---------------------------------------------------------------------------

class Participant:
    """
    A single participant in the PCIP study.

    Ground-truth parameters (never seen by the estimator)
    ------------------------------------------------------
    C_init  : shape [K]        — initial latent cognitive ability per domain
    A_star  : scalar float     — individual optimal arousal level ∈ [0, 1]
    u_flat  : shape [J_total]  — utility (preference) for each game, time-invariant

    State trajectory (updated by the simulator, stored for analysis)
    ----------------------------------------------------------------
    C_traj  : shape [T+1, K]       — ability at start of each session (+ final)
    Q_traj  : shape [T,   K]       — cumulative impact after each session
    n_traj  : shape [T+1, J_total] — prior play counts at start of each session
    """

    def __init__(
        self,
        i:       int,
        C_init:  Tensor,    # shape [K]
        A_star:  float,
        u_flat:  Tensor,    # shape [J_total]
        T:       int,       # total number of sessions (needed for pre-allocation)
    ) -> None:
        self.i       = i
        self.K       = C_init.shape[0]
        self.J_total = u_flat.shape[0]
        self.T       = T

        # ------------------------------------------------------------------
        # Ground-truth parameters  (immutable after construction)
        # ------------------------------------------------------------------
        self.C_init = C_init.clone().float()
        self.A_star = float(A_star)
        self.u_flat = u_flat.clone().float()

        # ------------------------------------------------------------------
        # Mutable current state  (advanced session-by-session by simulator)
        # ------------------------------------------------------------------
        self.C_current: Tensor = self.C_init.clone()
        self.Q_current: Tensor = torch.zeros(self.K)
        self.n_current: Tensor = torch.zeros(self.J_total)

        # ------------------------------------------------------------------
        # Pre-allocated trajectory storage
        # ------------------------------------------------------------------
        self.C_traj: Tensor = torch.zeros(T + 1, self.K)
        self.Q_traj: Tensor = torch.zeros(T,     self.K)
        self.n_traj: Tensor = torch.zeros(T + 1, self.J_total)

        # Seed trajectory with initial state
        self.C_traj[0] = self.C_current
        self.n_traj[0] = self.n_current.clone()

    # ------------------------------------------------------------------
    # Called by the simulator after each session
    # ------------------------------------------------------------------

    def record_session(
        self,
        t_idx:   int,      # 0-indexed  (session 1 → t_idx = 0)
        Q_new:   Tensor,   # Q_ik^{t+1}, shape [K]
        C_new:   Tensor,   # C_ik^{t+2} (ability for next session), shape [K]
        n_after: Tensor,   # play counts after this session, shape [J_total]
    ) -> None:
        """
        Store updated state after session t_idx (0-indexed) and advance
        current state pointers.

        Called once per session by the simulator after all games are played
        and the Q / C updates have been computed.
        """
        assert 0 <= t_idx < self.T, f"t_idx={t_idx} out of range [0, {self.T})"

        self.Q_traj[t_idx]     = Q_new.detach().clone()
        self.C_traj[t_idx + 1] = C_new.detach().clone()
        self.n_traj[t_idx + 1] = n_after.detach().clone()

        # Advance current state
        self.Q_current = Q_new.detach().clone()
        self.C_current = C_new.detach().clone()
        self.n_current = n_after.detach().clone()

    def reset(self) -> None:
        """
        Reset to initial conditions so the same participant can be re-simulated
        (e.g., under a different policy or as the no-intervention baseline).
        """
        self.C_current = self.C_init.clone()
        self.Q_current = torch.zeros(self.K)
        self.n_current = torch.zeros(self.J_total)

        self.C_traj.zero_()
        self.Q_traj.zero_()
        self.n_traj.zero_()

        self.C_traj[0] = self.C_current
        self.n_traj[0] = self.n_current.clone()

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @property
    def intervention_effect(self) -> Tensor:
        """
        C_traj − C_init : shape [T+1, K]

        The cumulative cognitive gain relative to baseline at each time point.
        Index 0 = 0 (no effect before any session).
        Index t = ability improvement after t sessions.
        Comparable to the no-intervention trajectory (which stays at 0).
        """
        return self.C_traj - self.C_init.unsqueeze(0)

    @property
    def final_ability(self) -> Tensor:
        """C_ik^{T+1} : shape [K] — ability after all sessions."""
        return self.C_traj[-1]

    @property
    def total_gain(self) -> Tensor:
        """Σ_k (C_ik^{T+1} − C_ik^1) : scalar — summed gain across all domains."""
        return (self.C_traj[-1] - self.C_init).sum()

    def __repr__(self) -> str:
        return (
            f"Participant(i={self.i}, "
            f"C_init={self.C_init.tolist()}, "
            f"A*={self.A_star:.3f})"
        )


# ---------------------------------------------------------------------------
# ParticipantPool
# ---------------------------------------------------------------------------

class ParticipantPool:
    """
    Pool of I participants generated from population-level hyperparameters.

    Hierarchical generation model:
        C_ik^1  ~ clamp( N(mu_C,       sigma_C),       0, 10 )   shape [I, K]
        A_i*    ~ clamp( N(mu_A_star,  sigma_A_star),  0,  1 )   shape [I]
        u_ijk   ~ sigmoid( N(mu_u_logit, sigma_u_logit) )         shape [I, J_total]

    This mirrors the hierarchical structure in estimator/parameters.py, where
    deviations δ from population means are estimated with L2 regularisation.

    Usage
    -----
    pool = ParticipantPool.generate(I=50, K=3, J_total=15, T=20,
                                    hyperparams=cfg['population'], seed=0)
    p    = pool[3]                     # Participant object
    C    = pool.C_init_tensor()        # shape [I, K]
    """

    def __init__(self, participants: list[Participant]) -> None:
        self._participants = participants
        self.I       = len(participants)
        self.K       = participants[0].K       if participants else 0
        self.J_total = participants[0].J_total if participants else 0
        self.T       = participants[0].T       if participants else 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        I:           int,
        K:           int,
        J_total:     int,
        T:           int,
        hyperparams: dict,
        seed:        int | None = None,
    ) -> ParticipantPool:
        """
        Generate I participants by sampling from population hyperparameters.

        Parameters
        ----------
        I           : number of participants
        K           : number of cognitive domains
        J_total     : total number of games  (from GamePool)
        T           : number of sessions
        hyperparams : dict with keys from config `simulation.population`
        seed        : random seed for reproducibility
        """
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        mu_C        = hyperparams['mu_C']
        sigma_C     = hyperparams['sigma_C']
        mu_A        = hyperparams['mu_A_star']
        sigma_A     = hyperparams['sigma_A_star']
        mu_u        = hyperparams['mu_u_logit']
        sigma_u     = hyperparams['sigma_u_logit']

        # Vectorised sampling for all I participants
        # C_init: shape [I, K]
        C_all = torch.clamp(
            mu_C + sigma_C * torch.randn(I, K, generator=rng),
            min=0.0, max=10.0,
        )
        # A_star: shape [I]
        A_all = torch.clamp(
            mu_A + sigma_A * torch.randn(I, generator=rng),
            min=0.0, max=1.0,
        )
        # u_ijk: shape [I, J_total]
        u_all = torch.sigmoid(
            mu_u + sigma_u * torch.randn(I, J_total, generator=rng)
        )

        participants = [
            Participant(
                i=i,
                C_init=C_all[i],
                A_star=float(A_all[i]),
                u_flat=u_all[i],
                T=T,
            )
            for i in range(I)
        ]
        return cls(participants)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, i: int) -> Participant:
        return self._participants[i]

    def __len__(self) -> int:
        return self.I

    def __iter__(self):
        return iter(self._participants)

    def reset_all(self) -> None:
        """Reset every participant to initial conditions."""
        for p in self._participants:
            p.reset()

    # ------------------------------------------------------------------
    # Bulk tensor access (for the simulator's batched forward pass)
    # ------------------------------------------------------------------

    def C_init_tensor(self) -> Tensor:
        """Initial abilities, shape [I, K]."""
        return torch.stack([p.C_init for p in self._participants])

    def A_star_tensor(self) -> Tensor:
        """Optimal arousal levels, shape [I]."""
        return torch.tensor([p.A_star for p in self._participants])

    def u_tensor(self) -> Tensor:
        """Individual-game utilities, shape [I, J_total]."""
        return torch.stack([p.u_flat for p in self._participants])

    def C_traj_tensor(self) -> Tensor:
        """Full ability trajectories, shape [I, T+1, K]."""
        return torch.stack([p.C_traj for p in self._participants])

    def Q_traj_tensor(self) -> Tensor:
        """Full cumulative-impact trajectories, shape [I, T, K]."""
        return torch.stack([p.Q_traj for p in self._participants])

    def intervention_effect_tensor(self) -> Tensor:
        """Intervention effects, shape [I, T+1, K]."""
        return torch.stack([p.intervention_effect for p in self._participants])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Descriptive statistics of the generated population."""
        C  = self.C_init_tensor()         # [I, K]
        A  = self.A_star_tensor()         # [I]
        u  = self.u_tensor()              # [I, J_total]
        lines = [
            f"ParticipantPool  I={self.I}  K={self.K}  J_total={self.J_total}  T={self.T}",
            f"  C_init (per domain):",
        ]
        for k in range(self.K):
            lines.append(
                f"    domain {k}: mean={C[:,k].mean():.2f}  std={C[:,k].std():.2f}"
                f"  min={C[:,k].min():.2f}  max={C[:,k].max():.2f}"
            )
        lines.append(
            f"  A*     : mean={A.mean():.3f}  std={A.std():.3f}"
            f"  min={A.min():.3f}  max={A.max():.3f}"
        )
        lines.append(
            f"  u_ijk  : mean={u.mean():.3f}  std={u.std():.3f}"
            f"  min={u.min():.3f}  max={u.max():.3f}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ParticipantPool(I={self.I}, K={self.K}, J_total={self.J_total})"
