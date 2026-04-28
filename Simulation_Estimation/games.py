"""
Games module: ground-truth game entities and the game pool.

Classes:
    Game          — a single game with its identity and ground-truth parameters
    GamePool      — the full library of games, used by the simulator
    GameStructure — structural information only (no ground-truth params),
                    passed to the estimator so it cannot access true values

Flat indexing scheme
--------------------
Games are identified by (j, k) where j is the local index within domain k.
Internally a contiguous flat index is used:

    flat_idx(j, k) = Σ_{l=0}^{k-1} Nk[l]  +  j

For uniform Nk this simplifies to  k * Nk + j.
All tensors that span games use this flat ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Game  (single entity)
# ---------------------------------------------------------------------------

@dataclass
class Game:
    """
    A single cognitive game with its identity and ground-truth parameters.

    Parameters are sampled during GamePool.generate() and stored here as
    Python floats (plain values, not nn.Parameters).  The estimator never
    sees these — it only receives GameStructure.

    Attributes
    ----------
    j       : local game index within domain k  (0-indexed)
    k       : domain index  (0-indexed)
    zeta_jk : initial effectiveness ∈ [0, 1]  — how cognitively beneficial
              this game is when played for the first time
    beta0_jk: Bias saturation rate  (> 0)
    beta1_jk: Bias saturation shape (> 0, Weibull-like exponent)
    """
    j:        int
    k:        int
    zeta_jk:  float
    beta0_jk: float
    beta1_jk: float

    def __repr__(self) -> str:
        return (
            f"Game(j={self.j}, k={self.k}, "
            f"ζ={self.zeta_jk:.3f}, β₀={self.beta0_jk:.3f}, β₁={self.beta1_jk:.3f})"
        )


# ---------------------------------------------------------------------------
# GameStructure  (structural info only — safe to pass to the estimator)
# ---------------------------------------------------------------------------

@dataclass
class GameStructure:
    """
    Structural information about the game pool with NO ground-truth parameters.

    This is the only game-related object the estimator receives.  It tells the
    estimator how many games exist, how they map to domains, and how to index
    them — but reveals nothing about the true ζ, β₀, β₁ values.

    Attributes
    ----------
    K               : number of cognitive domains
    Nk              : list of game counts per domain, len = K
    J_total         : total number of games  (= sum(Nk))
    game_domain_idx : LongTensor shape [J_total] — domain index for each flat game
    game_local_idx  : LongTensor shape [J_total] — local j index for each flat game
    """
    K:               int
    Nk:              list[int]
    J_total:         int
    game_domain_idx: Tensor   # shape [J_total], dtype long
    game_local_idx:  Tensor   # shape [J_total], dtype long
    domain_names:    list[str] = field(default_factory=list)

    def flat_idx(self, j: int, k: int) -> int:
        """Convert (j, k) to flat index."""
        return sum(self.Nk[:k]) + j

    def domain_slice(self, k: int) -> slice:
        """Return the slice of flat indices belonging to domain k."""
        start = sum(self.Nk[:k])
        return slice(start, start + self.Nk[k])

    def domain_flat_indices(self, k: int) -> list[int]:
        """Return a list of flat indices for all games in domain k."""
        start = sum(self.Nk[:k])
        return list(range(start, start + self.Nk[k]))

    def domain_name(self, k: int) -> str:
        """Return the name for domain k, falling back to 'Domain k'."""
        if self.domain_names and k < len(self.domain_names):
            return self.domain_names[k]
        return f"Domain {k}"

    def __repr__(self) -> str:
        return (
            f"GameStructure(K={self.K}, Nk={self.Nk}, "
            f"J_total={self.J_total})"
        )


# ---------------------------------------------------------------------------
# GamePool  (full library — used by simulator)
# ---------------------------------------------------------------------------

class GamePool:
    """
    Full library of cognitive games with ground-truth parameters.

    Created once at the start of a simulation run.  The simulator receives
    the full GamePool (with parameters).  The estimator receives only
    GamePool.to_structure() — a parameter-free view.

    Parameters are stored internally in flat tensors for fast lookup by the
    simulator during batched forward passes.

    Usage
    -----
    pool  = GamePool.generate(K=3, Nk=5, hyperparams=cfg['game_pool'], seed=42)
    game  = pool.get_game(j=2, k=1)
    struc = pool.to_structure()            # pass to estimator
    params= pool.parameter_tensors()       # pass to simulator
    """

    def __init__(
        self,
        K:               int,
        Nk:              list[int],
        games_by_domain: dict[int, list[Game]],
        domain_names:    list[str] | None = None,
    ) -> None:
        self.K       = K
        self.Nk      = Nk                   # list[int], len K
        self.J_total = sum(Nk)
        self._games  = games_by_domain      # dict[k → list[Game]]
        self.domain_names = domain_names or [f"Domain {k}" for k in range(K)]

        # Pre-build flat tensors for O(1) simulator access
        all_games = list(self._iter_games())
        self._zeta  = torch.tensor([g.zeta_jk  for g in all_games], dtype=torch.float32)
        self._beta0 = torch.tensor([g.beta0_jk for g in all_games], dtype=torch.float32)
        self._beta1 = torch.tensor([g.beta1_jk for g in all_games], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        K:          int,
        Nk:         int | list[int],
        hyperparams: dict,
        seed:       int | None = None,
    ) -> GamePool:
        """
        Generate a GamePool by sampling game parameters with domain hierarchy.

        Sampling model (per domain k):
            ζ_jk   ~ sigmoid( mu_zeta_k  + N(0, sigma_zeta²)  )
            β₀,jk  ~ exp(     mu_beta0_k + N(0, sigma_beta0²) )
            β₁,jk  ~ exp(     mu_beta1_k + N(0, sigma_beta1²) )

        Domain-level means (mu_zeta_k, etc.) allow systematic differences
        between domains — e.g., EF games can have higher effectiveness than
        MEM games.  Within-domain spreads (sigma_*) are shared.

        Parameters
        ----------
        K           : number of domains
        Nk          : games per domain — int (uniform) or list[int]
        hyperparams : dict with keys from config `simulation.game_pool`
        seed        : random seed for reproducibility
        """
        if isinstance(Nk, int):
            Nk = [Nk] * K
        assert len(Nk) == K, f"len(Nk)={len(Nk)} must equal K={K}"

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        # Domain-level means (lists of length K)
        mu_zeta  = hyperparams['domain_mu_zeta_logit']
        mu_beta0 = hyperparams['domain_mu_log_beta0']
        mu_beta1 = hyperparams['domain_mu_log_beta1']
        assert len(mu_zeta) == K, (
            f"domain_mu_zeta_logit has {len(mu_zeta)} entries but K={K}"
        )

        # Within-domain spreads (shared)
        sigma_zeta  = hyperparams['sigma_zeta_logit']
        sigma_beta0 = hyperparams['sigma_log_beta0']
        sigma_beta1 = hyperparams['sigma_log_beta1']

        # Domain names (optional)
        domain_names = hyperparams.get(
            'domain_names', [f"Domain {k}" for k in range(K)]
        )

        # Sample games per domain with domain-specific means
        games_by_domain: dict[int, list[Game]] = {}
        for k in range(K):
            n_games = Nk[k]
            zeta_k = torch.sigmoid(
                mu_zeta[k] + sigma_zeta * torch.randn(n_games, generator=rng)
            )
            beta0_k = torch.exp(
                mu_beta0[k] + sigma_beta0 * torch.randn(n_games, generator=rng)
            )
            beta1_k = torch.exp(
                mu_beta1[k] + sigma_beta1 * torch.randn(n_games, generator=rng)
            )

            games_by_domain[k] = [
                Game(
                    j=j, k=k,
                    zeta_jk=float(zeta_k[j]),
                    beta0_jk=float(beta0_k[j]),
                    beta1_jk=float(beta1_k[j]),
                )
                for j in range(n_games)
            ]

        return cls(K=K, Nk=Nk, games_by_domain=games_by_domain,
                   domain_names=domain_names)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_game(self, j: int, k: int) -> Game:
        """Return the Game object for game j in domain k."""
        return self._games[k][j]

    def games_in_domain(self, k: int) -> list[Game]:
        """Return all Game objects for domain k."""
        return self._games[k]

    def flat_idx(self, j: int, k: int) -> int:
        """Convert (j, k) to flat index (consistent with GameStructure)."""
        return sum(self.Nk[:k]) + j

    def _iter_games(self) -> Iterator[Game]:
        """Iterate over all games in flat order (k=0 first, then k=1, ...)."""
        for k in range(self.K):
            yield from self._games[k]

    # ------------------------------------------------------------------
    # Parameter tensors for the simulator
    # ------------------------------------------------------------------

    def parameter_tensors(self) -> dict[str, Tensor]:
        """
        Return ground-truth game parameters as flat tensors, shape [J_total].

        Keys: 'zeta', 'beta0', 'beta1'
        These are passed to practice_bias() and game_effectiveness() in core/.
        """
        return {
            'zeta':  self._zeta.clone(),
            'beta0': self._beta0.clone(),
            'beta1': self._beta1.clone(),
        }

    # ------------------------------------------------------------------
    # Structure export (for the estimator)
    # ------------------------------------------------------------------

    def to_structure(self) -> GameStructure:
        """
        Return a parameter-free view of the pool.
        Pass this to the estimator — it contains no ground-truth values.
        """
        domain_idx = torch.zeros(self.J_total, dtype=torch.long)
        local_idx  = torch.zeros(self.J_total, dtype=torch.long)
        idx = 0
        for k in range(self.K):
            for j in range(self.Nk[k]):
                domain_idx[idx] = k
                local_idx[idx]  = j
                idx += 1
        return GameStructure(
            K=self.K,
            Nk=self.Nk,
            J_total=self.J_total,
            game_domain_idx=domain_idx,
            game_local_idx=local_idx,
            domain_names=list(self.domain_names),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Descriptive statistics of generated game parameters, per domain."""
        lines = [
            f"GamePool  K={self.K}  Nk={self.Nk}  J_total={self.J_total}",
            f"  Domains: {self.domain_names}",
        ]
        for k in range(self.K):
            start = sum(self.Nk[:k])
            end   = start + self.Nk[k]
            z_k = self._zeta[start:end]
            b0_k = self._beta0[start:end]
            b1_k = self._beta1[start:end]
            lines.append(
                f"  {self.domain_names[k]:>6s}  "
                f"zeta: mean={z_k.mean():.3f} std={z_k.std():.3f} "
                f"[{z_k.min():.3f}, {z_k.max():.3f}]  "
                f"beta0: mean={b0_k.mean():.3f}  beta1: mean={b1_k.mean():.3f}"
            )
        lines.append(
            f"  Overall: zeta mean={self._zeta.mean():.3f}  "
            f"beta0 mean={self._beta0.mean():.3f}  beta1 mean={self._beta1.mean():.3f}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GamePool(K={self.K}, Nk={self.Nk}, J_total={self.J_total})"
