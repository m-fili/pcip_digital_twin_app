"""
RandomPolicy: stateless random game selection with uniform difficulty.

Selects M games uniformly at random (without replacement across all J_total
games) and assigns each a random difficulty drawn uniformly from [d_min, d_max].

This policy is used as the no-personalization baseline: it does not adapt
to participant performance or history, making it useful for comparing against
adaptive policies.

Parameters
----------
d_min, d_max : difficulty range (same scale as latent ability C, i.e., [0, 10])
seed         : optional RNG seed for reproducibility
"""

from __future__ import annotations

import torch
from torch import Tensor

from games import GameStructure
from policy.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    """
    Selects M games uniformly at random; assigns random difficulty in [d_min, d_max].

    This policy is completely stateless: select() ignores all inputs and
    update() is a no-op.  The same RNG state is advanced each call so that
    replays with the same seed produce identical trajectories.

    Parameters
    ----------
    I             : number of participants (unused — policy is participant-agnostic)
    M             : games per session
    game_structure: GameStructure from GamePool.to_structure()
    d_min         : minimum difficulty  (default 1.0)
    d_max         : maximum difficulty  (default 9.0)
    seed          : RNG seed for reproducibility (None = non-deterministic)
    """

    def __init__(
        self,
        I:              int,
        M:              int,
        game_structure: GameStructure,
        d_min:          float = 1.0,
        d_max:          float = 9.0,
        seed:           int | None = None,
    ) -> None:
        super().__init__(I, M, game_structure)
        self.d_min = d_min
        self.d_max = d_max
        self._rng  = torch.Generator()
        self._seed = seed
        if seed is not None:
            self._rng.manual_seed(seed)

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def select(
        self,
        i:         int,
        t:         int,
        C_current: Tensor,
        n_current: Tensor,
    ) -> list[tuple[int, int, float]]:
        """
        Sample M games uniformly at random and assign random difficulties.

        Returns
        -------
        list of M tuples (j, k, d) in play order (m = 0 first)
        """
        J = self.gs.J_total

        # Sample M flat indices without replacement
        flat_perm = torch.randperm(J, generator=self._rng)[:self.M]

        # Random difficulties in [d_min, d_max]
        d_vals = (
            self.d_min
            + (self.d_max - self.d_min)
            * torch.rand(self.M, generator=self._rng)
        )

        selection = []
        for idx, flat in enumerate(flat_perm.tolist()):
            j = int(self.gs.game_local_idx[flat].item())
            k = int(self.gs.game_domain_idx[flat].item())
            d = float(d_vals[idx].item())
            selection.append((j, k, d))

        return selection

    def update(
        self,
        i:       int,
        t:       int,
        results: list[tuple[int, int, float, float]],
    ) -> None:
        """No-op: random policy does not learn from outcomes."""

    def reset(self) -> None:
        """Re-seed the RNG to its initial state."""
        if self._seed is not None:
            self._rng.manual_seed(self._seed)

    def __repr__(self) -> str:
        return (
            f"RandomPolicy(M={self.M}, "
            f"d=[{self.d_min},{self.d_max}], "
            f"J_total={self.gs.J_total})"
        )
