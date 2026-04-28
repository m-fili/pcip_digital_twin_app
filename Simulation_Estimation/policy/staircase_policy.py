"""
StaircasePolicy: 3-up-1-down adaptive difficulty with round-robin game selection.

Game selection
--------------
Games are cycled round-robin within each session.  The full ordered list of
J_total games (in flat index order) is rotated by M positions each session,
so every game is visited equally often over time.

Difficulty adaptation  (3-up-1-down rule)
------------------------------------------
For each game j in domain k, a per-participant difficulty level d[i, flat] is
maintained.  After each play:
    - If the last 3 consecutive plays of game (j, k) by participant i all had
      OGS >= threshold (default 70.0), increase difficulty by step_up.
    - If any play had OGS < threshold, decrease difficulty by step_down.

This targets approximately 75% accuracy (the 3-up-1-down equilibrium).

Difficulty is clamped to [d_min, d_max] at all times.

Parameters
----------
I             : number of participants
M             : games per session
game_structure: GameStructure from GamePool.to_structure()
d_init        : initial difficulty for every game/participant  (default 5.0)
d_min         : difficulty floor   (default 1.0)
d_max         : difficulty ceiling  (default 9.0)
step_up       : difficulty increase after 3 consecutive successes  (default 0.3)
step_down     : difficulty decrease after 1 failure                (default 0.5)
threshold     : OGS threshold for success/failure  (default 30.0)
               Set at the 25th percentile of OGS at δ=0 so the
               3-up-1-down rule equilibrates at ~75% success at the sweet spot.
               Derivation: threshold = exp(log(OGS_mean(δ=0)) − 0.674·σ_OGS)
                         ≈ exp(log(34) − 0.674·0.15) ≈ 30.
"""

from __future__ import annotations

import torch
from torch import Tensor

from games import GameStructure
from policy.base_policy import BasePolicy


class StaircasePolicy(BasePolicy):
    """
    3-up-1-down adaptive difficulty with round-robin game selection.

    Internal state (per participant, per game):
        d_current : shape [I, J_total]  — current difficulty level
        consec    : shape [I, J_total]  — consecutive successes since last failure
        rotation  : shape [I]           — current offset into the flat game list
    """

    def __init__(
        self,
        I:              int,
        M:              int,
        game_structure: GameStructure,
        d_init:         float = 5.0,
        d_min:          float = 1.0,
        d_max:          float = 9.0,
        step_up:        float = 1.0,   # 2x step_down — compensates for 3-up-1-down lag in tracking ability growth
        step_down:      float = 0.5,
        threshold:      float = 35.0,  # 3-up-1-down equilibrium at p=79.4%; threshold=35 gives ~81%
    ) -> None:
        super().__init__(I, M, game_structure)
        self.d_init    = d_init
        self.d_min     = d_min
        self.d_max     = d_max
        self.step_up   = step_up
        self.step_down = step_down
        self.threshold = threshold

        J = self.gs.J_total
        self.d_current = torch.full((I, J), d_init)   # shape [I, J_total]
        self.consec    = torch.zeros(I, J, dtype=torch.long)   # consecutive successes
        self.rotation  = torch.zeros(I, dtype=torch.long)       # rotation offset

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
        Select M games in round-robin order starting from rotation[i].

        Returns
        -------
        list of M tuples (j, k, d) in play order
        """
        J   = self.gs.J_total
        off = int(self.rotation[i].item())

        selection = []
        for m in range(self.M):
            flat = (off + m) % J
            j    = int(self.gs.game_local_idx[flat].item())
            k    = int(self.gs.game_domain_idx[flat].item())
            d    = float(self.d_current[i, flat].item())
            selection.append((j, k, d))

        # Advance rotation offset for next session
        self.rotation[i] = (off + self.M) % J

        return selection

    def update(
        self,
        i:       int,
        t:       int,
        results: list[tuple[int, int, float, float]],
    ) -> None:
        """
        Apply 3-up-1-down rule to update difficulty for each played game.

        results : list of (j, k, d, ogs) tuples
        """
        J = self.gs.J_total

        for j, k, d, ogs in results:
            flat = self.gs.flat_idx(j, k)

            if ogs >= self.threshold:
                # Success: increment consecutive counter
                self.consec[i, flat] += 1
                if int(self.consec[i, flat].item()) >= 3:
                    self.d_current[i, flat] = float(
                        torch.clamp(
                            self.d_current[i, flat] + self.step_up,
                            self.d_min, self.d_max,
                        )
                    )
                    self.consec[i, flat] = 0  # reset after step
            else:
                # Failure: decrease difficulty, reset consecutive counter
                self.d_current[i, flat] = float(
                    torch.clamp(
                        self.d_current[i, flat] - self.step_down,
                        self.d_min, self.d_max,
                    )
                )
                self.consec[i, flat] = 0

    def reset(self) -> None:
        """Reset all internal state to initial conditions."""
        J = self.gs.J_total
        self.d_current.fill_(self.d_init)
        self.consec.zero_()
        self.rotation.zero_()

    def __repr__(self) -> str:
        return (
            f"StaircasePolicy(I={self.I}, M={self.M}, "
            f"d_init={self.d_init}, "
            f"step_up={self.step_up}, step_down={self.step_down}, "
            f"threshold={self.threshold})"
        )
