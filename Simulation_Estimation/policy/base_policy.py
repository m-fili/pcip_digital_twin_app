"""
Base policy interface for game selection and difficulty adaptation.

A policy is responsible for deciding, at the start of each session, which M
games to present to participant i and at what difficulty level.  After the
session results are observed the policy may update its internal state.

Interface
---------
    select(i, t, C_current, n_current)
        → list of M (j, k, d) tuples  [game index, domain, difficulty]

    update(i, t, results)
        → None  (update internal state after observing outcomes)

The simulator calls these two methods in sequence for every (participant,
session) pair.  Policies must be stateful across sessions (track history)
but are initialised once at the start of a simulation run.

Note: policies receive C_current and n_current for information but are NOT
required to use them.  RandomPolicy ignores them; StaircasePolicy uses
n_current to track play counts but not C_current.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from games import GameStructure


class BasePolicy(ABC):
    """
    Abstract base class for game-selection / difficulty-adaptation policies.

    Parameters
    ----------
    I             : number of participants
    M             : games per session
    game_structure: structural info (K, Nk, J_total, indices)
    """

    def __init__(
        self,
        I:              int,
        M:              int,
        game_structure: GameStructure,
    ) -> None:
        self.I  = I
        self.M  = M
        self.gs = game_structure

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every concrete policy
    # ------------------------------------------------------------------

    @abstractmethod
    def select(
        self,
        i:         int,
        t:         int,
        C_current: Tensor,   # shape [K] — current ability for participant i
        n_current: Tensor,   # shape [J_total] — prior play counts
    ) -> list[tuple[int, int, float]]:
        """
        Return an ordered list of M (j, k, d) tuples for participant i at
        session t (1-indexed).

        j : local game index within domain k
        k : domain index
        d : difficulty level (continuous float, same scale as C)

        The ordering matters: game m = 0 is played first, m = M-1 last.
        The simulator uses this order to compute the fatigue term.
        """
        ...

    @abstractmethod
    def update(
        self,
        i:       int,
        t:       int,
        results: list[tuple[int, int, float, float]],
    ) -> None:
        """
        Receive session outcome and update internal state.

        results : list of M tuples (j, k, d, ogs)
            j   : game local index
            k   : domain index
            d   : difficulty used
            ogs : observed game score (0-100)

        Called immediately after session t finishes for participant i.
        Stateless (no-op) policies may implement this as `pass`.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset internal state to initial conditions.

        Override in subclasses that maintain per-participant history.
        Default implementation is a no-op (stateless policies).
        """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"I={self.I}, M={self.M}, "
            f"K={self.gs.K}, J_total={self.gs.J_total})"
        )
