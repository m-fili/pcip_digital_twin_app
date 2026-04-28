"""
Dynamics layer: state update equations for cognitive ability.

Session-level recursion (forward in time, t = 1, ..., T):

    Q_ik^t     = ρ_Q · Q_ik^{t-1} + Π_ik^t        (cumulative impact update)
    C_ik^{t+1} = C_ik^t + η_k · Q_ik^t             (ability update)

Initial conditions (caller's responsibility):
    Q_ik^0 = 0         (no accumulated impact before the program starts)
    C_ik^1 = baseline  (latent initial ability, estimated from pre-assessment)

Interpretation:
    Q is an AR(1) accumulator — a discounted running sum of session gains.
    ρ_Q < 1 means the benefit of old sessions fades; recent sessions matter more.
    Ability C grows by η_k times the accumulated impact each session.

Note on aging: aging effect is assumed negligible for the PCIP program
duration (weeks to months). A simple global aging term may be added later.
"""

from __future__ import annotations

import torch
from torch import Tensor


def update_cumulative_impact(
    Q_prev: Tensor,
    Pi: Tensor,
    rho_Q: float | Tensor,
) -> Tensor:
    """
    Q^t = ρ_Q · Q^{t-1} + Π^t,    0 < ρ_Q < 1

    AR(1) discounted accumulator of session gains.
    Recent sessions are weighted more; old sessions fade at rate ρ_Q per session.

    Q_prev : Q_ik^{t-1}, shape [..., K]  (initialise to zeros for t=1)
    Pi     : Π_ik^t,     shape [..., K]  (total session gain per domain)
    rho_Q  : global decay factor ∈ (0, 1)  (estimated)
    """
    return rho_Q * Q_prev + Pi


def update_ability(
    C: Tensor,
    Q: Tensor,
    eta_k: float | Tensor,
) -> Tensor:
    """
    C^{t+1} = C^t + η_k · Q^t

    Cognitive ability at the start of the next session.
    η_k is the domain-specific learning rate: how much accumulated impact
    translates into lasting ability improvement.

    C    : C_ik^t,  shape [..., K]   (ability per domain, on [0, 10] scale)
    Q    : Q_ik^t,  shape [..., K]   (cumulative impact per domain)
    eta_k: shape [K] or scalar       (domain-specific, estimated)
    """
    return torch.clamp(C + eta_k * Q, min=0.0)


def overall_cognitive_score(C: Tensor) -> Tensor:
    """
    C_i^t = Σ_k C_ik^t   (sum ability across all K domains)

    C : shape [..., K]  →  returns shape [...]
    """
    return C.sum(dim=-1)
