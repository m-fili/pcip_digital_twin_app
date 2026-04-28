"""
Gain layer: computes the cognitive gain from each game played in a session.

Single-game gain (deterministic part):
    π^{t,m} = Z(n) · Ψ(δ) · E(u) − F(m, δ)

    Z : game effectiveness  — decays with repetition (novelty wear-off)
    Ψ : mismatch effect     — peaks at the optimal ability-difficulty gap δ*
    E : game engagement     — scales with individual-game utility u
    F : fatigue cost        — ramps with game order m within a session

Total session gain for domain k:
    Π_ik^t = Σ_{games in domain k played at session t} π^{t,m}

Noise ε_π ~ N(0, σ²_π) is added by the simulator, not here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .kernels import asymmetric_laplace_kernel


# ---------------------------------------------------------------------------
# Game Effectiveness  Z
# ---------------------------------------------------------------------------

def game_effectiveness(
    n: Tensor,
    zeta_jk: float | Tensor,
    zeta_min: float | Tensor,
    lambda_Z: float | Tensor,
) -> Tensor:
    """
    Z(n) = ζ^min + (1 − ζ^min) · ζ_jk · exp(−λ_Z · n)

    Effectiveness decays exponentially with prior play count n.
    Floor ζ^min ensures games never become completely ineffective.

    n      : number of times game (j,k) was played before this session
    zeta_jk: initial (maximum) effectiveness ∈ [0,1]  (game-specific, estimated)
    zeta_min: global effectiveness floor              (global, estimated)
    lambda_Z: global decay rate per play              (global, estimated)
    """
    return zeta_min + (1.0 - zeta_min) * zeta_jk * torch.exp(-lambda_Z * n)


# ---------------------------------------------------------------------------
# Mismatch Effect  Ψ
# ---------------------------------------------------------------------------

def mismatch_effect(
    delta: Tensor,
    delta_star: float | Tensor,
    kappa_L: float | Tensor,
    kappa_R: float | Tensor,
) -> Tensor:
    """
    Ψ(δ) = ψ(δ | δ*, κ_L, κ_R)   (asymmetric Laplace kernel)

    Peaks at the optimal gap δ* (the "sweet spot").
    κ_L > κ_R:
      - Too easy (δ > δ*, right side): fast decay  → boredom kills gain quickly
      - Too hard (δ < δ*, left side) : slow decay  → moderate overchallenge still yields gain

    delta_star: optimal challenge gap     (global, estimated)
    kappa_L   : left-side bandwidth       (global, estimated)
    kappa_R   : right-side bandwidth      (global, estimated)
    """
    return asymmetric_laplace_kernel(delta, delta_star, kappa_L, kappa_R)


# ---------------------------------------------------------------------------
# Game Engagement  E
# ---------------------------------------------------------------------------

def game_engagement(
    u: Tensor,
    u_min: float | Tensor,
    alpha_u: float | Tensor,
) -> Tensor:
    """
    E(u) = u^min + (1 − u^min) · u^{α_u}

    Scales gain by how much the individual enjoys this game.
    α_u > 1: convex — low utility is heavily penalised (near-zero engagement
    almost eliminates cognitive gain regardless of other factors).

    u    : individual-game utility ∈ [0,1]  (latent, time-invariant)
    u_min: global engagement floor           (global, estimated)
    alpha_u: convexity exponent (> 1)        (global, estimated)
    """
    u_safe = torch.clamp(u, min=1e-8, max=1.0 - 1e-8)  # numerical safety
    return u_min + (1.0 - u_min) * u_safe ** alpha_u


# ---------------------------------------------------------------------------
# Fatigue Cost  F
# ---------------------------------------------------------------------------

def fatigue_cost(
    m: int,
    M: int,
    delta: Tensor,
    delta_star: float | Tensor,
    rho0_F: float | Tensor,
    rho_h_F: float | Tensor,
    rho_e_F: float | Tensor,
    tau_F: float | Tensor,
) -> Tensor:
    """
    F = ρ₀F · ℓ^{t,m} · [1 + ρhF · hF(δ) + ρeF · eF(δ)]

    Fatigue accumulates over game order m within a session.

    Baseline ramp:
        ℓ^{t,m} = (m−1)/(M−1)   [0 at first game, 1 at last; 0 when M=1]

    Mismatch-induced extra fatigue (both directions are costly):
        hF(δ) = 1 − exp(−(δ* − δ)₊ / τF)   [too-hard games: δ < δ*]
        eF(δ) = 1 − exp(−(δ  − δ*)₊ / τF)  [too-easy games: δ > δ*]

    m   : 1-indexed game order within the session  (1 ≤ m ≤ M)
    M   : total games per session  (structural parameter)
    rho0_F : baseline fatigue scale           (global, estimated)
    rho_h_F: hard-mismatch fatigue multiplier (global, estimated)
    rho_e_F: easy-mismatch fatigue multiplier (global, estimated)
    tau_F  : mismatch fatigue decay scale     (global, estimated)
    """
    ell = 0.0 if M <= 1 else (m - 1.0) / (M - 1.0)

    h_F = 1.0 - torch.exp(-torch.relu(delta_star - delta) / tau_F)
    e_F = 1.0 - torch.exp(-torch.relu(delta - delta_star) / tau_F)

    return rho0_F * ell * (1.0 + rho_h_F * h_F + rho_e_F * e_F)


# ---------------------------------------------------------------------------
# Single-game gain  π  and  session total  Π
# ---------------------------------------------------------------------------

def single_game_gain(
    Z: Tensor,
    Psi: Tensor,
    E: Tensor,
    F: Tensor,
) -> Tensor:
    """
    Deterministic single-game gain:  π = Z · Ψ · E − F

    Noise ε_π ~ N(0, σ²_π) is added by the simulator.
    The estimator uses this as the predicted gain (π can be negative if fatigue
    exceeds the benefit, e.g., at the end of a long session with a mismatched game).
    """
    return Z * Psi * E - F


def total_session_gain(pi_list: list[Tensor]) -> Tensor:
    """
    Π_ik^t = Σ_m π_ijk^{t,m}   (sum over games played in domain k this session)

    pi_list : list of per-game gain tensors (all same shape, e.g. [I] for a batch)
    Returns a tensor of the same shape as each element in pi_list.
    If pi_list is empty (no games played in this domain this session), returns 0.
    """
    if not pi_list:
        return torch.tensor(0.0)
    return torch.stack(pi_list, dim=0).sum(dim=0)
