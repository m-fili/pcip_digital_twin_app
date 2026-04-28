"""
core — Pure PyTorch functions implementing all PCIP model equations.

All functions are stateless and differentiable (compatible with autograd).
They accept tensors of any broadcast-compatible shape, making them usable
for both sequential simulation (per-game, per-session) and batched estimation
(all individuals × games × sessions at once).

Modules:
    kernels     — utility kernel functions (Gaussian, Asymmetric Laplace, SoftPlus)
    observation — ES, Bias, Emot, Arousal, Valence, OGS
    gain        — Z, Ψ, E, F, single-game gain π, session total Π
    dynamics    — cumulative impact Q and ability C update equations
"""

from .kernels import (
    gaussian_kernel,
    asymmetric_laplace_kernel,
    softplus_beta,
)

from .observation import (
    ability_difficulty_gap,
    expected_score,
    practice_bias,
    arousal,
    valence,
    emotional_regulator,
    obs_game_score_mean,
)

from .gain import (
    game_effectiveness,
    mismatch_effect,
    game_engagement,
    fatigue_cost,
    single_game_gain,
    total_session_gain,
)

from .dynamics import (
    update_cumulative_impact,
    update_ability,
    overall_cognitive_score,
)

__all__ = [
    # kernels
    "gaussian_kernel",
    "asymmetric_laplace_kernel",
    "softplus_beta",
    # observation
    "ability_difficulty_gap",
    "expected_score",
    "practice_bias",
    "arousal",
    "valence",
    "emotional_regulator",
    "obs_game_score_mean",
    # gain
    "game_effectiveness",
    "mismatch_effect",
    "game_engagement",
    "fatigue_cost",
    "single_game_gain",
    "total_session_gain",
    # dynamics
    "update_cumulative_impact",
    "update_ability",
    "overall_cognitive_score",
]
