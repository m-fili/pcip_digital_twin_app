"""
Observation layer: computes the Observed Game Score (OGS) and its components.

Decomposition (multiplicative, log-additive with noise):
    OGS  =  ES  ·  Bias  ·  Emot  ·  exp(ε_OGS)
    log(OGS) = log(ES) + log(Bias) + log(Emot) + ε_OGS,   ε_OGS ~ N(0, σ²_OGS)

Functions here return deterministic (noise-free) quantities.
The simulator adds noise; the estimator uses these as predictions.

Component roles:
    ES   — IRT-style expected performance given ability-difficulty gap
    Bias — score inflation from game familiarity (independent of true learning)
    Emot — performance scaling from emotional state (arousal × valence)
"""

from __future__ import annotations

import torch
from torch import Tensor

from .kernels import gaussian_kernel, softplus_beta


# ---------------------------------------------------------------------------
# Gap
# ---------------------------------------------------------------------------

def ability_difficulty_gap(C: Tensor, d: Tensor) -> Tensor:
    """
    δ = C − d

    Signed gap between cognitive ability and game difficulty.
      δ > 0 : game is easy  (ability exceeds difficulty)
      δ < 0 : game is hard  (difficulty exceeds ability)
      δ = δ*: optimal challenge ("sweet spot")

    Both C and d live on [0, 10].
    """
    return C - d


# ---------------------------------------------------------------------------
# Expected Score (ES)
# ---------------------------------------------------------------------------

def expected_score(
    delta: Tensor,
    gamma0: float | Tensor,
    gamma1: float | Tensor,
) -> Tensor:
    """
    IRT-style expected score:  ES = 100 · σ(γ₀ + γ₁ δ)  ∈ (0, 100)

    gamma0 : intercept — controls score at δ = 0  (ES = 100·σ(γ₀))
    gamma1 : sensitivity — how steeply score rises with δ (γ₁ > 0)
    """
    return 100.0 * torch.sigmoid(gamma0 + gamma1 * delta)


# ---------------------------------------------------------------------------
# Practice Bias
# ---------------------------------------------------------------------------

def practice_bias(
    n: Tensor,
    delta: Tensor,
    beta0_jk: float | Tensor,
    beta1_jk: float | Tensor,
    Bk_min: float | Tensor,
    Bk_max: float | Tensor,
    kappa_B: float | Tensor,
    delta_B_star: float | Tensor,
) -> Tensor:
    """
    Practice Bias:  Bias = 1 + ω_B · (1 − exp(−(β₀ · n)^β₁))

    Captures score inflation from repeated exposure to the same game,
    independent of true cognitive learning.

    n=0 → Bias = 1 (no inflation before first play)
    n→∞ → Bias → 1 + ω_B  (saturates at maximum inflation)

    ω_B = B_k^min + (B_k^max − B_k^min) · φ(δ | δ_B*, κ_B)
    The maximum inflation is gap-modulated: strongest near the optimal gap δ_B*.

    beta0_jk : rate of saturation  (game-specific)
    beta1_jk : shape of saturation (game-specific, Weibull-like)
    """
    omega_B = Bk_min + (Bk_max - Bk_min) * gaussian_kernel(delta, delta_B_star, kappa_B)
    # Clamp base to avoid 0^beta gradient issues when n=0
    base = torch.clamp(beta0_jk * n, min=1e-8)
    saturation = 1.0 - torch.exp(-(base ** beta1_jk))
    return 1.0 + omega_B * saturation


# ---------------------------------------------------------------------------
# Arousal model  (predefined parameters — not estimated)
# ---------------------------------------------------------------------------

def arousal(
    delta: Tensor,
    u: Tensor,
    A_min: float,
    gamma0_A: float,
    gamma1_A: float,
    K0: float,
    rho_u: float,
) -> Tensor:
    """
    Arousal model (parameters are predefined, not estimated):
        A = A^min + (1 − A^min) · σ(γ₀A + γ₁A(u − 0.5)) · φ(δ | 0, K(u))
        K(u) = K₀ + ρ_u · u

    Interpretation:
    - Peaks when game difficulty matches ability exactly (δ = 0)
    - The sigmoid term scales arousal with utility u (more engaged → more aroused)
    - K(u) widens the bandwidth for high-utility players (sustain arousal over
      a broader difficulty range)

    u  : individual-game utility ∈ [0, 1]  (latent, time-invariant)
    """
    K_u = K0 + rho_u * u
    sigmoid_term = torch.sigmoid(gamma0_A + gamma1_A * (u - 0.5))
    kernel_term  = gaussian_kernel(delta, 0.0, K_u)
    return A_min + (1.0 - A_min) * sigmoid_term * kernel_term


# ---------------------------------------------------------------------------
# Valence model  (predefined parameters — not estimated)
# ---------------------------------------------------------------------------

def valence(
    u: Tensor,
    delta: Tensor,
    gamma0_V: float,
    gamma_Vu: float,
    gamma_Vh: float,
    gamma_Ve: float,
    tau_V: float,
    beta_Vh: float = 5.0,
    beta_Ve: float = 5.0,
) -> Tensor:
    """
    Valence model (parameters are predefined, not estimated):
        V = tanh([γ₀V + γVu(2u−1) − γVh·((-δ)₊^βVh)² − γVe·((δ)₊^βVe)²] / τV)

    V ∈ (−1, 1):  positive = pleasant, negative = unpleasant.

    Three drivers:
      Utility   : γVu(2u−1)         — higher utility → more positive
      Frustration: γVh·((-δ)₊^βVh)² — too-hard games reduce valence
      Boredom   : γVe·((δ)₊^βVe)²  — too-easy games reduce valence

    beta_Vh = beta_Ve = 5 are fixed design choices (sharp SoftPlus ≈ ReLU).
    tau_V is a temperature parameter (sharpness of the tanh response).
    """
    # Smooth approximations of max(0, -δ) and max(0, δ)
    hard_sp = softplus_beta(-delta, beta_Vh)   # ≈ max(0, −δ)  (frustration)
    easy_sp = softplus_beta( delta, beta_Ve)   # ≈ max(0,  δ)  (boredom)

    numerator = (
        gamma0_V
        + gamma_Vu * (2.0 * u - 1.0)
        - gamma_Vh * (hard_sp ** 2)
        - gamma_Ve * (easy_sp ** 2)
    )
    return torch.tanh(numerator / tau_V)


# ---------------------------------------------------------------------------
# Emotional Regulator
# ---------------------------------------------------------------------------

def emotional_regulator(
    A: Tensor,
    V: Tensor,
    A_i_star: float | Tensor,
    E_min: float | Tensor,
    kappa_A: float | Tensor,
    kappa_V: float | Tensor,
    alpha_A: float | Tensor,
    alpha_V: float | Tensor,
) -> Tensor:
    """
    Emotional Regulator:
        Emot = E^min + (1−E^min) · φ(A | A_i*, κA)^αA · φ(V | 1, κV)^αV

    Emot ∈ [E^min, 1].  Scales performance based on current emotional state.

    Peaks when A = A_i* (individual's optimal arousal) and V = 1 (max valence).

    A, V  : observed (via survey / facial emotion recognition)
    A_i*  : individual-specific optimal arousal  (individual-level parameter)
    E_min : global floor (performance persists even in poor emotional state)
    """
    arousal_factor = gaussian_kernel(A, A_i_star, kappa_A) ** alpha_A
    valence_factor = gaussian_kernel(V, 1.0, kappa_V) ** alpha_V
    return E_min + (1.0 - E_min) * arousal_factor * valence_factor


# ---------------------------------------------------------------------------
# Observed Game Score (deterministic part)
# ---------------------------------------------------------------------------

def obs_game_score_mean(ES: Tensor, Bias: Tensor, Emot: Tensor) -> Tensor:
    """
    Deterministic component of OGS:  ES · Bias · Emot  ∈ (0, 100)

    Multiplicative noise exp(ε_OGS), ε_OGS ~ N(0, σ²_OGS), is added by the
    simulator.  The estimator uses this as the predicted mean (on original scale).

    In log space:  log(OGS_mean) = log(ES) + log(Bias) + log(Emot)
    Loss is computed as MSE on log scale:
        L = [log(OGS_obs) − log(OGS_pred)]²
    """
    return ES * Bias * Emot
