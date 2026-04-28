"""Numpy implementations of the 9 PCIP model components used in sensitivity analysis.

Ported from SA_01_OGS, SA_02_EmotionalRegulation, SA_03_PracticeBias,
SA_04_SessionGain notebooks. All functions accept scalar or array inputs.
"""
import numpy as np


def expit(x):
    """Logistic sigmoid, numerically stable for large |x|."""
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ---------------------------------------------------------------------------
# 1. Expected Score (ES)
# ---------------------------------------------------------------------------
def expected_score(delta, gamma0=1.0, gamma1=1.5):
    """ES(delta) = 100 * sigmoid(gamma0 + gamma1 * delta)."""
    delta = np.asarray(delta, dtype=float)
    return 100.0 * expit(gamma0 + gamma1 * delta)


# ---------------------------------------------------------------------------
# 2. Game Effectiveness Z(n)
# ---------------------------------------------------------------------------
def game_effectiveness(n, zeta=0.8, zeta_min=0.5, lambda_Z=0.03):
    """Z(n) with exponential decay from initial effectiveness toward zeta_min."""
    n = np.asarray(n, dtype=float)
    return zeta_min + (1.0 - zeta_min) * (zeta * np.exp(-lambda_Z * n))


# ---------------------------------------------------------------------------
# 3. Mismatch Effect Psi(delta)
# ---------------------------------------------------------------------------
def mismatch(delta, delta_star=0.0, kappa_L=3.0, kappa_R=1.0):
    """Asymmetric Laplace kernel peaked at delta_star."""
    delta = np.asarray(delta, dtype=float)
    left = np.maximum(0.0, delta_star - delta)
    right = np.maximum(0.0, delta - delta_star)
    return np.exp(-(left / kappa_L) - (right / kappa_R))


# ---------------------------------------------------------------------------
# 4. Engagement E(u)
# ---------------------------------------------------------------------------
def engagement(u, u_min=0.2, alpha_u=2.0):
    u = np.asarray(u, dtype=float)
    return u_min + (1.0 - u_min) * np.power(u, alpha_u)


# ---------------------------------------------------------------------------
# 5. Fatigue F(m, delta)
# ---------------------------------------------------------------------------
def fatigue(m, M, delta, delta_star=0.0,
            rho0=0.2, rho_h=0.8, rho_e=0.4,
            tau_h=1.5, tau_e=1.5):
    """Per-game fatigue for game order m in {1..M}. Scalar or array inputs."""
    m = np.asarray(m, dtype=float)
    if M <= 1:
        l = np.zeros_like(m)
    else:
        l = (m - 1.0) / (M - 1.0)

    hard = np.maximum(0.0, delta_star - delta)
    easy = np.maximum(0.0, delta - delta_star)
    hF = 1.0 - np.exp(-hard / tau_h)
    eF = 1.0 - np.exp(-easy / tau_e)
    return rho0 * l * (1.0 + rho_h * hF + rho_e * eF)


# ---------------------------------------------------------------------------
# 6. Arousal A(delta, u)
# ---------------------------------------------------------------------------
def arousal(delta, u, A_min=0.2, gamma0_A=-0.5, gamma1_A=15.0,
            kappa0_Au=0.1, rho_u=20.0):
    """Utility-gated Gaussian kernel in delta; bandwidth grows with u."""
    delta = np.asarray(delta, dtype=float)
    u = np.asarray(u, dtype=float)
    K = kappa0_Au + rho_u * u
    s = expit(gamma0_A + gamma1_A * (u - 0.5))
    phi = np.exp(-(delta ** 2) / np.maximum(K, 1e-12))
    return A_min + (1.0 - A_min) * (s * phi)


# ---------------------------------------------------------------------------
# 7. Valence V(delta, u)
# ---------------------------------------------------------------------------
def valence(delta, u, gamma0_v=0.0, gamma_Vu=10.0,
            gamma_vh=0.06, gamma_ve=0.06, tau=5.0):
    """tanh-link valence: utility drives sign, delta contributes quadratic penalties."""
    delta = np.asarray(delta, dtype=float)
    u = np.asarray(u, dtype=float)
    hard = np.maximum(0.0, -delta)
    easy = np.maximum(0.0, delta)
    z = (gamma0_v
         + gamma_Vu * (2.0 * u - 1.0)
         - gamma_vh * hard ** 2
         - gamma_ve * easy ** 2)
    return np.tanh(z / tau)


# ---------------------------------------------------------------------------
# 8. Emotional Regulator Emot(A, V)
# ---------------------------------------------------------------------------
def emot(A, V, A_star=0.6, E_min=0.2, kappa_A=0.15, kappa_V=0.5,
         alpha_A=0.1, alpha_V=0.4):
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    A_kernel = np.exp(-((A - A_star) ** 2) / np.maximum(kappa_A, 1e-12))
    V_kernel = np.exp(-((1.0 - V) ** 2) / np.maximum(kappa_V, 1e-12))
    E = (A_kernel ** alpha_A) * (V_kernel ** alpha_V)
    return E_min + (1.0 - E_min) * E


# ---------------------------------------------------------------------------
# 9. Practice Bias B(n, delta)
# ---------------------------------------------------------------------------
def omega_B(delta, B_min=0.05, B_max=0.4, delta_B_star=0.0, kappa_B=5.0):
    delta = np.asarray(delta, dtype=float)
    return B_min + (B_max - B_min) * np.exp(
        -((delta - delta_B_star) ** 2) / np.maximum(kappa_B, 1e-12)
    )


def practice_bias(n, delta, beta0=0.1, beta1=2.0,
                  B_min=0.05, B_max=0.4, delta_B_star=0.0, kappa_B=5.0):
    n = np.asarray(n, dtype=float)
    delta = np.asarray(delta, dtype=float)
    omega = omega_B(delta, B_min, B_max, delta_B_star, kappa_B)
    raw = 1.0 - np.exp(-((beta0 * n) ** beta1))
    return 1.0 + omega * raw
