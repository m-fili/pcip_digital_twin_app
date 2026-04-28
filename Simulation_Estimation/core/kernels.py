"""
Utility kernel functions used throughout the PCIP model.

All functions operate on PyTorch tensors and support broadcasting.
Parameters typed as `float | Tensor` accept either Python scalars or tensors,
relying on PyTorch's broadcasting for mixed-type arithmetic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def gaussian_kernel(
    x: Tensor,
    x0: float | Tensor,
    kappa: float | Tensor,
) -> Tensor:
    """
    Gaussian kernel:  φ(x | x₀, κ) = exp(-(x - x₀)² / κ)

    Peaks at x = x₀; bandwidth controlled by κ > 0.
    Used in: Practice Bias modulation, Emotional Regulator, Arousal model.
    """
    return torch.exp(-((x - x0) ** 2) / kappa)


def asymmetric_laplace_kernel(
    x: Tensor,
    x0: float | Tensor,
    kappa_L: float | Tensor,
    kappa_R: float | Tensor,
) -> Tensor:
    """
    Asymmetric Laplace kernel:
        ψ(x | x₀, κ_L, κ_R) = exp(-(x - x₀)₊ / κ_R  -  (x₀ - x)₊ / κ_L)

    Peaks at x = x₀. Decays at different rates on each side:
      - Right side (x > x₀): decay rate 1/κ_R  →  small κ_R = fast decay
      - Left side  (x < x₀): decay rate 1/κ_L  →  large κ_L = slow decay

    In the mismatch effect (κ_L > κ_R): overchallenge (δ < δ*) is penalised
    less severely than equivalent boredom (δ > δ*).
    """
    right_excess = F.relu(x - x0) / kappa_R    # (x − x₀)₊ / κ_R
    left_excess  = F.relu(x0 - x) / kappa_L    # (x₀ − x)₊ / κ_L
    return torch.exp(-right_excess - left_excess)


def softplus_beta(x: Tensor, beta: float) -> Tensor:
    """
    Parametric SoftPlus:  (x)₊^(β) = (1/β) · log(1 + e^{βx})

    Smooth, differentiable approximation of relu(x).
    Approaches relu(x) as β → ∞.

    Fixed at β = 5 in the Valence model (design choice, not estimated).
    """
    # F.softplus uses a numerically stable implementation
    return F.softplus(x * beta) / beta
