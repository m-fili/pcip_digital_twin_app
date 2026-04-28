"""
estimator — Parameter estimation for the PCIP model.

Classes / functions
-------------------
ModelParameters  : nn.Module holding all trainable parameters with transforms
compute_loss     : differentiable forward pass + log-MSE + L2 regularisation
fit              : main Adam optimisation loop
compare_to_truth : RMSE/correlation vs ground truth (simulation study)
print_comparison : pretty-print compare_to_truth output
run_diagnostics  : comprehensive estimation quality report
"""

from .parameters   import ModelParameters
from .loss         import compute_loss, forward_pass
from .fit          import fit, compare_to_truth, print_comparison
from .diagnostics  import run_diagnostics

__all__ = [
    "ModelParameters",
    "compute_loss",
    "forward_pass",
    "fit",
    "compare_to_truth",
    "print_comparison",
    "run_diagnostics",
]
