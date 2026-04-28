"""
policy — Game-selection and difficulty-adaptation policies.

Classes
-------
BasePolicy      : abstract interface (select / update / reset)
RandomPolicy    : uniform random selection, random difficulty, no adaptation
StaircasePolicy : 3-up-1-down difficulty + round-robin game selection
"""

from .base_policy       import BasePolicy
from .random_policy     import RandomPolicy
from .staircase_policy  import StaircasePolicy

__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "StaircasePolicy",
]
