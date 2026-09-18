"""Darwin-Evolab Dream Subsystem.

Implements offline replay-driven policy improvement, counterfactual operator
reweighting, and Governor-verified self-adaptation based on Dream-RSI
(arXiv:2609.14858, Sept 2026).
"""

from .self_model_dream import (
    DreamReweightingResult,
    OperatorDistribution,
    OperatorReweighter,
    run_dream_reweighting,
    sample_dirichlet_weights,
)

__all__ = [
    "OperatorDistribution",
    "OperatorReweighter",
    "DreamReweightingResult",
    "sample_dirichlet_weights",
    "run_dream_reweighting",
]
