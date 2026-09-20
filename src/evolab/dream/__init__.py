"""Darwin-Evolab Dream Subsystem.

Implements offline replay-driven policy improvement, counterfactual operator
reweighting, and Governor-verified self-adaptation based on Dream-RSI
(arXiv:2609.14858, Sept 2026).
"""

from .budget_elasticity import (
    BudgetElasticityOptimizer,
    BudgetElasticityPolicy,
    DreamElasticityResult,
    ElasticityConfig,
    run_budget_elasticity_dreaming,
)
from .cross_validated_seeding import (
    CrossValidatedSeedingOptimizer,
    CrossValidatedSeedingResult,
    SeedCandidate,
    SeedingConfig,
    SeedingPolicy,
    compute_seed_affinity,
    mine_seeds_from_trees,
    run_cross_validated_seeding,
    train_test_split_trees,
)
from .self_model_dream import (
    DreamReweightingResult,
    OperatorDistribution,
    OperatorReweighter,
    run_dream_reweighting,
    sample_dirichlet_weights,
)

# Friendly alias
run_dream_budget_elasticity = run_budget_elasticity_dreaming

__all__ = [
    "OperatorDistribution",
    "OperatorReweighter",
    "DreamReweightingResult",
    "sample_dirichlet_weights",
    "run_dream_reweighting",
    "ElasticityConfig",
    "BudgetElasticityPolicy",
    "DreamElasticityResult",
    "BudgetElasticityOptimizer",
    "run_budget_elasticity_dreaming",
    "SeedCandidate",
    "SeedingConfig",
    "SeedingPolicy",
    "CrossValidatedSeedingResult",
    "CrossValidatedSeedingOptimizer",
    "compute_seed_affinity",
    "mine_seeds_from_trees",
    "train_test_split_trees",
    "run_cross_validated_seeding",
]

