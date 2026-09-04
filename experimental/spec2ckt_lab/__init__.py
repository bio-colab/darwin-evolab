"""Spec2Ckt Experimental Laboratory.

Distilling ARCS (Grammar-Constrained Circuit Synthesis) and CktGen (Spec-Conditioned Generative AI)
into Darwin-Evolab's Evolutionary Operating System.
"""

from .spec_types import TargetCircuitSpec, SpecTolerance
from .grammar_guard import PhysicalGrammarGuard, GrammarCheckResult
from .spec_conditioned_generator import SpecConditionedPrior, SpecConditionedGenerator
from .bandit_latent_optimizer import BanditLatentOptimizer
from .hybrid_evolution_engine import HybridSpecEvolutionEngine

__all__ = [
    "TargetCircuitSpec",
    "SpecTolerance",
    "PhysicalGrammarGuard",
    "GrammarCheckResult",
    "SpecConditionedPrior",
    "SpecConditionedGenerator",
    "BanditLatentOptimizer",
    "HybridSpecEvolutionEngine",
]
