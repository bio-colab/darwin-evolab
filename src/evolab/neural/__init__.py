"""src/evolab/neural/ — Neural Weight-Space Programming (WSP) Extension for Darwin-Evolab.

Enables programmatic inspection, mutation, and evolutionary synthesis of neural network weights
without backpropagation, adhering to strict Dual-Invariant governance (Zero Regressions).
"""
from __future__ import annotations

from .adapter import NeuralWeightAdapter
from .analyst import LLMWeightAnalyst, MockWeightAnalyst, WeightHypothesis
from .benchmark import ControlledToyBenchmark, NeuralProblemSpec, NeuralWeightEvaluator
from .experiment import (
    run_group_a_blind_ga,
    run_group_b_oneshot_llm,
    run_group_c_darwin_wsp,
    run_tri_factor_benchmark,
)
from .genome import NeuralWeightGenome
from .operators import (
    CounterfactualIntronPruner,
    LLMHypothesisMutation,
    RankOnePerturbationMutation,
    SpectralRescaleMutation,
    WeightSpaceOperator,
)

__all__ = [
    "NeuralWeightGenome",
    "NeuralWeightAdapter",
    "NeuralProblemSpec",
    "NeuralWeightEvaluator",
    "ControlledToyBenchmark",
    "WeightSpaceOperator",
    "SpectralRescaleMutation",
    "RankOnePerturbationMutation",
    "LLMHypothesisMutation",
    "CounterfactualIntronPruner",
    "WeightHypothesis",
    "MockWeightAnalyst",
    "LLMWeightAnalyst",
]
