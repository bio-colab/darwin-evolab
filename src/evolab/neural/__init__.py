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
    run_dwsp_ablation_study,
    run_dwsp_sensitivity_benchmark,
    run_dwsp_noise_sensitivity_check,
)
from .genome import NeuralWeightGenome
from .operators import (
    CounterfactualIntronPruner,
    LLMHypothesisMutation,
    NullSpaceProjectionMutation,
    RankOnePerturbationMutation,
    RankRPerturbationMutation,
    SpectralRescaleMutation,
    WeightSpaceOperator,
)
from .profiler import ActivationProfile, ActivationProfiler, LayerActivationStats

__all__ = [
    "NeuralWeightGenome",
    "NeuralWeightAdapter",
    "NeuralProblemSpec",
    "NeuralWeightEvaluator",
    "ControlledToyBenchmark",
    "WeightSpaceOperator",
    "SpectralRescaleMutation",
    "RankOnePerturbationMutation",
    "RankRPerturbationMutation",
    "NullSpaceProjectionMutation",
    "LLMHypothesisMutation",
    "CounterfactualIntronPruner",
    "WeightHypothesis",
    "MockWeightAnalyst",
    "LLMWeightAnalyst",
    "ActivationProfiler",
    "ActivationProfile",
    "LayerActivationStats",
    "run_group_a_blind_ga",
    "run_group_b_oneshot_llm",
    "run_group_c_darwin_wsp",
    "run_tri_factor_benchmark",
    "run_dwsp_ablation_study",
    "run_dwsp_sensitivity_benchmark",
    "run_dwsp_noise_sensitivity_check",
]
