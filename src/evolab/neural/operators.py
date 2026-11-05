"""src/evolab/neural/operators.py — Algebraic Weight-Space Mutation Operators.

Provides linear-algebraic and spectral mutation operators for neural weight-space evolution:
1. SpectralRescaleMutation: Truncated SVD singular value perturbation.
2. RankOnePerturbationMutation: Low-rank outer-product injection (ΔW = u ⊗ v).
3. LLMHypothesisMutation: Ingestion of structured hypothesis directions from LLM analyst.
4. CounterfactualIntronPruner: Threshold ablation to enforce parsimony.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import svd

from .genome import NeuralWeightGenome


def _sample_normal(r: Any, mean: float = 0.0, std: float = 1.0) -> float:
    if hasattr(r, "gauss"):
        return float(r.gauss(mean, std))
    if hasattr(r, "normal"):
        return float(r.normal(mean, std))
    return float(random.gauss(mean, std))


def _sample_choice(r: Any, seq: Any) -> Any:
    if hasattr(r, "choice"):
        c = r.choice(seq)
        return str(c) if hasattr(c, "dtype") else c
    return random.choice(seq)


class WeightSpaceOperator:
    """Base class for neural weight-space transformation operators."""

    def mutate(self, genome: NeuralWeightGenome, rng: Optional[random.Random] = None, **kwargs: Any) -> NeuralWeightGenome:
        raise NotImplementedError


class SpectralRescaleMutation(WeightSpaceOperator):
    """Truncated SVD singular-value perturbation.

    Decomposes the layer weight matrix W = U Σ V^T and scales selected singular
    vectors to modify representation geometry along principal directions.
    """

    def __init__(self, singular_index: int = 0, scale_factor: float = 1.1) -> None:
        self.singular_index = singular_index
        self.scale_factor = scale_factor

    def mutate(
        self,
        genome: NeuralWeightGenome,
        rng: Optional[random.Random] = None,
        layer_weights: Optional[Dict[str, np.ndarray]] = None,
        **kwargs: Any,
    ) -> NeuralWeightGenome:
        child = genome.clone()
        r = rng or random

        if not layer_weights:
            return child

        available_layers = [name for name, w in layer_weights.items() if w.ndim == 2]
        if not available_layers:
            return child

        target_layer = _sample_choice(r, available_layers)
        W = layer_weights[target_layer]

        # Compute SVD
        try:
            U, s, Vt = svd(W, full_matrices=False)
            k = min(self.singular_index, len(s) - 1)
            # Perturb singular direction: ΔW = (scale - 1) * s_k * (u_k ⊗ v_k)
            delta = (self.scale_factor - 1.0) * s[k] * np.outer(U[:, k], Vt[k, :])

            if target_layer in child.sparse_patches:
                child.sparse_patches[target_layer] = child.sparse_patches[target_layer] + delta
            else:
                child.sparse_patches[target_layer] = delta

            child.edit_lineage.append({
                "op": "svd_spectral_rescale",
                "layer": target_layer,
                "singular_index": k,
                "scale_factor": self.scale_factor,
            })
        except Exception:
            pass

        return child


class RankOnePerturbationMutation(WeightSpaceOperator):
    """Generates low-rank outer-product updates ΔW = η · (u ⊗ v).

    Emulates LoRA rank-1 updates directly in weight space without backpropagation.
    """

    def __init__(self, eta: float = 0.05, rank: int = 1) -> None:
        self.eta = eta
        self.rank = rank

    def mutate(
        self,
        genome: NeuralWeightGenome,
        rng: Optional[random.Random] = None,
        layer_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> NeuralWeightGenome:
        child = genome.clone()
        r = rng or random

        if not layer_shapes:
            # Fall back to active layers or return clone
            return child

        matrix_layers = [name for name, shape in layer_shapes.items() if len(shape) == 2]
        if not matrix_layers:
            return child

        target_layer = _sample_choice(r, matrix_layers)
        d_out, d_in = layer_shapes[target_layer]

        # Random normalized vectors
        u = np.array([_sample_normal(r, 0.0, 1.0) for _ in range(d_out)], dtype=float)
        v = np.array([_sample_normal(r, 0.0, 1.0) for _ in range(d_in)], dtype=float)
        u /= (np.linalg.norm(u) + 1e-8)
        v /= (np.linalg.norm(v) + 1e-8)

        # Delta matrix
        delta = self.eta * np.outer(u, v)

        if target_layer in child.sparse_patches:
            child.sparse_patches[target_layer] = child.sparse_patches[target_layer] + delta
        else:
            child.sparse_patches[target_layer] = delta

        child.edit_lineage.append({
            "op": "rank_one_perturbation",
            "layer": target_layer,
            "eta": self.eta,
        })
        return child


class LLMHypothesisMutation(WeightSpaceOperator):
    """Directly injects targeted parameter delta hypothesis from LLM diagnostic oracle."""

    def __init__(self, hypothesis: Dict[str, Any]) -> None:
        self.hypothesis = hypothesis

    def mutate(
        self,
        genome: NeuralWeightGenome,
        rng: Optional[random.Random] = None,
        **kwargs: Any,
    ) -> NeuralWeightGenome:
        child = genome.clone()
        target_layer = self.hypothesis.get("layer")
        delta = self.hypothesis.get("delta")

        if target_layer and delta is not None:
            delta_arr = np.asarray(delta, dtype=float)
            if target_layer in child.sparse_patches:
                child.sparse_patches[target_layer] = child.sparse_patches[target_layer] + delta_arr
            else:
                child.sparse_patches[target_layer] = np.copy(delta_arr)

            child.edit_lineage.append({
                "op": "llm_hypothesis_injection",
                "layer": target_layer,
                "hypothesis_id": self.hypothesis.get("id", "hyp_0"),
                "reasoning": self.hypothesis.get("reasoning", ""),
            })

        return child


class CounterfactualIntronPruner:
    """Threshold ablation operator that strips insignificant weights to prevent bloat."""

    def __init__(self, threshold: float = 1e-4) -> None:
        self.threshold = threshold

    def prune(self, genome: NeuralWeightGenome) -> int:
        return genome.prune_inactive_weights(self.threshold)
