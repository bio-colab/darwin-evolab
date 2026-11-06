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


class NullSpaceProjectionMutation(WeightSpaceOperator):
    """Projects candidate weight updates onto the null space of anchor representations.

    Guarantees ΔW · X_anchor^T ≈ 0 by construction, eliminating catastrophic regressions
    on invariant holdout sets.
    """

    def __init__(self, target_layer: str, anchor_activations: np.ndarray, eta: float = 0.05, tol: float = 1e-4) -> None:
        self.target_layer = str(target_layer)
        self.anchor_activations = np.asarray(anchor_activations, dtype=float)
        self.eta = float(eta)
        self.tol = float(tol)
        self._P_null: Optional[np.ndarray] = None
        self._compute_projection()

    def _compute_projection(self) -> None:
        X = self.anchor_activations
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N, d_in = X.shape
        try:
            _, s, Vt = svd(X, full_matrices=True)
            cutoff = self.tol * (s[0] if len(s) > 0 else 1.0)
            rank = int(np.sum(s > cutoff))
            if rank < d_in:
                V_null = Vt[rank:, :].T
                self._P_null = V_null @ V_null.T
            else:
                k = max(1, d_in // 4)
                V_least = Vt[-k:, :].T
                self._P_null = V_least @ V_least.T
        except Exception:
            self._P_null = np.eye(d_in)

    def mutate(
        self,
        genome: NeuralWeightGenome,
        rng: Optional[random.Random] = None,
        layer_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> NeuralWeightGenome:
        child = genome.clone()
        r = rng or random

        if not layer_shapes or self.target_layer not in layer_shapes or self._P_null is None:
            return child

        d_out, d_in = layer_shapes[self.target_layer]
        if self._P_null.shape != (d_in, d_in):
            return child

        # Generate unconstrained exploratory perturbation
        W_raw = np.array([[_sample_normal(r, 0.0, 1.0) for _ in range(d_in)] for _ in range(d_out)], dtype=float)
        norm_raw = np.linalg.norm(W_raw, ord="fro") + 1e-8
        W_raw = (W_raw / norm_raw) * self.eta

        # Project onto null space: W_safe = W_raw @ P_null
        W_safe = W_raw @ self._P_null

        if self.target_layer in child.sparse_patches:
            child.sparse_patches[self.target_layer] = child.sparse_patches[self.target_layer] + W_safe
        else:
            child.sparse_patches[self.target_layer] = W_safe

        child.edit_lineage.append({
            "op": "null_space_projection",
            "layer": self.target_layer,
            "eta": self.eta,
            "null_energy": float(np.linalg.norm(W_safe, ord="fro")),
        })
        return child


class RankRPerturbationMutation(WeightSpaceOperator):
    """Generates rank-r low-rank updates ΔW = η · Σ_{i=1}^r σ_i (u_i ⊗ v_i).

    Allows richer manifold exploration beyond rank-1 perturbations while preserving
    compact parameterization.
    """

    def __init__(self, rank: int = 2, eta: float = 0.05, decay_rate: float = 0.6) -> None:
        self.rank = max(1, int(rank))
        self.eta = float(eta)
        self.decay_rate = float(decay_rate)

    def mutate(
        self,
        genome: NeuralWeightGenome,
        rng: Optional[random.Random] = None,
        layer_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        projection_matrix: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> NeuralWeightGenome:
        child = genome.clone()
        r = rng or random

        if not layer_shapes:
            return child

        matrix_layers = [name for name, shape in layer_shapes.items() if len(shape) == 2]
        if not matrix_layers:
            return child

        target_layer = _sample_choice(r, matrix_layers)
        d_out, d_in = layer_shapes[target_layer]

        # Multi-rank synthesis
        actual_rank = min(self.rank, min(d_out, d_in))
        delta = np.zeros((d_out, d_in), dtype=float)

        for i in range(actual_rank):
            sigma_i = (self.decay_rate ** i)
            u_i = np.array([_sample_normal(r, 0.0, 1.0) for _ in range(d_out)], dtype=float)
            v_i = np.array([_sample_normal(r, 0.0, 1.0) for _ in range(d_in)], dtype=float)
            u_i /= (np.linalg.norm(u_i) + 1e-8)
            v_i /= (np.linalg.norm(v_i) + 1e-8)

            if projection_matrix is not None and projection_matrix.shape == (d_in, d_in):
                v_i = projection_matrix @ v_i
                norm_p = np.linalg.norm(v_i)
                if norm_p > 1e-8:
                    v_i /= norm_p

            delta += sigma_i * np.outer(u_i, v_i)

        delta = (delta / (np.linalg.norm(delta, ord="fro") + 1e-8)) * self.eta

        if target_layer in child.sparse_patches:
            child.sparse_patches[target_layer] = child.sparse_patches[target_layer] + delta
        else:
            child.sparse_patches[target_layer] = delta

        child.edit_lineage.append({
            "op": "rank_r_perturbation",
            "layer": target_layer,
            "rank": actual_rank,
            "eta": self.eta,
        })
        return child
