"""src/evolab/neural/benchmark.py — Controlled Toy Benchmark for Weight-Space Evolution.

Provides a reproducible, hermetic testbed:
1. 3-layer MLP classification model (8 -> 16 -> 4) with clean baseline weights.
2. Injected defect causing 0% recall on Class 3 while preserving 100% on Classes 0-2.
3. Dual-Invariant evaluation suite: Target Suite (FAIL_TO_PASS) & Anchor Suite (PASS_TO_PASS).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from evolab.evaluators import Evaluator, FitnessResult
from .genome import NeuralWeightGenome


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _forward_mlp(x: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    """Forward pass through 2-layer MLP with ReLU activation."""
    w1 = weights["layer1.weight"]
    b1 = weights["layer1.bias"]
    w2 = weights["layer2.weight"]
    b2 = weights["layer2.bias"]

    # x shape: (N, 8)
    h = _relu(np.dot(x, w1.T) + b1)
    logits = np.dot(h, w2.T) + b2
    return logits


@dataclass
class NeuralProblemSpec:
    """Specification of a neural weight-space optimization problem."""
    name: str
    base_model_id: str
    base_weights: Dict[str, np.ndarray]
    layer_shapes: Dict[str, Tuple[int, ...]]
    target_class: int
    fail_to_pass_x: np.ndarray
    fail_to_pass_y: np.ndarray
    pass_to_pass_x: np.ndarray
    pass_to_pass_y: np.ndarray


class ControlledToyBenchmark:
    """Generates synthetic 4-class classification benchmark with an intentional localized defect."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        self.num_classes = 4
        self.input_dim = 8
        self.hidden_dim = 16

        # 1. Generate clean weights
        self.clean_weights = self._build_clean_weights()

        # 2. Generate datasets
        self.train_x, self.train_y = self._generate_data(samples_per_class=25)
        self.test_x, self.test_y = self._generate_data(samples_per_class=20)

        # 3. Create defective weights by damaging class 3 in layer2
        self.defective_weights = self._inject_defect(self.clean_weights, target_class=3)

    def _build_clean_weights(self) -> Dict[str, np.ndarray]:
        # Orthogonal-like centers for each class
        centers = np.eye(self.num_classes, self.input_dim) * 3.0
        # Layer 1: map input features to hidden
        w1 = np.zeros((self.hidden_dim, self.input_dim), dtype=float)
        b1 = np.zeros(self.hidden_dim, dtype=float)
        for c in range(self.num_classes):
            w1[c * 4 : (c + 1) * 4, :] = centers[c] * 0.5
            b1[c * 4 : (c + 1) * 4] = 0.1

        # Layer 2: map hidden to 4 output classes
        w2 = np.zeros((self.num_classes, self.hidden_dim), dtype=float)
        b2 = np.zeros(self.num_classes, dtype=float)
        for c in range(self.num_classes):
            w2[c, c * 4 : (c + 1) * 4] = 1.0
            b2[c] = 0.5

        return {
            "layer1.weight": w1,
            "layer1.bias": b1,
            "layer2.weight": w2,
            "layer2.bias": b2,
        }

    def _generate_data(self, samples_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        centers = np.eye(self.num_classes, self.input_dim) * 3.0
        for c in range(self.num_classes):
            noise = self.rng.randn(samples_per_class, self.input_dim) * 0.2
            cluster = centers[c] + noise
            xs.append(cluster)
            ys.append(np.full(samples_per_class, c, dtype=int))
        return np.vstack(xs), np.concatenate(ys)

    def _inject_defect(self, weights: Dict[str, np.ndarray], target_class: int = 3) -> Dict[str, np.ndarray]:
        defective = {k: np.copy(v) for k, v in weights.items()}
        # Zero the projection weights for target class in Layer 2
        # This causes Class 3 recall to drop to 0% because its logit stays at base bias while other classes dominate
        defective["layer2.weight"][target_class, :] = 0.0
        return defective

    def get_spec(self) -> NeuralProblemSpec:
        """Returns structured problem specification."""
        # Fail to pass: Class 3 test samples
        mask_c3 = self.test_y == 3
        fail_x = self.test_x[mask_c3]
        fail_y = self.test_y[mask_c3]

        # Pass to pass: Classes 0, 1, 2 test samples
        mask_anchor = self.test_y != 3
        pass_x = self.test_x[mask_anchor]
        pass_y = self.test_y[mask_anchor]

        layer_shapes = {k: v.shape for k, v in self.defective_weights.items()}

        return NeuralProblemSpec(
            name="toy_4class_defect_repair",
            base_model_id="toy_mlp_8x16x4",
            base_weights=self.defective_weights,
            layer_shapes=layer_shapes,
            target_class=3,
            fail_to_pass_x=fail_x,
            fail_to_pass_y=fail_y,
            pass_to_pass_x=pass_x,
            pass_to_pass_y=pass_y,
        )


class NeuralWeightEvaluator(Evaluator):
    """Dual-invariant evaluator for neural weight candidates."""

    def __init__(self, spec: NeuralProblemSpec) -> None:
        self.spec = spec
        self.name = "neural_weight_evaluator"

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        genome = getattr(target, "genome", target)
        if not isinstance(genome, NeuralWeightGenome):
            return FitnessResult(score=0.0)

        # Clone base weights and apply candidate patch
        patched_weights = {k: np.copy(v) for k, v in self.spec.base_weights.items()}
        patched_weights = genome.apply_to_model(patched_weights)

        # 1. Evaluate target suite (FAIL_TO_PASS: Class 3)
        logits_fail = _forward_mlp(self.spec.fail_to_pass_x, patched_weights)
        preds_fail = np.argmax(logits_fail, axis=1)
        fail_acc = float(np.mean(preds_fail == self.spec.fail_to_pass_y) * 100.0)

        # 2. Evaluate anchor suite (PASS_TO_PASS: Classes 0, 1, 2)
        logits_anchor = _forward_mlp(self.spec.pass_to_pass_x, patched_weights)
        preds_anchor = np.argmax(logits_anchor, axis=1)
        anchor_acc = float(np.mean(preds_anchor == self.spec.pass_to_pass_y) * 100.0)

        # 3. Dual Invariant verification
        # 100% anchor accuracy required AND >=96% target recall required; any regression penalizes fitness heavily
        regressions = int(np.sum(preds_anchor != self.spec.pass_to_pass_y))
        passed_holdout = bool(regressions == 0 and anchor_acc >= 99.9 and fail_acc >= 96.0)

        if regressions > 0:
            # Penalize regression severely: 1 regression drops score by at least 15 points
            penalty = regressions * 15.0
            score = max(0.0, fail_acc - penalty)
        else:
            score = fail_acc

        sub_scores = {
            "target_defect_recall": round(fail_acc, 2),
            "anchor_suite_accuracy": round(anchor_acc, 2),
            "regressions_count": regressions,
        }

        artifacts = {
            "passed_holdout": passed_holdout,
            "scale": genome.scale,
            "active_patches": list(genome.sparse_patches.keys()),
        }

        return FitnessResult(
            score=round(score, 4),
            sub_scores=sub_scores,
            artifacts=artifacts,
            passed_holdout=passed_holdout,
        )
