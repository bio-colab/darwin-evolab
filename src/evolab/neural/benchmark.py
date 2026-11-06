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


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _forward_mlp(x: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    """Forward pass through shallow (ReLU) or deep (GELU) MLP."""
    if "layer3.weight" in weights:
        # Deep architecture: 8 -> 32 -> 16 -> 48 -> 32 -> 16 -> 4 with GELU
        num_layers = len([k for k in weights.keys() if k.endswith(".weight")])
        h = x
        for idx in range(1, num_layers):
            w = weights[f"layer{idx}.weight"]
            b = weights[f"layer{idx}.bias"]
            h = _gelu(np.dot(h, w.T) + b)
        w_last = weights[f"layer{num_layers}.weight"]
        b_last = weights[f"layer{num_layers}.bias"]
        return np.dot(h, w_last.T) + b_last
    else:
        # Standard shallow 3-layer MLP: 8 -> 16 -> 4
        w1 = weights["layer1.weight"]
        b1 = weights["layer1.bias"]
        w2 = weights["layer2.weight"]
        b2 = weights["layer2.bias"]
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
    penultimate_layer: str = "layer2.weight"


class ControlledToyBenchmark:
    """Generates synthetic 4-class classification benchmark with an intentional localized defect.

    Supports both:
    - "shallow": 3-layer MLP (8 -> 16 -> 4) with ReLU activation (default).
    - "deep": Deep 6-layer MLP (8 -> 32 -> 16 -> 48 -> 32 -> 16 -> 4) with GELU activation.
    """

    def __init__(self, seed: int = 42, architecture: str = "shallow") -> None:
        self.rng = np.random.RandomState(seed)
        self.architecture = str(architecture).lower()
        self.num_classes = 4
        self.input_dim = 8

        # 1. Generate clean weights
        if self.architecture == "deep":
            self.clean_weights = self._build_deep_clean_weights()
            self.penultimate_layer = "layer6.weight"
            self.base_model_id = "deep_mlp_8x32x16x48x32x16x4"
        else:
            self.hidden_dim = 16
            self.clean_weights = self._build_clean_weights()
            self.penultimate_layer = "layer2.weight"
            self.base_model_id = "toy_mlp_8x16x4"

        # 2. Generate datasets
        self.train_x, self.train_y = self._generate_data(samples_per_class=25)
        self.test_x, self.test_y = self._generate_data(samples_per_class=20)

        # 3. Create defective weights by damaging class 3 in output layer
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

    def _build_deep_clean_weights(self) -> Dict[str, np.ndarray]:
        """Build deep 6-layer MLP weights (8 -> 32 -> 16 -> 48 -> 32 -> 16 -> 4) with GELU."""
        centers = np.eye(self.num_classes, self.input_dim) * 3.0
        weights: Dict[str, np.ndarray] = {}

        # Layer 1: 32 x 8
        w1 = np.zeros((32, self.input_dim), dtype=float)
        b1 = np.full(32, 0.1, dtype=float)
        for c in range(self.num_classes):
            w1[c * 8 : (c + 1) * 8, :] = centers[c] * 0.25
        weights["layer1.weight"] = w1
        weights["layer1.bias"] = b1

        # Layer 2: 16 x 32
        w2 = np.zeros((16, 32), dtype=float)
        b2 = np.full(16, 0.1, dtype=float)
        for c in range(self.num_classes):
            w2[c * 4 : (c + 1) * 4, c * 8 : c * 8 + 4] = 0.5
        weights["layer2.weight"] = w2
        weights["layer2.bias"] = b2

        # Layer 3: 48 x 16
        w3 = np.zeros((48, 16), dtype=float)
        b3 = np.full(48, 0.1, dtype=float)
        for c in range(self.num_classes):
            w3[c * 12 : (c + 1) * 12, c * 4 : (c + 1) * 4] = 0.5
        weights["layer3.weight"] = w3
        weights["layer3.bias"] = b3

        # Layer 4: 32 x 48
        w4 = np.zeros((32, 48), dtype=float)
        b4 = np.full(32, 0.1, dtype=float)
        for c in range(self.num_classes):
            w4[c * 8 : (c + 1) * 8, c * 12 : c * 12 + 8] = 0.5
        weights["layer4.weight"] = w4
        weights["layer4.bias"] = b4

        # Layer 5: 16 x 32
        w5 = np.zeros((16, 32), dtype=float)
        b5 = np.full(16, 0.1, dtype=float)
        for c in range(self.num_classes):
            w5[c * 4 : (c + 1) * 4, c * 8 : c * 8 + 4] = 0.5
        weights["layer5.weight"] = w5
        weights["layer5.bias"] = b5

        # Layer 6: 4 x 16
        w6 = np.zeros((self.num_classes, 16), dtype=float)
        b6 = np.full(self.num_classes, 0.5, dtype=float)
        for c in range(self.num_classes):
            w6[c, c * 4 : (c + 1) * 4] = 1.0
        weights["layer6.weight"] = w6
        weights["layer6.bias"] = b6

        return weights

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
        # Zero the projection weights for target class in the output layer
        # This causes Class 3 recall to drop to 0% because its logit stays at base bias while other classes dominate
        defective[self.penultimate_layer][target_class, :] = 0.0
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
            name=f"{self.architecture}_4class_defect_repair",
            base_model_id=self.base_model_id,
            base_weights=self.defective_weights,
            layer_shapes=layer_shapes,
            target_class=3,
            fail_to_pass_x=fail_x,
            fail_to_pass_y=fail_y,
            pass_to_pass_x=pass_x,
            pass_to_pass_y=pass_y,
            penultimate_layer=self.penultimate_layer,
        )


class NeuralWeightEvaluator(Evaluator):
    """Dual-invariant evaluator for neural weight candidates."""

    def __init__(
        self,
        spec: NeuralProblemSpec,
        min_fail_acc: float = 96.0,
        min_anchor_acc: float = 99.7,
        max_regressions: int = 0,
    ) -> None:
        self.spec = spec
        self.name = "neural_weight_evaluator"
        self.min_fail_acc = float(min_fail_acc)
        self.min_anchor_acc = float(min_anchor_acc)
        self.max_regressions = int(max_regressions)

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        genome = getattr(target, "genome", target)
        if not isinstance(genome, NeuralWeightGenome):
            return FitnessResult(score=0.0)

        # Clone base weights and apply candidate patch within patch_scope for zero parameter drift
        patched_weights = {k: np.copy(v) for k, v in self.spec.base_weights.items()}
        with genome.patch_scope(patched_weights):
            # 1. Evaluate target suite (FAIL_TO_PASS: Class 3)
            logits_fail = _forward_mlp(self.spec.fail_to_pass_x, patched_weights)
            preds_fail = np.argmax(logits_fail, axis=1)
            fail_acc = float(np.mean(preds_fail == self.spec.fail_to_pass_y) * 100.0)

            # 2. Evaluate anchor suite (PASS_TO_PASS: Classes 0, 1, 2)
            logits_anchor = _forward_mlp(self.spec.pass_to_pass_x, patched_weights)
            preds_anchor = np.argmax(logits_anchor, axis=1)
            anchor_acc = float(np.mean(preds_anchor == self.spec.pass_to_pass_y) * 100.0)

        # 3. Dual Invariant verification
        # 0 regressions required AND anchor accuracy >= min_anchor_acc (99.7%) AND target recall >= min_fail_acc (96.0%)
        regressions = int(np.sum(preds_anchor != self.spec.pass_to_pass_y))
        passed_holdout = bool(
            regressions <= self.max_regressions
            and anchor_acc >= self.min_anchor_acc
            and fail_acc >= self.min_fail_acc
        )

        if regressions > self.max_regressions:
            # Penalize regression severely: each regression drops score by 15 points
            penalty = (regressions - self.max_regressions) * 15.0
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
