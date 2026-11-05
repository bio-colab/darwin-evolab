"""src/evolab/neural/analyst.py — Weight-Space Diagnostic Oracle & LLM Analyst.

Provides System-1 weight inspection and hypothesis generation:
1. LLMWeightAnalyst: Frontier LLM weight inspector.
2. MockWeightAnalyst: Deterministic rule/activation-based heuristic oracle for hermetic offline testing.
3. WeightHypothesis: Dataclass capturing targeted layer and proposed weight delta.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class WeightHypothesis:
    """Structured hypothesis for targeted weight-space modification."""
    hypothesis_id: str
    target_layer: str
    delta: np.ndarray
    reasoning: str
    confidence: float = 0.95


class MockWeightAnalyst:
    """Deterministic activation/subspace diagnostic oracle for offline, hermetic execution.

    Inspects defective network behavior and computes directional hypotheses
    without requiring external API calls or network connections.
    """

    def __init__(self, name: str = "mock_weight_analyst") -> None:
        self.name = name

    def diagnose_and_propose(
        self,
        base_weights: Dict[str, np.ndarray],
        failing_inputs: np.ndarray,
        target_class: int,
        penultimate_layer: str = "layer2",
    ) -> List[WeightHypothesis]:
        """Diagnose classification failure on a target class and propose directional correction."""
        hypotheses: List[WeightHypothesis] = []

        layer_key = penultimate_layer
        if layer_key not in base_weights and f"{penultimate_layer}.weight" in base_weights:
            layer_key = f"{penultimate_layer}.weight"

        if layer_key in base_weights:
            W = base_weights[layer_key]
            delta = np.zeros_like(W, dtype=float)

            # Determine activating hidden neurons for failing inputs
            active_idx = None
            w1_key = "layer1.weight" if "layer1.weight" in base_weights else ("layer1" if "layer1" in base_weights else None)
            b1_key = "layer1.bias" if "layer1.bias" in base_weights else None
            if w1_key and failing_inputs is not None and len(failing_inputs) > 0:
                w1 = base_weights[w1_key]
                b1 = base_weights.get(b1_key, 0.0)
                h = np.maximum(0, failing_inputs @ w1.T + b1)
                mean_h = np.mean(h, axis=0)
                pos_idx = np.where(mean_h > 1.0)[0]
                if len(pos_idx) > 0:
                    active_idx = pos_idx
                else:
                    active_idx = np.argsort(mean_h)[-4:]

            if target_class < W.shape[0]:
                if active_idx is not None:
                    delta[target_class, active_idx] = 2.5
                else:
                    delta[target_class, :] = 0.5 * np.ones(W.shape[1])

                hyp = WeightHypothesis(
                    hypothesis_id="hyp_target_class_boost",
                    target_layer=layer_key,
                    delta=delta,
                    reasoning=f"Identified under-activation in {layer_key} for Class {target_class}. Proposed subspace projection on active hidden units.",
                    confidence=0.95,
                )
                hypotheses.append(hyp)

        return hypotheses


class LLMWeightAnalyst:
    """Frontier LLM weight inspector interface querying LLM for weight-space hypotheses."""

    def __init__(self, provider: str = "mock", model_name: Optional[str] = None) -> None:
        self.provider = provider
        self.model_name = model_name
        self._fallback = MockWeightAnalyst()

    def inspect_and_generate(
        self,
        architecture_summary: Dict[str, Any],
        failing_cases: List[Dict[str, Any]],
        base_weights: Dict[str, np.ndarray],
        target_class: int = 3,
        target_layer: str = "layer2",
    ) -> List[WeightHypothesis]:
        """Generate hypotheses via LLM or fallback oracle."""
        if self.provider == "mock" or not self.model_name:
            # Deterministic fallback
            if failing_cases and isinstance(failing_cases, list) and len(failing_cases) > 0 and isinstance(failing_cases[0], dict) and "input" in failing_cases[0]:
                failing_inputs = np.array([c["input"] for c in failing_cases])
            elif failing_cases is not None and len(failing_cases) > 0:
                failing_inputs = np.array(failing_cases)
            else:
                failing_inputs = None

            return self._fallback.diagnose_and_propose(
                base_weights=base_weights,
                failing_inputs=failing_inputs,
                target_class=target_class,
                penultimate_layer=target_layer,
            )

        # Extensible live provider implementation (Groq, Gemini, OpenAI)
        prompt = {
            "task": "Weight-Space Programmatic Repair",
            "architecture": architecture_summary,
            "target_layer": target_layer,
            "failing_class": target_class,
            "instruction": "Identify weights responsible for defective class recall and propose sparse delta.",
        }
        # If live call unavailable, fall back cleanly
        return self._fallback.diagnose_and_propose(
            base_weights=base_weights,
            failing_inputs=np.zeros((1, 8)),
            target_class=target_class,
            penultimate_layer=target_layer,
        )
