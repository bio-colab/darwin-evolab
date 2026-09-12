"""
evaluator.py — Biologically grounded evaluator for Neuromorphic Connectome Motifs.

Evaluates candidate circuits against Drosophila sensory-response benchmarks,
measuring firing accuracy, directional selectivity index (DSI), and silicon parsimony.
"""
from __future__ import annotations

from typing import Any

from evolab.evaluators import Evaluator, FitnessResult
from .genome import NeuromorphicCircuitGenome
from .motifs import MotifDataset
from .spec import NeuromorphicSpec


class NeuromorphicEvaluator(Evaluator):
    """
    Evaluates temporal neuromorphic circuits against Drosophila connectome benchmark datasets.
    """

    def __init__(self, spec: NeuromorphicSpec, dataset: MotifDataset) -> None:
        self.spec = spec
        self.dataset = dataset

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        genome = getattr(target, "genome", target)
        if not isinstance(genome, NeuromorphicCircuitGenome):
            return FitnessResult(score=0.0)

        # Execute temporal simulation across the sensory benchmark
        sim_outputs = genome.simulate_sequence(self.dataset.inputs)

        # 1. Functional accuracy against expected spike trains
        total_bits = len(self.dataset.expected_outputs) * self.spec.num_outputs
        matching_bits = 0
        true_positives = 0
        false_positives = 0
        target_positives = 0

        for sim_step, exp_step in zip(sim_outputs, self.dataset.expected_outputs):
            for s_bit, e_bit in zip(sim_step, exp_step):
                if s_bit == e_bit:
                    matching_bits += 1
                if e_bit == 1:
                    target_positives += 1
                    if s_bit == 1:
                        true_positives += 1
                elif s_bit == 1 and e_bit == 0:
                    false_positives += 1

        accuracy = (matching_bits / max(total_bits, 1)) * 100.0

        # 2. Directional Selectivity / Sensitivity (TPR)
        tpr = (true_positives / max(target_positives, 1)) * 100.0 if target_positives > 0 else 100.0
        fp_rate = (false_positives / max(total_bits - target_positives, 1)) * 100.0

        # 3. Parsimony pressure (Koza-style small penalty for transistor bloat)
        active_nodes = len(genome.active_nodes())
        transistor_count = genome.transistor_count()
        parsimony_penalty = min(5.0, (transistor_count / 20.0) * 0.5)

        # Balanced score: 70% overall bit accuracy + 30% event sensitivity - parsimony
        raw_score = (accuracy * 0.70) + (tpr * 0.30) - (fp_rate * 0.10) - parsimony_penalty
        final_score = max(0.0, min(100.0, round(raw_score, 4)))

        return FitnessResult(
            score=final_score,
            sub_scores={
                "accuracy": round(accuracy, 2),
                "true_positive_rate": round(tpr, 2),
                "false_positive_rate": round(fp_rate, 2),
                "active_nodes": active_nodes,
                "transistor_count": transistor_count,
            },
            artifacts={
                "motif": self.spec.motif_name,
                "dataset_length": len(self.dataset.inputs),
            },
        )
