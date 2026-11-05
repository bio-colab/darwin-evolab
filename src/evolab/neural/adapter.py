"""src/evolab/neural/adapter.py — DomainAdapter for Neural Weight-Space Programming.

Connects Darwin-Evolab's core evolutionary engine to neural parameter manifolds.
Fulfills the canonical DomainAdapter contract.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from evolab.adapters import DomainAdapter, register_domain_adapter
from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import Individual

from .analyst import MockWeightAnalyst
from .benchmark import ControlledToyBenchmark, NeuralProblemSpec, NeuralWeightEvaluator
from .genome import NeuralWeightGenome
from .operators import RankOnePerturbationMutation, SpectralRescaleMutation


class NeuralWeightAdapter(DomainAdapter[NeuralWeightGenome, NeuralProblemSpec, Dict[str, Any]]):
    """Universal DomainAdapter for neural weight-space evolution and surgery."""

    @property
    def name(self) -> str:
        return "neural_weight_space"

    def parse_spec(self, raw_input: Any) -> NeuralProblemSpec:
        """Parses and validates domain-specific problem specification."""
        if isinstance(raw_input, NeuralProblemSpec):
            return raw_input

        if isinstance(raw_input, dict):
            scenario = raw_input.get("scenario", "toy_4class_defect_repair")
            if scenario == "toy_4class_defect_repair":
                seed = int(raw_input.get("seed", 42))
                bench = ControlledToyBenchmark(seed=seed)
                return bench.get_spec()

        # Default fallback
        bench = ControlledToyBenchmark()
        return bench.get_spec()

    def build_population(
        self,
        spec: NeuralProblemSpec,
        size: int,
        rng: random.Random,
    ) -> List[Individual]:
        """Initializes a diverse population of candidate weight-space genomes."""
        population: List[Individual] = []

        # 1. Individual 0: Baseline zero-patch candidate
        g0 = NeuralWeightGenome(base_model_id=spec.base_model_id)
        population.append(Individual(genome=g0, species="spec_neural_weight", _index=0))

        # 2. Individual 1: LLM/Analyst Hypothesis seed
        analyst = MockWeightAnalyst()
        hypotheses = analyst.diagnose_and_propose(
            base_weights=spec.base_weights,
            failing_inputs=spec.fail_to_pass_x,
            target_class=spec.target_class,
            penultimate_layer="layer2.weight",
        )

        if hypotheses:
            h = hypotheses[0]
            g_hyp = NeuralWeightGenome(
                base_model_id=spec.base_model_id,
                sparse_patches={h.target_layer: np.copy(h.delta)},
                scale=1.0,
                edit_lineage=[{"op": "analyst_hypothesis", "reasoning": h.reasoning}],
            )
            population.append(Individual(genome=g_hyp, species="spec_neural_weight", _index=1))

        # 3. Fill remaining population with algebraic mutations (Rank-1 and SVD perturbations)
        rank1_op = RankOnePerturbationMutation(eta=0.1)
        svd_op = SpectralRescaleMutation(singular_index=0, scale_factor=1.2)

        while len(population) < size:
            idx = len(population)
            parent_g = g_hyp.clone() if hypotheses and rng.random() < 0.6 else g0.clone()

            if rng.random() < 0.5:
                child_g = rank1_op.mutate(parent_g, rng=rng, layer_shapes=spec.layer_shapes)
            else:
                child_g = svd_op.mutate(parent_g, rng=rng, layer_weights=spec.base_weights)

            child_g.scale = float(rng.uniform(0.5, 1.8))
            population.append(Individual(genome=child_g, species="spec_neural_weight", _index=idx))

        return population

    def build_evaluator(self, spec: NeuralProblemSpec) -> Evaluator:
        """Constructs dual-invariant neural evaluator."""
        return NeuralWeightEvaluator(spec)

    def export_solution(
        self,
        individual: Individual,
        spec: NeuralProblemSpec,
        output_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Exports winning genome and generates deployable artifacts."""
        genome: NeuralWeightGenome = individual.genome
        patched_weights = {k: np.copy(v) for k, v in spec.base_weights.items()}
        patched_weights = genome.apply_to_model(patched_weights)

        # Compute final verification
        evaluator = self.build_evaluator(spec)
        res = evaluator.evaluate(individual)

        summary = {
            "status": "success" if res.passed_holdout and res.score >= 99.0 else "partial",
            "base_model_id": spec.base_model_id,
            "best_individual_id": individual.id,
            "fitness_score": res.score,
            "sub_scores": res.sub_scores,
            "passed_holdout": res.passed_holdout,
            "genome_description": genome.describe(),
            "genome_serialization": genome.serialize(),
        }

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        return summary


# Register the domain adapter globally into evolab's registry
register_domain_adapter("neural_weight_space", NeuralWeightAdapter())
