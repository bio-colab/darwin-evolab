"""
adapter.py — Canonical DomainAdapter connecting Neuromorphic Motifs to Darwin-Evolab.
"""
from __future__ import annotations

import random
from typing import Any

from evolab.adapters import DomainAdapter, register_domain_adapter
from evolab.evaluators import Evaluator
from evolab.genome import Individual

from .evaluator import NeuromorphicEvaluator
from .exporter import NeuromorphicExporter
from .genome import NeuromorphicCircuitGenome
from .motifs import generate_emd_dataset, generate_olfactory_dataset
from .spec import NeuromorphicSpec


class NeuromorphicAdapter(DomainAdapter):
    """
    Domain Adapter driver for synthesizing biological Drosophila neural circuits
    and exporting synthesizable FPGA Verilog RTL modules.
    """

    @property
    def name(self) -> str:
        return "neuromorphic"

    def parse_spec(self, raw_input: Any) -> NeuromorphicSpec:
        if isinstance(raw_input, NeuromorphicSpec):
            return raw_input
        if isinstance(raw_input, dict):
            return NeuromorphicSpec(
                motif_name=str(raw_input.get("motif_name", "hassenstein_reichardt_emd")),
                num_inputs=int(raw_input.get("num_inputs", 2)),
                num_outputs=int(raw_input.get("num_outputs", 2)),
                sequence_length=int(raw_input.get("sequence_length", 24)),
                max_nodes=int(raw_input.get("max_nodes", 16)),
                target_accuracy=float(raw_input.get("target_accuracy", 95.0)),
                clocked=bool(raw_input.get("clocked", True)),
                metadata=dict(raw_input.get("metadata", {})),
            )
        return NeuromorphicSpec()

    def build_population(
        self, spec: NeuromorphicSpec, size: int, rng: random.Random
    ) -> list[Individual]:
        return [
            Individual(
                genome=NeuromorphicCircuitGenome.random(
                    num_inputs=spec.num_inputs,
                    num_outputs=spec.num_outputs,
                    num_nodes=spec.max_nodes,
                    rng=rng,
                ),
                species="spec_neuromorphic",
            )
            for _ in range(size)
        ]

    def build_evaluator(self, spec: NeuromorphicSpec) -> Evaluator:
        if spec.motif_name == "olfactory_lateral_inhibition":
            dataset = generate_olfactory_dataset(length=spec.sequence_length, seed=42)
        else:
            dataset = generate_emd_dataset(length=spec.sequence_length, seed=42)
        return NeuromorphicEvaluator(spec, dataset)

    def export_solution(
        self, individual: Individual, spec: NeuromorphicSpec
    ) -> dict[str, Any]:
        genome: NeuromorphicCircuitGenome = individual.genome
        module_name = f"drosophila_{spec.motif_name}_motif"
        verilog_code = NeuromorphicExporter.export_verilog(genome, module_name=module_name)
        ascii_diagram = NeuromorphicExporter.export_ascii(genome, spec)

        fitness_val = (
            float(individual.fitness.score)
            if hasattr(individual.fitness, "score")
            else float(individual.fitness)
        )
        return {
            "motif_name": spec.motif_name,
            "fitness": fitness_val,
            "verilog_rtl": verilog_code,
            "ascii_diagram": ascii_diagram,
            "active_nodes": len(genome.active_nodes()),
            "transistor_count": genome.transistor_count(),
            "physical_claim": False,
        }


# Automatically register adapter with central registry
register_domain_adapter("neuromorphic", NeuromorphicAdapter())
