"""adapter.py — DomainAdapter Implementation for EvoMaze.

Bridges the universal Darwin-Evolab evolutionary kernel to spatial
maze search-based procedural content generation.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from evolab.adapters import DomainAdapter, register_domain_adapter
from evolab.evaluators import Evaluator
from evolab.genome import Individual

from .evaluator import MazeEvaluator
from .exporter import MazeExporter
from .genome import MazeGenome
from .spec import MazeSpec


class EvoMazeAdapter(DomainAdapter[MazeGenome, MazeSpec, dict[str, Any]]):
    """Canonical domain adapter driver for EvoMaze."""

    @property
    def name(self) -> str:
        return "evomaze"

    def parse_spec(self, raw_input: Any) -> MazeSpec:
        """Parses raw input (dict, JSON string, or MazeSpec) into a validated MazeSpec."""
        if isinstance(raw_input, MazeSpec):
            return raw_input
        if isinstance(raw_input, dict):
            return MazeSpec.from_dict(raw_input)
        if isinstance(raw_input, str):
            p = Path(raw_input)
            if p.exists() and p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    return MazeSpec.from_dict(json.load(f))
            try:
                return MazeSpec.from_dict(json.loads(raw_input))
            except json.JSONDecodeError:
                pass
        return MazeSpec()

    def build_population(
        self, spec: MazeSpec, size: int, rng: random.Random
    ) -> list[Individual]:
        """Initializes a valid population of topological maze genomes."""
        population: list[Individual] = []
        num_edges = (spec.width - 1) * spec.height + spec.width * (spec.height - 1)

        for _ in range(size):
            weights = [rng.random() for _ in range(num_edges)]
            loop_gene = rng.random() if spec.target_loops > 0 else 0.0
            genome = MazeGenome(
                width=spec.width,
                height=spec.height,
                edge_weights=weights,
                loop_threshold=loop_gene,
                start=spec.start,
                exit_pt=spec.exit,
            )
            population.append(Individual(genome=genome, species="spec_maze"))

        return population

    def build_evaluator(self, spec: MazeSpec) -> Evaluator:
        """Constructs a deterministic graph-topology evaluator."""
        return MazeEvaluator(spec)

    def export_solution(
        self,
        individual: Individual,
        spec: MazeSpec,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Exports the optimal maze genome into a deployable tilemap and ASCII artifact."""
        genome = getattr(individual, "genome", individual)
        if not isinstance(genome, MazeGenome):
            raise TypeError(f"Expected MazeGenome, got {type(genome)}")

        exporter = MazeExporter(genome, spec)
        data = exporter.to_dict(include_ascii=True)

        if output_path is not None:
            saved_file = exporter.save_json(output_path)
            data["saved_path"] = str(saved_file)

        return data


# Auto-register driver into Darwin-Evolab central registry
register_domain_adapter("evomaze", EvoMazeAdapter())
register_domain_adapter("maze", EvoMazeAdapter())
