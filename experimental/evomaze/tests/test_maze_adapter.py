"""test_maze_adapter.py — Comprehensive Unit and Integration Tests for EvoMaze.

Verifies the mathematical guarantees of solvability, genome contracts,
evaluator topological metric calculations, and engine integration.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import pytest

from evolab.adapters import get_domain_adapter, list_domain_adapters
from evolab.engine import EvolutionEngine
from experimental.evomaze import (
    DisjointSet,
    EvoMazeAdapter,
    MazeEvaluator,
    MazeExporter,
    MazeGenome,
    MazeSpec,
    build_tilemap_grid,
    canonical_edges,
)


def test_disjoint_set() -> None:
    ds = DisjointSet(5)
    assert ds.find(0) == 0
    assert ds.find(1) == 1
    assert ds.union(0, 1) is True
    assert ds.find(0) == ds.find(1)
    # Redundant union
    assert ds.union(0, 1) is False
    assert ds.union(2, 3) is True
    assert ds.union(1, 3) is True
    assert ds.find(0) == ds.find(2)


def test_spec_validation() -> None:
    spec = MazeSpec(width=11, height=11, start=(0, 0), exit=(10, 10))
    assert spec.width == 11
    assert spec.height == 11

    # To/From dict
    d = spec.to_dict()
    restored = MazeSpec.from_dict(d)
    assert restored.width == spec.width
    assert restored.start == spec.start
    assert restored.exit == spec.exit

    # Invalid dimensions
    with pytest.raises(ValueError, match="at least 3x3"):
        MazeSpec(width=2, height=10)

    # Out of bounds coordinates
    with pytest.raises(ValueError, match="out of bounds"):
        MazeSpec(width=5, height=5, start=(10, 0))

    # Identical start and exit
    with pytest.raises(ValueError, match="cannot be identical"):
        MazeSpec(width=5, height=5, start=(2, 2), exit=(2, 2))


def test_canonical_edges() -> None:
    edges = canonical_edges(3, 3)
    # For a 3x3 grid: horizontal = 2 * 3 = 6, vertical = 3 * 2 = 6, total = 12
    assert len(edges) == 12
    # Ensure all edges are unique
    assert len(set(edges)) == 12


def test_genome_contract() -> None:
    rng = random.Random(42)
    g1 = MazeGenome(width=5, height=5)
    assert len(g1.edge_weights) == (4 * 5 + 5 * 4)

    # Clone
    g2 = g1.clone()
    assert g1.distance_to(g2) == 0.0
    assert g1.fingerprint() == g2.fingerprint()
    assert g1.edge_weights == g2.edge_weights

    # Mutation
    g3 = g1.mutate(rng=rng, mutation_rate=0.5, mutation_scale=0.5)
    assert g1.distance_to(g3) > 0.0
    assert len(g3.edge_weights) == len(g1.edge_weights)

    # Crossover
    child = g1.crossover(g3, rng=rng)
    assert len(child.edge_weights) == len(g1.edge_weights)

    # Serialization
    ser = g1.serialize()
    assert ser["type"] == "MazeGenome"
    assert "fingerprint" in ser
    assert len(ser["edge_weights"]) == len(g1.edge_weights)

    # Behavioral Descriptors
    desc = g1.describe()
    assert "path_length" in desc
    assert "dead_end_count" in desc
    assert "junction_count" in desc
    assert desc["path_length"] > 0


def test_guaranteed_solvability_invariant() -> None:
    """Mathematical Invariant: 100% of generated and mutated mazes MUST be solvable."""
    rng = random.Random(12345)
    sizes = [(5, 5), (7, 7), (11, 9), (15, 15)]

    for w, h in sizes:
        for _ in range(15):
            genome = MazeGenome(width=w, height=h, start=(0, 0), exit_pt=(w - 1, h - 1))
            adj = genome.get_adjacency()

            # Verify complete connectedness via BFS from (0, 0)
            visited = set()
            queue = [(0, 0)]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        queue.append(nbr)

            # Every cell in the grid MUST be reached
            assert len(visited) == w * h, f"Unreachable cells found in {w}x{h} maze!"

            # Start and exit must be connected
            assert (w - 1, h - 1) in visited

            # Now mutate heavily and re-verify
            mutated = genome.mutate(rng=rng, mutation_rate=0.4, mutation_scale=0.4)
            mut_adj = mutated.get_adjacency()
            mut_visited = set()
            mut_queue = [(0, 0)]
            while mut_queue:
                curr = mut_queue.pop(0)
                if curr in mut_visited:
                    continue
                mut_visited.add(curr)
                for nbr in mut_adj[curr]:
                    if nbr not in mut_visited:
                        mut_queue.append(nbr)
            assert len(mut_visited) == w * h, "Mutated maze became disconnected!"


def test_evaluator_metrics() -> None:
    spec = MazeSpec(
        width=7,
        height=7,
        start=(0, 0),
        exit=(6, 6),
        target_path_length=15.0,
        target_dead_ends=5,
        target_loops=1,
    )
    evaluator = MazeEvaluator(spec)
    assert evaluator.deterministic is True

    genome = MazeGenome(width=7, height=7, loop_threshold=0.5)
    res = evaluator.evaluate(genome)

    assert res.score >= 0.0 and res.score <= 100.0
    assert res.passed_holdout is True
    assert "path_length" in res.artifacts
    assert "junction_count" in res.artifacts
    assert "dead_end_count" in res.artifacts
    assert res.artifacts["path_length"] >= 12  # Manhattan distance is 12


def test_exporter_ascii_and_json(tmp_path: Path) -> None:
    spec = MazeSpec(width=5, height=5, start=(0, 0), exit=(4, 4))
    genome = MazeGenome(width=5, height=5, start=(0, 0), exit_pt=(4, 4))
    exporter = MazeExporter(genome, spec)

    ascii_rep = exporter.to_ascii(show_solution=True)
    assert "S " in ascii_rep
    assert "##" in ascii_rep
    # Test custom wall character
    unicode_rep = exporter.to_ascii(wall_char="██")
    assert "██" in unicode_rep

    d = exporter.to_dict()
    assert d["format"] == "darwin-evomaze/v1"
    assert d["logical_dimensions"] == {"width": 5, "height": 5}
    # Expanded grid is (2*5+1) x (2*5+1) = 11 x 11
    assert d["tilemap_dimensions"] == {"width": 11, "height": 11}
    assert len(d["tilemap_grid"]) == 11
    assert len(d["tilemap_grid"][0]) == 11

    # File export
    json_file = tmp_path / "test_maze.json"
    saved_path = exporter.save_json(json_file)
    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metrics"]["solvable"] is True


def test_adapter_and_engine_integration(tmp_path: Path) -> None:
    """Full end-to-end evolutionary optimization using EvolutionEngine and EvoMazeAdapter."""
    adapter = EvoMazeAdapter()
    assert adapter.name == "evomaze"

    spec = adapter.parse_spec({
        "width": 7,
        "height": 7,
        "target_path_length": 20.0,
        "target_dead_ends": 6,
        "target_loops": 0,
        "population_size": 10,
        "generations": 5,
        "seed": 99,
    })

    rng = random.Random(spec.seed)
    population = adapter.build_population(spec, spec.population_size, rng)
    assert len(population) == spec.population_size

    evaluator = adapter.build_evaluator(spec)

    # Run EvolutionEngine
    engine = EvolutionEngine(
        population_size=spec.population_size,
        seed=spec.seed,
        fitness_fn=evaluator,
    )

    report = engine.run(spec.generations, initial_population=population)
    assert report["total_generations"] == spec.generations
    best = report["best_individual"]
    assert best is not None
    assert best["fitness"] > 0.0

    # Test export
    out_json = tmp_path / "evolved_maze.json"
    best_ind = engine.best_ever or engine.population[0]
    artifact = adapter.export_solution(best_ind, spec, output_path=out_json)
    assert "tilemap_grid" in artifact
    assert "metrics" in artifact
    assert out_json.exists()


def test_central_registry_lookup() -> None:
    available = list_domain_adapters()
    assert "evomaze" in available

    adapter1 = get_domain_adapter("evomaze")
    assert isinstance(adapter1, EvoMazeAdapter)

    adapter2 = get_domain_adapter("maze")
    assert isinstance(adapter2, EvoMazeAdapter)
