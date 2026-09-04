"""
Tests for Deb's Fast Non-Dominated Sorting, Crowding Distance, and NSGA2Engine.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import pytest

from evolab.pareto import (
    Objective,
    MultiObjectiveResult,
    dominates,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    crowded_comparison,
    NSGA2Engine,
    build_silicon_multiobjective_evaluator,
)
from evolab.genome import Individual, FloatGenome
from evolab.cgp_logic import create_random_cgp_genome
from evolab.cli import main


def test_pareto_dominance_logic():
    objs = [
        Objective("cost", direction="minimize"),
        Objective("speed", direction="maximize"),
    ]

    # sol_a is better on cost and speed -> dominates sol_b
    sol_a = {"cost": 10.0, "speed": 100.0}
    sol_b = {"cost": 20.0, "speed": 50.0}
    assert dominates(sol_a, sol_b, objs) is True
    assert dominates(sol_b, sol_a, objs) is False

    # sol_c is better on cost, worse on speed -> neither dominates
    sol_c = {"cost": 5.0, "speed": 40.0}
    assert dominates(sol_a, sol_c, objs) is False
    assert dominates(sol_c, sol_a, objs) is False


def test_fast_non_dominated_sorting():
    objs = [
        Objective("obj1", direction="minimize"),
        Objective("obj2", direction="minimize"),
    ]

    # Points on 2D trade-off
    pts = [
        {"obj1": 1.0, "obj2": 5.0},  # A (non-dominated)
        {"obj1": 2.0, "obj2": 3.0},  # B (non-dominated)
        {"obj1": 5.0, "obj2": 1.0},  # C (non-dominated)
        {"obj1": 3.0, "obj2": 4.0},  # D (dominated by B)
        {"obj1": 6.0, "obj2": 6.0},  # E (dominated by all)
    ]

    inds = [Individual(FloatGenome([float(i)]), species="spec_test") for i in range(len(pts))]

    def score_fn(ind: Individual) -> dict[str, float]:
        idx = int(ind.genome.values[0])
        return pts[idx]

    fronts = fast_non_dominated_sort(inds, objs, score_fn)
    assert len(fronts) >= 2

    # Front 0 must contain A (0), B (1), C (2)
    front_0_indices = {int(ind.genome.values[0]) for ind in fronts[0]}
    assert front_0_indices == {0, 1, 2}


def test_crowding_distance_boundary_assignment():
    objs = [Objective("x", direction="minimize")]
    inds = [
        Individual(FloatGenome([1.0]), species="spec_test"),
        Individual(FloatGenome([2.0]), species="spec_test"),
        Individual(FloatGenome([5.0]), species="spec_test"),
    ]
    for ind in inds:
        ind._pareto_meta = MultiObjectiveResult(scores={"x": ind.genome.values[0]})

    calculate_crowding_distance(inds, objs)
    # Boundary points must receive infinite crowding distance
    assert inds[0]._pareto_meta.crowding_distance == float("inf")
    assert inds[2]._pareto_meta.crowding_distance == float("inf")
    assert 0.0 < inds[1]._pareto_meta.crowding_distance < float("inf")


def test_nsga2_silicon_engine_run(tmp_path):
    # 4-objective Silicon optimization: Half Adder
    truth_table = [
        ((0, 0), (0, 0)),
        ((0, 1), (1, 0)),
        ((1, 0), (1, 0)),
        ((1, 1), (0, 1)),
    ]
    objs, eval_fn = build_silicon_multiobjective_evaluator(truth_table)
    assert len(objs) == 4

    rng = random.Random(42)
    init_pop = [
        Individual(create_random_cgp_genome(2, 2, 8, rng=rng), species="spec_logic")
        for _ in range(8)
    ]

    engine = NSGA2Engine(
        objectives=objs,
        evaluate_vector_fn=eval_fn,
        population_size=8,
        generations=3,
        seed=42,
    )
    result = engine.run(initial_population=init_pop, generations=3)

    assert result["generations"] == 3
    assert len(result["front_0"]) > 0
    assert "correctness" in result["front_0"][0]["scores"]
    assert "power" in result["front_0"][0]["scores"]
    assert "delay" in result["front_0"][0]["scores"]
    assert "area" in result["front_0"][0]["scores"]

    out_file = tmp_path / "pareto_front.json"
    engine.export_pareto_front(out_file)
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["algorithm"] == "NSGA-II"
    assert len(data["frontier"]) == len(result["front_0"])


def test_cli_nsga2_pareto_export(tmp_path):
    out_json = tmp_path / "cli_pareto.json"
    cmd = [
        "evolve",
        "--engine", "nsga2",
        "--expr", "S = A ^ B; C = A & B",
        "-g", "2",
        "-p", "4",
        "--pareto-export", str(out_json),
    ]
    ret = main(cmd)
    assert ret == 0
    assert out_json.is_file()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["algorithm"] == "NSGA-II"
    assert len(data["frontier"]) > 0
