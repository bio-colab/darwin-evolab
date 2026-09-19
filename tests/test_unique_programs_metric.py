"""tests/test_unique_programs_metric.py — Tests for Unique Programs Diversity Metric."""

from __future__ import annotations

import random
from typing import Any

import pytest

from evolab.code_fixtures import make_code_population, scenario_click_parser
from evolab.engine import EvolutionEngine
from evolab.engine_telemetry import calculate_unique_programs
from evolab.genome import Individual
from evolab.repair import RepairEdit, RepairGenome
from evolab.self_model import diagnose_run


def test_calculate_unique_programs_numeric():
    """Verify unique program calculation for numeric/vector populations."""
    # 1. Fully identical population
    pop_identical = [
        Individual(genome=[1.0, 2.0, 3.0], species="spec_0", fitness=10.0)
        for _ in range(10)
    ]
    res_ident = calculate_unique_programs(pop_identical)
    assert res_ident["unique_count"] == 1
    assert res_ident["unique_ratio"] == 0.1

    # 2. Fully distinct population
    pop_distinct = [
        Individual(genome=[float(i), 0.0, 0.0], species="spec_0", fitness=10.0)
        for i in range(10)
    ]
    res_dist = calculate_unique_programs(pop_distinct)
    assert res_dist["unique_count"] == 10
    assert res_dist["unique_ratio"] == 1.0

    # 3. Empty population
    assert calculate_unique_programs([]) == {"unique_count": 0, "unique_ratio": 0.0}


def test_calculate_unique_programs_code():
    """Verify unique program calculation for code/AST repair genomes."""
    from evolab.code_fixtures import scenario_click_parser
    from evolab.repair import catalog_sources

    sc = scenario_click_parser()
    cat = catalog_sources(sc.sources)

    g1 = RepairGenome(sources=sc.sources, target_file=sc.target_file, edits=[])
    g2 = RepairGenome(sources=sc.sources, target_file=sc.target_file, edits=[cat[0]])
    g3 = RepairGenome(sources=sc.sources, target_file=sc.target_file, edits=[cat[1]])

    pop = [
        Individual(genome=g1, species="spec_0"),
        Individual(genome=g1, species="spec_0"),
        Individual(genome=g2, species="spec_0"),
        Individual(genome=g3, species="spec_0"),
    ]

    res = calculate_unique_programs(pop)
    assert res["unique_count"] == 3
    assert res["unique_ratio"] == 0.75


def test_engine_records_unique_programs_in_history_and_report():
    """Verify EvolutionEngine tracks unique_programs and report_builder outputs them."""
    flat_fitness = lambda ind: 50.0
    engine = EvolutionEngine(
        fitness_fn=flat_fitness,
        population_size=10,
        seed=42,
    )
    report = engine.run(3)

    assert "unique_programs_final" in report
    assert "unique_programs_ratio_final" in report
    assert report["unique_programs_final"] is not None
    assert 0.0 < report["unique_programs_ratio_final"] <= 1.0

    for gen_data in report["history"]:
        assert "unique_programs" in gen_data
        assert "unique_programs_ratio" in gen_data
        assert 1 <= gen_data["unique_programs"] <= 10
        assert 0.1 <= gen_data["unique_programs_ratio"] <= 1.0


def test_self_model_interoception_reflects_unique_programs():
    """Verify diagnose_run captures unique_programs telemetry in inner state."""
    history = [
        {
            "generation": i + 1,
            "best_fitness": 80.0,
            "mean_fitness": 79.0,
            "diversity": 0.5,
            "unique_programs": 8 if i < 4 else 6,
            "unique_programs_ratio": 0.8 if i < 4 else 0.6,
            "energy_spent": 10,
        }
        for i in range(5)
    ]

    diag = diagnose_run(history)
    assert "inner" in diag
    assert diag["inner"]["unique_programs"] == 6
    assert diag["inner"]["unique_programs_ratio"] == 0.6
