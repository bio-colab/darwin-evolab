"""Tests for code evolution fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.code_fixtures import (
    get_all_scenarios,
    scenario_click_parser,
    scenario_lru_cache_logic,
    scenario_multi_file_config,
    scenario_requests_auth_url,
)
from evolab.patch import PatchGenome


def test_scenario_registry_and_initialization():
    scenarios = get_all_scenarios()
    assert len(scenarios) == 4
    names = {s.name for s in scenarios}
    assert "click_cli_parser" in names
    assert "requests_http_helper" in names
    assert "lru_cache_logic" in names
    assert "multi_file_config" in names

    for s in scenarios:
        assert s.target_file in s.sources
        assert len(s.test_cases) > 0
        assert len(s.holdout_cases) > 0


def test_click_parser_scenario_evaluator():
    sc = scenario_click_parser()
    evaluator = sc.create_evaluator()
    res = evaluator.evaluate(PatchGenome())
    assert res.score == 40.0
    assert not res.passed_holdout
    assert "compiled" in res.sub_scores


def test_requests_scenario_evaluator():
    sc = scenario_requests_auth_url()
    evaluator = sc.create_evaluator()
    res = evaluator.evaluate(PatchGenome())
    assert res.score == pytest.approx(46.67, abs=0.01)
    assert not res.passed_holdout


def test_lru_cache_scenario_evaluator():
    sc = scenario_lru_cache_logic()
    evaluator = sc.create_evaluator()
    res = evaluator.evaluate(PatchGenome())
    assert res.score == 36.0
    assert not res.passed_holdout


def test_multi_file_config_scenario():
    sc = scenario_multi_file_config()
    assert len(sc.sources) == 2
    evaluator = sc.create_evaluator()
    res = evaluator.evaluate(PatchGenome())
    assert res.score == 20.0
    assert not res.passed_holdout


def test_code_population_and_engine_loop():
    from evolab.code_fixtures import make_code_population, scenario_click_parser
    from evolab.engine import EvolutionEngine
    import random

    sc = scenario_click_parser()
    rng = random.Random(1)
    pop = make_code_population(sc, size=6, rng=rng)
    assert len(pop) == 6
    assert all(hasattr(ind.genome, "to_code") for ind in pop)

    engine = EvolutionEngine(
        population_size=6,
        seed=1,
        early_stop_fitness=100.0,
        stagnation_patience=8,
        sharing_mode="off",
        fitness_fn=sc.create_evaluator(),
    )
    report = engine.run(3, initial_population=pop)
    assert report["total_generations"] == 3
    assert report["best_individual"]["fitness"] >= 20.0
    assert all(name.startswith("spec_") for name in report["species_distribution"])


def test_repair_genes_compose_and_inherit():
    from evolab.repair import RepairEdit, RepairGenome, catalog_edits, apply_edits
    from evolab.code_fixtures import scenario_click_parser, scenario_lru_cache_logic

    click = scenario_click_parser().sources["cli_parser.py"]
    kinds = {e.kind for e in catalog_edits(click)}
    assert "bool_flip" in kinds
    assert "index_flip" in kinds
    assert "int_wrap" in kinds

    a = RepairGenome(source=click, edits=[e for e in catalog_edits(click) if e.kind == "bool_flip"][:1])
    b = RepairGenome(source=click, edits=[e for e in catalog_edits(click) if e.kind == "index_flip"][:1])
    child = a.crossover(b)
    assert a.edit_keys() <= child.edit_keys()
    assert b.edit_keys() <= child.edit_keys()
    assert "debug" in child.to_code()
    compile(child.to_code(), "<click>", "exec")

    lru = scenario_lru_cache_logic().sources["lru_store.py"]
    lru_kinds = {e.kind for e in catalog_edits(lru)}
    assert "pop_to_front" in lru_kinds
    assert "hit_move_to_end" in lru_kinds
    both = [e for e in catalog_edits(lru) if e.kind in ("pop_to_front", "hit_move_to_end")]
    fixed = apply_edits(lru, both)
    ns = {}
    exec(compile(fixed, "<lru>", "exec"), ns)
    assert ns["manage_lru"](["a", "b", "c", "a", "d"], 3) == ["c", "a", "d"]


def test_greedy_repairs_all_builtin_scenarios():
    from evolab.code_fixtures import get_all_scenarios
    from evolab.repair import greedy_repair

    for sc in get_all_scenarios():
        genome, history, n_eval = greedy_repair(
            sc.sources, sc.target_file, sc.create_evaluator()
        )
        score, hold = sc.create_evaluator().evaluate(genome).score, sc.create_evaluator().evaluate(genome).passed_holdout
        assert score == 100.0, (sc.name, score, genome.to_code())
        assert hold is True, sc.name
        assert n_eval >= 1


def test_load_scenario_file_and_diff(tmp_path):
    from evolab.code_fixtures import load_scenario_file, scenario_click_parser
    from evolab.repair import greedy_repair, unified_source_diff

    sc = scenario_click_parser()
    spec = tmp_path / "click.json"
    spec.write_text(__import__("json").dumps({
        "name": "from_file",
        "target_file": sc.target_file,
        "func_name": sc.func_name,
        "sources": sc.sources,
        "test_cases": [{"args": list(a), "expected": e} for a, e in sc.test_cases],
        "holdout_cases": [{"args": list(a), "expected": e} for a, e in sc.holdout_cases],
    }), encoding="utf-8")
    loaded = load_scenario_file(spec)
    genome, _, _ = greedy_repair(loaded.sources, loaded.target_file, loaded.create_evaluator())
    diff = unified_source_diff(loaded.sources, genome.apply_to())
    assert "--- a/cli_parser.py" in diff
    assert loaded.create_evaluator().evaluate(genome).score == 100.0


def test_register_repair_pattern_extends_catalog():
    import ast
    from evolab.repair import (
        RepairEdit,
        catalog_edits,
        list_repair_patterns,
        make_edit,
        register_repair_pattern,
        unregister_repair_pattern,
    )

    assert "bool_flip" in list_repair_patterns()

    @register_repair_pattern("const_zero")
    def find_const_zero(tree, file=""):
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 8000:
                edit = make_edit("const_zero", node, file)
                if edit:
                    out.append(edit)
        return out

    try:
        src = "x = 8000\n"
        kinds = {e.kind for e in catalog_edits(src)}
        assert "const_zero" in kinds
    finally:
        unregister_repair_pattern("const_zero")
    assert "const_zero" not in list_repair_patterns()
