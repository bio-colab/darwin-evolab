"""Tests validating John R. Koza's Parsimony Pressure and Bloat Elimination (Milestone M1)."""
import pytest
from pathlib import Path

from evolab.repair import (
    RepairGenome,
    RepairEdit,
    parsimony_prune,
    greedy_repair,
    catalog_sources,
)
from evolab.evaluators import FunctionTestEvaluator, SandboxFunctionTestEvaluator
from evolab.adapters import SoftwareRepairAdapter, SoftwareRepairSpec
from evolab.code_fixtures import scenario_click_parser, scenario_requests_auth_url


def test_evaluator_parsimony_penalty_applied():
    """Koza Parsimony: bloated genomes receive a calibrated penalty when parsimony_weight > 0."""
    scenario = scenario_click_parser()
    base_eval = FunctionTestEvaluator(
        base_sources=scenario.sources,
        target_file=scenario.target_file,
        func_name=scenario.func_name,
        test_cases=scenario.test_cases,
        parsimony_weight=0.0,
    )
    parsimonious_eval = FunctionTestEvaluator(
        base_sources=scenario.sources,
        target_file=scenario.target_file,
        func_name=scenario.func_name,
        test_cases=scenario.test_cases,
        parsimony_weight=0.5,
    )

    catalog = catalog_sources(scenario.sources)
    assert len(catalog) >= 3

    # Baseline: empty edits
    empty_genome = RepairGenome(sources=dict(scenario.sources), target_file=scenario.target_file, edits=[])
    res_base_empty = base_eval.evaluate(empty_genome)
    res_pars_empty = parsimonious_eval.evaluate(empty_genome)
    # Zero edits -> zero penalty -> identical scores
    assert res_base_empty.score == res_pars_empty.score

    # Bloated genome: add 3 edits
    bloated_genome = RepairGenome(
        sources=dict(scenario.sources),
        target_file=scenario.target_file,
        edits=catalog[:3],
    )
    res_base_bloat = base_eval.evaluate(bloated_genome)
    res_pars_bloat = parsimonious_eval.evaluate(bloated_genome)

    # Parsimonious score must be strictly less than base score by the penalty amount
    expected_penalty = min(5.0, round(0.5 * 3, 4))
    assert res_pars_bloat.artifacts.get("parsimony_penalty") == expected_penalty
    assert res_pars_bloat.sub_scores.get("parsimony_penalty") == expected_penalty
    assert res_pars_bloat.score == max(0.0, round(res_base_bloat.score - expected_penalty, 2))


def test_sandbox_evaluator_parsimony_penalty():
    """SandboxFunctionTestEvaluator also computes parsimony penalty consistently."""
    scenario = scenario_click_parser()
    evaluator = SandboxFunctionTestEvaluator(
        base_sources=scenario.sources,
        target_file=scenario.target_file,
        func_name=scenario.func_name,
        test_cases=scenario.test_cases,
        parsimony_weight=0.25,
    )
    catalog = catalog_sources(scenario.sources)
    two_edit_genome = RepairGenome(
        sources=dict(scenario.sources),
        target_file=scenario.target_file,
        edits=catalog[:2],
    )
    res = evaluator.evaluate(two_edit_genome)
    assert res.artifacts.get("parsimony_penalty") == 0.5
    assert res.sub_scores.get("parsimony_penalty") == 0.5


def test_parsimony_prune_eliminates_redundant_introns():
    """Koza Shrink Pass: parsimony_prune strips redundant edits from multi-edit solutions."""
    scenario = scenario_click_parser()
    evaluator = FunctionTestEvaluator(
        base_sources=scenario.sources,
        target_file=scenario.target_file,
        func_name=scenario.func_name,
        test_cases=scenario.test_cases,
    )

    # Find the legitimate 3-edit solution for click_cli_parser
    solution, _, _ = greedy_repair(
        sources=scenario.sources,
        target_file=scenario.target_file,
        evaluator=evaluator,
    )
    assert len(solution.edits) == 3
    assert evaluator.evaluate(solution).score == 100.0

    # Add an extra redundant/irrelevant edit to make a 4-edit bloated genome
    catalog = catalog_sources(scenario.sources)
    essential_loci = {e.locus() for e in solution.edits}
    extra_edits = [e for e in catalog if e.locus() not in essential_loci]
    assert len(extra_edits) > 0

    bloated_genome = RepairGenome(
        sources=dict(scenario.sources),
        target_file=scenario.target_file,
        edits=solution.edits + [extra_edits[0]],
    )
    assert len(bloated_genome.edits) == 4

    # Execute Koza Parsimony Shrink
    pruned_genome, evals_used = parsimony_prune(
        sources=scenario.sources,
        target_file=scenario.target_file,
        genome=bloated_genome,
        evaluator=evaluator,
    )

    # Pruned genome must eliminate the redundant edit and retain strictly the 3 essential edits
    assert len(pruned_genome.edits) == 3
    assert {e.locus() for e in pruned_genome.edits} == essential_loci

    # Score and holdout must be 100% optimal
    res_pruned = evaluator.evaluate(pruned_genome)
    assert res_pruned.score == 100.0


def test_greedy_repair_parsimony_lexicographic():
    """Greedy repair with parsimony_shrink produces clean minimal patch on click_cli_parser."""
    scenario = scenario_click_parser()
    evaluator = FunctionTestEvaluator(
        base_sources=scenario.sources,
        target_file=scenario.target_file,
        func_name=scenario.func_name,
        test_cases=scenario.test_cases,
    )
    repaired, history, evals = greedy_repair(
        sources=scenario.sources,
        target_file=scenario.target_file,
        evaluator=evaluator,
        parsimony_shrink=True,
    )
    assert len(repaired.edits) == 3
    assert history[-1]["best_fitness"] == 100.0


def test_adapter_spec_parsimony_weight_integration():
    """SoftwareRepairAdapter respects parsimony_weight when building evaluators."""
    adapter = SoftwareRepairAdapter()
    spec = adapter.parse_spec({
        "sources": {"calc.py": "def add(a, b): return a + b\n"},
        "target_file": "calc.py",
        "func_name": "add",
        "tests": [((1, 2), 3)],
        "parsimony_weight": 0.15,
    })
    assert spec.parsimony_weight == 0.15
    evaluator = adapter.build_evaluator(spec)
    assert getattr(evaluator, "parsimony_weight", None) == 0.15
