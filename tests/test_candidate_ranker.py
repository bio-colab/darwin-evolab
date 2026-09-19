"""tests/test_candidate_ranker.py — Tests for System-One / Heuristic candidate ranking in greedy repair."""

from __future__ import annotations

import pytest

from evolab.code_fixtures import SCENARIO_REGISTRY
from evolab.repair import RepairEdit, greedy_repair, greedy_run_report


def test_candidate_ranker_reorders_edits_and_reduces_search():
    """Verify that a candidate_ranker prioritizes target edits, reducing evaluation count."""
    sc = SCENARIO_REGISTRY["click_cli_parser"]()
    evaluator = sc.create_evaluator()

    # 1. Baseline repair without ranker
    g_base, _, evals_base = greedy_repair(
        sources=sc.sources,
        target_file=sc.target_file,
        evaluator=evaluator,
        max_evals=50,
    )
    assert len(g_base.edits) > 0

    # 2. Perfect or heuristic ranker: prioritizes the exact successful edit kind
    winning_kind = g_base.edits[0].kind

    def heuristic_ranker(candidates: list[RepairEdit]) -> list[RepairEdit]:
        # Prioritize candidates matching winning_kind
        return sorted(candidates, key=lambda e: (0 if e.kind == winning_kind else 1))

    g_ranked, _, evals_ranked = greedy_repair(
        sources=sc.sources,
        target_file=sc.target_file,
        evaluator=evaluator,
        max_evals=50,
        candidate_ranker=heuristic_ranker,
    )

    assert len(g_ranked.edits) > 0
    # Search with heuristic ranking should resolve in fewer or equal evaluations
    assert evals_ranked <= evals_base


def test_candidate_ranker_in_greedy_run_report():
    """Verify greedy_run_report accepts and passes candidate_ranker without regressions."""
    sc = SCENARIO_REGISTRY["requests_http_helper"]()
    evaluator = sc.create_evaluator()

    def dummy_ranker(candidates: list[RepairEdit], current: Any = None, ev: Any = None) -> list[RepairEdit]:
        return list(reversed(candidates))

    report = greedy_run_report(
        sources=sc.sources,
        target_file=sc.target_file,
        evaluator=evaluator,
        candidate_ranker=dummy_ranker,
    )
    assert "best_individual" in report
    assert report["best_individual"]["fitness"] >= 99.7
