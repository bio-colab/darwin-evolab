"""Tests for Adaptive Budget Elasticity and Stagnation Breaking via Dreaming (Dream-RSI Idea 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.dream import (
    BudgetElasticityOptimizer,
    BudgetElasticityPolicy,
    DreamElasticityResult,
    ElasticityConfig,
    run_budget_elasticity_dreaming,
)
from evolab.replay_simulator import DiscoveryNode, DiscoveryTree, ReplayObjective


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def test_elasticity_config_serialization():
    """Verify ElasticityConfig roundtrip serialization and default invariants."""
    cfg = ElasticityConfig(
        patience=3,
        min_delta=2.5,
        burst_multiplier=4,
        breakthrough_threshold=85.0,
        max_stagnant_depth=12,
    )
    d = cfg.to_dict()
    assert d["patience"] == 3
    assert d["min_delta"] == 2.5
    assert d["burst_multiplier"] == 4
    assert d["breakthrough_threshold"] == 85.0
    assert d["max_stagnant_depth"] == 12

    restored = ElasticityConfig.from_dict(d)
    assert restored.patience == 3
    assert restored.min_delta == 2.5
    assert restored.burst_multiplier == 4
    assert restored.breakthrough_threshold == 85.0
    assert restored.max_stagnant_depth == 12


def test_should_stop_branch_stagnation_detection():
    """Verify should_stop_branch flags branches only after consecutive stagnation."""
    tree = DiscoveryTree(root_id="root")
    tree.add_node(DiscoveryNode(node_id="root", score=0.0))
    # Step 1: Big improvement
    tree.add_node(DiscoveryNode(node_id="s1", parent_id="root", score=40.0))
    # Step 2: Stagnant (delta = 0.0 < min_delta 1.0)
    tree.add_node(DiscoveryNode(node_id="s2", parent_id="s1", score=40.0))
    # Step 3: Stagnant (delta = 0.0 < min_delta 1.0) -> Stagnation count = 2
    tree.add_node(DiscoveryNode(node_id="s3", parent_id="s2", score=40.0))

    policy = BudgetElasticityPolicy(config=ElasticityConfig(patience=2, min_delta=1.0))

    # s1 improved from root (40 > 0) -> should NOT stop
    assert policy.should_stop_branch("s1", tree) is False
    # s2 stagnated for 1 step < patience 2 -> should NOT stop
    assert policy.should_stop_branch("s2", tree) is False
    # s3 stagnated for 2 consecutive steps == patience 2 -> SHOULD stop
    assert policy.should_stop_branch("s3", tree) is True


def test_should_stop_branch_never_stops_resolved_solutions():
    """Verify solutions that achieve 100% or pass holdout are never stopped."""
    tree = DiscoveryTree(root_id="root")
    tree.add_node(DiscoveryNode(node_id="root", score=0.0))
    tree.add_node(DiscoveryNode(node_id="s1", parent_id="root", score=100.0, passed_holdout=True))
    tree.add_node(DiscoveryNode(node_id="s2", parent_id="s1", score=100.0, passed_holdout=True))

    policy = BudgetElasticityPolicy(config=ElasticityConfig(patience=1))
    assert policy.should_stop_branch("s2", tree) is False


def test_conservation_mode_empty_batch():
    """Verify that when all leaves have stagnated, policy enters Conservation Mode and halts."""
    tree = DiscoveryTree(root_id="root")
    tree.add_node(DiscoveryNode(node_id="root", score=0.0))
    # Both leaves stagnate
    tree.add_node(DiscoveryNode(node_id="b1", parent_id="root", score=20.0))
    tree.add_node(DiscoveryNode(node_id="b1_stag", parent_id="b1", score=20.0))

    tree.add_node(DiscoveryNode(node_id="b2", parent_id="root", score=30.0))
    tree.add_node(DiscoveryNode(node_id="b2_stag", parent_id="b2", score=30.0))

    policy = BudgetElasticityPolicy(config=ElasticityConfig(patience=1, min_delta=1.0))
    # Both leaves b1_stag and b2_stag should be flagged as dead ends
    chosen = policy.select_batch(tree, batch_width=2)
    # Conservation mode returns empty list to stop burning budget
    assert chosen == []


def test_burst_mode_expands_width():
    """Verify that breakthrough nodes trigger burst expansion with multiplied width."""
    tree = DiscoveryTree(root_id="root")
    tree.add_node(DiscoveryNode(node_id="root", score=0.0))
    # Normal leaf
    tree.add_node(DiscoveryNode(node_id="norm", parent_id="root", score=30.0))
    # Breakthrough leaf (score 90.0 >= 80.0 threshold)
    tree.add_node(DiscoveryNode(node_id="breakthrough", parent_id="root", score=90.0))
    tree.add_node(DiscoveryNode(node_id="other", parent_id="root", score=50.0))

    policy = BudgetElasticityPolicy(
        config=ElasticityConfig(breakthrough_threshold=80.0, burst_multiplier=3, patience=5)
    )

    # With batch_width=1, burst expansion should expand width to 1 * 3 = 3
    chosen = policy.select_batch(tree, batch_width=1)
    assert len(chosen) >= 2
    # Highest scoring breakthrough node should be first
    assert chosen[0] == "breakthrough"


def test_budget_elasticity_optimizer_governor_acceptance():
    """Verify that BudgetElasticityOptimizer achieves Governor ACCEPT with >= 25% savings and 0 regressions."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    assert report_file.is_file(), f"Missing artifact: {report_file}"

    trees = DiscoveryTree.from_swe_bench_report(report_file)
    optimizer = BudgetElasticityOptimizer(trees=trees)

    result = optimizer.optimize(n_samples=500, seed=42)

    # Governor Acceptance Verification
    assert result.governor_verdict["decision"] == "ACCEPT"
    assert result.governor_verdict["reasons"] == ["all_gates_passed"]
    assert result.governor_verdict["regressions"] == 0
    assert result.governor_verdict["worst_c"] >= result.governor_verdict["worst_b"]

    # Quantitative savings target: 25% to 40%
    assert result.mean_evaluations_saved_percent >= 25.0
    assert result.delta_mean_value > 0.0

    # 100% solution retention (no regressions on solved instances)
    assert result.instances_retained_percent == 100.0


def test_run_budget_elasticity_dreaming_artifact_export(tmp_path: Path):
    """Verify run_budget_elasticity_dreaming writes valid, complete artifact."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    out_file = tmp_path / "dream_elasticity_test.json"

    result = run_budget_elasticity_dreaming(
        report_path=report_file,
        output_report_path=out_file,
        n_samples=200,
        seed=101,
    )

    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))

    assert "optimal_config" in data
    assert "governor_verdict" in data
    assert data["governor_verdict"]["decision"] == "ACCEPT"
    assert data["mean_evaluations_saved_percent"] >= 25.0
    assert len(data["per_instance_records"]) == 10
