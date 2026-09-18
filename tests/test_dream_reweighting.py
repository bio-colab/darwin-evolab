"""Tests for Autonomous Operator Reweighting via Dreaming (Dream-RSI Idea 1)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evolab.dream import (
    DreamReweightingResult,
    OperatorDistribution,
    OperatorReweighter,
    run_dream_reweighting,
    sample_dirichlet_weights,
)
from evolab.replay_simulator import DiscoveryNode, DiscoveryTree, ReplayObjective


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def test_operator_distribution_simplex_invariants():
    """Verify OperatorDistribution normalizes weights to sum to 1.0 and calculates entropy."""
    dist = OperatorDistribution(weights={"InsertGuard": 2.0, "SwapCondition": 1.0, "BoundaryFlip": 1.0})
    weights = dist.weights
    assert abs(sum(weights.values()) - 1.0) < 1e-5
    assert weights["InsertGuard"] == pytest.approx(0.5, abs=1e-3)
    assert weights["SwapCondition"] == pytest.approx(0.25, abs=1e-3)

    # Uniform distribution
    u_dist = OperatorDistribution.uniform(["opA", "opB", "opC", "opD"])
    assert len(u_dist.weights) == 4
    for w in u_dist.weights.values():
        assert w == pytest.approx(0.25, abs=1e-3)

    # Entropy of uniform is ln(K)
    assert u_dist.entropy() == pytest.approx(math.log(4), abs=1e-3)


def test_sample_dirichlet_weights():
    """Verify Dirichlet sampling generates valid probability distributions on the simplex."""
    ops = ["InsertGuard", "BoundaryFlip", "OffByOne", "DeleteStatement"]
    for alpha in [0.2, 1.0, 5.0]:
        dist = sample_dirichlet_weights(ops, alpha=alpha)
        assert abs(sum(dist.weights.values()) - 1.0) < 1e-5
        for w in dist.weights.values():
            assert w >= 0.0


def test_operator_reweighter_annotation_and_extraction():
    """Verify OperatorReweighter recognizes and annotates operators across discovery trees."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    assert report_file.is_file(), f"Missing artifact: {report_file}"

    trees = DiscoveryTree.from_swe_bench_report(report_file)
    assert len(trees) == 10

    reweighter = OperatorReweighter(trees=trees)
    assert len(reweighter.operators) >= 4

    # Check that all non-root nodes in all trees have an annotated operator
    for tree in reweighter.trees:
        for nid, node in tree.nodes.items():
            if nid != tree.root_id:
                assert "operator" in node.metadata
                assert node.metadata["operator"] in reweighter.operators


def test_counterfactual_replay_baseline_equality():
    """Verify that under uniform baseline weights, counterfactual evaluations match baseline search effort."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    trees = DiscoveryTree.from_swe_bench_report(report_file)
    reweighter = OperatorReweighter(trees=trees)

    u_dist = OperatorDistribution.uniform(reweighter.operators)
    mean_val, values, evals_list, regressions = reweighter.evaluate_distribution(u_dist)

    assert regressions == 0
    assert len(values) == 10
    assert len(evals_list) == 10

    # For each tree, baseline evals should match total recorded non-root attempts
    for i, tree in enumerate(trees):
        recorded_attempts = max(1, tree.size() - 1)
        # Counterfactual under uniform weights should equal recorded attempts
        assert evals_list[i] == pytest.approx(float(recorded_attempts), abs=1e-2)


def test_governor_acceptance_and_zero_regression():
    """Verify Dreaming optimization discovers weights that pass all Governor gates with p < 0.05."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    trees = DiscoveryTree.from_swe_bench_report(report_file)
    reweighter = OperatorReweighter(trees=trees)

    # Run optimization with 5,000 Dirichlet candidates
    result = reweighter.optimize(n_samples=5_000, seed=42)

    # Governor verification contract
    assert result.governor_verdict["decision"] == "ACCEPT"
    assert result.governor_verdict["reasons"] == ["all_gates_passed"]
    assert result.governor_verdict["regressions"] == 0
    assert result.delta_mean_value > 0.0
    assert result.mean_evaluations_saved_percent > 0.0

    # Statistical significance
    assert result.p_value < 0.05
    assert result.cohen_d > 0.5  # Substantial positive effect size


def test_dream_reweighting_artifact_export_and_schema(tmp_path: Path):
    """Verify run_dream_reweighting generates valid, complete JSON audit artifact."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    out_file = tmp_path / "dream_reweighting_test.json"

    result = run_dream_reweighting(
        report_path=report_file,
        output_report_path=out_file,
        n_samples=1_000,
        seed=123,
    )

    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))

    assert data["total_candidates_sampled"] == 1000
    assert "baseline_weights" in data
    assert "optimal_weights" in data
    assert "governor_verdict" in data
    assert data["governor_verdict"]["decision"] in ("ACCEPT", "REJECT")
    assert "per_instance_records" in data
    assert len(data["per_instance_records"]) == 10


def test_synthetic_branching_reweighting_speedup():
    """Verify that on branching trees, boosting the winning operator reduces search effort."""
    tree = DiscoveryTree(root_id="root", name="synthetic_apr_tree")
    tree.add_node(DiscoveryNode(node_id="root", score=0.0))

    # Dead ends produced by SwapCondition and DeleteStatement
    for i in range(1, 10):
        tree.add_node(DiscoveryNode(
            node_id=f"dead_{i}",
            parent_id="root",
            score=20.0,
            metadata={"operator": "SwapCondition"},
        ))

    # Winning path reached via InsertGuard
    tree.add_node(DiscoveryNode(
        node_id="win_step_1",
        parent_id="root",
        score=70.0,
        metadata={"operator": "InsertGuard"},
    ))
    tree.add_node(DiscoveryNode(
        node_id="win_step_2",
        parent_id="win_step_1",
        score=100.0,
        passed_holdout=True,
        metadata={"operator": "InsertGuard"},
    ))

    reweighter = OperatorReweighter(
        trees=[tree],
        operators=["InsertGuard", "SwapCondition", "DeleteStatement"],
    )

    baseline_dist = OperatorDistribution.uniform(reweighter.operators)
    _, _, evals_base, _ = reweighter.evaluate_distribution(baseline_dist)

    # Favorable distribution: heavily weight InsertGuard, downweight SwapCondition
    favorable_dist = OperatorDistribution(weights={
        "InsertGuard": 0.80,
        "SwapCondition": 0.10,
        "DeleteStatement": 0.10,
    })
    _, _, evals_fav, _ = reweighter.evaluate_distribution(favorable_dist)

    # Search effort under favorable weights should be significantly lower
    assert evals_fav[0] < evals_base[0]
    savings_pct = (evals_base[0] - evals_fav[0]) / evals_base[0] * 100.0
    assert savings_pct > 30.0  # More than 30% evaluations saved!
