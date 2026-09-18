"""Tests for M8 & M9 Cross-Validated Seeding and Holdout Tree Separation via Dreaming (Dream-RSI Idea 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.dream import (
    CrossValidatedSeedingOptimizer,
    CrossValidatedSeedingResult,
    SeedCandidate,
    SeedingConfig,
    SeedingPolicy,
    compute_seed_affinity,
    mine_seeds_from_trees,
    run_cross_validated_seeding,
    train_test_split_trees,
)
from evolab.replay_simulator import (
    DiscoveryNode,
    DiscoveryTree,
    GreedyBestFirstPolicy,
    ReplayObjective,
)


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def test_seeding_config_serialization():
    """Verify SeedingConfig roundtrip serialization and default invariants."""
    cfg = SeedingConfig(
        affinity_threshold=0.55,
        dead_gate_filtering=True,
        composition_depth=3,
        warm_start_bonus_ratio=0.30,
        max_active_seeds=4,
    )
    d = cfg.to_dict()
    assert d["affinity_threshold"] == 0.55
    assert d["dead_gate_filtering"] is True
    assert d["composition_depth"] == 3
    assert d["warm_start_bonus_ratio"] == 0.30
    assert d["max_active_seeds"] == 4

    restored = SeedingConfig.from_dict(d)
    assert restored.affinity_threshold == 0.55
    assert restored.dead_gate_filtering is True
    assert restored.composition_depth == 3
    assert restored.warm_start_bonus_ratio == 0.30
    assert restored.max_active_seeds == 4


def test_seed_candidate_serialization():
    """Verify SeedCandidate roundtrip serialization."""
    seed = SeedCandidate(
        seed_id="test_seed_01",
        source_tree="pallets__flask-4992",
        source_repo="pallets/flask",
        motif_type="composition",
        target_operators=["InsertGuard", "SwapCondition"],
        pattern_signature="pallets/flask:comp:InsertGuard+SwapCondition",
        confidence=0.92,
        expected_savings=2.5,
        metadata={"depth": 3},
    )
    d = seed.to_dict()
    assert d["seed_id"] == "test_seed_01"
    assert d["source_repo"] == "pallets/flask"
    assert d["target_operators"] == ["InsertGuard", "SwapCondition"]

    restored = SeedCandidate.from_dict(d)
    assert restored.seed_id == "test_seed_01"
    assert restored.confidence == 0.92
    assert restored.expected_savings == 2.5
    assert restored.metadata["depth"] == 3


def test_train_test_split_trees_disjoint():
    """Verify train/test split creates non-empty, strictly disjoint discovery tree sets."""
    trees = [
        DiscoveryTree(root_id="r1", name=f"tree_{i}")
        for i in range(10)
    ]
    train, test = train_test_split_trees(trees, train_ratio=0.6, seed=42)

    assert len(train) == 6
    assert len(test) == 4
    train_names = {t.name for t in train}
    test_names = {t.name for t in test}
    assert train_names.isdisjoint(test_names)

    # Determinism
    train2, test2 = train_test_split_trees(trees, train_ratio=0.6, seed=42)
    assert [t.name for t in train] == [t.name for t in train2]
    assert [t.name for t in test] == [t.name for t in test2]


def test_mine_seeds_from_trees():
    """Verify seed mining extracts composition and dead gate motifs from high-scoring trajectories."""
    tree = DiscoveryTree(root_id="root", name="pallets__test-01")
    tree.add_node(DiscoveryNode(node_id="root", score=0.0, workspace_snapshot={"repo": "pallets/test"}))
    tree.add_node(DiscoveryNode(node_id="n1", parent_id="root", score=0.0, metadata={"operator": "ConstantMutate"}))
    tree.add_node(DiscoveryNode(node_id="n2", parent_id="n1", score=60.0, metadata={"operator": "InsertGuard"}))
    tree.add_node(DiscoveryNode(node_id="n3", parent_id="n2", score=100.0, passed_holdout=True, metadata={"operator": "BoundaryFlip"}))

    seeds = mine_seeds_from_trees([tree], min_score=70.0)
    assert len(seeds) >= 1

    # Composition seed should contain InsertGuard and BoundaryFlip
    comp_seeds = [s for s in seeds if s.motif_type == "composition"]
    assert len(comp_seeds) >= 1
    assert "BoundaryFlip" in comp_seeds[0].target_operators or "InsertGuard" in comp_seeds[0].target_operators
    assert comp_seeds[0].confidence == 1.0


def test_compute_seed_affinity():
    """Verify affinity function rewards exact repo match and shared organizational family."""
    seed = SeedCandidate(
        seed_id="s1",
        source_tree="pallets__flask-4992",
        source_repo="pallets/flask",
        motif_type="composition",
        target_operators=["InsertGuard"],
        pattern_signature="pallets/flask:comp:InsertGuard",
        confidence=1.0,
        expected_savings=2.0,
    )

    # Exact repo match
    tree_exact = DiscoveryTree(root_id="r", name="pallets__flask-1234")
    tree_exact.add_node(DiscoveryNode(node_id="r", workspace_snapshot={"repo": "pallets/flask"}))
    aff_exact = compute_seed_affinity(seed, tree_exact)

    # Shared family match (pallets/click vs pallets/flask)
    tree_family = DiscoveryTree(root_id="r", name="pallets__click-1608")
    tree_family.add_node(DiscoveryNode(node_id="r", workspace_snapshot={"repo": "pallets/click"}))
    aff_family = compute_seed_affinity(seed, tree_family)

    # Unrelated repo (sympy vs pallets)
    tree_unrelated = DiscoveryTree(root_id="r", name="sympy__sympy-13480")
    tree_unrelated.add_node(DiscoveryNode(node_id="r", workspace_snapshot={"repo": "sympy/sympy"}))
    aff_unrelated = compute_seed_affinity(seed, tree_unrelated)

    assert aff_exact > aff_family
    assert aff_family > aff_unrelated
    assert aff_unrelated < 0.40


def test_seeding_policy_affinity_gating():
    """Verify that SeedingPolicy falls back to baseline when affinity is sub-threshold."""
    tree = DiscoveryTree(root_id="root", name="sympy__sympy-13480")
    tree.add_node(DiscoveryNode(node_id="root", score=0.0, workspace_snapshot={"repo": "sympy/sympy"}))
    tree.add_node(DiscoveryNode(node_id="c1", parent_id="root", score=40.0))

    unrelated_seed = SeedCandidate(
        seed_id="s_pallets",
        source_tree="pallets__flask-4992",
        source_repo="pallets/flask",
        motif_type="composition",
        target_operators=["InsertGuard"],
        pattern_signature="pallets/flask:comp:InsertGuard",
        confidence=1.0,
        expected_savings=2.0,
    )

    # At root expansion (T^0 = {root})
    tree_root_only = DiscoveryTree(root_id="root", name="sympy__sympy-13480")
    tree_root_only.add_node(DiscoveryNode(node_id="root", score=0.0, workspace_snapshot={"repo": "sympy/sympy"}))

    policy = SeedingPolicy(seeds=[unrelated_seed], config=SeedingConfig(affinity_threshold=0.80))
    chosen_root = policy.select_batch(tree_root_only, batch_width=1)
    assert chosen_root == ["root"]

    # At subsequent expansion: fallback matches GreedyBestFirst exactly
    greedy = GreedyBestFirstPolicy()
    assert policy.select_batch(tree, batch_width=1) == greedy.select_batch(tree, batch_width=1) == ["c1"]


def test_cross_validated_seeding_optimizer_governor_acceptance():
    """Verify that CrossValidatedSeedingOptimizer achieves Governor ACCEPT with p < 0.05 and 0 regressions."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    assert report_file.is_file(), f"Missing artifact: {report_file}"

    trees = DiscoveryTree.from_swe_bench_report(report_file)
    optimizer = CrossValidatedSeedingOptimizer(trees=trees, k_folds=5, seed=42)

    result = optimizer.optimize(n_samples=500, seed=42)

    # Governor Acceptance Verification on Out-of-Fold Test Worlds
    assert result.governor_verdict["decision"] == "ACCEPT"
    assert result.governor_verdict["reasons"] == ["all_gates_passed"]
    assert result.governor_verdict["regressions"] == 0
    assert result.governor_verdict["worst_c"] >= result.governor_verdict["worst_b"]

    # Statistical Significance Verification (resolving historical p = 0.2967)
    assert result.p_value < 0.05
    assert result.cohen_d > 0.5
    assert result.delta_mean_test_value > 0.0
    assert result.mean_test_evaluations_saved_percent > 10.0

    # 100% test instance coverage with 0 regressions
    assert result.total_instances_evaluated == len(trees)
    assert result.test_regressions == 0


def test_run_cross_validated_seeding_artifact_export(tmp_path: Path):
    """Verify run_cross_validated_seeding writes complete JSON artifact with historical comparison."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    out_file = tmp_path / "dream_seeding_test.json"

    result = run_cross_validated_seeding(
        report_path=report_file,
        output_report_path=out_file,
        k_folds=5,
        n_samples=200,
        seed=42,
    )

    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))

    assert "optimal_config" in data
    assert "governor_verdict" in data
    assert data["governor_verdict"]["decision"] == "ACCEPT"
    assert data["p_value"] < 0.05
    assert "historical_comparison" in data
    assert data["historical_comparison"]["historical_p_value"] == 0.2967
    assert data["historical_comparison"]["dream_rsi_test_holdout_verdict"] == "ACCEPT"
    assert len(data["per_instance_test_records"]) == 10
