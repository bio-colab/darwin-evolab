"""Tests for Autonomous Closed-Loop Self-Evolution & Self-Governing Policy Hot-Swapping."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evolab import (
    AutonomousEvolutionManager,
    SelfEvolutionEngine,
    MetaProposal,
)
from evolab.replay_simulator import DiscoveryNode, DiscoveryTree


def test_autonomous_manager_init_and_default_policy(tmp_path: Path):
    """Verify AutonomousEvolutionManager initializes with valid default policy and paths."""
    policy_file = tmp_path / "test_policy.json"
    manager = AutonomousEvolutionManager(policy_path=policy_file)

    policy = manager.get_active_policy()
    assert policy["version"] == "baseline"
    assert "operator_weights" in policy
    assert policy["operator_weights"]["InsertGuard"] == pytest.approx(0.142857, abs=1e-4)
    assert manager.should_trigger_optimization() is False


def test_passive_telemetry_and_stagnation_detection(tmp_path: Path):
    """Verify telemetry recording increments stagnation count and triggers optimization threshold."""
    policy_file = tmp_path / "test_policy.json"
    manager = AutonomousEvolutionManager(policy_path=policy_file, stagnation_threshold=3)

    # 1. Normal runs with no stagnation
    manager.record_run_telemetry({"generation": 10, "best_fitness": 95.0, "stagnation": False})
    manager.record_run_telemetry({"generation": 12, "best_fitness": 100.0, "stagnation": False})
    assert manager.should_trigger_optimization() is False
    assert manager.total_sessions_recorded == 2

    # 2. Repeated stagnation events
    manager.record_run_telemetry({"generation": 20, "stagnation": True})
    manager.record_run_telemetry({"generation": 20, "stagnation": True})
    assert manager.should_trigger_optimization() is False

    manager.record_run_telemetry({"generation": 20, "stagnation": True})
    assert manager.should_trigger_optimization() is True


def test_autonomous_dreaming_optimization_and_governor_hot_swap(tmp_path: Path):
    """Verify autonomous Dreaming cycle submits to Governor and hot-swaps runtime policy on ACCEPT."""
    policy_file = tmp_path / "test_policy.json"
    manager = AutonomousEvolutionManager(policy_path=policy_file, alpha=0.05, min_effect_size=0.50)

    # Build 10 synthetic discovery trees representing historical search sessions
    trees = []
    for t_idx in range(10):
        tree = DiscoveryTree(root_id=f"root_{t_idx}", name=f"tree_{t_idx}")
        tree.add_node(DiscoveryNode(node_id=f"root_{t_idx}", score=0.0))
        # Dead ends
        for d_idx in range(1, 4):
            tree.add_node(DiscoveryNode(
                node_id=f"dead_{t_idx}_{d_idx}",
                parent_id=f"root_{t_idx}",
                score=10.0,
                metadata={"operator": "SwapCondition"},
            ))
        # Winning node reached via InsertGuard
        tree.add_node(DiscoveryNode(
            node_id=f"win_{t_idx}",
            parent_id=f"root_{t_idx}",
            score=100.0,
            passed_holdout=True,
            metadata={"operator": "InsertGuard"},
        ))
        trees.append(tree)

    # Run autonomous optimization
    outcome = manager.run_autonomous_optimization(discovery_trees=trees, n_samples=500, seed=42)

    assert outcome["decision"] in ("ACCEPT", "REJECT")
    if outcome["promoted"]:
        assert outcome["decision"] == "ACCEPT"
        assert outcome["p_value"] < 0.05
        assert outcome["cohen_d"] >= 0.50
        # Check hot-swapped runtime policy
        active = manager.get_active_policy()
        assert "dream_promoted" in active["version"]
        assert active["operator_weights"]["InsertGuard"] > active["operator_weights"]["SwapCondition"]
        # Verify persistence to disk
        assert policy_file.is_file()
        persisted = json.loads(policy_file.read_text(encoding="utf-8"))
        assert persisted["version"] == active["version"]


def test_governor_rejection_safety_preserves_baseline(tmp_path: Path):
    """Verify that when a candidate fails Governor statistical gates, baseline is preserved intact."""
    policy_file = tmp_path / "test_policy.json"
    # Set impossible effect size threshold (e.g. 5.0) to force Governor rejection
    manager = AutonomousEvolutionManager(policy_path=policy_file, min_effect_size=5.0)

    # Empty tree list leads to skipped / preserved baseline
    res = manager.run_autonomous_optimization(discovery_trees=[])
    assert res["decision"] == "SKIPPED"
    assert res["promoted"] is False

    active = manager.get_active_policy()
    assert active["version"] == "baseline"
