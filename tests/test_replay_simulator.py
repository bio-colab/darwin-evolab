"""tests/test_replay_simulator.py — Automated Verification for Dream-RSI Replay Simulator.

Validates:
1. DiscoveryNode serialization and metadata preservation.
2. DiscoveryTree topology, invariants, and serialization.
3. Ingestion of official committed benchmark reports (SWE-bench Lite, duplicate evals probe).
4. Episodic experience reconstruction from ExperienceStore.
5. Exact mathematical implementation of the Dream-RSI Replay Objective (Equation 1).
6. Deterministic, zero-execution-cost replay rollout across simulated worlds.
7. Off-policy policy differentiation and rapid evaluation speed.
"""
from __future__ import annotations

import time
from pathlib import Path
import pytest

from evolab.replay_simulator import (
    DiscoveryNode,
    DiscoveryTree,
    ReplaySimulator,
    ReplayObjective,
    ExplorationPolicy,
    GreedyBestFirstPolicy,
    BreadthFirstPolicy,
    AdaptivePlateauPolicy,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"


def test_discovery_node_serialization():
    """Verify DiscoveryNode serializes to/from dict preserving state and metadata."""
    node = DiscoveryNode(
        node_id="n1",
        parent_id="root",
        workspace_snapshot={"edits": ["InsertGuard"], "target": "app.py"},
        score=95.5,
        cost=0.012,
        passed_holdout=True,
        metadata={"operator": "InsertGuard", "line": 42},
        children_ids=["n2", "n3"],
    )
    d = node.to_dict()
    restored = DiscoveryNode.from_dict(d)

    assert restored.node_id == "n1"
    assert restored.parent_id == "root"
    assert restored.workspace_snapshot["edits"] == ["InsertGuard"]
    assert restored.score == 95.5
    assert restored.cost == 0.012
    assert restored.passed_holdout is True
    assert restored.metadata["line"] == 42
    assert restored.children_ids == ["n2", "n3"]


def test_discovery_tree_topology_and_invariants():
    """Verify tree construction, child pointers, depth, and best-node retrieval."""
    tree = DiscoveryTree(root_id="root", name="test_tree")
    root = DiscoveryNode(node_id="root", score=10.0)
    c1 = DiscoveryNode(node_id="c1", parent_id="root", score=40.0)
    c2 = DiscoveryNode(node_id="c2", parent_id="root", score=30.0)
    c1_1 = DiscoveryNode(node_id="c1_1", parent_id="c1", score=90.0)

    tree.add_node(root)
    tree.add_node(c1)
    tree.add_node(c2)
    tree.add_node(c1_1)

    assert tree.size() == 4
    assert tree.max_depth() == 3
    assert set(tree.nodes["root"].children_ids) == {"c1", "c2"}
    assert tree.nodes["c1"].children_ids == ["c1_1"]
    assert len(tree.leaves()) == 2  # c2 and c1_1
    assert tree.best_node().node_id == "c1_1"
    assert tree.best_node().score == 90.0

    # Serialization roundtrip
    tree_dict = tree.to_dict()
    restored_tree = DiscoveryTree.from_dict(tree_dict)
    assert restored_tree.size() == 4
    assert restored_tree.max_depth() == 3
    assert restored_tree.best_node().score == 90.0


def test_discovery_tree_from_run_report():
    """Verify building DiscoveryTree from a run report with generation history."""
    report_data = {
        "total_generations": 3,
        "history": [
            {"generation": 1, "best_fitness": 40.0, "mean_fitness": 30.0, "edits": 1},
            {"generation": 2, "best_fitness": 70.0, "mean_fitness": 55.0, "edits": 2},
            {"generation": 3, "best_fitness": 100.0, "mean_fitness": 80.0, "edits": 3},
        ],
        "best_individual": {
            "fitness": 100.0,
            "code": "def solve(): return True",
            "edits": ["SwapCondition"],
            "passed_holdout": True,
        },
    }

    tree = DiscoveryTree.from_run_report(report_data, tree_name="synthetic_run")
    assert tree.size() == 4  # root + 3 generations
    assert tree.best_node().score == 100.0
    assert tree.max_depth() == 4


def test_discovery_tree_from_swe_bench_report():
    """Verify extracting DiscoveryTrees from official SWE-bench Lite benchmark artifact."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    assert report_file.is_file(), f"Missing required empirical artifact: {report_file}"

    trees = DiscoveryTree.from_swe_bench_report(report_file)
    assert len(trees) == 10  # 10 instances tested

    # Check first instance sympy__sympy-13480
    sympy_tree = next(t for t in trees if "sympy" in t.name)
    assert sympy_tree.size() == 4  # root + 3 evaluations_used
    assert sympy_tree.best_node().score == 100.0
    assert sympy_tree.best_node().passed_holdout is True

    # Check unresolved instance e.g. click-1608
    click_tree = next(t for t in trees if "click" in t.name)
    assert click_tree.size() == 20  # root + 19 evaluations_used
    assert click_tree.best_node().score < 100.0
    assert click_tree.best_node().passed_holdout is False


def test_discovery_tree_from_experience_store(tmp_path: Path):
    """Verify reconstructing discovery trees from episodic experience records in SQLite."""
    import sqlite3
    from evolab.experience import ExperienceStore

    db_file = tmp_path / "exp_test.db"
    store = ExperienceStore(db_file)

    # Insert mock experiences across 2 independent runs
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO experiences (
            run_id, eval_index, problem_fingerprint, func_name, target_file,
            genome_class, edit_kinds, edit_loci, n_edits, score, fitness_delta,
            is_new_best, passed_holdout, eval_ms, outcome, created_at
        ) VALUES 
        ('run_A', 1, 'fp_01', 'func1', 'app.py', 'code', '["InsertGuard"]', '[10]', 1, 50.0, 50.0, 1, 0, 15.0, 'improvement', '2026-09-18T00:00:00Z'),
        ('run_A', 2, 'fp_01', 'func1', 'app.py', 'code', '["SwapCondition"]', '[15]', 2, 100.0, 50.0, 1, 1, 20.0, 'success', '2026-09-18T00:00:01Z'),
        ('run_B', 1, 'fp_01', 'func1', 'app.py', 'code', '["DeleteStatement"]', '[5]', 1, 20.0, 20.0, 1, 0, 12.0, 'neutral', '2026-09-18T00:00:02Z')
    """)
    conn.commit()
    conn.close()

    trees = DiscoveryTree.from_experience_store(store, problem_fingerprint="fp_01")
    assert len(trees) == 2  # run_A and run_B

    tree_a = next(t for t in trees if "run_A" in t.name)
    assert tree_a.size() == 3  # root + 2 evals
    assert tree_a.best_node().score == 100.0
    assert tree_a.best_node().passed_holdout is True

    tree_b = next(t for t in trees if "run_B" in t.name)
    assert tree_b.size() == 2  # root + 1 eval
    assert tree_b.best_node().score == 20.0


def test_replay_objective_dream_rsi_formula():
    """Verify Dream-RSI Replay Objective equation:
    
    V = max s_v - beta_1 * N + beta_2 * (N / k*)
    """
    obj = ReplayObjective(beta1=0.05, beta2=0.02)

    # Synthetic tree with 1 root + 10 nodes (N = 10 attempts), max_score = 100.0
    tree = DiscoveryTree(root_id="r")
    tree.add_node(DiscoveryNode(node_id="r", score=0.0))
    for i in range(1, 11):
        tree.add_node(DiscoveryNode(node_id=f"n{i}", parent_id="r", score=10.0 * i))

    # Test with k* = 2 rounds (high batching parallelism)
    # Expected: 100.0 - 0.05 * 10 + 0.02 * (10 / 2) = 100.0 - 0.5 + 0.1 = 99.6
    v_parallel = obj.compute(tree, rounds_completed=2)
    assert abs(v_parallel - 99.6) < 1e-6

    # Test with k* = 10 rounds (pure sequential)
    # Expected: 100.0 - 0.05 * 10 + 0.02 * (10 / 10) = 100.0 - 0.5 + 0.02 = 99.52
    v_sequential = obj.compute(tree, rounds_completed=10)
    assert abs(v_sequential - 99.52) < 1e-6

    # Parallel strategy receives higher score under equal solution quality
    assert v_parallel > v_sequential


def test_replay_simulator_deterministic_rollout():
    """Verify that ReplaySimulator exposes children deterministically without live execution."""
    # Build tree with 2 branches:
    # root -> b1 (50) -> b1_1 (100)
    #      -> b2 (40) -> b2_1 (60)
    full_tree = DiscoveryTree(root_id="root", name="oracle_tree")
    full_tree.add_node(DiscoveryNode(node_id="root", score=0.0))
    full_tree.add_node(DiscoveryNode(node_id="b1", parent_id="root", score=50.0))
    full_tree.add_node(DiscoveryNode(node_id="b2", parent_id="root", score=40.0))
    full_tree.add_node(DiscoveryNode(node_id="b1_1", parent_id="b1", score=100.0))
    full_tree.add_node(DiscoveryNode(node_id="b2_1", parent_id="b2", score=60.0))

    sim = ReplaySimulator(objective=ReplayObjective(beta1=0.1, beta2=0.0))
    policy = GreedyBestFirstPolicy()

    # Rollout:
    # Round 1: policy expands root -> reveals b1 (earliest child of root)
    # Round 2: policy expands b1 (score 50 > root 0) -> reveals b1_1 (score 100)
    # At Round 2: b1_1 is discovered (score 100)
    revealed, rounds, score = sim.simulate(full_tree, policy, max_rounds=2, batch_width=1)

    assert rounds == 2
    assert "b1_1" in revealed.nodes
    assert revealed.best_node().score == 100.0
    # Score: 100.0 - 0.1 * 3 attempts (b1, b2, b1_1) = 99.7
    assert abs(score - 99.7) < 1e-6


def test_off_policy_evaluation_speed_and_differentiation():
    """Verify 100+ policy rollouts evaluate in milliseconds and differentiate strategies."""
    report_file = REPORTS_DIR / "swe_bench_lite_subset.json"
    worlds = DiscoveryTree.from_swe_bench_report(report_file)
    assert len(worlds) == 10

    greedy_pol = GreedyBestFirstPolicy()
    breadth_pol = BreadthFirstPolicy()
    adaptive_pol = AdaptivePlateauPolicy(patience=1)

    obj = ReplayObjective(beta1=0.05, beta2=0.05)

    t0 = time.perf_counter()
    # Evaluate 30 simulated world rollouts (3 policies x 10 worlds)
    score_greedy = obj.evaluate_mean_score(greedy_pol, worlds, max_rounds=5, batch_width=2)
    score_breadth = obj.evaluate_mean_score(breadth_pol, worlds, max_rounds=5, batch_width=2)
    score_adaptive = obj.evaluate_mean_score(adaptive_pol, worlds, max_rounds=5, batch_width=2)
    dur = time.perf_counter() - t0

    # Must be exceptionally fast: zero compiler/subprocess/pytest calls!
    assert dur < 0.10, f"Replay simulation took too long ({dur:.4f}s) - must be pure memory lookup"

    # All scores must be valid positive values
    assert score_greedy > 0.0
    assert score_breadth > 0.0
    assert score_adaptive > 0.0

    # Verify that the objective differentiates between exploration policies
    scores = [score_greedy, score_breadth, score_adaptive]
    assert len(set(round(s, 4) for s in scores)) >= 2, f"Policies failed to differentiate: {scores}"


def test_empty_and_edge_case_rollouts():
    """Verify edge cases: empty trees, max_rounds=0, and instant policy termination."""
    empty_tree = DiscoveryTree(root_id="r")
    sim = ReplaySimulator()
    pol = GreedyBestFirstPolicy()

    revealed, rounds, score = sim.simulate(empty_tree, pol, max_rounds=10)
    assert rounds == 0
    assert score == 0.0

    class StopInstantlyPolicy(ExplorationPolicy):
        def select_batch(self, revealed_tree, batch_width):
            return []  # Immediate stop

    tree = DiscoveryTree(root_id="r")
    tree.add_node(DiscoveryNode(node_id="r", score=50.0))
    tree.add_node(DiscoveryNode(node_id="c1", parent_id="r", score=90.0))

    rev, rounds, score = sim.simulate(tree, StopInstantlyPolicy(), max_rounds=10)
    assert rounds == 0
    assert rev.size() == 1
    assert score == 50.0


def test_policy_differentiation_on_branching_trees():
    """Verify that Greedy and Breadth-First exploration policies take distinct trajectories on branching trees."""
    tree = DiscoveryTree(root_id="root", name="branching_world")
    tree.add_node(DiscoveryNode(node_id="root", score=0.0))
    # Branch 1 (shallow, moderate score):
    tree.add_node(DiscoveryNode(node_id="b1", parent_id="root", score=30.0))
    tree.add_node(DiscoveryNode(node_id="b1_1", parent_id="b1", score=35.0))
    # Branch 2 (deep promising, high score):
    tree.add_node(DiscoveryNode(node_id="b2", parent_id="root", score=70.0))
    tree.add_node(DiscoveryNode(node_id="b2_1", parent_id="b2", score=90.0))
    tree.add_node(DiscoveryNode(node_id="b2_2", parent_id="b2_1", score=100.0, passed_holdout=True))

    sim = ReplaySimulator(objective=ReplayObjective(beta1=0.01, beta2=0.01))

    # Run with max_rounds=2, batch_width=1
    # Round 1: both expand root, revealing b1 (30) and b2 (70).
    # Round 2:
    # Greedy expands b2 (score 70 > 30) -> reveals b2_1 (score 90). Best score = 90.
    # BreadthFirst expands b1 (depth 1, earliest in frontier) -> reveals b1_1 (score 35). Best score = 70.
    rev_g, rounds_g, score_g = sim.simulate(tree, GreedyBestFirstPolicy(), max_rounds=2, batch_width=1)
    rev_b, rounds_b, score_b = sim.simulate(tree, BreadthFirstPolicy(), max_rounds=2, batch_width=1)

    assert rev_g.best_node().score == 90.0
    assert rev_b.best_node().score == 70.0
    assert score_g > score_b

