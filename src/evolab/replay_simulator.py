"""replay_simulator.py — Discovery Trees & Zero-Execution-Cost Replay Simulator.

Implements the foundational concepts from Dream-RSI (Zheng et al., arXiv:2609.14858):
1. Accumulated evolutionary search history serves as an executable Replay Simulator.
2. Discovery trajectories are structured as Discovery Trees T = (V, E, r).
3. Offline exploration policies are simulated and scored over pre-stored nodes with zero
   compiler, subprocess, test-case, or SPICE simulation overhead.
4. The Replay Objective balances solution quality, attempt cost, and batch parallelism:
   V = max s_v - beta_1 * N + beta_2 * (N / k*)
"""
from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class DiscoveryNode:
    """A single node in a historical discovery tree representing an evaluated state."""

    node_id: str
    parent_id: str | None = None
    workspace_snapshot: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    cost: float = 0.0  # Execution time (seconds) or evaluation count delta
    passed_holdout: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    children_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "workspace_snapshot": self.workspace_snapshot,
            "score": self.score,
            "cost": self.cost,
            "passed_holdout": self.passed_holdout,
            "metadata": self.metadata,
            "children_ids": list(self.children_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveryNode:
        return cls(
            node_id=str(data["node_id"]),
            parent_id=data.get("parent_id"),
            workspace_snapshot=dict(data.get("workspace_snapshot", {})),
            score=float(data.get("score", 0.0)),
            cost=float(data.get("cost", 0.0)),
            passed_holdout=bool(data.get("passed_holdout", False)),
            metadata=dict(data.get("metadata", {})),
            children_ids=list(data.get("children_ids", [])),
        )


class DiscoveryTree:
    """Rooted directed acyclic discovery tree T = (V, E, r)."""

    def __init__(self, root_id: str, name: str = "discovery_tree") -> None:
        self.root_id = root_id
        self.name = name
        self.nodes: dict[str, DiscoveryNode] = {}

    def add_node(self, node: DiscoveryNode) -> None:
        """Inserts a node into the tree, maintaining child pointers."""
        self.nodes[node.node_id] = node
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.node_id not in parent.children_ids:
                parent.children_ids.append(node.node_id)

    def get_node(self, node_id: str) -> DiscoveryNode | None:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> list[DiscoveryNode]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def leaves(self) -> list[DiscoveryNode]:
        """Returns all nodes that have no recorded children."""
        return [node for node in self.nodes.values() if not node.children_ids]

    def best_node(self) -> DiscoveryNode | None:
        """Returns the highest-scoring node in the tree."""
        if not self.nodes:
            return None
        return max(self.nodes.values(), key=lambda n: n.score)

    def size(self) -> int:
        return len(self.nodes)

    def max_depth(self) -> int:
        """Calculates maximum depth from root."""
        if not self.nodes or self.root_id not in self.nodes:
            return 0

        def _depth(nid: str) -> int:
            node = self.nodes.get(nid)
            if not node or not node.children_ids:
                return 1
            return 1 + max(_depth(cid) for cid in node.children_ids if cid in self.nodes)

        return _depth(self.root_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "name": self.name,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveryTree:
        tree = cls(root_id=str(data["root_id"]), name=data.get("name", "discovery_tree"))
        for nid, ndata in data.get("nodes", {}).items():
            tree.add_node(DiscoveryNode.from_dict(ndata))
        return tree

    @classmethod
    def from_run_report(
        cls,
        report_data: dict[str, Any] | str | Path,
        tree_name: str | None = None,
    ) -> DiscoveryTree:
        """Constructs a DiscoveryTree from a standard Darwin-Evolab RunReport JSON artifact."""
        if isinstance(report_data, (str, Path)):
            path = Path(report_data)
            data = json.loads(path.read_text(encoding="utf-8"))
            name = tree_name or path.stem
        else:
            data = report_data
            name = tree_name or "run_report"

        root_id = "node_000_root"
        tree = cls(root_id=root_id, name=name)

        # Create root node representing baseline/unmodified state
        history = data.get("history", [])
        init_fitness = 0.0
        if history and isinstance(history, list):
            init_fitness = float(history[0].get("best_fitness", 0.0))

        root_node = DiscoveryNode(
            node_id=root_id,
            parent_id=None,
            workspace_snapshot={"source": data.get("best_individual", {}).get("code", "")},
            score=init_fitness,
            cost=0.0,
            passed_holdout=False,
            metadata={"generation": 0, "stage": "initial"},
        )
        tree.add_node(root_node)

        # Build trajectory nodes from generation history or best individual
        prev_id = root_id
        for idx, item in enumerate(history):
            gen = item.get("generation", idx + 1)
            node_id = f"node_gen_{gen:03d}"
            fit = float(item.get("best_fitness", 0.0))
            edits = int(item.get("edits", 0))
            node = DiscoveryNode(
                node_id=node_id,
                parent_id=prev_id,
                workspace_snapshot={"generation": gen, "edits": edits},
                score=fit,
                cost=float(item.get("eval_cost", 1.0)),
                passed_holdout=fit >= 99.0,
                metadata={"generation": gen, "mean_fitness": item.get("mean_fitness")},
            )
            tree.add_node(node)
            prev_id = node_id

        # If a best individual exists with higher score, append/link it as candidate discovery
        best_ind = data.get("best_individual")
        if best_ind and isinstance(best_ind, dict):
            best_fit = float(best_ind.get("fitness", 0.0))
            if best_fit > (tree.best_node().score if tree.best_node() else 0.0):
                best_id = "node_best_solution"
                b_node = DiscoveryNode(
                    node_id=best_id,
                    parent_id=prev_id,
                    workspace_snapshot={"code": best_ind.get("code", ""), "edits": best_ind.get("edits", [])},
                    score=best_fit,
                    cost=1.0,
                    passed_holdout=bool(best_ind.get("passed_holdout", False)),
                    metadata={"species": best_ind.get("species", "default")},
                )
                tree.add_node(b_node)

        return tree

    @classmethod
    def from_swe_bench_report(
        cls,
        report_path: str | Path,
    ) -> list[DiscoveryTree]:
        """Extracts individual DiscoveryTrees for each instance in a SWE-bench evaluation report."""
        path = Path(report_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        trees: list[DiscoveryTree] = []
        instances = data.get("instances", [])
        for inst in instances:
            iid = inst.get("instance_id", "unknown_instance")
            root_id = f"root_{iid}"
            tree = cls(root_id=root_id, name=iid)

            root = DiscoveryNode(
                node_id=root_id,
                parent_id=None,
                workspace_snapshot={"repo": inst.get("repo", ""), "target_file": inst.get("target_file", "")},
                score=0.0,
                cost=0.0,
                passed_holdout=False,
                metadata={"status": "initial"},
            )
            tree.add_node(root)

            evals = int(inst.get("evaluations_used", inst.get("evaluations", 1)))
            resolved = bool(inst.get("resolved", False))
            time_sec = float(inst.get("execution_time_seconds", inst.get("time_seconds", 0.0)))

            # Represent search attempts leading to final outcome
            prev_id = root_id
            for step in range(1, evals + 1):
                step_id = f"{iid}_step_{step:02d}"
                is_final = step == evals
                score = 100.0 if (is_final and resolved) else min(50.0, step * 10.0)
                patch = inst.get("patch_preview", inst.get("patch_applied", "")) if is_final else ""
                node = DiscoveryNode(
                    node_id=step_id,
                    parent_id=prev_id,
                    workspace_snapshot={"step": step, "patch": patch},
                    score=score,
                    cost=time_sec / max(1, evals),
                    passed_holdout=is_final and resolved,
                    metadata={"step": step, "resolved": is_final and resolved},
                )
                tree.add_node(node)
                prev_id = step_id

            trees.append(tree)

        return trees

    @classmethod
    def from_experience_store(
        cls,
        store: Any,
        problem_fingerprint: str | None = None,
    ) -> list[DiscoveryTree]:
        """Constructs discovery trees from episodic experience rows in ExperienceStore."""
        import sqlite3

        db_path = getattr(store, "db_path", None) or getattr(store, "path", None)
        if not db_path or not Path(db_path).exists():
            return []

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        if problem_fingerprint:
            cur.execute(
                "SELECT * FROM experiences WHERE problem_fingerprint = ? ORDER BY run_id, eval_index",
                (problem_fingerprint,),
            )
        else:
            cur.execute("SELECT * FROM experiences ORDER BY run_id, eval_index")

        rows = cur.fetchall()
        conn.close()

        # Group rows by run_id
        runs: dict[str, list[Any]] = {}
        for r in rows:
            runs.setdefault(r["run_id"], []).append(r)

        trees: list[DiscoveryTree] = []
        for run_id, exp_rows in runs.items():
            root_id = f"root_{run_id}"
            tree = cls(root_id=root_id, name=f"run_{run_id}")

            first = exp_rows[0]
            root = DiscoveryNode(
                node_id=root_id,
                parent_id=None,
                workspace_snapshot={"target_file": first["target_file"], "func": first["func_name"]},
                score=0.0,
                cost=0.0,
                passed_holdout=False,
                metadata={"fingerprint": first["problem_fingerprint"]},
            )
            tree.add_node(root)

            prev_id = root_id
            for r in exp_rows:
                idx = r["eval_index"]
                nid = f"{run_id}_eval_{idx:03d}"
                node = DiscoveryNode(
                    node_id=nid,
                    parent_id=prev_id,
                    workspace_snapshot={"kinds": r["edit_kinds"], "loci": r["edit_loci"]},
                    score=float(r["score"]),
                    cost=float(r["eval_ms"]) / 1000.0,
                    passed_holdout=bool(r["passed_holdout"]),
                    metadata={"outcome": r["outcome"], "is_new_best": bool(r["is_new_best"])},
                )
                tree.add_node(node)
                prev_id = nid

            trees.append(tree)

        return trees


@dataclass
class ReplayObjective:
    """Calculates the Dream-RSI objective function (Equation 1):
    
    V_i^m = max_{v in T^{m, k*}} s_v - beta_1 * N_i^m + beta_2 * (N_i^m / k*)
    
    Parameters:
        beta1: Penalty coefficient on evaluation attempt count (cost efficiency).
        beta2: Bonus coefficient on parallelism / attempts per decision round.
    """

    beta1: float = 0.05
    beta2: float = 0.02

    def compute_raw(
        self,
        max_score: float,
        n_attempts: float,
        rounds_completed: int,
    ) -> float:
        """Computes replay objective score directly from raw metrics."""
        cost_penalty = self.beta1 * n_attempts
        k_star = max(1, rounds_completed)
        parallel_bonus = self.beta2 * (n_attempts / k_star) if rounds_completed > 0 else 0.0
        return float(max_score - cost_penalty + parallel_bonus)

    def compute(
        self,
        revealed_tree: DiscoveryTree,
        rounds_completed: int,
    ) -> float:
        """Computes replay score V for a completed policy trajectory."""
        if not revealed_tree.nodes:
            return 0.0

        # Best solution quality attained in revealed subtree
        best = revealed_tree.best_node()
        max_score = best.score if best else 0.0
        n_attempts = max(0, revealed_tree.size() - 1)
        return self.compute_raw(max_score, float(n_attempts), rounds_completed)

    def evaluate_mean_score(
        self,
        policy: ExplorationPolicy,
        worlds: Sequence[DiscoveryTree],
        max_rounds: int = 20,
        batch_width: int = 4,
    ) -> float:
        """Evaluates policy across multiple replay worlds and returns mean score V^m."""
        if not worlds:
            return 0.0

        sim = ReplaySimulator(objective=self)
        scores = [
            sim.simulate(tree, policy, max_rounds=max_rounds, batch_width=batch_width)[2]
            for tree in worlds
        ]
        return float(sum(scores) / len(scores))


class ExplorationPolicy(ABC):
    """Abstract interface for programmable search policies under Replay Simulation."""

    @abstractmethod
    def select_batch(
        self,
        revealed_tree: DiscoveryTree,
        batch_width: int,
    ) -> list[str]:
        """Selects up to `batch_width` node IDs from the revealed tree to continue/expand.
        
        Returning an empty list signals stopping the search rollout.
        """
        pass


class GreedyBestFirstPolicy(ExplorationPolicy):
    """Exploration policy that prioritizes expanding frontier leaf nodes with highest score."""

    def select_batch(
        self,
        revealed_tree: DiscoveryTree,
        batch_width: int,
    ) -> list[str]:
        leaves = revealed_tree.leaves()
        if not leaves:
            return [revealed_tree.root_id]

        # Sort leaves descending by score
        sorted_leaves = sorted(leaves, key=lambda n: n.score, reverse=True)
        return [node.node_id for node in sorted_leaves[:batch_width]]


class BreadthFirstPolicy(ExplorationPolicy):
    """Exploration policy that explores unexpanded nodes in shallowest-first order."""

    def select_batch(
        self,
        revealed_tree: DiscoveryTree,
        batch_width: int,
    ) -> list[str]:
        leaves = revealed_tree.leaves()
        if not leaves:
            return [revealed_tree.root_id]
        return [node.node_id for node in leaves[:batch_width]]


class AdaptivePlateauPolicy(ExplorationPolicy):
    """Exploration policy that detects fitness plateaus and pivots to alternative branches."""

    def __init__(self, patience: int = 2) -> None:
        self.patience = patience
        self._stagnation_count = 0
        self._last_best_score = -1.0

    def select_batch(
        self,
        revealed_tree: DiscoveryTree,
        batch_width: int,
    ) -> list[str]:
        best = revealed_tree.best_node()
        current_best = best.score if best else 0.0

        if current_best > self._last_best_score:
            self._last_best_score = current_best
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1

        leaves = revealed_tree.leaves()
        if not leaves:
            return [revealed_tree.root_id]

        if self._stagnation_count >= self.patience:
            # Stagnation detected: force pivot to root or least-explored shallow leaves
            self._stagnation_count = 0
            return [revealed_tree.root_id]

        # Otherwise continue greedy exploitation
        sorted_leaves = sorted(leaves, key=lambda n: n.score, reverse=True)
        return [node.node_id for node in sorted_leaves[:batch_width]]


class ReplaySimulator:
    """In-memory zero-execution-cost replay simulator over historical DiscoveryTrees."""

    def __init__(self, objective: ReplayObjective | None = None) -> None:
        self.objective = objective or ReplayObjective()

    def simulate(
        self,
        full_tree: DiscoveryTree,
        policy: ExplorationPolicy,
        max_rounds: int = 20,
        batch_width: int = 4,
    ) -> tuple[DiscoveryTree, int, float]:
        """Simulates an exploration policy rollout over the historical discovery tree.
        
        Returns:
            (revealed_tree, rounds_completed, replay_objective_score)
        """
        if not full_tree.nodes or full_tree.root_id not in full_tree.nodes:
            empty_tree = DiscoveryTree(root_id="empty")
            return empty_tree, 0, 0.0

        # Step 0: Initialize revealed tree with root node only: T^{m, 0} = {r}
        revealed_tree = DiscoveryTree(root_id=full_tree.root_id, name=f"revealed_{full_tree.name}")
        root_node_copy = copy.deepcopy(full_tree.nodes[full_tree.root_id])
        root_node_copy.children_ids = []  # Children are unrevealed at step 0
        revealed_tree.add_node(root_node_copy)

        completed_rounds = 0

        for round_k in range(max_rounds):
            # Check if all recorded nodes have been revealed
            if revealed_tree.size() >= full_tree.size():
                break

            # Policy chooses batch of nodes to expand based on revealed tree
            chosen_batch = policy.select_batch(revealed_tree, batch_width)
            if not chosen_batch:
                # Policy chooses to terminate rollout early
                break

            newly_revealed = 0

            for node_id in chosen_batch:
                full_node = full_tree.get_node(node_id)
                if not full_node:
                    continue

                # Find recorded children of full_node that are still unrevealed
                unrevealed_children = [
                    cid for cid in full_node.children_ids
                    if cid in full_tree.nodes and cid not in revealed_tree.nodes
                ]

                # Expand node: reveal all recorded unrevealed children from the full tree
                for target_cid in unrevealed_children:
                    child_copy = copy.deepcopy(full_tree.nodes[target_cid])
                    child_copy.children_ids = []
                    revealed_tree.add_node(child_copy)
                    newly_revealed += 1

            if newly_revealed == 0:
                # No new nodes could be revealed from the chosen batch
                break

            completed_rounds += 1

        # Compute replay objective score
        v_score = self.objective.compute(revealed_tree, completed_rounds)
        return revealed_tree, completed_rounds, v_score
