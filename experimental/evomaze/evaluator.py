"""evaluator.py — Graph Topology Evaluator for EvoMaze.

Computes pathfinding (A*/BFS), branching entropy, dead-end distribution,
and cycle counts to score spatial mazes against objective specifications.
"""
from __future__ import annotations

from collections import deque
import math
from typing import Any

from evolab.evaluators import Evaluator, FitnessResult
from .genome import MazeGenome
from .spec import MazeSpec


class MazeEvaluator(Evaluator):
    """Evaluates spatial mazes based on graph topological metrics."""

    def __init__(self, spec: MazeSpec) -> None:
        self.spec = spec

    @property
    def deterministic(self) -> bool:
        return True

    def _extract_graph_metrics(
        self, genome: MazeGenome
    ) -> dict[str, Any]:
        """Runs graph traversals to extract all structural metrics."""
        adj = genome.get_adjacency(max_loops=self.spec.target_loops)
        start = self.spec.start
        exit_pt = self.spec.exit

        # 1. BFS Shortest Path
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        dist: dict[tuple[int, int], int] = {start: 0}
        queue = deque([start])

        while queue:
            curr = queue.popleft()
            if curr == exit_pt:
                break
            for nbr in adj.get(curr, []):
                if nbr not in dist:
                    dist[nbr] = dist[curr] + 1
                    parent[nbr] = curr
                    queue.append(nbr)

        # Reconstruct path
        path: list[tuple[int, int]] = []
        if exit_pt in dist:
            curr_node: tuple[int, int] | None = exit_pt
            while curr_node is not None:
                path.append(curr_node)
                curr_node = parent[curr_node]
            path.reverse()

        path_length = len(path) - 1 if path else 0
        path_set = set(path)

        # 2. Degree Analysis & Decision Junctions
        dead_ends: list[tuple[int, int]] = []
        junctions: list[tuple[int, int]] = []
        path_junctions: list[tuple[int, int]] = []

        for cell, neighbors in adj.items():
            deg = len(neighbors)
            if deg == 1 and cell != start and cell != exit_pt:
                dead_ends.append(cell)
            elif deg >= 3:
                junctions.append(cell)
                if cell in path_set:
                    path_junctions.append(cell)

        # 3. Branching Ratio
        branching_ratio = len(path_junctions) / max(1, path_length)

        # 4. Cycles / Loops
        active_edges = genome.get_active_edges(max_loops=self.spec.target_loops)
        num_vertices = genome.width * genome.height
        cycle_count = max(0, len(active_edges) - (num_vertices - 1))

        # 5. Dead-End Depth (Average distance to nearest junction)
        total_depth = 0
        for de in dead_ends:
            curr = de
            p_node = None
            depth = 0
            while True:
                nbrs = [n for n in adj[curr] if n != p_node]
                if not nbrs or len(adj[curr]) >= 3 or curr == start or curr == exit_pt:
                    break
                p_node = curr
                curr = nbrs[0]
                depth += 1
            total_depth += depth

        avg_dead_end_depth = total_depth / max(1, len(dead_ends))

        return {
            "path": path,
            "path_length": path_length,
            "branching_ratio": branching_ratio,
            "junction_count": len(junctions),
            "path_junction_count": len(path_junctions),
            "dead_end_count": len(dead_ends),
            "avg_dead_end_depth": avg_dead_end_depth,
            "cycle_count": cycle_count,
            "solvable": path_length > 0,
        }

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        genome = getattr(target, "genome", target)
        if not isinstance(genome, MazeGenome):
            return FitnessResult(score=0.0)

        metrics = self._extract_graph_metrics(genome)
        if not metrics["solvable"]:
            return FitnessResult(score=0.0, artifacts=metrics)

        subscores: list[float] = []
        weights: list[float] = []

        # Target Path Length
        if self.spec.target_path_length is not None:
            err = abs(metrics["path_length"] - self.spec.target_path_length)
            norm_err = err / max(1.0, self.spec.target_path_length)
            subscore = max(0.0, 100.0 * (1.0 - norm_err))
            subscores.append(subscore)
            weights.append(3.0)
        else:
            # Maximize path length (tortuosity)
            max_possible = genome.width * genome.height - 1
            subscore = min(100.0, (metrics["path_length"] / max(1.0, max_possible)) * 100.0)
            subscores.append(subscore)
            weights.append(2.0)

        # Target Branching Ratio
        if self.spec.target_branching_ratio is not None:
            err = abs(metrics["branching_ratio"] - self.spec.target_branching_ratio)
            subscore = max(0.0, 100.0 - err * 250.0)
            subscores.append(subscore)
            weights.append(2.0)

        # Target Dead Ends
        if self.spec.target_dead_ends is not None:
            err = abs(metrics["dead_end_count"] - self.spec.target_dead_ends)
            norm_err = err / max(1.0, self.spec.target_dead_ends)
            subscore = max(0.0, 100.0 * (1.0 - norm_err))
            subscores.append(subscore)
            weights.append(2.0)

        # Target Loops / Cycles
        if self.spec.target_loops is not None:
            err = abs(metrics["cycle_count"] - self.spec.target_loops)
            subscore = max(0.0, 100.0 - err * 30.0)
            subscores.append(subscore)
            weights.append(1.5)

        total_weight = sum(weights)
        if total_weight > 0:
            final_score = sum(s * w for s, w in zip(subscores, weights)) / total_weight
        else:
            final_score = 100.0

        return FitnessResult(
            score=round(max(0.0, min(100.0, final_score)), 3),
            passed_holdout=metrics["solvable"],
            artifacts={
                "path_length": metrics["path_length"],
                "junction_count": metrics["junction_count"],
                "path_junction_count": metrics["path_junction_count"],
                "dead_end_count": metrics["dead_end_count"],
                "avg_dead_end_depth": round(metrics["avg_dead_end_depth"], 2),
                "cycle_count": metrics["cycle_count"],
            },
        )
