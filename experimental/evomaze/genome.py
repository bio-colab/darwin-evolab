"""genome.py — Topological Spanning-Tree Genome for EvoMaze.

Guarantees 100% solvability and connectivity by construction using
a Minimum Spanning Tree (MST) over candidate grid edges, with
controlled cycle injection (braiding).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from typing import Any

from evolab.genome import EvolabGenome


def canonical_edges(width: int, height: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Generates a deterministic, sorted list of all internal grid edges."""
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    # Horizontal edges: (x, y) <-> (x + 1, y)
    for y in range(height):
        for x in range(width - 1):
            edges.append(((x, y), (x + 1, y)))
    # Vertical edges: (x, y) <-> (x, y + 1)
    for y in range(height - 1):
        for x in range(width):
            edges.append(((x, y), (x, y + 1)))
    return edges


class DisjointSet:
    """Disjoint Set (Union-Find) with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return False
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            self.parent[root_j] = root_i
            self.rank[root_i] += 1
        return True


class MazeGenome(EvolabGenome):
    """Topological graph genome for evolving spatial mazes.

    Represents edge weights on a 2D planar cell graph. The maze passages
    are computed via Kruskal's Minimum Spanning Tree algorithm, which
    mathematically guarantees 100% reachability and solvability.
    """

    def __init__(
        self,
        width: int,
        height: int,
        edge_weights: list[float] | None = None,
        loop_threshold: float = 0.0,
        start: tuple[int, int] = (0, 0),
        exit_pt: tuple[int, int] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.exit = exit_pt if exit_pt is not None else (width - 1, height - 1)
        self.num_edges = (width - 1) * height + width * (height - 1)

        if edge_weights is None:
            self.edge_weights = [random.random() for _ in range(self.num_edges)]
        else:
            if len(edge_weights) != self.num_edges:
                raise ValueError(
                    f"Expected {self.num_edges} edge weights for {width}x{height} maze, got {len(edge_weights)}"
                )
            self.edge_weights = [float(w) for w in edge_weights]

        self.loop_threshold = float(max(0.0, min(1.0, loop_threshold)))

        # Cache of active edges
        self._cached_active_edges: set[tuple[tuple[int, int], tuple[int, int]]] | None = None
        self._cached_fingerprint: str | None = None

    def __len__(self) -> int:
        return len(self.edge_weights)

    def clone(self) -> MazeGenome:
        """Returns an independent deep copy of this genome."""
        return MazeGenome(
            width=self.width,
            height=self.height,
            edge_weights=list(self.edge_weights),
            loop_threshold=self.loop_threshold,
            start=self.start,
            exit_pt=self.exit,
        )

    def _cell_to_id(self, cell: tuple[int, int]) -> int:
        return cell[1] * self.width + cell[0]

    def get_active_edges(self, max_loops: int = 0) -> set[tuple[tuple[int, int], tuple[int, int]]]:
        """Resolves the active maze passages using Kruskal's MST algorithm.

        Guarantees that all cells are mutually reachable (100% solvable).
        """
        if self._cached_active_edges is not None:
            return self._cached_active_edges

        edges = canonical_edges(self.width, self.height)
        # Pair edges with weights and sort ascending
        indexed_edges = list(enumerate(edges))
        indexed_edges.sort(key=lambda item: self.edge_weights[item[0]])

        uf = DisjointSet(self.width * self.height)
        active: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        rejected: list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = []

        for idx, edge in indexed_edges:
            u_id = self._cell_to_id(edge[0])
            v_id = self._cell_to_id(edge[1])
            if uf.union(u_id, v_id):
                # Standardize edge orientation (lower coordinate first)
                u, v = sorted(edge)
                active.add((u, v))
            else:
                rejected.append((idx, edge))

        # Optional cycle injection (braiding)
        if max_loops > 0 and rejected:
            # Number of extra loops proportional to loop_threshold
            loops_to_add = min(max_loops, int(round(self.loop_threshold * max_loops)))
            for _, edge in rejected[:loops_to_add]:
                u, v = sorted(edge)
                active.add((u, v))

        self._cached_active_edges = active
        return active

    def get_adjacency(self, max_loops: int = 0) -> dict[tuple[int, int], list[tuple[int, int]]]:
        """Returns an adjacency list mapping each cell to its connected neighbors."""
        active = self.get_active_edges(max_loops)
        adj: dict[tuple[int, int], list[tuple[int, int]]] = {
            (x, y): [] for y in range(self.height) for x in range(self.width)
        }
        for u, v in active:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def fingerprint(self) -> str:
        """Stable SHA-256 hash identifying phenotypic topology."""
        if self._cached_fingerprint is not None:
            return self._cached_fingerprint

        active = self.get_active_edges()
        sorted_active = sorted(active)
        raw = json.dumps([[[u[0], u[1]], [v[0], v[1]]] for u, v in sorted_active])
        self._cached_fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._cached_fingerprint

    def distance_to(self, other: EvolabGenome) -> float:
        """Normalized Euclidean distance between edge weight vectors."""
        if not isinstance(other, MazeGenome):
            return 10.0
        if self.width != other.width or self.height != other.height:
            return 10.0

        sq_sum = sum((w1 - w2) ** 2 for w1, w2 in zip(self.edge_weights, other.edge_weights))
        sq_sum += (self.loop_threshold - other.loop_threshold) ** 2
        return math.sqrt(sq_sum / (len(self.edge_weights) + 1))

    def serialize(self) -> dict[str, Any]:
        """JSON-safe serialization of the genome."""
        return {
            "type": "MazeGenome",
            "width": self.width,
            "height": self.height,
            "start": list(self.start),
            "exit": list(self.exit),
            "edge_weights": [round(w, 4) for w in self.edge_weights],
            "loop_threshold": round(self.loop_threshold, 4),
            "fingerprint": self.fingerprint(),
        }

    def describe(self) -> dict[str, float | int | str]:
        """Behavioral descriptors for MAP-Elites / quality diversity archiving."""
        adj = self.get_adjacency()
        # Compute shortest path length from start to exit via BFS
        visited = {self.start: 0}
        queue = [self.start]
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            if curr == self.exit:
                break
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited[neighbor] = visited[curr] + 1
                    queue.append(neighbor)

        path_length = visited.get(self.exit, 0)
        dead_ends = sum(1 for cell, neighbors in adj.items() if len(neighbors) == 1)
        junctions = sum(1 for cell, neighbors in adj.items() if len(neighbors) >= 3)
        total_active_edges = len(self.get_active_edges())
        cycle_count = total_active_edges - (self.width * self.height - 1)

        return {
            "path_length": path_length,
            "dead_end_count": dead_ends,
            "junction_count": junctions,
            "cycle_count": max(0, cycle_count),
        }

    def mutate(
        self,
        rng: random.Random | None = None,
        mutation_rate: float = 0.15,
        mutation_scale: float = 0.25,
        **kwargs: Any,
    ) -> MazeGenome:
        """Perturbs a subset of edge weights and loop threshold."""
        r = rng if rng is not None else random.Random()
        new_weights = list(self.edge_weights)

        for i in range(len(new_weights)):
            if r.random() < mutation_rate:
                delta = r.gauss(0.0, mutation_scale)
                new_weights[i] = max(0.0, min(1.0, new_weights[i] + delta))

        new_loop = self.loop_threshold
        if r.random() < mutation_rate:
            new_loop = max(0.0, min(1.0, self.loop_threshold + r.gauss(0.0, mutation_scale)))

        return MazeGenome(
            width=self.width,
            height=self.height,
            edge_weights=new_weights,
            loop_threshold=new_loop,
            start=self.start,
            exit_pt=self.exit,
        )

    def crossover(self, other: EvolabGenome, rng: random.Random | None = None) -> MazeGenome:
        """Uniform crossover between two maze genomes."""
        if not isinstance(other, MazeGenome):
            return self.clone()

        r = rng if rng is not None else random.Random()
        child_weights: list[float] = []
        for w1, w2 in zip(self.edge_weights, other.edge_weights):
            child_weights.append(w1 if r.random() < 0.5 else w2)

        child_loop = self.loop_threshold if r.random() < 0.5 else other.loop_threshold

        return MazeGenome(
            width=self.width,
            height=self.height,
            edge_weights=child_weights,
            loop_threshold=child_loop,
            start=self.start,
            exit_pt=self.exit,
        )
