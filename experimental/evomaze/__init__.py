"""EvoMaze — Search-Based Procedural Content Generation (SBPCG) Domain Adapter.

Universal spatial maze and graph optimization track for Darwin-Evolab.
"""
from __future__ import annotations

from .adapter import EvoMazeAdapter
from .evaluator import MazeEvaluator
from .exporter import MazeExporter, build_tilemap_grid
from .genome import DisjointSet, MazeGenome, canonical_edges
from .spec import MazeSpec

__all__ = [
    "EvoMazeAdapter",
    "MazeSpec",
    "MazeGenome",
    "MazeEvaluator",
    "MazeExporter",
    "DisjointSet",
    "canonical_edges",
    "build_tilemap_grid",
]
