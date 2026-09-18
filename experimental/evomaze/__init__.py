"""EvoMaze — Search-Based Procedural Content Generation (SBPCG) Domain Adapter.

Status: FROZEN / EXPLORATORY RESEARCH TRACK (physical_claim: false)
Universal spatial maze and graph optimization track for Darwin-Evolab.
Preserved in a frozen state under experimental/ for academic reproducibility.
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
