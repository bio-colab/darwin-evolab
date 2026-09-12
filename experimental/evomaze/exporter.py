"""exporter.py — Visualization and Tilemap Exporters for EvoMaze.

Exports evolved mazes into ANSI/Unicode ASCII text representations
and standardized JSON Tilemap schemas compatible with Godot 4 and Unity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .genome import MazeGenome
from .spec import MazeSpec


def build_tilemap_grid(
    genome: MazeGenome,
    spec: MazeSpec,
    solution_path: list[tuple[int, int]] | None = None,
) -> list[list[int]]:
    """Constructs an expanded (2W+1) x (2H+1) discrete tilemap matrix.

    Tile values:
      0: Passage
      1: Wall
      2: Start
      3: Exit
      4: Solution Path
    """
    grid_w = genome.width * 2 + 1
    grid_h = genome.height * 2 + 1
    # Initialize all cells as solid walls (1)
    grid = [[1 for _ in range(grid_w)] for _ in range(grid_h)]

    # Open logical cells as passages (0)
    for y in range(genome.height):
        for x in range(genome.width):
            grid[y * 2 + 1][x * 2 + 1] = 0

    # Carve active edges
    active_edges = genome.get_active_edges(max_loops=spec.target_loops)
    for u, v in active_edges:
        ux, uy = u
        vx, vy = v
        # Passage coordinate between u and v
        wall_x = ux + vx + 1
        wall_y = uy + vy + 1
        grid[wall_y][wall_x] = 0

    # Highlight solution path
    if solution_path:
        for i in range(len(solution_path)):
            cx, cy = solution_path[i]
            grid[cy * 2 + 1][cx * 2 + 1] = 4
            if i + 1 < len(solution_path):
                nx, ny = solution_path[i + 1]
                grid[cy + ny + 1][cx + nx + 1] = 4

    # Mark start and exit
    sx, sy = spec.start
    ex, ey = spec.exit
    grid[sy * 2 + 1][sx * 2 + 1] = 2
    grid[ey * 2 + 1][ex * 2 + 1] = 3

    return grid


class MazeExporter:
    """Renders evolved mazes to text and machine-readable data."""

    def __init__(self, genome: MazeGenome, spec: MazeSpec) -> None:
        self.genome = genome
        self.spec = spec

    def to_ascii(
        self,
        show_solution: bool = True,
        wall_char: str = "##",
        path_char: str = "  ",
        start_char: str = "S ",
        exit_char: str = "E ",
        sol_char: str = "..",
    ) -> str:
        """Renders the maze as an ASCII/Unicode string for terminal inspection."""
        from .evaluator import MazeEvaluator

        ev = MazeEvaluator(self.spec)
        metrics = ev._extract_graph_metrics(self.genome)
        sol_path = metrics["path"] if show_solution else None

        grid = build_tilemap_grid(self.genome, self.spec, sol_path)
        char_map = {
            0: path_char,
            1: wall_char,
            2: start_char,
            3: exit_char,
            4: sol_char,
        }

        lines: list[str] = []
        for row in grid:
            line = "".join(char_map.get(cell, "??") for cell in row)
            lines.append(line)

        return "\n".join(lines)

    def to_dict(self, include_ascii: bool = True) -> dict[str, Any]:
        """Exports the maze into a structured dictionary suitable for JSON serialization."""
        from .evaluator import MazeEvaluator

        ev = MazeEvaluator(self.spec)
        metrics = ev._extract_graph_metrics(self.genome)

        grid = build_tilemap_grid(self.genome, self.spec, metrics["path"])
        grid_w = len(grid[0])
        grid_h = len(grid)

        data: dict[str, Any] = {
            "format": "darwin-evomaze/v1",
            "logical_dimensions": {"width": self.genome.width, "height": self.genome.height},
            "tilemap_dimensions": {"width": grid_w, "height": grid_h},
            "start": list(self.spec.start),
            "exit": list(self.spec.exit),
            "metrics": {
                "path_length": metrics["path_length"],
                "branching_ratio": round(metrics["branching_ratio"], 3),
                "junction_count": metrics["junction_count"],
                "path_junction_count": metrics["path_junction_count"],
                "dead_end_count": metrics["dead_end_count"],
                "avg_dead_end_depth": round(metrics["avg_dead_end_depth"], 2),
                "cycle_count": metrics["cycle_count"],
                "solvable": metrics["solvable"],
            },
            "tilemap_grid": grid,
            "solution_path": [list(pt) for pt in metrics["path"]],
        }

        if include_ascii:
            data["ascii_preview"] = self.to_ascii(show_solution=True)

        return data

    def save_json(self, file_path: str | Path, indent: int = 2) -> Path:
        """Saves the tilemap dictionary to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        return path
