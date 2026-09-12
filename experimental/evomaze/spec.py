"""spec.py — Specification data structures for EvoMaze.

Defines the configuration parameters and optimization targets for
procedurally generated spatial mazes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MazeSpec:
    """Specification configuration for generating an evolved spatial maze."""

    width: int = 15
    height: int = 15
    start: tuple[int, int] = (0, 0)
    exit: tuple[int, int] = field(default_factory=lambda: (14, 14))
    target_path_length: float | None = None
    target_branching_ratio: float | None = None
    target_dead_ends: int | None = None
    target_loops: int = 0
    population_size: int = 20
    generations: int = 25
    seed: int = 42

    def __post_init__(self) -> None:
        if self.width < 3 or self.height < 3:
            raise ValueError(f"Maze dimensions must be at least 3x3, got {self.width}x{self.height}")
        sx, sy = self.start
        ex, ey = self.exit
        if not (0 <= sx < self.width and 0 <= sy < self.height):
            raise ValueError(f"Start coordinate {self.start} is out of bounds for {self.width}x{self.height}")
        if not (0 <= ex < self.width and 0 <= ey < self.height):
            raise ValueError(f"Exit coordinate {self.exit} is out of bounds for {self.width}x{self.height}")
        if self.start == self.exit:
            raise ValueError("Start and exit coordinates cannot be identical")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MazeSpec:
        """Constructs a MazeSpec from a dictionary with sensible defaults."""
        width = int(data.get("width", 15))
        height = int(data.get("height", 15))
        start_raw = data.get("start", (0, 0))
        exit_raw = data.get("exit", (width - 1, height - 1))

        start = (int(start_raw[0]), int(start_raw[1]))
        exit_pt = (int(exit_raw[0]), int(exit_raw[1]))

        return cls(
            width=width,
            height=height,
            start=start,
            exit=exit_pt,
            target_path_length=float(data["target_path_length"]) if "target_path_length" in data and data["target_path_length"] is not None else None,
            target_branching_ratio=float(data["target_branching_ratio"]) if "target_branching_ratio" in data and data["target_branching_ratio"] is not None else None,
            target_dead_ends=int(data["target_dead_ends"]) if "target_dead_ends" in data and data["target_dead_ends"] is not None else None,
            target_loops=int(data.get("target_loops", 0)),
            population_size=int(data.get("population_size", 20)),
            generations=int(data.get("generations", 25)),
            seed=int(data.get("seed", 42)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "start": list(self.start),
            "exit": list(self.exit),
            "target_path_length": self.target_path_length,
            "target_branching_ratio": self.target_branching_ratio,
            "target_dead_ends": self.target_dead_ends,
            "target_loops": self.target_loops,
            "population_size": self.population_size,
            "generations": self.generations,
            "seed": self.seed,
        }
