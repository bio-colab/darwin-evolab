"""map_elites_archive.py — Quality-Diversity (QD) MAP-Elites Archive and Dispatcher.

Maintains an N x M behavioral archive (default 10 x 10 = 100 niches) discovered during
evolutionary calibration. Provides automated PDF descriptor analysis and runtime
niche-specialized policy dispatch for incoming documents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

from .genome import ProfileGenome, ProfilePolicy


def compute_document_descriptors(pdf_bytes: bytes) -> tuple[float, float]:
    """Computes normalized behavioral descriptors (D1: cluster_density, D2: table_sensitivity) from PDF geometry.

    Returns:
        tuple[float, float]: (cluster_density, table_sensitivity), both in [0.0, 1.0].
    """
    if not HAS_FITZ:
        return 0.5, 0.5

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    line_gaps: list[float] = []
    font_sizes: list[float] = []
    has_table_drawings = 0
    total_drawings = 0
    col_x0s: list[float] = []

    for page in doc:
        drawings = page.get_drawings()
        total_drawings += len(drawings)
        rect_lines = [
            d for d in drawings
            if d.get("type") in ("s", "f", "fs") and len(d.get("items", [])) > 0
        ]
        if len(rect_lines) >= 4:
            has_table_drawings += len(rect_lines)

        blocks = page.get_text("rawdict").get("blocks", [])
        prev_y1: float | None = None
        for b in blocks:
            if b.get("type") == 0:  # text block
                for line in b.get("lines", []):
                    y0, y1 = line["bbox"][1], line["bbox"][3]
                    x0 = line["bbox"][0]
                    col_x0s.append(round(x0, 1))
                    if prev_y1 is not None and y0 >= prev_y1:
                        line_gaps.append(y0 - prev_y1)
                    prev_y1 = y1
                    for sp in line.get("spans", []):
                        font_sizes.append(float(sp.get("size", 11.0)))
    doc.close()

    # D1: Cluster Density (0.0 to 1.0): median line gap relative to font size
    avg_size = (sum(font_sizes) / len(font_sizes)) if font_sizes else 11.0
    if line_gaps:
        sorted_gaps = sorted(line_gaps)
        med_gap = sorted_gaps[len(sorted_gaps) // 2]
    else:
        med_gap = avg_size * 0.2

    gap_ratio = med_gap / max(1.0, avg_size)
    cluster_density = float(max(0.02, min(0.98, gap_ratio / 1.2)))

    # D2: Table Tendency / Sensitivity (0.0 to 1.0)
    col_counts: dict[float, int] = {}
    for x in col_x0s:
        col_counts[x] = col_counts.get(x, 0) + 1
    distinct_cols = sum(1 for c, cnt in col_counts.items() if cnt >= 2)

    table_score = 0.05
    if has_table_drawings >= 4 or total_drawings >= 8:
        table_score += 0.55
    if has_table_drawings >= 10 or total_drawings >= 16:
        table_score += 0.25
    if distinct_cols >= 4:
        table_score += 0.15

    table_sensitivity = float(max(0.05, min(0.95, table_score)))
    return round(cluster_density, 4), round(table_sensitivity, 4)


@dataclass
class ArchiveCell:
    """A single niche cell in the MAP-Elites behavioral archive."""

    coord: tuple[int, int]
    descriptors: dict[str, float]
    fitness: float
    policy: ProfilePolicy
    fingerprint: str
    last_evaluated_gen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "coord": list(self.coord),
            "descriptors": self.descriptors,
            "fitness": round(self.fitness, 4),
            "fingerprint": self.fingerprint,
            "last_evaluated_gen": self.last_evaluated_gen,
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArchiveCell:
        coord = tuple(d["coord"])
        policy = ProfilePolicy.from_dict(d["policy"])
        return cls(
            coord=(int(coord[0]), int(coord[1])),
            descriptors=d.get("descriptors", {}),
            fitness=float(d.get("fitness", 0.0)),
            policy=policy,
            fingerprint=str(d.get("fingerprint", "")),
            last_evaluated_gen=int(d.get("last_evaluated_gen", 0)),
        )


@dataclass
class MAPElitesArchive:
    """Behavioral quality-diversity grid preserving the elite policy per niche."""

    grid_x: int = 10
    grid_y: int = 10
    scale_x: float = 1.0
    scale_y: float = 1.0
    descriptor_names: list[str] = field(default_factory=lambda: ["cluster_density", "table_sensitivity"])
    cells: dict[tuple[int, int], ArchiveCell] = field(default_factory=dict)
    calibrated_at: str = ""

    def coord_from_values(self, val_x: float, val_y: float) -> tuple[int, int]:
        """Maps normalized continuous descriptor values to discrete grid coordinates."""
        cx = min(self.grid_x - 1, max(0, int(val_x / self.scale_x * self.grid_x)))
        cy = min(self.grid_y - 1, max(0, int(val_y / self.scale_y * self.grid_y)))
        return cx, cy

    def update_cell(
        self,
        coord: tuple[int, int],
        fitness: float,
        policy: ProfilePolicy,
        descriptors: dict[str, float],
        fingerprint: str,
        gen: int = 0,
    ) -> bool:
        """Inserts or replaces the elite in a niche if fitness is strictly superior."""
        cur = self.cells.get(coord)
        if cur is None or fitness > cur.fitness:
            self.cells[coord] = ArchiveCell(
                coord=coord,
                descriptors=descriptors,
                fitness=fitness,
                policy=policy,
                fingerprint=fingerprint,
                last_evaluated_gen=gen,
            )
            return True
        return False

    def get_cell(self, coord: tuple[int, int]) -> ArchiveCell | None:
        return self.cells.get(coord)

    def get_champion(self) -> ArchiveCell | None:
        """Returns the highest fitness elite across all occupied niches."""
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda c: c.fitness)

    def get_nearest_elite(self, val_x: float, val_y: float) -> ArchiveCell | None:
        """Retrieves the exact elite for (val_x, val_y), or the closest populated niche elite."""
        if not self.cells:
            return None

        target_coord = self.coord_from_values(val_x, val_y)
        if target_coord in self.cells:
            return self.cells[target_coord]

        # Nearest neighbor in continuous descriptor space
        best_cell: ArchiveCell | None = None
        min_dist = float("inf")

        for cell in self.cells.values():
            cx_val = cell.descriptors.get(self.descriptor_names[0], (cell.coord[0] + 0.5) / self.grid_x)
            cy_val = cell.descriptors.get(self.descriptor_names[1], (cell.coord[1] + 0.5) / self.grid_y)
            dist = math.hypot(val_x - cx_val, val_y - cy_val)
            if dist < min_dist:
                min_dist = dist
                best_cell = cell

        return best_cell

    @property
    def total_capacity(self) -> int:
        return self.grid_x * self.grid_y

    @property
    def occupied_count(self) -> int:
        return len(self.cells)

    @property
    def coverage(self) -> float:
        return self.occupied_count / max(1, self.total_capacity)

    @property
    def qd_score(self) -> float:
        """Quality Diversity Score: sum of fitnesses across all occupied cells."""
        return sum(cell.fitness for cell in self.cells.values())

    def to_dict(self) -> dict[str, Any]:
        champion = self.get_champion()
        return {
            "title": "MAP-Elites Calibrated Behavioral Archive",
            "grid_dimensions": [self.grid_x, self.grid_y],
            "total_niches": self.total_capacity,
            "occupied_niches": self.occupied_count,
            "coverage": round(self.coverage, 4),
            "qd_score": round(self.qd_score, 4),
            "descriptor_names": self.descriptor_names,
            "calibrated_at": self.calibrated_at,
            "champion": champion.to_dict() if champion else None,
            "cells": [c.to_dict() for c in sorted(self.cells.values(), key=lambda x: x.coord)],
        }

    def save_json(self, path: str | Path) -> Path:
        out_p = Path(path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_p

    @classmethod
    def load_json(cls, path: str | Path) -> MAPElitesArchive:
        in_p = Path(path).resolve()
        with open(in_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        dims = data.get("grid_dimensions", [10, 10])
        archive = cls(
            grid_x=dims[0],
            grid_y=dims[1],
            descriptor_names=data.get("descriptor_names", ["cluster_density", "table_sensitivity"]),
            calibrated_at=data.get("calibrated_at", ""),
        )
        for cd in data.get("cells", []):
            cell = ArchiveCell.from_dict(cd)
            archive.cells[cell.coord] = cell
        return archive


class MAPElitesPolicyDispatcher:
    """Dispatches specialized heuristic extraction policies according to document layout descriptors."""

    def __init__(self, archive: MAPElitesArchive | str | Path) -> None:
        if isinstance(archive, (str, Path)):
            self.archive = MAPElitesArchive.load_json(archive)
        elif isinstance(archive, MAPElitesArchive):
            self.archive = archive
        else:
            raise TypeError(f"Expected MAPElitesArchive or path, got {type(archive)}")

    def dispatch(self, pdf_bytes: bytes) -> tuple[ProfilePolicy, ArchiveCell, bool]:
        """Analyzes PDF geometry, retrieves optimal specialized elite policy, and flags exact niche match.

        Returns:
            tuple[ProfilePolicy, ArchiveCell, bool]:
                - The selected specialized ProfilePolicy.
                - The matched ArchiveCell.
                - True if the document's niche had an exact elite; False if nearest neighbor fallback was used.
        """
        d1, d2 = compute_document_descriptors(pdf_bytes)
        target_coord = self.archive.coord_from_values(d1, d2)
        exact_cell = self.archive.get_cell(target_coord)

        if exact_cell is not None:
            return exact_cell.policy, exact_cell, True

        nearest_cell = self.archive.get_nearest_elite(d1, d2)
        if nearest_cell is not None:
            return nearest_cell.policy, nearest_cell, False

        champion = self.archive.get_champion()
        if champion is not None:
            return champion.policy, champion, False

        # Fallback to default
        default_policy = ProfilePolicy()
        dummy_cell = ArchiveCell(
            coord=(0, 0),
            descriptors={"cluster_density": d1, "table_sensitivity": d2},
            fitness=0.0,
            policy=default_policy,
            fingerprint="default",
        )
        return default_policy, dummy_cell, False
