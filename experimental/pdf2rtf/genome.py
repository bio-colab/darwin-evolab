"""genome.py — ProfileGenome and ProfilePolicy for PDF-to-RTF Extraction.

Represents tunable heuristic policies (spacing ratios, paragraph clustering deltas,
alignment tolerances, and table detection thresholds) as an EvolabGenome for
evolutionary calibration via Darwin-Evolab and MAP-Elites.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from evolab.genome import EvolabGenome


@dataclass
class ProfilePolicy:
    """Tunable extraction heuristic policy."""

    space_gap_ratio: float = 0.25  # Fraction of font size for inter-span space
    para_split_delta_ratio: float = 1.40  # Multiple of line height to split paragraphs
    align_tolerance_pt: float = 5.0  # Points tolerance for center/right alignment
    line_spacing_round_pt: float = 0.5  # Rounding grid for vertical line spacing
    table_col_align_tol_pt: float = 8.0  # Column boundary grouping tolerance
    table_min_rows: int = 2  # Minimum rows to classify as table
    table_min_cols: int = 2  # Minimum columns to classify as table

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProfilePolicy:
        return cls(
            space_gap_ratio=float(d.get("space_gap_ratio", 0.25)),
            para_split_delta_ratio=float(d.get("para_split_delta_ratio", 1.40)),
            align_tolerance_pt=float(d.get("align_tolerance_pt", 5.0)),
            line_spacing_round_pt=float(d.get("line_spacing_round_pt", 0.5)),
            table_col_align_tol_pt=float(d.get("table_col_align_tol_pt", 8.0)),
            table_min_rows=int(d.get("table_min_rows", 2)),
            table_min_cols=int(d.get("table_min_cols", 2)),
        )


# Parameter bounds for valid search space exploration
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "space_gap_ratio": (0.10, 0.50),
    "para_split_delta_ratio": (1.05, 2.50),
    "align_tolerance_pt": (1.0, 15.0),
    "line_spacing_round_pt": (0.25, 2.0),
    "table_col_align_tol_pt": (2.0, 25.0),
    "table_min_rows": (2.0, 5.0),
    "table_min_cols": (2.0, 5.0),
}


@dataclass
class ProfileGenome(EvolabGenome):
    """Evolutionary genome parameterizing PDF-to-RTF heuristic policies."""

    space_gap_ratio: float = 0.25
    para_split_delta_ratio: float = 1.40
    align_tolerance_pt: float = 5.0
    line_spacing_round_pt: float = 0.5
    table_col_align_tol_pt: float = 8.0
    table_min_rows: int = 2
    table_min_cols: int = 2

    def __post_init__(self) -> None:
        self.clamp_bounds()

    def clamp_bounds(self) -> None:
        """Clamp parameters within physically valid ranges."""
        self.space_gap_ratio = max(0.10, min(0.50, float(self.space_gap_ratio)))
        self.para_split_delta_ratio = max(1.05, min(2.50, float(self.para_split_delta_ratio)))
        self.align_tolerance_pt = max(1.0, min(15.0, float(self.align_tolerance_pt)))
        self.line_spacing_round_pt = max(0.25, min(2.0, float(self.line_spacing_round_pt)))
        self.table_col_align_tol_pt = max(2.0, min(25.0, float(self.table_col_align_tol_pt)))
        self.table_min_rows = max(2, min(5, int(round(self.table_min_rows))))
        self.table_min_cols = max(2, min(5, int(round(self.table_min_cols))))

    def __len__(self) -> int:
        """Number of tunable heuristic policy dimensions."""
        return 7

    def clone(self) -> ProfileGenome:
        return copy.deepcopy(self)

    def fingerprint(self) -> str:
        """Deterministic fingerprint representing genome parameters."""
        raw = (
            f"{self.space_gap_ratio:.4f}:{self.para_split_delta_ratio:.4f}:"
            f"{self.align_tolerance_pt:.2f}:{self.line_spacing_round_pt:.2f}:"
            f"{self.table_col_align_tol_pt:.2f}:{self.table_min_rows}:{self.table_min_cols}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        """Normalized L1 distance in [0, 1] across parameter dimensions."""
        if not isinstance(other, ProfileGenome):
            raise TypeError(f"Cannot compare ProfileGenome with {type(other)}")

        total_norm_dist = 0.0
        keys = [
            "space_gap_ratio",
            "para_split_delta_ratio",
            "align_tolerance_pt",
            "line_spacing_round_pt",
            "table_col_align_tol_pt",
            "table_min_rows",
            "table_min_cols",
        ]
        for k in keys:
            lo, hi = PARAM_BOUNDS[k]
            v1 = getattr(self, k)
            v2 = getattr(other, k)
            total_norm_dist += abs(v1 - v2) / (hi - lo)

        return total_norm_dist / len(keys)

    def serialize(self) -> dict[str, Any]:
        return {
            "space_gap_ratio": round(self.space_gap_ratio, 4),
            "para_split_delta_ratio": round(self.para_split_delta_ratio, 4),
            "align_tolerance_pt": round(self.align_tolerance_pt, 2),
            "line_spacing_round_pt": round(self.line_spacing_round_pt, 2),
            "table_col_align_tol_pt": round(self.table_col_align_tol_pt, 2),
            "table_min_rows": self.table_min_rows,
            "table_min_cols": self.table_min_cols,
            "fingerprint": self.fingerprint(),
        }

    def describe(self) -> dict[str, float | int | str]:
        """Behavioral descriptors for MAP-Elites archive coordinate projection."""
        # D1: Clustering density: high space gap + high split delta means sparse clustering
        lo_s, hi_s = PARAM_BOUNDS["space_gap_ratio"]
        lo_p, hi_p = PARAM_BOUNDS["para_split_delta_ratio"]
        norm_s = (self.space_gap_ratio - lo_s) / (hi_s - lo_s)
        norm_p = (self.para_split_delta_ratio - lo_p) / (hi_p - lo_p)
        cluster_density = round((norm_s * 0.5) + (norm_p * 0.5), 4)

        # D2: Table sensitivity: high col align tol + low min rows means aggressive table formation
        lo_t, hi_t = PARAM_BOUNDS["table_col_align_tol_pt"]
        norm_t = (self.table_col_align_tol_pt - lo_t) / (hi_t - lo_t)
        table_sensitivity = round(norm_t, 4)

        return {
            "cluster_density": cluster_density,
            "table_sensitivity": table_sensitivity,
            "table_min_rows": self.table_min_rows,
        }

    def mutate(self, rng: random.Random | None = None, sigma: float = 0.15, **kwargs: Any) -> ProfileGenome:
        """Perturb parameters with bounded Gaussian noise."""
        r = rng if rng is not None else random
        child = self.clone()

        # Mutate float parameters
        child.space_gap_ratio += r.gauss(0, (0.50 - 0.10) * sigma)
        child.para_split_delta_ratio += r.gauss(0, (2.50 - 1.05) * sigma)
        child.align_tolerance_pt += r.gauss(0, (15.0 - 1.0) * sigma)
        child.line_spacing_round_pt += r.gauss(0, (2.0 - 0.25) * sigma)
        child.table_col_align_tol_pt += r.gauss(0, (25.0 - 2.0) * sigma)

        # Mutate integer parameters probabilistically
        if r.random() < 0.30:
            child.table_min_rows += r.choice([-1, 1])
        if r.random() < 0.30:
            child.table_min_cols += r.choice([-1, 1])

        child.clamp_bounds()
        return child

    def crossover(self, other: EvolabGenome, rng: random.Random | None = None) -> ProfileGenome:
        """Blend crossover combining parents' policies."""
        if not isinstance(other, ProfileGenome):
            return self.clone()

        r = rng if rng is not None else random
        alpha = r.uniform(0.2, 0.8)

        child = ProfileGenome(
            space_gap_ratio=alpha * self.space_gap_ratio + (1 - alpha) * other.space_gap_ratio,
            para_split_delta_ratio=alpha * self.para_split_delta_ratio + (1 - alpha) * other.para_split_delta_ratio,
            align_tolerance_pt=alpha * self.align_tolerance_pt + (1 - alpha) * other.align_tolerance_pt,
            line_spacing_round_pt=alpha * self.line_spacing_round_pt + (1 - alpha) * other.line_spacing_round_pt,
            table_col_align_tol_pt=alpha * self.table_col_align_tol_pt + (1 - alpha) * other.table_col_align_tol_pt,
            table_min_rows=int(round(alpha * self.table_min_rows + (1 - alpha) * other.table_min_rows)),
            table_min_cols=int(round(alpha * self.table_min_cols + (1 - alpha) * other.table_min_cols)),
        )
        child.clamp_bounds()
        return child

    def to_policy(self) -> ProfilePolicy:
        """Produce runtime ProfilePolicy instance."""
        return ProfilePolicy(
            space_gap_ratio=self.space_gap_ratio,
            para_split_delta_ratio=self.para_split_delta_ratio,
            align_tolerance_pt=self.align_tolerance_pt,
            line_spacing_round_pt=self.line_spacing_round_pt,
            table_col_align_tol_pt=self.table_col_align_tol_pt,
            table_min_rows=self.table_min_rows,
            table_min_cols=self.table_min_cols,
        )

    @classmethod
    def from_policy(cls, policy: ProfilePolicy) -> ProfileGenome:
        return cls(
            space_gap_ratio=policy.space_gap_ratio,
            para_split_delta_ratio=policy.para_split_delta_ratio,
            align_tolerance_pt=policy.align_tolerance_pt,
            line_spacing_round_pt=policy.line_spacing_round_pt,
            table_col_align_tol_pt=policy.table_col_align_tol_pt,
            table_min_rows=policy.table_min_rows,
            table_min_cols=policy.table_min_cols,
        )
