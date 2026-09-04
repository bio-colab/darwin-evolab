"""pvt_evaluator.py — PVT-Aware Evaluation and Conservative Hindsight Replay.

Distilled from PPAAS (ICCAD 2025):
1. Skip-on-Fail simulation acceleration: Evaluates nominal corner (TT) first;
   if basic electrical/operating sanity fails, immediately aborts remaining corners (SS, FF)
   to save ~80% simulation compute.
2. Worst-case robust PVT signoff across process, voltage, and temperature corners.
3. Conservative Hindsight Experience Replay (HER): Retains non-dominated intermediate
   circuits as goal-conditioned exemplars on the empirical Pareto frontier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import FloatGenome, Individual
from evolab.pareto import Objective

from .opamp_benchmark import (
    OpAmpPerformanceMetrics,
    OpAmpSizing,
    evaluate_opamp_analytical,
)
from .sky130_pdk import Sky130Corner


@dataclass
class MultiCornerEvaluationResult:
    """Consolidated performance metrics across evaluated PVT corners."""
    nominal_metrics: OpAmpPerformanceMetrics
    corner_metrics: dict[Sky130Corner, OpAmpPerformanceMetrics]
    skipped_corners: bool
    passed_nominal: bool
    worst_gain_db: float
    worst_gbw_mhz: float
    worst_pm_deg: float
    max_power_uw: float
    all_corners_pass: bool
    execution_time_savings_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skipped_corners": self.skipped_corners,
            "passed_nominal": self.passed_nominal,
            "worst_gain_db": self.worst_gain_db,
            "worst_gbw_mhz": self.worst_gbw_mhz,
            "worst_pm_deg": self.worst_pm_deg,
            "max_power_uw": self.max_power_uw,
            "all_corners_pass": self.all_corners_pass,
            "corners_evaluated": [c.value for c in self.corner_metrics.keys()],
            "execution_time_savings_pct": self.execution_time_savings_pct,
        }


class PVTAwareOpAmpEvaluator(Evaluator):
    """Evaluates OpAmp designs across SkyWater 130nm PVT corners with Skip-on-Fail acceleration.

    Distilled from PPAAS (ICCAD 2025):
    - Nominal TT corner evaluated first.
    - If nominal gain < min_nominal_gain_db or nominal PM < min_nominal_pm_deg,
      subsequent corners are skipped, saving compute while reliably rejecting unviable candidates.
    """

    def __init__(
        self,
        target_gain_db: float = 60.0,
        target_gbw_mhz: float = 10.0,
        target_pm_deg: float = 60.0,
        target_max_power_uw: float = 600.0,
        corners: Sequence[Sky130Corner] | None = None,
        skip_on_fail: bool = True,
        min_nominal_gain_db: float = 15.0,
        min_nominal_pm_deg: float = 15.0,
    ):
        self.target_gain_db = target_gain_db
        self.target_gbw_mhz = target_gbw_mhz
        self.target_pm_deg = target_pm_deg
        self.target_max_power_uw = target_max_power_uw
        self.corners = list(corners) if corners is not None else [
            Sky130Corner.TT,
            Sky130Corner.SS,
            Sky130Corner.FF,
        ]
        self.skip_on_fail = skip_on_fail
        self.min_nominal_gain_db = min_nominal_gain_db
        self.min_nominal_pm_deg = min_nominal_pm_deg

    @property
    def deterministic(self) -> bool:
        return True

    def _extract_sizing(self, ind_or_genome: Any) -> OpAmpSizing:
        g = getattr(ind_or_genome, "genome", ind_or_genome)
        if isinstance(g, FloatGenome) or hasattr(g, "values") or hasattr(g, "genes"):
            vals = list(getattr(g, "values", getattr(g, "genes", [])))
            return OpAmpSizing.from_normalized_vector(vals)
        elif isinstance(g, OpAmpSizing):
            return g
        return OpAmpSizing()

    def evaluate_multi_corner(self, ind_or_genome: Any) -> MultiCornerEvaluationResult:
        """Evaluates sizing across PVT corners with PPAAS Skip-on-Fail logic."""
        sizing = self._extract_sizing(ind_or_genome)

        # 1. Evaluate Nominal corner first (TT)
        nominal_corner = Sky130Corner.TT
        nom_metrics = evaluate_opamp_analytical(sizing, corner=nominal_corner)

        # Check nominal sanity
        passed_nominal = (
            nom_metrics.gain_db >= self.min_nominal_gain_db
            and nom_metrics.pm_deg >= self.min_nominal_pm_deg
        )

        results: dict[Sky130Corner, OpAmpPerformanceMetrics] = {nominal_corner: nom_metrics}

        total_corners = len(self.corners)
        if self.skip_on_fail and not passed_nominal:
            # Skip remaining corners to save compute
            savings_pct = max(0.0, (total_corners - 1) / max(total_corners, 1) * 100.0)
            return MultiCornerEvaluationResult(
                nominal_metrics=nom_metrics,
                corner_metrics=results,
                skipped_corners=True,
                passed_nominal=False,
                worst_gain_db=nom_metrics.gain_db,
                worst_gbw_mhz=nom_metrics.gbw_mhz,
                worst_pm_deg=nom_metrics.pm_deg,
                max_power_uw=nom_metrics.power_uw,
                all_corners_pass=False,
                execution_time_savings_pct=round(savings_pct, 1),
            )

        # 2. Evaluate remaining corners
        for c in self.corners:
            if c == nominal_corner:
                continue
            m = evaluate_opamp_analytical(sizing, corner=c)
            results[c] = m

        worst_gain = min(m.gain_db for m in results.values())
        worst_gbw = min(m.gbw_mhz for m in results.values())
        worst_pm = min(m.pm_deg for m in results.values())
        max_power = max(m.power_uw for m in results.values())

        all_pass = all(
            m.gain_db >= self.target_gain_db
            and m.gbw_mhz >= self.target_gbw_mhz
            and m.pm_deg >= self.target_pm_deg
            and m.power_uw <= self.target_max_power_uw
            for m in results.values()
        )

        return MultiCornerEvaluationResult(
            nominal_metrics=nom_metrics,
            corner_metrics=results,
            skipped_corners=False,
            passed_nominal=True,
            worst_gain_db=round(worst_gain, 2),
            worst_gbw_mhz=round(worst_gbw, 2),
            worst_pm_deg=round(worst_pm, 2),
            max_power_uw=round(max_power, 2),
            all_corners_pass=all_pass,
            execution_time_savings_pct=0.0,
        )

    def evaluate(self, ind: Individual) -> FitnessResult:
        res = self.evaluate_multi_corner(ind)

        if res.skipped_corners:
            # Heavily penalized for non-functional nominal operating point
            penalized_score = max(0.0, res.nominal_metrics.gain_db / max(self.target_gain_db, 1.0) * 10.0)
            return FitnessResult(
                score=round(penalized_score, 3),
                passed_holdout=False,
                artifacts={
                    "multi_corner": res.to_dict(),
                    "note": "PPAAS skip_on_fail triggered: unviable nominal operating point",
                },
            )

        # Robust multi-corner fitness based on worst-case corners
        gain_score = min(res.worst_gain_db / max(self.target_gain_db, 1.0), 1.25) * 35.0
        gbw_score = min(res.worst_gbw_mhz / max(self.target_gbw_mhz, 0.1), 1.5) * 25.0

        if res.worst_pm_deg >= 60.0:
            pm_score = 25.0
        elif res.worst_pm_deg >= 45.0:
            pm_score = 15.0 + (res.worst_pm_deg - 45.0) / 15.0 * 10.0
        else:
            pm_score = max(res.worst_pm_deg / 45.0 * 15.0, 0.0)

        power_score = max(0.0, 15.0 * (1.0 - (res.max_power_uw / (self.target_max_power_uw * 2.0))))
        total_score = max(0.0, min(gain_score + gbw_score + pm_score + power_score, 100.0))

        return FitnessResult(
            score=round(total_score, 3),
            passed_holdout=res.all_corners_pass,
            artifacts={
                "multi_corner": res.to_dict(),
                "worst_gain_db": res.worst_gain_db,
                "worst_gbw_mhz": res.worst_gbw_mhz,
                "worst_pm_deg": res.worst_pm_deg,
                "max_power_uw": res.max_power_uw,
            },
        )

    def build_pareto_objectives(self) -> list[Objective]:
        """PVT-Aware robust objectives for multi-objective Pareto optimization."""
        return [
            Objective(name="Worst_Gain_dB", direction="maximize", weight=1.0),
            Objective(name="Worst_GBW_MHz", direction="maximize", weight=1.0),
            Objective(name="Worst_PM_Stability", direction="maximize", weight=1.0),
            Objective(name="Max_Power_uW", direction="minimize", weight=1.0),
        ]

    def evaluate_pareto_vector(self, ind: Individual) -> dict[str, float]:
        res = self.evaluate_multi_corner(ind)
        return {
            "Worst_Gain_dB": res.worst_gain_db,
            "Worst_GBW_MHz": res.worst_gbw_mhz,
            "Worst_PM_Stability": -abs(res.worst_pm_deg - 65.0),
            "Max_Power_uW": res.max_power_uw,
        }


@dataclass
class HindsightExperience:
    """Stored analog sizing solution retained via Conservative Hindsight Replay."""
    sizing: OpAmpSizing
    gain_db: float
    gbw_mhz: float
    pm_deg: float
    power_uw: float
    corner: Sky130Corner = Sky130Corner.TT
    generation: int = 0


class ConservativeHindsightReplay:
    """Conservative Hindsight Experience Replay buffer for analog sizing.

    Distilled from PPAAS (ICCAD 2025):
    Maintains an archive of empirical Pareto-frontier solutions. When an individual
    misses a strict user target (e.g. 70dB gain) but achieves an interesting alternative
    trade-off (e.g. high bandwidth or ultra-low power), it is preserved as an empirical
    goal-conditioned exemplar rather than discarded.
    """

    def __init__(self, max_capacity: int = 200):
        self.max_capacity = max_capacity
        self.archive: list[HindsightExperience] = []

    def __len__(self) -> int:
        return len(self.archive)

    def _dominates(self, a: HindsightExperience, b: HindsightExperience) -> bool:
        """Checks if a Pareto-dominates b across (Gain, GBW, PM, -Power)."""
        a_pm_dev = abs(a.pm_deg - 65.0)
        b_pm_dev = abs(b.pm_deg - 65.0)

        not_worse = (
            a.gain_db >= b.gain_db
            and a.gbw_mhz >= b.gbw_mhz
            and a_pm_dev <= b_pm_dev
            and a.power_uw <= b.power_uw
        )
        strictly_better = (
            a.gain_db > b.gain_db
            or a.gbw_mhz > b.gbw_mhz
            or a_pm_dev < b_pm_dev
            or a.power_uw < b.power_uw
        )
        return not_worse and strictly_better

    def add(
        self,
        sizing: OpAmpSizing,
        metrics: OpAmpPerformanceMetrics,
        generation: int = 0,
    ) -> bool:
        """Adds a candidate if it is non-dominated or represents a viable Pareto frontier point."""
        # Sanity check: must be functional
        if metrics.gain_db < 20.0 or metrics.pm_deg < 20.0:
            return False

        candidate = HindsightExperience(
            sizing=sizing,
            gain_db=metrics.gain_db,
            gbw_mhz=metrics.gbw_mhz,
            pm_deg=metrics.pm_deg,
            power_uw=metrics.power_uw,
            generation=generation,
        )

        # Check if candidate is dominated by existing archive
        for existing in self.archive:
            if self._dominates(existing, candidate):
                return False

        # Filter out existing entries that the candidate now dominates
        self.archive = [e for e in self.archive if not self._dominates(candidate, e)]

        self.archive.append(candidate)

        # Trim archive if over capacity by removing points closest to neighbors
        if len(self.archive) > self.max_capacity:
            self.archive.sort(key=lambda x: x.gain_db + x.gbw_mhz, reverse=True)
            self.archive = self.archive[: self.max_capacity]

        return True

    def sample_seeds(self, count: int = 5) -> list[OpAmpSizing]:
        """Returns diverse seed sizings from the current Pareto archive."""
        if not self.archive:
            return [OpAmpSizing()]
        import random
        selected = random.sample(self.archive, min(count, len(self.archive)))
        return [e.sizing for e in selected]
