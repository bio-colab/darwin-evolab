"""spec_types.py — Target Circuit Specifications and Scoring for Spec2Ckt Lab.

Standardized specifications inspired by CktBench and OCB (Open Circuit Benchmark).
Provides continuous multi-objective satisfaction metrics and tolerance gating.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from evolab.silicon.opamp_benchmark import OpAmpPerformanceMetrics


@dataclass
class SpecTolerance:
    """Allowed tolerance margins for target specifications."""
    gain_rel_tol: float = 0.05    # 5% relative tolerance
    gbw_rel_tol: float = 0.10     # 10% relative tolerance
    pm_abs_deg_tol: float = 5.0   # 5 degrees tolerance
    power_rel_slack: float = 0.10 # Allow up to 10% above max power
    sr_rel_slack: float = 0.10    # Allow up to 10% below min slew rate


@dataclass
class TargetCircuitSpec:
    """Target performance requirements for analog circuit synthesis."""
    name: str = "custom_opamp"
    gain_db: float = 60.0          # Target low-frequency gain (dB)
    gbw_mhz: float = 10.0          # Target Gain-Bandwidth Product (MHz)
    pm_deg: float = 60.0           # Target Phase Margin (degrees)
    max_power_uw: float = 600.0    # Maximum allowable power dissipation (uW)
    min_slew_rate_v_us: float = 8.0  # Minimum Slew Rate (V/us)
    min_cmrr_db: float = 50.0      # Minimum CMRR (dB)

    def to_spec_vector(self) -> list[float]:
        """Returns normalized target vector [gain, gbw, pm, power, sr]."""
        return [
            self.gain_db / 100.0,
            self.gbw_mhz / 50.0,
            self.pm_deg / 90.0,
            self.max_power_uw / 2000.0,
            self.min_slew_rate_v_us / 50.0,
        ]

    def evaluate_satisfaction(
        self,
        metrics: OpAmpPerformanceMetrics,
        tol: SpecTolerance | None = None,
    ) -> dict[str, Any]:
        """Evaluates whether the circuit metrics satisfy all specifications."""
        tol = tol or SpecTolerance()

        gain_ok = metrics.gain_db >= self.gain_db * (1.0 - tol.gain_rel_tol)
        gbw_ok = metrics.gbw_mhz >= self.gbw_mhz * (1.0 - tol.gbw_rel_tol)
        pm_ok = metrics.pm_deg >= (self.pm_deg - tol.pm_abs_deg_tol)
        power_ok = metrics.power_uw <= self.max_power_uw * (1.0 + tol.power_rel_slack)
        sr_ok = metrics.slew_rate_v_us >= self.min_slew_rate_v_us * (1.0 - tol.sr_rel_slack)
        stable_ok = metrics.is_stable

        all_ok = all([gain_ok, gbw_ok, pm_ok, power_ok, sr_ok, stable_ok])

        # Continuous normalized satisfaction score [0.0, 1.0]
        score_gain = min(1.0, metrics.gain_db / max(self.gain_db, 1.0))
        score_gbw = min(1.0, metrics.gbw_mhz / max(self.gbw_mhz, 0.1))
        score_pm = min(1.0, metrics.pm_deg / max(self.pm_deg, 1.0)) if metrics.is_stable else 0.2
        score_pwr = min(1.0, self.max_power_uw / max(metrics.power_uw, 1.0))
        score_sr = min(1.0, metrics.slew_rate_v_us / max(self.min_slew_rate_v_us, 0.1))

        composite_score = 0.25 * score_gain + 0.25 * score_gbw + 0.20 * score_pm + 0.15 * score_pwr + 0.15 * score_sr
        if not metrics.is_stable:
            composite_score *= 0.5

        return {
            "all_satisfied": all_ok,
            "composite_score": round(composite_score, 4),
            "gain_ok": gain_ok,
            "gbw_ok": gbw_ok,
            "pm_ok": pm_ok,
            "power_ok": power_ok,
            "sr_ok": sr_ok,
            "stable_ok": stable_ok,
            "metrics": metrics.to_dict(),
        }

    def compute_fitness(self, metrics: OpAmpPerformanceMetrics) -> float:
        """Computes evolutionary scalar fitness from performance metrics against this spec."""
        res = self.evaluate_satisfaction(metrics)
        score = float(res["composite_score"])
        # Bonus for meeting all criteria simultaneously
        if res["all_satisfied"]:
            score += 0.5
        return score


# Standard benchmark targets representing real-world IC applications (CktBench style)
CKTBENCH_LOW_POWER = TargetCircuitSpec(
    name="CktBench_LowPower",
    gain_db=55.0,
    gbw_mhz=6.0,
    pm_deg=65.0,
    max_power_uw=300.0,
    min_slew_rate_v_us=5.0,
    min_cmrr_db=50.0,
)

CKTBENCH_BALANCED = TargetCircuitSpec(
    name="CktBench_Balanced",
    gain_db=65.0,
    gbw_mhz=12.0,
    pm_deg=60.0,
    max_power_uw=600.0,
    min_slew_rate_v_us=10.0,
    min_cmrr_db=55.0,
)

CKTBENCH_HIGH_GAIN = TargetCircuitSpec(
    name="CktBench_HighGain",
    gain_db=75.0,
    gbw_mhz=15.0,
    pm_deg=55.0,
    max_power_uw=850.0,
    min_slew_rate_v_us=12.0,
    min_cmrr_db=60.0,
)

CKTBENCH_HIGH_SPEED = TargetCircuitSpec(
    name="CktBench_HighSpeed",
    gain_db=60.0,
    gbw_mhz=22.0,
    pm_deg=52.0,
    max_power_uw=1100.0,
    min_slew_rate_v_us=18.0,
    min_cmrr_db=50.0,
)

BENCHMARK_SPECS: dict[str, TargetCircuitSpec] = {
    "low_power": CKTBENCH_LOW_POWER,
    "balanced": CKTBENCH_BALANCED,
    "high_gain": CKTBENCH_HIGH_GAIN,
    "high_speed": CKTBENCH_HIGH_SPEED,
}
