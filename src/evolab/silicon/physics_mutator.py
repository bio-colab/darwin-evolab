"""physics_mutator.py — Physics-Informed Diagnostic Repair Mutator.

Distilled from AnalogCoder-Pro (IEEE TCAD 2026):
Replaces blind random Gaussian perturbation with physics-directed diagnostic repairs:
- Diagnoses specific circuit deficiencies (phase margin loss/oscillation, DC gain deficit,
  bandwidth starvation, slew rate limitation, or power overrun).
- Maps symptoms to exact physical root causes (Cc, W6, L1, L3, Wtail, Ibias).
- Guarantees electrical & DRC rule compliance via PhysicalGrammarGuard.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

from evolab.genome import FloatGenome, Individual

from .grammar_guard import PhysicalGrammarGuard
from .opamp_benchmark import (
    OpAmpPerformanceMetrics,
    OpAmpSizing,
    evaluate_opamp_analytical,
)
from .sky130_pdk import Sky130Corner


@dataclass(frozen=True)
class CircuitDiagnosis:
    """Diagnostic report linking measured metric deficits to physical mechanisms."""
    pm_deficit: bool = False
    gain_deficit: bool = False
    gbw_deficit: bool = False
    slew_deficit: bool = False
    power_excess: bool = False
    dominant_symptom: str = "none"


class PhysicsInformedOpAmpMutator:
    """Physics-informed directed mutator for Two-Stage Miller CMOS OpAmps.

    Distilled from AnalogCoder-Pro:
    Connects closed-loop simulation diagnostics directly to physical transistor dimension adjustments.
    """

    def __init__(
        self,
        guard: PhysicalGrammarGuard | None = None,
        target_gain_db: float = 60.0,
        target_gbw_mhz: float = 10.0,
        target_pm_deg: float = 60.0,
        target_max_power_uw: float = 600.0,
        target_slew_v_us: float = 10.0,
    ):
        self.guard = guard or PhysicalGrammarGuard()
        self.target_gain_db = target_gain_db
        self.target_gbw_mhz = target_gbw_mhz
        self.target_pm_deg = target_pm_deg
        self.target_max_power_uw = target_max_power_uw
        self.target_slew_v_us = target_slew_v_us

    def diagnose(self, metrics: OpAmpPerformanceMetrics) -> CircuitDiagnosis:
        """Analyzes simulation metrics to identify active electrical bottlenecks."""
        pm_def = metrics.pm_deg < self.target_pm_deg
        gain_def = metrics.gain_db < self.target_gain_db
        gbw_def = metrics.gbw_mhz < self.target_gbw_mhz
        slew_def = metrics.slew_rate_v_us < self.target_slew_v_us
        power_exc = metrics.power_uw > self.target_max_power_uw

        # Determine dominant symptom priority: Stability > Slew > Gain > Bandwidth > Power
        if pm_def and metrics.pm_deg < 45.0:
            dominant = "critical_instability"
        elif pm_def:
            dominant = "phase_margin_deficit"
        elif gain_def:
            dominant = "dc_gain_deficit"
        elif gbw_def:
            dominant = "bandwidth_deficit"
        elif slew_def:
            dominant = "slew_rate_deficit"
        elif power_exc:
            dominant = "excess_power"
        else:
            dominant = "nominal_fine_tuning"

        return CircuitDiagnosis(
            pm_deficit=pm_def,
            gain_deficit=gain_def,
            gbw_deficit=gbw_def,
            slew_deficit=slew_def,
            power_excess=power_exc,
            dominant_symptom=dominant,
        )

    def mutate_sizing(
        self,
        sizing: OpAmpSizing,
        metrics: OpAmpPerformanceMetrics | None = None,
        rng: random.Random | None = None,
    ) -> OpAmpSizing:
        """Applies physics-directed adjustments to repair diagnosed deficits."""
        r = rng or random.Random()
        m = metrics or evaluate_opamp_analytical(sizing)
        diag = self.diagnose(m)

        new_s = copy.deepcopy(sizing)

        # 1. Stability Deficit (PM < target):
        # Action: Increase Cc to pull dominant pole inward; increase W6 to push output pole p2 and zero z1
        if diag.pm_deficit:
            cc_boost = r.uniform(1.15, 1.40)
            new_s.cc_pf = min(10.0, new_s.cc_pf * cc_boost)
            w6_boost = r.uniform(1.10, 1.30)
            new_s.w6_um = min(120.0, new_s.w6_um * w6_boost)

        # 2. Gain Deficit (Av < target):
        # Action: Increase L1 and L3 to increase output resistances ro1, ro3 (reduce channel length modulation)
        # Also increase W1/L1 ratio to increase gm1
        if diag.gain_deficit:
            l_boost = r.uniform(1.10, 1.30)
            new_s.l1_um = min(2.0, new_s.l1_um * l_boost)
            new_s.l3_um = min(2.0, new_s.l3_um * l_boost)
            w1_boost = r.uniform(1.10, 1.25)
            new_s.w1_um = min(50.0, new_s.w1_um * w1_boost)

        # 3. Bandwidth Deficit (GBW < target):
        # Action: Increase gm1 by increasing W1 and tail current (W5)
        if diag.gbw_deficit and not diag.pm_deficit:
            new_s.w1_um = min(50.0, new_s.w1_um * r.uniform(1.15, 1.35))
            new_s.w5_um = min(60.0, new_s.w5_um * r.uniform(1.10, 1.25))

        # 4. Slew Rate Deficit (SR < target):
        # Action: Increase tail current I5 (via W5 and Ibias)
        if diag.slew_deficit:
            new_s.w5_um = min(60.0, new_s.w5_um * r.uniform(1.15, 1.30))
            new_s.ibias_ua = min(50.0, new_s.ibias_ua * r.uniform(1.10, 1.25))

        # 5. Excess Power (P > target):
        # Action: Throttles bias current and second stage sink
        if diag.power_excess:
            new_s.ibias_ua = max(2.0, new_s.ibias_ua * r.uniform(0.80, 0.92))
            new_s.w7_um = max(2.0, new_s.w7_um * r.uniform(0.85, 0.95))

        # 6. Fine-tuning jitter if already compliant or close
        if diag.dominant_symptom == "nominal_fine_tuning":
            # Gentle explore
            param_names = ["w1_um", "w3_um", "w6_um", "cc_pf", "ibias_ua"]
            target_param = r.choice(param_names)
            curr = getattr(new_s, target_param)
            setattr(new_s, target_param, curr * r.uniform(0.95, 1.05))

        # 7. Apply PhysicalGrammarGuard to clamp and repair geometric / DRC violations
        repaired_s = self.guard.repair(new_s)
        return repaired_s

    def mutate_individual(
        self,
        ind: Individual,
        rng: random.Random | None = None,
    ) -> Individual:
        """Mutates an Individual's FloatGenome via physics-informed diagnostic repair."""
        vals = list(getattr(ind.genome, "values", getattr(ind.genome, "genes", [])))
        sizing = OpAmpSizing.from_normalized_vector(vals)
        metrics = evaluate_opamp_analytical(sizing)

        mutated_sizing = self.mutate_sizing(sizing, metrics=metrics, rng=rng)
        new_vec = mutated_sizing.to_normalized_vector()

        new_ind = copy.deepcopy(ind)
        new_ind.genome = FloatGenome(values=new_vec)
        new_ind.fitness = 0.0
        return new_ind
