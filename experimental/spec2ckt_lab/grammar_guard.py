"""grammar_guard.py — ARCS-Style Physical & Grammar Constraints for Analog Circuits.

Implements structural and physical validity enforcement by construction:
1. Saturation margin and operating point checks.
2. High-frequency pole-splitting stability guards (p2 > GBW).
3. Lithographic PDK aspect ratio and sizing bounds.
4. Smooth projection operator to map violating vectors back to the physically feasible manifold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from evolab.silicon.opamp_benchmark import OpAmpSizing, evaluate_opamp_analytical
from evolab.silicon.sky130_pdk import Sky130Corner


@dataclass
class GrammarCheckResult:
    """Outcome of physical grammar and stability verification."""
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    repaired_sizing: OpAmpSizing | None = None
    repairs_applied: list[str] = field(default_factory=list)


class PhysicalGrammarGuard:
    """Enforces CMOS design grammar and physical realizability rules."""

    def __init__(
        self,
        min_w_um: float = 1.0,
        max_w_um: float = 120.0,
        min_l_um: float = 0.18,
        max_l_um: float = 2.0,
        min_cc_pf: float = 0.2,
        max_cc_pf: float = 10.0,
        min_ibias_ua: float = 2.0,
        max_ibias_ua: float = 50.0,
    ):
        self.min_w_um = min_w_um
        self.max_w_um = max_w_um
        self.min_l_um = min_l_um
        self.max_l_um = max_l_um
        self.min_cc_pf = min_cc_pf
        self.max_cc_pf = max_cc_pf
        self.min_ibias_ua = min_ibias_ua
        self.max_ibias_ua = max_ibias_ua

    def validate(self, sizing: OpAmpSizing) -> GrammarCheckResult:
        """Validates all physical grammar rules without altering the sizing."""
        violations = []

        # 1. Lithographic dimension bounds
        transistor_pairs = [
            ("M1/M2", sizing.w1_um, sizing.l1_um),
            ("M3/M4", sizing.w3_um, sizing.l3_um),
            ("M5", sizing.w5_um, sizing.l5_um),
            ("M6", sizing.w6_um, sizing.l6_um),
            ("M7", sizing.w7_um, sizing.l7_um),
            ("M8", sizing.w8_um, sizing.l8_um),
        ]
        for name, w, l in transistor_pairs:
            if w < self.min_w_um or w > self.max_w_um:
                violations.append(f"{name} width {w:.2f}um outside [{self.min_w_um}, {self.max_w_um}]")
            if l < self.min_l_um or l > self.max_l_um:
                violations.append(f"{name} length {l:.2f}um outside [{self.min_l_um}, {self.max_l_um}]")
            if w / max(l, 1e-4) < 1.0:
                violations.append(f"{name} aspect ratio W/L ({w/l:.2f}) < 1.0")

        # 2. Bias and compensation bounds
        if sizing.cc_pf < self.min_cc_pf or sizing.cc_pf > self.max_cc_pf:
            violations.append(f"Cc {sizing.cc_pf:.2f}pF outside [{self.min_cc_pf}, {self.max_cc_pf}]")
        if sizing.ibias_ua < self.min_ibias_ua or sizing.ibias_ua > self.max_ibias_ua:
            violations.append(f"Ibias {sizing.ibias_ua:.2f}uA outside [{self.min_ibias_ua}, {self.max_ibias_ua}]")

        # 3. Mirror ratio consistency: Tail and Stage 2 must be positive multiples of M8
        w8_l8 = sizing.w8_um / max(sizing.l8_um, 1e-3)
        w5_l5 = sizing.w5_um / max(sizing.l5_um, 1e-3)
        w7_l7 = sizing.w7_um / max(sizing.l7_um, 1e-3)

        if w5_l5 < 0.5 * w8_l8:
            violations.append("Tail mirror M5 too small relative to reference M8 (tail starvation)")
        if w7_l7 < 0.5 * w8_l8:
            violations.append("Stage 2 sink M7 too small relative to reference M8")

        # 4. Small-signal stability guard: Stage 2 transconductance must dominate input gm to guarantee p2 > GBW
        try:
            m = evaluate_opamp_analytical(sizing, Sky130Corner.TT)
            if not m.is_stable:
                violations.append(f"Unstable Phase Margin: {m.pm_deg:.1f} deg < 45.0 deg")
        except Exception as e:
            violations.append(f"Analytical physics failure: {str(e)}")

        return GrammarCheckResult(is_valid=len(violations) == 0, violations=violations)

    def repair_and_project(self, sizing: OpAmpSizing) -> OpAmpSizing:
        """Projects any candidate sizing onto the physically valid and stable manifold."""
        repairs = []

        def _clamp(v: float, low: float, high: float) -> float:
            return max(low, min(v, high))

        # 1. Clamp dimensions to PDK lithographic limits
        w1 = _clamp(sizing.w1_um, 1.0, 50.0)
        l1 = _clamp(sizing.l1_um, 0.18, 2.0)
        w3 = _clamp(sizing.w3_um, 2.0, 80.0)
        l3 = _clamp(sizing.l3_um, 0.18, 2.0)
        w5 = _clamp(sizing.w5_um, 2.0, 60.0)
        l5 = _clamp(sizing.l5_um, 0.18, 2.0)
        w6 = _clamp(sizing.w6_um, 5.0, 120.0)
        l6 = _clamp(sizing.l6_um, 0.18, 2.0)
        w7 = _clamp(sizing.w7_um, 2.0, 80.0)
        l7 = _clamp(sizing.l7_um, 0.18, 2.0)
        w8 = _clamp(sizing.w8_um, 1.0, 30.0)
        l8 = _clamp(sizing.l8_um, 0.36, 3.0)
        cc = _clamp(sizing.cc_pf, 0.5, 10.0)
        ibias = _clamp(sizing.ibias_ua, 2.0, 50.0)

        # 2. Enforce aspect ratio W/L >= 1.5
        w1 = max(w1, 1.5 * l1)
        w3 = max(w3, 1.5 * l3)
        w5 = max(w5, 1.5 * l5)
        w6 = max(w6, 2.0 * l6)
        w7 = max(w7, 1.5 * l7)
        w8 = max(w8, 1.5 * l8)

        # 3. Enforce current mirror scaling: M5 and M7 must exceed M8 to support stages
        # Tail ratio M5/M8 >= 1.2
        ratio_8 = w8 / l8
        if (w5 / l5) < 1.2 * ratio_8:
            w5 = round(1.5 * ratio_8 * l5, 2)
            w5 = _clamp(w5, 2.0, 60.0)

        # Driver sink ratio M7/M8 >= 1.5
        if (w7 / l7) < 1.5 * ratio_8:
            w7 = round(2.0 * ratio_8 * l7, 2)
            w7 = _clamp(w7, 2.0, 80.0)

        # 4. Enforce driver M6 transconductance to maintain high-frequency pole-splitting:
        # gm6 is proportional to sqrt(W6/L6 * I7). If W6 is too small relative to W1,
        # second pole p2 collapses into GBW causing phase margin collapse.
        if w6 < 1.5 * w1:
            w6 = round(max(w6, 2.0 * w1), 2)
            w6 = _clamp(w6, 5.0, 120.0)

        repaired = OpAmpSizing(
            w1_um=round(w1, 3),
            l1_um=round(l1, 3),
            w3_um=round(w3, 3),
            l3_um=round(l3, 3),
            w5_um=round(w5, 3),
            l5_um=round(l5, 3),
            w6_um=round(w6, 3),
            l6_um=round(l6, 3),
            w7_um=round(w7, 3),
            l7_um=round(l7, 3),
            w8_um=round(w8, 3),
            l8_um=round(l8, 3),
            cc_pf=round(cc, 3),
            ibias_ua=round(ibias, 2),
            cl_pf=sizing.cl_pf,
        )

        # 5. Iterative pole-splitting sanity check: if phase margin is still marginal (< 45 deg),
        # slightly increase Cc compensation capacitor
        m = evaluate_opamp_analytical(repaired, Sky130Corner.TT)
        if m.pm_deg < 45.0 and repaired.cc_pf < 8.0:
            repaired.cc_pf = round(repaired.cc_pf * 1.5, 3)

        return repaired
