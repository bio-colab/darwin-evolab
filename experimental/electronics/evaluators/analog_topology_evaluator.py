"""
analog_topology_evaluator.py — Evaluator for evolving analog filter networks.

Evaluates an AnalogTopologyGenome against target AC frequency response specifications.
Uses NGSpiceBridge if available, or analytical RC ladder estimation with honest fallback tagging.
"""
from __future__ import annotations

import math
from typing import Any

from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import Individual
from ..models.analog_topology import AnalogComponentKind, AnalogTopologyGenome
from ..models.ngspice_bridge import NGSpiceBridge


class AnalogFilterTopologyEvaluator(Evaluator):
    """Evaluates analog topology genomes for target filter characteristics."""

    def __init__(
        self,
        target_kind: str = "lowpass",
        target_cutoff_hz: float = 1000.0,
        passband_gain_target_db: float = 0.0,
        stopband_attenuation_db: float = -20.0,
    ) -> None:
        self.target_kind = target_kind
        self.target_cutoff_hz = target_cutoff_hz
        self.passband_gain_target_db = passband_gain_target_db
        self.stopband_attenuation_db = stopband_attenuation_db
        self.bridge = NGSpiceBridge()

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, individual: Individual) -> FitnessResult:
        genome: AnalogTopologyGenome = individual.genome
        if not isinstance(genome, AnalogTopologyGenome):
            return FitnessResult(score=0.0)

        # 1. Structural validity check
        nodes = genome.get_all_nodes()
        if genome.input_node not in nodes or genome.output_node not in nodes:
            return FitnessResult(score=0.0, artifacts={"error": "disconnected_ports"})

        # 2. Check if ngspice is available
        if self.bridge.is_ngspice_available():
            netlist = genome.to_spice_netlist("Filter Evaluation")
            res = self.bridge.run_netlist(netlist)
            if res.success and res.gain_db:
                # Score based on measured frequency response
                score = max(0.0, min(100.0, 50.0 + res.gain_db))
                return FitnessResult(
                    score=score,
                    passed_holdout=True,
                    artifacts={
                        "gain_db": res.gain_db,
                        "tool_used": "ngspice",
                        "parts_count": len(genome.components),
                    },
                )

        # 3. Analytical estimation fallback (Honest fallback: passed_holdout=False per Oracle rules)
        # Find R and C in the circuit
        r_sum = sum(c.value for c in genome.components if c.kind == AnalogComponentKind.RESISTOR)
        c_sum = sum(c.value for c in genome.components if c.kind == AnalogComponentKind.CAPACITOR)

        if r_sum <= 0 or c_sum <= 0:
            return FitnessResult(score=10.0, passed_holdout=False, artifacts={"tool_used": "analytical_fallback", "note": "missing_R_or_C"})

        # Estimate cutoff: fc = 1 / (2 * pi * R * C)
        est_fc = 1.0 / (2.0 * math.pi * r_sum * c_sum)
        # Score based on proximity to target cutoff in log scale
        error_decades = abs(math.log10(max(est_fc, 1e-3)) - math.log10(self.target_cutoff_hz))
        score = max(0.0, 100.0 - error_decades * 35.0)

        return FitnessResult(
            score=round(score, 2),
            passed_holdout=False,  # Fallback never claims physical verification
            artifacts={
                "tool_used": "analytical_fallback",
                "estimated_fc_hz": round(est_fc, 1),
                "target_fc_hz": self.target_cutoff_hz,
                "parts_count": len(genome.components),
            },
        )
