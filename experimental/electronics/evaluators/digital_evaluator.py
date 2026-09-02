"""
digital_evaluator.py — Multi-Corner Physical Evaluator for 74HC Circuit Netlists.
Evaluates functional logic correctness, timing constraints, and quiescent power across
nominal and holdout PVT (Process, Voltage, Temperature) corners.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import Individual
from ..models.circuit_netlist import CircuitNetlistGenome
from ..models.logic import missing_functions
from ..models.validity import electrical_validity


@dataclass(frozen=True)
class OperatingCorner:
    """PVT operating corner definition."""

    name: str
    vcc_v: float
    temp_c: float


class MultiCorner74xxEvaluator(Evaluator):
    """Evaluates 74HC circuit netlists across nominal and holdout physical corners."""

    def __init__(
        self,
        truth_table: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
        max_delay_ns: float = 120.0,
        max_quiescent_ua: float = 100.0,
        functions_needed: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.truth_table = list(truth_table)
        self.max_delay_ns = max_delay_ns
        self.max_quiescent_ua = max_quiescent_ua
        self.functions_needed = tuple(functions_needed)

        # Standard physical corners (Nominal + Holdout worst-case corners)
        self.nominal_corner = OperatingCorner("Nominal_4.5V_25C", 4.5, 25.0)
        self.holdout_slow_corner = OperatingCorner("Holdout_4.5V_85C", 4.5, 85.0)
        self.holdout_fast_corner = OperatingCorner("Holdout_6.0V_25C", 6.0, 25.0)

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def spec_descriptor(self) -> dict:
        """Archive key material: everything that changes what a score means."""
        return {
            "family": "74xx_multicorner",
            "truth_table": [
                [list(vec), list(out) if not isinstance(out, dict) else out]
                for vec, out in self.truth_table
            ],
            "max_delay_ns": self.max_delay_ns,
            "max_quiescent_ua": self.max_quiescent_ua,
            "functions_needed": list(self.functions_needed),
        }

    def evaluate(self, target: Any, **kwargs: Any) -> FitnessResult:
        genome = target.genome if isinstance(target, Individual) else target
        if not isinstance(genome, CircuitNetlistGenome):
            return FitnessResult(score=0.0, passed_holdout=False, artifacts={"error": "Target is not CircuitNetlistGenome"})

        guard = electrical_validity(genome)
        if not guard["valid"]:
            return FitnessResult(
                score=0.0,
                passed_holdout=False,
                artifacts={"invalid": True, "violations": guard["violations"]},
            )

        circuit = genome.circuit

        # 1. Functional Logic Verification
        correct_vectors = 0
        settled_vectors = 0
        total_vectors = len(self.truth_table)

        for in_vec, expected_out in self.truth_table:
            actual_out, is_stable = circuit.simulate(in_vec)
            if is_stable:
                settled_vectors += 1
            if is_stable and tuple(actual_out) == tuple(expected_out):
                correct_vectors += 1

        functional_accuracy = correct_vectors / total_vectors
        settled_ratio = settled_vectors / total_vectors if total_vectors else 1.0

        # 2. Multi-Corner Timing and Physical Delay
        nominal_delay = circuit.compute_critical_path_delay_ns(
            self.nominal_corner.vcc_v, self.nominal_corner.temp_c, 50.0
        )
        worst_delay = circuit.compute_critical_path_delay_ns(
            self.holdout_slow_corner.vcc_v, self.holdout_slow_corner.temp_c, 50.0
        )
        fast_delay = circuit.compute_critical_path_delay_ns(
            self.holdout_fast_corner.vcc_v, self.holdout_fast_corner.temp_c, 50.0
        )
        delay_cl15 = circuit.compute_critical_path_delay_ns(
            self.nominal_corner.vcc_v, self.nominal_corner.temp_c, 15.0
        )
        delay_cl50 = nominal_delay
        specs = circuit.ic_specs
        t_cl15 = (
            sum(s.timing.get_transition_ns(self.nominal_corner.vcc_v, self.nominal_corner.temp_c, 15.0) for s in specs)
            / len(specs)
            if specs
            else 0.0
        )
        t_cl50 = (
            sum(s.timing.get_transition_ns(self.nominal_corner.vcc_v, self.nominal_corner.temp_c, 50.0) for s in specs)
            / len(specs)
            if specs
            else 0.0
        )

        timing_score = max(0.0, 1.0 - (worst_delay / (self.max_delay_ns * 1.5)))

        # 3. Quiescent Current Budget
        total_icc = circuit.compute_quiescent_current_ua()
        power_score = max(0.0, 1.0 - (total_icc / (self.max_quiescent_ua * 1.5)))

        # Composite Fitness (70% functional truth table, 20% timing, 10% quiescent power)
        composite_score = 0.70 * functional_accuracy + 0.20 * timing_score + 0.10 * power_score

        # Success criteria: 100% truth table + delay within datasheet limits
        passed = (functional_accuracy == 1.0) and (worst_delay <= self.max_delay_ns) and (total_icc <= self.max_quiescent_ua)

        return FitnessResult(
            score=round(composite_score * 100.0, 4),
            passed_holdout=passed,
            sub_scores={
                "functional_accuracy": round(functional_accuracy * 100.0, 2),
                "timing_score": round(timing_score * 100.0, 2),
                "power_score": round(power_score * 100.0, 2),
            },
            artifacts={
                "functional_accuracy": functional_accuracy,
                "nominal_delay_ns": nominal_delay,
                "worst_delay_ns": worst_delay,
                "fast_delay_ns": fast_delay,
                "delay_ns_cl15": delay_cl15,
                "delay_ns_cl50": delay_cl50,
                "transition_ns_cl15": round(t_cl15, 2),
                "transition_ns_cl50": round(t_cl50, 2),
                "quiescent_icc_ua": total_icc,
                "ic_count": len(genome.ic_packages),
                "wire_count": len(genome.connections),
                "settled_ratio": settled_ratio,
                "functions_used": circuit.functions_used(),
                "functions_needed": list(self.functions_needed or getattr(genome, "functions_needed", ())),
                "functions_missing": missing_functions(
                    circuit.functions_used(),
                    self.functions_needed or getattr(genome, "functions_needed", ()),
                ),
            },
        )
