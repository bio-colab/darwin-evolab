"""
exp1_datasheet_74xx_synthesis.py — Laboratory Experiment 1: 74xx Datasheet-Native Synthesis.
Synthesizes verified arithmetic/logic subsystems from real 74HC DIP packages under
multi-corner physical operating conditions.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..evaluators.digital_evaluator import MultiCorner74xxEvaluator
from ..models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
from ..models.independent_verifier import IndependentDigitalVerifier


def run_half_adder_synthesis_lab() -> dict[str, Any]:
    """Verify a known 74HC half-adder netlist. Search uses scenarios.half_adder instead.

    Ground truth target:
      Inputs: A (pin -1, 0), B (pin -1, 1)
      Outputs: Sum (pin -1, 100), Cout (pin -1, 101)
      Truth table:
        (0, 0) -> (0, 0)
        (0, 1) -> (1, 0)
        (1, 0) -> (1, 0)
        (1, 1) -> (0, 1)
    """
    t0 = time.perf_counter()

    # Ground truth truth table verified with independent analytical formula
    truth_table = []
    for a in (0, 1):
        for b in (0, 1):
            expected = IndependentDigitalVerifier.half_adder_reference(a, b)
            truth_table.append(((a, b), expected))

    evaluator = MultiCorner74xxEvaluator(truth_table=truth_table, max_delay_ns=150.0, max_quiescent_ua=60.0)

    # Human-engineered / target-evolved netlist using 74HC86 (XOR) and 74HC08 (AND):
    # IC 0: 74HC86 (XOR for Sum)
    #   Pin 1 (in1) <- Input A (pin -1, 0)
    #   Pin 2 (in2) <- Input B (pin -1, 1)
    #   Pin 3 (out) -> Primary Output Sum (pin -1, 100)
    # IC 1: 74HC08 (AND for Cout)
    #   Pin 1 (in1) <- Input A (pin -1, 0)
    #   Pin 2 (in2) <- Input B (pin -1, 1)
    #   Pin 3 (out) -> Primary Output Cout (pin -1, 101)
    connections = [
        Connection(PinRef(-1, 0), PinRef(0, 1)),   # Input A -> 74HC86 Pin 1
        Connection(PinRef(-1, 1), PinRef(0, 2)),   # Input B -> 74HC86 Pin 2
        Connection(PinRef(0, 3), PinRef(-1, 100)), # 74HC86 Pin 3 -> Sum Out

        Connection(PinRef(-1, 0), PinRef(1, 1)),   # Input A -> 74HC08 Pin 1
        Connection(PinRef(-1, 1), PinRef(1, 2)),   # Input B -> 74HC08 Pin 2
        Connection(PinRef(1, 3), PinRef(-1, 101)), # 74HC08 Pin 3 -> Cout Out
    ]

    candidate_genome = CircuitNetlistGenome(
        ic_packages=["74HC86", "74HC08"],
        connections=connections,
        num_inputs=2,
        num_outputs=2,
    )

    eval_result = evaluator.evaluate(candidate_genome)
    from ..models.equivalence import encode_half_adder_ref, verify_equivalent

    equiv = verify_equivalent(
        candidate_genome,
        lambda a, b: IndependentDigitalVerifier.half_adder_reference(a, b),
        encode_ref=encode_half_adder_ref,
    )
    dur_ms = (time.perf_counter() - t0) * 1000.0

    report = {
        "experiment": "Exp1_Datasheet_74xx_Half_Adder_Synthesis",
        "timestamp_ms": round(dur_ms, 2),
        "passed": eval_result.passed_holdout,
        "fitness_score": eval_result.score,
        "sub_scores": eval_result.sub_scores,
        "ic_packages_used": list(candidate_genome.ic_packages),
        "wire_count": len(candidate_genome.connections),
        "metrics": eval_result.artifacts,
        "verification_layers": {
            "layer_1_datasheet_specs": "TI SCLS099E (74HC86), TI SCLS089D (74HC08)",
            "layer_2_independent_formula": "Verified via IndependentDigitalVerifier.half_adder_reference",
            "layer_3_circuit_model": "BreadboardCircuit multi-IC gate propagation simulation",
            "layer_4_holdout_corners": "Nominal 5V/25C, Slow 4.5V/85C, Fast 5.5V/-40C",
            "layer_5_formal": equiv,
        },
    }
    return report


if __name__ == "__main__":
    rep = run_half_adder_synthesis_lab()
    print(json.dumps(rep, indent=2))
