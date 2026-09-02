"""Measure what the electronics track can actually do on realistic tasks."""
from __future__ import annotations

import json
from typing import Any

from evolab.engine import EvolutionEngine
from evolab.genome import Individual

from experimental.electronics.evaluators.digital_evaluator import MultiCorner74xxEvaluator
from experimental.electronics.evaluators.spice_evaluator import AnalogSizingEvaluator
from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
from experimental.electronics.models.equivalence import (
    encode_full_adder_ref,
    encode_half_adder_ref,
    verify_equivalent,
)
from experimental.electronics.models.independent_verifier import IndependentDigitalVerifier
from experimental.electronics.scenarios import prepare_electronics_run


def _ha_table():
    return [
        ((a, b), IndependentDigitalVerifier.half_adder_reference(a, b))
        for a in (0, 1)
        for b in (0, 1)
    ]


def _fa_table():
    return [
        ((a, b, c), IndependentDigitalVerifier.full_adder_reference(a, b, c))
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    ]


def known_half_adder() -> CircuitNetlistGenome:
    return CircuitNetlistGenome(
        ["74HC86", "74HC08"],
        [
            Connection(PinRef(-1, 0), PinRef(0, 1)),
            Connection(PinRef(-1, 1), PinRef(0, 2)),
            Connection(PinRef(0, 3), PinRef(-1, 100)),
            Connection(PinRef(-1, 0), PinRef(1, 1)),
            Connection(PinRef(-1, 1), PinRef(1, 2)),
            Connection(PinRef(1, 3), PinRef(-1, 101)),
        ],
        2,
        2,
        functions_needed=("XOR", "AND"),
    )


def known_full_adder() -> CircuitNetlistGenome:
    # IC0: A XOR B; IC1: (A^B) XOR Cin = Sum
    # IC2: A AND B; IC3: (A^B) AND Cin; IC4: OR -> Cout
    return CircuitNetlistGenome(
        ["74HC86", "74HC86", "74HC08", "74HC08", "74HC32"],
        [
            Connection(PinRef(-1, 0), PinRef(0, 1)),
            Connection(PinRef(-1, 1), PinRef(0, 2)),
            Connection(PinRef(0, 3), PinRef(1, 1)),
            Connection(PinRef(-1, 2), PinRef(1, 2)),
            Connection(PinRef(1, 3), PinRef(-1, 100)),
            Connection(PinRef(-1, 0), PinRef(2, 1)),
            Connection(PinRef(-1, 1), PinRef(2, 2)),
            Connection(PinRef(0, 3), PinRef(3, 1)),
            Connection(PinRef(-1, 2), PinRef(3, 2)),
            Connection(PinRef(2, 3), PinRef(4, 1)),
            Connection(PinRef(3, 3), PinRef(4, 2)),
            Connection(PinRef(4, 3), PinRef(-1, 101)),
        ],
        3,
        2,
        functions_needed=("XOR", "AND", "OR"),
    )


def run_probe() -> dict[str, Any]:
    ha_ev = MultiCorner74xxEvaluator(_ha_table(), max_delay_ns=150.0, max_quiescent_ua=60.0, functions_needed=("XOR", "AND"))
    fa_ev = MultiCorner74xxEvaluator(_fa_table(), max_delay_ns=200.0, max_quiescent_ua=120.0, functions_needed=("XOR", "AND", "OR"))

    ha = known_half_adder()
    ha_res = ha_ev.evaluate(ha)
    ha_eq = verify_equivalent(ha, IndependentDigitalVerifier.half_adder_reference, encode_ref=encode_half_adder_ref)

    fa = known_full_adder()
    fa_res = fa_ev.evaluate(fa)
    fa_eq = verify_equivalent(fa, IndependentDigitalVerifier.full_adder_reference, encode_ref=encode_full_adder_ref)

    search_ev, search_pop, _ = prepare_electronics_run("half_adder", 12, seed=7)
    raw_scores = [float(search_ev.evaluate(ind).score) for ind in search_pop]
    engine = EvolutionEngine(
        fitness_fn=search_ev,
        population_size=12,
        seed=7,
        early_stop_fitness=99.0,
        stagnation_patience=6,
        sharing_mode="off",
    )
    report = engine.run(8, initial_population=search_pop)
    best = report["best_individual"]

    analog_ev, analog_pop, _ = prepare_electronics_run("analog_sizing", 8, seed=3)
    analog_scores = [analog_ev.evaluate(ind) for ind in analog_pop]
    analog_best = max(analog_scores, key=lambda r: r.score)

    bjt_ev, bjt_pop, _ = prepare_electronics_run("bjt_ce_amp", 4, seed=1)
    bjt_res = bjt_ev.evaluate(bjt_pop[0])

    return {
        "known_half_adder": {
            "score": ha_res.score,
            "passed_holdout": ha_res.passed_holdout,
            "functions_used": ha_res.artifacts.get("functions_used"),
            "z3_equivalent": ha_eq.get("equivalent"),
            "z3_method": ha_eq.get("method"),
        },
        "known_full_adder": {
            "score": fa_res.score,
            "passed_holdout": fa_res.passed_holdout,
            "functional_accuracy": fa_res.artifacts.get("functional_accuracy"),
            "functions_used": fa_res.artifacts.get("functions_used"),
            "z3_equivalent": fa_eq.get("equivalent"),
            "z3_method": fa_eq.get("method"),
            "z3_cex": fa_eq.get("counterexample"),
        },
        "search_half_adder_from_random": {
            "init_best": max(raw_scores),
            "init_mean": round(sum(raw_scores) / len(raw_scores), 2),
            "after_8_gens_best": best.get("fitness"),
            "found_reference": best.get("fitness", 0) >= 80,
        },
        "analog_sizing_fallback": {
            "best_score": analog_best.score,
            "tool_used": analog_best.artifacts.get("tool_used"),
            "passed_holdout": analog_best.passed_holdout,
            "holdout_gain_db": analog_best.artifacts.get("holdout_gain_db"),
        },
        "bjt_circuit_config": {
            "score": bjt_res.score,
            "tool_used": bjt_res.artifacts.get("tool_used"),
            "passed_holdout": bjt_res.passed_holdout,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), indent=2, default=str))
