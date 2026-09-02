"""
Tests for MultiCorner74xxEvaluator and AnalogSizingEvaluator.
"""
from experimental.electronics.evaluators.digital_evaluator import MultiCorner74xxEvaluator
from experimental.electronics.evaluators.spice_evaluator import AnalogSizingEvaluator
from experimental.electronics.models.circuit_netlist import (
    CircuitNetlistGenome,
    Connection,
    PinRef,
)
from evolab.genome import FloatGenome


def test_multi_corner_evaluator_half_adder():
    truth_table = [
        ((0, 0), (0, 0)),
        ((0, 1), (1, 0)),
        ((1, 0), (1, 0)),
        ((1, 1), (0, 1)),
    ]
    evaluator = MultiCorner74xxEvaluator(truth_table=truth_table, max_delay_ns=150.0, functions_needed=("XOR", "AND"))

    # 74HC86 (XOR) + 74HC08 (AND)
    conns = [
        Connection(PinRef(-1, 0), PinRef(0, 1)),
        Connection(PinRef(-1, 1), PinRef(0, 2)),
        Connection(PinRef(0, 3), PinRef(-1, 100)),
        Connection(PinRef(-1, 0), PinRef(1, 1)),
        Connection(PinRef(-1, 1), PinRef(1, 2)),
        Connection(PinRef(1, 3), PinRef(-1, 101)),
    ]
    genome = CircuitNetlistGenome(["74HC86", "74HC08"], conns, 2, 2)
    res = evaluator.evaluate(genome)

    assert res.passed_holdout is True
    assert res.score > 80.0
    assert res.artifacts["functional_accuracy"] == 1.0
    assert "XOR" in res.artifacts["functions_used"]
    assert "AND" in res.artifacts["functions_used"]
    assert res.artifacts["functions_missing"] == []


def test_analog_sizing_evaluator():
    evaluator = AnalogSizingEvaluator()
    genome = FloatGenome([2.5, 0.35, 1.2, 0.35])
    res = evaluator.evaluate(genome)

    assert res.score > 85.0
    assert "gain_db" in res.artifacts
    assert res.artifacts.get("tool_used")
    assert "holdout_gain_db" in res.artifacts
    assert isinstance(res.passed_holdout, bool)


def test_oracle_blocks_physical_claim_on_fallback():
    from experimental.electronics.models.ngspice_bridge import SpiceSimulationResult
    from experimental.electronics.oracle import ElectronicsOracle

    fb = SpiceSimulationResult(True, 42.0, 10.0, 60.0, 2.0, tool_used="analytical_fallback")
    out = ElectronicsOracle().merge(fb, fb)
    assert out.tool_used == "analytical_fallback"
    assert out.physical_claim is False
    assert out.agreement is False
    assert "physical_claim_blocked_no_spice" in out.notes
