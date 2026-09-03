"""
Comprehensive tests for advanced electronics inputs:
- Safe Boolean Expression Parser
- Behavioral Verilog RTL Reader
- Analog Engineering Specs and Waveform Traces
- Pareto Objective Matrix
- CLI End-to-End integration with --expr, --verilog-in, and --waveform
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from experimental.electronics.inputs.boolean_expr import BooleanExpressionParser, parse_boolean_spec
from experimental.electronics.inputs.verilog_reader import VerilogRTLReader, parse_verilog_spec
from experimental.electronics.inputs.analog_spec import FilterSpec, AmplifierSpec, WaveformTraceSpec
from experimental.electronics.inputs.objective_matrix import ObjectiveMatrix
from evolab.cli import main


# --------------------------------------------------------------------------- #
# 1. Boolean Expression Parser Tests
# --------------------------------------------------------------------------- #

def test_boolean_expression_parser_single_and_multi():
    # Test half-adder equations
    raw = "Sum = A ^ B; Carry = A & B"
    res = parse_boolean_spec(raw)

    assert res.inputs == ("A", "B")
    assert res.outputs == ("Sum", "Carry")
    assert res.num_inputs == 2
    assert res.num_outputs == 2
    assert len(res.truth_table) == 4

    # Truth table lookup: (A, B) -> (Sum, Carry)
    # (0, 0) -> (0, 0)
    # (0, 1) -> (1, 0)
    # (1, 0) -> (1, 0)
    # (1, 1) -> (0, 1)
    tt = dict(res.truth_table)
    assert tt[(0, 0)] == (0, 0)
    assert tt[(0, 1)] == (1, 0)
    assert tt[(1, 0)] == (1, 0)
    assert tt[(1, 1)] == (0, 1)


def test_boolean_expression_word_operators_and_safety():
    raw = "out = (A AND B) OR (NOT C)"
    res = parse_boolean_spec(raw)
    assert res.inputs == ("A", "B", "C")
    assert len(res.truth_table) == 8

    # Ensure dangerous expressions are rejected
    with pytest.raises(ValueError):
        parse_boolean_spec("out = __import__('os').system('dir')")


# --------------------------------------------------------------------------- #
# 2. Verilog RTL Reader Tests
# --------------------------------------------------------------------------- #

def test_verilog_rtl_reader_full_adder(tmp_path):
    v_code = """
    module full_adder (
        input wire a,
        input wire b,
        input wire cin,
        output wire sum,
        output wire cout
    );
        assign sum = a ^ b ^ cin;
        assign cout = (a & b) | (cin & (a ^ b));
    endmodule
    """
    v_file = tmp_path / "full_adder.v"
    v_file.write_text(v_code, encoding="utf-8")

    spec = parse_verilog_spec(v_file)
    assert spec.module_name == "full_adder"
    assert "a" in spec.inputs and "b" in spec.inputs and "cin" in spec.inputs
    assert "sum" in spec.outputs and "cout" in spec.outputs
    assert len(spec.parse_result.truth_table) == 8


def test_verilog_rtl_reader_ternary_multiplexer(tmp_path):
    mux_code = """
    module mux2to1 (
        input wire sel,
        input wire d0,
        input wire d1,
        output wire y
    );
        assign y = sel ? d1 : d0;
    endmodule
    """
    mux_file = tmp_path / "mux2to1.v"
    mux_file.write_text(mux_code, encoding="utf-8")

    spec = parse_verilog_spec(mux_file)
    assert spec.module_name == "mux2to1"
    # When sel=0 -> y=d0; when sel=1 -> y=d1
    tt = dict(spec.parse_result.truth_table)
    inputs_order = list(spec.parse_result.inputs)
    sel_idx = inputs_order.index("sel")
    d0_idx = inputs_order.index("d0")
    d1_idx = inputs_order.index("d1")

    # Check vector: sel=0, d0=1, d1=0 -> y=1
    v1 = [0, 0, 0]
    v1[sel_idx], v1[d0_idx], v1[d1_idx] = 0, 1, 0
    assert tt[tuple(v1)][0] == 1

    # Check vector: sel=1, d0=1, d1=0 -> y=0
    v2 = [0, 0, 0]
    v2[sel_idx], v2[d0_idx], v2[d1_idx] = 1, 1, 0
    assert tt[tuple(v2)][0] == 0


# --------------------------------------------------------------------------- #
# 3. Analog Engineering Specs & Waveforms Tests
# --------------------------------------------------------------------------- #

def test_analog_filter_and_amp_specs():
    f_spec = FilterSpec(cutoff_hz=1000.0, stopband_attenuation_db=-30.0)
    score_pass = f_spec.evaluate_response(gain_db=0.0, measured_freq_hz=500.0)
    assert score_pass == 100.0

    amp_spec = AmplifierSpec(target_gain_db=20.0, min_bandwidth_mhz=1.0)
    score_amp = amp_spec.evaluate_metrics(gain_db=20.0, bandwidth_mhz=1.5, phase_margin_deg=65.0, power_mw=5.0)
    assert score_amp == 100.0


def test_waveform_trace_spec_from_csv(tmp_path):
    csv_file = tmp_path / "osc_wave.csv"
    csv_file.write_text("0.0, 0.0\n0.001, 2.5\n0.002, 4.8\n0.003, 5.0\n", encoding="utf-8")

    wf = WaveformTraceSpec.from_csv(csv_file)
    assert len(wf.times) == 4
    assert wf.voltages[-1] == 5.0

    # Test MSE match with identical values
    score_exact = wf.evaluate_mse(wf.times, wf.voltages)
    assert score_exact == 100.0


# --------------------------------------------------------------------------- #
# 4. Objective Matrix Tests
# --------------------------------------------------------------------------- #

def test_objective_matrix_presets():
    p_power = ObjectiveMatrix.from_preset("power")
    p_speed = ObjectiveMatrix.from_preset("speed")
    assert p_power.power_weight > p_power.delay_weight
    assert p_speed.delay_weight > p_speed.power_weight

    score = p_power.calculate_score(is_functional=True, accuracy=1.0, critical_delay_fo4=2.0, active_gates=4, toggle_ratio=0.2)
    assert score > 70.0


# --------------------------------------------------------------------------- #
# 5. CLI End-to-End Integration Tests
# --------------------------------------------------------------------------- #

def test_cli_direct_boolean_expr_synthesis(tmp_path):
    rep = tmp_path / "expr_rep.json"
    svg = tmp_path / "expr_circuit.svg"
    cmd = [
        "evolve",
        "--expr", "Sum = A ^ B; Cout = A & B",
        "-g", "2",
        "-p", "4",
        "--schematic-file", str(svg),
        "-o", str(rep),
    ]
    ret = main(cmd)
    assert ret in (0, 1)
    assert rep.is_file()
    assert svg.is_file()
    data = json.loads(rep.read_text(encoding="utf-8"))
    assert data["config"]["genome"] == "electronics"


def test_cli_verilog_in_synthesis(tmp_path):
    v_in = tmp_path / "simple_gate.v"
    v_in.write_text("module simple_gate(input wire a, input wire b, output wire y); assign y = a & b; endmodule\n", encoding="utf-8")
    v_out = tmp_path / "synth_out.v"
    rep = tmp_path / "v_rep.json"

    cmd = [
        "evolve",
        "--verilog-in", str(v_in),
        "--objective", "power",
        "-g", "2",
        "-p", "4",
        "--verilog-file", str(v_out),
        "-o", str(rep),
    ]
    ret = main(cmd)
    assert ret in (0, 1)
    assert rep.is_file()
    assert v_out.is_file()
    v_text = v_out.read_text(encoding="utf-8")
    assert "module synthesized_circuit" in v_text


def test_cli_waveform_matching(tmp_path):
    csv_file = tmp_path / "target_step.csv"
    csv_file.write_text("0.0, 0.0\n0.001, 1.0\n0.002, 2.0\n0.003, 3.0\n", encoding="utf-8")
    rep = tmp_path / "wf_rep.json"

    cmd = [
        "evolve",
        "--waveform", str(csv_file),
        "-g", "2",
        "-p", "4",
        "-o", str(rep),
    ]
    ret = main(cmd)
    assert ret in (0, 1)
    assert rep.is_file()
