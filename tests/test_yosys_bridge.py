"""Unit test suite for Yosys RTL synthesis bridge and comparative silicon benchmarking."""
from __future__ import annotations

import pytest

from evolab.cgp_logic import CGPGenome, CGPNode, GateType
from evolab.silicon.yosys_bridge import (
    YosysCellReport,
    YosysComparisonReport,
    YosysSynthesisBridge,
)


def test_yosys_bridge_builtin_fallback():
    bridge = YosysSynthesisBridge(yosys_bin="__non_existent_yosys_executable__")
    assert bridge.is_available is False

    verilog_code = """
module my_adder(input a, input b, output s, output c);
  assign s = a ^ b;
  assign c = a & b;
endmodule
"""
    rep = bridge.run_synthesis(verilog_code)
    assert isinstance(rep, YosysCellReport)
    assert rep.total_cells >= 2
    assert rep.is_yosys_native is False
    assert "$_XOR_" in rep.cell_breakdown
    assert "$_AND_" in rep.cell_breakdown


def test_yosys_bridge_cgp_comparison():
    # Construct a Half-Adder CGPGenome
    nodes = [
        CGPNode(GateType.XOR, 0, 1),
        CGPNode(GateType.AND, 0, 1),
    ]
    cgp = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])

    bridge = YosysSynthesisBridge(yosys_bin="__mock_fallback__")
    comp = bridge.compare_cgp_with_yosys(cgp, top_module="half_adder")

    assert isinstance(comp, YosysComparisonReport)
    assert comp.cgp_gate_count == 2
    assert comp.yosys_cell_count >= 2
    assert comp.area_ratio > 0.0
    assert "OPTIMAL" in comp.efficiency_verdict or "COMPETITIVE" in comp.efficiency_verdict
    assert comp.is_synthesizable is True
    assert "tool_used" in comp.to_dict()


def test_yosys_bridge_parse_json_or_text():
    bridge = YosysSynthesisBridge()
    sample_text = """
3. Executing STAT pass (printing statistics).
=== cgp_circuit ===
   Number of wires:                 12
   Number of wire bits:             12
   Number of public wires:           4
   Number of public wire bits:       4
   Number of memories:               0
   Number of memory bits:            0
   Number of processes:              0
   Number of cells:                  5
     $_AND_                          2
     $_OR_                           1
     $_XOR_                          2
"""
    rep = bridge._parse_yosys_json_or_text(sample_text)
    assert rep.total_cells == 5
    assert rep.num_wires == 12
    assert rep.cell_breakdown["$_AND_"] == 2
    assert rep.cell_breakdown["$_OR_"] == 1
    assert rep.cell_breakdown["$_XOR_"] == 2
