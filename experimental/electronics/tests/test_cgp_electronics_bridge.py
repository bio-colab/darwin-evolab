"""Tests for Cartesian Genetic Programming (CGP) integration in the electronics track."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from experimental.electronics.scenarios import prepare_electronics_run, list_electronics_scenarios
from evolab.cgp_logic import CGPGenome
from evolab.cli import main


def test_cgp_scenarios_registered():
    scenarios = list_electronics_scenarios()
    assert "cgp_adder" in scenarios
    assert "cgp_alu" in scenarios
    assert "cgp_comparator" in scenarios


def test_cgp_scenario_preparation():
    ev_adder, pop_adder, name_adder = prepare_electronics_run("cgp_adder", 4, seed=42)
    assert name_adder == "cgp_adder"
    assert len(pop_adder) == 4
    assert isinstance(pop_adder[0].genome, CGPGenome)
    assert pop_adder[0].genome.num_inputs == 3
    assert pop_adder[0].genome.num_outputs == 2

    ev_alu, pop_alu, name_alu = prepare_electronics_run("cgp_alu", 4, seed=42)
    assert name_alu == "cgp_alu"
    assert len(pop_alu) == 4
    assert isinstance(pop_alu[0].genome, CGPGenome)

    ev_comp, pop_comp, name_comp = prepare_electronics_run("cgp_comparator", 4, seed=42)
    assert name_comp == "cgp_comparator"
    assert len(pop_comp) == 4
    assert isinstance(pop_comp[0].genome, CGPGenome)
    assert pop_comp[0].genome.num_inputs == 4
    assert pop_comp[0].genome.num_outputs == 3


def test_cli_cgp_adder_and_verilog_export(tmp_path):
    verilog_out = tmp_path / "adder_synth.v"
    report_out = tmp_path / "cgp_report.json"

    cmd = [
        "evolve",
        "--genome", "electronics",
        "--scenario", "cgp_adder",
        "-g", "2",
        "-p", "4",
        "--seed", "42",
        "--verilog-file", str(verilog_out),
        "-o", str(report_out),
    ]

    ret = main(cmd)
    assert ret in (0, 1)
    assert report_out.is_file()
    assert verilog_out.is_file()

    v_content = verilog_out.read_text(encoding="utf-8")
    assert "module synthesized_circuit" in v_content
    assert "endmodule" in v_content
