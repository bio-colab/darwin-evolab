"""
Unit and integration tests for the interactive Silicon Workbench UI/UX generator.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from experimental.electronics.ui.workbench_generator import generate_workbench_html, save_workbench_html
from evolab.cgp_logic import create_random_cgp_genome
from evolab.cli import main
import random


def test_generate_workbench_html_cgp():
    rng = random.Random(42)
    genome = create_random_cgp_genome(num_inputs=2, num_outputs=2, num_nodes=8, rng=rng)
    meta = {
        "scenario": "test_cgp_logic",
        "fitness": 98.5,
        "generations": 5,
        "candidates": 20,
    }
    html = generate_workbench_html(genome, metadata=meta, title="Test Silicon Workbench")

    assert "<!DOCTYPE html>" in html
    assert "Live Circuit Simulator" in html
    assert "Dual-Channel Oscilloscope" in html
    assert "Verilog-2001 RTL" in html
    assert "SPICE Netlist" in html
    assert "WebUSB / FPGA Programmer" in html
    assert "Test Silicon Workbench" in html


def test_save_workbench_html_file(tmp_path):
    rng = random.Random(42)
    genome = create_random_cgp_genome(num_inputs=3, num_outputs=2, num_nodes=10, rng=rng)
    out_file = tmp_path / "workbench.html"

    p = save_workbench_html(genome, out_file, metadata={"scenario": "adder_3in"})
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert len(text) > 5000
    assert "evaluateLogic" in text
    assert "renderScope" in text


def test_cli_ui_file_flag(tmp_path):
    html_file = tmp_path / "cli_workbench.html"
    rep_file = tmp_path / "rep.json"
    cmd = [
        "evolve",
        "--expr", "Sum = A ^ B; Cout = A & B",
        "-g", "2",
        "-p", "4",
        "--ui-file", str(html_file),
        "-o", str(rep_file),
    ]
    ret = main(cmd)
    assert ret in (0, 1)
    assert html_file.is_file()
    assert rep_file.is_file()
    content = html_file.read_text(encoding="utf-8")
    assert "Live Circuit Simulator" in content
    assert "Dual-Channel Oscilloscope" in content


def test_workbench_webusb_and_fpga_integration():
    """Verifies that WebUSB flasher profiles, loopback mock engine, and FPGA specs are properly embedded."""
    rng = random.Random(42)
    genome = create_random_cgp_genome(num_inputs=2, num_outputs=2, num_nodes=6, rng=rng)
    meta = {
        "scenario": "full_adder",
        "fitness": 99.2,
        "fpga_target": "artix7_35t",
    }
    html = generate_workbench_html(genome, metadata=meta, title="Artix-7 WebUSB Workbench")

    # 1. Verify FPGA specs embedded in datasheet
    assert "Artix-7" in html
    assert "AMD/Xilinx" in html
    assert "Fmax" in html
    assert "Dynamic Power" in html

    # 2. Verify WebUSB programmer controls & terminal
    assert "usb-terminal" in html
    assert "usb-progress-bar" in html
    assert "btn-usb-connect" in html
    assert "btn-usb-flash" in html
    assert "btn-usb-loopback" in html

    # 3. Verify WebUSB JS profiles
    assert "FTDI FT2232H" in html
    assert "TinyFPGA BX" in html
    assert "Raspberry Pi Pico" in html
    assert "navigator.usb.requestDevice" in html
    assert "runHardwareLoopback" in html


def test_workbench_analog_opamp_mode():
    """Verifies that an OpAmpSizing circuit automatically enables analog workbench mode with all interactive controls."""
    from evolab.silicon.opamp_benchmark import OpAmpSizing

    sizing = OpAmpSizing(
        w1_um=15.0,
        l1_um=0.36,
        w3_um=30.0,
        l3_um=0.36,
        w5_um=20.0,
        l5_um=0.36,
        w6_um=60.0,
        l6_um=0.36,
        w7_um=30.0,
        l7_um=0.36,
        w8_um=8.0,
        l8_um=0.72,
        cc_pf=3.5,
        ibias_ua=15.0,
    )
    meta = {
        "scenario": "two_stage_opamp_sky130",
        "tech_node": "skywater130",
        "fitness": 96.4,
    }
    html = generate_workbench_html(sizing, metadata=meta, title="Sky130 Analog OpAmp Studio")

    # 1. Verify Mode is set to analog by default
    assert 'data-initial-mode="analog"' in html
    assert "btn-mode-analog" in html
    assert "btn-mode-digital" in html

    # 2. Verify Analog sections
    assert "Live AC Bode Plot (Frequency Response)" in html
    assert "Sky130 Transistor Sizing Lab" in html
    assert "Non-Dominated Pareto Frontier (Gain vs Power)" in html
    assert "CircuitGenome Modular Blocks" in html

    # 3. Verify Interactive Sizing Sliders
    assert 'id="slider-w1"' in html
    assert 'id="slider-l1"' in html
    assert 'id="slider-w6"' in html
    assert 'id="slider-cc"' in html
    assert 'id="slider-ibias"' in html
    assert 'id="btn-physics-repair"' in html

    # 4. Verify PVT Corner Switchers
    assert 'id="btn-corner-tt"' in html
    assert 'id="btn-corner-ss"' in html
    assert 'id="btn-corner-ff"' in html

    # 5. Verify In-Browser JavaScript Physics Engine
    assert "computeTransistorOP" in html
    assert "renderBodePlot" in html
    assert "renderPareto" in html
    assert "updateAnalogUI" in html


def test_workbench_modular_circuit_mode():
    """Verifies that a ModularOpAmpCircuit object generates valid dual-mode HTML with modular blocks."""
    from evolab.silicon.modular_circuit import ModularOpAmpCircuit

    circuit = ModularOpAmpCircuit()
    html = generate_workbench_html(circuit, metadata={"scenario": "modular_opamp"})

    assert 'data-initial-mode="analog"' in html
    assert "Differential Input Pair" in html
    assert "Active Current Mirror Load" in html
    assert "Common-Source Driver Stage" in html
    assert "Miller Compensation Network" in html
    assert "Sol-A" in html
    assert "Sol-E" in html


