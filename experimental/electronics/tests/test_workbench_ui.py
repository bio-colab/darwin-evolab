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

