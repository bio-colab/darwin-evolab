"""Tests for the SVG Schematic Exporter in experimental/electronics/instruments/schematic.py."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from experimental.electronics.instruments.schematic import circuit_to_svg, save_circuit_svg
from experimental.electronics.proposal import seed_for_controller
from evolab.cli import main


def test_circuit_to_svg_rendering():
    # Build a known half-adder seed
    genome = seed_for_controller(("XOR", "AND"), 2, 2)
    circuit = genome.circuit

    svg_str = circuit_to_svg(circuit, title="Test Half Adder")
    assert "<svg" in svg_str
    assert "</svg>" in svg_str
    assert "Test Half Adder" in svg_str
    assert "IN 0" in svg_str
    assert "IN 1" in svg_str
    assert "OUT 0" in svg_str
    assert "OUT 1" in svg_str
    assert "74HC86" in svg_str
    assert "74HC08" in svg_str
    assert "<path" in svg_str


def test_save_circuit_svg_file(tmp_path):
    genome = seed_for_controller(("XOR", "AND"), 2, 2)
    svg_file = tmp_path / "schematics" / "adder.svg"

    saved = save_circuit_svg(genome, svg_file, title="Saved Adder Schematic")
    assert saved.is_file()
    content = saved.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "Saved Adder Schematic" in content


def test_cli_schematic_flag(tmp_path):
    svg_out = tmp_path / "cli_schematic.svg"
    rep_out = tmp_path / "cli_rep.json"

    cmd = [
        "evolve",
        "--genome", "electronics",
        "--scenario", "half_adder",
        "-g", "1",
        "-p", "4",
        "--schematic-file", str(svg_out),
        "-o", str(rep_out),
    ]

    ret = main(cmd)
    assert ret in (0, 1)
    assert svg_out.is_file()
    content = svg_out.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "74HC" in content
