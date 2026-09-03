"""Tests for Analog Circuit Topology Synthesis in experimental/electronics/models/analog_topology.py."""
from __future__ import annotations

import json
from pathlib import Path
import random
import pytest

from experimental.electronics.models.analog_topology import (
    AnalogComponent,
    AnalogComponentKind,
    AnalogTopologyGenome,
)
from experimental.electronics.evaluators.analog_topology_evaluator import AnalogFilterTopologyEvaluator
from experimental.electronics.scenarios import prepare_electronics_run
from evolab.genome import Individual
from evolab.cli import main


def test_analog_component_and_netlist_generation():
    r1 = AnalogComponent(kind=AnalogComponentKind.RESISTOR, name="R1", nodes=("in", "out"), value=1000.0)
    c1 = AnalogComponent(kind=AnalogComponentKind.CAPACITOR, name="C1", nodes=("out", "0"), value=1e-7)
    q1 = AnalogComponent(kind=AnalogComponentKind.BJT, name="Q1", nodes=("c", "b", "e"), value=0.0)

    assert r1.to_spice_line() == "R1 in out 1000"
    assert "1.000000e-07" in c1.to_spice_line()
    assert "Q1 c b e NPN_GENERIC" == q1.to_spice_line()

    genome = AnalogTopologyGenome([r1, c1, q1])
    netlist = genome.to_spice_netlist("Filter")
    assert "* Filter" in netlist
    assert "R1 in out 1000" in netlist
    assert ".ac dec" in netlist


def test_analog_topology_mutations():
    rng = random.Random(42)
    r1 = AnalogComponent(kind=AnalogComponentKind.RESISTOR, name="R1", nodes=("in", "out"), value=1000.0)
    c1 = AnalogComponent(kind=AnalogComponentKind.CAPACITOR, name="C1", nodes=("out", "0"), value=1e-7)
    genome = AnalogTopologyGenome([r1, c1])

    # Test mutation
    mutated = genome.mutate(rng)
    assert isinstance(mutated, AnalogTopologyGenome)
    assert mutated.fingerprint() != genome.fingerprint() or len(mutated.components) != len(genome.components)


def test_analog_filter_evaluator():
    evaluator = AnalogFilterTopologyEvaluator(target_cutoff_hz=1000.0)
    r1 = AnalogComponent(kind=AnalogComponentKind.RESISTOR, name="R1", nodes=("in", "out"), value=10000.0)
    c1 = AnalogComponent(kind=AnalogComponentKind.CAPACITOR, name="C1", nodes=("out", "0"), value=1.59e-8)
    genome = AnalogTopologyGenome([r1, c1])
    ind = Individual(genome=genome, species="spec_electronics")

    res = evaluator.evaluate(ind)
    assert res.score > 70.0  # Close to 1000 Hz
    assert "estimated_fc_hz" in res.artifacts or "gain_db" in res.artifacts


def test_cli_analog_filter_synthesis(tmp_path):
    report_file = tmp_path / "filter_report.json"
    cmd = [
        "evolve",
        "--genome", "electronics",
        "--scenario", "analog_filter_synthesis",
        "-g", "2",
        "-p", "4",
        "--seed", "42",
        "-o", str(report_file),
    ]

    ret = main(cmd)
    assert ret in (0, 1)
    assert report_file.is_file()
    result = json.loads(report_file.read_text(encoding="utf-8"))
    assert result["config"]["scenario"] == "analog_filter_synthesis"
    assert result["total_generations"] == 2
