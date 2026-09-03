"""Tests for free custom electronics CLI inputs (--spec and --netlist)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from experimental.electronics.bridge import prepare_custom_electronics_run
from evolab.cli import main


def test_prepare_custom_digital_electronics_run(tmp_path):
    spec_path = tmp_path / "custom_logic.json"
    spec_data = {
        "name": "custom_logic",
        "inputs": 2,
        "outputs": 1,
        "truth_table": [
            [[0, 0], [0]],
            [[0, 1], [1]],
            [[1, 0], [1]],
            [[1, 1], [0]],
        ],
        "functions_needed": ["XOR"],
        "max_delay_ns": 120.0,
        "max_quiescent_ua": 50.0,
    }
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")

    evaluator, pop, name = prepare_custom_electronics_run(
        spec_path=spec_path,
        netlist_path=None,
        population_size=6,
        seed=123,
    )

    assert name == "custom_logic"
    assert len(pop) == 6
    assert pop[0].species == "spec_electronics"

    # Evaluate the first individual
    res = evaluator.evaluate(pop[0])
    assert res.score >= 0.0
    assert "functional_accuracy" in res.artifacts or "correct" in res.artifacts


def test_cli_custom_electronics_run(tmp_path):
    spec_path = tmp_path / "custom_adder.json"
    spec_data = {
        "name": "custom_half_adder",
        "inputs": 2,
        "outputs": 2,
        "truth_table": [
            [[0, 0], [0, 0]],
            [[0, 1], [1, 0]],
            [[1, 0], [1, 0]],
            [[1, 1], [0, 1]],
        ],
        "functions_needed": ["XOR", "AND"],
        "max_delay_ns": 150.0,
        "max_quiescent_ua": 60.0,
    }
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")
    report_path = tmp_path / "custom_electronics_report.json"

    cmd = [
        "evolve",
        "--spec", str(spec_path),
        "-g", "2",
        "-p", "4",
        "--seed", "42",
        "-o", str(report_path),
    ]

    ret = main(cmd)
    # Target reached or not, it should execute cleanly without exceptions
    assert ret in (0, 1)
    assert report_path.is_file()

    result = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["config"]["genome"] == "electronics"
    assert result["config"]["scenario"] == "custom_half_adder"
    assert result["total_generations"] == 2
    assert result["total_candidates_evaluated"] > 0


def test_cli_electronics_reports_and_patches(tmp_path):
    spec_path = tmp_path / "custom_adder.json"
    spec_data = {
        "name": "synth_adder",
        "inputs": 2,
        "outputs": 2,
        "truth_table": [
            [[0, 0], [0, 0]],
            [[0, 1], [1, 0]],
            [[1, 0], [1, 0]],
            [[1, 1], [0, 1]],
        ],
        "functions_needed": ["XOR", "AND"],
    }
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")
    patch_out = tmp_path / "circuit.patch"
    summary_out = tmp_path / "summary.md"
    report_path = tmp_path / "report.json"

    cmd = [
        "evolve",
        "--spec", str(spec_path),
        "-g", "2",
        "-p", "4",
        "--patch-file", str(patch_out),
        "--summary-file", str(summary_out),
        "-o", str(report_path),
    ]

    ret = main(cmd)
    assert ret in (0, 1)
    assert patch_out.is_file()
    assert summary_out.is_file()

    patch_content = patch_out.read_text(encoding="utf-8")
    assert "SYNTHESIZED NETLIST" in patch_content

    summary_content = summary_out.read_text(encoding="utf-8")
    assert "Circuit Synthesis Report" in summary_content
    assert "synth_adder" in summary_content


def test_cli_electronics_hybrid_llm_breaker(tmp_path):
    spec_path = tmp_path / "custom_adder.json"
    spec_data = {
        "name": "llm_adder",
        "inputs": 2,
        "outputs": 2,
        "truth_table": [
            [[0, 0], [0, 0]],
            [[0, 1], [1, 0]],
            [[1, 0], [1, 0]],
            [[1, 1], [0, 1]],
        ],
        "functions_needed": ["XOR", "AND"],
    }
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")
    report_path = tmp_path / "report_llm.json"

    cmd = [
        "evolve",
        "--spec", str(spec_path),
        "-g", "1",
        "-p", "4",
        "--target", "99.0",
        "--llm", "mock",
        "-o", str(report_path),
    ]

    ret = main(cmd)
    assert ret in (0, 1)
    assert report_path.is_file()

    result = json.loads(report_path.read_text(encoding="utf-8"))
    history = result.get("history", [])
    assert any("llm_circuit" in str(h) for h in history) or result["total_candidates_evaluated"] >= 2
