"""Tests for Pytest scenario loader and CLI integration."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest

from evolab.code_fixtures import load_pytest_scenario
from evolab.repair import greedy_repair


def test_load_pytest_scenario_extraction(tmp_path):
    app_py = tmp_path / "app.py"
    app_py.write_text(
        "def add_five(x):\n"
        "    return x + 4\n",   # off-by-one bug
        encoding="utf-8",
    )

    test_py = tmp_path / "test_app.py"
    test_py.write_text(
        "def test_add_five():\n"
        "    assert add_five(0) == 5\n"
        "    assert add_five(1) == 6\n"
        "    assert add_five(10) == 15\n",
        encoding="utf-8",
    )

    scenario = load_pytest_scenario([app_py], test_py, func_name="add_five")
    assert scenario.name == "test_app"
    assert scenario.func_name == "add_five"
    # Should have extracted 3 cases, splitting into test_cases and holdout_cases
    assert len(scenario.test_cases) == 2
    assert len(scenario.holdout_cases) == 1

    # Now verify that greedy repair actually fixes this bug using the extracted scenario!
    ev = scenario.create_evaluator()
    repaired, history, n_eval = greedy_repair(
        scenario.sources,
        scenario.target_file,
        ev,
    )
    res = ev.evaluate(repaired)
    assert res.score == 100.0
    assert res.passed_holdout is True
    assert "return x + 5" in repaired.to_code()


def test_cli_evolve_with_pytest_flag(tmp_path):
    app_py = tmp_path / "app.py"
    app_py.write_text(
        "def is_positive(x):\n"
        "    return x <= 0\n",  # boundary bug
        encoding="utf-8",
    )

    test_py = tmp_path / "test_app.py"
    test_py.write_text(
        "def test_is_positive():\n"
        "    assert is_positive(5) == False\n"
        "    assert is_positive(-2) == True\n",
        encoding="utf-8",
    )

    from evolab.cli import main
    # Run CLI command with --pytest
    cmd = [
        "evolve",
        "--source", str(app_py),
        "--pytest", str(test_py),
        "--func", "is_positive",
        "--diff",
        "-o", str(tmp_path / "report.json"),
    ]
    exit_code = main(cmd)
    assert exit_code == 0
    assert (tmp_path / "report.json").is_file()
