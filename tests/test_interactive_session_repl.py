"""tests/test_interactive_session_repl.py — Test stateful session engine & interactive REPL kernel."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evolab.repl import ReplKernel
from evolab.session import SessionManager


def test_session_manager_param_lifecycle():
    sm = SessionManager()
    assert sm.get_param("population_size") == 16
    assert sm.get_param("seed") == 42

    # Set new param and verify undo
    sm.set_param("population_size", 64)
    assert sm.get_param("population_size") == 64

    sm.set_param("seed", 999)
    assert sm.get_param("seed") == 999

    # Undo seed change
    undone = sm.undo()
    assert undone is True
    assert sm.get_param("seed") == 42
    assert sm.get_param("population_size") == 64

    # Undo population change
    undone = sm.undo()
    assert undone is True
    assert sm.get_param("population_size") == 16

    # Redo population change
    redone = sm.redo()
    assert redone is True
    assert sm.get_param("population_size") == 64

    # Reset
    sm.reset()
    assert sm.get_param("population_size") == 16


def test_session_manager_persistence(tmp_path: Path):
    sess_file = tmp_path / "custom_session.json"
    sm = SessionManager(session_path=sess_file)
    sm.set_param("population_size", 128)
    sm.set_param("landscape", "sphere")
    saved_path = sm.save()
    assert saved_path.is_file()

    # Load in new manager instance
    sm2 = SessionManager(session_path=sess_file)
    loaded = sm2.load()
    assert loaded is True
    assert sm2.get_param("population_size") == 128
    assert sm2.get_param("landscape") == "sphere"


def test_repl_kernel_commands():
    sm = SessionManager()
    kernel = ReplKernel(session=sm, output_json=True)

    # 1. status
    res_status = kernel.execute_line("status")
    assert res_status["status"] == "success"
    assert "params" in res_status["session"]

    # 2. set & get
    res_set = kernel.execute_line("set population_size 32")
    assert res_set["status"] == "success"
    assert res_set["param"] == "population_size"
    assert res_set["value"] == 32

    res_get = kernel.execute_line("get population_size")
    assert res_get["status"] == "success"
    assert res_get["value"] == 32

    # 3. undo
    res_undo = kernel.execute_line("undo")
    assert res_undo["status"] == "success"
    assert sm.get_param("population_size") == 16

    # 4. eval candidate
    res_eval = kernel.execute_line("eval [0.0, 0.0]")
    assert res_eval["status"] == "success"
    assert isinstance(res_eval["score"], (int, float))

    # 5. run optimization
    res_run = kernel.execute_line("run 2")
    assert res_run["status"] == "success"
    assert res_run["generations"] == 2
    assert "best_fitness" in res_run


def test_repl_kernel_script_execution():
    sm = SessionManager()
    kernel = ReplKernel(session=sm, output_json=True)

    script = "set seed 77; set generations 2; run 2"
    results = kernel.execute_script(script)
    assert len(results) == 3
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "success"
    assert results[2]["status"] == "success"
    assert sm.get_param("seed") == 77


def test_cli_repl_exec():
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolab.cli",
            "repl",
            "--exec",
            "set population_size 32; eval [0.0, 0.0]",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] == "success"
    assert len(data["results"]) == 2
    assert data["results"][0]["param"] == "population_size"
    assert data["results"][1]["status"] == "success"
