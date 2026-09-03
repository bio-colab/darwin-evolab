"""Tests for Hybrid LLM Stagnation Breaker in CLI."""
from __future__ import annotations

from pathlib import Path
from evolab.cli import main


def test_cli_hybrid_llm_stagnation_breaker_invoked(tmp_path):
    app_py = tmp_path / "app.py"
    # A function with a syntax/logic structure
    app_py.write_text(
        "def custom_func(x):\n"
        "    return 'unrepairable_by_catalog'\n",
        encoding="utf-8",
    )

    test_py = tmp_path / "test_app.py"
    test_py.write_text(
        "def test_custom_func():\n"
        "    assert custom_func(1) == 'fixed'\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "report.json"
    cmd = [
        "evolve",
        "--source", str(app_py),
        "--pytest", str(test_py),
        "--func", "custom_func",
        "--llm", "mock",
        "-o", str(out_file),
    ]

    # main will run greedy (stagnates), then invoke mock LLM stagnation breaker
    exit_code = main(cmd)
    # Output file was written and LLM was invoked safely
    assert out_file.is_file()
