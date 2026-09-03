"""Tests for output reporters: terminal diagnostics, GitHub Markdown, Git Patch, and in-place apply."""
from __future__ import annotations

from pathlib import Path
import pytest

from evolab.reporters import (
    apply_in_place,
    format_git_patch,
    format_markdown_summary,
    format_terminal_diagnostics,
)
from evolab.cli import main


def test_format_terminal_diagnostics():
    result = {
        "best_individual": {"fitness": 100.0, "passed_holdout": True, "id": "ind_01"},
        "total_candidates_evaluated": 12,
        "total_generations": 3,
    }
    diag = format_terminal_diagnostics(
        result,
        baseline_fitness=40.0,
        baseline_failures=["arg 1 failed"],
    )
    assert "REPAIR DIAGNOSTICS" in diag
    assert "Baseline Score  :  40.00%" in diag
    assert "Final Status    : SUCCESS" in diag
    assert "PASSED (100% generalization, zero overfitting)" in diag


def test_format_markdown_summary():
    result = {
        "best_individual": {"fitness": 100.0, "passed_holdout": True},
        "total_candidates_evaluated": 10,
        "total_generations": 2,
        "history": [{"generation": 1, "best_fitness": 50.0, "edits": 0, "added": "init"}],
    }
    md = format_markdown_summary(result, diff_text="--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n-x\n+y\n")
    assert "## 🧬 Darwin-Evolab Automated Repair Report" in md
    assert "100.0% fitness" in md
    assert "```diff" in md
    assert "+y" in md


def test_format_git_patch_and_apply_in_place(tmp_path):
    src_file = tmp_path / "app.py"
    src_file.write_text("def test():\n    return False\n", encoding="utf-8")

    scenario = type("DummyScenario", (), {
        "sources": {str(src_file): src_file.read_text(encoding="utf-8")},
        "target_file": str(src_file),
    })()

    repaired = {str(src_file): "def test():\n    return True\n"}
    patch = format_git_patch(scenario, repaired, commit_msg="fix test return")
    assert "Subject: [PATCH] fix test return" in patch
    assert "-    return False" in patch
    assert "+    return True" in patch

    # Test in-place apply with backup
    modified = apply_in_place(scenario, repaired, create_backup=True)
    assert len(modified) == 1
    assert src_file.read_text(encoding="utf-8") == "def test():\n    return True\n"
    bak_file = tmp_path / "app.py.bak"
    assert bak_file.is_file()
    assert bak_file.read_text(encoding="utf-8") == "def test():\n    return False\n"


def test_cli_output_flags(tmp_path):
    app_py = tmp_path / "app.py"
    app_py.write_text("def inc(x):\n    return x + 0\n", encoding="utf-8")

    test_py = tmp_path / "test_app.py"
    test_py.write_text("def test_inc():\n    assert inc(1) == 2\n", encoding="utf-8")

    patch_out = tmp_path / "repair.patch"
    summary_out = tmp_path / "summary.md"
    rep_json = tmp_path / "report.json"

    cmd = [
        "evolve",
        "--source", str(app_py),
        "--pytest", str(test_py),
        "--patch-file", str(patch_out),
        "--summary-file", str(summary_out),
        "--apply",
        "-o", str(rep_json),
    ]

    ret = main(cmd)
    assert ret == 0
    assert patch_out.is_file()
    assert summary_out.is_file()
    assert "Darwin-Evolab" in summary_out.read_text(encoding="utf-8")
    assert "Subject: [PATCH]" in patch_out.read_text(encoding="utf-8")

    # Verify in-place patching happened!
    assert "return x + 1" in app_py.read_text(encoding="utf-8")
    assert (tmp_path / "app.py.bak").is_file()
