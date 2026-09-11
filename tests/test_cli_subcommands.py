"""Unit tests for evolab CLI subcommands (repair, optimize, audit, init)."""
from __future__ import annotations

import json
from pathlib import Path
from evolab.cli import main


def test_cli_repair_subcommand(tmp_path: Path):
    out_file = tmp_path / "report_repair.json"
    rc = main(["repair", "--scenario", "click_cli_parser", "--quiet", "-o", str(out_file)])
    assert rc == 0
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["best_individual"]["fitness"] >= 99.7


def test_cli_optimize_subcommand(tmp_path: Path):
    out_file = tmp_path / "report_opt.json"
    rc = main(["optimize", "-g", "2", "-p", "4", "-s", "42", "--quiet", "-o", str(out_file)])
    # optimize on 2 generations may not hit 99.7, so rc can be 0 or 1, but file must exist and be valid
    assert rc in (0, 1)
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["total_generations"] == 2


def test_cli_init_subcommand(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["init", "--format", "json", "--force"])
    assert rc == 0
    assert (tmp_path / ".evolab.json").is_file()

    rc_toml = main(["init", "--format", "toml", "--force"])
    assert rc_toml == 0
    assert (tmp_path / "evolab.toml").is_file()


def test_cli_audit_subcommand(tmp_path: Path):
    out_audit = tmp_path / "self_audit.json"
    rc = main(["audit", "--quiet", "-o", str(out_audit)])
    assert rc in (0, 1)
    assert out_audit.is_file()
    data = json.loads(out_audit.read_text(encoding="utf-8"))
    assert "overall_verdict" in data
    assert "governor" in data
    assert "code_benchmarks" in data
    assert data["code_benchmarks"]["passed"] == 4
