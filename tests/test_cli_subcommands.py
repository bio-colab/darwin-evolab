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


def test_cli_progressive_disclosure(tmp_path: Path, capsys):
    out_file = tmp_path / "report_lean.json"
    main(["repair", "--scenario", "click_cli_parser", "--quiet", "-o", str(out_file)])
    captured = capsys.readouterr().out
    assert "Saved to        :" in captured
    assert "use --diagnose for detailed fault analysis" in captured
    assert "REPAIR DIAGNOSTICS" not in captured

    out_diag = tmp_path / "report_diag.json"
    main(["repair", "--scenario", "click_cli_parser", "--diagnose", "--quiet", "-o", str(out_diag)])
    captured_diag = capsys.readouterr().out
    assert "REPAIR DIAGNOSTICS" in captured_diag
    assert "Richness         :" in captured_diag


def test_cli_argument_groups_help():
    from evolab.cli import build_parser
    parser = build_parser()
    sub_action = [a for a in parser._actions if a.dest == "command"][0]
    p_evo = sub_action.choices["evolve"]
    help_text = p_evo.format_help()
    assert "Target Specification & Scenario Inputs:" in help_text
    assert "Evolution Budget & Hyperparameters:" in help_text
    assert "Diagnostics, Output & Reporting:" in help_text
    assert "Silicon & Domain Synthesis (Hardware Track):" in help_text

