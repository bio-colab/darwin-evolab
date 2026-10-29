"""tests/test_agent_cli_contracts.py — Autonomous Agent Machine Contract Verification.

Ensures evolab conforms to deterministic agent-native contracts:
1. Universal --json support across root and all subcommands.
2. 100% stdout purity: only pure parseable JSON on stdout in JSON mode.
3. Structured error payloads with consistent error schema on failures.
4. Deterministic exit codes:
   - 0 for successful execution and report generation.
   - 1 for internal/engine operational crash.
   - 2 for syntax or input argument errors.
5. Introspection capability for agents via root and help flags.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}


def run_cmd(args: list[str], input_str: str | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Helper to execute evolab via subprocess."""
    full_cmd = [sys.executable, "-m", "evolab.cli"] + args
    return subprocess.run(
        full_cmd,
        input=input_str,
        capture_output=True,
        text=True,
        cwd=str(cwd or ROOT),
        env=ENV,
    )


# -------------------------------------------------------------------------
# Root & Introspection Contracts
# -------------------------------------------------------------------------

def test_root_json_introspection():
    """Agent contract: evolab --json returns full command inventory."""
    proc = run_cmd(["--json"])
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert "commands" in data
    assert "eval" in data["commands"]
    assert "audit" in data["commands"]


def test_version_json():
    """Agent contract: evolab --version --json returns structured version."""
    proc = run_cmd(["--version", "--json"])
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert "version" in data


def test_help_json():
    """Agent contract: evolab --help --json returns structured commands schema."""
    proc = run_cmd(["--help", "--json"])
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert isinstance(data["commands"], dict)
    assert "optimize" in data["commands"]


def test_root_prefix_flag_order():
    """Agent contract: evolab --json <command> works identically to evolab <command> --json."""
    proc = run_cmd(["--json", "eval", "[0.0, 0.0]"])
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["score"] == 100.0


# -------------------------------------------------------------------------
# Oracle 'eval' Machine Contracts
# -------------------------------------------------------------------------

def test_eval_json_success():
    """Agent contract: eval coordinates return structured numeric oracle result."""
    proc = run_cmd(["eval", "[0.0, 0.0]", "--json"])
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["score"] == 100.0
    assert "artifacts" in data
    assert "evaluation_time_ms" in data


def test_eval_json_from_stdin():
    """Agent contract: eval accepts piped stdin vector in json mode."""
    proc = run_cmd(["eval", "-", "--json"], input_str="[0.0, 0.0, 0.0]")
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["score"] == 100.0


def test_eval_json_invalid_coordinates_error():
    """Agent contract: invalid candidate coordinates emit structured JSON error."""
    proc = run_cmd(["eval", "not_a_valid_vector", "--json"])
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["status"] == "error"
    assert data["code"] == "INVALID_COORDINATES"
    assert data["exit_code"] == 2


def test_eval_json_empty_candidate():
    """Agent contract: empty input emits structured EMPTY_CANDIDATE error."""
    proc = run_cmd(["eval", "-", "--json"], input_str="")
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["status"] == "error"
    assert data["code"] in ("EMPTY_CANDIDATE", "MISSING_CANDIDATE")


# -------------------------------------------------------------------------
# 'inspect' Machine Contracts
# -------------------------------------------------------------------------

def test_inspect_json_missing_file():
    """Agent contract: inspect nonexistent file emits structured JSON error."""
    proc = run_cmd(["inspect", "nonexistent_rep.json", "--json"])
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["status"] == "error"
    assert data["code"] == "FILE_NOT_FOUND"


def test_inspect_json_valid_report(tmp_path: Path):
    """Agent contract: inspect valid report returns structured report object."""
    rep_file = tmp_path / "rep.json"
    rep_file.write_text((ROOT / "run_report.json").read_text(encoding="utf-8"), encoding="utf-8")
    proc = run_cmd(["inspect", str(rep_file), "--json"], cwd=tmp_path)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "total_generations" in data


# -------------------------------------------------------------------------
# 'init' Machine Contracts
# -------------------------------------------------------------------------

def test_init_json_success(tmp_path: Path):
    """Agent contract: init in clean folder returns structured init status."""
    proc = run_cmd(["init", "--json"], cwd=tmp_path)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert ".evolab.json" in data["file"]
    assert (tmp_path / ".evolab.json").exists()


def test_init_json_exists_error(tmp_path: Path):
    """Agent contract: init when config already exists emits structured error."""
    (tmp_path / ".evolab.json").write_text('{"name": "test"}', encoding="utf-8")
    proc = run_cmd(["init", "--json"], cwd=tmp_path)
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["status"] == "error"
    assert data["code"] == "CONFIG_EXISTS"


def test_init_json_force_overwrite(tmp_path: Path):
    """Agent contract: init --force overwrites successfully."""
    (tmp_path / "evolab.toml").write_text("# existing")
    proc = run_cmd(["init", "--json", "--force"], cwd=tmp_path)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["status"] == "success"


# -------------------------------------------------------------------------
# 'audit' Machine Contracts
# -------------------------------------------------------------------------

def test_audit_json():
    """Agent contract: audit --json outputs pure structured scorecard."""
    proc = run_cmd(["audit", "--json"])
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "timestamp" in data
    assert "overall_verdict" in data
    assert "code_benchmarks" in data
    assert "numeric_benchmarks" in data


# -------------------------------------------------------------------------
# 'harness' Machine Contracts
# -------------------------------------------------------------------------

def test_harness_init_manifest_json(tmp_path: Path):
    """Agent contract: harness init-manifest --json scaffolds loop manifest."""
    out_m = tmp_path / "loop.json"
    proc = run_cmd(["harness", "init-manifest", "-o", str(out_m), "--json"], cwd=tmp_path)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert out_m.exists()


def test_harness_missing_action_json():
    """Agent contract: bare harness without action emits structured error."""
    proc = run_cmd(["harness", "--json"])
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["status"] == "error"


# -------------------------------------------------------------------------
# Search & Optimization Machine Contracts
# -------------------------------------------------------------------------

def test_optimize_json_exit_code_zero(tmp_path: Path):
    """Agent contract: optimize execution returning valid report yields exit code 0."""
    out_r = tmp_path / "opt_rep.json"
    proc = run_cmd(["optimize", "-g", "2", "-p", "4", "-o", str(out_r), "--json"], cwd=tmp_path)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "total_generations" in data
    assert data["total_generations"] == 2
    assert "best_individual" in data


def test_repair_json_unknown_scenario():
    """Agent contract: repair with unknown scenario emits structured UNKNOWN_SCENARIO error."""
    proc = run_cmd(["repair", "--scenario", "nonexistent_scenario_xyz", "--json"])
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["status"] == "error"
    assert data["code"] == "UNKNOWN_SCENARIO"
    assert "details" in data
    assert "available_scenarios" in data["details"]


def test_evolve_json_numeric_exit_code_zero(tmp_path: Path):
    """Agent contract: evolve with numeric genome and json flag exits with 0."""
    out_r = tmp_path / "evo_rep.json"
    proc = run_cmd([
        "evolve", "--engine", "ga", "--genome", "numeric",
        "-g", "2", "-p", "4", "-o", str(out_r), "--json"
    ], cwd=tmp_path)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "best_individual" in data


# -------------------------------------------------------------------------
# Syntax & Parser Robustness Contracts
# -------------------------------------------------------------------------

def test_syntax_error_json():
    """Agent contract: invalid command line arguments in --json mode emit structured SYNTAX_ERROR."""
    proc = run_cmd(["--unknown-flag-12345", "--json"])
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["status"] == "error"
    assert data["code"] == "SYNTAX_ERROR"
    assert data["exit_code"] == 2


def test_wizard_in_json_mode():
    """Agent contract: wizard in json mode cleanly rejects with TTY_REQUIRED."""
    proc = run_cmd(["wizard", "--json"])
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["status"] == "error"
    assert data["code"] == "TTY_REQUIRED"
