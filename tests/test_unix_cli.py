"""tests/test_unix_cli.py — Automated verification for UNIX toolchain etiquette.

Validates:
1. evolab --version prints the version and exits with 0.
2. evolab eval acts as a composable Fitness Oracle via arguments and stdin.
3. evolab optimize --format json keeps stdout 100% clean and parseable by jq/json.
4. evolab inspect reads report JSON from stdin via '-' or auto-piped stdin.
5. BrokenPipeError is caught gracefully with POSIX exit code 141.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_version_flag():
    """Verify evolab --version prints version string and exits with code 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "--version"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0
    assert "evolab 0.5.0" in proc.stdout


def test_cli_eval_from_argument():
    """Verify evolab eval evaluates candidate vector passed as CLI argument."""
    proc = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "eval", "[0.0, 0.0, 0.0]"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0
    score = float(proc.stdout.strip())
    assert score == 100.0


def test_cli_eval_from_stdin():
    """Verify evolab eval reads candidate vector piped into stdin."""
    proc = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "eval", "-"],
        input="[0.0, 0.0, 0.0]",
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0
    score = float(proc.stdout.strip())
    assert score == 100.0


def test_cli_eval_json_format():
    """Verify evolab eval --format json outputs structured JSON with loss and artifacts."""
    proc = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "eval", "[1.0, 2.0]", "--landscape", "sphere", "--format", "json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "score" in data
    assert data["artifacts"]["landscape"] == "sphere"
    assert data["artifacts"]["dimension"] == 2


def test_cli_stdout_purity_format_json():
    """Verify stdout contains ONLY parseable JSON without banner or status noise when --format json is used."""
    proc = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "optimize", "-g", "2", "-p", "4", "-s", "1", "--format", "json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode in (0, 1)
    # stdout MUST be valid JSON starting with '{'
    clean_stdout = proc.stdout.strip()
    assert clean_stdout.startswith("{"), f"stdout does not start with '{{': {clean_stdout[:60]}"
    parsed = json.loads(clean_stdout)
    assert "total_generations" in parsed
    assert "best_individual" in parsed
    # Banners like 'Engine: ...' must have gone to stderr
    assert "Engine: GA" in proc.stderr

    # Run with --quiet: stderr should be completely silent (Rule of Silence)
    proc_quiet = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "optimize", "-g", "2", "-p", "4", "-s", "1", "--format", "json", "--quiet"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc_quiet.returncode in (0, 1)
    assert proc_quiet.stderr == ""


def test_cli_inspect_stdin():
    """Verify evolab inspect reads report directly from stdin via '-' or piped input."""
    # Generate report in memory
    sample_report = {
        "total_generations": 2,
        "total_candidates_evaluated": 8,
        "best_individual": {
            "id": "gen_02_ind_00",
            "fitness": 99.8,
            "species": "spec_default",
            "passed_holdout": True,
        },
        "species_distribution": {"spec_default": 4},
        "early_stop_triggered": True,
        "timestamp_utc": "2026-09-12T12:00:00Z",
        "engine_version": "evolab-engine/0.5.0",
    }
    raw_json = json.dumps(sample_report)

    # Test inspect -
    proc_dash = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "inspect", "-"],
        input=raw_json,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc_dash.returncode == 0, proc_dash.stderr
    assert "Source           : <stdin>" in proc_dash.stdout

    # Test inspect without argument (auto stdin)
    proc_auto = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "inspect"],
        input=raw_json,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc_auto.returncode == 0, proc_auto.stderr
    assert "Source           : <stdin>" in proc_auto.stdout


def test_cli_broken_pipe_handling():
    """Verify BrokenPipeError is caught cleanly and exits with POSIX code 141."""
    script = (
        "import sys\n"
        "from evolab.cli import main\n"
        "class BrokenStdout:\n"
        "    def write(self, s):\n"
        "        raise BrokenPipeError('Broken pipe')\n"
        "    def flush(self):\n"
        "        pass\n"
        "    def isatty(self):\n"
        "        return False\n"
        "    def fileno(self):\n"
        "        return 1\n"
        "sys.stdout = BrokenStdout()\n"
        "sys.exit(main(['eval', '[0.0, 0.0]']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 141
    # Traceback should NOT be printed to stderr
    assert "Traceback" not in proc.stderr


def test_cli_telemetry_stream(tmp_path: Path):
    """Verify --telemetry-stream emits real-time JSONL generation events."""
    telemetry_file = tmp_path / "telemetry.jsonl"
    proc = subprocess.run(
        [
            sys.executable, "-m", "evolab.cli", "optimize",
            "-g", "3", "-p", "4", "-s", "42",
            "--telemetry-stream", str(telemetry_file),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode in (0, 1)
    assert telemetry_file.is_file()
    lines = [line.strip() for line in telemetry_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 3  # At least 3 generation events
    records = [json.loads(line) for line in lines]
    gen_records = [r for r in records if r.get("event") == "generation"]
    assert len(gen_records) == 3
    assert gen_records[0]["gen"] == 1
    assert "best_fitness" in gen_records[0]
    assert "mean_fitness" in gen_records[0]
    assert "diversity" in gen_records[0]

    complete_records = [r for r in records if r.get("event") == "completed"]
    assert len(complete_records) == 1
    assert complete_records[0]["total_generations"] == 3


def test_external_process_evaluator_unit(tmp_path: Path):
    """Unit test for ExternalProcessEvaluator communicating via JSON-RPC 2.0."""
    from evolab.ipc_evaluator import ExternalProcessEvaluator

    driver_code = (
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "    if not line.strip(): continue\n"
        "    req = json.loads(line)\n"
        "    if req.get('method') == 'shutdown': break\n"
        "    if req.get('method') == 'evaluate':\n"
        "        cand = req['params']['candidate']\n"
        "        score = 100.0 if cand == [1.0, 1.0] else 42.0\n"
        "        resp = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'fitness': score}}\n"
        "        sys.stdout.write(json.dumps(resp) + '\\n')\n"
        "        sys.stdout.flush()\n"
    )
    driver_script = tmp_path / "mock_driver.py"
    driver_script.write_text(driver_code, encoding="utf-8")

    evaluator = ExternalProcessEvaluator(driver_cmd=[sys.executable, str(driver_script)])
    try:
        res1 = evaluator.evaluate([1.0, 1.0])
        assert res1.score == 100.0

        res2 = evaluator.evaluate([0.0, 0.0])
        assert res2.score == 42.0
    finally:
        evaluator.close()


def test_cli_external_driver(tmp_path: Path):
    """Verify evolab optimize runs with an external JSON-RPC 2.0 evaluation driver."""
    driver_code = (
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "    if not line.strip(): continue\n"
        "    req = json.loads(line)\n"
        "    if req.get('method') == 'shutdown': break\n"
        "    if req.get('method') == 'evaluate':\n"
        "        cand = req['params']['candidate']\n"
        "        dist = sum(x**2 for x in cand)\n"
        "        score = round(100.0 / (1.0 + dist), 4)\n"
        "        resp = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'fitness': score}}\n"
        "        sys.stdout.write(json.dumps(resp) + '\\n')\n"
        "        sys.stdout.flush()\n"
    )
    driver_script = tmp_path / "ext_driver.py"
    driver_script.write_text(driver_code, encoding="utf-8")

    driver_cmd = f"{sys.executable} {driver_script}"
    proc = subprocess.run(
        [
            sys.executable, "-m", "evolab.cli", "optimize",
            "-g", "3", "-p", "4", "-s", "1",
            "--external-driver", driver_cmd,
            "--format", "json",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode in (0, 1)
    data = json.loads(proc.stdout)
    assert data["total_generations"] == 3
    assert data["best_individual"]["fitness"] > 0.0


def test_cli_repair_stdin_piping(tmp_path: Path):
    """Verify piping source code to evolab repair via stdin produces clean patch targeting target-file."""
    test_code = (
        "def test_calc():\n"
        "    assert compute(2) == 4\n"
        "    assert compute(3) == 6\n"
    )
    test_file = tmp_path / "test_calc.py"
    test_file.write_text(test_code, encoding="utf-8")

    buggy_source = "def compute(x):\n    return x + 2\n"

    proc = subprocess.run(
        [
            sys.executable, "-m", "evolab.cli", "repair",
            "--source", "-",
            "--target-file", "calc.py",
            "--pytest", str(test_file),
            "--format", "patch",
            "--quiet",
        ],
        input=buggy_source,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode in (0, 1)
    # Rule of silence: stderr must be completely empty with --quiet
    assert proc.stderr == ""
    # stdout should start with patch header or be empty if target reached/unreached
    if proc.stdout.strip():
        assert "--- a/calc.py" in proc.stdout
        assert "+++ b/calc.py" in proc.stdout
