"""Unit tests for SelfAuditSupervisor and governance harness."""
from __future__ import annotations

import io
from pathlib import Path
from evolab.supervisor import SelfAuditSupervisor


def test_self_audit_supervisor_run(tmp_path: Path):
    out_path = tmp_path / "test_audit.json"
    buf = io.StringIO()
    supervisor = SelfAuditSupervisor(
        candidate_mode="directed",
        full_suite=False,
        output_path=str(out_path),
        quiet=False,
        stream=buf,
    )
    report = supervisor.run_audit()

    assert out_path.is_file()
    assert report["overall_verdict"] in ("PASS", "NEEDS_REVIEW")
    assert report["code_benchmarks"]["total"] == 4
    assert report["code_benchmarks"]["passed"] == 4
    assert report["code_benchmarks"]["wilson_ci_95"][1] == 1.0
    assert "invariants" in report
    assert report["invariants"]["zero_code_regressions"] is True

    scorecard_output = buf.getvalue()
    assert "EVOLAB AUTONOMOUS AUDIT SCORECARD" in scorecard_output
    assert "Code Repair Benchmark" in scorecard_output
