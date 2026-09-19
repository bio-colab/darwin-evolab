"""tests/test_swe_bench_300.py — Automated Tests for the Full 300-Instance SWE-bench Suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.swe_bench import SWEBenchAdapter

ROOT_DIR = Path(__file__).resolve().parent.parent
FIXTURES_300_DIR = ROOT_DIR / "src" / "evolab" / "fixtures" / "swe_bench_300"
REPORT_300_PATH = ROOT_DIR / "reports" / "swe_bench_lite_300.json"


def test_swe_bench_300_fixtures_integrity():
    """Verify that exactly 300 industrial benchmark fixtures exist and parse cleanly."""
    adapter = SWEBenchAdapter()
    fixture_files = sorted(FIXTURES_300_DIR.glob("*.json"))
    assert len(fixture_files) == 300, f"Expected exactly 300 fixtures, found {len(fixture_files)}"

    for f in fixture_files:
        spec = adapter.parse_spec(f)
        assert spec.instance_id, f"Missing instance_id in {f.name}"
        assert spec.repo, f"Missing repo in {f.name}"
        assert spec.target_file, f"Missing target_file in {f.name}"
        assert spec.sources, f"Missing sources in {f.name}"
        assert len(spec.fail_to_pass_tests) >= 1, f"Missing fail_to_pass_tests in {f.name}"
        assert len(spec.pass_to_pass_tests) >= 1, f"Missing pass_to_pass_tests in {f.name}"


def test_swe_bench_300_empirical_report():
    """Verify that the official N=300 empirical report satisfies dual invariants and hardware disclosure."""
    assert REPORT_300_PATH.is_file(), f"Report not found: {REPORT_300_PATH}"
    data = json.loads(REPORT_300_PATH.read_text(encoding="utf-8"))

    assert data.get("total_instances") == 300
    assert data.get("resolved_count") >= 100
    assert data.get("pass_rate_percent") >= 33.0
    assert len(data.get("instances", [])) == 300

    hw = data.get("hardware_disclosure", {})
    assert "i5-8350U" in hw.get("processor", "")
    assert hw.get("memory_ram_gb") == 8.0

    for inst in data["instances"]:
        if inst["resolved"]:
            assert inst["fail_to_pass_passed"] is True
            assert inst["pass_to_pass_clean"] is True
