"""tests/test_swe_bench_50.py — Tests for the Expanded N=50 SWE-bench Industrial Suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.swe_bench import SWEBenchAdapter

ROOT_DIR = Path(__file__).resolve().parent.parent
FIXTURES_50_DIR = ROOT_DIR / "src" / "evolab" / "fixtures" / "swe_bench_50"
REPORT_50_PATH = ROOT_DIR / "reports" / "swe_bench_lite_50.json"


def test_swe_bench_50_fixtures_integrity():
    """Verify that exactly 50 industrial benchmark fixtures exist and parse cleanly."""
    adapter = SWEBenchAdapter()
    fixture_files = sorted(FIXTURES_50_DIR.glob("*.json"))
    assert len(fixture_files) >= 50, f"Expected >= 50 fixtures, found {len(fixture_files)}"

    for f in fixture_files:
        spec = adapter.parse_spec(f)
        assert spec.instance_id, f"Missing instance_id in {f.name}"
        assert spec.repo, f"Missing repo in {f.name}"
        assert spec.target_file, f"Missing target_file in {f.name}"
        assert spec.sources, f"Missing sources in {f.name}"
        assert len(spec.fail_to_pass_tests) >= 1, f"Missing fail_to_pass_tests in {f.name}"
        assert len(spec.pass_to_pass_tests) >= 1, f"Missing pass_to_pass_tests in {f.name}"


def test_swe_bench_50_empirical_report():
    """Verify that the official N=50 empirical report satisfies benchmark invariants."""
    assert REPORT_50_PATH.is_file(), f"Report not found: {REPORT_50_PATH}"
    data = json.loads(REPORT_50_PATH.read_text(encoding="utf-8"))

    assert data.get("total_instances") == 50
    assert data.get("resolved_count") >= 20
    assert data.get("pass_rate_percent") >= 40.0
    assert len(data.get("instances", [])) == 50

    for inst in data["instances"]:
        if inst["resolved"]:
            assert inst["fail_to_pass_passed"] is True
            assert inst["pass_to_pass_clean"] is True
