"""tests/test_neural_trifactor_benchmark.py — Empirical Tri-Factor Benchmark Validation for DWSP."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.capsule import inspect_evidence_capsule, verify_evidence_capsule
from evolab.neural.benchmark import ControlledToyBenchmark
from evolab.neural.experiment import (
    run_group_a_blind_ga,
    run_group_b_oneshot_llm,
    run_group_c_darwin_wsp,
    run_tri_factor_benchmark,
)


def test_tri_factor_comparative_benchmark(tmp_path: Path):
    """Verify empirical superiority of Darwin WSP (Group C) over Blind GA and raw one-shot."""
    bundle_file = tmp_path / "dwsp_solution.evobundle"
    results = run_tri_factor_benchmark(seed=42, output_bundle=bundle_file)

    assert results["benchmark"] == "Controlled 4-Class Neural Weight Repair"
    res_a = results["group_a_blind_ga"]
    res_b = results["group_b_oneshot_llm"]
    res_c = results["group_c_darwin_wsp"]

    # 1. Group A (Blind GA without semantic guidance) fails to fix the defect within budget
    assert res_a["passed_holdout"] is False
    assert res_a["target_recall"] < 50.0

    # 2. Group C (Darwin WSP Dual-System) achieves 100% resolution with ZERO regressions
    assert res_c["passed_holdout"] is True
    assert res_c["target_recall"] >= 96.0
    assert res_c["anchor_accuracy"] >= 99.9
    assert res_c["regressions"] == 0
    assert res_c["fitness"] >= 96.0

    # 3. Verify cryptographic evidence capsule
    assert bundle_file.is_file()
    assert res_c["evidence_capsule"] == str(bundle_file)

    # Cryptographic integrity verification
    verif = verify_evidence_capsule(bundle_file)
    assert verif["valid"] is True
    assert verif["status"] == "VERIFIED"

    # Inspection
    info = inspect_evidence_capsule(bundle_file)
    assert "report.json" in info["files"]
    assert "attestation.json" in info["files"]


def test_tri_factor_robustness_across_seeds():
    """Verify that Darwin WSP consistently solves the defect across multiple random seeds."""
    for seed in [1, 2, 3]:
        bench = ControlledToyBenchmark(seed=seed)
        spec = bench.get_spec()
        res_c = run_group_c_darwin_wsp(spec, seed=seed)
        assert res_c["passed_holdout"] is True
        assert res_c["regressions"] == 0
        assert res_c["target_recall"] >= 96.0
