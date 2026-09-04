"""Tests for SWE-bench Lite industrial APR adapter and dual-invariant validation."""
from pathlib import Path
import random
import pytest

from evolab.swe_bench import (
    SWEBenchAdapter,
    SWEBenchEvaluator,
    SWEBenchInstance,
    SWEBenchResolution,
)
from evolab.repair import RepairGenome
from evolab.genome import Individual
from evolab.cli import main


FIXTURES_DIR = Path(__file__).parent.parent / "src" / "evolab" / "fixtures" / "swe_bench"


def test_parse_swe_bench_instance_from_file_and_dict():
    adapter = SWEBenchAdapter()
    file_path = FIXTURES_DIR / "sympy__sympy_13480.json"
    assert file_path.exists(), f"Missing fixture: {file_path}"

    spec = adapter.parse_spec(str(file_path))
    assert isinstance(spec, SWEBenchInstance)
    assert spec.instance_id == "sympy__sympy-13480"
    assert spec.repo == "sympy/sympy"
    assert spec.target_file == "sympy/functions/elementary/hyperbolic.py"
    assert len(spec.fail_to_pass_tests) == 1
    assert len(spec.pass_to_pass_tests) == 3

    # Direct dict parsing
    spec_dict = adapter.parse_spec({
        "instance_id": "custom_test_1",
        "repo": "repo/custom",
        "target_file": "mod.py",
        "sources": {"mod.py": "def solve(x): return x + 1\n"},
        "fail_to_pass_tests": [[[1], 2]],
        "pass_to_pass_tests": [[[0], 1]],
    })
    assert spec_dict.instance_id == "custom_test_1"
    assert spec_dict.repo == "repo/custom"


def test_swe_bench_evaluator_dual_invariants():
    from evolab.repair import catalog_sources
    spec = SWEBenchInstance(
        instance_id="dummy-001",
        repo="dummy/repo",
        problem_statement="dummy bug",
        target_file="dummy.py",
        sources={"dummy.py": "def solve(x: int) -> int:\n    if x > 0:\n        return 1\n    return 0\n"},
        fail_to_pass_tests=[[[0], 1]],  # Expected 1 for 0 (currently returns 0 -> fails)
        pass_to_pass_tests=[[[5], 1], [[-1], 0]],  # Returns 1 and 0 -> passes
    )
    evaluator = SWEBenchEvaluator(spec, func_name="solve")

    # 1. Buggy genome (no edits): FAIL_TO_PASS fails, PASS_TO_PASS passes
    buggy_genome = RepairGenome(sources=spec.sources, target_file=spec.target_file, edits=[])
    res_buggy = evaluator.evaluate(buggy_genome)
    assert res_buggy.artifacts["resolved"] is False
    assert res_buggy.artifacts["fail_to_pass_passed"] is False
    assert res_buggy.artifacts["pass_to_pass_clean"] is True
    assert res_buggy.score < 100.0

    # 2. Fixed genome (boundary_cmp edit): both pass 100%
    boundary_edits = [e for e in catalog_sources(spec.sources) if e.kind == "boundary_cmp"]
    assert len(boundary_edits) > 0
    fixed_genome = RepairGenome(sources=spec.sources, target_file=spec.target_file, edits=boundary_edits)
    res_fixed = evaluator.evaluate(fixed_genome)
    assert res_fixed.artifacts["resolved"] is True
    assert res_fixed.artifacts["fail_to_pass_passed"] is True
    assert res_fixed.artifacts["pass_to_pass_clean"] is True
    assert res_fixed.score == 100.0


def test_swe_bench_adapter_population_and_species():
    adapter = SWEBenchAdapter()
    spec = adapter.parse_spec(str(FIXTURES_DIR / "sympy__sympy_13480.json"))
    rng = random.Random(42)

    pop = adapter.build_population(spec, size=6, rng=rng)
    assert len(pop) == 6
    for ind in pop:
        assert isinstance(ind, Individual)
        assert ind.species.startswith("spec_")
        assert ind.species == "spec_swe_bench"


def test_swe_bench_solve_sympy_instance():
    adapter = SWEBenchAdapter()
    spec = adapter.parse_spec(str(FIXTURES_DIR / "sympy__sympy_13480.json"))
    resolution = adapter.solve_instance(spec, max_evals=16)

    assert resolution.resolved is True
    assert resolution.fail_to_pass_passed is True
    assert resolution.pass_to_pass_clean is True
    assert resolution.evaluations_used <= 10
    assert resolution.execution_time_seconds < 1.0
    assert "--- a/sympy/functions/elementary/hyperbolic.py" in resolution.generated_patch
    assert "if val >= 0.0:" in resolution.generated_patch


def test_swe_bench_solve_pytest_instance():
    adapter = SWEBenchAdapter()
    spec = adapter.parse_spec(str(FIXTURES_DIR / "pytest_dev__pytest_5227.json"))
    resolution = adapter.solve_instance(spec, max_evals=16)

    assert resolution.resolved is True
    assert resolution.fail_to_pass_passed is True
    assert resolution.pass_to_pass_clean is True
    assert resolution.evaluations_used <= 10
    assert resolution.execution_time_seconds < 1.0
    assert "--- a/_pytest/logging.py" in resolution.generated_patch
    assert "if is_custom:" in resolution.generated_patch


def test_swe_bench_export_solution(tmp_path):
    adapter = SWEBenchAdapter()
    spec = adapter.parse_spec(str(FIXTURES_DIR / "sympy__sympy_13480.json"))
    resolution = adapter.solve_instance(spec, max_evals=16)
    assert resolution.resolved is True

    out_patch = tmp_path / "repaired.patch"
    individual = Individual(
        genome=RepairGenome(sources=spec.sources, target_file=spec.target_file, edits=[]),
        species="spec_swe_bench",
    )
    res_dict = adapter.export_solution(individual, spec, output_path=out_patch)
    assert out_patch.exists()
    assert "instance_id" in res_dict
    assert res_dict["instance_id"] == "sympy__sympy-13480"


def test_cli_swe_bench_execution(tmp_path):
    patch_out = tmp_path / "cli_swe.patch"
    fixture_path = FIXTURES_DIR / "sympy__sympy_13480.json"
    cmd = [
        "evolve",
        "--swe-bench", str(fixture_path),
        "--patch-out", str(patch_out),
    ]
    ret = main(cmd)
    assert ret == 0
    assert patch_out.exists()
    content = patch_out.read_text(encoding="utf-8")
    assert "if val >= 0.0:" in content

