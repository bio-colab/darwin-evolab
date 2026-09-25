"""test_harness.py — Comprehensive Unit & Integration Tests for 2026 Loop & Harness Subsystem.

Verifies:
1. Declarative LoopManifest serialization, loading, and schema validation (loop.json).
2. Additive Baseline Verification Gate:
   - Evaluates strictly on candidate delta (E_cand \\ E_base = empty).
   - Allows preexisting type debt while rejecting newly introduced type errors.
   - Strictly enforces zero regressions (PASS_TO_PASS = 100%).
   - Enforces AST purity.
3. End-to-end DeterministicHarness lifecycle execution.
4. Evolutionary Trajectory Distillation to reusable O(1) workflow.json recipes.
5. Deterministic WorkflowExecutor instant replay without search.
6. LLM Context Governor token tracking, budget ceiling, and surgical AST pruning.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from evolab.code_fixtures import scenario_click_parser, scenario_requests_auth_url
from evolab.harness import (
    AdditiveBaselineGate,
    AdditiveVerificationReport,
    BaselineSnapshot,
    ContextBudget,
    DeterministicHarness,
    GoalSpec,
    GovernorSpec,
    HarnessExecutionReport,
    LLMContextGovernor,
    LoopBudget,
    LoopManifest,
    LoopSpec,
    TargetSpec,
    WorkflowAction,
    WorkflowExecutor,
    WorkflowManifest,
    distill_trajectory_to_workflow,
)


def test_loop_manifest_serialization_and_roundtrip(tmp_path: Path):
    """Verify LoopManifest declarative serialization, disk saving, and roundtrip deserialization."""
    manifest = LoopManifest.for_repair(
        name="test_apr_manifest",
        sources={"app.py": "def f(): return 1\n"},
        target_file="app.py",
        test_code="assert f() == 2\n",
        max_evals=32,
        seed=123,
    )
    d = manifest.to_dict()
    assert d["$schema"] == "https://evolab.dev/schemas/loop.v1.json"
    assert d["name"] == "test_apr_manifest"
    assert d["goal"]["goal_type"] == "repair"
    assert d["loop"]["budget"]["max_evaluations"] == 32
    assert d["target"]["verification_mode"] == "additive_baseline"

    # Save to loop.json
    loop_file = tmp_path / "loop.json"
    manifest.save(loop_file)
    assert loop_file.is_file()

    # Load from disk
    restored = LoopManifest.load(loop_file)
    assert restored.name == manifest.name
    assert restored.goal.target_file == "app.py"
    assert restored.loop.random_seed == 123
    assert restored.target.enforce_additive_type_check is True


def test_additive_baseline_gate_allows_preexisting_type_errors():
    """Verify that AdditiveBaselineGate passes if preexisting type errors are unchanged, but rejects new errors."""
    gate = AdditiveBaselineGate(enforce_purity=True, enforce_type_check=True)

    # Baseline has preexisting type errors
    base_snapshot = BaselineSnapshot(
        code_hash="mock_hash",
        passing_tests=["test_sanity"],
        failing_tests=["test_bug"],
        type_errors=[
            'Incompatible types in assignment (expression has type "str", variable has type "int")',
            'Cannot determine type of "unknown_obj"',
        ],
        fitness_score=50.0,
    )

    # Candidate 1: Fixed bug, still has preexisting type errors, NO NEW type errors
    cand1_sources = {"main.py": "def solve():\n    return 42\n"}
    # Mock type checker returning exact same preexisting type errors
    class MockTypeGateClean:
        def check_code(self, code: str):
            from evolab.type_gate import TypeCheckResult
            return TypeCheckResult(
                passed=False,
                error_count=2,
                errors=list(base_snapshot.type_errors),
                tool_used="mypy",
            )

    gate.type_check_gate = MockTypeGateClean()

    # Mock evaluator showing bug resolved and no regression
    class MockEvaluatorSuccess:
        def evaluate(self, target: Any, context: Any = None):
            from evolab.evaluators import FitnessResult
            return FitnessResult(
                score=100.0,
                artifacts={"passing_tests": ["test_sanity", "test_bug"], "failing_tests": []},
            )

    rep1 = gate.verify(
        candidate_sources=cand1_sources,
        target_file="main.py",
        baseline=base_snapshot,
        evaluator=MockEvaluatorSuccess(),
    )

    assert rep1.passed is True
    assert rep1.verdict == "ACCEPT"
    assert len(rep1.new_type_errors) == 0
    assert rep1.baseline_type_error_count == 2
    assert "test_bug" in rep1.fail_to_pass_resolved
    assert "test_sanity" in rep1.pass_to_pass_preserved
    assert len(rep1.regressions) == 0

    # Candidate 2: Introduced a NEW type error
    class MockTypeGateNewError:
        def check_code(self, code: str):
            from evolab.type_gate import TypeCheckResult
            return TypeCheckResult(
                passed=False,
                error_count=3,
                errors=list(base_snapshot.type_errors) + ["Incompatible return type (got float, expected int)"],
                tool_used="mypy",
            )

    gate.type_check_gate = MockTypeGateNewError()
    rep2 = gate.verify(
        candidate_sources=cand1_sources,
        target_file="main.py",
        baseline=base_snapshot,
        evaluator=MockEvaluatorSuccess(),
    )

    assert rep2.passed is False
    assert rep2.verdict == "REJECT"
    assert len(rep2.new_type_errors) == 1
    assert "new_type_errors_introduced: 1" in rep2.reasons


def test_additive_baseline_gate_rejects_regressions():
    """Verify that any regression on preexisting passing tests is rejected."""
    gate = AdditiveBaselineGate(enforce_purity=False, enforce_type_check=False)

    base_snapshot = BaselineSnapshot(
        code_hash="mock_hash",
        passing_tests=["test_feature_a", "test_feature_b"],
        failing_tests=["test_bug_c"],
        type_errors=[],
        fitness_score=66.6,
    )

    # Candidate fixed bug_c but broke feature_a!
    class MockEvaluatorRegression:
        def evaluate(self, target: Any, context: Any = None):
            from evolab.evaluators import FitnessResult
            return FitnessResult(
                score=66.6,
                artifacts={"passing_tests": ["test_feature_b", "test_bug_c"], "failing_tests": ["test_feature_a"]},
            )

    rep = gate.verify(
        candidate_sources={"main.py": ""},
        target_file="main.py",
        baseline=base_snapshot,
        evaluator=MockEvaluatorRegression(),
    )

    assert rep.passed is False
    assert rep.verdict == "REJECT"
    assert "test_feature_a" in rep.regressions
    assert any("regressions_detected" in r for r in rep.reasons)


def test_deterministic_harness_execution_and_distillation(tmp_path: Path):
    """Verify end-to-end execution of DeterministicHarness and workflow generation."""
    sc = scenario_requests_auth_url()
    manifest = LoopManifest(
        name="test_requests_harness",
        goal=GoalSpec(
            goal_type="repair",
            description="Fix requests auth helper bugs",
            target_file=sc.target_file,
            sources=dict(sc.sources),
            metadata={"scenario": "requests_http_helper"},
            fitness_target=99.7,
        ),
        loop=LoopSpec(
            strategy="greedy_ast_prior",
            budget=LoopBudget(max_evaluations=32, timeout_seconds=10.0),
            first_ascent=True,
            random_seed=42,
        ),
        target=TargetSpec(
            verification_mode="additive_baseline",
            enforce_additive_type_check=False,
            enforce_ast_purity=True,
            distill_trajectory_on_success=True,
        ),
    )

    harness = DeterministicHarness(manifest)
    report = harness.run()

    assert report.status == "SUCCESS"
    assert report.best_score >= 99.7
    assert report.evaluations_consumed > 0
    assert report.verification is not None
    assert report.verification.passed is True

    # Verify distilled workflow was created
    wf = report.distilled_workflow
    assert wf is not None
    assert wf.target_file == sc.target_file
    assert len(wf.actions) >= 1
    assert "dict_access" in wf.defect_signatures
    assert wf.search_evaluations_saved == report.evaluations_consumed

    # Test saving workflow.json
    wf_file = tmp_path / "workflow.json"
    wf.save(wf_file)
    assert wf_file.is_file()

    # Replay distilled workflow on fresh buggy sources with O(1) zero-search complexity
    restored_wf = WorkflowManifest.load(wf_file)
    success, elapsed_ms, repaired = WorkflowExecutor.replay_and_verify(
        sources=sc.sources,
        workflow=restored_wf,
        evaluator=sc.create_evaluator(),
    )
    assert success is True
    assert elapsed_ms < 50.0  # Must be fast deterministic replay (< 50ms)
    assert "query = '&'.join(items)" in repaired[sc.target_file]


def test_llm_context_governor_surgical_pruning_and_budget():
    """Verify LLMContextGovernor manages token budgets and surgical AST pruning."""
    budget = ContextBudget(
        max_context_window_tokens=60,  # Tight token window
        max_session_tokens=300,
        surgical_radius_lines=5,
        max_output_tokens=50,
    )
    governor = LLMContextGovernor(budget=budget)

    # Large code snippet (> 60 tokens)
    large_code = "\n".join([f"def distant_function_{i}():\n    return {i}" for i in range(30)])
    large_code += "\n\ndef target_faulty_func(x):\n    return x - 1\n"
    large_code += "\n".join([f"def after_function_{i}():\n    return {i}" for i in range(30)])

    tokens_before = governor.estimate_tokens(large_code)
    assert tokens_before > 60

    # Prune keeping focus on target_faulty_func (around line 62)
    pruned = governor.surgically_prune_code(large_code, target_lines=[62], max_tokens=60)
    tokens_after = governor.estimate_tokens(pruned)

    assert tokens_after < tokens_before
    assert "target_faulty_func" in pruned
    assert "Context Governor: omitted non-target AST block" in pruned
    assert governor.telemetry.pruning_interventions == 1

    # Test token accounting and budget depletion
    assert governor.can_proceed(estimated_prompt_tokens=50) is True
    governor.record_usage(prompt_tokens=150, completion_tokens=100)  # Total 250 used
    assert governor.remaining_session_tokens == 50

    # Remaining 50 is not enough for 50 prompt + 50 output
    assert governor.can_proceed(estimated_prompt_tokens=50) is False
    assert governor.telemetry.budget_exhausted is True


def test_cli_harness_lifecycle(tmp_path: Path):
    """Verify CLI harness commands: init-manifest -> run -> replay."""
    from evolab.cli import _run_cli

    manifest_file = tmp_path / "cli_loop.json"
    report_file = tmp_path / "cli_report.json"
    wf_file = tmp_path / "cli_workflow.json"
    target_py = tmp_path / "http_helpers.py"

    # 1. init-manifest
    ret_init = _run_cli([
        "harness", "init-manifest",
        "--scenario", "requests_http_helper",
        "-o", str(manifest_file),
    ])
    assert ret_init == 0
    assert manifest_file.is_file()

    # 2. run harness
    ret_run = _run_cli([
        "harness", "run",
        "-m", str(manifest_file),
        "-o", str(report_file),
        "--save-workflow", str(wf_file),
        "--quiet",
    ])
    assert ret_run == 0
    assert report_file.is_file()
    assert wf_file.is_file()

    rep_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert rep_data["status"] == "SUCCESS"
    assert rep_data["best_score"] >= 99.7

    # 3. replay workflow on buggy source
    sc = scenario_requests_auth_url()
    target_py.write_text(sc.sources[sc.target_file], encoding="utf-8")

    ret_replay = _run_cli([
        "harness", "replay",
        "-w", str(wf_file),
        "-s", str(target_py),
        "--apply",
    ])
    assert ret_replay == 0
    fixed_code = target_py.read_text(encoding="utf-8")
    assert "query = '&'.join(items)" in fixed_code


def test_mcp_harness_tools(tmp_path: Path):
    """Verify MCP tools for harness execution, additive verification, and replay."""
    from evolab.mcp.tools import (
        tool_execute_harness,
        tool_verify_additive_baseline,
        tool_replay_workflow,
    )

    sc = scenario_requests_auth_url()
    manifest = LoopManifest.for_repair(
        name="test_mcp_loop",
        sources=sc.sources,
        target_file=sc.target_file,
        max_evals=32,
        scenario_name="requests_http_helper",
    )
    manifest_p = tmp_path / "mcp_manifest.json"
    manifest.save(manifest_p)

    wf_out = tmp_path / "mcp_wf.json"
    res_exec = tool_execute_harness(
        manifest_path=str(manifest_p),
        save_distilled_workflow=True,
        workflow_output_path=str(wf_out),
    )
    assert res_exec["success"] is True
    assert res_exec["status"] == "SUCCESS"
    assert res_exec["verification"]["passed"] is True
    assert wf_out.is_file()

    # Test tool_verify_additive_baseline
    res_verify = tool_verify_additive_baseline(
        candidate_code=sc.sources[sc.target_file],
        target_file=sc.target_file,
        scenario_name="requests_http_helper",
    )
    assert res_verify["success"] is False  # baseline has defect

    # Test tool_replay_workflow
    res_replay = tool_replay_workflow(
        workflow_path=str(wf_out),
        sources=sc.sources,
    )
    assert res_replay["success"] is True
    assert res_replay["actions_applied"] >= 1
    assert "query = '&'.join(items)" in res_replay["repaired_code"]

