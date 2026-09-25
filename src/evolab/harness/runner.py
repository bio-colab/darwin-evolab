"""runner.py — Deterministic Harness Execution Engine for Darwin-Evolab.

Implements the 2026 Modern Harness Standard:
Drives a declarative LoopManifest through deterministic lifecycle phases:
1. INIT & BASELINE_CAPTURE: Records preexisting errors, passing tests, and type state.
2. SEARCH_LOOP: Runs governed evolutionary search within explicit budget bounds.
3. ADDITIVE_VERIFICATION: Validates candidate against baseline-scoped additive gates.
4. DISTILLATION: Transforms winning solution into reusable O(1) workflow.json.
5. REPORT: Emits structured execution report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from evolab.code_fixtures import SCENARIO_REGISTRY
from evolab.evaluators import Evaluator, FunctionTestEvaluator

from evolab.harness.distillation import (
    WorkflowExecutor,
    WorkflowManifest,
    distill_trajectory_to_workflow,
)
from evolab.harness.manifest import LoopManifest
from evolab.harness.verify import (
    AdditiveBaselineGate,
    AdditiveVerificationReport,
    BaselineSnapshot,
)
from evolab.patch import PatchGenome
from evolab.repair import greedy_repair


@dataclass
class HarnessExecutionReport:
    """Complete structured execution report produced by DeterministicHarness."""

    manifest_name: str
    status: str  # "SUCCESS" | "FAILED" | "REJECTED_BY_GATE" | "TIMEOUT"
    evaluations_consumed: int = 0
    duration_ms: float = 0.0
    best_score: float = 0.0
    fixed_sources: dict[str, str] = field(default_factory=dict)
    baseline_snapshot: BaselineSnapshot | None = None
    verification: AdditiveVerificationReport | None = None
    distilled_workflow: WorkflowManifest | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_name": self.manifest_name,
            "status": self.status,
            "evaluations_consumed": self.evaluations_consumed,
            "duration_ms": self.duration_ms,
            "best_score": self.best_score,
            "fixed_sources": self.fixed_sources,
            "baseline_snapshot": self.baseline_snapshot.to_dict() if self.baseline_snapshot else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "distilled_workflow": self.distilled_workflow.to_dict() if self.distilled_workflow else None,
            "telemetry": self.telemetry,
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class DeterministicHarness:
    """Production harness driving declarative LoopManifests deterministically."""

    def __init__(self, manifest: LoopManifest) -> None:
        self.manifest = manifest
        self.verify_gate = AdditiveBaselineGate(
            enforce_purity=manifest.target.enforce_ast_purity,
            enforce_type_check=manifest.target.enforce_additive_type_check,
        )

    def _build_evaluator(self) -> tuple[Evaluator, dict[str, str], str]:
        """Resolves evaluator and sources from the goal specification."""
        goal = self.manifest.goal
        target_file = goal.target_file
        sources = dict(goal.sources)

        # 1. If scenario name specified in metadata
        scenario_name = goal.metadata.get("scenario")
        if scenario_name and scenario_name in SCENARIO_REGISTRY:
            sc = SCENARIO_REGISTRY[scenario_name]()
            return sc.create_evaluator(), dict(sc.sources), sc.target_file

        # 2. If test_cases provided
        if goal.test_cases:
            func_name = goal.target_function or "solution"
            test_cases_tuples = [
                (tuple(tc.get("args", [])), tc.get("expected"))
                for tc in goal.test_cases
            ]
            evaluator = FunctionTestEvaluator(
                base_sources=sources,
                target_file=target_file,
                func_name=func_name,
                test_cases=test_cases_tuples,
            )
            return evaluator, sources, target_file

        # 3. If test_code containing assert statements provided
        if goal.test_code:
            evaluator = self._build_assert_evaluator(sources, target_file, goal.test_code)
            return evaluator, sources, target_file

        # Fallback dummy evaluator
        class DummyEvaluator(Evaluator):
            def evaluate(self, target: Any, context: Any = None):
                from evolab.evaluators import FitnessResult
                return FitnessResult(score=100.0, artifacts={"resolved": True})

        return DummyEvaluator(), sources, target_file

    def _build_assert_evaluator(
        self, sources: dict[str, str], target_file: str, test_code: str
    ) -> Evaluator:
        from evolab.evaluators import FitnessResult
        from evolab.patch import apply_patch

        class AssertEvaluator(Evaluator):
            def evaluate(self, target: Any, context: Any = None) -> FitnessResult:
                current_sources = dict(sources)
                if isinstance(target, PatchGenome):
                    current_sources = apply_patch(current_sources, target)
                elif hasattr(target, "sources"):
                    current_sources = dict(target.sources)

                code = current_sources.get(target_file, "")
                combined = f"{code}\n\n# --- Test Harness Assertions ---\n{test_code}\n"
                ns: dict[str, Any] = {}
                try:
                    exec(combined, ns, ns)  # nosec B102
                    return FitnessResult(
                        score=100.0,
                        artifacts={
                            "resolved": True,
                            "passing_tests": ["test_asserts"],
                            "failing_tests": [],
                        },
                    )
                except AssertionError as exc:
                    return FitnessResult(
                        score=25.0,
                        artifacts={
                            "resolved": False,
                            "passing_tests": [],
                            "failing_tests": [f"assertion_failed: {exc}"],
                        },
                    )
                except Exception as exc:
                    return FitnessResult(
                        score=0.0,
                        artifacts={
                            "resolved": False,
                            "passing_tests": [],
                            "failing_tests": [f"runtime_error: {exc}"],
                        },
                    )

        return AssertEvaluator()

    def run(self) -> HarnessExecutionReport:
        """Executes the complete deterministic harness lifecycle."""
        t0 = time.perf_counter()
        evaluator, sources, target_file = self._build_evaluator()

        # Phase 1: Capture Additive Baseline Snapshot
        baseline_snapshot = self.verify_gate.capture_baseline(
            sources=sources,
            target_file=target_file,
            evaluator=evaluator,
        )

        # Phase 2: Execute Optimization Search Loop
        budget = self.manifest.loop.budget
        winning_genome = None
        evals_consumed = 0
        best_score = baseline_snapshot.fitness_score

        if self.manifest.goal.goal_type == "repair":
            win_repair, history, evals_consumed = greedy_repair(
                sources=sources,
                target_file=target_file,
                evaluator=evaluator,
                max_evals=budget.max_evaluations,
                first_ascent=self.manifest.loop.first_ascent,
            )
            winning_genome = win_repair
            if hasattr(win_repair, "apply_to"):
                fixed_sources = win_repair.apply_to()
            else:
                fixed_sources = dict(win_repair.sources)
            if history:
                best_score = float(history[-1].get("best_fitness", baseline_snapshot.fitness_score))
            else:
                best_score = baseline_snapshot.fitness_score
        else:
            fixed_sources = dict(sources)

        # Phase 3: Additive Baseline Verification
        verification = self.verify_gate.verify(
            candidate_sources=fixed_sources,
            target_file=target_file,
            baseline=baseline_snapshot,
            evaluator=evaluator,
            candidate_genome=winning_genome,
        )

        status = "SUCCESS" if verification.passed else "REJECTED_BY_GATE"

        # Phase 4: Trajectory Distillation to Reusable Workflow
        distilled_wf = None
        if verification.passed and self.manifest.target.distill_trajectory_on_success:
            distilled_wf = distill_trajectory_to_workflow(
                initial_sources=sources,
                fixed_sources=fixed_sources,
                target_file=target_file,
                workflow_name=f"Workflow: {self.manifest.name}",
                search_evaluations_consumed=evals_consumed,
            )

        dur_ms = (time.perf_counter() - t0) * 1000.0

        return HarnessExecutionReport(
            manifest_name=self.manifest.name,
            status=status,
            evaluations_consumed=evals_consumed,
            duration_ms=round(dur_ms, 2),
            best_score=round(best_score, 2),
            fixed_sources=fixed_sources,
            baseline_snapshot=baseline_snapshot,
            verification=verification,
            distilled_workflow=distilled_wf,
            telemetry={
                "strategy": self.manifest.loop.strategy,
                "evaluations_budget": budget.max_evaluations,
                "verification_mode": self.manifest.target.verification_mode,
            },
        )
