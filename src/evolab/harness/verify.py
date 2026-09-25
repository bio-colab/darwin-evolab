"""verify.py — Additive Baseline Verification Gate for Darwin-Evolab.

Implements the 2026 Modern Harness Standard for deterministic verification:
- Baseline-scoped gating: Evaluates candidate code strictly against the baseline delta.
- FAIL_TO_PASS: Verifies that target defect tests are cleanly resolved.
- PASS_TO_PASS: Guarantees zero regressions on preexisting passing tests.
- Additive Static Analysis: Enforces Delta_errors = E_candidate - E_baseline = empty.
  Preexisting codebase type debt or warnings do NOT cause candidate rejection.
- AST Purity Guard: Enforces zero illegal shell calls, dynamic exec, or evasive constructs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Callable

from evolab.ast_genome import validate_code_purity
from evolab.evaluators import Evaluator, FitnessResult
from evolab.patch import PatchGenome
from evolab.type_gate import EliteTypeCheckGate, TypeCheckResult


@dataclass
class BaselineSnapshot:
    """Captured baseline state of the target codebase before optimization."""

    code_hash: str
    passing_tests: list[str] = field(default_factory=list)
    failing_tests: list[str] = field(default_factory=list)
    type_errors: list[str] = field(default_factory=list)
    fitness_score: float = 0.0
    ast_pure: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdditiveVerificationReport:
    """Complete diagnostic report for additive baseline verification."""

    passed: bool
    verdict: str  # "ACCEPT" | "REJECT"
    fail_to_pass_resolved: list[str] = field(default_factory=list)
    pass_to_pass_preserved: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    baseline_type_error_count: int = 0
    candidate_type_error_count: int = 0
    new_type_errors: list[str] = field(default_factory=list)
    cured_type_errors: list[str] = field(default_factory=list)
    ast_violations: list[str] = field(default_factory=list)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_markdown(self) -> str:
        icon = "[PASS]" if self.passed else "[FAIL]"
        lines = [
            f"# Additive Baseline Verification Report {icon}",
            f"- **Verdict**: `{self.verdict}`",
            f"- **Candidate Score**: `{self.score:.2f}`",
            f"- **FAIL_TO_PASS Resolved**: `{len(self.fail_to_pass_resolved)}` tests",
            f"- **PASS_TO_PASS Preserved**: `{len(self.pass_to_pass_preserved)}` tests",
            f"- **Regressions**: `{len(self.regressions)}`",
            f"- **Additive Type Errors Introduced**: `{len(self.new_type_errors)}` (Preexisting: {self.baseline_type_error_count})",
            f"- **Preexisting Type Errors Cured**: `{len(self.cured_type_errors)}`",
            f"- **AST Purity Violations**: `{len(self.ast_violations)}`",
            f"- **Reasons**: {', '.join(self.reasons) if self.reasons else 'None'}",
        ]
        return "\n".join(lines)


class AdditiveBaselineGate:
    """Verifies candidates using baseline-scoped delta evaluation."""

    def __init__(
        self,
        type_check_gate: EliteTypeCheckGate | None = None,
        enforce_purity: bool = True,
        enforce_type_check: bool = True,
    ) -> None:
        self.type_check_gate = type_check_gate or EliteTypeCheckGate(enabled=True)
        self.enforce_purity = enforce_purity
        self.enforce_type_check = enforce_type_check

    def capture_baseline(
        self,
        sources: dict[str, str],
        target_file: str,
        evaluator: Evaluator | None = None,
    ) -> BaselineSnapshot:
        """Executes verification on baseline sources to record existing errors."""
        code = sources.get(target_file, "")
        code_h = hashlib.sha256(code.encode("utf-8")).hexdigest()

        # 1. AST purity check
        ast_pure, _ = validate_code_purity(code)

        # 2. Type errors
        type_errs: list[str] = []
        if self.enforce_type_check:
            t_res = self.type_check_gate.check_code(code)
            type_errs = list(t_res.errors)

        # 3. Test execution
        passing: list[str] = []
        failing: list[str] = []
        score = 0.0

        if evaluator:
            try:
                base_genome = PatchGenome()
                res = evaluator.evaluate(base_genome)
                score = res.score
                artifacts = res.artifacts or {}
                raw_passing = artifacts.get("passing_tests", [])
                if isinstance(raw_passing, list):
                    passing = [str(x) for x in raw_passing]
                raw_failing = artifacts.get("failing_tests") or artifacts.get("failures") or []
                if isinstance(raw_failing, list):
                    failing = [str(x) for x in raw_failing]

                if not passing and not failing:
                    # In standard scenarios with score < 100, record default failure marker
                    if score < 99.7:
                        failing.append("target_defect_assertion")
                    else:
                        passing.append("target_defect_assertion")
            except Exception as exc:
                failing.append(f"baseline_exception: {exc}")

        return BaselineSnapshot(
            code_hash=code_h,
            passing_tests=passing,
            failing_tests=failing,
            type_errors=type_errs,
            fitness_score=score,
            ast_pure=ast_pure,
        )

    def verify(
        self,
        candidate_sources: dict[str, str],
        target_file: str,
        baseline: BaselineSnapshot,
        evaluator: Evaluator | None = None,
        candidate_genome: Any = None,
    ) -> AdditiveVerificationReport:
        """Verifies candidate code against the captured baseline."""
        code = candidate_sources.get(target_file, "")
        reasons: list[str] = []

        # 1. AST Purity Gate
        violations: list[str] = []
        if self.enforce_purity:
            is_pure, v_list = validate_code_purity(code)
            if not is_pure:
                violations.extend(v_list)
                reasons.append("ast_purity_violation")

        # 2. Additive Static Type Analysis Gate
        cand_type_errors: list[str] = []
        new_type_errors: list[str] = []
        cured_type_errors: list[str] = []

        if self.enforce_type_check:
            t_res = self.type_check_gate.check_code(code)
            cand_type_errors = list(t_res.errors)

            base_set = set(baseline.type_errors)
            cand_set = set(cand_type_errors)

            # Additive scoping: only errors in candidate NOT present in baseline
            new_type_errors = sorted(list(cand_set - base_set))
            cured_type_errors = sorted(list(base_set - cand_set))

            if new_type_errors:
                reasons.append(f"new_type_errors_introduced: {len(new_type_errors)}")

        # 3. Dynamic Evaluation & Regression Gate
        cand_passing: list[str] = []
        cand_failing: list[str] = []
        cand_score = 0.0
        resolved_tests: list[str] = []
        regressions: list[str] = []
        preserved_tests: list[str] = []

        if evaluator:
            target_eval = candidate_genome if candidate_genome is not None else PatchGenome()
            try:
                res = evaluator.evaluate(target_eval)
                cand_score = res.score
                artifacts = res.artifacts or {}
                raw_passing = artifacts.get("passing_tests", [])
                if isinstance(raw_passing, list):
                    cand_passing = [str(x) for x in raw_passing]
                raw_failing = artifacts.get("failing_tests") or artifacts.get("failures") or []
                if isinstance(raw_failing, list):
                    cand_failing = [str(x) for x in raw_failing]

                if not cand_passing and not cand_failing:
                    if cand_score >= 99.7:
                        cand_passing.append("target_defect_assertion")
                    else:
                        cand_failing.append("target_defect_assertion")

                # Compute FAIL_TO_PASS (resolved) and regressions
                base_failing_set = set(baseline.failing_tests)
                base_passing_set = set(baseline.passing_tests)
                cand_passing_set = set(cand_passing)
                cand_failing_set = set(cand_failing)

                if cand_score >= 99.7:
                    for f in base_failing_set:
                        if f not in cand_failing_set:
                            cand_passing_set.add(f)

                resolved_tests = sorted(list(base_failing_set.intersection(cand_passing_set)))
                preserved_tests = sorted(list(base_passing_set.intersection(cand_passing_set)))
                regressions = sorted(list(base_passing_set.intersection(cand_failing_set)))

                if regressions:
                    reasons.append(f"regressions_detected: {len(regressions)}")

                if cand_score < 99.7 and not resolved_tests:
                    reasons.append("target_defect_not_resolved")

            except Exception as exc:
                reasons.append(f"evaluation_exception: {exc}")

        # Final Verdict
        passed = (
            len(violations) == 0
            and len(new_type_errors) == 0
            and len(regressions) == 0
            and (cand_score >= 99.7 or len(resolved_tests) > 0)
        )
        verdict = "ACCEPT" if passed else "REJECT"
        if not reasons:
            reasons.append("all_additive_gates_passed")

        return AdditiveVerificationReport(
            passed=passed,
            verdict=verdict,
            fail_to_pass_resolved=resolved_tests,
            pass_to_pass_preserved=preserved_tests,
            regressions=regressions,
            baseline_type_error_count=len(baseline.type_errors),
            candidate_type_error_count=len(cand_type_errors),
            new_type_errors=new_type_errors,
            cured_type_errors=cured_type_errors,
            ast_violations=violations,
            score=cand_score,
            reasons=reasons,
        )
