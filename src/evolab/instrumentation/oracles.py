"""oracles.py — Security Invariant Oracles and Differential Evaluators."""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .invariants import (
    ExpectedBehaviorInvariant,
    HoldoutGeneralizationInvariant,
    Invariant,
    LatencyDilationInvariant,
    NoEnvironmentIntrospectionInvariant,
    NoMemorySafetyViolationInvariant,
    NoPrivilegeBoundaryViolationInvariant,
    NoResourceExhaustionInvariant,
    NoStateCorruptionInvariant,
    Observation,
    OracleVerdict,
    Target,
    Violation,
    ViolationSeverity,
)
from .taxonomy import FaultCategory


class SecurityOracle:
    """Evaluates an Observation against an invariant suite to generate a security-grounded verdict."""

    def __init__(self, invariants: Sequence[Invariant] | None = None):
        self.invariants: list[Invariant] = list(invariants) if invariants is not None else [
            NoPrivilegeBoundaryViolationInvariant(),
            NoEnvironmentIntrospectionInvariant(),
            NoMemorySafetyViolationInvariant(),
            NoStateCorruptionInvariant(),
            NoResourceExhaustionInvariant(),
            ExpectedBehaviorInvariant(),
            LatencyDilationInvariant(),
            HoldoutGeneralizationInvariant(),
        ]

    def add_invariant(self, invariant: Invariant) -> None:
        self.invariants.append(invariant)

    def evaluate(self, obs: Observation, target: Target | None = None, expected: Any = None) -> OracleVerdict:
        violations: list[Violation] = []
        for inv in self.invariants:
            v = inv.check(obs, target=target, expected=expected)
            if v is not None:
                violations.append(v)

        max_sev = max((v.severity for v in violations), default=ViolationSeverity.NONE)
        # Calculate graded security score: starts at 100, penalized by severity weights
        penalty_table = {
            ViolationSeverity.NONE: 0.0,
            ViolationSeverity.INFO: 2.0,
            ViolationSeverity.LOW: 10.0,
            ViolationSeverity.MEDIUM: 25.0,
            ViolationSeverity.HIGH: 50.0,
            ViolationSeverity.CRITICAL: 100.0,
        }
        total_penalty = sum(penalty_table.get(v.severity, 15.0) for v in violations)
        security_score = max(0.0, 100.0 - total_penalty)
        passed = (len(violations) == 0) or (max_sev <= ViolationSeverity.INFO)

        return OracleVerdict(
            passed=passed,
            security_score=security_score,
            max_severity=max_sev,
            violations=violations,
            observation=obs,
        )


# ===========================================================================
# Pillar C: Differential Execution Engine
# ===========================================================================

@dataclass
class BehavioralDelta:
    """Structured differential analysis between baseline and mutated candidate."""

    baseline_obs: Observation
    candidate_obs: Observation
    functional_divergence: bool
    output_distance: float | None = None
    latency_ratio: float = 1.0
    fault_divergence: bool = False
    side_effect_divergence: bool = False
    classification: str = "IDENTICAL"  # IDENTICAL, EQUIVALENT, REGRESSION, DIVERGENT, CRASH_DIFFERENTIAL, SECURITY_ANOMALY

    def serialize(self) -> dict[str, Any]:
        return {
            "functional_divergence": self.functional_divergence,
            "output_distance": round(self.output_distance, 4) if self.output_distance is not None else None,
            "latency_ratio": round(self.latency_ratio, 3),
            "fault_divergence": self.fault_divergence,
            "side_effect_divergence": self.side_effect_divergence,
            "classification": self.classification,
            "baseline_fault": self.baseline_obs.fault_category.value,
            "candidate_fault": self.candidate_obs.fault_category.value,
        }

    def is_benign(self) -> bool:
        return self.classification in ("IDENTICAL", "EQUIVALENT", "PERFORMANCE_IMPROVEMENT")


class DifferentialExecutor:
    """Executes baseline vs candidate across shared inputs and computes behavioral deltas."""

    def __init__(self, oracle: SecurityOracle | None = None):
        self.oracle = oracle or SecurityOracle()

    def compare_observations(self, base_obs: Observation, cand_obs: Observation) -> BehavioralDelta:
        # 1. Check functional divergence
        base_ret = base_obs.return_value
        cand_ret = cand_obs.return_value
        if isinstance(base_ret, dict) and "return_value" in base_ret:
            base_ret = base_ret["return_value"]
        if isinstance(cand_ret, dict) and "return_value" in cand_ret:
            cand_ret = cand_ret["return_value"]

        func_div = (base_ret != cand_ret)
        out_dist: float | None = None
        if isinstance(base_ret, (int, float)) and isinstance(cand_ret, (int, float)):
            out_dist = abs(float(base_ret) - float(cand_ret))

        # 2. Check latency ratio
        base_dur = max(base_obs.duration_ms, 0.001)
        cand_dur = max(cand_obs.duration_ms, 0.001)
        latency_ratio = cand_dur / base_dur
        if cand_obs.metadata is not None:
            cand_obs.metadata["latency_ratio"] = latency_ratio
        else:
            cand_obs.metadata = {"latency_ratio": latency_ratio}

        # 3. Fault & side-effect divergence
        fault_div = (base_obs.fault_category != cand_obs.fault_category)
        base_state = base_obs.state_diff.get("changed", []) + base_obs.state_diff.get("added", [])
        cand_state = cand_obs.state_diff.get("changed", []) + cand_obs.state_diff.get("added", [])
        side_div = (base_state != cand_state) or (len(base_obs.telemetry_events) != len(cand_obs.telemetry_events))

        # 4. Classify differential
        if cand_obs.metadata.get("environment_divergence") or cand_obs.metadata.get("dual_context_divergent"):
            classification = "ENVIRONMENT_AWARE_DIVERGENCE"
        elif cand_obs.fault_category in (FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION, FaultCategory.STATE_CORRUPTION):
            classification = "SECURITY_ANOMALY"
        elif cand_obs.fault_category in (FaultCategory.MEMORY_SAFETY_SIGNAL, FaultCategory.UNEXPECTED_TERMINATION):
            classification = "CRASH_DIFFERENTIAL"
        elif fault_div:
            classification = "REGRESSION" if base_obs.is_clean() else "FAULT_DIVERGENCE"
        elif func_div:
            classification = "DIVERGENT"
        elif side_div:
            classification = "SIDE_EFFECT_ANOMALY"
        elif latency_ratio < 0.85:
            classification = "PERFORMANCE_IMPROVEMENT"
        elif latency_ratio > 1.4:
            classification = "PERFORMANCE_REGRESSION"
        else:
            classification = "EQUIVALENT" if not (base_obs.stdout == cand_obs.stdout and base_obs.stderr == cand_obs.stderr) else "IDENTICAL"

        return BehavioralDelta(
            baseline_obs=base_obs,
            candidate_obs=cand_obs,
            functional_divergence=func_div,
            output_distance=out_dist,
            latency_ratio=latency_ratio,
            fault_divergence=fault_div,
            side_effect_divergence=side_div,
            classification=classification,
        )

    def execute_single(
        self,
        baseline: Target,
        candidate: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        runner: Any = None,
    ) -> BehavioralDelta:
        """Executes a single test case differentially on baseline and candidate targets."""
        from ..sandbox import SandboxRunner

        runner = runner or SandboxRunner()
        kwargs = kwargs or {}

        base_res = runner.run_function(baseline.sources, baseline.target_file, baseline.func_name, args, kwargs)
        cand_res = runner.run_function(candidate.sources, candidate.target_file, candidate.func_name, args, kwargs)

        base_obs = Observation.from_execution_result(baseline.name, base_res)
        cand_obs = Observation.from_execution_result(candidate.name, cand_res)

        return self.compare_observations(base_obs, cand_obs)

    def execute_dual_context(
        self,
        candidate: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        runner_a: Any = None,
        runner_b: Any = None,
    ) -> BehavioralDelta:
        """Executes candidate across two differentially configured sandbox/mirrored contexts.

        Detects environment-awareness and anti-sandbox heuristics by verifying that candidate
        behavior remains identical whether running in strict sandbox or host-mirrored context.
        """
        from ..sandbox import SandboxConfig, SandboxRunner

        kwargs = kwargs or {}
        # Context A: Default strict sandbox
        runner_a = runner_a or SandboxRunner(SandboxConfig(allow_network=False))
        # Context B: Mirrored host simulation
        runner_b = runner_b or SandboxRunner(SandboxConfig(allow_network=True))

        res_a = runner_a.run_function(candidate.sources, candidate.target_file, candidate.func_name, args, kwargs)
        res_b = runner_b.run_function(candidate.sources, candidate.target_file, candidate.func_name, args, kwargs)

        obs_a = Observation.from_execution_result(f"{candidate.name}_Sandbox", res_a)
        obs_b = Observation.from_execution_result(f"{candidate.name}_Mirrored", res_b)

        delta = self.compare_observations(obs_a, obs_b)
        if delta.functional_divergence:
            obs_b.metadata["environment_divergence"] = True
            delta.classification = "ENVIRONMENT_AWARE_DIVERGENCE"

        return delta


# ===========================================================================
# Invariant-Driven Evolutionary Evaluators (Connecting Oracle to Selection)
# ===========================================================================

class SecurityEvaluator:
    """Evaluator that couples SecurityOracle and Invariants directly into fitness for evolutionary selection.

    Modes:
      - 'hardening' (Immune / Protective):
          Selection penalizes invariant violations. Candidates must solve the task while
          strictly obeying security invariants.
          Fitness = FunctionalScore * (SecurityScore / 100.0)

      - 'discovery' (Adversarial / Red-Teaming / Fuzzing):
          Selection rewards invariant violations. Evolution actively searches for inputs
          or mutations that trigger high-severity faults and security breaches.
          Fitness = (MaxSeverityScore * 20.0) + (100.0 - SecurityScore) * 0.5
    """

    def __init__(
        self,
        base_sources: dict[str, str],
        target_file: str = "target.py",
        func_name: str = "solve",
        test_cases: Sequence[tuple[tuple[Any, ...], Any]] | None = None,
        oracle: SecurityOracle | None = None,
        mode: str = "hardening",
        config: Any = None,
        holdout_test_cases: Sequence[tuple[tuple[Any, ...], Any]] | None = None,
    ):
        from ..sandbox import SandboxConfig, SandboxRunner

        self.base_sources = dict(base_sources)
        self.target_file = target_file
        self.func_name = func_name
        self.test_cases = list(test_cases or [])
        self.holdout_test_cases = list(holdout_test_cases or [])
        self.oracle = oracle or SecurityOracle()
        self.mode = mode.lower()  # "hardening" or "discovery"
        self.config = config or SandboxConfig()
        self.runner = SandboxRunner(self.config)

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> Any:
        from ..evaluators import FitnessResult
        from ..genome import Individual

        t0 = time.perf_counter()
        patch = target.genome if isinstance(target, Individual) else target

        try:
            if hasattr(patch, "apply_to"):
                applied = patch.apply_to(self.base_sources)
            elif hasattr(patch, "to_code"):
                applied = dict(self.base_sources)
                applied[self.target_file] = patch.to_code()
            elif isinstance(patch, str):
                applied = dict(self.base_sources)
                applied[self.target_file] = patch
            else:
                applied = dict(self.base_sources)
        except Exception as err:
            duration = (time.perf_counter() - t0) * 1000.0
            return FitnessResult(
                score=0.0,
                sub_scores={"compiled": 0.0, "security_score": 0.0},
                artifacts={"error": str(err)},
                evaluation_time_ms=duration,
            )

        # Check static AST purity
        candidate_code = applied.get(self.target_file, "")
        purity_violations = []
        if candidate_code:
            from ..ast_genome import validate_code_purity
            _, purity_violations = validate_code_purity(candidate_code)

        # Run test suite or single function
        exec_res = self.runner.run_test_suite(
            sources=applied,
            target_file=self.target_file,
            func_name=self.func_name,
            test_cases=self.test_cases,
        )

        obs = Observation.from_execution_result("candidate", exec_res)
        if purity_violations:
            if obs.metadata is not None:
                obs.metadata["purity_violations"] = purity_violations
            else:
                obs.metadata = {"purity_violations": purity_violations}

        # Check holdout generalization test cases to detect semantic overfitting
        holdout_ratio = 1.0
        if self.holdout_test_cases and exec_res.success:
            holdout_res = self.runner.run_test_suite(
                sources=applied,
                target_file=self.target_file,
                func_name=self.func_name,
                test_cases=self.holdout_test_cases,
            )
            h_val = holdout_res.return_value if isinstance(holdout_res.return_value, dict) else {}
            h_passed = h_val.get("tests_passed", 1 if holdout_res.success else 0)
            h_total = len(self.holdout_test_cases)
            holdout_ratio = h_passed / h_total if h_total > 0 else 1.0
            if obs.metadata is None:
                obs.metadata = {}
            obs.metadata["holdout_ratio"] = holdout_ratio
            obs.metadata["holdout_failures"] = h_total - h_passed
            obs.metadata["holdout_total"] = h_total

        expected_output = self.test_cases[0][1] if self.test_cases else None
        verdict = self.oracle.evaluate(obs, expected=expected_output)

        val = exec_res.return_value if isinstance(exec_res.return_value, dict) else {}
        tests_passed = val.get("tests_passed", 1 if exec_res.success else 0)
        total_tests = max(len(self.test_cases), 1)
        functional_ratio = tests_passed / total_tests
        functional_score = 20.0 + 80.0 * functional_ratio if exec_res.success else 0.0

        sev_score = int(verdict.max_severity)  # 0 to 5

        if self.mode == "discovery":
            # Evolutionary goal: Maximize invariant violations (find security holes / crashes)
            discovery_table = {0: 0.0, 1: 15.0, 2: 35.0, 3: 60.0, 4: 85.0, 5: 100.0}
            score = discovery_table.get(sev_score, 0.0)
        else:
            # Hardening mode: Maximize functional correctness WHILE maintaining security & holdout generalization
            if sev_score >= ViolationSeverity.HIGH or holdout_ratio == 0.0:
                score = 0.0
            else:
                security_multiplier = verdict.security_score / 100.0
                score = functional_score * (holdout_ratio ** 2) * security_multiplier

        duration = (time.perf_counter() - t0) * 1000.0
        return FitnessResult(
            score=round(score, 2),
            sub_scores={
                "functional_score": round(functional_score, 2),
                "security_score": round(verdict.security_score, 2),
                "max_severity": float(sev_score),
            },
            artifacts={
                "verdict": verdict.serialize(),
                "fault_category": obs.fault_category.value,
                "violation_count": len(verdict.violations),
            },
            evaluation_time_ms=duration,
        )

    def __call__(self, target: Any) -> float:
        return self.evaluate(target).score


class DifferentialEvaluator:
    """Evaluator that uses DifferentialExecutor against a baseline target to drive selection."""

    def __init__(
        self,
        baseline_target: Target,
        target_file: str = "target.py",
        func_name: str = "solve",
        test_cases: Sequence[tuple[tuple[Any, ...], Any]] | None = None,
        mode: str = "discovery",  # "discovery" (find deviations) or "regression" (maintain fidelity)
        config: Any = None,
    ):
        from ..sandbox import SandboxConfig, SandboxRunner

        self.baseline_target = baseline_target
        self.target_file = target_file
        self.func_name = func_name
        self.test_cases = list(test_cases or [])
        self.mode = mode.lower()
        self.config = config or SandboxConfig()
        self.runner = SandboxRunner(self.config)
        self.diff_exec = DifferentialExecutor()

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> Any:
        from ..evaluators import FitnessResult
        from ..genome import Individual

        t0 = time.perf_counter()
        patch = target.genome if isinstance(target, Individual) else target

        try:
            if hasattr(patch, "apply_to"):
                applied = patch.apply_to(self.baseline_target.sources)
            elif hasattr(patch, "to_code"):
                applied = dict(self.baseline_target.sources)
                applied[self.target_file] = patch.to_code()
            elif isinstance(patch, str):
                applied = dict(self.baseline_target.sources)
                applied[self.target_file] = patch
            else:
                applied = dict(self.baseline_target.sources)
        except Exception as err:
            duration = (time.perf_counter() - t0) * 1000.0
            return FitnessResult(
                score=0.0,
                sub_scores={"compiled": 0.0},
                artifacts={"error": str(err)},
                evaluation_time_ms=duration,
            )

        cand_target = Target(name="candidate", sources=applied, target_file=self.target_file, func_name=self.func_name)

        divergences = 0
        total = max(len(self.test_cases), 1)
        deltas = []
        max_dist = 0.0

        for args, expected in self.test_cases:
            delta = self.diff_exec.execute_single(self.baseline_target, cand_target, args=args, runner=self.runner)
            deltas.append(delta)
            if delta.functional_divergence or delta.fault_divergence:
                divergences += 1
            if delta.output_distance is not None:
                max_dist = max(max_dist, delta.output_distance)

        div_ratio = divergences / total

        if self.mode == "discovery":
            # Reward behavioral divergence from baseline
            score = round(div_ratio * 70.0 + min(30.0, max_dist * 3.0), 2)
        else:
            # Regression mode: reward fidelity to baseline
            score = round((1.0 - div_ratio) * 100.0, 2)

        duration = (time.perf_counter() - t0) * 1000.0
        return FitnessResult(
            score=score,
            sub_scores={"divergence_ratio": round(div_ratio * 100.0, 2), "max_distance": round(max_dist, 2)},
            artifacts={"deltas": [d.serialize() for d in deltas]},
            evaluation_time_ms=duration,
        )

    def __call__(self, target: Any) -> float:
        return self.evaluate(target).score


