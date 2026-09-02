"""Evaluator contract and concrete evaluators for numerical and code evolution."""
from __future__ import annotations

import sys
import time
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .genome import EvolabGenome, Individual
from .patch import PatchGenome


@dataclass
class FitnessResult:
    """Standard evaluation result across any fitness landscape."""

    score: float                         # [0.0, 100.0]
    sub_scores: dict[str, float] = field(default_factory=dict)
    passed_holdout: bool | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    evaluation_time_ms: float = 0.0

    def serialize(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
            "passed_holdout": self.passed_holdout,
            "artifacts": self.artifacts,
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
        }


class Evaluator(ABC):
    """Standard evaluator interface."""

    @abstractmethod
    def evaluate(
        self, target: EvolabGenome | Individual | list[float], context: dict[str, Any] | None = None
    ) -> FitnessResult:
        """Evaluates target and returns a FitnessResult."""

    @property
    @abstractmethod
    def deterministic(self) -> bool:
        """True if same input yields identical results."""

    @property
    def cost_estimate(self) -> str:
        """'free' | 'cheap' | 'expensive'"""
        return "cheap"

    def __call__(self, target: EvolabGenome | Individual | list[float]) -> float:
        """Convenience method to allow Evaluator to be passed as standard fitness_fn."""
        res = self.evaluate(target)
        return res.score


class NumericEvaluator(Evaluator):
    """Wraps an existing numeric callable fitness_fn."""

    def __init__(self, fn: Callable[[Any], float], name: str = "numeric_landscape"):
        self.fn = fn
        self.name = name

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def evaluate(
        self, target: EvolabGenome | Individual | list[float], context: dict[str, Any] | None = None
    ) -> FitnessResult:
        t0 = time.perf_counter()
        score = float(self.fn(target))
        duration = (time.perf_counter() - t0) * 1000.0
        return FitnessResult(
            score=score,
            sub_scores={"numeric_fitness": score},
            evaluation_time_ms=duration,
        )


class CompileCheckEvaluator(Evaluator):
    """Evaluates whether a PatchGenome produces syntactically valid and compilable code."""

    def __init__(self, base_sources: dict[str, str]):
        self.base_sources = base_sources

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def evaluate(
        self, target: EvolabGenome | Individual | list[float], context: dict[str, Any] | None = None
    ) -> FitnessResult:
        t0 = time.perf_counter()
        patch = target.genome if isinstance(target, Individual) else target
        if not isinstance(patch, PatchGenome):
            raise TypeError(f"CompileCheckEvaluator expects PatchGenome, got {type(patch)}")

        try:
            applied = patch.apply_to(self.base_sources)
        except Exception as err:
            duration = (time.perf_counter() - t0) * 1000.0
            return FitnessResult(
                score=0.0,
                sub_scores={"applied": 0.0, "compiled": 0.0},
                artifacts={"error": f"Patch application error: {err}"},
                evaluation_time_ms=duration,
            )

        compiled_count = 0
        total_files = len(applied)
        errors = {}

        for file_path, code in applied.items():
            try:
                compile(code, file_path, "exec")
                compiled_count += 1
            except SyntaxError as e:
                errors[file_path] = f"SyntaxError at line {e.lineno}: {e.msg}"

        compilation_ratio = (compiled_count / total_files) if total_files > 0 else 1.0
        score = round(compilation_ratio * 100.0, 2)
        duration = (time.perf_counter() - t0) * 1000.0

        return FitnessResult(
            score=score,
            sub_scores={"applied": 100.0, "compiled": score},
            artifacts={"errors": errors, "compiled_files": compiled_count, "total_files": total_files},
            evaluation_time_ms=duration,
        )


class FunctionTestEvaluator(Evaluator):
    """Evaluates a PatchGenome by compiling the source code and executing unit test cases."""

    def __init__(
        self,
        base_sources: dict[str, str],
        target_file: str,
        func_name: str,
        test_cases: Sequence[tuple[tuple[Any, ...], Any]],
        holdout_cases: Sequence[tuple[tuple[Any, ...], Any]] | None = None,
        timeout_seconds: float = 1.0,
    ):
        self.base_sources = base_sources
        self.target_file = target_file
        self.func_name = func_name
        self.test_cases = list(test_cases)
        self.holdout_cases = list(holdout_cases) if holdout_cases is not None else None
        self.timeout_seconds = timeout_seconds
        self.last_suspicion_map = None

    #: Observable state that ``evaluate`` mutates as a side effect. The
    #: evaluation cache (evolab.experience.EvaluationCache) snapshots these
    #: attributes on a miss and restores them on a hit so a cached result is
    #: indistinguishable from a fresh one — including for the SBFL-narrowed
    #: mutator, which reads ``last_suspicion_map`` AFTER the evaluation.
    cacheable_state_attrs = ("last_suspicion_map",)

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def _run_cases(
        self, func: Callable, cases: list[tuple[tuple[Any, ...], Any]]
    ) -> tuple[int, int, list[str]]:
        passed = 0
        details = []
        for args, expected in cases:
            try:
                res = func(*args)
                if res == expected:
                    passed += 1
                else:
                    details.append(f"Expected {expected}, got {res} for args {args}")
            except Exception as e:
                details.append(f"Exception {type(e).__name__}: {e} for args {args}")
        return passed, len(cases), details

    def _refresh_suspicion(self, code: str, func: Callable) -> None:
        try:
            from .suspicion import LineCoverageTracer, build_suspicion_map
            runs = []
            for args, expected in self.test_cases:
                tracer = LineCoverageTracer(self.target_file)
                with tracer:
                    try:
                        ok = func(*args) == expected
                    except Exception:
                        ok = False
                runs.append((set(tracer.executed_lines), bool(ok)))
            self.last_suspicion_map = build_suspicion_map(
                code, runs, target_func=self.func_name
            )
        except Exception:
            self.last_suspicion_map = None

    def evaluate(
        self, target: EvolabGenome | Individual | list[float], context: dict[str, Any] | None = None
    ) -> FitnessResult:
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
                raise TypeError(f"FunctionTestEvaluator expects PatchGenome or ASTGenome, got {type(patch)}")
            
            injected_mods = []
            try:
                for other_path, other_code in applied.items():
                    if other_path != self.target_file:
                        mod_name = other_path.removesuffix(".py")
                        other_compiled = compile(other_code, other_path, "exec")
                        other_ns: dict[str, Any] = {}
                        exec(other_compiled, other_ns)  # nosec B102  # In-process execution intentional for trusted benchmarks; untrusted code uses SandboxFunctionTestEvaluator.
                        mod = types.ModuleType(mod_name)
                        mod.__dict__.update(other_ns)
                        sys.modules[mod_name] = mod
                        injected_mods.append(mod_name)

                code = applied[self.target_file]
                compiled = compile(code, self.target_file, "exec")
                namespace: dict[str, Any] = {}
                exec(compiled, namespace)  # nosec B102  # In-process execution intentional for trusted benchmarks; untrusted code uses SandboxFunctionTestEvaluator.
                if self.func_name not in namespace:
                    duration = (time.perf_counter() - t0) * 1000.0
                    return FitnessResult(
                        score=10.0,
                        sub_scores={"compiled": 100.0, "found_func": 0.0, "tests_passed": 0.0},
                        artifacts={"error": f"Function {self.func_name} not found in namespace"},
                        evaluation_time_ms=duration,
                    )

                func = namespace[self.func_name]
                passed, total, failure_details = self._run_cases(func, self.test_cases)
                self._refresh_suspicion(code, func)
                test_ratio = passed / total if total > 0 else 1.0

                # Base score: 20 points for compilation + 80 points for test passing ratio
                score = round(20.0 + 80.0 * test_ratio, 2)

                passed_holdout = None
                if self.holdout_cases is not None:
                    h_passed, h_total, _ = self._run_cases(func, self.holdout_cases)
                    passed_holdout = (h_passed == h_total)

                duration = (time.perf_counter() - t0) * 1000.0
                return FitnessResult(
                    score=score,
                    sub_scores={"compiled": 100.0, "tests_passed": round(test_ratio * 100.0, 2)},
                    passed_holdout=passed_holdout,
                    artifacts={"passed_tests": passed, "total_tests": total, "failures": failure_details},
                    evaluation_time_ms=duration,
                )
            finally:
                for mod_name in injected_mods:
                    sys.modules.pop(mod_name, None)

        except Exception as err:
            duration = (time.perf_counter() - t0) * 1000.0
            return FitnessResult(
                score=0.0,
                sub_scores={"compiled": 0.0, "tests_passed": 0.0},
                artifacts={"error": str(err)},
                evaluation_time_ms=duration,
            )


class SandboxFunctionTestEvaluator(Evaluator):
    """Evaluates a PatchGenome or ASTGenome safely inside an isolated subprocess sandbox."""

    def __init__(
        self,
        base_sources: dict[str, str],
        target_file: str,
        func_name: str,
        test_cases: Sequence[tuple[tuple[Any, ...], Any]],
        holdout_cases: Sequence[tuple[tuple[Any, ...], Any]] | None = None,
        config: Any = None,
    ):
        from .sandbox import SandboxConfig, SandboxRunner

        self.base_sources = base_sources
        self.target_file = target_file
        self.func_name = func_name
        self.test_cases = list(test_cases)
        self.holdout_cases = list(holdout_cases) if holdout_cases is not None else None
        self.config = config or SandboxConfig()
        self.runner = SandboxRunner(self.config)

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def evaluate(
        self, target: EvolabGenome | Individual | list[float], context: dict[str, Any] | None = None
    ) -> FitnessResult:
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
                raise TypeError(f"SandboxFunctionTestEvaluator expects PatchGenome or ASTGenome, got {type(patch)}")
        except Exception as err:
            duration = (time.perf_counter() - t0) * 1000.0
            return FitnessResult(
                score=0.0,
                sub_scores={"applied": 0.0, "compiled": 0.0},
                artifacts={"error": f"Patch application error: {err}"},
                evaluation_time_ms=duration,
            )

        exec_res = self.runner.run_test_suite(
            sources=applied,
            target_file=self.target_file,
            func_name=self.func_name,
            test_cases=self.test_cases,
            holdout_cases=self.holdout_cases,
        )

        duration = (time.perf_counter() - t0) * 1000.0

        if exec_res.timeout_triggered:
            return FitnessResult(
                score=0.0,
                sub_scores={"compiled": 0.0, "tests_passed": 0.0},
                passed_holdout=False,
                artifacts={
                    "error": exec_res.error,
                    "timeout_triggered": True,
                    "stdout": exec_res.stdout,
                    "stderr": exec_res.stderr,
                },
                evaluation_time_ms=duration,
            )

        if not exec_res.success:
            return FitnessResult(
                score=10.0 if "not found in" in (exec_res.error or "") else 0.0,
                sub_scores={"compiled": 0.0, "tests_passed": 0.0},
                passed_holdout=False,
                artifacts={
                    "error": exec_res.error,
                    "stdout": exec_res.stdout,
                    "stderr": exec_res.stderr,
                },
                evaluation_time_ms=duration,
            )

        val = exec_res.return_value or {}
        passed = val.get("tests_passed", 0)
        total = val.get("total_tests", len(self.test_cases))
        holdout_passed = val.get("holdout_passed", None) if self.holdout_cases is not None else None

        test_ratio = passed / total if total > 0 else 1.0
        score = round(20.0 + 80.0 * test_ratio, 2)

        return FitnessResult(
            score=score,
            sub_scores={"compiled": 100.0, "tests_passed": round(test_ratio * 100.0, 2)},
            passed_holdout=holdout_passed,
            artifacts={
                "passed_tests": passed,
                "total_tests": total,
                "failures": val.get("details", []),
                "stdout": exec_res.stdout,
            },
            evaluation_time_ms=duration,
        )


class GeneralizationEvaluator(Evaluator):
    """Evaluates generalization gap between train fitness and holdout test fitness."""

    def __init__(
        self,
        train_evaluator: Evaluator | Callable[[Any], float],
        test_evaluator: Evaluator | Callable[[Any], float],
        max_overfit_gap: float = 30.0,
        gap_penalty_weight: float = 1.0,
    ) -> None:
        self.train_evaluator = train_evaluator
        self.test_evaluator = test_evaluator
        self.max_overfit_gap = max_overfit_gap
        self.gap_penalty_weight = gap_penalty_weight

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def evaluate(
        self, target: EvolabGenome | Individual | list[float], context: dict[str, Any] | None = None
    ) -> FitnessResult:
        t0 = time.perf_counter()
        train_score = (
            float(self.train_evaluator(target))
            if callable(self.train_evaluator)
            else self.train_evaluator.evaluate(target).score
        )
        test_score = (
            float(self.test_evaluator(target))
            if callable(self.test_evaluator)
            else self.test_evaluator.evaluate(target).score
        )
        duration = (time.perf_counter() - t0) * 1000.0

        gap = max(0.0, train_score - test_score)
        is_overfit = gap > self.max_overfit_gap

        # Penalize overfitted fitness score to reflect true generalization
        penalized_score = max(0.0, train_score - self.gap_penalty_weight * max(0.0, gap - self.max_overfit_gap))

        return FitnessResult(
            score=round(penalized_score, 4),
            sub_scores={
                "train_score": round(train_score, 4),
                "test_score": round(test_score, 4),
                "generalization_gap": round(gap, 4),
            },
            passed_holdout=not is_overfit,
            artifacts={
                "is_overfit": is_overfit,
                "gap": gap,
                "train_score": train_score,
                "test_score": test_score,
            },
            evaluation_time_ms=duration,
        )


class AdversarialRobustEvaluator(Evaluator):
    """Evaluates fitness while penalizing syntactic bloat, junk genes, and artificial inflation."""

    def __init__(
        self,
        base_evaluator: Evaluator | Callable[[Any], float],
        baseline_length: int = 4,
        bloat_penalty_per_gene: float = 5.0,
        stealth_resolution_epsilon: float = 1e-5,
    ) -> None:
        self.base_evaluator = base_evaluator
        self.baseline_length = baseline_length
        self.bloat_penalty_per_gene = bloat_penalty_per_gene
        self.stealth_resolution_epsilon = stealth_resolution_epsilon

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def evaluate(
        self, target: EvolabGenome | Individual | list[float], context: dict[str, Any] | None = None
    ) -> FitnessResult:
        t0 = time.perf_counter()
        raw_score = (
            float(self.base_evaluator(target))
            if callable(self.base_evaluator)
            else self.base_evaluator.evaluate(target).score
        )
        duration = (time.perf_counter() - t0) * 1000.0

        # Measure gene count / syntactic length
        if hasattr(target, "values"):
            length = len(target.values)
        elif hasattr(target, "genes"):
            length = len(target.genes)
        elif isinstance(target, list):
            length = len(target)
        else:
            length = self.baseline_length

        bloat_excess = max(0, length - self.baseline_length)
        bloat_penalty = bloat_excess * self.bloat_penalty_per_gene
        robust_score = max(0.0, raw_score - bloat_penalty)

        return FitnessResult(
            score=round(robust_score, 4),
            sub_scores={"raw_score": round(raw_score, 4), "bloat_penalty": round(bloat_penalty, 4)},
            artifacts={"bloat_excess": bloat_excess, "raw_score": raw_score},
            evaluation_time_ms=duration,
        )


