"""
swe_bench.py — SWE-bench Lite Standard Industrial APR Adapter for Darwin-Evolab.

Enables Darwin-Evolab to ingest, localize, evolve, and verify real-world software
engineering issues mined from GitHub in the official SWE-bench Lite dataset format.
Enforces the dual invariant: 100% of FAIL_TO_PASS tests must pass, and 100% of
PASS_TO_PASS tests must remain unbroken (zero regressions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

from .adapters import DomainAdapter
from .evaluators import Evaluator, FitnessResult, FunctionTestEvaluator
from .genome import Individual
from .repair import RepairGenome, catalog_sources, greedy_repair
from .reporters import format_git_patch


@dataclass
class SWEBenchInstance:
    """Represents a standard SWE-bench Lite issue instance."""

    instance_id: str
    repo: str
    problem_statement: str
    target_file: str
    sources: dict[str, str]
    fail_to_pass_tests: list[tuple[tuple[Any, ...], Any]]
    pass_to_pass_tests: list[tuple[tuple[Any, ...], Any]]
    fail_to_pass_names: list[str] = field(default_factory=list)
    pass_to_pass_names: list[str] = field(default_factory=list)
    base_commit: str = "HEAD"
    reference_patch: str | None = None


@dataclass
class SWEBenchResolution:
    """Evaluation verdict for a SWE-bench issue repair."""

    instance_id: str
    resolved: bool
    fail_to_pass_passed: bool
    pass_to_pass_clean: bool
    generated_patch: str
    evaluations_used: int
    execution_time_seconds: float
    target_file: str


class SWEBenchEvaluator(Evaluator):
    """Rigorous SWE-bench evaluator: evaluates FAIL_TO_PASS and verifies zero regressions on PASS_TO_PASS."""

    def __init__(
        self,
        instance: SWEBenchInstance,
        func_name: str = "solve",
    ) -> None:
        self.instance = instance
        self.func_name = func_name
        self.fail_evaluator = FunctionTestEvaluator(
            base_sources=instance.sources,
            target_file=instance.target_file,
            func_name=func_name,
            test_cases=instance.fail_to_pass_tests,
        )
        self.pass_evaluator = FunctionTestEvaluator(
            base_sources=instance.sources,
            target_file=instance.target_file,
            func_name=func_name,
            test_cases=instance.pass_to_pass_tests,
        )

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        t0 = time.perf_counter()
        # 1. Evaluate FAIL_TO_PASS (Must pass all)
        f_res = self.fail_evaluator.evaluate(target, context)
        # 2. Evaluate PASS_TO_PASS (Must remain completely green)
        p_res = self.pass_evaluator.evaluate(target, context)
        duration = (time.perf_counter() - t0) * 1000.0

        all_fail_to_pass_ok = f_res.score >= 99.9
        all_pass_to_pass_ok = p_res.score >= 99.9

        # Primary score: weighted combination
        # 60% weight on fixing failing tests, 40% on preserving existing tests
        score = (f_res.score * 0.6) + (p_res.score * 0.4)
        if all_fail_to_pass_ok and all_pass_to_pass_ok:
            score = 100.0

        return FitnessResult(
            score=round(score, 2),
            sub_scores={
                "fail_to_pass": f_res.score,
                "pass_to_pass": p_res.score,
            },
            passed_holdout=all_pass_to_pass_ok,
            artifacts={
                "resolved": bool(all_fail_to_pass_ok and all_pass_to_pass_ok),
                "fail_to_pass_passed": all_fail_to_pass_ok,
                "pass_to_pass_clean": all_pass_to_pass_ok,
            },
            evaluation_time_ms=duration,
        )


class SWEBenchAdapter(DomainAdapter):
    """Domain driver for SWE-bench Lite industrial APR benchmarks."""

    @property
    def name(self) -> str:
        return "swe_bench"

    def parse_spec(self, raw_input: Any) -> SWEBenchInstance:
        if isinstance(raw_input, SWEBenchInstance):
            return raw_input
        elif isinstance(raw_input, (str, Path)):
            p = Path(raw_input)
            if not p.is_file():
                raise FileNotFoundError(f"SWE-bench instance file not found: {p}")
            data = json.loads(p.read_text(encoding="utf-8"))
        elif isinstance(raw_input, dict):
            data = raw_input
        else:
            raise TypeError(f"Unsupported SWE-bench input type: {type(raw_input)}")

        return SWEBenchInstance(
            instance_id=str(data.get("instance_id", "custom_instance")),
            repo=str(data.get("repo", "unknown_repo")),
            problem_statement=str(data.get("problem_statement", "")),
            target_file=str(data.get("target_file", list(data.get("sources", {}).keys())[0] if data.get("sources") else "main.py")),
            sources=dict(data.get("sources", {})),
            fail_to_pass_tests=list(data.get("fail_to_pass_tests", [])),
            pass_to_pass_tests=list(data.get("pass_to_pass_tests", [])),
            fail_to_pass_names=list(data.get("FAIL_TO_PASS", [])),
            pass_to_pass_names=list(data.get("PASS_TO_PASS", [])),
            base_commit=str(data.get("base_commit", "HEAD")),
            reference_patch=data.get("patch"),
        )

    def build_population(self, spec: SWEBenchInstance, size: int, rng: random.Random) -> list[Individual]:
        edits = catalog_sources(spec.sources)
        pop: list[Individual] = [
            Individual(
                genome=RepairGenome(sources=dict(spec.sources), target_file=spec.target_file, edits=[]),
                species="spec_swe_bench",
            )
        ]
        if edits:
            for _ in range(size - 1):
                k = min(len(edits), rng.randint(1, 3))
                sample = rng.sample(edits, k)
                pop.append(
                    Individual(
                        genome=RepairGenome(sources=dict(spec.sources), target_file=spec.target_file, edits=sample),
                        species="spec_swe_bench",
                    )
                )
        else:
            while len(pop) < size:
                pop.append(pop[0].clone())
        return pop

    def build_evaluator(self, spec: SWEBenchInstance) -> Evaluator:
        func_name = "solve"
        if spec.sources and spec.target_file in spec.sources:
            code = spec.sources[spec.target_file]
            for line in code.splitlines():
                line_s = line.strip()
                if line_s.startswith("def ") and "(" in line_s:
                    func_name = line_s.split("def ")[1].split("(")[0].strip()
                    break

        return SWEBenchEvaluator(instance=spec, func_name=func_name)

    def solve_instance(
        self,
        spec: SWEBenchInstance,
        max_evals: int = 32,
    ) -> SWEBenchResolution:
        """Executes targeted AST + Ochiai SBFL guided repair on the SWE-bench instance."""
        t0 = time.perf_counter()
        evaluator = self.build_evaluator(spec)

        winning_genome, history, n_evals = greedy_repair(
            sources=spec.sources,
            target_file=spec.target_file,
            evaluator=evaluator,
            max_evals=max_evals,
            prioritize_by_suspicion=True,
        )

        res = evaluator.evaluate(winning_genome)
        duration = time.perf_counter() - t0

        repaired_sources = dict(spec.sources)
        repaired_sources[spec.target_file] = winning_genome.to_code()
        patch_text = format_git_patch(spec, repaired_sources)

        is_resolved = bool(res.artifacts.get("resolved", False))

        return SWEBenchResolution(
            instance_id=spec.instance_id,
            resolved=is_resolved,
            fail_to_pass_passed=bool(res.artifacts.get("fail_to_pass_passed", False)),
            pass_to_pass_clean=bool(res.artifacts.get("pass_to_pass_clean", False)),
            generated_patch=patch_text,
            evaluations_used=n_evals,
            execution_time_seconds=round(duration, 3),
            target_file=spec.target_file,
        )

    def export_solution(
        self,
        individual: Individual,
        spec: SWEBenchInstance,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        repaired_code = individual.genome.to_code() if hasattr(individual.genome, "to_code") else ""
        repaired_sources = dict(spec.sources)
        repaired_sources[spec.target_file] = repaired_code

        diff_patch = format_git_patch(spec, repaired_sources)
        if output_path:
            p = Path(output_path)
            if p.parent and str(p.parent):
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(diff_patch, encoding="utf-8")

        return {
            "instance_id": spec.instance_id,
            "git_patch": diff_patch,
            "target_file": spec.target_file,
        }
