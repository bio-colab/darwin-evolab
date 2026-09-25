"""manifest.py — Declarative Loop Manifest (Goal -> Loop -> Target) for Darwin-Evolab.

Implements the 2026 Modern Harness Standard (Harness Books / Claude Code):
Declarative specification defining:
- Goal: Objective, problem type, target file, constraints, and fitness threshold.
- Loop: Optimization strategy, budget ceilings, patience, plateau breaking, and governor gates.
- Target: Verification oracle, additive baseline scoping, and regression tolerances.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class GoalSpec:
    """Declarative specification of the agent's optimization goal."""

    goal_type: str = "repair"  # "repair" | "optimize" | "synthesize" | "benchmark"
    description: str = ""
    target_file: str = "solution.py"
    sources: dict[str, str] = field(default_factory=dict)
    test_code: str | None = None
    pytest_file: str | None = None
    test_cases: list[dict[str, Any]] | None = None
    target_function: str | None = None
    fitness_target: float = 99.7
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalSpec:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LoopBudget:
    """Explicit resource and evaluation limits for the loop."""

    max_evaluations: int = 64
    max_generations: int = 30
    timeout_seconds: float = 30.0
    max_context_tokens: int = 8192

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopBudget:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GovernorSpec:
    """Statistical governor verification settings."""

    enabled: bool = True
    alpha: float = 0.05
    epsilon: float = 1e-6
    min_effect_size: float = 0.4
    strict_zero_regressions: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GovernorSpec:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LoopSpec:
    """Declarative specification of the evolutionary loop behavior."""

    strategy: str = "greedy_ast_prior"  # "greedy_ast_prior" | "evolutionary_ga" | "cgp_silicon" | "island_swarm"
    budget: LoopBudget = field(default_factory=LoopBudget)
    patience: int = 15
    plateau_breaking: bool = True
    taboo_memory: bool = True
    first_ascent: bool = True
    candidate_ranker: str = "learned_prior"  # "learned_prior" | "sbfl_only" | "uniform"
    governor: GovernorSpec = field(default_factory=GovernorSpec)
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "budget": self.budget.to_dict(),
            "patience": self.patience,
            "plateau_breaking": self.plateau_breaking,
            "taboo_memory": self.taboo_memory,
            "first_ascent": self.first_ascent,
            "candidate_ranker": self.candidate_ranker,
            "governor": self.governor.to_dict(),
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopSpec:
        budget_d = data.get("budget", {})
        budget = LoopBudget.from_dict(budget_d) if isinstance(budget_d, dict) else LoopBudget()
        gov_d = data.get("governor", {})
        governor = GovernorSpec.from_dict(gov_d) if isinstance(gov_d, dict) else GovernorSpec()

        return cls(
            strategy=data.get("strategy", "greedy_ast_prior"),
            budget=budget,
            patience=data.get("patience", 15),
            plateau_breaking=data.get("plateau_breaking", True),
            taboo_memory=data.get("taboo_memory", True),
            first_ascent=data.get("first_ascent", True),
            candidate_ranker=data.get("candidate_ranker", "learned_prior"),
            governor=governor,
            random_seed=data.get("random_seed", 42),
        )


@dataclass
class TargetSpec:
    """Declarative specification of the target verification oracle and gates."""

    verification_mode: str = "additive_baseline"  # "additive_baseline" | "absolute" | "isolated_sandbox"
    oracle_type: str = "pytest"  # "pytest" | "assertions" | "truth_table_2k" | "custom_evaluator"
    regression_tolerance: int = 0
    enforce_additive_type_check: bool = True
    enforce_ast_purity: bool = True
    sandbox_isolation: bool = True
    distill_trajectory_on_success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetSpec:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LoopManifest:
    """Top-level Declarative Loop Manifest (Goal -> Loop -> Target)."""

    name: str
    version: str = "1.0.0"
    goal: GoalSpec = field(default_factory=GoalSpec)
    loop: LoopSpec = field(default_factory=LoopSpec)
    target: TargetSpec = field(default_factory=TargetSpec)
    schema_uri: str = "https://evolab.dev/schemas/loop.v1.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "$schema": self.schema_uri,
            "name": self.name,
            "version": self.version,
            "goal": self.goal.to_dict(),
            "loop": self.loop.to_dict(),
            "target": self.target.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopManifest:
        goal_d = data.get("goal", {})
        loop_d = data.get("loop", {})
        target_d = data.get("target", {})

        return cls(
            name=data.get("name", "evolab_loop"),
            version=data.get("version", "1.0.0"),
            goal=GoalSpec.from_dict(goal_d),
            loop=LoopSpec.from_dict(loop_d),
            target=TargetSpec.from_dict(target_d),
            schema_uri=data.get("$schema", "https://evolab.dev/schemas/loop.v1.json"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> LoopManifest:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path | str) -> LoopManifest:
        p = Path(path)
        return cls.from_json(p.read_text(encoding="utf-8"))

    @classmethod
    def for_repair(
        cls,
        name: str,
        sources: dict[str, str],
        target_file: str,
        pytest_file: str | None = None,
        test_code: str | None = None,
        max_evals: int = 64,
        seed: int = 42,
        scenario_name: str | None = None,
    ) -> LoopManifest:
        """Factory helper creating a validated loop manifest for APR repair."""
        metadata = {"scenario": scenario_name} if scenario_name else {}
        return cls(
            name=name,
            goal=GoalSpec(
                goal_type="repair",
                description=f"Automated Program Repair on {target_file}",
                target_file=target_file,
                sources=sources,
                pytest_file=pytest_file,
                test_code=test_code,
                metadata=metadata,
                fitness_target=99.7,
            ),
            loop=LoopSpec(
                strategy="greedy_ast_prior",
                budget=LoopBudget(max_evaluations=max_evals, timeout_seconds=20.0),
                random_seed=seed,
            ),
            target=TargetSpec(
                verification_mode="additive_baseline",
                enforce_additive_type_check=True,
                enforce_ast_purity=True,
                distill_trajectory_on_success=True,
            ),
        )
