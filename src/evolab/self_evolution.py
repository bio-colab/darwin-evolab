"""self_evolution.py — Closed Self-Improvement Loop & Autonomous Meta-Evolution.

Part of Darwin-Evolab Pillar 2: Autonomous Self-Evolution.
Enables the system to inspect its own performance, diagnose evolutionary bottlenecks,
generate structured algorithmic/hyperparameter adaptations, evaluate them in a
sandboxed arena, and submit them to the vaccinated Governor for atomic self-commitment.
"""

from __future__ import annotations

import copy
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from evolab.self_model import (
    diagnose_run,
    govern_modification,
    render_self_modification_proposal,
    wilson_interval,
)


@dataclass
class MetaProposal:
    """A proposed structural, algorithmic, or parameter self-modification."""
    proposal_id: int
    category: str  # e.g., "diversity_boost", "elastic_budget", "operator_weighting", "compositional"
    parameters: dict[str, Any]
    rationale: str
    status: str = "PENDING"  # PENDING, ACCEPTED, REJECTED
    verdict: dict[str, Any] | None = None
    created_at_epoch: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "category": self.category,
            "parameters": dict(self.parameters),
            "rationale": self.rationale,
            "status": self.status,
            "verdict": self.verdict,
            "created_at_epoch": self.created_at_epoch,
        }


class SelfEvolutionEngine:
    """Closed self-improvement engine managing the inspect -> propose -> evaluate -> govern cycle."""

    def __init__(
        self,
        base_config: dict[str, Any] | None = None,
        alpha: float = 0.05,
        min_effect_size: float = 0.20,
    ):
        self.config = dict(base_config or {
            "immigrant_fraction": 0.0,
            "mutation_rate": 0.15,
            "crossover_rate": 0.70,
            "plateau_gens": 5,
            "budget_elasticity": False,
            "compositional_depth": 1,
        })
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.proposal_counter = 0
        self.history: list[MetaProposal] = []
        self.dead_ends: set[str] = set()

    def inspect_telemetry(self, run_history: list[dict[str, Any]], holdout_gap: float | None = None) -> dict[str, Any]:
        """Perform interoceptive and exteroceptive diagnosis on run telemetry."""
        return diagnose_run(run_history, holdout_gap=holdout_gap)

    def generate_proposals(self, diagnosis: dict[str, Any]) -> list[MetaProposal]:
        """Synthesize candidate self-modifications addressing diagnosed bottlenecks."""
        proposals: list[MetaProposal] = []

        is_converged = diagnosis.get("premature_convergence", False)
        is_stagnant = diagnosis.get("stagnation", False)
        is_overfitting = diagnosis.get("overfit_risk", False)

        # 1. Address Premature Convergence: Inject immigrants or boost mutation
        if is_converged:
            cand_immigrant = round(min(0.35, self.config.get("immigrant_fraction", 0.0) + 0.15), 2)
            key = f"immigrant_fraction:{cand_immigrant}"
            if key not in self.dead_ends:
                self.proposal_counter += 1
                proposals.append(
                    MetaProposal(
                        proposal_id=self.proposal_counter,
                        category="diversity_injection",
                        parameters={"immigrant_fraction": cand_immigrant},
                        rationale="Diagnosed premature convergence; inject immigrants to sustain gene diversity.",
                    )
                )

        # 2. Address Stagnation: Expand compositional depth or enable budget elasticity
        if is_stagnant:
            if not self.config.get("budget_elasticity", False):
                key = "budget_elasticity:True"
                if key not in self.dead_ends:
                    self.proposal_counter += 1
                    proposals.append(
                        MetaProposal(
                            proposal_id=self.proposal_counter,
                            category="elastic_budget",
                            parameters={"budget_elasticity": True},
                            rationale="Diagnosed plateau stagnation; engage budget elasticity to terminate dead paths.",
                        )
                    )

            cand_depth = self.config.get("compositional_depth", 1) + 1
            if cand_depth <= 3:
                key = f"compositional_depth:{cand_depth}"
                if key not in self.dead_ends:
                    self.proposal_counter += 1
                    proposals.append(
                        MetaProposal(
                            proposal_id=self.proposal_counter,
                            category="compositional_expansion",
                            parameters={"compositional_depth": cand_depth},
                            rationale="Single-point mutation stalled; increase compositional lookahead depth.",
                        )
                    )

        # 3. Address Overfitting: Restrict mutation rate
        if is_overfitting:
            cand_mut = round(max(0.05, self.config.get("mutation_rate", 0.15) * 0.8), 3)
            key = f"mutation_rate:{cand_mut}"
            if key not in self.dead_ends:
                self.proposal_counter += 1
                proposals.append(
                    MetaProposal(
                        proposal_id=self.proposal_counter,
                        category="parsimony_shrink",
                        parameters={"mutation_rate": cand_mut},
                        rationale="High holdout generalization gap detected; reduce mutation entropy.",
                    )
                )

        return proposals

    def evaluate_in_arena(
        self,
        proposal: MetaProposal,
        benchmark_evaluator: Callable[[dict[str, Any], int], float],
        seeds: Sequence[int] = (10, 20, 30, 40, 50, 60, 70),
    ) -> tuple[list[float], list[float], int]:
        """Run sandboxed paired A/B evaluation between baseline and candidate configuration."""
        baseline_scores: list[float] = []
        candidate_scores: list[float] = []
        regressions = 0

        # Construct candidate config
        cand_config = copy.deepcopy(self.config)
        cand_config.update(proposal.parameters)

        for s in seeds:
            s_base = benchmark_evaluator(self.config, s)
            s_cand = benchmark_evaluator(cand_config, s)

            baseline_scores.append(s_base)
            candidate_scores.append(s_cand)

            if s_cand < s_base - 0.01:
                regressions += 1

        return baseline_scores, candidate_scores, regressions

    def step(
        self,
        proposal: MetaProposal,
        baseline_scores: list[float],
        candidate_scores: list[float],
        regressions: int = 0,
    ) -> dict[str, Any]:
        """Submit proposal results to the vaccinated Governor and execute atomic self-commit on ACCEPT."""
        verdict = govern_modification(
            baseline=baseline_scores,
            candidate=candidate_scores,
            regressions=regressions,
            alpha=self.alpha,
            min_effect_size=self.min_effect_size,
        )
        proposal.verdict = verdict
        decision = verdict.get("decision", "REJECT")
        proposal.status = decision

        self.history.append(proposal)

        if decision == "ACCEPT":
            # Atomic update of engine configuration!
            self.config.update(proposal.parameters)
            outcome = {
                "action": "CONFIG_UPDATED",
                "proposal_id": proposal.proposal_id,
                "category": proposal.category,
                "new_config": dict(self.config),
                "verdict": verdict,
            }
        else:
            # Register dead-end parameter mutation
            for k, v in proposal.parameters.items():
                self.dead_ends.add(f"{k}:{v}")
            outcome = {
                "action": "PROPOSAL_REJECTED",
                "proposal_id": proposal.proposal_id,
                "reasons": verdict.get("reasons", []),
                "verdict": verdict,
            }

        return outcome


class AutonomousEvolutionManager:
    """Default-active autonomous self-evolution manager.

    Conducts passive telemetry monitoring, autonomous bottleneck detection,
    background Dreaming replay optimization over discovery trees, and
    statistically vaccinated Governor auto-promotion with hot-swapping.
    """

    def __init__(
        self,
        policy_path: str | Path = ".evolab/autonomous_policy.json",
        alpha: float = 0.05,
        min_effect_size: float = 0.50,
        stagnation_threshold: int = 3,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_events_count = 0
        self.total_sessions_recorded = 0
        self.promotion_history: list[dict[str, Any]] = []

        # Default baseline policy
        self.active_policy: dict[str, Any] = {
            "operator_weights": {
                "InsertGuard": 0.142857,
                "BoundaryFlip": 0.142857,
                "SwapCondition": 0.142857,
                "DeleteStatement": 0.142857,
                "OffByOne": 0.142857,
                "BinOpFlip": 0.142857,
                "ConstantMutate": 0.142857,
            },
            "immigrant_fraction": 0.0,
            "budget_elasticity": False,
            "compositional_depth": 1,
            "version": "baseline",
            "last_updated_epoch": time.time(),
        }

        self._load_persisted_policy()

    def _load_persisted_policy(self) -> None:
        if self.policy_path.is_file():
            try:
                data = json.loads(self.policy_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.active_policy.update(data)
            except Exception:
                pass

    def save_persisted_policy(self) -> None:
        try:
            self.policy_path.parent.mkdir(parents=True, exist_ok=True)
            self.policy_path.write_text(json.dumps(self.active_policy, indent=2), encoding="utf-8")
        except Exception:
            pass

    def record_run_telemetry(self, telemetry: dict[str, Any]) -> None:
        """Passively records search telemetry and checks for stagnation bottlenecks."""
        self.total_sessions_recorded += 1
        is_stagnant = telemetry.get("stagnation", False) or telemetry.get("plateau_detected", False)
        if is_stagnant:
            self.stagnation_events_count += 1

    def should_trigger_optimization(self) -> bool:
        """Determines if evolutionary stagnation warrants an autonomous Dreaming cycle."""
        return self.stagnation_events_count >= self.stagnation_threshold

    def run_autonomous_optimization(
        self,
        discovery_trees: Sequence[Any] | None = None,
        report_path: str | Path | None = None,
        n_samples: int = 2000,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Runs background counterfactual Dreaming optimization and submits to the Governor."""
        from evolab.dream.self_model_dream import OperatorReweighter
        from evolab.replay_simulator import DiscoveryTree

        trees = list(discovery_trees) if discovery_trees else []
        if not trees and report_path:
            path = Path(report_path)
            if path.is_file():
                trees = DiscoveryTree.from_swe_bench_report(path)

        if not trees:
            return {
                "decision": "SKIPPED",
                "reason": "no_discovery_trees_available",
                "promoted": False,
            }

        reweighter = OperatorReweighter(trees=trees)
        result = reweighter.optimize(n_samples=n_samples, seed=seed)

        verdict = result.governor_verdict
        is_accepted = (
            verdict.get("decision") == "ACCEPT"
            and result.p_value < self.alpha
            and result.cohen_d >= self.min_effect_size
        )

        record = {
            "timestamp_epoch": time.time(),
            "p_value": result.p_value,
            "cohen_d": result.cohen_d,
            "evals_saved_percent": result.mean_evaluations_saved_percent,
            "verdict": verdict,
            "promoted": is_accepted,
        }
        self.promotion_history.append(record)

        if is_accepted:
            # Hot-swap runtime active policy!
            self.active_policy["operator_weights"] = result.optimal_weights
            self.active_policy["version"] = f"dream_promoted_{len(self.promotion_history)}"
            self.active_policy["last_updated_epoch"] = time.time()
            self.save_persisted_policy()
            # Reset stagnation counter
            self.stagnation_events_count = 0
            return {
                "decision": "ACCEPT",
                "promoted": True,
                "optimal_weights": result.optimal_weights,
                "p_value": result.p_value,
                "cohen_d": result.cohen_d,
                "evals_saved_percent": result.mean_evaluations_saved_percent,
            }
        else:
            return {
                "decision": "REJECT",
                "promoted": False,
                "reasons": verdict.get("reasons", ["criteria_not_met"]),
                "p_value": result.p_value,
                "cohen_d": result.cohen_d,
            }

    def get_active_policy(self) -> dict[str, Any]:
        """Returns the currently active, promoted evolutionary parameters."""
        return dict(self.active_policy)

