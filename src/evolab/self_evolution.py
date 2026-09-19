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
