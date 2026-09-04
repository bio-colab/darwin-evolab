"""bandit_latent_optimizer.py — CktGen-style Test-Time Bandit Latent Search.

Promoted to production core: Performs rapid test-time optimization on the
conditioned latent manifold without retraining. Uses Upper Confidence Bound (UCB1)
multi-armed bandit search to refine initial seeds within milliseconds.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .grammar_guard import PhysicalGrammarGuard
from .opamp_benchmark import OpAmpSizing, evaluate_opamp_analytical
from .sky130_pdk import Sky130Corner
from .spec_conditioned_prior import TargetCircuitSpec


@dataclass
class BanditArm:
    """An exploration arm representing a candidate sizing and its local neighborhood."""
    arm_id: int
    sizing: OpAmpSizing
    fitness: float
    pull_count: int = 1
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, self.pull_count)


class BanditLatentOptimizer:
    """Test-time multi-armed bandit optimizer for circuit candidates."""

    def __init__(
        self,
        exploration_weight: float = 0.5,
        step_sigma: float = 0.04,
        seed: int = 42,
    ):
        self.c = exploration_weight
        self.sigma = step_sigma
        self.guard = PhysicalGrammarGuard()
        self.rng = random.Random(seed)

    def optimize_seeds(
        self,
        initial_sizings: Sequence[OpAmpSizing],
        target_spec: TargetCircuitSpec,
        budget_trials: int = 25,
    ) -> list[OpAmpSizing]:
        """Refines initial seed candidates using UCB1 bandit search in normalized space."""
        if not initial_sizings:
            return []

        arms: list[BanditArm] = []
        for i, sizing in enumerate(initial_sizings):
            m = evaluate_opamp_analytical(sizing, Sky130Corner.TT)
            fit = target_spec.compute_fitness(m)
            arms.append(BanditArm(arm_id=i, sizing=sizing, fitness=fit, pull_count=1, total_reward=fit))

        best_sizing = max(arms, key=lambda a: a.fitness).sizing
        best_fitness = max(a.fitness for a in arms)

        total_pulls = len(arms)
        for trial in range(budget_trials):
            total_pulls += 1
            best_ucb = -float("inf")
            chosen_arm = arms[0]
            for arm in arms:
                ucb = arm.mean_reward + self.c * math.sqrt(math.log(total_pulls) / arm.pull_count)
                if ucb > best_ucb:
                    best_ucb = ucb
                    chosen_arm = arm

            norm_vec = chosen_arm.sizing.to_normalized_vector()
            perturbed_vec = []
            for val in norm_vec:
                noise = self.rng.gauss(0.0, self.sigma)
                perturbed_vec.append(max(0.0, min(val + noise, 1.0)))

            candidate_sizing = self.guard.repair_and_project(
                OpAmpSizing.from_normalized_vector(perturbed_vec)
            )

            m = evaluate_opamp_analytical(candidate_sizing, Sky130Corner.TT)
            candidate_fitness = target_spec.compute_fitness(m)

            chosen_arm.pull_count += 1
            chosen_arm.total_reward += candidate_fitness

            if candidate_fitness > chosen_arm.fitness:
                chosen_arm.sizing = candidate_sizing
                chosen_arm.fitness = candidate_fitness

            if candidate_fitness > best_fitness:
                best_fitness = candidate_fitness
                best_sizing = candidate_sizing

        arms.sort(key=lambda a: a.fitness, reverse=True)
        return [arm.sizing for arm in arms]
