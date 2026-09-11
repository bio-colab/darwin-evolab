"""
src/evolab/holland.py — Foundations of John H. Holland (1975).
Implements:
1. Schema Representation & Holland's Schema Theorem (Order, Defining Length, Disruption Bound).
2. SchemaTracker: Empirical verification of Holland's reproductive lower bound in evolving populations.
3. HollandTrialAllocator: Chapter 5 Multi-Armed Bandit Optimal Allocation of Trials.

References:
    Holland, J. H. (1975). Adaptation in Natural and Artificial Systems: An Introductory
    Analysis with Applications to Biology, Control, and Artificial Intelligence.
    University of Michigan Press. (Reprinted by MIT Press, 1992).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Schema:
    """Representation of a Holland Schema (hyperplane template) over a chromosome.

    Elements in pattern:
        - Specific symbol/allele (e.g. 0, 1, 'A', etc.)
        - '*' wildcard (don't-care symbol that matches any allele)

    Properties:
        - order o(H): Number of defined (non-wildcard) positions.
        - defining_length delta(H): Distance between the first and last defined positions.
    """

    pattern: tuple[Any, ...]

    def __post_init__(self):
        if not isinstance(self.pattern, tuple):
            object.__setattr__(self, "pattern", tuple(self.pattern))

    @property
    def order(self) -> int:
        """Holland Order o(H): count of specific fixed alleles."""
        return sum(1 for gene in self.pattern if gene != "*")

    @property
    def defining_length(self) -> int:
        """Holland Defining Length delta(H): index difference between outermost fixed positions."""
        fixed_indices = [i for i, gene in enumerate(self.pattern) if gene != "*"]
        if len(fixed_indices) <= 1:
            return 0
        return fixed_indices[-1] - fixed_indices[0]

    def matches(self, chromosome: Sequence[Any]) -> bool:
        """Determines if a candidate chromosome is an instance of this schema."""
        if len(chromosome) != len(self.pattern):
            return False
        for expected, actual in zip(self.pattern, chromosome):
            if expected != "*" and expected != actual:
                return False
        return True

    def survival_bound(self, p_c: float, p_m: float, chromosome_length: int | None = None) -> float:
        """Computes Holland's schema survival probability under single-point crossover and bit mutation:

            S(H) >= 1 - p_c * (delta(H) / (l - 1)) - o(H) * p_m

        Where l is chromosome length, p_c is crossover probability, and p_m is mutation probability.
        """
        l = chromosome_length if chromosome_length is not None else len(self.pattern)
        denom = max(l - 1, 1)
        crossover_disruption = p_c * (self.defining_length / denom)
        mutation_disruption = self.order * p_m
        bound = 1.0 - crossover_disruption - mutation_disruption
        return max(0.0, min(1.0, round(bound, 6)))


class SchemaTracker:
    """Monitors schema representation and validates Holland's Schema Theorem across generations.

    Theorem Lower Bound:
        E[m(H, t+1)] >= m(H, t) * (f(H, t) / f_bar(t)) * [1 - p_c * delta(H)/(l-1) - o(H)*p_m]
    """

    def __init__(self, schemas: Sequence[Schema]):
        self.schemas = list(schemas)

    def count_instances(self, schema: Schema, population: Sequence[Any]) -> int:
        """Counts how many individuals in the population instantiate the schema."""
        count = 0
        for ind in population:
            chrom = getattr(ind, "genome", ind)
            values = getattr(chrom, "values", chrom)
            if schema.matches(values):
                count += 1
        return count

    def schema_fitness(self, schema: Schema, population: Sequence[Any]) -> float:
        """Computes the mean empirical fitness of all individuals matching schema H."""
        total_fit = 0.0
        matching_count = 0
        for ind in population:
            chrom = getattr(ind, "genome", ind)
            values = getattr(chrom, "values", chrom)
            if schema.matches(values):
                fit = getattr(ind, "fitness", 0.0)
                total_fit += float(fit)
                matching_count += 1
        if matching_count == 0:
            return 0.0
        return total_fit / matching_count

    def population_mean_fitness(self, population: Sequence[Any]) -> float:
        """Computes average population fitness f_bar(t)."""
        if not population:
            return 0.0
        total = sum(float(getattr(ind, "fitness", 0.0)) for ind in population)
        return total / len(population)

    def compute_holland_lower_bound(
        self,
        schema: Schema,
        population: Sequence[Any],
        p_c: float = 0.7,
        p_m: float = 0.01,
        chromosome_length: int | None = None,
    ) -> float:
        """Evaluates the theoretical expectation lower bound for schema count in next generation."""
        m_t = self.count_instances(schema, population)
        if m_t == 0:
            return 0.0
        f_H = self.schema_fitness(schema, population)
        f_bar = self.population_mean_fitness(population)
        if f_bar <= 0.0:
            return 0.0
        fitness_ratio = f_H / f_bar
        survival = schema.survival_bound(p_c, p_m, chromosome_length)
        return round(m_t * fitness_ratio * survival, 4)


@dataclass
class HollandTrialAllocator:
    """Optimal Allocation of Trials across competing operators (Holland 1975, Chapter 5).

    Balances exploitation of high-payoff operators with exploration of less-tested operators
    using an Upper Confidence Bound (UCB) logarithmic loss minimization criterion.

    Formula:
        Priority_k = mu_hat_k + c * sqrt(2 * ln(N) / n_k)
    """

    arms: list[str]
    c_exploration: float = 1.0
    counts: dict[str, int] = field(default_factory=dict)
    total_rewards: dict[str, float] = field(default_factory=dict)
    mean_rewards: dict[str, float] = field(default_factory=dict)
    total_trials: int = 0

    def __post_init__(self):
        for arm in self.arms:
            self.counts.setdefault(arm, 0)
            self.total_rewards.setdefault(arm, 0.0)
            self.mean_rewards.setdefault(arm, 0.0)

    def select_arm(self) -> str:
        """Selects the operator arm that maximizes Holland's trial allocation index."""
        # Initial exploration: sample every untried arm at least once
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm

        best_arm = self.arms[0]
        best_priority = -float("inf")
        log_total = math.log(max(self.total_trials, 1))

        for arm in self.arms:
            n_k = self.counts[arm]
            mu_k = self.mean_rewards[arm]
            exploration_bonus = self.c_exploration * math.sqrt(2.0 * log_total / n_k)
            priority = mu_k + exploration_bonus
            if priority > best_priority:
                best_priority = priority
                best_arm = arm

        return best_arm

    def update(self, arm: str, reward: float) -> None:
        """Updates empirical mean payoff statistics following an observed mutation trial."""
        if arm not in self.counts:
            self.arms.append(arm)
            self.counts[arm] = 0
            self.total_rewards[arm] = 0.0
            self.mean_rewards[arm] = 0.0

        r = float(reward)
        self.counts[arm] += 1
        self.total_trials += 1
        self.total_rewards[arm] += r
        self.mean_rewards[arm] = self.total_rewards[arm] / self.counts[arm]

    def allocation_ratios(self) -> dict[str, float]:
        """Returns empirical fraction of total trials allocated to each operator arm."""
        if self.total_trials == 0:
            return {arm: 1.0 / len(self.arms) for arm in self.arms}
        return {arm: round(self.counts[arm] / self.total_trials, 4) for arm in self.arms}

    def describe(self) -> dict[str, Any]:
        return {
            "total_trials": self.total_trials,
            "c_exploration": self.c_exploration,
            "counts": dict(self.counts),
            "mean_rewards": {k: round(v, 4) for k, v in self.mean_rewards.items()},
            "allocation_ratios": self.allocation_ratios(),
        }
