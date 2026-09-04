"""
pareto.py — True Multi-Objective Optimization and NSGA-II Engine for Darwin-Evolab.

Implements Deb's Fast Non-Dominated Sorting and Crowding Distance assignment (NSGA-II)
to discover and export true Pareto-optimal trade-off frontiers across arbitrary
objective dimensions (e.g. Silicon Power vs Area vs Delay vs Functional Correctness).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any

from .config import EngineConfig
from .genome import Individual


@dataclass(frozen=True)
class Objective:
    """Specification for an optimization objective."""

    name: str
    direction: str = "maximize"  # "maximize" or "minimize"
    weight: float = 1.0

    def is_better(self, val_a: float, val_b: float) -> bool:
        if self.direction == "maximize":
            return val_a > val_b
        return val_a < val_b

    def is_no_worse(self, val_a: float, val_b: float) -> bool:
        if self.direction == "maximize":
            return val_a >= val_b
        return val_a <= val_b


@dataclass
class MultiObjectiveResult:
    """Holds objective evaluations and Pareto metadata for an individual."""

    scores: dict[str, float]
    pareto_rank: int = 0
    crowding_distance: float = 0.0

    def get(self, name: str, default: float = 0.0) -> float:
        return self.scores.get(name, default)


def dominates(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    objectives: Sequence[Objective],
) -> bool:
    """Determines whether solution A Pareto-dominates solution B."""
    at_least_one_strictly_better = False

    for obj in objectives:
        val_a = scores_a.get(obj.name, 0.0)
        val_b = scores_b.get(obj.name, 0.0)

        if not obj.is_no_worse(val_a, val_b):
            return False
        if obj.is_better(val_a, val_b):
            at_least_one_strictly_better = True

    return at_least_one_strictly_better


def fast_non_dominated_sort(
    individuals: Sequence[Individual],
    objectives: Sequence[Objective],
    score_fn: Callable[[Individual], dict[str, float]],
) -> list[list[Individual]]:
    """Fast Non-Dominated Sorting algorithm (Deb et al., 2002).

    Complexity: O(M * N^2) where M is number of objectives, N is population size.
    Returns: List of Pareto fronts [F_0, F_1, F_2, ...] where F_0 is the non-dominated front.
    """
    n = len(individuals)
    if n == 0:
        return []

    scores = [score_fn(ind) for ind in individuals]
    s_p: list[list[int]] = [[] for _ in range(n)]
    n_p: list[int] = [0] * n
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(scores[p], scores[q], objectives):
                s_p[p].append(q)
            elif dominates(scores[q], scores[p], objectives):
                n_p[p] += 1

        if n_p[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in s_p[p]:
                n_p[q] -= 1
                if n_p[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Remove trailing empty front if present
    if fronts and not fronts[-1]:
        fronts.pop()

    # Annotate Pareto rank onto individuals
    result: list[list[Individual]] = []
    for rank_idx, front_indices in enumerate(fronts):
        front_inds: list[Individual] = []
        for idx in front_indices:
            ind = individuals[idx]
            if not hasattr(ind, "_pareto_meta") or ind._pareto_meta is None:
                ind._pareto_meta = MultiObjectiveResult(scores=scores[idx])
            ind._pareto_meta.pareto_rank = rank_idx
            ind._pareto_meta.scores = scores[idx]
            front_inds.append(ind)
        result.append(front_inds)

    return result


def calculate_crowding_distance(
    front: list[Individual],
    objectives: Sequence[Objective],
) -> list[Individual]:
    """Assigns crowding distance to solutions in a front to preserve diversity along the trade-off surface."""
    l = len(front)
    if l == 0:
        return front

    for ind in front:
        if not hasattr(ind, "_pareto_meta") or ind._pareto_meta is None:
            ind._pareto_meta = MultiObjectiveResult(scores={})
        ind._pareto_meta.crowding_distance = 0.0

    if l <= 2:
        for ind in front:
            ind._pareto_meta.crowding_distance = float("inf")
        return front

    for obj in objectives:
        front.sort(key=lambda ind: ind._pareto_meta.scores.get(obj.name, 0.0))

        front[0]._pareto_meta.crowding_distance = float("inf")
        front[-1]._pareto_meta.crowding_distance = float("inf")

        f_min = front[0]._pareto_meta.scores.get(obj.name, 0.0)
        f_max = front[-1]._pareto_meta.scores.get(obj.name, 0.0)
        denom = abs(f_max - f_min) + 1e-9

        for i in range(1, l - 1):
            if not math.isinf(front[i]._pareto_meta.crowding_distance):
                prev_val = front[i - 1]._pareto_meta.scores.get(obj.name, 0.0)
                next_val = front[i + 1]._pareto_meta.scores.get(obj.name, 0.0)
                front[i]._pareto_meta.crowding_distance += abs(next_val - prev_val) / denom

    return front


def crowded_comparison(ind_a: Individual, ind_b: Individual) -> int:
    """Crowded-comparison operator (Deb et al., 2002).

    A is preferred over B if:
    1. A has a better (lower) Pareto rank, or
    2. A and B have the same rank, but A has greater crowding distance.
    Returns: -1 if A < B (A is preferred), 1 if A > B, 0 if identical.
    """
    meta_a = getattr(ind_a, "_pareto_meta", None)
    meta_b = getattr(ind_b, "_pareto_meta", None)

    if meta_a is None or meta_b is None:
        return 0

    if meta_a.pareto_rank < meta_b.pareto_rank:
        return -1
    elif meta_a.pareto_rank > meta_b.pareto_rank:
        return 1
    else:
        if meta_a.crowding_distance > meta_b.crowding_distance:
            return -1
        elif meta_a.crowding_distance < meta_b.crowding_distance:
            return 1
        return 0


class NSGA2Engine:
    """Non-Dominated Sorting Genetic Algorithm II (NSGA-II) Optimization Engine.

    Elitist multi-objective evolutionary engine discovering Pareto-optimal fronts
    for complex multi-criteria engineering systems (such as Silicon circuits).
    """

    def __init__(
        self,
        objectives: Sequence[Objective],
        evaluate_vector_fn: Callable[[Individual], dict[str, float]],
        config: EngineConfig | None = None,
        population_size: int = 16,
        generations: int = 20,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        seed: int | None = None,
    ) -> None:
        self.objectives = list(objectives)
        self.evaluate_vector_fn = evaluate_vector_fn
        self.config = config or EngineConfig(population_size=population_size, generations=generations, seed=seed)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = random.Random(seed)
        self.history: list[dict[str, Any]] = []
        self.best_front: list[Individual] = []

    def _binary_tournament_selection(self, population: list[Individual]) -> Individual:
        a = self.rng.choice(population)
        b = self.rng.choice(population)
        cmp = crowded_comparison(a, b)
        return a if cmp <= 0 else b

    def run(
        self,
        initial_population: list[Individual] | None = None,
        generations: int | None = None,
    ) -> dict[str, Any]:
        gens = generations or self.generations
        n = self.population_size

        if initial_population:
            pop = [ind.clone() for ind in initial_population]
        else:
            raise ValueError("NSGA2Engine requires an initial_population to be supplied.")

        # Ensure initial population matches target size
        while len(pop) < n:
            pop.append(self.rng.choice(pop).clone())
        pop = pop[:n]

        # Initial ranking
        fronts = fast_non_dominated_sort(pop, self.objectives, self.evaluate_vector_fn)
        for f in fronts:
            calculate_crowding_distance(f, self.objectives)

        for gen in range(1, gens + 1):
            # 1. Generate Offspring Population Q_t of size N
            offspring: list[Individual] = []
            while len(offspring) < n:
                parent1 = self._binary_tournament_selection(pop)
                parent2 = self._binary_tournament_selection(pop)

                if hasattr(parent1.genome, "crossover") and self.rng.random() < self.crossover_rate:
                    child_g = parent1.genome.crossover(parent2.genome, rng=self.rng)
                else:
                    child_g = parent1.genome.clone()

                if hasattr(child_g, "mutate") and self.rng.random() < self.mutation_rate:
                    child_g = child_g.mutate(rng=self.rng)

                offspring.append(Individual(genome=child_g, species=parent1.species))

            # 2. Form R_t = P_t U Q_t (size 2N)
            combined = pop + offspring

            # 3. Fast Non-Dominated Sorting on combined population
            all_fronts = fast_non_dominated_sort(combined, self.objectives, self.evaluate_vector_fn)

            # 4. Fill Next Generation P_{t+1}
            new_pop: list[Individual] = []
            front_idx = 0

            while front_idx < len(all_fronts) and len(new_pop) + len(all_fronts[front_idx]) <= n:
                calculate_crowding_distance(all_fronts[front_idx], self.objectives)
                new_pop.extend(all_fronts[front_idx])
                front_idx += 1

            if len(new_pop) < n and front_idx < len(all_fronts):
                last_front = all_fronts[front_idx]
                calculate_crowding_distance(last_front, self.objectives)
                # Sort descending by crowding distance
                last_front.sort(key=lambda ind: ind._pareto_meta.crowding_distance, reverse=True)
                remainder = n - len(new_pop)
                new_pop.extend(last_front[:remainder])

            pop = new_pop
            fronts = fast_non_dominated_sort(pop, self.objectives, self.evaluate_vector_fn)
            for f in fronts:
                calculate_crowding_distance(f, self.objectives)

            self.best_front = fronts[0] if fronts else []

            # Record generational statistics
            gen_stat = {
                "generation": gen,
                "pareto_front_size": len(self.best_front),
                "total_fronts": len(fronts),
                "front_0_metrics": [ind._pareto_meta.scores for ind in self.best_front[:5]],
            }
            self.history.append(gen_stat)

        return {
            "generations": gens,
            "pareto_front_size": len(self.best_front),
            "front_0": [
                {
                    "scores": ind._pareto_meta.scores,
                    "crowding_distance": ind._pareto_meta.crowding_distance,
                    "genome_fingerprint": ind.genome.fingerprint() if hasattr(ind.genome, "fingerprint") else str(ind),
                }
                for ind in self.best_front
            ],
            "history": self.history,
        }

    def export_pareto_front(self, output_path: str | Path) -> None:
        """Exports the non-dominated Pareto front trade-off matrix to JSON."""
        p = Path(output_path)
        if p.parent and str(p.parent):
            p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "algorithm": "NSGA-II",
            "objectives": [
                {"name": obj.name, "direction": obj.direction, "weight": obj.weight}
                for obj in self.objectives
            ],
            "pareto_front_size": len(self.best_front),
            "frontier": [
                {
                    "scores": ind._pareto_meta.scores,
                    "crowding_distance": ind._pareto_meta.crowding_distance if not math.isinf(ind._pareto_meta.crowding_distance) else "inf",
                    "genome": ind.genome.serialize() if hasattr(ind.genome, "serialize") else str(ind.genome),
                }
                for ind in self.best_front
            ],
        }
        p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_silicon_multiobjective_evaluator(
    truth_table: Sequence[tuple[Sequence[int], Sequence[int]]],
    switching_stream: Sequence[Sequence[int]] | None = None,
) -> tuple[list[Objective], Callable[[Individual], dict[str, float]]]:
    """Constructs the canonical 4-objective Silicon optimization problem:
    1. Correctness (%) [maximize]
    2. Power (Dynamic switching toggles) [minimize]
    3. Delay (Critical path levels) [minimize]
    4. Area (Active gate count) [minimize]
    """
    objectives = [
        Objective("correctness", direction="maximize"),
        Objective("power", direction="minimize"),
        Objective("delay", direction="minimize"),
        Objective("area", direction="minimize"),
    ]

    if switching_stream is None:
        r = random.Random(42)
        stream = [vec for vec, _ in truth_table]
        for _ in range(16):
            stream.append(r.choice([vec for vec, _ in truth_table]))
    else:
        stream = list(switching_stream)

    def evaluate_silicon_vector(ind: Individual) -> dict[str, float]:
        genome = getattr(ind, "genome", ind)
        if not hasattr(genome, "evaluate_truth_table"):
            return {"correctness": 0.0, "power": 999.0, "delay": 999.0, "area": 999.0}

        tt_metrics = genome.evaluate_truth_table(truth_table)
        sw_metrics = genome.evaluate_switching_activity(stream)

        return {
            "correctness": round(tt_metrics.truth_table_accuracy * 100.0, 2),
            "power": float(sw_metrics.total_toggles),
            "delay": float(tt_metrics.critical_path_delay),
            "area": float(tt_metrics.active_gate_count),
        }

    return objectives, evaluate_silicon_vector
