"""Causal Evolutionary Memory (CEM) core — darwin-evolab v4.0.

Four layers per the v4 proposal, adapted honestly to the numeric-genome
simulation (see class docstrings for scope notes):

  L1 ChangeDetector        — multi-signal CUSUM (fitness + species counts)
  L2 MemoryBank (TMI)      — entries carry context, staleness, reliability
  L3 MemoryInjector        — dose-calibrated, weakest-replacement, sandboxed
  L4 Immunity              — exponential staleness, euthanasia, quarantine

Budget honesty: every sandboxed evaluation performed by the injector is
counted in ``memory_evals_used`` and surfaced in the run report — it is
part of the experiment's cost, never free.
"""
from __future__ import annotations

import enum
import math
import random
import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ChangeType(enum.Enum):
    STABLE = "STABLE"
    DRIFT = "DRIFT"
    COLLAPSE = "COLLAPSE"
    SHOCK = "SHOCK"


@dataclass
class NullInterventionProof:
    """Records that a null intervention produced identical trajectory.

    Used by the Null Intervention Contract (audit A25): any mechanism
    with dose=0, empty bank, or full quarantine must produce the exact
    same trajectory as its disabled counterpart.
    """
    variant: str
    trajectory_hash: str
    best_fitness: float
    total_generations: int


class ChangeDetector:
    """Multi-signal CUSUM detector over fitness and species richness."""

    def __init__(self, window: int = 20, threshold_k: float = 0.5,
                 threshold_h: float = 5.0) -> None:
        self.window = window
        self.threshold_k = threshold_k
        self.threshold_h = threshold_h
        self.fitness_history: deque = deque(maxlen=window)
        self.species_counts: deque = deque(maxlen=window)
        self.cusum_fitness = 0.0
        self.cusum_species = 0.0

    def detect(self, fitness: float, n_species: int) -> ChangeType:
        mu = (statistics.mean(self.fitness_history)
              if self.fitness_history else fitness)
        prev_species = (self.species_counts[-1] if self.species_counts
                        else n_species)

        self.cusum_fitness = max(
            0.0, self.cusum_fitness + (mu - fitness) - self.threshold_k)
        species_drop = max(0, prev_species - n_species)
        self.cusum_species = max(
            0.0, self.cusum_species + species_drop - 0.5)

        self.fitness_history.append(fitness)
        self.species_counts.append(n_species)

        if self.cusum_fitness > self.threshold_h:
            if self.cusum_species > 2.0:
                return ChangeType.SHOCK
            return ChangeType.DRIFT
        if species_drop > max(1, n_species // 2) and n_species >= 1:
            return ChangeType.COLLAPSE
        return ChangeType.STABLE


@dataclass
class MemoryEntry:
    genome: Any
    fitness_at_archive: float
    generation_archived: int
    survival_duration: int = 0
    recall_count: int = 0
    successes: int = 0
    staleness_generations: float = 0.0
    signature: tuple = ()
    quarantined: bool = False
    phenotypic_hash: str | None = None
    last_evaluated_gen: int = 0

    @property
    def success_rate(self) -> float:
        if self.recall_count <= 0:
            return 0.5
        return self.successes / self.recall_count

    def age(self, current_generation: int) -> int:
        return max(0, current_generation - self.generation_archived)


class TemporalMemoryIndex:
    """Layer-2 index: entries + staleness ageing + euthanasia protocol."""

    def __init__(self, staleness_tau: float = 100.0,
                 euthanasia_after_recalls: int = 3,
                 euthanasia_max_failure_rate: float = 0.7,
                 max_entries: int = 200) -> None:
        self.tau = staleness_tau
        self.euthanasia_recalls = euthanasia_after_recalls
        self.euthanasia_fail_rate = euthanasia_max_failure_rate
        self.max_entries = max_entries
        self.entries: list[MemoryEntry] = []
        self.graveyard: list[MemoryEntry] = []

    def upsert(self, genome: Any, fitness: float,
               generation: int, signature: tuple,
               phenotypic_hash: str | None = None) -> MemoryEntry:
        # Check exact genotypic match
        for e in self.entries:
            if e.genome == genome:
                e.fitness_at_archive = max(e.fitness_at_archive, fitness)
                e.generation_archived = generation
                e.last_evaluated_gen = generation
                e.signature = signature
                if phenotypic_hash:
                    e.phenotypic_hash = phenotypic_hash
                return e

        # Check phenotypic equivalence hash match
        if phenotypic_hash is not None:
            for e in self.entries:
                if e.phenotypic_hash == phenotypic_hash:
                    if fitness >= e.fitness_at_archive:
                        g_stored = genome.clone() if hasattr(genome, "clone") else list(genome)
                        e.genome = g_stored
                        e.fitness_at_archive = fitness
                        e.generation_archived = generation
                        e.last_evaluated_gen = generation
                        e.signature = signature
                    return e

        g_stored = genome.clone() if hasattr(genome, "clone") else list(genome)
        entry = MemoryEntry(genome=g_stored, fitness_at_archive=fitness,
                            generation_archived=generation,
                            signature=signature,
                            phenotypic_hash=phenotypic_hash,
                            last_evaluated_gen=generation)
        self.entries.append(entry)
        self._enforce_cap()
        return entry

    def _enforce_cap(self) -> None:
        """Cap bank size: evict lowest (fitness × recency) first (A22 fix)."""
        while len(self.entries) > self.max_entries:
            worst = min(
                self.entries,
                key=lambda e: (
                    e.fitness_at_archive
                    * math.exp(-e.staleness_generations / max(self.tau, 1e-9))
                ),
            )
            self.entries.remove(worst)

    def age_all(self, generations_elapsed: float) -> None:
        """Exponential staleness (audit-facing: documented, never deletes)."""
        for e in self.entries:
            e.survival_duration += generations_elapsed
            e.staleness_generations += generations_elapsed

    def staleness_factor(self, e: MemoryEntry) -> float:
        raw = 1.0 - math.exp(-e.staleness_generations / max(self.tau, 1e-9))
        # high success rate slows decay
        return raw * (1.0 - 0.5 * e.success_rate)

    def reusability(self, e: MemoryEntry,
                    current_signature: tuple) -> float:
        if not e.signature or not current_signature:
            sim = 0.0
        else:
            a, b = e.signature, current_signature
            na = max(1e-9, math.sqrt(sum(x * x for x in a)))
            nb = max(1e-9, math.sqrt(sum(x * x for x in b)))
            dot = sum(x * y for x, y in zip(a, b))
            sim = max(-1.0, min(1.0, dot / (na * nb)))
        recency = math.exp(-self.staleness_factor(e))
        reliability = e.success_rate if e.recall_count > 0 else 0.5
        alignment = max(0.0, min(1.0, (sim + 1.0) / 2.0))
        return 0.5 * alignment + 0.3 * reliability + 0.2 * recency

    def top_k(self, k: int, current_signature: tuple) -> list[MemoryEntry]:
        pool = [e for e in self.entries if not e.quarantined]
        pool.sort(key=lambda e: self.reusability(e, current_signature),
                  reverse=True)
        return pool[:max(0, k)]

    def euthanize_sweep(self) -> int:
        doomed = [
            e for e in self.entries
            if e.recall_count >= self.euthanasia_recalls
            and e.success_rate < (1.0 - self.euthanasia_fail_rate)
        ]
        for e in doomed:
            self.graveyard.append(e)
            self.entries.remove(e)
        if len(self.graveyard) > self.max_entries:
            self.graveyard = self.graveyard[-self.max_entries:]
        return len(doomed)


DOSE_BY_CHANGE = {
    ChangeType.STABLE: 0.0,
    ChangeType.DRIFT: 0.05,
    ChangeType.COLLAPSE: 0.10,
    ChangeType.SHOCK: None,   # resolved to max_rate at inject-time
}


class MemoryInjector:
    """Layer-3: dose-calibrated, sandboxed, weakest-replacement injection."""

    def __init__(self, max_injection_rate: float = 0.15) -> None:
        self.max_rate = max_injection_rate

    @staticmethod
    def compute_dose(change: ChangeType, max_rate: float) -> float:
        base = DOSE_BY_CHANGE[change]
        return max_rate if base is None else base

    def inject(self, population: list, bank: TemporalMemoryIndex,
               sandbox_score: Callable[[list[float]], float],
               change: ChangeType, current_signature: tuple,
               rng: random.Random, generation: int) -> dict:
        dose = self.compute_dose(change, self.max_rate)
        # A22 fix + A25 null-intervention contract: dose > 0 must yield
        # at least one candidate, but dose=0 or empty bank is a true no-op
        if dose <= 0 or not bank.entries:
            return {"dose": dose, "requested": 0, "injected": 0,
                    "failed": 0, "euthanized": 0, "generation": generation,
                    "null_intervention": True}
        k = max(1, round(len(population) * dose))
        stats = {"dose": dose, "requested": k, "injected": 0,
                 "failed": 0, "euthanized": 0}
        if not bank.entries:
            return stats

        candidates = bank.top_k(k, current_signature)
        fitnesses = [ind.fitness for ind in population]
        median_f = statistics.median(fitnesses)

        for entry in candidates:
            entry.recall_count += 1
            # Forced Staleness Re-evaluation on current landscape:
            score = sandbox_score(entry.genome)
            entry.last_evaluated_gen = generation
            entry.fitness_at_archive = score
            if score >= median_f:
                weakest = min(range(len(population)),
                              key=lambda i: population[i].fitness)
                target = population[weakest]
                target.genome = entry.genome.clone() if hasattr(entry.genome, "clone") else list(entry.genome)
                target.fitness = float(score)
                target.last_evaluated_gen = generation
                entry.successes += 1
                stats["injected"] += 1
            else:
                stats["failed"] += 1
        # exact-count accounting replaces the proposal's EMA: success_rate is
        # successes/recall_count, so failures already penalise naturally.
        stats["euthanized"] = bank.euthanize_sweep()
        stats["generation"] = generation
        return stats
