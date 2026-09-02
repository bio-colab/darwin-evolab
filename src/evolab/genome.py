"""Genome representation: EvolabGenome contract, FloatGenome, and Individual."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

Species = str
GENOME_RANGE = 10.0


class EvolabGenome(ABC):
    """The contract that any genome representation in darwin-evolab must fulfill.

    """

    @abstractmethod
    def clone(self) -> EvolabGenome:
        """Independent deep copy."""

    @abstractmethod
    def fingerprint(self) -> str:
        """Stable hash identifying identity."""

    @abstractmethod
    def distance_to(self, other: EvolabGenome) -> float:
        """Non-negative symmetric distance metric, returns 0 for identity."""

    @abstractmethod
    def serialize(self) -> Any:
        """JSON-safe serialization for reporting."""

    @abstractmethod
    def describe(self) -> dict[str, float | int | str]:
        """Behavioral descriptors for MAP-Elites / analysis."""

    def mutate(self, rng: random.Random | None = None, **kwargs: Any) -> EvolabGenome:
        """Default mutation hook. Override in subclasses."""
        return self.clone()

    def crossover(self, other: EvolabGenome, rng: random.Random | None = None) -> EvolabGenome:
        """Default crossover hook. Override in subclasses."""
        return self.clone() if (rng or random).random() < 0.5 else other.clone()


@dataclass
class FloatGenome(EvolabGenome):
    """Vector genome implementation representing a list of real numbers."""

    values: list[float]

    @property
    def genes(self) -> list[float]:
        return self.values

    @genes.setter
    def genes(self, val: list[float]) -> None:
        self.values = list(val)

    def clone(self) -> FloatGenome:
        return FloatGenome(values=list(self.values))

    def fingerprint(self) -> str:
        raw = json.dumps([round(float(v), 6) for v in self.values])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, FloatGenome):
            raise TypeError(f"Cannot compare FloatGenome with {type(other)}")
        if len(self.values) != len(other.values):
            raise ValueError(
                f"FloatGenomes must have equal length ({len(self.values)} vs {len(other.values)})"
            )
        deltas = [abs(x - y) for x, y in zip(self.values, other.values)]
        return sum(deltas) / (len(deltas) * GENOME_RANGE)

    def structural_distance(self, other: FloatGenome) -> float:
        """Computes structural / permutation-aware distance separating refactoring from logic changes."""
        n = len(self.values)
        if n == 0 or len(other.values) == 0:
            return 0.0
        min_len = min(n, len(other.values))
        pos_deltas = sum(abs(self.values[i] - other.values[i]) for i in range(min_len)) / (min_len * GENOME_RANGE)
        val_deltas = sum(abs(x - y) for x, y in zip(sorted(self.values), sorted(other.values))) / (min_len * GENOME_RANGE)
        return 0.7 * val_deltas + 0.3 * pos_deltas

    def segment_similarity(self, other: FloatGenome, window_size: int = 2) -> float:
        """Computes maximum sub-segment similarity for crossover provenance tracking."""
        n1, n2 = len(self.values), len(other.values)
        if n1 == 0 or n2 == 0:
            return 0.0
        w = min(window_size, n1, n2)
        max_sim = 0.0
        for i in range(n1 - w + 1):
            seg1 = self.values[i : i + w]
            for j in range(n2 - w + 1):
                seg2 = other.values[j : j + w]
                dist = sum(abs(a - b) for a, b in zip(seg1, seg2)) / (w * GENOME_RANGE)
                sim = max(0.0, 1.0 - dist)
                max_sim = max(max_sim, sim)
        return max_sim

    def serialize(self) -> list[float]:
        return list(self.values)

    def describe(self) -> dict[str, float]:
        n = len(self.values)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "slope": 0.0}
        mean_v = sum(self.values) / n
        std_v = statistics.pstdev(self.values) if n > 1 else 0.0
        if n > 1:
            mid = n // 2
            first_h = sum(self.values[:mid]) / max(1, mid)
            second_h = sum(self.values[mid:]) / max(1, n - mid)
            slope_v = (second_h - first_h) / (std_v + 1e-6)
        else:
            slope_v = 0.0
        return {"mean": round(mean_v, 4), "std": round(std_v, 4), "slope": round(slope_v, 4)}

    def mutate(
        self,
        rng: random.Random | None = None,
        sigma: float = 0.5,
        clip_bounds: tuple[float, float] = (-5.0, 5.0),
        **kwargs: Any,
    ) -> FloatGenome:
        r = rng or random
        new_values = list(self.values)
        low, high = clip_bounds
        kind = kwargs.get("kind", "semantic")
        n = len(self.values)
        if n == 0:
            return FloatGenome(values=[])

        if kind == "light":
            # Charter contract: light edits 1..2 genes with fine precision
            num_genes = min(n, r.choice([1, 2]))
            for idx in r.sample(range(n), num_genes):
                delta = r.gauss(0.0, sigma)
                new_values[idx] = max(low, min(high, new_values[idx] + delta))
        else:
            for idx in range(n):
                delta = r.gauss(0.0, sigma)
                new_values[idx] = max(low, min(high, new_values[idx] + delta))
        return FloatGenome(values=new_values)

    def crossover(
        self,
        other: EvolabGenome,
        rng: random.Random | None = None,
        method: str = "single_point",
        **kwargs: Any,
    ) -> FloatGenome:
        if not isinstance(other, FloatGenome):
            raise TypeError(f"Cannot crossover FloatGenome with {type(other)}")
        if len(self.values) != len(other.values):
            raise ValueError(
                f"crossover requires equal-length parent genomes ({len(self.values)} vs {len(other.values)})"
            )
        r = rng or random
        n = len(self.values)
        if n <= 1:
            return FloatGenome(values=list(self.values))

        mode = kwargs.get("mode", method)
        if mode == "blend":
            alpha = r.uniform(0.25, 0.75)
            child_vals = [alpha * a + (1.0 - alpha) * b for a, b in zip(self.values, other.values)]
        elif mode == "uniform":
            child_vals = [a if r.random() < 0.5 else b for a, b in zip(self.values, other.values)]
        else:
            cut = r.randint(1, max(1, n - 1))
            child_vals = list(self.values[:cut]) + list(other.values[cut:])
        return FloatGenome(values=child_vals)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, idx: int | slice):
        return self.values[idx]

    def __setitem__(self, idx: int | slice, val: Any) -> None:
        self.values[idx] = val


def rank_distance(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Computes normalized Spearman-like rank distance between two numerical sequences."""
    n = min(len(v1), len(v2))
    if n <= 1:
        return 0.0
    r1 = [sorted(range(n), key=lambda i: v1[i]).index(i) for i in range(n)]
    r2 = [sorted(range(n), key=lambda i: v2[i]).index(i) for i in range(n)]
    diff = sum(abs(a - b) for a, b in zip(r1, r2))
    max_diff = n * (n - 1) / 2.0 if n > 1 else 1.0
    return diff / max(1e-6, max_diff)


def multiset_distance(v1: Sequence[float], v2: Sequence[float], range_scale: float = GENOME_RANGE) -> float:
    """Computes permutation-invariant Earth Mover's / sorted distance between two numerical multisets."""
    n = min(len(v1), len(v2))
    if n == 0:
        return 0.0
    s1, s2 = sorted(v1[:n]), sorted(v2[:n])
    return sum(abs(a - b) for a, b in zip(s1, s2)) / (n * range_scale)


@dataclass
class Individual:
    genome: EvolabGenome
    species: Species
    fitness: float = 0.0
    adjusted_fitness: float = 0.0
    lineage: dict = field(default_factory=dict)
    last_evaluated_gen: int = 0
    _generation: int = field(default=0, repr=False)
    _index: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.genome, list):
            self.genome = FloatGenome(values=list(self.genome))

    @property
    def id(self) -> str:
        return f"gen_{self._generation:02d}_ind_{self._index:02d}"

    @property
    def genes(self) -> list[float]:
        if hasattr(self.genome, "genes"):
            return self.genome.genes
        elif hasattr(self.genome, "values"):
            return self.genome.values
        return []


Genome = FloatGenome


def random_individual(
    species: Species, size: int = 16, rng: random.Random | None = None
) -> Individual:
    rng = rng or random
    return Individual(
        genome=FloatGenome(values=[rng.uniform(-5.0, 5.0) for _ in range(size)]),
        species=species,
    )
