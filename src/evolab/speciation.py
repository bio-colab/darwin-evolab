"""Speciation: distance metric, species assignment, and configuration."""
from __future__ import annotations

from .genome import GENOME_RANGE, Individual, Species

SPECIES_POOL: dict[Species, dict] = {
    "spec_dynamic_programming": {"sigma": 0.6, "crossover": "uniform"},
    "spec_bit_manipulation": {"sigma": 1.4, "crossover": "single_point"},
}

DYNAMIC_SPECIES_CFG = {"sigma": 1.0, "crossover": "uniform"}


def species_cfg(species: Species) -> dict:
    return SPECIES_POOL.get(species, DYNAMIC_SPECIES_CFG)


def genomic_distance(
    a: Individual, b: Individual, c1: float, c2: float, c3: float = 0.0
) -> float:
    """NEAT-*inspired* distance for fixed-length float genomes.

    Naming precision (external review): NOT true NEAT — no historical
    markings/innovation numbers. Composite: categorical tag disagreement +
    normalised mean absolute gene delta + optional positional max-delta.

    Positional blindness (A19 BJ-1): mean|dg| ignores gene ORDER.
    Enable c3 to separate order-permuted identities; default 0 preserves
    historical calibration.
    """
    if len(a.genome) != len(b.genome):
        raise ValueError(
            f"genomic_distance requires equal-length genomes "
            f"({len(a.genome)} vs {len(b.genome)})"
        )
    d_tag = 0.0 if a.species == b.species else 1.0
    deltas = [abs(x - y) for x, y in zip(a.genome, b.genome)]
    total = c1 * d_tag + c2 * (sum(deltas) / (len(deltas) * GENOME_RANGE))
    if c3:
        total += c3 * (max(deltas) / GENOME_RANGE)
    return total

