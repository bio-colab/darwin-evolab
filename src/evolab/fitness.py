"""Fitness landscapes: smooth proxy + ragged stress terrain."""
from __future__ import annotations

from .genome import Individual
from .landscapes import _clip


def default_fitness(ind: Individual) -> float:
    """Synthetic proxy landscape: per-gene closeness to a constant target.

    NOTE (post-audit): smooth, near-unimodal continuous problem.
    Validates mechanism dynamics, NOT transferability to rugged AST
    landscapes. Historical dead `speed_bonus` term removed with
    bit-identical outputs.
    """
    target = 3.0
    err = sum(abs(g - target) for g in ind.genome) / (len(ind.genome) * 8.0)
    return round(min(100.0, max(0.0, (1.0 - err)) * 100.0), 2)


def ragged_fitness(ind: Individual) -> float:
    """Multimodal stress landscape (Rastrigin-flavoured)."""
    n = len(ind.genome)
    base = sum((g - 3.0) ** 2 / (n * 12.5) for g in ind.genome)
    rp = (
        0.06
        * n
        * sum(__import__("math").cos(2.4 * (g - 3.0)) for g in ind.genome)
        / n
    )
    return _clip(100.0 - base * 100.0 - abs(rp) * 10.0)


