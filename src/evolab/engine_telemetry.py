"""engine_telemetry.py — Diversity, causality, and descriptor telemetry for EvolutionEngine.

Extracted from engine.py for modular decomposition.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Callable


def _desc_mean(genome: Any) -> float:
    if hasattr(genome, "describe"):
        desc = genome.describe()
        for k in ("mean", "node_count", "hunk_count", "lines_added"):
            if k in desc:
                return float(desc[k])
    if hasattr(genome, "values"):
        vals = genome.values
        return sum(vals) / max(len(vals), 1)
    if hasattr(genome, "genes"):
        return sum(genome.genes) / max(len(genome.genes), 1)
    if isinstance(genome, list):
        return sum(genome) / max(len(genome), 1)
    if hasattr(genome, "fingerprint"):
        return float(int(genome.fingerprint()[:4], 16) % 100) / 10.0
    return 0.0


def _desc_std(genome: Any) -> float:
    if hasattr(genome, "describe"):
        desc = genome.describe()
        for k in ("slope", "std", "max_depth", "lines_removed", "stmt_count"):
            if k in desc:
                return float(desc[k])
    if hasattr(genome, "values"):
        vals = genome.values
        if len(vals) > 1:
            mid = len(vals) // 2
            f_h = sum(vals[:mid]) / max(1, mid)
            s_h = sum(vals[mid:]) / max(1, len(vals) - mid)
            std_v = statistics.pstdev(vals)
            return (s_h - f_h) / (std_v + 1e-6)
        return 0.0
    if hasattr(genome, "genes"):
        g = genome.genes
        return statistics.pstdev(g) if len(g) > 1 else 0.0
    if isinstance(genome, list):
        return statistics.pstdev(genome) if len(genome) > 1 else 0.0
    if hasattr(genome, "fingerprint"):
        return float(int(genome.fingerprint()[4:8], 16) % 50) / 10.0
    return 0.0


def pattern_similarity(a: tuple, b: tuple) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot / (na * nb)))


def build_causal_summary(causal_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate causal event statistics per mutation type."""
    by_type: dict[str, list[float]] = {}
    for e in causal_events:
        t = e["mutation_type"]
        by_type.setdefault(t, []).append(e["fitness_delta"])
    summary = {}
    for t, deltas in sorted(by_type.items()):
        n = len(deltas)
        positive = sum(1 for d in deltas if d > 0)
        summary[t] = {
            "count": n,
            "mean_delta": round(statistics.mean(deltas), 4) if n else 0.0,
            "positive_rate": round(positive / n, 3) if n else 0.0,
            "std_delta": round(statistics.stdev(deltas), 4) if n > 1 else 0.0,
        }
    return {
        "total_events": len(causal_events),
        "by_mutation_type": summary,
        "note": "fitness_delta = child_fitness - mean(parent_fitness); "
                "correlation ≠ causation",
    }


def calculate_population_diversity(
    pop: list[Any], distance_fn: Callable[[Any, Any], float]
) -> float:
    """Honest mean pairwise genomic distance in [0, ~1]."""
    try:
        n = len(pop)
        if n < 2:
            return 0.0
        # Cap O(n^2) cost: sample at most 16 individuals deterministically.
        sample = pop[:16] if n > 16 else pop
        total = 0.0
        count = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                try:
                    total += float(distance_fn(sample[i], sample[j]))
                    count += 1
                except Exception:
                    continue
        if count:
            return round(total / count, 4)
        fits = [float(getattr(ind, "fitness", 0.0)) for ind in pop]
        spread = max(fits) - min(fits) if fits else 0.0
        return round(max(0.0, min(spread / 100.0, 1.0)), 4)
    except Exception:
        return 0.0


def count_cache_misses(fitness_fn: Any) -> int:
    """Best-effort cache-miss total through wrapper chains."""
    try:
        seen = set()
        node = fitness_fn
        for _ in range(6):
            if node is None or id(node) in seen:
                break
            seen.add(id(node))
            try:
                misses = getattr(node, "misses", None)
                if isinstance(misses, int):
                    return max(int(misses), 0)
            except Exception:
                pass
            try:
                stats = getattr(node, "stats", None)
                if callable(stats):
                    st = stats()
                else:
                    st = stats
                if isinstance(st, dict) and isinstance(st.get("misses"), int):
                    return max(int(st["misses"]), 0)
            except Exception:
                pass
            try:
                node = getattr(node, "raw", None)
            except Exception:
                break
        return 0
    except Exception:
        return 0


def calculate_unique_programs(pop: list[Any]) -> dict[str, Any]:
    """Computes count and ratio of unique program representations in the population.

    Supports:
    - Code genomes (RepairGenome, ASTGenome, CSTGenome) via to_code() or edit_keys()
    - Symbolic / Circuit genomes via string / netlist
    - Numeric / Bitvector genomes via tuple representation
    """
    if not pop:
        return {"unique_count": 0, "unique_ratio": 0.0}
    seen = set()
    for ind in pop:
        g = getattr(ind, "genome", ind)
        if hasattr(g, "to_code") and callable(g.to_code):
            try:
                rep = g.to_code()
            except Exception:
                rep = str(getattr(g, "edits", g))
        elif hasattr(g, "edit_keys") and callable(g.edit_keys):
            rep = tuple(sorted(g.edit_keys()))
        elif isinstance(g, (list, tuple)):
            rep = tuple(g)
        elif hasattr(g, "fingerprint") and callable(g.fingerprint):
            try:
                rep = g.fingerprint()
            except Exception:
                rep = str(g)
        else:
            rep = str(g)
        seen.add(rep)

    count = len(seen)
    ratio = round(count / max(1, len(pop)), 4)
    return {"unique_count": count, "unique_ratio": ratio}


def calculate_clonal_drift_metrics(
    pop: list[Any],
    best_fitness_history: list[float] | None = None,
    stagnation_window: int = 5,
    min_clonal_ratio: float = 0.5,
) -> dict[str, Any]:
    """Detects when an apparent rise in population mean fitness is merely clonal drift
    (blind replication of a single genotype) rather than true evolutionary exploration.

    Returns:
    - unique_expressions: count of distinct representations
    - max_clone_count: number of copies of the most frequent genotype
    - clonal_ratio: proportion of population belonging to duplicated clones
    - is_clonal_drift_stagnation: True if best fitness has stagnated while clones dominate
    """
    if not pop:
        return {
            "unique_expressions": 0,
            "max_clone_count": 0,
            "clonal_ratio": 0.0,
            "is_clonal_drift_stagnation": False,
            "dominant_genotype": None,
        }

    from collections import Counter
    counts: Counter = Counter()

    for ind in pop:
        g = getattr(ind, "genome", ind)
        if hasattr(g, "root") and hasattr(g.root, "to_pretty_str"):
            rep = g.root.to_pretty_str()
        elif hasattr(g, "to_code") and callable(g.to_code):
            try:
                rep = g.to_code()
            except Exception:
                rep = str(g)
        elif hasattr(g, "fingerprint") and callable(g.fingerprint):
            try:
                rep = g.fingerprint()
            except Exception:
                rep = str(g)
        else:
            rep = str(g)
        counts[rep] += 1

    total = len(pop)
    unique_count = len(counts)
    most_common_rep, max_clones = counts.most_common(1)[0]
    clonal_ratio = round((total - unique_count) / max(1, total), 4)

    stagnated = False
    if best_fitness_history and len(best_fitness_history) >= stagnation_window:
        recent = best_fitness_history[-stagnation_window:]
        if max(recent) - min(recent) < 1e-5:
            stagnated = True

    is_stagnation = bool(stagnated and (max_clones / total >= min_clonal_ratio or clonal_ratio >= min_clonal_ratio))

    return {
        "unique_expressions": unique_count,
        "max_clone_count": max_clones,
        "dominant_genotype": most_common_rep,
        "clonal_ratio": clonal_ratio,
        "is_clonal_drift_stagnation": is_stagnation,
    }


__all__ = [
    "_desc_mean",
    "_desc_std",
    "pattern_similarity",
    "build_causal_summary",
    "calculate_population_diversity",
    "calculate_unique_programs",
    "calculate_clonal_drift_metrics",
    "count_cache_misses",
]
