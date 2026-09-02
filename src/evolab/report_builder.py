"""Assemble the run report after the evolution loop finishes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .genome import Individual


def _pareto_front(pop: list[Individual]) -> list[Individual]:
    def dominates(a: Individual, b: Individual) -> bool:
        if isinstance(a.genome, list):
            a_mem = -sum(abs(g) for g in a.genome)
            b_mem = -sum(abs(g) for g in b.genome)
        elif hasattr(a.genome, "describe"):
            a_desc = a.genome.describe()
            b_desc = b.genome.describe()
            a_mem = -float(a_desc.get("node_count", a_desc.get("hunk_count", 0)))
            b_mem = -float(b_desc.get("node_count", b_desc.get("hunk_count", 0)))
        else:
            a_mem = 0.0
            b_mem = 0.0
        ge = (a.fitness >= b.fitness) and (a_mem >= b_mem)
        gt = (a.fitness > b.fitness) or (a_mem > b_mem)
        return ge and gt

    return [
        ind
        for i, ind in enumerate(pop)
        if not any(dominates(o, ind) for j, o in enumerate(pop) if j != i)
    ]


def _lineage_edges(population: list[Individual]) -> tuple[list[dict], dict[str, int]]:
    lineage_edges = [
        {"child": ind.id, **ind.lineage}
        for ind in population
        if ind.lineage
    ]
    operators: dict[str, int] = {}
    for e in lineage_edges:
        op = e.get("operator", "?")
        base = "crossover+mutation_light" if op.endswith("_light") else (
            "crossover+mutation_semantic" if op.endswith("_semantic") else op
        )
        operators[base] = operators.get(base, 0) + 1
    return lineage_edges, operators


def _best_payload(engine: Any, best_ever: Individual, final_gen: int) -> dict[str, Any]:
    holdout = None
    suspicion_top: list[dict] = []
    if best_ever is not None and hasattr(engine.fitness_fn, "evaluate"):
        try:
            final_res = engine.fitness_fn.evaluate(best_ever)
            holdout = final_res.passed_holdout
        except Exception:
            holdout = None
    smap = getattr(engine.fitness_fn, "last_suspicion_map", None)
    if smap is not None and hasattr(smap, "get_top_nodes"):
        suspicion_top = [
            {
                "line": n.line_no,
                "node": n.node_type,
                "score": n.suspicion_score,
            }
            for n in smap.get_top_nodes(top_k=5, min_score=0.0)
        ]

    best_payload = {
        "id": f"gen_{best_ever._generation:02d}_ind_{best_ever._index:02d}",
        "fitness": best_ever.fitness,
        "species": best_ever.species,
        "genome_size": len(best_ever.genome),
        "last_evaluated_gen": getattr(best_ever, "last_evaluated_gen", final_gen),
    }
    if holdout is not None:
        best_payload["passed_holdout"] = holdout
    if suspicion_top:
        best_payload["suspicion_top"] = suspicion_top
    if best_ever is not None and hasattr(best_ever.genome, "to_code"):
        try:
            best_payload["code"] = best_ever.genome.to_code()
        except Exception:
            pass
    return best_payload


def build_run_report(
    engine: Any,
    *,
    population: list[Individual],
    history: list[dict],
    species_history: list,
    stagnation_events: list,
    early_stop: bool,
    best_ever: Individual,
    exploit_start: int,
    final_gen: int,
) -> dict[str, Any]:
    from .engine import ENGINE_VERSION, SCHEMA_VERSION

    species_distribution: dict[str, int] = {}
    for ind in population:
        species_distribution[ind.species] = (
            species_distribution.get(ind.species, 0) + 1
        )

    front = _pareto_front(sorted(population, key=lambda i: i.fitness, reverse=True))
    lineage_edges, operators = _lineage_edges(population)
    filled = len(engine._archive)
    total_cells = engine.me_grid_x * engine.me_grid_y
    total_mutations = sum(engine._mutation_stats.values())
    code_mode = bool(getattr(engine, "_code_mode", False))

    report: dict[str, Any] = {
        "total_generations": final_gen,
        "total_candidates_evaluated": final_gen * engine.population_size,
        "best_individual": _best_payload(engine, best_ever, final_gen),
        "species_distribution": {
            k: v for k, v in sorted(species_distribution.items())
        },
        "early_stop_triggered": early_stop,
        "history": history,
        "species_history": species_history,
        "stagnation_events": stagnation_events,
        "speciation": {
            "metric": "delta = c1*tag_diff + c2*mean|dg|/GENOME_RANGE"
                      " + c3*max|dg|/GENOME_RANGE",
            "family": "composite tag+numeric (NEAT-inspired; no historical markings)",
            "enabled": engine.speciation_enabled,
            "c1": engine.dist_c1,
            "c2": engine.dist_c2,
            "c3": engine.dist_c3,
            "threshold": engine.speciation_threshold,
            "dynamic_species_created": engine._dyn_species_seq,
        },
        "lineage_summary": {
            "note": "final-generation edges; full genealogy graph ships with "
                    "Genome abstraction (v4 design, spec 10-d5)",
            "edges_final_gen": len(lineage_edges),
            "operators_final_gen": dict(sorted(operators.items())),
            "sample_edges": lineage_edges[:6],
        },
        "operator_history": engine._operator_history,
        "causal_events": engine._causal_events,
        "causal_summary": engine._build_causal_summary(),
        "fitness_sharing": False if code_mode else engine.fitness_sharing,
        "hybrid_mutation": {
            "light_share": engine.hybrid_light_share,
            **engine._mutation_stats,
            "total_mutations": total_mutations,
            "est_llm_calls_saved_pct": round(
                100.0 * engine._mutation_stats["light"] / total_mutations, 1
            )
            if total_mutations
            else 0.0,
            "mutation_accounting": {
                "semantic_uses_mutation_rate": True,
                "light_uses_mutation_rate": False,
                "light_genes_per_event": "1..2 (uniform)",
                "gene_edit_rate_pct": round(
                    100.0 * engine._gene_edits / engine._gene_slots, 2
                )
                if engine._gene_slots
                else 0.0,
            },
        },
        "map_elites": {
            "grid": [engine.me_grid_x, engine.me_grid_y],
            "descriptors": [
                getattr(fn, "__name__", f"desc_{i}")
                for i, fn in enumerate(engine._descriptor_fns)
            ],
            "filled_cells": filled,
            "coverage_pct": round(100.0 * filled / total_cells, 1),
            "archive_best_fitness": max(
                (ind.fitness for ind in engine._archive.values()), default=0.0
            ),
        },
        "pareto_front": {
            "objectives": ["fitness_max", "genome_compactness_max"],
            "size": len(front),
            "members": [
                {"id": f"gen_{i._generation:02d}_ind_{i._index:02d}",
                 "fitness": i.fitness}
                for i in front[:8]
            ],
        },
        "sharing_schedule": {
            "mode": engine.sharing_mode,
            "exploit_after_gen": exploit_start,
        },
        "sharing_effective": {
            "exploration_phase": (
                (not code_mode)
                and bool(engine.fitness_sharing and engine.sharing_mode != "off")
            ),
            "exploitation_phase": (
                (not code_mode)
                and engine.sharing_mode == "static"
                and engine.fitness_sharing
            ),
        },
        "decision_log": engine._decision_log,
        "constraints": {
            "n_hard": len(engine._hard_constraints),
            "violations_total": engine._constraint_violations,
        },
        "memory": {
            "enabled": engine.memory_enabled,
            "entries_active": len(engine._memory_bank.entries),
            "graveyard_size": len(engine._memory_bank.graveyard),
            "recalls_total": sum(
                e.recall_count for e in engine._memory_bank.entries
            ) + sum(e.recall_count for e in engine._memory_bank.graveyard),
            "injections_total": sum(
                s["injected"] for s in getattr(engine, "_injection_stats", [])
            ),
            "memory_evals_used": engine._memory_evals_used,
            "last_change": (
                engine._last_change.value if engine._last_change else None
            ),
            "quarantine_until_gen": engine._quarantine_until_gen,
        },
        "config": {
            "population_size": engine.population_size,
            "elite_count": engine.elite_count,
            "mutation_rate": engine.mutation_rate,
            "early_stop_fitness": engine.early_stop_fitness,
            "seed": engine.rng_seed,
            "genome_size": engine.genome_size,
            "stagnation_patience": engine.stagnation_patience,
            "immigrant_fraction": engine.immigrant_fraction,
            "fitness_sharing": False if code_mode else engine.fitness_sharing,
            "dist_c1": engine.dist_c1,
            "dist_c2": engine.dist_c2,
            "speciation_threshold": engine.speciation_threshold,
            "hybrid_light_share": engine.hybrid_light_share,
            "speciation_enabled": engine.speciation_enabled,
            "mutation_enabled": engine.mutation_enabled,
            "sharing_mode": "off" if code_mode else engine.sharing_mode,
            "exploit_after_frac": engine.exploit_after_frac,
            "fitness_range": list(engine.fitness_range),
            "evaluator": engine._evaluator_name,
            "dist_c3": engine.dist_c3,
            "distance_metric": (
                "custom" if engine._custom_distance is not None
                else engine.distance_metric
            ),
            "eval_repeats": engine.eval_repeats,
            "stability_penalty": engine.stability_penalty,
        },
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine_version": ENGINE_VERSION,
        "schema_version": SCHEMA_VERSION,
    }
    if engine.record_population_snapshots:
        report["population_snapshots"] = engine._population_snapshots
    if engine.record_archive_solutions:
        report["archive_cells"] = [
            {
                "cell": list(cell),
                "fitness": ind.fitness,
                "genome": ind.genome.serialize() if hasattr(ind.genome, "serialize") else list(ind.genome),
                "last_evaluated_gen": getattr(ind, "last_evaluated_gen", 0),
            }
            for cell, ind in sorted(engine._archive.items())
        ]
    return report
