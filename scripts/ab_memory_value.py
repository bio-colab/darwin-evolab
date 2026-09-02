#!/usr/bin/env python3
"""Phase-3 A/B harness: does cross-run experience memory reduce the number of
evaluations needed to reach the first holdout-passing repair?

Pre-registered protocol (written down BEFORE any measurement — no scenario,
seed or budget may be tuned after peeking at results; that would be selection
bias dressed up as evidence):

  Teacher phase   For each scenario, ``--teachers`` GA runs (seeds 9901..)
                  record real experiences into one store. Teachers run with
                  the prior DISABLED: inherited knowledge must be plain
                  recorded experience, not prior-influenced experience.
  Test phase      For each seed s in ``--seeds``, two PAIRED runs:
                    control arm  recorder with prior_enabled=False -> the
                                 exact pre-memory behavior, identical
                                 instrumentation, fresh empty store.
                    memory arm   recorder with the prior enabled, reading a
                                 FILE COPY of the teacher snapshot. The prior
                                 is frozen at run start (cache_ttl=1e9) so
                                 every run inherits exactly the same
                                 knowledge; student writes go to the copy and
                                 never leak into later runs.
  Metric          evals-to-first-holdout-success via
                  ExperienceStore.run_metrics(run_id); a run that never
                  passes holdout is censored at its total evaluation count.
  Primary number  search_efficiency_gain = 1 - M/N where M is the mean cost
                  of the memory arm and N the mean cost of the control arm
                  over the same paired seeds.

Decision rule (registered here, before the run):
  The memory prior KEEPS default-on status only if overall gain > 0 AND the
  non-tied paired wins are >= losses. Otherwise this harness documents that
  cross-run memory did not prove value on the repository's own benchmarks —
  and that is the reported result, whatever it is.

Usage (from the repository root):
  PYTHONPATH=src python scripts/ab_memory_value.py
  PYTHONPATH=src python scripts/ab_memory_value.py --seeds 1-5 --out /tmp/ab.json

Stdlib only. Deterministic given fixed --seeds/--teachers/--generations/
--population: teacher seeds, scenario set and budgets are all fixed inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab import EngineConfig, EvolutionEngine  # noqa: E402
from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population  # noqa: E402
from evolab.experience import (  # noqa: E402
    ExperienceStore,
    attach_experience_recorder,
    problem_fingerprint,
)

TEACHER_SEED_START = 9901
FROZEN_PRIOR_KWARGS = {"cache_ttl": 1e9}


def clean_env() -> None:
    """Harness controls wiring explicitly; ambient env must not leak in."""
    os.environ.pop("EVOLAB_EXPERIENCE", None)
    os.environ.pop("EVOLAB_EXPERIENCE_DB", None)


def run_one(
    scenario,
    db_path: Path,
    seed: int,
    run_id: str,
    *,
    prior_enabled: bool,
    generations: int,
    population: int,
) -> None:
    ev = scenario.create_evaluator()
    wired = attach_experience_recorder(
        ev,
        scenario.sources,
        scenario.target_file,
        scenario.func_name,
        db_path=db_path,
        run_id=run_id,
        prior_enabled=prior_enabled,
        prior_kwargs=dict(FROZEN_PRIOR_KWARGS) if prior_enabled else None,
    )
    engine = EvolutionEngine(
        fitness_fn=wired,
        config=EngineConfig(
            generations=generations,
            population_size=population,
            elite_count=1,
            genome_size=2,
            seed=seed,
        ),
    )
    initial = make_code_population(scenario, population, random.Random(seed))
    engine.run(generations=generations, initial_population=initial)


def arm_cost(metrics: dict[str, Any]) -> int:
    """Censored runs count at their total evaluation budget."""
    if metrics["first_success_eval"] is not None:
        return int(metrics["first_success_eval"])
    return int(metrics["evals_total"])


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def sign_pvalue(wins: int, losses: int) -> float | None:
    """Exact two-sided sign-test p over non-tied pairs (Binomial(n, 0.5))."""
    n = wins + losses
    if n == 0:
        return None
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def keeps_default_on(pooled: dict[str, Any]) -> bool:
    """The pre-registered decision rule, as a function so it is testable:
    memory keeps default-on status iff overall gain > 0 AND the non-tied
    paired wins are >= losses. No post-hoc reinterpretation."""
    gain = pooled.get("search_efficiency_gain")
    paired = pooled.get("paired", {})
    return bool(
        gain is not None
        and gain > 0
        and paired.get("memory_wins", 0) >= paired.get("control_wins", 0)
    )


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired runs into the honest headline numbers."""
    ctrl_costs = [arm_cost(p["control"]) for p in pairs]
    mem_costs = [arm_cost(p["memory"]) for p in pairs]
    wins = sum(1 for p in pairs if arm_cost(p["memory"]) < arm_cost(p["control"]))
    losses = sum(1 for p in pairs if arm_cost(p["memory"]) > arm_cost(p["control"]))
    ties = len(pairs) - wins - losses
    n_mean = mean(ctrl_costs)
    m_mean = mean(mem_costs)
    gain = (1.0 - m_mean / n_mean) if n_mean > 0 else None
    return {
        "pairs": len(pairs),
        "N_mean_evals": round(n_mean, 3),
        "M_mean_evals": round(m_mean, 3),
        "search_efficiency_gain": (round(gain, 4) if gain is not None else None),
        "successes": {
            "control": sum(1 for p in pairs if p["control"]["first_success_eval"] is not None),
            "memory": sum(1 for p in pairs if p["memory"]["first_success_eval"] is not None),
        },
        "paired": {"memory_wins": wins, "control_wins": losses, "ties": ties},
        "sign_test_p": sign_pvalue(wins, losses),
    }


def run_scenario(
    scenario,
    seeds: list[int],
    teachers: int,
    generations: int,
    population: int,
    workdir: Path,
) -> dict[str, Any]:
    clean_env()
    fingerprint = None
    teacher_db = workdir / f"teacher_{scenario.name}.db"

    # --- teacher phase: plain recorded experience, prior disabled ---------
    for i in range(teachers):
        rid = f"teacher_{scenario.name}_t{i}"
        run_one(
            scenario,
            teacher_db,
            TEACHER_SEED_START + i,
            rid,
            prior_enabled=False,
            generations=generations,
            population=population,
        )
    tstore = ExperienceStore(teacher_db)
    stats = tstore.stats(
        problem_fingerprint(scenario.sources, scenario.target_file, scenario.func_name)
    )
    tstore.close()

    # --- paired test phase -------------------------------------------------
    pairs: list[dict[str, Any]] = []
    for s in seeds:
        ctrl_db = workdir / f"ctrl_{scenario.name}_s{s}.db"
        ctrl_rid = f"ctrl_{scenario.name}_s{s}"
        run_one(
            scenario,
            ctrl_db,
            s,
            ctrl_rid,
            prior_enabled=False,
            generations=generations,
            population=population,
        )
        ctrl_store = ExperienceStore(ctrl_db)
        ctrl_metrics = ctrl_store.run_metrics(ctrl_rid)
        ctrl_store.close()
        ctrl_db.unlink()

        mem_db = workdir / f"mem_{scenario.name}_s{s}.db"
        shutil.copyfile(teacher_db, mem_db)
        mem_rid = f"mem_{scenario.name}_s{s}"
        run_one(
            scenario,
            mem_db,
            s,
            mem_rid,
            prior_enabled=True,
            generations=generations,
            population=population,
        )
        mem_store = ExperienceStore(mem_db)
        mem_metrics = mem_store.run_metrics(mem_rid)
        mem_store.close()
        mem_db.unlink()

        pairs.append(
            {
                "seed": s,
                "control": ctrl_metrics,
                "memory": mem_metrics,
            }
        )

    summary = summarize_pairs(pairs)
    return {
        "scenario": scenario.name,
        "difficulty": scenario.difficulty,
        "teacher_experiences": stats["total_experiences"],
        "per_edit_kind": stats["per_edit_kind"],
        **summary,
        "runs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="1-10", help="inclusive range, e.g. 1-10")
    ap.add_argument("--teachers", type=int, default=3)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--scenarios", default=",".join(SCENARIO_REGISTRY))
    ap.add_argument("--out", default="reports/ab_memory_value.json")
    args = ap.parse_args(argv)

    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIO_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown scenarios: {unknown}; available: {list(SCENARIO_REGISTRY)}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="evolab_ab_") as td:
        workdir = Path(td)
        for name in names:
            scenario = SCENARIO_REGISTRY[name]()
            print(f"=== {name} (difficulty={scenario.difficulty}) ===", flush=True)
            res = run_scenario(
                scenario, seeds, args.teachers, args.generations, args.population, workdir
            )
            results.append(res)
            print(
                f"  teacher_experiences={res['teacher_experiences']}  "
                f"N={res['N_mean_evals']}  M={res['M_mean_evals']}  "
                f"gain={res['search_efficiency_gain']}  "
                f"successes={res['successes']}  paired={res['paired']}",
                flush=True,
            )

    pooled = summarize_pairs([p for r in results for p in r["runs"]])
    keeps = keeps_default_on(pooled)
    verdict = {
        "rule": "keep default-on memory iff overall gain > 0 AND memory_wins >= control_wins (registered before the run)",
        "overall_gain": pooled["search_efficiency_gain"],
        "keeps_default_on": keeps,
    }
    report = {
        "protocol": {
            "seeds": seeds,
            "teachers": args.teachers,
            "teacher_seeds": [TEACHER_SEED_START + i for i in range(args.teachers)],
            "generations": args.generations,
            "population": args.population,
            "teacher_prior": "disabled (plain experience only)",
            "memory_prior": "enabled, frozen snapshot of the teacher store (cache_ttl=1e9)",
            "control_prior": "hard-disabled (prior_enabled=False), identical instrumentation",
            "metric": "evals to first holdout success; censored at total evals",
        },
        "results": results,
        "overall": pooled,
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print(
        f"OVERALL (pooled {pooled['pairs']} paired runs): "
        f"N={pooled['N_mean_evals']}  M={pooled['M_mean_evals']}  "
        f"gain={pooled['search_efficiency_gain']}  paired={pooled['paired']}  p={pooled['sign_test_p']}"
    )
    print(f"VERDICT: keeps_default_on={keeps}  (rule: {verdict['rule']})")
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
