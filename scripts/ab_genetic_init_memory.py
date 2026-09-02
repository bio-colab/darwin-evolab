#!/usr/bin/env python3
"""M8 A/B harness: memory via the genetic channel — trap-aware initialization.

Forensic basis (established before this protocol was written):
  * The CEM genetic-memory system (memory.py) is alive only on the numeric
    path — archiving (list-of-numbers gate), injection (isinstance-list gate)
    and signature/sandbox (float-vector math) all exclude RepairGenome.
  * Cross-run POSITIVE genetic memory (re-seeding remembered programs) is
    blocked by the payload wall: the experience store records skeletons
    (kinds + loci) by design, never payloads — and its value class
    (cross-run program reuse) is already measured: the M2 probe found
    97.5% cross-run duplicate-evaluation savings, i.e. reuse belongs to
    the cache/archive family, not to a new hypothesis.
  * Cross-run NEGATIVE genetic memory IS possible with existing data:
    single-edit failures carry (kind, locus) — enough to know which doors
    are dead. This is the agenda's "trap-aware initialization": memory
    decides WHICH GENOTYPES EXIST at generation 0 (population composition
    = the genetic state), instead of biasing the mutation operator during
    search (the variation channel — measured twice, v2 and M7, no
    detectable effect at this power).

Pre-registered protocol (written BEFORE any measurement — no scenario,
seed, arm, threshold or rule may be tuned after peeking at results):

  Teacher phase   Per scenario, 3 GA runs (seeds 9901..9903), prior
                  DISABLED, no avoidance — byte-identical to the v2/M7
                  teacher phase so the frozen snapshots stay comparable.
  Arms            For each seed s in 1..30, four runs per scenario, each on
                  a copy of the frozen teacher snapshot; ALL arms run with
                  prior_enabled=False (M8's channel is the population):
                    control      avoid=None, redraws=0 (legacy behavior)
                    redraw_once  avoid=None, redraws=1 — mechanism-free
                                 init perturbation; isolates re-draw noise
                                 from memory-directed avoidance (the v2
                                 R2 discipline, adapted to the genetic
                                 channel)
                    avoid_f2     avoid=avoidance_set(fp, min_failures=2)
                    avoid_f3     avoid=avoidance_set(fp, min_failures=3)
                  Dead-door definition (registered before measurement):
                  a (file, lineno, col_offset, kind) key mined from
                  SINGLE-edit experiences with >= min_failures attempts
                  and ZERO holdout successes. Multi-edit rows are never
                  mined; one success rescues a door forever.
  Budget          generations=8, population=12, elite_count=1 (identical
                  to v1/v2/M7 so results are comparable).
  Primary metric  holdout successes within budget, out of 30 per scenario
                  per arm (120 pooled).
  Secondary       median evals-to-first-success among that arm's successes.
  Statistics      two-sided Fisher exact test on the pooled 2x2 table,
                  unpaired (same rationale as v2: any mechanism that
                  consumes the RNG stream differently diverges trajectories).

Decision rules (registered here, before the run):
  R1'' An arm is classified vs control as
         "promising"   iff pooled successes > control AND Fisher p < 0.05
                       AND the surplus sign holds in >= 3 of 4 scenarios;
         "harmful"     iff pooled successes < control AND Fisher p < 0.05
                       AND the deficit sign holds in >= 3 of 4 scenarios;
         otherwise     "no_detectable_effect at this power".
  R2'' If redraw_once shows a detectable effect vs control, init
       re-drawing alone moves the needle at this power, and the memory
       verdicts must then be drawn vs redraw_once — not against control.
  R3'' Any "promising" classification does NOT enable avoidance by
       default: it only registers that a follow-up tuning experiment is
       warranted. Default stays off until a follow-up registered protocol
       proves gain > 0 on this instrument.
  R4'' Dose-response: if avoid_f2 and avoid_f3 disagree in classification,
       the verdict is "dose-sensitive at this power" — no default either
       way, follow-up required.
  Transparency (committed with the report): per-scenario dead-door set
  sizes for both doses and the teacher experience count.

Usage (from the repository root):
  PYTHONPATH=src python scripts/ab_genetic_init_memory.py
  PYTHONPATH=src python scripts/ab_genetic_init_memory.py --seeds 1-5 \
      --scenarios requests_http_helper --out /tmp/ab8.json

Stdlib only. Deterministic given fixed --seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
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
TEACHERS = 3
ARMS: list[tuple[str, int | None]] = [
    ("control", None),      # avoid=None, redraws=0
    ("redraw_once", None),  # avoid=None, redraws=1 (mechanism-free isolation)
    ("avoid_f2", 2),        # dead doors with >= 2 failed attempts
    ("avoid_f3", 3),        # dead doors with >= 3 failed attempts
]
FISHER_ALPHA = 0.05
CONSISTENCY_SCENARIOS = 3  # of 4


def clean_env() -> None:
    os.environ.pop("EVOLAB_EXPERIENCE", None)
    os.environ.pop("EVOLAB_EXPERIENCE_DB", None)


def run_one(
    scenario,
    db_path: Path,
    seed: int,
    run_id: str,
    *,
    generations: int,
    population: int,
    avoid_loci: set | None,
    redraws: int,
) -> None:
    ev = scenario.create_evaluator()
    wired = attach_experience_recorder(
        ev,
        scenario.sources,
        scenario.target_file,
        scenario.func_name,
        db_path=db_path,
        run_id=run_id,
        prior_enabled=False,  # M8 runs the genetic channel, never the prior
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
    initial = make_code_population(
        scenario, population, random.Random(seed),
        avoid_loci=avoid_loci, redraws=redraws,
    )
    engine.run(generations=generations, initial_population=initial)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]].

    Rows = (test arm, control), columns = (success, no-success):
    a = arm successes, b = arm failures, c = control successes, d = control
    failures. Hypergeometric two-sided via the probability-mass convention.
    """
    n = a + b + c + d
    r1 = a + b
    c1 = a + c
    if min(n, r1, c1) == 0 or max(r1, c1) > n:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)

    lo, hi = max(0, c1 - (n - r1)), min(c1, r1)
    p_obs = prob(a)
    tail = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs + 1e-12)
    return min(1.0, tail)


def classify_effect(
    arm_successes: int,
    control_successes: int,
    fisher_p: float,
    per_scenario_deltas: list[int],
) -> str:
    """R1'' as a function: testable, no post-hoc reinterpretation."""
    n_pos = sum(1 for d in per_scenario_deltas if d > 0)
    n_neg = sum(1 for d in per_scenario_deltas if d < 0)
    consistent = n_pos >= CONSISTENCY_SCENARIOS or n_neg >= CONSISTENCY_SCENARIOS
    if fisher_p < FISHER_ALPHA and consistent:
        if arm_successes > control_successes:
            return "promising"
        if arm_successes < control_successes:
            return "harmful"
    return "no_detectable_effect"


def median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 1) if xs else None


def summarize_arm(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [
        r["first_success_eval"] if r["first_success_eval"] is not None else r["evals_total"]
        for r in run_rows
    ]
    success_costs = [r["first_success_eval"] for r in run_rows if r["first_success_eval"] is not None]
    return {
        "runs": len(run_rows),
        "successes_within_budget": len(success_costs),
        "median_evals_among_successes": median(success_costs),
        "mean_censored_cost": round(sum(costs) / len(costs), 2) if costs else 0.0,
    }


def dead_doors_report(store: ExperienceStore, fingerprint: str) -> dict[str, Any]:
    """Transparency: the exact memory content the avoid arms consumed."""
    out: dict[str, Any] = {}
    for arm, f in (("avoid_f2", 2), ("avoid_f3", 3)):
        doors = store.avoidance_set(fingerprint, min_failures=f)
        if doors is None:
            out[arm] = {"size": 0, "no_single_edit_data": True, "doors": []}
        else:
            out[arm] = {
                "size": len(doors),
                "no_single_edit_data": False,
                "doors": sorted(f"{file}:{ln}:{col}:{kind}" for file, ln, col, kind in doors),
            }
    return out


def run_scenario(
    scenario,
    seeds: list[int],
    generations: int,
    population: int,
    workdir: Path,
) -> dict[str, Any]:
    clean_env()
    name = scenario.name
    fingerprint = problem_fingerprint(
        scenario.sources, scenario.target_file, scenario.func_name
    )
    teacher_db = workdir / f"teacher_{name}.db"
    for i in range(TEACHERS):
        run_one(
            scenario,
            teacher_db,
            TEACHER_SEED_START + i,
            f"teacher_{name}_t{i}",
            generations=generations,
            population=population,
            avoid_loci=None,
            redraws=0,
        )
    tstore = ExperienceStore(teacher_db)
    teacher_stats = tstore.stats(fingerprint)
    avoidance = dead_doors_report(tstore, fingerprint)
    tstore.close()

    avoid_by_arm: dict[str, set | None] = {}
    tstore = ExperienceStore(teacher_db)
    for arm, f in ARMS:
        if arm.startswith("avoid_"):
            avoid_by_arm[arm] = tstore.avoidance_set(fingerprint, min_failures=int(f))
        elif arm == "redraw_once":
            avoid_by_arm[arm] = None
        else:
            avoid_by_arm[arm] = None
    tstore.close()

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm, _ in ARMS}
    for s in seeds:
        for arm, _f in ARMS:
            db = workdir / f"{arm}_{name}_s{s}.db"
            shutil.copyfile(teacher_db, db)  # same snapshot lineage for every arm
            rid = f"{arm}_{name}_s{s}"
            run_one(
                scenario,
                db,
                s,
                rid,
                generations=generations,
                population=population,
                avoid_loci=avoid_by_arm[arm],
                redraws=1 if arm == "redraw_once" else 0,
            )
            store = ExperienceStore(db)
            metrics = store.run_metrics(rid)
            store.close()
            db.unlink()
            arm_rows[arm].append(metrics)

    per_arm = {arm: summarize_arm(rows) for arm, rows in arm_rows.items()}
    deltas = {
        arm: per_arm[arm]["successes_within_budget"] - per_arm["control"]["successes_within_budget"]
        for arm, _ in ARMS
        if arm != "control"
    }
    return {
        "scenario": name,
        "difficulty": scenario.difficulty,
        "teacher_experiences": teacher_stats["total_experiences"],
        "avoidance_memory": avoidance,
        "per_arm": per_arm,
        "success_deltas_vs_control": deltas,
        "runs": arm_rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="1-30", help="inclusive range, e.g. 1-30")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--scenarios", default=",".join(SCENARIO_REGISTRY))
    ap.add_argument("--out", default="reports/ab_genetic_init_memory.json")
    args = ap.parse_args(argv)

    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIO_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown scenarios: {unknown}; available: {list(SCENARIO_REGISTRY)}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="evolab_ab8_") as td:
        for name in names:
            scenario = SCENARIO_REGISTRY[name]()
            print(f"=== {name} (difficulty={scenario.difficulty}) ===", flush=True)
            res = run_scenario(scenario, seeds, args.generations, args.population, Path(td))
            results.append(res)
            line = ", ".join(
                f"{arm}={res['per_arm'][arm]['successes_within_budget']}/{len(seeds)}"
                for arm, _ in ARMS
            )
            print(f"  successes within budget: {line}", flush=True)

    # pooled analysis (unpaired — registered rationale in the module docstring)
    pooled_ctrl = sum(
        r["per_arm"]["control"]["successes_within_budget"] for r in results
    )
    pooled: dict[str, Any] = {}
    for arm, _ in ARMS:
        if arm == "control":
            continue
        arm_succ = sum(r["per_arm"][arm]["successes_within_budget"] for r in results)
        ctrl_succ = pooled_ctrl
        n_each = len(seeds) * len(results)
        p = fisher_exact_2x2(arm_succ, n_each - arm_succ, ctrl_succ, n_each - ctrl_succ)
        deltas = [r["success_deltas_vs_control"][arm] for r in results]
        cls = classify_effect(arm_succ, ctrl_succ, p, deltas)
        pooled[arm] = {
            "successes_within_budget": arm_succ,
            "vs_control_delta": arm_succ - ctrl_succ,
            "fisher_p": round(p, 5),
            "per_scenario_deltas": deltas,
            "classification_vs_control": cls,
        }

    # R2'' refinement: if the mechanism-free redraw arm is detectable, memory
    # verdicts are drawn vs redraw_once instead of control (same rule shape
    # as v2's prior_s00).
    redraw_detectable = pooled.get("redraw_once", {}).get("classification_vs_control") != "no_detectable_effect"
    if redraw_detectable:
        once_succ = pooled["redraw_once"]["successes_within_budget"]
        n_each = len(seeds) * len(results)
        for arm in ("avoid_f2", "avoid_f3"):
            arm_succ = pooled[arm]["successes_within_budget"]
            p = fisher_exact_2x2(arm_succ, n_each - arm_succ, once_succ, n_each - once_succ)
            deltas = [
                r["per_arm"][arm]["successes_within_budget"]
                - r["per_arm"]["redraw_once"]["successes_within_budget"]
                for r in results
            ]
            pooled[arm]["classification_vs_redraw"] = classify_effect(
                arm_succ, once_succ, p, deltas
            )

    # R4'' dose-response
    cls_f2 = pooled.get("avoid_f2", {}).get("classification_vs_control")
    cls_f3 = pooled.get("avoid_f3", {}).get("classification_vs_control")
    dose_sensitive = cls_f2 != cls_f3

    verdict = {
        "rules_registered": [
            "R1'': promising/harmful iff Fisher p<0.05 AND sign-consistent in >=3/4 scenarios",
            "R2'': if redraw_once (mechanism-free init perturbation) is detectable, memory verdicts are drawn vs redraw_once",
            "R3'': 'promising' only registers a follow-up tuning experiment; default stays off",
            "R4'': if avoid_f2 and avoid_f3 disagree, verdict is dose-sensitive — no default either way",
        ],
        "redraw_detectable": redraw_detectable,
        "dose_sensitive": dose_sensitive,
        "pooled": pooled,
    }
    report = {
        "protocol": {
            "seeds": seeds,
            "teachers": TEACHERS,
            "teacher_seeds": [TEACHER_SEED_START + i for i in range(TEACHERS)],
            "generations": args.generations,
            "population": args.population,
            "arms": {
                "control": "avoid=None, redraws=0, prior disabled (legacy behavior)",
                "redraw_once": "avoid=None, redraws=1 (mechanism-free init perturbation)",
                "avoid_f2": "avoid=avoidance_set(min_failures=2), prior disabled",
                "avoid_f3": "avoid=avoidance_set(min_failures=3), prior disabled",
            },
            "dead_door_definition": "(file, lineno, col_offset, kind) mined from SINGLE-edit experiences with >= min_failures attempts and ZERO holdout successes; multi-edit rows never mined; one success rescues a door forever",
            "primary_metric": "holdout successes within budget (censoring-free count)",
            "secondary_metric": "median evals-to-first-success among successes",
            "analysis": "unpaired across arms (any mechanism consuming the RNG stream differently diverges trajectories — documented v1 weakness)",
            "statistics": f"two-sided Fisher exact on pooled 2x2, alpha={FISHER_ALPHA}",
        },
        "results": results,
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    ctrl_line = f"control={pooled_ctrl}/{len(seeds) * len(results)}"
    print(f"POOLED successes within budget: {ctrl_line}, " + ", ".join(
        f"{arm}={v['successes_within_budget']} ({v['classification_vs_control']}, p={v['fisher_p']})"
        for arm, v in pooled.items()
    ))
    if redraw_detectable:
        print("R2'' active: memory verdicts drawn vs redraw_once:")
        for arm in ("avoid_f2", "avoid_f3"):
            print(f"  {arm} vs redraw_once: {pooled[arm].get('classification_vs_redraw')}")
    if dose_sensitive:
        print("R4'' active: dose-sensitive classification — no default either way")
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
