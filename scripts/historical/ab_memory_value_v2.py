#!/usr/bin/env python3
"""A/B harness v2 (M1): a measurement instrument hardened against the two
weaknesses documented after v1 (reports/ab_memory_value.json):

  W1  Censoring-dominated mean. v1's primary metric (mean evals-to-first-
      success, censored runs counted at full budget) let 2 extra censored
      runs in one arm drag a scenario mean by -63.8% while the sign test
      said p=0.51. v2's PRIMARY metric is the holdout success count within
      the fixed budget (a binomial count, censoring-free), with the median
      cost among successes as secondary. The v1 mean-gain is still reported
      for continuity but is explicitly demoted: it takes part in NO decision.

  W2  Nominal pairing. With the prior enabled, ``rng.choices`` consumes the
      RNG stream differently, so the memory arm's trajectory diverges from
      its same-seed control twin completely — even at strength 0.0 (v1
      analysis: max weight deviation from uniform was 2.1 percentage points
      yet the scenario moved -63.8%). v2 therefore analyzes arms UNPAIRED,
      and adds a strength=0.0 RESHUFFLE arm that isolates the pure
      trajectory-reshuffle effect from the bias effect.

Pre-registered protocol (written BEFORE any measurement — no scenario, seed,
arm or rule may be tuned after peeking at results):

  Teacher phase   Per scenario, 3 GA runs (seeds 9901..9903), prior DISABLED,
                  record experiences into one store (identical to v1).
  Arms            For each seed s in 1..30, four runs per scenario, each on a
                  copy of the frozen teacher snapshot (control: fresh store):
                    control    prior_enabled=False (exact pre-memory behavior)
                    prior_s00  prior enabled, strength=0.0  (pure reshuffle)
                    prior_s05  prior enabled, strength=0.5  (v1's config)
                    prior_s09  prior enabled, strength=0.9  (strong bias)
                  All prior arms read the same frozen snapshot
                  (cache_ttl=1e9); student writes never leak across runs.
  Budget          generations=8, population=12, elite_count=1, genome_size=2
                  (identical to v1 so results are comparable).
  Primary metric  holdout successes within budget, out of 30 per scenario
                  per arm (120 pooled).
  Secondary       median evals-to-first-success among that arm's successes.
  Demoted         mean censored cost and search_efficiency_gain (v1 metric),
                  reported for continuity only.
  Statistics      two-sided Fisher exact test on the pooled 2x2 table
                  (arm vs control, success vs no-success).

Decision rules (registered here, before the run):
  R1  An arm is classified vs control as
        "promising"   iff pooled successes > control AND Fisher p < 0.05
                      AND the surplus sign holds in >= 3 of 4 scenarios;
        "harmful"     iff pooled successes < control AND Fisher p < 0.05
                      AND the deficit sign holds in >= 3 of 4 scenarios;
        otherwise     "no_detectable_effect at this power".
  R2  The same rule classifies prior_s00 vs control. If prior_s00 shows a
      detectable effect, trajectory reshuffling alone moves the needle at
      this power, and the BIAS verdict must then be drawn against prior_s00
      (bias effect = arm@s vs arm@0) with the same rule — not against
      control.
  R3  Any "promising" classification does NOT re-enable the prior by
      default: it only registers that a follow-up tuning experiment is
      warranted. Default stays off until a follow-up registered protocol
      proves gain > 0 on this instrument.

Usage (from the repository root):
  PYTHONPATH=src python scripts/ab_memory_value_v2.py
  PYTHONPATH=src python scripts/ab_memory_value_v2.py --seeds 1-5 --scenarios requests_http_helper --out /tmp/ab2.json

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
ARMS: list[tuple[str, float | None]] = [
    ("control", None),      # prior hard-disabled
    ("prior_s00", 0.0),     # pure reshuffle
    ("prior_s05", 0.5),     # v1's configuration
    ("prior_s09", 0.9),     # strong bias
]
FROZEN_CACHE_TTL = 1e9
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
    strength: float | None,
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
        prior_enabled=strength is not None,
        prior_kwargs=(
            {"cache_ttl": FROZEN_CACHE_TTL, "strength": strength}
            if strength is not None
            else None
        ),
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
    """R1 as a function: testable, no post-hoc reinterpretation."""
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


def run_scenario(
    scenario,
    seeds: list[int],
    generations: int,
    population: int,
    workdir: Path,
) -> dict[str, Any]:
    clean_env()
    name = scenario.name
    teacher_db = workdir / f"teacher_{name}.db"
    for i in range(TEACHERS):
        run_one(
            scenario,
            teacher_db,
            TEACHER_SEED_START + i,
            f"teacher_{name}_t{i}",
            strength=None,
            generations=generations,
            population=population,
        )
    tstore = ExperienceStore(teacher_db)
    teacher_stats = tstore.stats(
        problem_fingerprint(scenario.sources, scenario.target_file, scenario.func_name)
    )
    tstore.close()

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm, _ in ARMS}
    for s in seeds:
        for arm, strength in ARMS:
            if arm == "control":
                db = workdir / f"ctrl_{name}_s{s}.db"
                shutil.copyfile(teacher_db, db)  # same snapshot lineage; prior disabled
            else:
                db = workdir / f"{arm}_{name}_s{s}.db"
                shutil.copyfile(teacher_db, db)
            rid = f"{arm}_{name}_s{s}"
            run_one(
                scenario,
                db,
                s,
                rid,
                strength=strength,
                generations=generations,
                population=population,
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
    ap.add_argument("--out", default="reports/ab_memory_value_v2.json")
    args = ap.parse_args(argv)

    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIO_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown scenarios: {unknown}; available: {list(SCENARIO_REGISTRY)}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="evolab_ab2_") as td:
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

    # R2 refinement: if the reshuffle arm is detectable, bias verdicts are
    # drawn against prior_s00 instead of control (same registered rule).
    reshuffle_detectable = pooled.get("prior_s00", {}).get("classification_vs_control") != "no_detectable_effect"
    if reshuffle_detectable:
        base_rows = None  # vs prior_s00 requires per-run costs; counts comparison:
        s00_succ = pooled["prior_s00"]["successes_within_budget"]
        for arm in ("prior_s05", "prior_s09"):
            arm_succ = pooled[arm]["successes_within_budget"]
            n_each = len(seeds) * len(results)
            p = fisher_exact_2x2(arm_succ, n_each - arm_succ, s00_succ, n_each - s00_succ)
            deltas = [r["per_arm"][arm]["successes_within_budget"] - r["per_arm"]["prior_s00"]["successes_within_budget"] for r in results]
            pooled[arm]["classification_vs_reshuffle"] = classify_effect(arm_succ, s00_succ, p, deltas)

    verdict = {
        "rules_registered": [
            "R1: promising/harmful iff Fisher p<0.05 AND sign-consistent in >=3/4 scenarios",
            "R2: if prior_s00 (pure reshuffle) is detectable, bias verdicts are drawn vs prior_s00",
            "R3: 'promising' only registers a follow-up tuning experiment; default stays off",
        ],
        "reshuffle_detectable": reshuffle_detectable,
        "pooled": pooled,
    }
    report = {
        "protocol": {
            "seeds": seeds,
            "teachers": TEACHERS,
            "teacher_seeds": [TEACHER_SEED_START + i for i in range(TEACHERS)],
            "generations": args.generations,
            "population": args.population,
            "arms": {arm: ("prior disabled" if s is None else f"prior strength={s}, frozen teacher snapshot (cache_ttl=1e9)") for arm, s in ARMS},
            "primary_metric": "holdout successes within budget (censoring-free count)",
            "secondary_metric": "median evals-to-first-success among successes",
            "demoted_metric": "mean censored cost / v1 mean-gain (continuity only, no decision role)",
            "analysis": "unpaired across arms (weighted rng consumes the RNG stream differently; pairing is nominal — documented v1 weakness)",
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
    if reshuffle_detectable:
        print("R2 active: bias verdicts drawn vs prior_s00:")
        for arm in ("prior_s05", "prior_s09"):
            print(f"  {arm} vs reshuffle: {pooled[arm].get('classification_vs_reshuffle')}")
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
