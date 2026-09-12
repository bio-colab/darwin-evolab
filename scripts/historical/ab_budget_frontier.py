"""Budget-frontier dose-response (BF) — pre-registered measurement, no mechanism.

BACKGROUND (registered before the run):
  M9's closing verdict: memory CONTENT arrived honestly in every measured
  channel (v2 kind-prior, M7 sequence-prior, M8 avoidance, M9 composition
  seeding) yet the success metric never moved detectably — the bottleneck is
  the search dynamics itself. The last deferred hypothesis worth building is
  "extend CEM (memory.py) to RepairGenomes with a longer budget". CEM is
  structurally dead on the repair path (three gates: archive serialize()
  wants a numeric list, injection wants isinstance(genome, list), signature
  math is float vectors) AND behaviorally dead at the standard budget
  (static fitness => CUSUM sees STABLE => dose 0.0 forever). Building the
  mechanism is justified ONLY IF the search can convert a longer budget into
  holdout successes at all. This protocol measures exactly that — cold
  start, no memory, no prior, no mechanism.

DESIGN:
  Four SCENARIO_REGISTRY scenarios x 30 seeds (1..30, identical to the v2
  student phase) x generation ladder {8, 16, 32} at population=12,
  elite_count=1, genome_size=2, prior_enabled=False, seed_keys=None /
  seed_count=0 (byte-identical legacy cold population). The ONLY varying
  parameter is generations. Config is otherwise byte-identical to the M9
  control arm (run_one in scripts/ab_composition_seeding.py).

  Prefix semantics (verified structurally BEFORE registration): the engine
  re-seeds RNG at run start and draws lazily per generation; the single
  generations-dependent computation (exploit_start) is consumed only by
  sharing_on(), which is disabled in code mode; both early stops depend on
  stagnation_patience (15) / early_stop_fitness (None), never on the total
  generation count. Therefore a G-generation run with seed s executes
  exactly the first G generations of the same-seed trajectory at any larger
  budget: success counts are non-decreasing in budget BY CONSTRUCTION, and
  the level-8 arm doubles as an exact replication of the M9 control arm.

REGISTERED RULES (written before any run):
  R1''''' Primary test: per-scenario success-within-budget counts at each
         ladder level; Fisher exact 2x2 (level 32 vs level 8), unpaired as
         registered in M8/M9 (conservative under the paired prefix design),
         alpha=0.05.
  R2''''' Replication anchor: the level-8 arm MUST exactly reproduce the M9
         control arm (click_cli_parser 5/30, requests_http_helper 30/30,
         lru_cache_logic 27/30, multi_file_config 30/30; pooled 92/120).
         Any mismatch => STOP, investigate, no verdict until resolved.
  R3''''' Prefix determinism: successes must be non-decreasing in budget
         for identical seeds. Any regression => engine nondeterminism =>
         stop all interpretation, file as a bug.
  R4''''' Cost honesty: report the evals_total distribution per level
         (stagnation truncation = effective budget actually consumed),
         median first_success_eval among successes per level, mean censored
         cost, and the pooled success-vs-eval-budget curve derived from the
         32-generation runs at gen-equivalent thresholds (97, 145, 193, 241,
         289, 337, 385 evals = 8..32 generations at population 12).
  R5''''' CEM build gate (registered BEFORE the run): build the
         CEM-on-RepairGenome mechanism ONLY IF at least one scenario shows
         Delta >= +3 successes (level 32 vs level 8) with Fisher p < 0.05
         (and R3''''' holds trivially). Delta in {1, 2}: no build, report
         descriptively. No scenario passing: the honest verdict is "budget
         alone does not open the frontier up to 32 generations (~3.97x
         evals) under the engine's stagnation patience" — the
         CEM-with-longer-budget premise is falsified at this scale; no
         mechanism is built and the remaining deferred hypothesis
         (to_code() cache key) becomes next.
  R6''''' Transparency: per-run raw rows (scenario, seed, generations,
         first_success_eval, first_success_score, evals_total, first-success
         edit kinds/loci, wall seconds) committed with the report JSON.

PRE-REGISTERED POWER NOTE: two of the four scenarios sit at ceiling 30/30 in
  the M9 control (zero headroom — they can only stay flat). The house
  consistency rule (>= 3 of 4 scenarios same direction) is retained as
  registered but cannot draw on ceiling scenarios; any global claim
  therefore rides on the two headroom scenarios (click 5/30, lru 27/30) and
  is labeled as such. Per-scenario verdicts are the decision layer.

DEVIATION from M9: NO teacher phase and NO memory arms — this measures pure
  search dynamics from a cold start; memory absence is the design, not an
  omission.

Usage (from the repository root):
  PYTHONPATH=src python scripts/ab_budget_frontier.py
  PYTHONPATH=src python scripts/ab_budget_frontier.py --seeds 1-5 \
      --scenarios click_cli_parser --out /tmp/bf.json

Stdlib only. Deterministic given fixed --seeds.

===========================================================================
BF-2 — GOVERNOR SUSPENSION (registered AFTER BF-1's results were seen and
BF-1 was committed, BEFORE any BF-2 run; BF-1's verdict stands untouched —
this is a separate sequential protocol, not a reinterpretation):

  MOTIVATION: BF-1's R4''''' finding — 120/120 runs truncated by
  stagnation_patience=15 at g32 (193-349 of 385 nominal evals consumed).
  BF-1's verdict is explicitly scoped "under the engine's stagnation
  patience". BF-2 asks the remaining question: is the frontier closed
  because the search is saturated, or because the governor resigns before
  the budget is spent?

  DESIGN: identical to BF-1 except stagnation_patience=32 is passed
  explicitly — provably inert within a 32-generation ladder
  (gens_since_improvement <= 31 at the last generation), so every run
  consumes its full nominal budget. The governor check consumes no RNG and
  only breaks the loop, so trajectories remain prefix-compatible with
  BF-1 wherever the governor never fired (g8: patience needs 15 stagnant
  gens — inert; g16: BF-1 showed zero truncation at g16 — inert).

  REGISTERED RULES (written before any BF-2 run):
  S1  Replication anchor: the level-8 arm MUST exactly reproduce the M9
      control / BF-1 level-8 numbers (5/30/27/30). Any mismatch => STOP.
  S2  Continuity check: the level-16 arm MUST exactly reproduce BF-1's
      level-16 numbers (6/30/27/30) — the governor never fired at g16 in
      BF-1. Any mismatch => nondeterminism => STOP. (Verified in the
      external analysis from the two committed reports.)
  S3  Full consumption: EVERY g32 run must show evals_total = 385 exactly.
      Any run below 385 => protocol bug => stop all interpretation.
  S4  Primary: Fisher exact 2x2 per scenario (g32 vs g8), alpha=0.05, and
      the SAME build gate as R5''''' (Delta >= +3 successes AND p < 0.05,
      no regressions). OPEN => the CEM premise is alive at full-consumption
      budget; a CEM-on-RepairGenome build (M10) may be registered under
      this config. CLOSED => the budget hypothesis is falsified up to full
      consumption of 385 evals (~3.97x standard) — final death of
      "CEM with longer budget"; the remaining deferred hypothesis is the
      to_code() cache key.
  S5  Transparency: per-run raw rows committed with the report; README
      subsection; both protocols remain separately registered.

  BF-2 invocation (registered):
    PYTHONPATH=src python scripts/ab_budget_frontier.py \
        --stagnation-patience 32 --out reports/budget_frontier_full_consumption.json
===========================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab import EngineConfig, EvolutionEngine  # noqa: E402
from evolab.code_fixtures import (  # noqa: E402
    SCENARIO_REGISTRY,
    make_code_population,
)
from evolab.experience import (  # noqa: E402
    ExperienceStore,
    attach_experience_recorder,
    problem_fingerprint,
)

SEEDS = list(range(1, 31))
LEVELS = (8, 16, 32)
POPULATION = 12
FISHER_ALPHA = 0.05
BUILD_GATE_DELTA = 3
M9_CONTROL_ANCHOR = {
    "click_cli_parser": 5,
    "requests_http_helper": 30,
    "lru_cache_logic": 27,
    "multi_file_config": 30,
}
GEN_EQUIV_THRESHOLDS = (97, 145, 193, 241, 289, 337, 385)


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
    stagnation_patience: int | None = None,
) -> None:
    """One cold-start run. With ``stagnation_patience=None`` (default) the
    engine's own default (15) applies — byte-identical legacy behavior, i.e.
    exactly the BF-1 configuration. A non-None value is the BF-2 governor
    suspension."""
    ev = scenario.create_evaluator()
    wired = attach_experience_recorder(
        ev,
        scenario.sources,
        scenario.target_file,
        scenario.func_name,
        db_path=db_path,
        run_id=run_id,
        prior_enabled=False,  # pure search dynamics: no prior, no memory
    )
    cfg_kwargs: dict[str, Any] = dict(
        generations=generations,
        population_size=POPULATION,
        elite_count=1,
        genome_size=2,
        seed=seed,
    )
    if stagnation_patience is not None:
        cfg_kwargs["stagnation_patience"] = stagnation_patience
    engine = EvolutionEngine(
        fitness_fn=wired,
        config=EngineConfig(**cfg_kwargs),
    )
    initial = make_code_population(
        scenario, POPULATION, random.Random(seed), seed_keys=None, seed_count=0
    )
    engine.run(generations=generations, initial_population=initial)


def first_success_edits(store: ExperienceStore, run_id: str, fse: int) -> list[dict[str, Any]]:
    row = store._conn.execute(
        """SELECT edit_kinds, edit_loci FROM experiences
           WHERE run_id = ? AND eval_index = ?""",
        (run_id, fse),
    ).fetchone()
    if row is None:
        return []
    kinds = json.loads(row[0]) or []
    loci = json.loads(row[1]) or []
    return [
        {"kind": str(k), "lineno": int(l[1]), "col": int(l[2])}
        for k, l in zip(kinds, loci)
        if isinstance(l, list) and len(l) == 3
    ]


def run_cell(
    scenario,
    seeds: list[int],
    generations: int,
    workdir: Path,
    *,
    stagnation_patience: int | None = None,
) -> list[dict[str, Any]]:
    """All seeds at one budget level. Fresh DB per run, extracted then removed."""
    rows: list[dict[str, Any]] = []
    name = scenario.name
    for seed in seeds:
        db = workdir / f"bf_{name}_g{generations}_s{seed}.db"
        rid = f"bf_{name}_g{generations}_s{seed}"
        t0 = time.perf_counter()
        run_one(
            scenario, db, seed, rid,
            generations=generations,
            stagnation_patience=stagnation_patience,
        )
        wall = round(time.perf_counter() - t0, 3)
        store = ExperienceStore(db)
        m = store.run_metrics(rid)
        fse = m.get("first_success_eval")
        row: dict[str, Any] = {
            "scenario": name,
            "seed": seed,
            "generations": generations,
            "first_success_eval": fse,
            "first_success_score": m.get("first_success_score"),
            "evals_total": m.get("evals_total"),
            "wall_seconds": wall,
        }
        if fse is not None:
            row["first_success_edits"] = first_success_edits(store, rid, fse)
        store.close()
        db.unlink()
        rows.append(row)
    return rows


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] (registered house
    implementation, identical to M8/M9 harnesses).
    Rows = (level 32, level 8), columns = (success, no-success)."""
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


def median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 1) if xs else None


def summarize_level(rows: list[dict[str, Any]]) -> dict[str, Any]:
    succ = [r["first_success_eval"] for r in rows if r["first_success_eval"] is not None]
    costs = [
        r["first_success_eval"] if r["first_success_eval"] is not None else r["evals_total"]
        for r in rows
    ]
    truncated = sum(1 for r in rows if (r["evals_total"] or 0) < rows[0]["generations"] * POPULATION)
    return {
        "runs": len(rows),
        "successes_within_budget": len(succ),
        "median_evals_among_successes": median(succ),
        "mean_censored_cost": round(sum(costs) / len(costs), 2) if costs else 0.0,
        "evals_total_min": min((r["evals_total"] or 0) for r in rows) if rows else 0,
        "evals_total_max": max((r["evals_total"] or 0) for r in rows) if rows else 0,
        "runs_truncated_by_stagnation": truncated,
        "wall_seconds_total": round(sum(r["wall_seconds"] for r in rows), 1),
    }


def frontier_curve(rows32: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R4''''': pooled success-vs-eval-budget curve from 32-gen runs."""
    out = []
    for t in GEN_EQUIV_THRESHOLDS:
        out.append({
            "eval_budget": t,
            "gen_equivalent": round(t / POPULATION, 2),
            "successes": sum(
                1 for r in rows32
                if r["first_success_eval"] is not None and r["first_success_eval"] <= t
            ),
        })
    return out


def apply_build_gate(
    per_scenario: dict[str, dict[str, Any]],
    lo_key: str = "8",
    hi_key: str = "32",
) -> dict[str, Any]:
    """R5''''' as a function: testable, no post-hoc reinterpretation."""
    passing = []
    regressions = []
    for name, d in per_scenario.items():
        delta = d["successes"][hi_key] - d["successes"][lo_key]
        if delta < 0:
            regressions.append(name)
        if delta >= BUILD_GATE_DELTA and d["fisher_p"] < FISHER_ALPHA:
            passing.append(name)
    if regressions:
        return {"gate": "DETERMINISM_FAILURE", "regressions": regressions,
                "build_cem_mechanism": False}
    if passing:
        return {"gate": "OPEN", "passing_scenarios": passing,
                "build_cem_mechanism": True}
    return {"gate": "CLOSED", "passing_scenarios": [],
            "build_cem_mechanism": False,
            "verdict": ("budget alone does not open the frontier up to 32 "
                        "generations (~3.97x evals) under the engine's "
                        "stagnation patience; CEM-with-longer-budget "
                        "premise falsified at this scale; no mechanism "
                        "built")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-30")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--levels", default="8,16,32")
    ap.add_argument("--stagnation-patience", type=int, default=None,
                    help="None (default) = engine default 15 (BF-1 legacy); "
                         "an explicit value is the BF-2 governor suspension")
    ap.add_argument("--out", default="reports/budget_frontier.json")
    args = ap.parse_args()

    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    if len(levels) < 2 or any(b <= a for a, b in zip(levels, levels[1:])):
        raise SystemExit("--levels needs >= 2 strictly increasing values")
    lo_key, hi_key = str(levels[0]), str(levels[-1])

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    names = (
        [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios
        else list(SCENARIO_REGISTRY)
    )
    unknown = [n for n in names if n not in SCENARIO_REGISTRY]
    if unknown:
        raise SystemExit(
            f"unknown scenarios: {unknown}; available: {list(SCENARIO_REGISTRY)}"
        )
    scenarios = [SCENARIO_REGISTRY[n]() for n in names]

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="bf_") as td:
        workdir = Path(td)
        for scenario in scenarios:
            cells: dict[int, list[dict[str, Any]]] = {}
            for gens in levels:
                cells[gens] = run_cell(
                    scenario, seeds, gens, workdir,
                    stagnation_patience=args.stagnation_patience,
                )
            per_level = {g: summarize_level(rows) for g, rows in cells.items()}
            a = per_level[levels[-1]]["successes_within_budget"]
            c = per_level[levels[0]]["successes_within_budget"]
            n = len(seeds)
            p_hi = fisher_exact_2x2(a, n - a, c, n - c)
            p_mid = (
                fisher_exact_2x2(
                    per_level[levels[1]]["successes_within_budget"],
                    n - per_level[levels[1]]["successes_within_budget"],
                    c, n - c,
                )
                if len(levels) > 2 else None
            )
            results.append({
                "scenario": scenario.name,
                "difficulty": scenario.difficulty,
                "fingerprint": problem_fingerprint(
                    scenario.sources, scenario.target_file, scenario.func_name
                ),
                "per_level": {str(g): per_level[g] for g in sorted(cells)},
                "successes": {str(g): per_level[g]["successes_within_budget"] for g in sorted(cells)},
                f"fisher_p_{hi_key}_vs_{lo_key}": round(p_hi, 6),
                f"fisher_p_{str(levels[1])}_vs_{lo_key}": (
                    round(p_mid, 6) if p_mid is not None else None
                ),
                f"delta_{hi_key}_vs_{lo_key}": a - c,
                "frontier_curve": frontier_curve(cells[levels[-1]]),
                "runs": {str(g): cells[g] for g in sorted(cells)},
            })

    # R2''''': replication anchor check (only meaningful on the full seed set
    # and the registered ladder)
    anchor_ok = True
    anchor_detail: dict[str, Any] = {}
    if (seeds == SEEDS and len(scenarios) == len(SCENARIO_REGISTRY)
            and levels == LEVELS):
        for r in results:
            want = M9_CONTROL_ANCHOR[r["scenario"]]
            got = r["successes"][lo_key]
            anchor_detail[r["scenario"]] = {"expected": want, "got": got, "ok": got == want}
            if got != want:
                anchor_ok = False

    # R3''''': prefix determinism check (highest level vs lowest level)
    regressions = [
        {"scenario": r["scenario"], "seed": row["seed"],
         "fse_high": hi_row["first_success_eval"]}
        for r in results
        for hi_row, lo_row in (
            [(h, l) for h in r["runs"][hi_key] for l in r["runs"][lo_key]
             if h["seed"] == l["seed"]]
        )
        if (lo_row["first_success_eval"] is not None
            and hi_row["first_success_eval"] is None)
        or (
            lo_row["first_success_eval"] is not None
            and hi_row["first_success_eval"] is not None
            and hi_row["first_success_eval"] > lo_row["first_success_eval"]
        )
    ]

    per_scenario_gate = {
        r["scenario"]: {
            "successes": r["successes"],
            "delta": r[f"delta_{hi_key}_vs_{lo_key}"],
            "fisher_p": r[f"fisher_p_{hi_key}_vs_{lo_key}"],
        }
        for r in results
    }
    gate = apply_build_gate(per_scenario_gate, lo_key, hi_key)

    verdict = {
        "replication_anchor_ok": anchor_ok,
        "replication_anchor_detail": anchor_detail,
        "prefix_determinism_regressions": regressions,
        "build_gate": gate,
    }

    out = {
        "protocol": {
            "name": "budget_frontier",
            "seeds": seeds,
            "levels": list(levels),
            "population": POPULATION,
            "elite_count": 1,
            "genome_size": 2,
            "prior_enabled": False,
            "memory_arms": None,
            "stagnation_patience": args.stagnation_patience,
            "rules": ["R1'''''", "R2'''''", "R3'''''", "R4'''''", "R5'''''", "R6'''''"]
            if args.stagnation_patience is None
            else ["S1", "S2", "S3", "S4", "S5"],
            "m9_control_anchor": M9_CONTROL_ANCHOR,
            "gen_equiv_thresholds": list(GEN_EQUIV_THRESHOLDS),
        },
        "results": results,
        "verdict": verdict,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps({
        "scenarios": [
            {k: r[k] for k in ("scenario", "successes", "fisher_p_32_vs_8", "delta_32_vs_8")}
            for r in results
        ],
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
