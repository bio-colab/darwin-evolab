#!/usr/bin/env python3
"""A/B harness for M7 (sequence memory): does conditioning the mutation
prior on the parent's edit-kind PREFIX add value over per-kind marginals,
on the hardened v2 instrument?

Background (registered context, not tunable): v2 (reports/ab_memory_value_v2.json)
showed the per-kind prior is a no-op on this instrument at every dose
(p>=0.88, flat dose-response) — and v1's scare reading (-63.8%) was an
instrument artifact of trajectory reshuffling. M7's hypothesis is NOT that
the kind prior secretly works; it is that per-kind marginals are blind to
COMBINATORIAL success structure (the lru lesson: successes are recipes,
not rates). The honest test is therefore head-to-head against the kind
prior at identical strength, on identical frozen memory, plus the same
absolute (vs control) and reshuffle (s00) checks v2 ran.

Pre-registered protocol (written BEFORE any measurement — no scenario,
seed, arm or rule may be tuned after peeking at results):

  Teacher phase   Per scenario, 3 GA runs (seeds 9901..9903), prior
                  DISABLED, recorded into one store (identical to v2).
  Arms            For each seed s in 1..30, four runs per scenario, each on
                  a copy of the frozen teacher snapshot (control: fresh
                  copy, prior disabled):
                    control   prior_enabled=False (exact pre-memory behavior)
                    kind_s05  ExperienceMutationPrior,  strength=0.5
                    seq_s00   ExperienceSequencePrior, strength=0.0 (pure
                              reshuffle isolation for the sequence arm)
                    seq_s05   ExperienceSequencePrior, strength=0.5 (M7)
                  All prior arms read the same frozen snapshot
                  (cache_ttl=1e9); student writes never leak across runs.
  Budget          generations=8, population=12, elite_count=1, genome_size=2
                  (identical to v1/v2 so results are comparable).
  Primary metric  holdout successes within budget, out of 30 per scenario
                  per arm (120 pooled).
  Secondary       median evals-to-first-success among that arm's successes.
  Statistics      two-sided Fisher exact test on the pooled 2x2 table.

Decision rules (registered here, before the run):
  R1'  Each arm is classified vs control as
         "promising"   iff pooled successes > control AND Fisher p < 0.05
                       AND the surplus sign holds in >= 3 of 4 scenarios;
         "harmful"     iff pooled successes < control AND Fisher p < 0.05
                       AND the deficit sign holds in >= 3 of 4 scenarios;
         otherwise     "no_detectable_effect at this power".
  R2'  If seq_s00 (pure reshuffle) is detectable vs control, the seq_s05
       BIAS verdict must be drawn vs seq_s00 (same rule), not vs control.
  R3'  Any "promising" classification does NOT change defaults: the prior
       stays opt-in/off. It only registers that a follow-up experiment is
       warranted.
  R4'  M7's PRIMARY registered comparison: seq_s05 vs kind_s05 head-to-head
       (same rule R1', same alpha). "promising" here means sequence
       conditioning carries information per-kind marginals miss; "harmful"
       means the conditioning hurts relative to the incumbent; anything
       else means no measurable difference at this power.
  R5'  Per-scenario deltas (including lru_cache_logic, the scenario that
       motivated M7) are DESCRIPTIVE ONLY — no per-scenario gate exists.
       Reading gates per-scenario after peeking is exactly how v1 misled
       itself; that failure mode is forbidden here.

Usage (from the repository root):
  PYTHONPATH=src python scripts/ab_sequence_memory.py
  PYTHONPATH=src python scripts/ab_sequence_memory.py --seeds 1-3 --scenarios lru_cache_logic --out /tmp/ab_seq.json

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
# (arm name, mode or None for control, strength)
ARMS: list[tuple[str, str | None, float | None]] = [
    ("control", None, None),
    ("kind_s05", "kind", 0.5),
    ("seq_s00", "sequence", 0.0),
    ("seq_s05", "sequence", 0.5),
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
    mode: str | None,
    strength: float | None,
    generations: int,
    population: int,
) -> None:
    ev = scenario.create_evaluator()
    prior_kwargs = None
    if mode is not None:
        prior_kwargs = {
            "mode": mode,
            "cache_ttl": FROZEN_CACHE_TTL,
            "strength": strength,
        }
    wired = attach_experience_recorder(
        ev,
        scenario.sources,
        scenario.target_file,
        scenario.func_name,
        db_path=db_path,
        run_id=run_id,
        prior_enabled=mode is not None,
        prior_kwargs=prior_kwargs,
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
    (Identical implementation to ab_memory_value_v2 — unit-tested there.)
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
    base_successes: int,
    fisher_p: float,
    per_scenario_deltas: list[int],
) -> str:
    """R1' as a function: testable, no post-hoc reinterpretation."""
    n_pos = sum(1 for d in per_scenario_deltas if d > 0)
    n_neg = sum(1 for d in per_scenario_deltas if d < 0)
    consistent = n_pos >= CONSISTENCY_SCENARIOS or n_neg >= CONSISTENCY_SCENARIOS
    if fisher_p < FISHER_ALPHA and consistent:
        if arm_successes > base_successes:
            return "promising"
        if arm_successes < base_successes:
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
            mode=None,
            strength=None,
            generations=generations,
            population=population,
        )
    tstore = ExperienceStore(teacher_db)
    fp = problem_fingerprint(scenario.sources, scenario.target_file, scenario.func_name)
    teacher_stats = tstore.stats(fp)
    teacher_seq = tstore.sequence_stats(fp)
    tstore.close()

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm, _, _ in ARMS}
    for s in seeds:
        for arm, mode, strength in ARMS:
            db = workdir / f"{arm}_{name}_s{s}.db"
            shutil.copyfile(teacher_db, db)
            rid = f"{arm}_{name}_s{s}"
            run_one(
                scenario,
                db,
                s,
                rid,
                mode=mode,
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
        for arm, _, _ in ARMS
        if arm != "control"
    }
    return {
        "scenario": name,
        "difficulty": scenario.difficulty,
        "teacher_experiences": teacher_stats["total_experiences"],
        "teacher_transitions": len(teacher_seq.get("transitions") or {}),
        "per_arm": per_arm,
        "success_deltas_vs_control": deltas,
        "runs": arm_rows,
    }


def pairwise(
    arm_succ: int,
    base_succ: int,
    n_each: int,
    per_scenario_deltas: list[int],
) -> dict[str, Any]:
    p = fisher_exact_2x2(arm_succ, n_each - arm_succ, base_succ, n_each - base_succ)
    return {
        "successes_within_budget": arm_succ,
        "vs_base_delta": arm_succ - base_succ,
        "fisher_p": round(p, 5),
        "per_scenario_deltas": per_scenario_deltas,
        "classification": classify_effect(arm_succ, base_succ, p, per_scenario_deltas),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="1-30", help="inclusive range, e.g. 1-30")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--scenarios", default=",".join(SCENARIO_REGISTRY))
    ap.add_argument("--out", default="reports/ab_sequence_memory.json")
    args = ap.parse_args(argv)

    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIO_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown scenarios: {unknown}; available: {list(SCENARIO_REGISTRY)}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="evolab_abseq_") as td:
        for name in names:
            scenario = SCENARIO_REGISTRY[name]()
            print(f"=== {name} (difficulty={scenario.difficulty}) ===", flush=True)
            res = run_scenario(scenario, seeds, args.generations, args.population, Path(td))
            results.append(res)
            line = ", ".join(
                f"{arm}={res['per_arm'][arm]['successes_within_budget']}/{len(seeds)}"
                for arm, _, _ in ARMS
            )
            print(f"  successes within budget: {line}", flush=True)

    n_each = len(seeds) * len(results)
    pooled_ctrl = sum(
        r["per_arm"]["control"]["successes_within_budget"] for r in results
    )
    pooled: dict[str, dict[str, Any]] = {"control": {"successes_within_budget": pooled_ctrl}}

    # R1': every prior arm vs control
    for arm, _, _ in ARMS:
        if arm == "control":
            continue
        arm_succ = sum(r["per_arm"][arm]["successes_within_budget"] for r in results)
        deltas = [r["success_deltas_vs_control"][arm] for r in results]
        entry = pairwise(arm_succ, pooled_ctrl, n_each, deltas)
        entry["classification_vs_control"] = entry.pop("classification")
        pooled[arm] = entry

    # R4' (M7 primary): seq_s05 head-to-head vs kind_s05
    kind_succ = pooled["kind_s05"]["successes_within_budget"]
    seq_succ = pooled["seq_s05"]["successes_within_budget"]
    deltas_vs_kind = [
        r["per_arm"]["seq_s05"]["successes_within_budget"]
        - r["per_arm"]["kind_s05"]["successes_within_budget"]
        for r in results
    ]
    seq_vs_kind = pairwise(seq_succ, kind_succ, n_each, deltas_vs_kind)
    pooled["seq_s05"]["head_to_head_vs_kind_s05"] = {
        k: v for k, v in seq_vs_kind.items() if k != "successes_within_budget"
    }

    # R2': if the sequence reshuffle arm is detectable vs control, the bias
    # verdict for seq_s05 must be drawn vs seq_s00 instead.
    reshuffle_detectable = (
        pooled["seq_s00"]["classification_vs_control"] != "no_detectable_effect"
    )
    if reshuffle_detectable:
        s00_succ = pooled["seq_s00"]["successes_within_budget"]
        deltas_vs_s00 = [
            r["per_arm"]["seq_s05"]["successes_within_budget"]
            - r["per_arm"]["seq_s00"]["successes_within_budget"]
            for r in results
        ]
        pooled["seq_s05"]["classification_vs_reshuffle"] = pairwise(
            seq_succ, s00_succ, n_each, deltas_vs_s00
        )["classification"]

    verdict = {
        "rules_registered": [
            "R1': promising/harmful iff Fisher p<0.05 AND sign-consistent in >=3/4 scenarios (vs control)",
            "R2': if seq_s00 (pure reshuffle) is detectable, seq_s05 bias verdicts are drawn vs seq_s00",
            "R3': 'promising' only registers a follow-up experiment; defaults unchanged (prior stays opt-in/off)",
            "R4': M7 PRIMARY comparison = seq_s05 vs kind_s05 head-to-head (same rule, same alpha)",
            "R5': per-scenario deltas (incl. lru_cache_logic) are descriptive only — no per-scenario gate",
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
            "arms": {
                arm: (
                    "prior disabled"
                    if mode is None
                    else f"{mode} prior strength={strength}, frozen teacher snapshot (cache_ttl=1e9)"
                )
                for arm, mode, strength in ARMS
            },
            "primary_metric": "holdout successes within budget (censoring-free count)",
            "secondary_metric": "median evals-to-first-success among successes",
            "analysis": "unpaired across arms (weighted rng consumes the RNG stream differently; v2-documented)",
            "statistics": f"two-sided Fisher exact on pooled 2x2, alpha={FISHER_ALPHA}",
            "harness_lineage": "identical budget/teachers/metrics to ab_memory_value_v2.py (comparability)",
        },
        "results": results,
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    ctrl_line = f"control={pooled_ctrl}/{n_each}"
    print(f"POOLED successes within budget: {ctrl_line}, " + ", ".join(
        f"{arm}={v['successes_within_budget']} ({v['classification_vs_control']}, p={v['fisher_p']})"
        for arm, v in pooled.items()
        if arm != "control"
    ))
    print(f"R4' M7 primary — seq_s05 vs kind_s05: {pooled['seq_s05']['head_to_head_vs_kind_s05']}")
    if reshuffle_detectable:
        print("R2' active: seq_s05 bias verdict drawn vs seq_s00: "
              f"{pooled['seq_s05'].get('classification_vs_reshuffle')}")
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
