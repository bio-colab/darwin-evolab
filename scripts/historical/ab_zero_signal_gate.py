#!/usr/bin/env python3
"""A/B harness for M6 (zero-signal suppression): does the prior's self-gate
deliver byte-identical behavior to prior-off wherever the memory data
carries no differential signal — and what perturbation does it remove?

Background (registered context, not tunable): v2 measured the per-kind
prior's real stores as NEAR-UNIFORM everywhere (weight ratios <= 1.10:1 at
strength 0.5, i.e. <= 10% of the blend's maximal bias) with zero measured
search value at any dose (p >= 0.88) — and v1's scary -14.6%/-63.8%
readings were artifacts of pure RNG-stream reshuffling, not of bias. M6's
gate makes the honest collapse automatic: within the registered envelope
the prior returns None and the caller keeps its exact existing behavior.
M7 and M8 then measured two more memory channels on this same instrument:
both "no detectable effect". The layer-3 picture this harness closes:
the prior must not be ABLE to perturb a search it has nothing to say about.

Pre-registered protocol (written BEFORE any measurement — no scenario,
seed, arm, rule or threshold may be tuned after peeking at results):

  Teacher phase   Per scenario, 3 GA runs (seeds 9901..9903), prior
                  DISABLED, recorded into one store (identical to
                  v2/M7/M8 — comparability).
  Arms            For each seed s in 1..30, three runs per scenario, each
                  on a copy of the frozen teacher snapshot:
                    control      prior_enabled=False (exact pre-memory
                                 behavior, records only)
                    ungated_s05  ExperienceMutationPrior strength=0.5,
                                 zero_signal_gate=False — the exact
                                 pre-M6 v2-era behavior (isolation arm)
                    gated_s05    ExperienceMutationPrior strength=0.5,
                                 zero_signal_gate=True (M6, default)
                  All prior arms read the same frozen snapshot
                  (cache_ttl=1e9); student writes never leak across runs.
  Budget          generations=8, population=12, elite_count=1, genome_size=2
                  (identical to v1/v2/M7/M8 so results are comparable).
  PRIMARY metric  BYTE-IDENTITY: for each (scenario, seed), the gated
                  run's trajectory digest (sha256 over the full
                  per-generation history + evals_total +
                  first_success_eval) must equal the control run's.
                  Claimed endpoint: 120/120 identical pairs.
  Guard metric    pooled holdout successes within budget per arm vs
                  control (Fisher exact) — internal consistency only:
                  if the gate truly collapses to the null-intervention,
                  gated successes CANNOT differ from control.
  Descriptive     ungated vs control: Fisher classification (expected
                  replication of v2's null) + trajectory divergence rate
                  — the quantification of the perturbation M6 removes.

Decision rules (registered here, before the run):
  R1''' M6's PRIMARY claim is TRANSPARENCY, not fitness gain: success iff
        gated == control byte-identically on ALL 120 run-pairs.
  R2''' FALSIFIER: if ANY gated run diverges from its control, no success
        claim is made; the divergent pair's histories are dumped into the
        report and investigated before any further step (either the gate
        failed to fire where v2 said it should, or real signal exists
        somewhere — both must be understood, not argued away).
  R3''' Guard: gated-vs-control Fisher must show no detectable difference
        (identity implies it); a detectable difference would itself be a
        falsifier handled under R2'''.
  R4''' ungated-vs-control results are DESCRIPTIVE (v2 replication +
        perturbation quantification); no M6 claim rides on them.
  R5''' Defaults are unchanged by ANY outcome: the prior stays opt-in/off;
        the gate refines what an opt-in prior may do. This harness can
        never activate anything.
  R6''' Teachers, budget and metrics are byte-comparable with
        ab_memory_value_v2.py / ab_sequence_memory.py (M7) /
        ab_genetic_init_memory.py (M8).

Pre-registered PREDICTION (from v2's measured envelope + the M8 audit's
no-live-single-edit-loci finding): 120/120 byte-identical pairs.

Usage (from the repository root):
  PYTHONPATH=src python scripts/ab_zero_signal_gate.py
  PYTHONPATH=src python scripts/ab_zero_signal_gate.py --seeds 1-2 --out /tmp/m6.json

Stdlib only. Deterministic given fixed --seeds.
"""
from __future__ import annotations

import argparse
import hashlib
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
# (arm name, prior on?, zero_signal_gate) — None gate = prior disabled
ARMS: list[tuple[str, bool, bool | None]] = [
    ("control", False, None),
    ("ungated_s05", True, False),
    ("gated_s05", True, True),
]
FROZEN_CACHE_TTL = 1e9
FISHER_ALPHA = 0.05
CONSISTENCY_SCENARIOS = 3  # of 4


def clean_env() -> None:
    os.environ.pop("EVOLAB_EXPERIENCE", None)
    os.environ.pop("EVOLAB_EXPERIENCE_DB", None)


def trajectory_digest(history: list[dict], metrics: dict[str, Any]) -> str:
    """Canonical digest of a run's trajectory (M6 primary endpoint)."""
    payload = {
        "history": history,
        "evals_total": metrics.get("evals_total"),
        "first_success_eval": metrics.get("first_success_eval"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_one(
    scenario,
    db_path: Path,
    seed: int,
    run_id: str,
    *,
    prior_on: bool,
    gate: bool | None,
    generations: int,
    population: int,
) -> tuple[dict[str, Any], list[dict]]:
    ev = scenario.create_evaluator()
    prior_kwargs = None
    if prior_on:
        prior_kwargs = {
            "mode": "kind",
            "cache_ttl": FROZEN_CACHE_TTL,
            "strength": 0.5,
        }
        if gate is not None:
            prior_kwargs["zero_signal_gate"] = gate
    wired = attach_experience_recorder(
        ev,
        scenario.sources,
        scenario.target_file,
        scenario.func_name,
        db_path=db_path,
        run_id=run_id,
        prior_enabled=prior_on,
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
    report = engine.run(generations=generations, initial_population=initial)
    history = list(report.get("history") or [])
    store = ExperienceStore(db_path)
    metrics = store.run_metrics(run_id)
    store.close()
    return metrics, history


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]].

    Rows = (test arm, control), columns = (success, no-success).
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
    """R1'-family rule as a function (same shape as M7's classify)."""
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


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [
        r["first_success_eval"] if r["first_success_eval"] is not None else r["evals_total"]
        for r in rows
    ]
    success_costs = [r["first_success_eval"] for r in rows if r["first_success_eval"] is not None]
    return {
        "runs": len(rows),
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
            prior_on=False,
            gate=None,
            generations=generations,
            population=population,
        )
    tstore = ExperienceStore(teacher_db)
    fp = problem_fingerprint(scenario.sources, scenario.target_file, scenario.func_name)
    teacher_stats = tstore.stats(fp)
    tstore.close()

    runs: dict[str, list[dict[str, Any]]] = {arm: [] for arm, _, _ in ARMS}
    full_histories: dict[tuple[str, int], list[dict]] = {}
    for s in seeds:
        for arm, prior_on, gate in ARMS:
            db = workdir / f"{arm}_{name}_s{s}.db"
            shutil.copyfile(teacher_db, db)
            rid = f"{arm}_{name}_s{s}"
            metrics, history = run_one(
                scenario,
                db,
                s,
                rid,
                prior_on=prior_on,
                gate=gate,
                generations=generations,
                population=population,
            )
            db.unlink()
            runs[arm].append(
                {
                    "seed": s,
                    "digest": trajectory_digest(history, metrics),
                    "evals_total": metrics.get("evals_total"),
                    "first_success_eval": metrics.get("first_success_eval"),
                    "best_fitness": history[-1]["best_fitness"] if history else None,
                }
            )
            full_histories[(arm, s)] = history

    per_arm = {arm: summarize_arm(rows) for arm, rows in runs.items()}

    # PRIMARY (R1'''): byte identity of gated vs control, per pair.
    identical = 0
    divergent_pairs: list[dict[str, Any]] = []
    identity_by_scenario = 0
    total_pairs = len(seeds)
    for row_c, row_g in zip(runs["control"], runs["gated_s05"]):
        if row_c["digest"] == row_g["digest"]:
            identical += 1
            identity_by_scenario += 1
        else:
            divergent_pairs.append(
                {
                    "seed": row_c["seed"],
                    "control_history": full_histories[("control", row_c["seed"])],
                    "gated_history": full_histories[("gated_s05", row_c["seed"])],
                }
            )

    # Descriptive (R4'''): ungated divergence rate = the perturbation M6 removes.
    ungated_divergent = sum(
        1 for rc, ru in zip(runs["control"], runs["ungated_s05"])
        if rc["digest"] != ru["digest"]
    )

    return {
        "scenario": name,
        "difficulty": scenario.difficulty,
        "teacher_experiences": teacher_stats["total_experiences"],
        "per_arm": per_arm,
        "runs": runs,
        "byte_identity": {
            "identical_pairs": identity_by_scenario,
            "total_pairs": total_pairs,
        },
        "ungated_divergent_pairs": ungated_divergent,
        "_divergent_details": divergent_pairs[:1],  # falsifier dump (first only)
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="1-30", help="inclusive range, e.g. 1-30")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--scenarios", default=",".join(SCENARIO_REGISTRY))
    ap.add_argument("--out", default="reports/ab_zero_signal_gate.json")
    args = ap.parse_args(argv)

    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIO_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown scenarios: {unknown}; available: {list(SCENARIO_REGISTRY)}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="evolab_abm6_") as td:
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
            bi = res["byte_identity"]
            print(
                f"  byte identity gated==control: {bi['identical_pairs']}/{bi['total_pairs']}"
                f" | ungated divergent: {res['ungated_divergent_pairs']}/{len(seeds)}",
                flush=True,
            )

    n_each = len(seeds) * len(results)
    pooled_ctrl = sum(r["per_arm"]["control"]["successes_within_budget"] for r in results)
    pooled: dict[str, Any] = {"control": {"successes_within_budget": pooled_ctrl}}
    for arm, _, _ in ARMS:
        if arm == "control":
            continue
        arm_succ = sum(r["per_arm"][arm]["successes_within_budget"] for r in results)
        deltas = [
            r["per_arm"][arm]["successes_within_budget"]
            - r["per_arm"]["control"]["successes_within_budget"]
            for r in results
        ]
        p = fisher_exact_2x2(arm_succ, n_each - arm_succ, pooled_ctrl, n_each - pooled_ctrl)
        pooled[arm] = {
            "successes_within_budget": arm_succ,
            "vs_control_delta": arm_succ - pooled_ctrl,
            "fisher_p": round(p, 5),
            "per_scenario_deltas": deltas,
            "classification_vs_control": classify_effect(arm_succ, pooled_ctrl, p, deltas),
        }

    total_identical = sum(r["byte_identity"]["identical_pairs"] for r in results)
    total_pairs = sum(r["byte_identity"]["total_pairs"] for r in results)
    falsifiers = [
        {"scenario": r["scenario"], **d}
        for r in results
        for d in r["_divergent_details"]
    ]
    ungated_div = sum(r["ungated_divergent_pairs"] for r in results)

    verdict = {
        "rules_registered": [
            "R1''': PRIMARY claim = transparency, success iff gated==control byte-identical on ALL pairs",
            "R2''': any gated-vs-control divergence is a falsifier: dump, investigate, no success claim",
            "R3''': guard: gated-vs-control Fisher must show no detectable difference (identity implies it)",
            "R4''': ungated-vs-control is descriptive (v2 replication + perturbation quantification)",
            "R5''': defaults unchanged by any outcome: the prior stays opt-in/off",
            "R6''': teachers/budget/metrics byte-comparable with v2/M7/M8 harnesses",
        ],
        "pre_registered_prediction": "120/120 byte-identical pairs (v2 envelope + M8 audit)",
        "byte_identity": {
            "identical_pairs": total_identical,
            "total_pairs": total_pairs,
            "success": total_identical == total_pairs,
        },
        "falsifier_dump": falsifiers if falsifiers else None,
        "ungated_divergence": {
            "divergent_pairs": ungated_div,
            "total_pairs": total_pairs,
            "reading": "the RNG-stream perturbation the M6 gate removes",
        },
        "guard_and_descriptive_fisher": pooled,
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
                    "prior disabled (records only)"
                    if not prior_on
                    else f"kind prior strength=0.5, zero_signal_gate={gate}, "
                    f"frozen teacher snapshot (cache_ttl=1e9)"
                )
                for arm, prior_on, gate in ARMS
            },
            "primary_metric": (
                "byte-identity of full per-generation history + evals_total + "
                "first_success_eval (sha256 digest), gated vs control per (scenario, seed)"
            ),
            "guard_metric": "pooled holdout successes within budget, Fisher exact",
            "harness_lineage": "identical budget/teachers/metrics to v2/M7/M8 harnesses",
        },
        "results": results,
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    identity_ok = total_identical == total_pairs
    identity_label = "SUCCESS" if identity_ok else "FALSIFIED (R2''')"
    print(
        f"BYTE IDENTITY (R1'''): {total_identical}/{total_pairs} pairs identical"
        f" -> {identity_label}"
    )
    print(
        f"ungated divergence (R4'''): {ungated_div}/{total_pairs} pairs diverged"
        " (the perturbation M6 removes)"
    )
    print(
        "guard Fisher (R3'''): "
        + ", ".join(
            f"{arm}={v['successes_within_budget']} ({v['classification_vs_control']}, p={v['fisher_p']})"
            for arm, v in pooled.items()
            if arm != "control"
        )
    )
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
