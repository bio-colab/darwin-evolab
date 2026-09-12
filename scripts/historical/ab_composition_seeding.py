#!/usr/bin/env python3
"""M9 A/B harness: composition-seeded initialization — positive memory via
the genetic channel, replay-proof by construction.

Forensic basis (established BEFORE this protocol was written; recon used
only the committed instrument and teacher-lineage runs):
  * Materialization: a successful program is rebuildable from its stored
    skeleton (kind, locus) on this instrument — the pattern catalog
    regenerates payloads deterministically from the frozen scenario
    sources, and apply_edits is one-pass over the original tree (no line
    shifting, order-independent). Verified empirically: rebuilt genomes
    scored exactly 100.0 (byte-equal to the stored score) and passed
    holdout on requests/lru/multi_file. The M8 "payload wall" is real for
    the GENERIC store schema (no payload column) but surmountable on a
    frozen-source instrument — this protocol is the registered test of
    what that unlocks.
  * Winner uniqueness: across all successful teacher runs, each scenario
    has exactly ONE distinct successful composition (requests:
    auth_prefix@(2,13)+string_sep@(4,12) x34; lru:
    hit_move_to_end@(5,12)+pop_to_front@(7,12) x27; multi_file:
    compare_flip@(4,15)+swap_int_args@(5,15) x41; click with 24 teachers:
    bool_flip@(5,30)+int_wrap@(7,12)+index_flip@(9,29) x17, 2/24 runs
    succeeded). Zero single-edit successes anywhere (third confirmation
    of the M8 audit conclusion).
  * Subset hazard (the design-forcing discovery): on click, the 2-subset
    {int_wrap@(7,12), index_flip@(9,29)} of the remembered 3-edit winner
    PASSES holdout (score 60) — remembered winners are not minimal, so
    seeding k >= 2 plants candidate ANSWERS (replay: the cache/archive
    value class, outside the hypothesis space by the repo's registered
    position). Hence k=1 BY CONSTRUCTION: every single-edit genotype is a
    proven dead door (committed M8 firecheck: the mined dead doors are the
    entire catalog; zero single-edit successes in all teacher data).
    Seeded individuals structurally CANNOT pass at generation zero — all
    gain, if any, is search completion guided by a remembered fragment.
  * Zero-signal contract: no successful multi-edit rows ->
    composition_seeds returns None -> mem arms behave exactly like
    control (M6 lesson: memory with nothing to say must not shuffle).

REGISTERED DEVIATION from the v2/M7/M8 instrument (disclosed, structural
necessity — registered here before any measurement):
  Teacher phase 3 -> 24 runs (seeds 9901..9924). Composition memory
  requires at least one successful teacher run; click's registered
  3-teacher phase yields zero successes (measured: 0/3), which would
  leave the only high-headroom scenario (control 5/30) without memory and
  the whole protocol without power. P(click empty | 24 teachers) ~=
  0.83^24 ~= 0.8%. The STUDENT phase is identical to v2/M7/M8: 30 seeds,
  budget generations=8 / population=12 / elite_count=1, prior disabled,
  no avoidance — the genetic channel only.

Arms (per scenario x seed; 4 arms = 480 student runs + 96 teacher runs):
  control   seed_keys=None, seed_count=0 (legacy init, byte-identical)
  rand_s6   seed_count=6; each seeded individual gets ONE edit drawn
            uniformly from the catalog via the run's rng — the
            mechanism-free matched warm start (same count, same k=1,
            no memory direction)
  mem_s3    seed_count=3; each seeded individual gets ONE edit from the
            remembered winner (round-robin over the winner's edits)
  mem_s6    seed_count=6; same rule as mem_s3
  Dose axis: s3 vs s6 (R5''''). Isolation axis: rand_s6 vs mem_s6 (R3'''').

Budget     generations=8, population=12, elite_count=1 (identical to
           v1/v2/M7/M8 student phases so the headline metric stays
           comparable).
Primary    holdout successes within budget, out of 30 per scenario per
           arm (120 pooled).
Secondary  median evals-to-first-success among that arm's successes.
Statistics two-sided Fisher exact on the pooled 2x2, unpaired (same
           rationale as v2/M7/M8: any mechanism that consumes the RNG
           stream differently diverges trajectories, so pairing is fake).

Decision rules (registered here, BEFORE the run):
  R1'''' Classification vs control (identical to M8's R1''):
         "promising" iff pooled successes > control AND Fisher p < 0.05
         AND surplus sign holds in >= 3/4 scenarios; "harmful" iff
         < control AND p < 0.05 AND deficit consistent; else
         "no_detectable_effect at this power".
  R2'''' Replay firewall (the R2 discipline of this protocol):
         (a) k=1 is structural (mining contract n_edits>=2 + one edit per
             seeded individual);
         (b) the report must show ZERO gen-0 passes (first_success_eval
             <= population size) in every arm — any gen-0 pass in a mem
             arm falsifies the replay-proof assumption and forces
             reinterpretation before any value claim;
         (c) first-pass composition accounting is committed per arm:
             winner_full (first passing genotype == remembered winner),
             winner_subset (proper subset of remembered loci — a
             memory-guided discovery, counted as search success), other.
             A "promising" verdict whose successes are predominantly
             winner_full is re-labeled replay by this rule — no
             memory-value claim may be made.
  R3'''' Isolation primacy: the memory-value verdict is drawn mem_s6 vs
         rand_s6 (matched warm start). If mem beats control but not
         rand_s6, the verdict is "warm-start effect, no memory-specific
         value".
  R4'''' Gate: any "promising" classification does NOT enable seeding by
         default — it only registers that a follow-up registered protocol
         is warranted. Default stays off until gain > 0 is proven on this
         instrument.
  R5'''' Dose: if mem_s3 and mem_s6 disagree in classification vs
         control, the verdict is "dose-sensitive at this power" — no
         default either way.
  R6'''' Transparency (committed with the report): per-scenario memory
         content (winners with kinds/loci/counts, the exact seed keys,
         k used), teacher success counts, the teacher snapshot's
         single-edit success count (replay-proof verification: must be 0),
         per-arm gen-0 pass counts, and per-arm first-pass composition
         classification.

Usage (from the repository root):
  PYTHONPATH=src python scripts/ab_composition_seeding.py
  PYTHONPATH=src python scripts/ab_composition_seeding.py --seeds 1-5 \
      --scenarios requests_http_helper --out /tmp/ab9.json

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
from evolab.code_fixtures import (  # noqa: E402
    SCENARIO_REGISTRY,
    make_code_population,
)
from evolab.experience import (  # noqa: E402
    ExperienceStore,
    attach_experience_recorder,
    problem_fingerprint,
)
from evolab.repair import catalog_sources  # noqa: E402

TEACHER_SEED_START = 9901
TEACHERS = 24  # registered deviation: 3 -> 24 (see header)
ARMS: list[str] = ["control", "rand_s6", "mem_s3", "mem_s6"]
SEED_COUNTS: dict[str, int] = {"control": 0, "rand_s6": 6, "mem_s3": 3, "mem_s6": 6}
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
    seed_keys: list[tuple[str, int, int, str]] | None,
    seed_count: int,
) -> None:
    """One student run. The rand arm's key randomness lives in
    ``_rand_keys`` (called by the harness before this), so every arm's
    population construction is deterministic given its keys and seed."""
    ev = scenario.create_evaluator()
    wired = attach_experience_recorder(
        ev,
        scenario.sources,
        scenario.target_file,
        scenario.func_name,
        db_path=db_path,
        run_id=run_id,
        prior_enabled=False,  # M9 runs the genetic channel, never the prior
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
        seed_keys=seed_keys, seed_count=seed_count,
    )
    engine.run(generations=generations, initial_population=initial)


def _rand_keys(scenario, seed: int, count: int) -> list[tuple[str, int, int, str]]:
    """Mechanism-free warm start: ONE uniform catalog edit per seeded slot,
    drawn from the run's own rng stream (deterministic given seed)."""
    rng = random.Random(seed)
    catalog = catalog_sources(scenario.sources)
    return [
        (e.file, e.lineno, e.col_offset, e.kind)
        for e in (catalog[rng.randrange(len(catalog))] for _ in range(count))
    ]


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]].
    Rows = (test arm, control), columns = (success, no-success)."""
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
    """R1'''' as a function: testable, no post-hoc reinterpretation."""
    n_pos = sum(1 for d in per_scenario_deltas if d > 0)
    n_neg = sum(1 for d in per_scenario_deltas if d < 0)
    consistent = n_pos >= CONSISTENCY_SCENARIOS or n_neg >= CONSISTENCY_SCENARIOS
    if fisher_p < FISHER_ALPHA and consistent:
        if arm_successes > control_successes:
            return "promising"
        if arm_successes < control_successes:
            return "harmful"
    return "no_detectable_effect"


def classify_first_pass(
    first_pass_edits: list[tuple[str, int, int]],
    winner_loci: set[tuple[int, int]],
) -> str:
    """R2''''(c): classify the first passing genotype against the remembered
    winner. ``first_pass_edits``: [(kind, lineno, col), ...]; winner_loci:
    {(lineno, col), ...} of the remembered winner (same file)."""
    if not first_pass_edits:
        return "other"
    loci = {(ln, col) for _, ln, col in first_pass_edits}
    if loci == winner_loci:
        return "winner_full"
    if loci < winner_loci:
        return "winner_subset"
    return "other"


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


def memory_report(store: ExperienceStore, fingerprint: str, scenario) -> dict[str, Any]:
    """R6'''': the exact memory content the mem arms consume, plus the
    replay-proof verification (teacher single-edit successes must be 0)."""
    row = store._conn.execute(
        """SELECT COUNT(*) FROM experiences
           WHERE problem_fingerprint = ? AND passed_holdout = 1
             AND n_edits = 1""",
        (fingerprint,),
    ).fetchone()
    single_successes = int(row[0]) if row else 0
    winners = store.composition_seeds(fingerprint, max_winners=3)
    out: dict[str, Any] = {
        "teacher_single_edit_successes": single_successes,
        "replay_proof_holds": single_successes == 0,
        "memory_present": winners is not None,
        "winners": [],
    }
    if winners:
        for w in winners:
            out["winners"].append(
                {
                    "count": w["count"],
                    "edits": [
                        {"kind": k, "file": f, "lineno": ln, "col": c}
                        for k, f, ln, c in w["edits"]
                    ],
                }
            )
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
            scenario, teacher_db, TEACHER_SEED_START + i, f"teacher_{name}_t{i}",
            generations=generations, population=population,
            seed_keys=None, seed_count=0,
        )
    tstore = ExperienceStore(teacher_db)
    mem = memory_report(tstore, fingerprint, scenario)
    tstore.close()

    # the remembered winner's keys (mem arms) — None when memory absent
    mem_keys: list[tuple[str, int, int, str]] | None = None
    winner_loci: set[tuple[int, int]] = set()
    if mem["memory_present"]:
        mem_keys = [
            (e["file"], e["lineno"], e["col"], e["kind"])
            for e in mem["winners"][0]["edits"]
        ]
        winner_loci = {(e["lineno"], e["col"]) for e in mem["winners"][0]["edits"]}

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    first_pass: dict[str, dict[str, int]] = {
        arm: {"winner_full": 0, "winner_subset": 0, "other": 0} for arm in ARMS
    }
    gen0_passes: dict[str, int] = {arm: 0 for arm in ARMS}

    for s in seeds:
        for arm in ARMS:
            db = workdir / f"{arm}_{name}_s{s}.db"
            shutil.copyfile(teacher_db, db)  # same snapshot lineage for every arm
            rid = f"{arm}_{name}_s{s}"
            count = SEED_COUNTS[arm]
            if arm == "rand_s6":
                keys = _rand_keys(scenario, s, count)
            else:
                keys = mem_keys
            run_one(
                scenario, db, s, rid,
                generations=generations, population=population,
                seed_keys=keys, seed_count=count,
            )
            store = ExperienceStore(db)
            metrics = store.run_metrics(rid)
            # R2''''(c): classify the first passing genotype
            fse = metrics.get("first_success_eval")
            if fse is not None:
                if fse <= population:
                    gen0_passes[arm] += 1  # R2''''(b): must stay zero
                row = store._conn.execute(
                    """SELECT edit_kinds, edit_loci FROM experiences
                       WHERE run_id = ? AND eval_index = ?""",
                    (rid, fse),
                ).fetchone()
                if row is not None:
                    kinds = json.loads(row[0]) or []
                    loci = json.loads(row[1]) or []
                    edits = [
                        (str(k), int(l[1]), int(l[2]))
                        for k, l in zip(kinds, loci)
                        if isinstance(l, list) and len(l) == 3
                    ]
                    first_pass[arm][classify_first_pass(edits, winner_loci)] += 1
            store.close()
            db.unlink()
            arm_rows[arm].append(metrics)

    per_arm = {arm: summarize_arm(rows) for arm, rows in arm_rows.items()}
    deltas = {
        arm: per_arm[arm]["successes_within_budget"] - per_arm["control"]["successes_within_budget"]
        for arm in ARMS if arm != "control"
    }
    return {
        "scenario": name,
        "difficulty": scenario.difficulty,
        "teachers": TEACHERS,
        "memory": mem,
        "k_used": 1,
        "seed_keys_used": [
            {"kind": k, "file": f, "lineno": ln, "col": c}
            for f, ln, c, k in (mem_keys or [])
        ],
        "per_arm": per_arm,
        "success_deltas_vs_control": deltas,
        "gen0_passes": gen0_passes,
        "first_pass_composition": first_pass,
        "runs": arm_rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="1-30", help="inclusive range, e.g. 1-30")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--scenarios", default=",".join(SCENARIO_REGISTRY))
    ap.add_argument("--out", default="reports/ab_composition_seeding.json")
    args = ap.parse_args(argv)

    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIO_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown scenarios: {unknown}; available: {list(SCENARIO_REGISTRY)}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="evolab_ab9_") as td:
        for name in names:
            scenario = SCENARIO_REGISTRY[name]()
            print(f"=== {name} (difficulty={scenario.difficulty}) ===", flush=True)
            res = run_scenario(scenario, seeds, args.generations, args.population, Path(td))
            results.append(res)
            line = ", ".join(
                f"{arm}={res['per_arm'][arm]['successes_within_budget']}/{len(seeds)}"
                for arm in ARMS
            )
            print(f"  successes within budget: {line}", flush=True)
            print(
                f"  memory present={res['memory']['memory_present']} "
                f"single-edit successes={res['memory']['teacher_single_edit_successes']} "
                f"gen0 passes={res['gen0_passes']}",
                flush=True,
            )

    # ---- pooled analysis ----
    n_each = len(seeds) * len(results)
    pooled_ctrl = sum(r["per_arm"]["control"]["successes_within_budget"] for r in results)
    pooled: dict[str, Any] = {}
    for arm in ARMS:
        if arm == "control":
            continue
        arm_succ = sum(r["per_arm"][arm]["successes_within_budget"] for r in results)
        p = fisher_exact_2x2(arm_succ, n_each - arm_succ, pooled_ctrl, n_each - pooled_ctrl)
        sc_deltas = [
            r["per_arm"][arm]["successes_within_budget"] - r["per_arm"]["control"]["successes_within_budget"]
            for r in results
        ]
        pooled[arm] = {
            "successes_within_budget": arm_succ,
            "vs_control_delta": arm_succ - pooled_ctrl,
            "fisher_p_vs_control": round(p, 5),
            "per_scenario_deltas": sc_deltas,
            "classification_vs_control": classify_effect(
                arm_succ, pooled_ctrl, p, sc_deltas
            ),
        }
    # R3'''': the memory-value verdict is drawn mem_s6 vs rand_s6
    mem6 = pooled["mem_s6"]["successes_within_budget"]
    rand6 = pooled["rand_s6"]["successes_within_budget"]
    p_mem_vs_rand = fisher_exact_2x2(mem6, n_each - mem6, rand6, n_each - rand6)
    mem_vs_rand = {
        "mem_s6": mem6,
        "rand_s6": rand6,
        "delta": mem6 - rand6,
        "fisher_p": round(p_mem_vs_rand, 5),
        "memory_value_classification": (
            "memory_specific_value" if (p_mem_vs_rand < FISHER_ALPHA and mem6 > rand6)
            else "warm_start_or_noise"
        ),
    }
    # R2''''(b): replay firewall
    gen0_total = {
        arm: sum(r["gen0_passes"][arm] for r in results) for arm in ARMS
    }
    # R5'''': dose
    dose_sensitive = (
        pooled["mem_s3"]["classification_vs_control"]
        != pooled["mem_s6"]["classification_vs_control"]
    )
    verdict = {
        "rules_registered": [
            "R1'''': promising/harmful iff Fisher p<0.05 AND sign-consistent in >=3/4 scenarios",
            "R2'''': replay firewall — k=1 structural; gen-0 passes must be 0; "
            "first-pass accounting committed; winner_full-dominated gain re-labeled replay",
            "R3'''': memory-value verdict drawn mem_s6 vs rand_s6 (matched warm start)",
            "R4'''': promising only registers a follow-up; default stays off",
            "R5'''': mem_s3 vs mem_s6 disagreement => dose-sensitive, no default",
            "R6'''': transparency block committed with the report",
        ],
        "pooled_control": pooled_ctrl,
        "pooled": pooled,
        "mem_vs_rand_matched": mem_vs_rand,
        "gen0_passes_total": gen0_total,
        "replay_firewall_holds": all(v == 0 for v in gen0_total.values()),
        "dose_sensitive": dose_sensitive,
        "deviation_registered": "teacher phase 3 -> 24 (structural necessity, header)",
    }

    report = {
        "protocol": {
            "seeds": seeds,
            "teachers": TEACHERS,
            "teacher_seeds": list(range(TEACHER_SEED_START, TEACHER_SEED_START + TEACHERS)),
            "generations": args.generations,
            "population": args.population,
            "k": 1,
            "arms": {
                "control": "seed_keys=None, seed_count=0 (legacy init)",
                "rand_s6": "seed_count=6, one uniform catalog edit per seeded slot (mechanism-free)",
                "mem_s3": "seed_count=3, one remembered winner edit per seeded slot (round-robin)",
                "mem_s6": "seed_count=6, same rule as mem_s3",
            },
            "deviation": (
                "teacher phase 3 -> 24 runs (seeds 9901..9924): composition memory "
                "requires >=1 successful teacher run; click's 3-teacher phase yields "
                "0 successes (measured). Student phase identical to v2/M7/M8."
            ),
            "dead_mechanism_note": (
                "k=1 replay-proof by construction: single-edit genotypes are proven "
                "dead doors (committed M8 firecheck); winners mined with n_edits>=2 only."
            ),
        },
        "results": results,
        "verdict": verdict,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport written to {out_path}", flush=True)

    print("\n=== pooled verdict (registered rules) ===")
    for arm in ("rand_s6", "mem_s3", "mem_s6"):
        v = pooled[arm]
        print(
            f"{arm:8s} {v['successes_within_budget']:3d} "
            f"({v['vs_control_delta']:+d}) p={v['fisher_p_vs_control']:.5f} "
            f"-> {v['classification_vs_control']}"
        )
    print(
        f"R3'''': mem_s6 {mem6} vs rand_s6 {rand6} "
        f"p={mem_vs_rand['fisher_p']:.5f} -> {mem_vs_rand['memory_value_classification']}"
    )
    print(f"R2'''': gen0 passes {gen0_total} (firewall holds: {verdict['replay_firewall_holds']})")
    print(f"R5'''': dose_sensitive={dose_sensitive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
