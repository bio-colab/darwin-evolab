#!/usr/bin/env python3
"""Cache-key swap A/B (CK): materialized-sources identity for the M3 cache.

BACKGROUND (registered before the post-swap measurement):
  The M3 cache (evolab.experience.EvaluationCache) keys edit-genomes on the
  ordered edit recipe (kinds+loci+payloads) + base hash — a safe-false
  choice: different recipes producing the SAME program never merge (never
  wrong, sometimes wasteful). The committed probe measured the cost:
  click visited 438 distinct edit keys producing only 77 distinct programs
  (5.7x path duplication); realized savings 54.8% vs a 92.1% cross-run /
  72.8% within-run ground-truth ceiling. This swap re-keys edit-genomes on
  the CANONICAL JSON OF THE APPLIED SOURCES (``RepairGenome.apply_to()`` —
  the full materialized state, same identity semantics as the existing
  source-genome branch). Two recipes reaching the same materialized state
  now merge; two different materialized states never do (dict equality is
  per-file text equality — the evaluation-relevant state in full). The M3
  probe's own registered rule anticipated this: "otherwise the cache must
  key on the program source hash".

  Scope note (honest): the per-run cache can only realize WITHIN-run
  duplicates. Registered within-run ground-truth ceilings (committed probe,
  seeds 1-10): click 72.8%, requests 92.0%, lru 92.2%, multi_file 95.9%.
  The 92.1% figure in the M3 disclosure is the CROSS-run ceiling and needs
  a persistent cache — out of scope here; the disclosure is corrected in
  the README as part of this work.

REGISTERED RULES (written before the post-swap run; baselines from
committed reports only — reports/duplicate_evals_probe.json and
reports/duplicate_evals_probe_cached.json):

  C1  Identity: for every (scenario, seed in 1-10) the hash-sequence
      digest of every evaluator attempt must be EQUAL across all three
      wirings — no-cache, old-key cache (baseline capture), new-key cache
      (after capture) — and per-seed DB metrics (rows, first_success_eval,
      first_success_score, evals_total) must match. Any divergence means
      evaluation is not a pure function of the materialized program —
      STOP, file as a bug, revert.
  C2  Savings (realized, cached mode, pooled over seeds 1-10 per scenario):
      click >= 70% (baseline 54.8%, within-run ceiling 72.8%);
      requests >= 90% (baseline 88.1%, ceiling 92.0%);
      lru >= 91% (baseline 89.0%, ceiling 92.2%);
      multi_file >= 91% (baseline 89.1%, ceiling 95.9%);
      and NO scenario may regress by more than 1 percentage point vs its
      committed baseline.
  C3  Overhead: mean wall-clock per run (cached mode) with the new key
      must not exceed the old-key baseline mean by more than 5%.
  C4  Adoption: the swap stays default ONLY if C1 and C2 and C3 all hold;
      otherwise revert to the recipe key and document.

INVOCATIONS (registered):
  PYTHONPATH=src python scripts/ab_cache_key_swap.py --mode baseline \
      --out reports/cache_key_baseline.json     # BEFORE the swap
  PYTHONPATH=src python scripts/ab_cache_key_swap.py --mode after \
      --out reports/cache_key_after.json        # AFTER the swap
  PYTHONPATH=src python scripts/ab_cache_key_swap.py --mode compare \
      --baseline reports/cache_key_baseline.json \
      --after reports/cache_key_after.json \
      --out reports/cache_key_verdict.json

Stdlib only. Deterministic given fixed seeds; protocol byte-identical to
the committed probe (seeds 1-10, generations=8, population=12, prior off).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_duplicate_evals as probe_mod  # noqa: E402  (committed probe reused, no drift)

from evolab import EngineConfig, EvolutionEngine  # noqa: E402
from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population  # noqa: E402
from evolab.experience import (  # noqa: E402
    ExperienceStore,
    attach_eval_cache,
    attach_experience_recorder,
)

SEEDS = list(range(1, 11))
GENERATIONS = 8
POPULATION = 12

BASELINE_SAVINGS = {  # committed: reports/duplicate_evals_probe_cached.json
    "click_cli_parser": 0.548,
    "requests_http_helper": 0.881,
    "lru_cache_logic": 0.890,
    "multi_file_config": 0.891,
}
BASELINE_C2_FLOOR = {  # registered in C2 above
    "click_cli_parser": 0.70,
    "requests_http_helper": 0.90,
    "lru_cache_logic": 0.91,
    "multi_file_config": 0.91,
}
C2_MAX_REGRESSION = 0.01
C3_MAX_OVERHEAD = 0.05


def seq_digest(hashes: list[str]) -> str:
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def capture_wiring(
    scenario_name: str,
    seeds: list[int],
    workdir: Path,
    use_cache: bool,
) -> dict[str, Any]:
    """One wiring (no-cache or cached) over all seeds — mirrors the committed
    probe's run loop exactly, adding per-seed digests, wall time and DB metrics."""
    probe_mod.clean_env()
    scenario = SCENARIO_REGISTRY[scenario_name]()
    db_path = workdir / f"ck_{scenario_name}.db"
    per_seed: list[dict[str, Any]] = []
    cache_acc = {"hits": 0, "misses": 0, "bypasses": 0}

    for s in seeds:
        inner = scenario.create_evaluator()
        if use_cache:
            inner = attach_eval_cache(inner)
        probe = probe_mod.DuplicateProbe(inner)
        run_id = f"probe_{scenario_name}_s{s}"
        wired = attach_experience_recorder(
            probe,
            scenario.sources,
            scenario.target_file,
            scenario.func_name,
            db_path=db_path,
            run_id=run_id,
            prior_enabled=False,
        )
        engine = EvolutionEngine(
            fitness_fn=wired,
            config=EngineConfig(
                generations=GENERATIONS,
                population_size=POPULATION,
                elite_count=1,
                genome_size=2,
                seed=s,
            ),
        )
        import random

        initial = make_code_population(scenario, POPULATION, random.Random(s))
        t0 = time.perf_counter()
        engine.run(generations=GENERATIONS, initial_population=initial)
        wall = round(time.perf_counter() - t0, 4)

        store = ExperienceStore(db_path)
        m = store.run_metrics(run_id)
        store.close()

        row: dict[str, Any] = {
            "seed": s,
            "seq_digest": seq_digest(probe.hashes),
            "evals": len(probe.hashes),
            "wall_seconds": wall,
            "rows": m.get("evals_total"),
            "first_success_eval": m.get("first_success_eval"),
            "first_success_score": m.get("first_success_score"),
        }
        if use_cache:
            cs = dict(probe.stats)
            for k in cache_acc:
                cache_acc[k] += cs.get(k, 0)
            row["cache"] = cs
        per_seed.append(row)

    attempts = sum(r["evals"] for r in per_seed)
    out: dict[str, Any] = {
        "scenario": scenario_name,
        "seeds": seeds,
        "per_seed": per_seed,
        "attempts": attempts,
    }
    if use_cache:
        raw_calls = cache_acc["misses"] + cache_acc["bypasses"]
        out["cache"] = {
            **cache_acc,
            "raw_calls": raw_calls,
            "realized_savings": round(1 - raw_calls / attempts, 4) if attempts else 0.0,
            "mean_wall_seconds": round(
                sum(r["wall_seconds"] for r in per_seed) / len(per_seed), 4
            ),
        }
    else:
        out["mean_wall_seconds"] = round(
            sum(r["wall_seconds"] for r in per_seed) / len(per_seed), 4
        )
    return out


def capture_all(mode: str, out: Path) -> None:
    results = []
    with tempfile.TemporaryDirectory(prefix="ck_") as td:
        workdir = Path(td)
        for name in SCENARIO_REGISTRY:
            nocache = capture_wiring(name, SEEDS, workdir, use_cache=False)
            cached = capture_wiring(name, SEEDS, workdir, use_cache=True)
            results.append({"scenario": name, "nocache": nocache, "cached": cached})
    report = {
        "mode": mode,
        "protocol": {
            "seeds": SEEDS,
            "generations": GENERATIONS,
            "population": POPULATION,
            "prior": "disabled",
            "probe": "reuses scripts/probe_duplicate_evals.py primitives",
        },
        "results": results,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{mode}] wrote {out}")
    for r in results:
        c = r["cached"]["cache"]
        print(
            f"  {r['scenario']}: savings {c['realized_savings']:.1%} "
            f"(raw {c['raw_calls']}/{r['cached']['attempts']}) "
            f"wall/run cached {c['mean_wall_seconds']}s nocache {r['nocache']['mean_wall_seconds']}s"
        )


def compare(baseline_path: Path, after_path: Path, out: Path) -> int:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    aft = json.loads(after_path.read_text(encoding="utf-8"))
    verdict: dict[str, Any] = {"C1": {}, "C2": {}, "C3": {}, "C4": {}}

    c1_ok = True
    for rb, ra in zip(base["results"], aft["results"]):
        name = rb["scenario"]
        per_seed_ok = []
        for s, sn, sa in zip(SEEDS, rb["nocache"]["per_seed"], ra["nocache"]["per_seed"]):
            s_old = next(x for x in rb["cached"]["per_seed"] if x["seed"] == s)
            s_new = next(x for x in ra["cached"]["per_seed"] if x["seed"] == s)
            eq = (
                s_old["seq_digest"] == s_new["seq_digest"] == sa["seq_digest"]
                and s_old["rows"] == s_new["rows"] == sa["rows"]
                and s_old["first_success_eval"] == s_new["first_success_eval"] == sa["first_success_eval"]
                and s_old["first_success_score"] == s_new["first_success_score"] == sa["first_success_score"]
            )
            per_seed_ok.append(eq)
            if not eq:
                c1_ok = False
        verdict["C1"][name] = {
            "all_seeds_identical": all(per_seed_ok),
            "failing_seeds": [s for s, ok in zip(SEEDS, per_seed_ok) if not ok],
        }

    c2_rows = {}
    c2_ok = True
    for ra in aft["results"]:
        name = ra["scenario"]
        new_sav = ra["cached"]["cache"]["realized_savings"]
        old_sav = BASELINE_SAVINGS[name]
        ok = new_sav >= BASELINE_C2_FLOOR[name] and new_sav >= old_sav - C2_MAX_REGRESSION
        c2_ok = c2_ok and ok
        c2_rows[name] = {
            "old_realized": old_sav,
            "new_realized": new_sav,
            "floor_registered": BASELINE_C2_FLOOR[name],
            "ok": ok,
        }
    verdict["C2"] = {"per_scenario": c2_rows, "all_ok": c2_ok}

    c3_rows = {}
    c3_ok = True
    for rb, ra in zip(base["results"], aft["results"]):
        name = rb["scenario"]
        old_w = rb["cached"]["cache"]["mean_wall_seconds"]
        new_w = ra["cached"]["cache"]["mean_wall_seconds"]
        ratio = new_w / old_w if old_w else float("inf")
        ok = ratio <= 1 + C3_MAX_OVERHEAD
        c3_ok = c3_ok and ok
        c3_rows[name] = {"old_mean_wall": old_w, "new_mean_wall": new_w,
                         "ratio": round(ratio, 4), "ok": ok}
    verdict["C3"] = {"per_scenario": c3_rows, "all_ok": c3_ok}

    adopt = c1_ok and c2_ok and c3_ok
    verdict["C4"] = {
        "c1": c1_ok, "c2": c2_ok, "c3": c3_ok,
        "adopt_swap_as_default": adopt,
        "verdict": ("swap adopted — materialized-sources identity: byte-identical "
                    "trajectories, savings at registered floors, no overhead"
                    if adopt else
                    "REVERT — at least one registered rule failed"),
    }
    out.write_text(json.dumps({"baseline": str(baseline_path), "after": str(after_path),
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0 if adopt else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["baseline", "after", "compare"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default="reports/cache_key_baseline.json")
    ap.add_argument("--after", default="reports/cache_key_after.json")
    args = ap.parse_args()

    if args.mode == "compare":
        out = Path(args.out or "reports/cache_key_verdict.json")
        return compare(Path(args.baseline), Path(args.after), out)

    out = Path(args.out or (
        "reports/cache_key_baseline.json" if args.mode == "baseline"
        else "reports/cache_key_after.json"
    ))
    capture_all(args.mode, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
