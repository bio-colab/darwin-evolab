#!/usr/bin/env python3
"""Duplicate-evaluation probe (M2): how many evaluator calls are spent on
programs that were already evaluated — within the same run and across runs
of the same problem?

Measurement only. No search behavior is modified; the probe wraps the raw
evaluator, hashes the materialized program of every evaluated individual
(ground truth: sha256 of ``RepairGenome.to_code()`` — the actual source the
evaluator tests) and never alters scores, holdout flags or RNG streams.

Why ground truth and not the store's (edit_kinds, edit_loci) proxy:
``RepairEdit`` carries a ``payload`` field, so two evaluations can share the
same kinds+loci while producing DIFFERENT programs. The probe therefore also
measures how well the proxy would work as a cache key (a wrong-keyed cache
returns wrong scores — forbidden), reporting proxy collisions explicitly.

Pre-registered thresholds (written BEFORE any measurement — the point of the
probe is to decide, not to rationalize):

  - An evaluation cache (M3) is built only if pooled within-run duplicate
    evaluations are >= 5% of all evaluations (0.95 Wilson lower bound >= 3%).
  - A persistent cross-run cache is considered only if cross-run source
    savings (evals saved by recalling another run's evaluated program) are
    also >= 5% of all evaluations.
  - The (edit_kinds, edit_loci) proxy is eligible as a cache key only if it
    produces ZERO collisions (distinct programs sharing one proxy key) in
    this probe; otherwise the cache must key on the program source hash.

Usage (from the repository root):
  PYTHONPATH=src python scripts/probe_duplicate_evals.py
  PYTHONPATH=src python scripts/probe_duplicate_evals.py --seeds 1-5 --out /tmp/probe.json
  PYTHONPATH=src python scripts/probe_duplicate_evals.py --cache   # verify realized savings with the M3 cache wired

With ``--cache`` the probe wraps the program-keyed EvaluationCache (M3):
probe(cache(raw)). The duplicate-rate numbers then become REALIZED savings:
cache misses are the raw invocations the engine could not avoid, hits are
the raw evaluations the cache eliminated — while scores and trajectories
stay identical (tests/test_eval_cache.py pins this contract).

Stdlib only. Deterministic given fixed --seeds: scenario bugs, populations
and search trajectories are all seed-controlled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab import EngineConfig, EvolutionEngine  # noqa: E402
from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population  # noqa: E402
from evolab.experience import (  # noqa: E402
    ExperienceStore,
    attach_eval_cache,
    attach_experience_recorder,
)

PROBE_SEEDS_DEFAULT = "1-10"


def clean_env() -> None:
    os.environ.pop("EVOLAB_EXPERIENCE", None)
    os.environ.pop("EVOLAB_EXPERIENCE_DB", None)


def program_hash(genome: Any) -> str:
    """Ground-truth identity of the evaluated program."""
    if hasattr(genome, "to_code"):
        return hashlib.sha256(genome.to_code().encode()).hexdigest()
    if hasattr(genome, "to_sources"):
        srcs = genome.to_sources()
        blob = json.dumps({k: srcs[k] for k in sorted(srcs)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()
    return hashlib.sha256(repr(genome).encode()).hexdigest()


class DuplicateProbe:
    """Wraps a raw evaluator; records sha256 of every evaluated program."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self.hashes: list[str] = []

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> Any:
        res = self.raw.evaluate(target, context) if context is not None else self.raw.evaluate(target)
        try:
            genome = target.genome if hasattr(target, "genome") else target
            self.hashes.append(program_hash(genome))
        except Exception:
            self.hashes.append("<probe_error>")
        return res

    def __call__(self, individual: Any) -> float:
        return float(self.evaluate(individual).score)

    @property
    def deterministic(self) -> bool:
        return getattr(self.raw, "deterministic", True)

    @property
    def cost_estimate(self) -> str:
        return getattr(self.raw, "cost_estimate", "cheap")

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.raw, attr)


def duplicate_stats(order: list[str]) -> dict[str, Any]:
    """Within-sequence duplicate accounting for one run's eval order."""
    seen: dict[str, int] = {}
    for h in order:
        seen[h] = seen.get(h, 0) + 1
    total = len(order)
    unique = len(seen)
    dup_evals = total - unique
    mults = sorted(seen.values(), reverse=True)
    return {
        "evals": total,
        "unique_programs": unique,
        "duplicate_evals": dup_evals,
        "duplicate_rate": round(dup_evals / total, 4) if total else 0.0,
        "max_multiplicity": mults[0] if mults else 0,
        "programs_seen_more_than_twice": sum(1 for m in mults if m > 2),
    }


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


def proxy_collision_count(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """How many DISTINCT programs share one (edit_kinds, edit_loci) proxy key?

    Returns (colliding_proxy_keys, distinct_programs_involved). Zero
    collisions means the proxy is a safe cache key; any collision forbids it.
    """
    by_proxy: dict[str, set[str]] = {}
    for row in rows:
        proxy = row["edit_kinds"] + "@" + row["edit_loci"]
        by_proxy.setdefault(proxy, set()).add(row["source_hash"])
    collisions = {k: v for k, v in by_proxy.items() if len(v) > 1}
    return len(collisions), sum(len(v) for v in collisions.values())


def probe_scenario(name: str, seeds: list[int], workdir: Path, generations: int, population: int, use_cache: bool = False) -> dict[str, Any]:
    clean_env()
    scenario = SCENARIO_REGISTRY[name]()
    db_path = workdir / f"probe_{name}.db"
    per_seed: list[dict[str, Any]] = []
    all_hashes: list[str] = []
    cache_acc = {"hits": 0, "misses": 0, "bypasses": 0}

    for s in seeds:
        inner = scenario.create_evaluator()
        if use_cache:
            inner = attach_eval_cache(inner)
        probe = DuplicateProbe(inner)
        run_id = f"probe_{name}_s{s}"
        wired = attach_experience_recorder(
            probe,
            scenario.sources,
            scenario.target_file,
            scenario.func_name,
            db_path=db_path,
            run_id=run_id,
            prior_enabled=False,  # pure baseline behavior, like v1's control arm
        )
        engine = EvolutionEngine(
            fitness_fn=wired,
            config=EngineConfig(
                generations=generations,
                population_size=population,
                elite_count=1,
                genome_size=2,
                seed=s,
            ),
        )
        initial = make_code_population(scenario, population, __import__("random").Random(s))
        engine.run(generations=generations, initial_population=initial)
        if use_cache:
            cs = dict(probe.stats)
            for k in cache_acc:
                cache_acc[k] += cs.get(k, 0)
        per_seed.append({"seed": s, **duplicate_stats(probe.hashes)})
        all_hashes.extend(probe.hashes)

    store = ExperienceStore(db_path)
    rows = store._conn.execute(
        "SELECT edit_kinds, edit_loci, run_id FROM experiences ORDER BY id"
    ).fetchall()
    store.close()
    # align store rows (insertion order) with probe hashes (evaluation order):
    # both observe every evaluate() call exactly once, in the same order.
    n_rows = len(rows)
    n_hash = len(all_hashes)
    aligned = n_rows == n_hash
    proxy_rows: list[dict[str, str]] = []
    if aligned:
        for (kinds, loci, _rid), h in zip(rows, all_hashes):
            proxy_rows.append({"edit_kinds": kinds, "edit_loci": loci, "source_hash": h})
    colliding_keys, colliding_programs = (
        proxy_collision_count(proxy_rows) if aligned else (-1, -1)
    )

    distinct_all = len(set(all_hashes))
    total = len(all_hashes)
    pooled_within = sum(p["duplicate_evals"] for p in per_seed)
    cross_saved = total - distinct_all
    out: dict[str, Any] = {
        "scenario": name,
        "seeds": seeds,
        "total_evals": total,
        "pooled_within_run_duplicate_evals": pooled_within,
        "within_run_duplicate_rate": round(pooled_within / total, 4) if total else 0.0,
        "within_run_rate_wilson_lb": round(wilson_lower(pooled_within, total), 4) if total else 0.0,
        "cross_run_saved_evals": cross_saved,
        "cross_run_savings_rate": round(cross_saved / total, 4) if total else 0.0,
        "distinct_programs_all_runs": distinct_all,
        "store_rows": n_rows,
        "probe_hashes": n_hash,
        "store_probe_aligned": aligned,
        "proxy_colliding_keys": colliding_keys,
        "proxy_colliding_programs": colliding_programs,
        "per_seed": per_seed,
    }
    if use_cache:
        out["cache_stats"] = {**cache_acc, "raw_evals_saved": cache_acc["hits"]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default=PROBE_SEEDS_DEFAULT, help="inclusive range, e.g. 1-10")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--scenarios", default=",".join(SCENARIO_REGISTRY))
    ap.add_argument("--out", default="reports/duplicate_evals_probe.json")
    ap.add_argument("--cache", action="store_true", help="wire the M3 EvaluationCache and report realized savings")
    args = ap.parse_args(argv)

    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="evolab_probe_") as td:
        for name in names:
            print(f"=== probe {name} (seeds {lo}-{hi}{', cache' if args.cache else ''}) ===", flush=True)
            res = probe_scenario(name, seeds, Path(td), args.generations, args.population, use_cache=args.cache)
            results.append(res)
            line = (
                f"  evals={res['total_evals']}  within-run dups={res['pooled_within_run_duplicate_evals']}"
                f" ({res['within_run_duplicate_rate']:.1%})  cross-run savings={res['cross_run_savings_rate']:.1%}"
                f"  aligned={res['store_probe_aligned']}  proxy_collisions={res['proxy_colliding_keys']}"
            )
            if args.cache and res.get("cache_stats"):
                cs = res["cache_stats"]
                line += (
                    f"\n  CACHE: raw_calls={cs['misses']}  saved={cs['hits']}"
                    f" ({cs['hits'] / max(1, cs['hits'] + cs['misses']):.1%} of raw work eliminated)"
                )
            print(line, flush=True)

    total = sum(r["total_evals"] for r in results)
    dups = sum(r["pooled_within_run_duplicate_evals"] for r in results)
    cross = sum(r["cross_run_saved_evals"] for r in results)
    collisions = sum(r["proxy_colliding_keys"] for r in results if r["proxy_colliding_keys"] >= 0)
    pooled_rate = dups / total if total else 0.0
    decision = {
        "thresholds_registered": {
            "build_eval_cache_if_within_run_rate_ge": 0.05,
            "build_eval_cache_if_wilson_lb_ge": 0.03,
            "consider_cross_run_cache_if_cross_rate_ge": 0.05,
            "proxy_key_allowed_only_if_zero_collisions": True,
        },
        "observed": {
            "total_evals": total,
            "within_run_duplicate_rate": round(pooled_rate, 4),
            "within_run_rate_wilson_lb": round(wilson_lower(dups, total), 4),
            "cross_run_savings_rate": round(cross / total, 4) if total else 0.0,
            "proxy_colliding_keys": collisions,
        },
        "build_eval_cache": bool(total and pooled_rate >= 0.05 and wilson_lower(dups, total) >= 0.03),
        "consider_cross_run_cache": bool(total and cross / total >= 0.05),
        "proxy_key_safe_for_cache": bool(collisions == 0),
    }
    report = {
        "protocol": {
            "seeds": seeds,
            "generations": args.generations,
            "population": args.population,
            "prior": "disabled (pure baseline behavior, v1 control-arm equivalent)",
            "identity": "sha256 of materialized program source (ground truth)",
        },
        "results": results,
        "decision": decision,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print(
        f"POOLED: {total} evals, within-run dups {dups} ({pooled_rate:.1%}), "
        f"cross-run savings {cross} ({cross / total:.1%}), proxy collisions {collisions}"
    )
    print(
        f"DECISION: build_eval_cache={decision['build_eval_cache']}  "
        f"consider_cross_run_cache={decision['consider_cross_run_cache']}  "
        f"proxy_key_safe={decision['proxy_key_safe_for_cache']}"
    )
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
