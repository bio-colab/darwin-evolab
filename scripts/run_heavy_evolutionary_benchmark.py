"""scripts/run_heavy_evolutionary_benchmark.py — Heavy Parallel Darwinian APR Benchmark.

Executes high-throughput, multi-core Darwinian Evolutionary and Compositional Search
across the SWE-bench Lite 300 suite without LLMs. Exploits Kaggle multi-core CPUs
to verify all 300 industrial instances with deep evaluation budgets and multi-seed search.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure evolab package is in path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evolab.swe_bench import SWEBenchAdapter, SWEBenchResolution
from evolab.experience import wilson_interval


def _solve_worker(task: tuple[str, str, int, str]) -> dict[str, Any]:
    fixture_path_str, instance_id, max_evals, mode = task
    adapter = SWEBenchAdapter()
    p = Path(fixture_path_str)
    spec = adapter.parse_spec(p)
    res = adapter.solve_instance(spec, max_evals=max_evals, mode=mode)
    return {
        "instance_id": res.instance_id,
        "resolved": bool(res.resolved),
        "fail_to_pass_passed": bool(res.fail_to_pass_passed),
        "pass_to_pass_clean": bool(res.pass_to_pass_clean),
        "evaluations_used": int(res.evaluations_used),
        "execution_time_seconds": float(res.execution_time_seconds),
        "target_file": str(res.target_file),
        "generated_patch": str(res.generated_patch) if res.resolved else "",
    }


def run_heavy_benchmark(
    max_evals: int = 128,
    mode: str = "compositional",
    workers: int | None = None,
    output_path: str = "reports/swe_bench_heavy_breakthroughs.json",
    limit: int | None = None,
) -> dict[str, Any]:
    fixtures_dir = SRC_DIR / "evolab" / "fixtures" / "swe_bench_300"
    if not fixtures_dir.exists():
        raise FileNotFoundError(f"SWE-bench fixtures directory not found: {fixtures_dir}")

    fixture_files = sorted(fixtures_dir.glob("*.json"))
    if limit is not None and limit > 0:
        fixture_files = fixture_files[:limit]

    total_tasks = len(fixture_files)
    num_workers = workers or max(1, multiprocessing.cpu_count())

    print("=" * 78)
    print("DARWIN-EVOLAB HEAVY EVOLUTIONARY APR ENGINE (ZERO-LLM INDUSTRIAL BENCHMARK)")
    print("=" * 78)
    print(f"[*] Benchmark suite: SWE-bench Lite (Distilled Industrial Suite)")
    print(f"[*] Total target instances: {total_tasks}")
    print(f"[*] Search mode: {mode.upper()} (Deep Multi-Hunk & Compositional AST)")
    print(f"[*] Evaluation budget per instance (E_max): {max_evals}")
    print(f"[*] Parallel CPU workers: {num_workers}")
    print("=" * 78)

    tasks = [
        (str(p), p.stem, max_evals, mode)
        for p in fixture_files
    ]

    t0 = time.perf_counter()
    resolutions: list[dict[str, Any]] = []
    resolved_count = 0
    total_evals = 0

    # Execute in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_solve_worker, t): t for t in tasks}
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            res = fut.result()
            resolutions.append(res)
            total_evals += res["evaluations_used"]
            if res["resolved"]:
                resolved_count += 1

            if completed % 25 == 0 or completed == total_tasks:
                elapsed = time.perf_counter() - t0
                rate = (resolved_count / completed) * 100.0
                print(
                    f"[{completed:3d}/{total_tasks:3d}] "
                    f"Resolved: {resolved_count:3d} ({rate:5.2f}%) | "
                    f"Evals used: {total_evals:6d} | "
                    f"Time: {elapsed:6.2f}s"
                )

    total_time = time.perf_counter() - t0
    resolutions.sort(key=lambda x: x["instance_id"])

    # Calculate Wilson Score 95% Confidence Interval
    pass_rate = resolved_count / total_tasks if total_tasks > 0 else 0.0
    w_low, w_high = wilson_interval(resolved_count, total_tasks)

    report = {
        "benchmark_suite": "swe_bench_lite_heavy_evolutionary",
        "benchmark_version": "2.0.0-surgical-native",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_instances": total_tasks,
        "resolved_count": resolved_count,
        "pass_rate_percent": round(pass_rate * 100.0, 2),
        "wilson_95_ci": [round(w_low, 4), round(w_high, 4)],
        "max_evals_budget": max_evals,
        "search_mode": mode,
        "total_evaluations_consumed": total_evals,
        "total_runtime_seconds": round(total_time, 2),
        "parallel_workers": num_workers,
        "instances": resolutions,
    }

    out_file = ROOT_DIR / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Also update reports/swe_bench_lite_300.json
    lite_300_file = ROOT_DIR / "reports" / "swe_bench_lite_300.json"
    lite_300_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("FINAL BENCHMARK VERDICT:")
    print(f"  * Total Evaluated: {total_tasks}")
    print(f"  * Total Resolved:  {resolved_count} / {total_tasks} ({pass_rate * 100.0:.2f}%)")
    print(f"  * Wilson 95% CI:   [{w_low:.4f}, {w_high:.4f}]")
    print(f"  * Total Evals:     {total_evals} (avg {total_evals/total_tasks:.1f} evals/instance)")
    print(f"  * Total Time:      {total_time:.2f}s ({total_time/total_tasks*1000:.1f} ms/instance)")
    print(f"  * Report saved to: {out_file}")
    print("=" * 78)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heavy Parallel Evolutionary APR Benchmark")
    parser.add_argument("--max-evals", type=int, default=128, help="Max evaluations budget per issue")
    parser.add_argument("--mode", type=str, default="compositional", choices=["greedy", "compositional"])
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances to evaluate")
    parser.add_argument("--output", type=str, default="reports/swe_bench_heavy_breakthroughs.json")
    args = parser.parse_args()

    run_heavy_benchmark(
        max_evals=args.max_evals,
        mode=args.mode,
        workers=args.workers,
        output_path=args.output,
        limit=args.limit,
    )
