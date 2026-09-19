"""scripts/run_swe_bench_50.py — Runner for the Expanded SWE-bench Lite Benchmark Suite (N=50).

Evaluates Darwin-Evolab across 50 real-world industrial issue instances mined from production
GitHub repositories (Django, Flask, Requests, Urllib3, Pytest, Pydantic, Tornado, SymPy, etc.).
Outputs raw empirical telemetry to reports/swe_bench_lite_50.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "src"))

from evolab.swe_bench import SWEBenchAdapter


def run_swe_bench_50(
    fixtures_dir: Path | str | None = None,
    output_report_path: Path | str | None = None,
    max_evals: int = 32,
) -> dict:
    f_dir = Path(fixtures_dir) if fixtures_dir else root_dir / "src" / "evolab" / "fixtures" / "swe_bench_50"
    out_path = Path(output_report_path) if output_report_path else root_dir / "reports" / "swe_bench_lite_50.json"

    fixture_files = sorted(f_dir.glob("*.json"))
    if not fixture_files:
        raise FileNotFoundError(f"No SWE-bench fixture files found in {f_dir}")

    adapter = SWEBenchAdapter()
    results = []
    t_start = time.perf_counter()

    print("=" * 80)
    print(f" Darwin-Evolab: SWE-bench Lite Expanded Industrial Suite (N={len(fixture_files)})")
    print(f" Evaluation Budget: max_evals={max_evals} | Strategy: AST + Ochiai SBFL")
    print("=" * 80)
    print(f"{'Instance ID':<35} | {'Repo':<20} | {'Status':<10} | {'Evals':<6} | {'Time (s)':<8}")
    print("-" * 80)

    resolved_count = 0
    total_evals = 0

    for fixture_file in fixture_files:
        spec = adapter.parse_spec(fixture_file)
        resolution = adapter.solve_instance(spec, max_evals=max_evals)

        status_str = "RESOLVED" if resolution.resolved else "UNRESOLVED"
        if resolution.resolved:
            resolved_count += 1
        total_evals += resolution.evaluations_used

        patch_snippet = resolution.generated_patch[:300] if resolution.generated_patch else ""

        entry = {
            "instance_id": resolution.instance_id,
            "repo": spec.repo,
            "problem_statement": spec.problem_statement,
            "target_file": resolution.target_file,
            "resolved": resolution.resolved,
            "fail_to_pass_passed": resolution.fail_to_pass_passed,
            "pass_to_pass_clean": resolution.pass_to_pass_clean,
            "evaluations_used": resolution.evaluations_used,
            "execution_time_seconds": resolution.execution_time_seconds,
            "patch_preview": patch_snippet,
        }
        results.append(entry)
        print(
            f"{resolution.instance_id:<35} | {spec.repo:<20} | {status_str:<10} | "
            f"{resolution.evaluations_used:<6} | {resolution.execution_time_seconds:<8.3f}"
        )

    total_time = time.perf_counter() - t_start
    pass_rate_pct = (resolved_count / len(fixture_files)) * 100.0

    print("=" * 80)
    print(f" SUMMARY: {resolved_count}/{len(fixture_files)} resolved ({pass_rate_pct:.1f}% empirical pass rate)")
    print(f" Total Evaluations: {total_evals} | Total Execution Time: {total_time:.2f}s")
    print("=" * 80)

    report = {
        "benchmark_suite": "SWE-bench Lite Expanded Industrial Suite (N=50)",
        "benchmark_version": "2.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_instances": len(fixture_files),
        "resolved_count": resolved_count,
        "pass_rate_percent": round(pass_rate_pct, 1),
        "max_evals_budget": max_evals,
        "total_evaluations_consumed": total_evals,
        "total_runtime_seconds": round(total_time, 3),
        "instances": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[Artifact Saved] Raw empirical report written to: {out_path}")
    return report


if __name__ == "__main__":
    run_swe_bench_50()
