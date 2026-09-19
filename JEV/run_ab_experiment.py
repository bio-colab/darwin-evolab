"""run_ab_experiment.py — Rigorous A/B Benchmark: Darwin-Evolab Baseline vs JEV-Guided.

Strictly private and local to JEV/ (ignored by Git).
Compares search effort (evaluations consumed), wall-clock latency, token usage,
and solution correctness across core micro-benchmarks and real-world SWE-bench Lite issues.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from evolab.code_fixtures import (
    scenario_click_parser,
    scenario_requests_auth_url,
    scenario_lru_cache_logic,
    scenario_multi_file_config,
)
from evolab.repair import RepairGenome, catalog_sources, greedy_repair, _score
from evolab.swe_bench import SWEBenchAdapter
from JEV.jev_client import JevClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SWE_FIXTURES_DIR = REPO_ROOT / "src" / "evolab" / "fixtures" / "swe_bench"
RESULTS_FILE = Path(__file__).resolve().parent / "ab_experiment_results.json"


def run_jev_greedy_repair(
    sources: dict[str, str],
    target_file: str,
    evaluator: Any,
    client: JevClient,
    problem_statement: str = "",
    max_evals: int = 100,
) -> tuple[RepairGenome, int, float, list[dict[str, Any]], dict[str, Any]]:
    """Executes greedy repair with JEV System One operator routing at each stagnation/decision point."""
    t0 = time.perf_counter()
    catalog = catalog_sources(sources)
    current = RepairGenome(sources=dict(sources), target_file=target_file, edits=[])
    best_score, best_hold = _score(evaluator, current)
    evaluations = 1
    taken = set(current.edit_keys())
    history = [{"generation": 1, "best_fitness": best_score, "edits": 0}]
    gen = 1

    calls_before = client.total_calls
    tokens_in_before = client.total_input_tokens
    tokens_out_before = client.total_output_tokens
    api_time_before = client.total_api_time_seconds

    improved = True
    while improved and best_score < 100.0 and evaluations < max_evals:
        improved = False
        available_edits = [e for e in catalog if e.locus() not in taken]
        if not available_edits:
            break

        # 1. Extract failure context
        current_res = evaluator.evaluate(current)
        failures = current_res.artifacts.get("failures", [])
        failure_text = failures[0] if failures else "Target function failed test assertions"
        if problem_statement:
            failure_text = f"Issue: {problem_statement}\nTest failure: {failure_text}"

        current_code = current.to_code()
        available_kinds = sorted(list(set(e.kind for e in available_edits)))

        # 2. Query Jev for operator probabilities
        probs = client.prioritize_operators(current_code, failure_text, available_kinds)

        # 3. Sort candidates by JEV probability descending
        def _jev_sort_key(e):
            p = probs.get(e.kind, 0.0)
            return (-p, e.file, e.lineno, e.col_offset)

        sorted_edits = sorted(available_edits, key=_jev_sort_key)

        best_trial = None
        best_trial_score = best_score
        best_trial_hold = best_hold
        best_edit = None

        for edit in sorted_edits:
            trial = RepairGenome(
                sources=dict(sources),
                target_file=target_file,
                edits=current.edits + [edit],
            )
            score, hold = _score(evaluator, trial)
            evaluations += 1

            if score <= best_score:
                continue
            if hold is False and best_hold is True:
                continue

            if score > best_trial_score:
                best_trial = trial
                best_trial_score = score
                best_trial_hold = hold
                best_edit = edit
                # First-ascent greedy: commit immediately on improvement
                break

            if evaluations >= max_evals:
                break

        if best_trial is not None and best_edit is not None:
            current = best_trial
            best_score = best_trial_score
            best_hold = best_trial_hold
            taken.add(best_edit.locus())
            improved = True
            history.append({
                "generation": gen + 1,
                "best_fitness": best_score,
                "edits": len(current.edits),
                "added": best_edit.kind,
                "lineno": best_edit.lineno,
            })
            gen += 1

    duration = time.perf_counter() - t0
    telemetry = {
        "jev_calls": client.total_calls - calls_before,
        "jev_tokens_in": client.total_input_tokens - tokens_in_before,
        "jev_tokens_out": client.total_output_tokens - tokens_out_before,
        "jev_api_time_seconds": round(client.total_api_time_seconds - api_time_before, 3),
    }
    return current, evaluations, duration, history, telemetry


def evaluate_scenario(
    name: str,
    sources: dict[str, str],
    target_file: str,
    evaluator: Any,
    client: JevClient,
    problem_statement: str = "",
    max_evals: int = 150,
) -> dict[str, Any]:
    """Runs paired A/B evaluation on a single scenario."""
    print(f"\n>>> Running Scenario: {name}")

    # Arm A: Baseline
    t0_base = time.perf_counter()
    base_genome, base_hist, base_evals = greedy_repair(
        sources=dict(sources),
        target_file=target_file,
        evaluator=evaluator,
        max_evals=max_evals,
        prioritize_by_suspicion=True,
    )
    base_time = time.perf_counter() - t0_base
    base_res = evaluator.evaluate(base_genome)
    base_resolved = (base_res.score >= 99.9 and base_res.passed_holdout is not False)
    print(f"  [Arm A - Baseline] Evals: {base_evals:3d} | Resolved: {str(base_resolved):<5} | Score: {base_res.score:5.1f} | Time: {base_time*1000:6.1f}ms")

    # Arm B: JEV-Guided
    jev_genome, jev_evals, jev_time, jev_hist, jev_telemetry = run_jev_greedy_repair(
        sources=dict(sources),
        target_file=target_file,
        evaluator=evaluator,
        client=client,
        problem_statement=problem_statement,
        max_evals=max_evals,
    )
    jev_res = evaluator.evaluate(jev_genome)
    jev_resolved = (jev_res.score >= 99.9 and jev_res.passed_holdout is not False)
    evals_saved = base_evals - jev_evals
    evals_saved_pct = round((evals_saved / max(1, base_evals)) * 100.0, 2)

    print(f"  [Arm B - JEV-RSI ] Evals: {jev_evals:3d} | Resolved: {str(jev_resolved):<5} | Score: {jev_res.score:5.1f} | Time: {jev_time:6.2f}s (API: {jev_telemetry['jev_api_time_seconds']}s)")
    print(f"  ==> Result: Evals Saved: {evals_saved} ({evals_saved_pct:+6.2f}%) | JEV Calls: {jev_telemetry['jev_calls']} | Tokens: {jev_telemetry['jev_tokens_in']} in / {jev_telemetry['jev_tokens_out']} out")

    return {
        "scenario": name,
        "baseline": {
            "evaluations": base_evals,
            "resolved": base_resolved,
            "score": base_res.score,
            "time_seconds": round(base_time, 4),
            "edits_count": len(base_genome.edits),
        },
        "jev_guided": {
            "evaluations": jev_evals,
            "resolved": jev_resolved,
            "score": jev_res.score,
            "time_seconds": round(jev_time, 4),
            "edits_count": len(jev_genome.edits),
            "telemetry": jev_telemetry,
        },
        "comparison": {
            "evaluations_saved": evals_saved,
            "evaluations_saved_percent": evals_saved_pct,
            "resolved_preserved": (base_resolved == jev_resolved) if base_resolved else jev_resolved,
        },
    }


def main():
    print("=" * 80)
    print(" Darwin-Evolab: Controlled A/B Experiment — With vs Without JEV System One")
    print(" Target: Measure Search Effort Reduction (Evaluations Saved) & Guidance Accuracy")
    print("=" * 80)

    client = JevClient()
    results = []

    # 1. Core APR Micro-Benchmarks
    scenarios = [
        ("click_cli_parser", scenario_click_parser()),
        ("requests_http_helper", scenario_requests_auth_url()),
        ("lru_cache_logic", scenario_lru_cache_logic()),
        ("multi_file_config", scenario_multi_file_config()),
    ]

    for name, sc in scenarios:
        ev = sc.create_evaluator()
        res = evaluate_scenario(
            name=name,
            sources=sc.sources,
            target_file=sc.target_file,
            evaluator=ev,
            client=client,
            problem_statement=sc.description,
            max_evals=150,
        )
        results.append(res)

    # 2. SWE-bench Lite Real-World Benchmark Instances
    swe_adapter = SWEBenchAdapter()
    swe_fixtures = [
        "pallets__flask_4992.json",
        "pallets__jinja_1155.json",
        "pytest_dev__pytest_5227.json",
        "sympy__sympy_13480.json",
    ]

    for fname in swe_fixtures:
        fpath = SWE_FIXTURES_DIR / fname
        if not fpath.exists():
            continue
        spec = swe_adapter.parse_spec(fpath)
        ev = swe_adapter.build_evaluator(spec)
        res = evaluate_scenario(
            name=f"swe_bench::{spec.instance_id}",
            sources=spec.sources,
            target_file=spec.target_file,
            evaluator=ev,
            client=client,
            problem_statement=spec.problem_statement,
            max_evals=32,
        )
        results.append(res)

    # Summary Statistics
    total_base_evals = sum(r["baseline"]["evaluations"] for r in results)
    total_jev_evals = sum(r["jev_guided"]["evaluations"] for r in results)
    total_saved_evals = total_base_evals - total_jev_evals
    overall_saved_pct = round((total_saved_evals / max(1, total_base_evals)) * 100.0, 2)

    base_resolved_count = sum(1 for r in results if r["baseline"]["resolved"])
    jev_resolved_count = sum(1 for r in results if r["jev_guided"]["resolved"])

    total_tokens_in = client.total_input_tokens
    total_tokens_out = client.total_output_tokens
    total_api_time = round(client.total_api_time_seconds, 2)

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_scenarios_evaluated": len(results),
        "total_baseline_evaluations": total_base_evals,
        "total_jev_evaluations": total_jev_evals,
        "total_evaluations_saved": total_saved_evals,
        "overall_evaluations_saved_percent": overall_saved_pct,
        "baseline_resolved_count": base_resolved_count,
        "jev_resolved_count": jev_resolved_count,
        "total_jev_calls": client.total_calls,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_api_time_seconds": total_api_time,
        "per_scenario_results": results,
    }

    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(" A/B BENCHMARK SUMMARY & EMPIRICAL TELEMETRY")
    print("=" * 80)
    print(f"{'Scenario / Benchmark':<35} | {'Base Evals':<10} | {'JEV Evals':<10} | {'Saved (%)':<12} | {'Base Res':<8} | {'JEV Res':<8}")
    print("-" * 80)
    for r in results:
        s_name = r["scenario"]
        b_ev = r["baseline"]["evaluations"]
        j_ev = r["jev_guided"]["evaluations"]
        pct = r["comparison"]["evaluations_saved_percent"]
        b_res = "PASS" if r["baseline"]["resolved"] else "FAIL"
        j_res = "PASS" if r["jev_guided"]["resolved"] else "FAIL"
        print(f"{s_name:<35} | {b_ev:<10} | {j_ev:<10} | {pct:+10.2f}% | {b_res:<8} | {j_res:<8}")

    print("-" * 80)
    print(f"{'TOTAL / OVERALL':<35} | {total_base_evals:<10} | {total_jev_evals:<10} | {overall_saved_pct:+10.2f}% | {base_resolved_count}/{len(results)}   | {jev_resolved_count}/{len(results)}")
    print("=" * 80)
    print(f"JEV System One Cumulative Telemetry:")
    print(f"  - Total API Calls:     {client.total_calls}")
    print(f"  - Total Input Tokens:  {total_tokens_in:,}")
    print(f"  - Total Output Tokens: {total_tokens_out:,}")
    print(f"  - Cumulative API Time: {total_api_time}s")
    print(f"  - Output Report:       {RESULTS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
