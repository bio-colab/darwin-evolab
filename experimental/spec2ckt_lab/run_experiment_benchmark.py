"""run_experiment_benchmark.py — Comprehensive Empirical Benchmark for Spec2Ckt Lab.

Runs head-to-head empirical evaluations across 4 standard CktBench specifications:
1. CktBench_LowPower
2. CktBench_Balanced
3. CktBench_HighGain
4. CktBench_HighSpeed

Compares:
- Cold-Start Darwin (Random Initial Population, No Spec Conditioning)
- Spec2Ckt Generative Darwin (Spec Conditioned Prior + ARCS Physical Grammar Guard + Latent Bandit)

Outputs detailed metrics table, convergence speedup, and PVT corner pass rates.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from evolab.silicon.sky130_pdk import Sky130Corner

from experimental.spec2ckt_lab.hybrid_evolution_engine import (
    HybridSpecEvolutionEngine,
    OptimizationResult,
)
from experimental.spec2ckt_lab.spec_types import BENCHMARK_SPECS, TargetCircuitSpec


def run_benchmarks():
    print("=" * 80)
    print("      DARWIN-EVOLAB : SPEC2CKT EXPERIMENTAL LABORATORY BENCHMARK")
    print("   Distilling ARCS (NYU) & CktGen (Engineering 2025) into Darwin OS")
    print("=" * 80)

    engine = HybridSpecEvolutionEngine(use_surrogate=True, seed=42)
    summary_results = []

    for name, spec in BENCHMARK_SPECS.items():
        print(f"\n[BENCHMARK] Target: {spec.name}")
        print(f"  Specs: Gain>={spec.gain_db}dB, GBW>={spec.gbw_mhz}MHz, PM>={spec.pm_deg}°, P<={spec.max_power_uw}uW, SR>={spec.min_slew_rate_v_us}V/us")

        # 1. Run Cold Start Baseline
        t0_cold = time.perf_counter()
        res_cold = engine.run_cold_start_baseline(spec, pop_size=20, max_generations=15)
        time_cold_ms = (time.perf_counter() - t0_cold) * 1000.0

        # 2. Run Spec2Ckt Generative Darwin
        t0_warm = time.perf_counter()
        res_warm = engine.run_spec2ckt_pipeline(
            spec, pop_size=20, max_generations=15, bandit_budget=12
        )
        time_warm_ms = (time.perf_counter() - t0_warm) * 1000.0

        # Calculate metrics
        cold_conv = res_cold.converged_generation if res_cold.converged_generation is not None else "> 15"
        warm_conv = res_warm.converged_generation if res_warm.converged_generation is not None else "> 15"

        sim_saving_pct = 0.0
        if res_cold.total_sim_calls > 0:
            sim_saving_pct = max(0.0, (res_cold.total_sim_calls - res_warm.total_sim_calls) / res_cold.total_sim_calls * 100.0)

        record = {
            "benchmark": spec.name,
            "cold_start": {
                "converged_gen": cold_conv,
                "all_satisfied": res_cold.all_specs_satisfied,
                "best_gain_db": res_cold.nominal_metrics.gain_db,
                "best_gbw_mhz": res_cold.nominal_metrics.gbw_mhz,
                "best_pm_deg": res_cold.nominal_metrics.pm_deg,
                "best_power_uw": res_cold.nominal_metrics.power_uw,
                "pvt_pass_rate": f"{res_cold.pvt_pass_rate * 100:.0f}%",
                "sim_calls": res_cold.total_sim_calls,
                "latency_ms": round(time_cold_ms, 1),
            },
            "spec2ckt_darwin": {
                "converged_gen": warm_conv,
                "all_satisfied": res_warm.all_specs_satisfied,
                "best_gain_db": res_warm.nominal_metrics.gain_db,
                "best_gbw_mhz": res_warm.nominal_metrics.gbw_mhz,
                "best_pm_deg": res_warm.nominal_metrics.pm_deg,
                "best_power_uw": res_warm.nominal_metrics.power_uw,
                "pvt_pass_rate": f"{res_warm.pvt_pass_rate * 100:.0f}%",
                "sim_calls": res_warm.total_sim_calls,
                "latency_ms": round(time_warm_ms, 1),
            },
            "speedup_sim_saving": f"{sim_saving_pct:.1f}%",
        }
        summary_results.append(record)

        print(f"  [Cold Start]      Conv Gen: {cold_conv:<5} | Satisfied: {res_cold.all_specs_satisfied!s:<5} | PVT Pass: {res_cold.pvt_pass_rate*100:.0f}% | Sims: {res_cold.total_sim_calls}")
        print(f"  [Spec2Ckt Darwin] Conv Gen: {warm_conv:<5} | Satisfied: {res_warm.all_specs_satisfied!s:<5} | PVT Pass: {res_warm.pvt_pass_rate*100:.0f}% | Sims: {res_warm.total_sim_calls}")

    # Print Summary Comparison Table
    print("\n" + "=" * 95)
    print(f"{'Benchmark Target':<22} | {'Cold Gen':<9} | {'Spec2Ckt Gen':<13} | {'Cold PVT':<9} | {'Spec2Ckt PVT':<13} | {'Sims Saved':<10}")
    print("-" * 95)
    for r in summary_results:
        print(
            f"{r['benchmark']:<22} | "
            f"{str(r['cold_start']['converged_gen']):<9} | "
            f"{str(r['spec2ckt_darwin']['converged_gen']):<13} | "
            f"{r['cold_start']['pvt_pass_rate']:<9} | "
            f"{r['spec2ckt_darwin']['pvt_pass_rate']:<13} | "
            f"{r['speedup_sim_saving']:<10}"
        )
    print("=" * 95)

    # Save benchmark report to local experimental directory
    out_file = Path("experimental/spec2ckt_lab/benchmark_report.json")
    out_file.write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Benchmark report saved locally to: {out_file.resolve()}\n")
    return summary_results


if __name__ == "__main__":
    run_benchmarks()
