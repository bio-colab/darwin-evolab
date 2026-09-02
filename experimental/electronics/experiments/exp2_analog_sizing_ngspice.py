"""
exp2_analog_sizing_ngspice.py — Laboratory Experiment 2: Comparative Analog Sizing.
Executes comparative ablation benchmark comparing Random Search, Standard GA, and
Darwin-EvoLab Engine with Speciation.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..benchmarks.comparative_runner import ComparativeSizingBenchmark


def run_comparative_analog_experiment() -> dict[str, Any]:
    """Runs comparative ablation on analog sizing and generates structured engineering summary."""
    t0 = time.perf_counter()
    bench = ComparativeSizingBenchmark(budget_evals=60, seed=42)
    results = bench.run_all()
    dur_sec = time.perf_counter() - t0

    summary = {
        "experiment": "Exp2_Comparative_Analog_Sizing_Benchmark",
        "benchmark_duration_sec": round(dur_sec, 3),
        "budget_evaluations_per_strategy": 60,
        "strategies": [
            {
                "strategy": r.strategy_name,
                "best_fitness": r.best_fitness,
                "generations": r.generations_run,
                "evaluations": r.evaluations_count,
                "target_met": r.target_met,
                "runtime_sec": r.duration_sec,
            }
            for r in results
        ],
    }
    return summary


if __name__ == "__main__":
    rep = run_comparative_analog_experiment()
    print(json.dumps(rep, indent=2))
