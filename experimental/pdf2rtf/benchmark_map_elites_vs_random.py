"""benchmark_map_elites_vs_random.py — Causal Attribution Benchmark: MAP-Elites vs. Random Search.

Preregistered scientific protocol comparing MAP-Elites Quality-Diversity evolution against
Uniform Random Search under an identical evaluation budget (B=5,000 candidate evaluations)
across K independent random seeds on a 10x10 behavioral grid (100 niches).

Measures:
1. Archive Coverage (% of niches occupied)
2. QD-Score (sum of niche elite fitnesses)
3. Max Fitness (champion discovery)
4. Mean Elite Fitness (QD-Score / occupied niches)
5. Illumination Trajectory over budget consumption
6. Paired t-test, p-values, and Cohen's d effect sizes
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Tuple

from evolab.engine import EvolutionEngine
from evolab.genome import Individual

from .calibrate import MultiDocumentCorpusEvaluator
from .corpus import create_golden_corpus
from .genome import PARAM_BOUNDS, ProfileGenome


def desc_cluster_density(genome: ProfileGenome) -> float:
    """Descriptor 1: Cluster density (0.0 to 1.0)."""
    return float(genome.describe()["cluster_density"])


def desc_table_sensitivity(genome: ProfileGenome) -> float:
    """Descriptor 2: Table sensitivity (0.0 to 1.0)."""
    return float(genome.describe()["table_sensitivity"])


def compute_cell_coordinate(
    d1: float,
    d2: float,
    grid_x: int = 10,
    grid_y: int = 10,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[int, int]:
    """Projects continuous 2D descriptors into discrete grid coordinates [0..grid-1]."""
    cx = min(grid_x - 1, max(0, int(d1 / scale_x * grid_x)))
    cy = min(grid_y - 1, max(0, int(d2 / scale_y * grid_y)))
    return (cx, cy)


def sample_uniform_genome(rng: random.Random) -> ProfileGenome:
    """Samples a genome uniformly from PARAM_BOUNDS across 10 dimensions."""
    return ProfileGenome(
        space_gap_ratio=rng.uniform(*PARAM_BOUNDS["space_gap_ratio"]),
        para_split_delta_ratio=rng.uniform(*PARAM_BOUNDS["para_split_delta_ratio"]),
        align_tolerance_pt=rng.uniform(*PARAM_BOUNDS["align_tolerance_pt"]),
        line_spacing_round_pt=rng.uniform(*PARAM_BOUNDS["line_spacing_round_pt"]),
        table_col_align_tol_pt=rng.uniform(*PARAM_BOUNDS["table_col_align_tol_pt"]),
        table_min_rows=int(round(rng.uniform(*PARAM_BOUNDS["table_min_rows"]))),
        table_min_cols=int(round(rng.uniform(*PARAM_BOUNDS["table_min_cols"]))),
        para_split_short_line_factor=rng.uniform(*PARAM_BOUNDS["para_split_short_line_factor"]),
        para_split_indent_factor=rng.uniform(*PARAM_BOUNDS["para_split_indent_factor"]),
        para_split_font_weight=rng.uniform(*PARAM_BOUNDS["para_split_font_weight"]),
    )


@dataclass
class ArchiveSummary:
    occupied_niches: int
    total_niches: int
    coverage_pct: float
    qd_score: float
    mean_fitness: float
    max_fitness: float
    archive: Dict[str, float]  # cell_key -> fitness


def evaluate_archive_state(
    archive: Dict[Tuple[int, int], float],
    total_niches: int = 100,
) -> ArchiveSummary:
    """Computes standard Quality-Diversity summary metrics from an archive dictionary."""
    occupied = len(archive)
    qd_score = sum(archive.values())
    mean_fit = (qd_score / occupied) if occupied > 0 else 0.0
    max_fit = max(archive.values()) if occupied > 0 else 0.0
    coverage = (occupied / total_niches) * 100.0
    serializable_archive = {f"{k[0]},{k[1]}": round(v, 4) for k, v in archive.items()}
    return ArchiveSummary(
        occupied_niches=occupied,
        total_niches=total_niches,
        coverage_pct=round(coverage, 2),
        qd_score=round(qd_score, 4),
        mean_fitness=round(mean_fit, 4),
        max_fitness=round(max_fit, 4),
        archive=serializable_archive,
    )


def run_random_search_run(
    seed: int,
    budget: int,
    evaluator: MultiDocumentCorpusEvaluator,
    grid_x: int = 10,
    grid_y: int = 10,
    snapshot_intervals: List[int] | None = None,
) -> Dict[str, Any]:
    """Executes a budget-matched Uniform Random Search baseline."""
    rng = random.Random(seed)
    archive: Dict[Tuple[int, int], float] = {}
    snapshots: Dict[str, Any] = {}
    snapshot_set = set(snapshot_intervals or [500, 1000, 2000, 3000, 4000, 5000])

    t_start = time.perf_counter()
    for eval_i in range(1, budget + 1):
        g = sample_uniform_genome(rng)
        ind = Individual(genome=g, species="spec_pdf_profile")
        eval_res = evaluator.evaluate(ind)
        fitness = float(eval_res.score if hasattr(eval_res, "score") else eval_res)

        d1 = desc_cluster_density(g)
        d2 = desc_table_sensitivity(g)
        cell = compute_cell_coordinate(d1, d2, grid_x, grid_y)

        if cell not in archive or fitness > archive[cell]:
            archive[cell] = fitness

        if eval_i in snapshot_set or eval_i == budget:
            s = evaluate_archive_state(archive, grid_x * grid_y)
            snapshots[str(eval_i)] = {
                "evaluations": eval_i,
                "occupied_niches": s.occupied_niches,
                "coverage_pct": s.coverage_pct,
                "qd_score": s.qd_score,
                "max_fitness": s.max_fitness,
                "mean_fitness": s.mean_fitness,
            }

    elapsed = time.perf_counter() - t_start
    final_summary = evaluate_archive_state(archive, grid_x * grid_y)

    return {
        "algorithm": "random_search",
        "seed": seed,
        "budget": budget,
        "elapsed_seconds": round(elapsed, 2),
        "evaluations_per_second": round(budget / max(0.001, elapsed), 2),
        "final_metrics": {
            "occupied_niches": final_summary.occupied_niches,
            "coverage_pct": final_summary.coverage_pct,
            "qd_score": final_summary.qd_score,
            "mean_fitness": final_summary.mean_fitness,
            "max_fitness": final_summary.max_fitness,
        },
        "snapshots": snapshots,
    }


def run_map_elites_run(
    seed: int,
    budget: int,
    evaluator: MultiDocumentCorpusEvaluator,
    grid_x: int = 10,
    grid_y: int = 10,
    pop_size: int = 100,
    snapshot_intervals: List[int] | None = None,
) -> Dict[str, Any]:
    """Executes a budget-matched MAP-Elites Evolutionary illumination run."""
    rng = random.Random(seed)
    generations = budget // pop_size
    snapshot_set = set(snapshot_intervals or [500, 1000, 2000, 3000, 4000, 5000])

    pop: List[Individual] = [Individual(genome=ProfileGenome(), species="spec_pdf_profile")]
    for _ in range(pop_size - 1):
        g = sample_uniform_genome(rng)
        pop.append(Individual(genome=g, species="spec_pdf_profile"))

    engine = EvolutionEngine(
        evaluator=evaluator,
        population_size=pop_size,
        generations=generations,
        mutation_rate=0.45,
        me_grid_x=grid_x,
        me_grid_y=grid_y,
        me_scale_x=1.0,
        me_scale_y=1.0,
        descriptors=[desc_cluster_density, desc_table_sensitivity],
        qd_selection=True,
        record_archive_solutions=True,
        seed=seed,
    )

    snapshots: Dict[str, Any] = {}

    def on_generation_evaluated(evt: Any) -> None:
        current_evals = evt.generation * pop_size
        archive = {coord: ind.fitness for coord, ind in engine._archive.items()}
        if current_evals in snapshot_set or evt.generation == generations:
            s = evaluate_archive_state(archive, grid_x * grid_y)
            snapshots[str(current_evals)] = {
                "evaluations": current_evals,
                "occupied_niches": s.occupied_niches,
                "coverage_pct": s.coverage_pct,
                "qd_score": s.qd_score,
                "max_fitness": s.max_fitness,
                "mean_fitness": s.mean_fitness,
            }

    from evolab.events import GenerationEvaluatedEvent
    engine.add_event_listener(GenerationEvaluatedEvent, on_generation_evaluated)

    t_start = time.perf_counter()
    engine.run(generations=generations, initial_population=pop)
    elapsed = time.perf_counter() - t_start

    final_archive = {coord: ind.fitness for coord, ind in engine._archive.items()}
    final_summary = evaluate_archive_state(final_archive, grid_x * grid_y)

    return {
        "algorithm": "map_elites",
        "seed": seed,
        "budget": budget,
        "generations": generations,
        "population_size": pop_size,
        "elapsed_seconds": round(elapsed, 2),
        "evaluations_per_second": round(budget / max(0.001, elapsed), 2),
        "final_metrics": {
            "occupied_niches": final_summary.occupied_niches,
            "coverage_pct": final_summary.coverage_pct,
            "qd_score": final_summary.qd_score,
            "mean_fitness": final_summary.mean_fitness,
            "max_fitness": final_summary.max_fitness,
        },
        "snapshots": snapshots,
    }


def compute_paired_statistics(
    me_vals: List[float],
    rs_vals: List[float],
) -> Dict[str, float]:
    """Computes paired t-test t-statistic, p-value, mean difference, and Cohen's d."""
    n = len(me_vals)
    if n < 2:
        return {"t_stat": 0.0, "p_val": 1.0, "mean_diff": 0.0, "cohens_d": 0.0}

    diffs = [m - r for m, r in zip(me_vals, rs_vals)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(var_diff) if var_diff > 0 else 1e-9

    se_diff = std_diff / math.sqrt(n)
    t_stat = mean_diff / se_diff

    # Standard approximation of 2-tailed p-value
    df = n - 1
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2))))

    cohens_d = mean_diff / std_diff if std_diff > 1e-8 else 0.0

    return {
        "mean_diff": round(mean_diff, 4),
        "t_stat": round(t_stat, 4),
        "p_val": round(max(0.0, min(1.0, p_val)), 6),
        "cohens_d": round(cohens_d, 4),
    }


def load_benchmark_corpus(corpus_name: str = "golden") -> List[Any]:
    """Loads benchmark evaluation corpus by identifier."""
    if corpus_name in ("dilemma", "dilemma_2"):
        from .word_dilemma import load_word_dilemma_corpus
        return load_word_dilemma_corpus()[:2]
    elif corpus_name == "dilemma_full":
        from .word_dilemma import load_word_dilemma_corpus
        return load_word_dilemma_corpus()
    else:
        return create_golden_corpus()[:2]


def run_comparative_benchmark(
    budget: int = 5000,
    seeds: List[int] | None = None,
    pop_size: int = 100,
    grid_x: int = 10,
    grid_y: int = 10,
    parallel: bool = True,
    corpus_name: str = "golden",
    save_report_path: Path | str | None = None,
) -> Dict[str, Any]:
    """Executes the full preregistered paired benchmark across all seeds."""
    if seeds is None:
        seeds = [42, 137, 256, 512, 1024]

    corpus = load_benchmark_corpus(corpus_name)

    print("=================================================================")
    print(" Causal Attribution: MAP-Elites Evolution vs. Random Search")
    print(f" Corpus Target:  {corpus_name} ({len(corpus)} documents)")
    print(f" Budget per Run: {budget} evaluations")
    print(f" Seeds ({len(seeds)}): {seeds}")
    print(f" Grid Niches:   {grid_x} x {grid_y} ({grid_x * grid_y} niches)")
    print(f" Parallel Mode: {parallel}")
    print("=================================================================\n")

    me_results: List[Dict[str, Any]] = []
    rs_results: List[Dict[str, Any]] = []

    if parallel and len(seeds) > 1:
        import concurrent.futures

        print(f"[Phase 1/2] Executing MAP-Elites across {len(seeds)} seeds in parallel (B={budget})...")
        t_me = time.perf_counter()

        def _task_me(s: int) -> Dict[str, Any]:
            ev = MultiDocumentCorpusEvaluator(corpus)
            res = run_map_elites_run(
                seed=s,
                budget=budget,
                evaluator=ev,
                grid_x=grid_x,
                grid_y=grid_y,
                pop_size=pop_size,
            )
            print(f"  [MAP-Elites] Seed {s} finished in {res['elapsed_seconds']}s -> Occupied: {res['final_metrics']['occupied_niches']}/100, QD-Score: {res['final_metrics']['qd_score']:.2f}")
            return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(seeds), 5)) as executor:
            me_results = list(executor.map(_task_me, seeds))
        print(f"  --> MAP-Elites Phase Complete in {time.perf_counter() - t_me:.2f}s\n")

        print(f"[Phase 2/2] Executing Uniform Random Search across {len(seeds)} seeds in parallel (B={budget})...")
        t_rs = time.perf_counter()

        def _task_rs(s: int) -> Dict[str, Any]:
            ev = MultiDocumentCorpusEvaluator(corpus)
            res = run_random_search_run(
                seed=s,
                budget=budget,
                evaluator=ev,
                grid_x=grid_x,
                grid_y=grid_y,
            )
            print(f"  [Random Search] Seed {s} finished in {res['elapsed_seconds']}s -> Occupied: {res['final_metrics']['occupied_niches']}/100, QD-Score: {res['final_metrics']['qd_score']:.2f}")
            return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(seeds), 5)) as executor:
            rs_results = list(executor.map(_task_rs, seeds))
        print(f"  --> Random Search Phase Complete in {time.perf_counter() - t_rs:.2f}s\n")

    else:
        evaluator = MultiDocumentCorpusEvaluator(corpus)
        for s in seeds:
            print(f"--- Running Seed {s} ---")
            print(f"  [1/2] Executing MAP-Elites Evolution (B={budget})...")
            me_res = run_map_elites_run(
                seed=s,
                budget=budget,
                evaluator=evaluator,
                grid_x=grid_x,
                grid_y=grid_y,
                pop_size=pop_size,
            )
            me_results.append(me_res)
            print(f"        Done in {me_res['elapsed_seconds']}s -> Occupied: {me_res['final_metrics']['occupied_niches']}/100, QD-Score: {me_res['final_metrics']['qd_score']:.2f}")

            print(f"  [2/2] Executing Uniform Random Search (B={budget})...")
            rs_res = run_random_search_run(
                seed=s,
                budget=budget,
                evaluator=evaluator,
                grid_x=grid_x,
                grid_y=grid_y,
            )
            rs_results.append(rs_res)
            print(f"        Done in {rs_res['elapsed_seconds']}s -> Occupied: {rs_res['final_metrics']['occupied_niches']}/100, QD-Score: {rs_res['final_metrics']['qd_score']:.2f}\n")

    # Aggregate Statistics
    me_cov = [r["final_metrics"]["coverage_pct"] for r in me_results]
    rs_cov = [r["final_metrics"]["coverage_pct"] for r in rs_results]

    me_qd = [r["final_metrics"]["qd_score"] for r in me_results]
    rs_qd = [r["final_metrics"]["qd_score"] for r in rs_results]

    me_max = [r["final_metrics"]["max_fitness"] for r in me_results]
    rs_max = [r["final_metrics"]["max_fitness"] for r in rs_results]

    me_mean = [r["final_metrics"]["mean_fitness"] for r in me_results]
    rs_mean = [r["final_metrics"]["mean_fitness"] for r in rs_results]

    stats_cov = compute_paired_statistics(me_cov, rs_cov)
    stats_qd = compute_paired_statistics(me_qd, rs_qd)
    stats_max = compute_paired_statistics(me_max, rs_max)

    def calc_mean_std(vals: List[float]) -> Tuple[float, float]:
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
        return round(m, 2), round(math.sqrt(var), 2)

    me_cov_m, me_cov_s = calc_mean_std(me_cov)
    rs_cov_m, rs_cov_s = calc_mean_std(rs_cov)

    me_qd_m, me_qd_s = calc_mean_std(me_qd)
    rs_qd_m, rs_qd_s = calc_mean_std(rs_qd)

    me_max_m, me_max_s = calc_mean_std(me_max)
    rs_max_m, rs_max_s = calc_mean_std(rs_max)

    me_mean_m, me_mean_s = calc_mean_std(me_mean)
    rs_mean_m, rs_mean_s = calc_mean_std(rs_mean)

    all_snapshot_keys = sorted(me_results[0]["snapshots"].keys(), key=lambda k: int(k))
    trajectory: List[Dict[str, Any]] = []
    for k in all_snapshot_keys:
        me_cov_step = sum(r["snapshots"][k]["coverage_pct"] for r in me_results) / len(seeds)
        rs_cov_step = sum(r["snapshots"][k]["coverage_pct"] for r in rs_results) / len(seeds)
        me_qd_step = sum(r["snapshots"][k]["qd_score"] for r in me_results) / len(seeds)
        rs_qd_step = sum(r["snapshots"][k]["qd_score"] for r in rs_results) / len(seeds)
        trajectory.append({
            "evaluations": int(k),
            "map_elites_coverage_mean": round(me_cov_step, 2),
            "random_search_coverage_mean": round(rs_cov_step, 2),
            "coverage_advantage": round(me_cov_step - rs_cov_step, 2),
            "map_elites_qd_score_mean": round(me_qd_step, 2),
            "random_search_qd_score_mean": round(rs_qd_step, 2),
            "qd_score_advantage": round(me_qd_step - rs_qd_step, 2),
        })

    report = {
        "title": "Causal Attribution Benchmark: MAP-Elites Evolution vs. Uniform Random Search",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preregistered_protocol": {
            "budget_evaluations": budget,
            "seeds": seeds,
            "num_seeds": len(seeds),
            "corpus_target": corpus_name,
            "grid_dimensions": [grid_x, grid_y],
            "total_niches": grid_x * grid_y,
            "population_size": pop_size,
            "evaluator_description": f"Deterministic MultiDocumentCorpusEvaluator ({corpus_name})",
        },
        "comparative_summary": {
            "coverage_pct": {
                "map_elites": {"mean": me_cov_m, "std": me_cov_s},
                "random_search": {"mean": rs_cov_m, "std": rs_cov_s},
                "difference": round(me_cov_m - rs_cov_m, 2),
                "paired_test": stats_cov,
                "causal_significance": bool(stats_cov["p_val"] < 0.05 and stats_cov["mean_diff"] > 0),
            },
            "qd_score": {
                "map_elites": {"mean": me_qd_m, "std": me_qd_s},
                "random_search": {"mean": rs_qd_m, "std": rs_qd_s},
                "difference": round(me_qd_m - rs_qd_m, 2),
                "paired_test": stats_qd,
                "causal_significance": bool(stats_qd["p_val"] < 0.05 and stats_qd["mean_diff"] > 0),
            },
            "max_fitness": {
                "map_elites": {"mean": me_max_m, "std": me_max_s},
                "random_search": {"mean": rs_max_m, "std": rs_max_s},
                "difference": round(me_max_m - rs_max_m, 2),
                "paired_test": stats_max,
            },
            "mean_elite_fitness": {
                "map_elites": {"mean": me_mean_m, "std": me_mean_s},
                "random_search": {"mean": rs_mean_m, "std": rs_mean_s},
                "difference": round(me_mean_m - rs_mean_m, 2),
            },
        },
        "illumination_trajectory": trajectory,
        "seed_level_runs": {
            "map_elites": me_results,
            "random_search": rs_results,
        },
    }

    if save_report_path:
        out_path = Path(save_report_path)
    else:
        filename = (
            "pdf2rtf_map_elites_vs_random_search_dilemma.json"
            if corpus_name in ("dilemma", "dilemma_2", "dilemma_full")
            else "pdf2rtf_map_elites_vs_random_search.json"
        )
        out_path = Path(__file__).resolve().parent.parent.parent / "reports" / filename

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=================================================================")
    print(" BENCHMARK COMPLETED")
    print(f" Target:    {corpus_name}")
    print(f" Coverage:  MAP-Elites {me_cov_m}% ± {me_cov_s}% vs. Random {rs_cov_m}% ± {rs_cov_s}% (Diff: {me_cov_m - rs_cov_m:+.2f}%, p={stats_cov['p_val']})")
    print(f" QD-Score:  MAP-Elites {me_qd_m} ± {me_qd_s} vs. Random {rs_qd_m} ± {rs_qd_s} (Diff: {me_qd_m - rs_qd_m:+.2f}, p={stats_qd['p_val']})")
    print(f" Max Fit:   MAP-Elites {me_max_m}% vs. Random {rs_max_m}%")
    print(f" Report saved: {out_path}")
    print("=================================================================\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAP-Elites vs. Random Search Causal Benchmark.")
    parser.add_argument("--budget", type=int, default=5000, help="Evaluation budget per run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 256, 512, 1024], help="List of random seeds.")
    parser.add_argument("--corpus", type=str, default="golden", choices=["golden", "dilemma", "dilemma_full"], help="Corpus target.")
    parser.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel execution.")
    parser.add_argument("--out", type=str, default=None, help="Output report JSON path.")
    args = parser.parse_args()

    run_comparative_benchmark(
        budget=args.budget,
        seeds=args.seeds,
        corpus_name=args.corpus,
        parallel=args.parallel,
        save_report_path=args.out,
    )
