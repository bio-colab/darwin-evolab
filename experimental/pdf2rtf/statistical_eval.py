"""statistical_eval.py — Multi-Seed Statistical Significance Evaluation.

Executes K independent evolutionary calibration runs across distinct random seeds,
evaluates each evolved champion against the unseen 12-document genuine Word holdout corpus,
and computes rigorous statistical metrics: Mean, Standard Deviation, Standard Error,
and 95% Confidence Intervals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Any

from evolab.engine import EvolutionEngine
from evolab.genome import Individual

from .benchmark import run_golden_benchmark
from .calibrate import MultiDocumentCorpusEvaluator
from .corpus import create_synthetic_dev_corpus
from .genome import PARAM_BOUNDS, ProfileGenome, ProfilePolicy
from .word_holdout import load_real_word_holdout


def evaluate_policy_on_corpus(policy: ProfilePolicy, corpus) -> dict[str, Any]:
    """Evaluates a single policy against a benchmark corpus."""
    rep = run_golden_benchmark(policy=policy, corpus=corpus)
    return {
        "composite_score": rep.average_composite_score,
        "text_integrity_pass_rate": rep.text_integrity_pass_rate,
        "overall_pass_rate": rep.overall_pass_rate,
        "all_passed": (rep.overall_pass_rate == 1.0),
    }


def run_multiseed_evaluation(
    num_seeds: int = 20,
    population_size: int = 12,
    generations: int = 4,
    save_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Runs evolutionary calibration over multiple independent seeds and records holdout performance."""
    dev_corpus = create_synthetic_dev_corpus()
    holdout_corpus = load_real_word_holdout()

    evaluator = MultiDocumentCorpusEvaluator(dev_corpus)

    seed_records: list[dict[str, Any]] = []
    holdout_scores: list[float] = []
    dev_scores: list[float] = []

    print(f"=== Running Multi-Seed Evaluation (K={num_seeds} seeds) ===")

    for s_idx in range(1, num_seeds + 1):
        t0 = time.perf_counter()
        rng = random.Random(s_idx)

        # Seed initial population
        pop: list[Individual] = [
            Individual(genome=ProfileGenome(), species="spec_pdf_profile")
        ]
        for _ in range(population_size - 1):
            g = ProfileGenome(
                space_gap_ratio=rng.uniform(*PARAM_BOUNDS["space_gap_ratio"]),
                para_split_delta_ratio=rng.uniform(*PARAM_BOUNDS["para_split_delta_ratio"]),
                align_tolerance_pt=rng.uniform(*PARAM_BOUNDS["align_tolerance_pt"]),
                line_spacing_round_pt=rng.uniform(*PARAM_BOUNDS["line_spacing_round_pt"]),
                table_col_align_tol_pt=rng.uniform(*PARAM_BOUNDS["table_col_align_tol_pt"]),
                table_min_rows=int(round(rng.uniform(*PARAM_BOUNDS["table_min_rows"]))),
                table_min_cols=int(round(rng.uniform(*PARAM_BOUNDS["table_min_cols"]))),
            )
            pop.append(Individual(genome=g, species="spec_pdf_profile"))

        # MAP-Elites descriptors
        desc_x = lambda g: g.describe()["cluster_density"]
        desc_y = lambda g: g.describe()["table_sensitivity"]

        engine = EvolutionEngine(
            evaluator=evaluator,
            population_size=population_size,
            generations=generations,
            mutation_rate=0.45,
            me_grid_x=4,
            me_grid_y=4,
            descriptors=[desc_x, desc_y],
            seed=s_idx,
        )

        engine.run(generations=generations, initial_population=pop)
        best_ind: Individual = pop[0]
        best_fitness = -1.0
        for cell_ind in engine._archive.values():
            if cell_ind.fitness > best_fitness:
                best_fitness = cell_ind.fitness
                best_ind = cell_ind
        champ_genome: ProfileGenome = getattr(best_ind, "genome", best_ind)
        champ_policy = champ_genome.to_policy()

        # Evaluate champion on training dev set
        dev_eval = evaluate_policy_on_corpus(champ_policy, dev_corpus)

        # Evaluate champion on UNSEEN real Word holdout set (N=12)
        holdout_eval = evaluate_policy_on_corpus(champ_policy, holdout_corpus)

        elapsed = time.perf_counter() - t0

        record = {
            "seed": s_idx,
            "dev_composite_score": round(dev_eval["composite_score"], 4),
            "holdout_composite_score": round(holdout_eval["composite_score"], 4),
            "text_integrity_pass_rate": round(holdout_eval["text_integrity_pass_rate"], 4),
            "holdout_all_passed": holdout_eval["all_passed"],
            "parameters": champ_policy.to_dict(),
            "elapsed_sec": round(elapsed, 2),
        }
        seed_records.append(record)
        dev_scores.append(dev_eval["composite_score"])
        holdout_scores.append(holdout_eval["composite_score"])

        print(
            f"  Seed {s_idx:02d}: Dev={dev_eval['composite_score']:.2%}, "
            f"Holdout={holdout_eval['composite_score']:.2%}, "
            f"Pass={holdout_eval['all_passed']} ({elapsed:.1f}s)"
        )

    # Statistical computation
    n = len(holdout_scores)
    mean_holdout = sum(holdout_scores) / n
    variance = sum((x - mean_holdout) ** 2 for x in holdout_scores) / (n - 1) if n > 1 else 0.0
    sd_holdout = math.sqrt(variance)
    se_holdout = sd_holdout / math.sqrt(n) if n > 0 else 0.0
    ci_95 = (mean_holdout - 1.96 * se_holdout, mean_holdout + 1.96 * se_holdout)

    mean_dev = sum(dev_scores) / n

    report = {
        "num_seeds": n,
        "population_size": population_size,
        "generations": generations,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "mean_holdout_composite": round(mean_holdout, 4),
            "sd_holdout_composite": round(sd_holdout, 4),
            "se_holdout_composite": round(se_holdout, 4),
            "ci_95_low": round(ci_95[0], 4),
            "ci_95_high": round(ci_95[1], 4),
            "min_holdout_composite": round(min(holdout_scores), 4),
            "max_holdout_composite": round(max(holdout_scores), 4),
            "mean_dev_composite": round(mean_dev, 4),
            "all_holdout_passed_rate": round(sum(1 for r in seed_records if r["holdout_all_passed"]) / n, 4),
            "text_integrity_pass_rate": round(sum(r["text_integrity_pass_rate"] for r in seed_records) / n, 4),
        },
        "seeds": seed_records,
    }

    print("\n=== Multi-Seed Statistical Summary ===")
    print(f"Mean Holdout Score:     {mean_holdout:.2%}")
    print(f"Standard Deviation:     {sd_holdout:.4f}")
    print(f"Standard Error (SE):    {se_holdout:.4f}")
    print(f"95% Confidence Interval: [{ci_95[0]:.2%}, {ci_95[1]:.2%}]")
    print(f"Min / Max Holdout:      {min(holdout_scores):.2%} / {max(holdout_scores):.2%}")
    print(f"Pass Rate across Seeds: {report['summary']['all_holdout_passed_rate']:.2%}")

    save_path = save_report_path or (
        Path(__file__).resolve().parent.parent.parent / "reports" / "pdf2rtf_multiseed_evaluation.json"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Multi-Seed Report Saved: {save_path}")

    return report


if __name__ == "__main__":
    run_multiseed_evaluation(num_seeds=20)
