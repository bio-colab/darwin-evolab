"""calibrate.py — Scaled Evolutionary Hyperparameter Calibration & MAP-Elites Archive Discovery.

Executes Quality-Diversity (QD) MAP-Elites evolutionary optimization across a 10-dimensional
heuristic and adaptive policy space on a 10x10 behavioral grid (100 niches).
Integrates Word-in-the-Loop evaluation into evolutionary training to completely eliminate
mirror blindness, and exports the full specialized niche archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

from evolab.engine import EvolutionEngine
from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import Individual

from .benchmark import run_golden_benchmark, run_holdout_benchmark
from .corpus import CorpusItem, create_golden_corpus, create_synthetic_dev_corpus
from .equivalence import MultiGateVerifier
from .genome import PARAM_BOUNDS, ProfileGenome, ProfilePolicy
from .map_elites_archive import ArchiveCell, MAPElitesArchive
from .pdf_extractor import PDFExtractor
from .rtf_emitter import emit_rtf
from .rtf_parser import parse_rtf
from .word_oracle import WordInTheLoopFitnessEvaluator


class MultiDocumentCorpusEvaluator(Evaluator):
    """Evaluates a ProfileGenome across the Golden Benchmark Corpus using internal MultiGateVerifier."""

    def __init__(self, corpus: list[CorpusItem]) -> None:
        self.corpus = corpus
        self.verifier = MultiGateVerifier()

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        genome = getattr(target, "genome", target)
        if not isinstance(genome, ProfileGenome):
            raise TypeError(f"Expected ProfileGenome, got {type(genome)}")

        policy = genome.to_policy()
        extractor = PDFExtractor(policy=policy)

        scores: list[float] = []
        sub_scores: dict[str, float] = {}

        for item in self.corpus:
            cand_ir = extractor.extract(item.pdf_bytes)
            rtf_text = emit_rtf(cand_ir)
            recon_ir = parse_rtf(rtf_text)
            report = self.verifier.verify(candidate=recon_ir, reference=item.reference_doc)

            scores.append(report.composite_score)
            for g in report.gates:
                sub_scores[f"{item.name}_{g.name}"] = g.score

        mean_score = (sum(scores) / len(scores)) if scores else 0.0

        return FitnessResult(
            score=mean_score,
            sub_scores=sub_scores,
            artifacts={"mean_score": mean_score, "document_scores": scores},
        )


def calibrate_golden_profile(
    population_size: int = 100,
    generations: int = 50,
    me_grid_x: int = 10,
    me_grid_y: int = 10,
    evaluator_mode: str = "word_com_hybrid",
    seed: int = 42,
    output_profile_path: str | Path | None = None,
    output_archive_path: str | Path | None = None,
    output_benchmark_path: str | Path | None = None,
    output_stats_path: str | Path | None = None,
) -> tuple[ProfilePolicy, MAPElitesArchive, dict[str, Any]]:
    """Runs scaled evolutionary calibration and produces audited calibrated profile, archive, and reports."""
    corpus = create_golden_corpus()
    rng = random.Random(seed)

    print(f"=== Starting Scaled MAP-Elites Evolution ===")
    print(f"  Population Size: {population_size}")
    print(f"  Generations:     {generations}")
    print(f"  Grid Niches:     {me_grid_x} x {me_grid_y} ({me_grid_x * me_grid_y} niches)")
    print(f"  Evaluator Mode:  {evaluator_mode}")
    print(f"  RNG Seed:        {seed}")

    # Configure Evaluator
    if evaluator_mode in ("word_com_hybrid", "word_com_pure"):
        # Use representative training archetypes for Word-in-the-loop training
        from .word_corpus_54 import load_corpus_54
        c54 = load_corpus_54()
        table_2x2_item = next(item for item in c54 if item.name == "word_seed_31_table_simple_2x2")
        train_corpus = [corpus[0], corpus[1], corpus[2], table_2x2_item]
        oracle_mode = "hybrid" if evaluator_mode == "word_com_hybrid" else "pure"
        evaluator: Evaluator = WordInTheLoopFitnessEvaluator(
            corpus=train_corpus,
            mode=oracle_mode,
            screening_threshold=0.90,
        )
    else:
        evaluator = MultiDocumentCorpusEvaluator(corpus)

    # Initial population: seed 1 with baseline policy, rest with random perturbations
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
            para_split_short_line_factor=rng.uniform(*PARAM_BOUNDS["para_split_short_line_factor"]),
            para_split_indent_factor=rng.uniform(*PARAM_BOUNDS["para_split_indent_factor"]),
            para_split_font_weight=rng.uniform(*PARAM_BOUNDS["para_split_font_weight"]),
        )
        pop.append(Individual(genome=g, species="spec_pdf_profile"))

    # Configure MAP-Elites descriptors
    desc_x = lambda g: g.describe()["cluster_density"]
    desc_y = lambda g: g.describe()["table_sensitivity"]

    engine = EvolutionEngine(
        evaluator=evaluator,
        population_size=population_size,
        generations=generations,
        mutation_rate=0.45,
        me_grid_x=me_grid_x,
        me_grid_y=me_grid_y,
        me_scale_x=1.0,
        me_scale_y=1.0,
        descriptors=[desc_x, desc_y],
        qd_selection=True,
        record_archive_solutions=True,
        seed=seed,
    )

    t_start = time.perf_counter()
    report = engine.run(generations=generations, initial_population=pop)
    total_time = time.perf_counter() - t_start

    # Flush Word COM if active
    if hasattr(evaluator, "flush"):
        evaluator.flush()

    # Build and populate MAPElitesArchive from engine._archive
    archive = MAPElitesArchive(
        grid_x=me_grid_x,
        grid_y=me_grid_y,
        descriptor_names=["cluster_density", "table_sensitivity"],
        calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    best_ind: Individual = pop[0]
    best_fitness = -1.0

    for cell_coord, cell_ind in engine._archive.items():
        g = cell_ind.genome
        p = g.to_policy()
        desc = g.describe()
        archive.update_cell(
            coord=cell_coord,
            fitness=cell_ind.fitness,
            policy=p,
            descriptors={"cluster_density": desc["cluster_density"], "table_sensitivity": desc["table_sensitivity"]},
            fingerprint=g.fingerprint(),
            gen=getattr(cell_ind, "last_evaluated_gen", generations),
        )
        if cell_ind.fitness > best_fitness:
            best_fitness = cell_ind.fitness
            best_ind = cell_ind

    champion_policy = best_ind.genome.to_policy()

    # Save full MAP-Elites archive JSON
    archive_out = Path(output_archive_path or (Path(__file__).parent / "calibrated_map_elites_archive.json"))
    archive.save_json(archive_out)
    print(f"Archive saved: {archive_out} (Occupied Niches: {archive.occupied_count}/{archive.total_capacity}, QD-Score: {archive.qd_score:.2f})")

    # Save monolithic champion profile JSON for backward compatibility
    profile_out = Path(output_profile_path or (Path(__file__).parent / "calibrated_word_profile.json"))
    profile_data = {
        "name": "calibrated_word_profile",
        "description": "Empirically calibrated heuristic extraction policy for MS Word-generated PDFs",
        "fitness_score": best_ind.fitness,
        "fingerprint": best_ind.genome.fingerprint(),
        "descriptors": best_ind.genome.describe(),
        "policy": champion_policy.to_dict(),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(profile_out, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    # Save MAP-Elites Archive Stats Report
    stats_out = Path(output_stats_path or (Path(__file__).resolve().parent.parent.parent / "reports" / "pdf2rtf_map_elites_archive_stats.json"))
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    archive_stats = {
        "title": "MAP-Elites Behavioral Archive Statistics Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "population_size": population_size,
            "generations": generations,
            "grid_dimensions": [me_grid_x, me_grid_y],
            "total_niches": archive.total_capacity,
            "evaluator_mode": evaluator_mode,
            "seed": seed,
            "total_elapsed_seconds": round(total_time, 2),
            "evaluations_per_second": round(report.get("total_candidates_evaluated", generations * population_size) / max(0.001, total_time), 2),
        },
        "archive_metrics": {
            "occupied_niches": archive.occupied_count,
            "coverage_percentage": round(archive.coverage * 100.0, 2),
            "qd_score": round(archive.qd_score, 4),
            "mean_fitness": round(archive.qd_score / max(1, archive.occupied_count), 4),
            "champion_fitness": round(best_fitness, 4),
            "champion_coord": list(best_ind.genome.describe().get("coord", (0, 0))) if hasattr(best_ind.genome, "describe") else [],
        },
        "niche_cells": [c.to_dict() for c in sorted(archive.cells.values(), key=lambda x: x.coord)],
    }
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(archive_stats, f, indent=2)

    # Run complete verification benchmark using champion policy on dev set
    benchmark_out = output_benchmark_path or (
        Path(__file__).resolve().parent.parent.parent / "reports" / "pdf2rtf_golden_benchmark.json"
    )
    bench_report = run_golden_benchmark(
        policy=champion_policy,
        corpus=corpus,
        save_report_path=benchmark_out,
    )

    # Strictly evaluate the champion policy on the OUT-OF-DISTRIBUTION Real Word Holdout set
    holdout_out = Path(__file__).resolve().parent.parent.parent / "reports" / "pdf2rtf_real_word_holdout_benchmark.json"
    holdout_report = run_holdout_benchmark(
        policy=champion_policy,
        save_report_path=holdout_out,
    )

    return champion_policy, archive, {
        "dev_benchmark": bench_report.to_dict(),
        "holdout_benchmark": holdout_report.to_dict(),
        "archive_stats": archive_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Scaled MAP-Elites Evolution for PDF2RTF.")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gens", type=int, default=50, help="Generations")
    parser.add_argument("--grid", type=int, default=10, help="Grid dimension (N x N)")
    parser.add_argument("--mode", type=str, default="word_com_hybrid", choices=["word_com_hybrid", "word_com_pure", "internal"])
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    pol, arch, res = calibrate_golden_profile(
        population_size=args.pop,
        generations=args.gens,
        me_grid_x=args.grid,
        me_grid_y=args.grid,
        evaluator_mode=args.mode,
        seed=args.seed,
    )
    dev_res = res["dev_benchmark"]
    hold_res = res["holdout_benchmark"]
    stats = res["archive_stats"]

    print("\n" + "=" * 60)
    print(f"Calibration Complete!")
    print(f"  Archive Niches Occupied: {stats['archive_metrics']['occupied_niches']}/{stats['configuration']['grid_dimensions'][0]**2} ({stats['archive_metrics']['coverage_percentage']}%)")
    print(f"  Quality-Diversity Score: {stats['archive_metrics']['qd_score']}")
    print(f"  Dev Synthetic Composite Score: {dev_res['average_composite_score']:.2%}")
    print(f"  Holdout Real Word Composite Score: {hold_res['average_composite_score']:.2%}")
    print(f"  Holdout Text Integrity Pass Rate: {hold_res['text_integrity_pass_rate']:.2%}")
    print(f"  Holdout Overall Pass Rate: {hold_res['overall_pass_rate']:.2%}")
    print("=" * 60)
