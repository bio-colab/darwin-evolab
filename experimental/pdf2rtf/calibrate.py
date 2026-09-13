"""calibrate.py — Evolutionary Hyperparameter Calibration across the Golden Corpus.

Executes MAP-Elites evolutionary optimization on the 7-dimensional heuristic policy
space to discover the highest-fidelity profile for Microsoft Word-generated PDFs.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any

from evolab.engine import EvolutionEngine
from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import Individual

from .benchmark import run_golden_benchmark, run_holdout_benchmark
from .corpus import CorpusItem, create_golden_corpus, create_synthetic_dev_corpus
from .equivalence import MultiGateVerifier
from .genome import PARAM_BOUNDS, ProfileGenome, ProfilePolicy
from .pdf_extractor import PDFExtractor
from .rtf_emitter import emit_rtf
from .rtf_parser import parse_rtf


class MultiDocumentCorpusEvaluator(Evaluator):
    """Evaluates a ProfileGenome across the entire Golden Benchmark Corpus."""

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

        # Arithmetic mean across all corpus archetypes
        mean_score = (sum(scores) / len(scores)) if scores else 0.0

        return FitnessResult(
            score=mean_score,
            sub_scores=sub_scores,
            artifacts={"mean_score": mean_score, "document_scores": scores},
        )


def calibrate_golden_profile(
    population_size: int = 12,
    generations: int = 4,
    seed: int = 42,
    output_profile_path: str | Path | None = None,
    output_benchmark_path: str | Path | None = None,
) -> tuple[ProfilePolicy, dict[str, Any]]:
    """Runs evolutionary calibration and produces audited calibrated profile and benchmark report."""
    corpus = create_golden_corpus()
    evaluator = MultiDocumentCorpusEvaluator(corpus)
    rng = random.Random(seed)

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
        me_grid_x=4,
        me_grid_y=4,
        descriptors=[desc_x, desc_y],
        seed=seed,
    )

    report = engine.run(generations=generations, initial_population=pop)

    # Select the champion individual from the MAP-Elites archive
    best_ind: Individual = pop[0]
    best_fitness = -1.0
    for cell_ind in engine._archive.values():
        if cell_ind.fitness > best_fitness:
            best_fitness = cell_ind.fitness
            best_ind = cell_ind

    champion_policy = best_ind.genome.to_policy()

    # Save calibrated profile JSON
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

    return champion_policy, {
        "dev_benchmark": bench_report.to_dict(),
        "holdout_benchmark": holdout_report.to_dict(),
    }


if __name__ == "__main__":
    policy, bench_results = calibrate_golden_profile()
    dev_res = bench_results["dev_benchmark"]
    hold_res = bench_results["holdout_benchmark"]
    print(f"Calibration Complete!")
    print(f"  Dev Synthetic Composite Score: {dev_res['average_composite_score']:.2%}")
    print(f"  Holdout Real Word Composite Score: {hold_res['average_composite_score']:.2%}")
    print(f"  Holdout Text Integrity Pass Rate: {hold_res['text_integrity_pass_rate']:.2%}")
    print(f"  Holdout Overall Pass Rate: {hold_res['overall_pass_rate']:.2%}")
