"""Unit and integration tests for PDF2RTF DomainAdapter, ProfileGenome, and Evolab Engine."""

from __future__ import annotations

import json
from pathlib import Path
import random
import pytest

from evolab.adapters import get_domain_adapter
from evolab.engine import EvolutionEngine
from evolab.evaluators import FitnessResult
from evolab.genome import Individual

from experimental.pdf2rtf.adapter import (
    PDF2RTFAdapter,
    PDF2RTFEvaluator,
    PDF2RTFSpec,
)
from experimental.pdf2rtf.genome import PARAM_BOUNDS, ProfileGenome, ProfilePolicy
from experimental.pdf2rtf.pdf_extractor import PDFExtractor


def test_profile_genome_contract():
    g1 = ProfileGenome(space_gap_ratio=0.25, para_split_delta_ratio=1.40)
    g2 = g1.clone()

    assert g1.fingerprint() == g2.fingerprint()
    assert g1.distance_to(g2) == 0.0

    # Bounds clamping test
    g_out_of_bounds = ProfileGenome(space_gap_ratio=0.01, para_split_delta_ratio=10.0)
    assert g_out_of_bounds.space_gap_ratio == PARAM_BOUNDS["space_gap_ratio"][0]
    assert g_out_of_bounds.para_split_delta_ratio == PARAM_BOUNDS["para_split_delta_ratio"][1]

    # Distance to different genome
    g3 = ProfileGenome(space_gap_ratio=0.50, para_split_delta_ratio=2.50)
    dist = g1.distance_to(g3)
    assert 0.0 < dist <= 1.0

    # MAP-Elites descriptors
    desc = g1.describe()
    assert "cluster_density" in desc
    assert "table_sensitivity" in desc
    assert 0.0 <= desc["cluster_density"] <= 1.0
    assert 0.0 <= desc["table_sensitivity"] <= 1.0

    # Mutation
    rng = random.Random(42)
    mutant = g1.mutate(rng=rng, sigma=0.2)
    assert mutant.fingerprint() != g1.fingerprint()
    assert PARAM_BOUNDS["space_gap_ratio"][0] <= mutant.space_gap_ratio <= PARAM_BOUNDS["space_gap_ratio"][1]

    # Crossover
    child = g1.crossover(g3, rng=rng)
    assert isinstance(child, ProfileGenome)

    # Serialization roundtrip
    s = g1.serialize()
    assert "space_gap_ratio" in s
    policy = g1.to_policy()
    assert policy.space_gap_ratio == g1.space_gap_ratio
    g_from_pol = ProfileGenome.from_policy(policy)
    assert g_from_pol.fingerprint() == g1.fingerprint()


def test_pdf_extractor_consumes_policy():
    policy = ProfilePolicy(space_gap_ratio=0.40, para_split_delta_ratio=1.80, align_tolerance_pt=10.0)
    extractor = PDFExtractor(policy=policy)
    assert extractor.space_gap_threshold_ratio == 0.40
    assert extractor.policy.para_split_delta_ratio == 1.80
    assert extractor.policy.align_tolerance_pt == 10.0


def test_pdf2rtf_spec_and_evaluator():
    spec = PDF2RTFSpec(name="test_calib")
    pdf_bytes = spec.get_pdf_bytes()
    assert pdf_bytes is not None and len(pdf_bytes) > 0
    assert spec.reference_doc is not None

    evaluator = PDF2RTFEvaluator(spec)
    assert evaluator.deterministic is True

    genome = ProfileGenome()
    result = evaluator.evaluate(genome)

    assert isinstance(result, FitnessResult)
    assert 0.0 <= result.score <= 1.0
    # Baseline profile should score high on synthetic text
    assert result.score > 0.80
    assert "gates" in result.artifacts


def test_domain_adapter_contract(tmp_path: Path):
    adapter = get_domain_adapter("pdf2rtf")
    assert isinstance(adapter, PDF2RTFAdapter)
    assert adapter.name == "pdf2rtf"

    # parse_spec
    spec = adapter.parse_spec(None)
    assert isinstance(spec, PDF2RTFSpec)

    # build_population
    rng = random.Random(123)
    pop = adapter.build_population(spec, size=6, rng=rng)
    assert len(pop) == 6
    for ind in pop:
        assert isinstance(ind, Individual)
        assert isinstance(ind.genome, ProfileGenome)
        assert ind.species == "spec_pdf_profile"

    # build_evaluator
    evaluator = adapter.build_evaluator(spec)
    assert isinstance(evaluator, PDF2RTFEvaluator)

    # export_solution
    best_ind = pop[0]
    best_ind.fitness = 0.985
    out_file = tmp_path / "calibrated_profile.json"
    exported = adapter.export_solution(best_ind, spec, output_path=out_file)

    assert exported["fitness"] == 0.985
    assert "policy" in exported
    assert "descriptors" in exported
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["fitness"] == 0.985
    assert loaded["policy"]["space_gap_ratio"] == best_ind.genome.space_gap_ratio


def test_mini_evolutionary_optimization_run(tmp_path: Path):
    """Verifies end-to-end integration into Darwin-Evolab's EvolutionEngine and MAP-Elites."""
    adapter = PDF2RTFAdapter()
    spec = adapter.parse_spec(None)
    rng = random.Random(42)
    pop = adapter.build_population(spec, size=6, rng=rng)
    evaluator = adapter.build_evaluator(spec)

    # Configure MAP-Elites descriptors
    desc_x = lambda g: g.describe()["cluster_density"]
    desc_y = lambda g: g.describe()["table_sensitivity"]

    engine = EvolutionEngine(
        evaluator=evaluator,
        population_size=6,
        generations=2,
        mutation_rate=0.4,
        me_grid_x=4,
        me_grid_y=4,
        descriptors=[desc_x, desc_y],
        seed=42,
    )

    report = engine.run(generations=2, initial_population=pop)
    assert report["total_generations"] >= 2
    assert "best_individual" in report
    assert report["best_individual"]["fitness"] > 0.0

    # Ensure MAP-Elites archive recorded elite solutions
    assert len(engine._archive) > 0

    # Pick an elite from archive to export solution
    elite_ind = next(iter(engine._archive.values()))
    assert isinstance(elite_ind.genome, ProfileGenome)

    out_file = tmp_path / "evolved_profile.json"
    solution = adapter.export_solution(elite_ind, spec, output_path=out_file)
    assert out_file.exists()
    assert solution["fitness"] == elite_ind.fitness
