"""test_dna_reader.py — Comprehensive tests for the Interoceptive DNA Reader.

Covers:
- Layer 0: DNASequencer (CGP, AST Repair, Float)
- Layer 1: GenomeAnalyzer (IntronDetector, SchemaMiner, NeutralityAnalyzer, BloatMonitor)
- Layer 2: EvolutionaryHistorian (Strategy Genome, Governor drift, Dream-RSI readiness)
- Facade: DNAReader end-to-end diagnostics
"""
from __future__ import annotations

import pytest

from evolab.cgp_logic import CGPGenome, CGPNode, GateType, create_random_cgp_genome
from evolab.dna_reader import (
    BloatDiagnostics,
    BloatMonitor,
    DNAReader,
    DNASequence,
    DNASequencer,
    EvolutionaryHistorian,
    GeneType,
    GeneUnit,
    IntronAblationReport,
    IntronDetector,
    NeutralityAnalyzer,
    NeutralityMetrics,
    Schema,
    SchemaMiner,
    StrategyGenomeReport,
)
from evolab.genome import FloatGenome, Individual
from evolab.repair import RepairEdit, RepairGenome


# ============================================================================
# Layer 0: DNASequencer Tests
# ============================================================================

def test_dna_sequencer_cgp():
    """Verify CGP netlist deconstruction into active exons and neutral introns."""
    cgp = create_random_cgp_genome(num_inputs=2, num_outputs=1, num_nodes=12)
    sequencer = DNASequencer()
    seq = sequencer.sequence(cgp)

    assert seq.genome_type == "CGPGenome"
    assert seq.total_genes == 12
    assert seq.active_genes == len(seq.exons)
    assert len(seq.introns) + len(seq.exons) == 12
    assert 0.0 <= seq.neutral_ratio <= 1.0
    assert len(seq.signature) > 0
    assert seq.raw_genome is cgp

    desc = seq.describe()
    assert desc["genome_type"] == "CGPGenome"
    assert desc["total_genes"] == 12
    assert desc["active_genes"] == len(seq.exons)


def test_dna_sequencer_ast_repair():
    """Verify AST RepairGenome sequencing into locus-indexed GeneUnits."""
    edit1 = RepairEdit(kind="cmp_op", file="test.py", lineno=10, col_offset=4)
    edit2 = RepairEdit(kind="boundary", file="test.py", lineno=25, col_offset=8)
    genome = RepairGenome(sources={"test.py": "x = 1\n"}, target_file="test.py", edits=[edit1, edit2])

    sequencer = DNASequencer()
    seq = sequencer.sequence(genome)

    assert seq.genome_type == "RepairGenome"
    assert seq.total_genes == 2
    assert len(seq.exons) == 2
    assert seq.exons[0].locus_id == "test.py:10:4"
    assert seq.exons[0].kind == "cmp_op"
    assert seq.exons[1].locus_id == "test.py:25:8"
    assert seq.exons[1].kind == "boundary"


def test_dna_sequencer_float_genome():
    """Verify numerical FloatGenome coordinate sequencing."""
    fgenome = FloatGenome(values=[1.23, -4.56, 7.89])
    sequencer = DNASequencer()
    seq = sequencer.sequence(fgenome)

    assert seq.genome_type == "FloatGenome"
    assert seq.total_genes == 3
    assert len(seq.exons) == 3
    assert seq.exons[0].locus_id == "dim_0"
    assert seq.exons[0].payload == 1.23


# ============================================================================
# Layer 1: IntronDetector & SchemaMiner & Neutrality & Bloat Tests
# ============================================================================

class MockEvaluator:
    """Evaluator that rewards edit1 but ignores edit2 (hitchhiker)."""
    def evaluate(self, genome: RepairGenome):
        kinds = [e.kind for e in genome.edits]
        if "essential_fix" in kinds:
            return type("Score", (), {"score": 1.0, "passed_holdout": True})()
        return type("Score", (), {"score": 0.0, "passed_holdout": False})()


def test_intron_detector_ast_prunes_hitchhikers():
    """Verify IntronDetector detects and ablates neutral hitchhiking AST edits."""
    edit1 = RepairEdit(kind="essential_fix", file="foo.py", lineno=5, col_offset=0)
    edit2 = RepairEdit(kind="neutral_hitchhiker", file="foo.py", lineno=20, col_offset=0)
    genome = RepairGenome(sources={"foo.py": "x = 0"}, target_file="foo.py", edits=[edit1, edit2])

    evaluator = MockEvaluator()
    detector = IntronDetector()
    report = detector.analyze_ast_introns(genome, evaluator)

    assert report.original_gene_count == 2
    assert report.functional_exons_count == 1
    assert report.hitchhiking_introns_count == 1
    assert report.fitness_preserved is True
    assert report.efficiency_gain_pct == 50.0
    assert len(report.pruned_genome.edits) == 1
    assert report.pruned_genome.edits[0].kind == "essential_fix"
    assert any("neutral_hitchhiker" in locus for locus in report.intron_loci)


def test_intron_detector_single_edit_no_ablation():
    """A single-edit genome requires no ablation and returns immediately."""
    edit1 = RepairEdit(kind="fix", file="foo.py", lineno=5, col_offset=0)
    genome = RepairGenome(sources={"foo.py": "x = 0"}, target_file="foo.py", edits=[edit1])
    report = IntronDetector().analyze_ast_introns(genome, MockEvaluator())

    assert report.hitchhiking_introns_count == 0
    assert report.efficiency_gain_pct == 0.0


def test_schema_miner_holland_building_blocks():
    """Verify Holland Building Block mining across a population."""
    e_good1 = RepairEdit(kind="null_check", file="app.py", lineno=10, col_offset=0)
    e_good2 = RepairEdit(kind="boundary", file="app.py", lineno=12, col_offset=0)
    e_bad = RepairEdit(kind="random_op", file="app.py", lineno=80, col_offset=0)

    # 3 individuals sharing the good pattern (fitness=1.0)
    pop = []
    for _ in range(3):
        g = RepairGenome(sources={"app.py": ""}, target_file="app.py", edits=[e_good1, e_good2])
        pop.append(Individual(genome=g, species="default", fitness=1.0))

    # 2 individuals with the bad op (fitness=0.1)
    for _ in range(2):
        g = RepairGenome(sources={"app.py": ""}, target_file="app.py", edits=[e_bad])
        pop.append(Individual(genome=g, species="default", fitness=0.1))

    miner = SchemaMiner()
    schemata = miner.mine_schemata(pop, min_support=2)

    assert len(schemata) > 0
    bb = [s for s in schemata if s.is_building_block]
    assert len(bb) >= 1
    # Best schema is the null_check + boundary building block
    top_bb = bb[0]
    assert top_bb.mean_fitness > 0.5
    assert top_bb.order <= 3
    assert top_bb.defining_length <= 15


def test_neutrality_analyzer_reservoir_status():
    """Verify NeutralityAnalyzer categorizes reservoir status correctly."""
    analyzer = NeutralityAnalyzer()

    # Nominal neutrality
    seq_nominal = DNASequence(
        genome_type="CGPGenome",
        exons=[GeneUnit(locus_id=f"e{i}", kind="NAND", payload=0) for i in range(10)],
        introns=[GeneUnit(locus_id=f"i{i}", kind="NOR", payload=0) for i in range(10)],
        total_genes=20,
        active_genes=10,
        neutral_ratio=0.5,
        signature="test_nom",
        raw_genome=None,
    )
    m_nominal = analyzer.analyze(seq_nominal)
    assert m_nominal.evolvability_buffer_status == "NOMINAL"

    # Starvation risk: ratio < 0.05 on large genome
    seq_starved = DNASequence(
        genome_type="CGPGenome",
        exons=[GeneUnit(locus_id=f"e{i}", kind="NAND", payload=0) for i in range(20)],
        introns=[],
        total_genes=20,
        active_genes=20,
        neutral_ratio=0.0,
        signature="test_starv",
        raw_genome=None,
    )
    m_starved = analyzer.analyze(seq_starved)
    assert m_starved.evolvability_buffer_status == "STARVATION_RISK"

    # Excessive neutrality: ratio > 0.95 on large genome
    seq_excess = DNASequence(
        genome_type="CGPGenome",
        exons=[GeneUnit(locus_id="e0", kind="NAND", payload=0)],
        introns=[GeneUnit(locus_id=f"i{i}", kind="NOR", payload=0) for i in range(29)],
        total_genes=30,
        active_genes=1,
        neutral_ratio=29 / 30,
        signature="test_excess",
        raw_genome=None,
    )
    m_excess = analyzer.analyze(seq_excess)
    assert m_excess.evolvability_buffer_status == "EXCESSIVE_NEUTRALITY"


def test_bloat_monitor_turner_ratio():
    """Verify Turner Bloat Equation and Koza Parsimony classification."""
    monitor = BloatMonitor()

    # Healthy Parsimony: size remains 10, fitness improved from 0.5 to 1.0
    diag_healthy = monitor.evaluate_bloat(
        current_size=10,
        initial_size=10,
        current_fitness=1.0,
        initial_fitness=0.5,
    )
    assert diag_healthy.turner_bloat_ratio == 0.5
    assert diag_healthy.is_bloated is False
    assert diag_healthy.verdict == "HEALTHY_PARSIMONY"

    # Pathological Bloat: size increased 3x (10 -> 30), fitness stalled (0.5 -> 0.51)
    diag_bloated = monitor.evaluate_bloat(
        current_size=30,
        initial_size=10,
        current_fitness=0.51,
        initial_fitness=0.5,
    )
    assert diag_bloated.turner_bloat_ratio >= 2.0
    assert diag_bloated.is_bloated is True
    assert diag_bloated.verdict == "PATHOLOGICAL_BLOAT"


# ============================================================================
# Layer 2: EvolutionaryHistorian Tests
# ============================================================================

def test_evolutionary_historian_governor_drift():
    """Verify EvolutionaryHistorian detects governor drift and Dream-RSI readiness."""
    historian = EvolutionaryHistorian()

    # Search history with diverse fitness trajectory across 10 generations
    history = [{"generation": i, "best_fitness": 0.1 * i} for i in range(10)]

    # Case 1: Healthy Governor (15% acceptance rate)
    decisions_healthy = [
        {"event": "governor_proposal", "action": "ACCEPT" if i < 2 else "REJECT", "operator": "ast_crossover"}
        for i in range(12)
    ]
    rep_healthy = historian.analyze_history(history, decisions_healthy)
    assert rep_healthy.governor_drift_detected is False
    assert rep_healthy.dream_replay_readiness is True
    assert rep_healthy.verdict == "DREAM_RSI_READY"

    # Case 2: Governor Drift (40% acceptance rate > 30% threshold: too permissive / delusion trap)
    decisions_drift = [
        {"event": "governor_proposal", "action": "ACCEPT" if i < 5 else "REJECT", "operator": "ast_crossover"}
        for i in range(12)
    ]
    rep_drift = historian.analyze_history(history, decisions_drift)
    assert rep_drift.governor_drift_detected is True
    assert rep_drift.verdict == "GOVERNOR_DRIFT_DETECTED"


# ============================================================================
# High-Level DNAReader Facade Tests
# ============================================================================

def test_dna_reader_facade_end_to_end():
    """Verify the high-level DNAReader facade unifies all 3 layers seamlessly."""
    reader = DNAReader()

    # 1. Individual scan with intron detection and bloat monitoring
    edit1 = RepairEdit(kind="essential_fix", file="src.py", lineno=1, col_offset=0)
    edit2 = RepairEdit(kind="neutral_junk", file="src.py", lineno=99, col_offset=0)
    genome = RepairGenome(sources={"src.py": "pass"}, target_file="src.py", edits=[edit1, edit2])
    setattr(genome, "fitness", 0.9)

    evaluator = MockEvaluator()
    scan = reader.read_genome(genome, evaluator=evaluator, initial_size=2, initial_fitness=0.9)

    assert scan["sequence"]["genome_type"] == "RepairGenome"
    assert scan["sequence"]["total_genes"] == 2
    assert scan["introns"]["hitchhiking_introns"] == 1
    assert scan["introns"]["functional_exons"] == 1
    assert scan["bloat"]["verdict"] == "HEALTHY_PARSIMONY"

    # 2. Population scan
    ind1 = Individual(genome=genome, species="default", fitness=0.9)
    ind2 = Individual(genome=genome, species="default", fitness=0.9)
    pop_diag = reader.read_population([ind1, ind2], min_support=2)
    assert pop_diag["population_size"] == 2
    assert pop_diag["schemata_mined"] >= 1

    # 3. History scan
    history = [{"generation": i, "best_fitness": float(i)} for i in range(10)]
    hist_diag = reader.read_history(history)
    assert hist_diag["total_generations"] == 10
    assert hist_diag["dream_replay_readiness"] is True
