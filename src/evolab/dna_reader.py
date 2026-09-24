"""dna_reader.py — Interoceptive DNA Reader, Schema Mining, Intron Detection, and Evolutionary History.

Architecture:
Part of Darwin-Evolab: Interoceptive Evolutionary Operating System.
Provides a unified introspection layer that deconstructs any digital DNA (Software AST,
Silicon CGP, or Numerical Float), distinguishes active exons from neutral introns,
mines Holland building block schemata, audits bloat, and reads search history
as a strategy genome for Dream-RSI and Governor self-governance.

Theoretical Foundations:
- Wagner & Altenberg (1996): Complex Adaptations and the Evolution of Evolvability.
- Holland (1975): Schema Theorem & Building Block Hypothesis.
- Julian Miller & Andrew Turner (EuroGP 2014): Neutral Drift and Why No Bloat in CGP.
- John Koza (1992): Parsimony Pressure and Intron Bloat Defense.
- DeepMind Dream-RSI (2026): Replay History as a Search Strategy Simulator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from .genome import EvolabGenome, FloatGenome, Individual


# ============================================================================
# Layer 0: DNA Sequencer & Canonical Gene Units
# ============================================================================

class GeneType(str, Enum):
    EXON = "EXON"         # Functional / active / expressed gene
    INTRON = "INTRON"     # Neutral / non-functional / inactive reservoir gene
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GeneUnit:
    """Canonical representation of an individual gene locus."""
    locus_id: str
    kind: str
    payload: Any
    gene_type: GeneType = GeneType.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "locus_id": self.locus_id,
            "kind": self.kind,
            "payload": str(self.payload),
            "gene_type": self.gene_type.value,
        }


@dataclass
class DNASequence:
    """Canonical sequence abstraction representing any digital genome."""
    genome_type: str
    exons: list[GeneUnit]
    introns: list[GeneUnit]
    total_genes: int
    active_genes: int
    neutral_ratio: float
    signature: str
    raw_genome: Any = field(repr=False)

    def describe(self) -> dict[str, Any]:
        return {
            "genome_type": self.genome_type,
            "total_genes": self.total_genes,
            "active_genes": self.active_genes,
            "intron_count": len(self.introns),
            "neutral_ratio": round(self.neutral_ratio, 4),
            "signature": self.signature,
        }


class DNASequencer:
    """Universal sequencer that translates heterogeneous genomes into canonical DNASequence."""

    def sequence(self, genome: Any) -> DNASequence:
        """Deconstruct any EvolabGenome or raw genome into canonical exons and introns."""
        inner = getattr(genome, "genome", genome)

        # 1. Cartesian Genetic Programming (CGP Logic)
        if hasattr(inner, "get_active_nodes") and hasattr(inner, "nodes"):
            return self._sequence_cgp(inner)

        # 2. Software AST Repair Genome
        if hasattr(inner, "edits") and hasattr(inner, "sources"):
            return self._sequence_ast(inner)

        # 3. Numerical Float / Array Genome
        if isinstance(inner, FloatGenome) or hasattr(inner, "values") or isinstance(inner, (list, tuple)):
            return self._sequence_float(inner)

        # Fallback generic genome
        return self._sequence_generic(inner)

    def _sequence_cgp(self, cgp: Any) -> DNASequence:
        active_indices = cgp.get_active_nodes()
        num_inputs = getattr(cgp, "num_inputs", 0)
        exons: list[GeneUnit] = []
        introns: list[GeneUnit] = []

        for i, node in enumerate(cgp.nodes):
            node_idx = num_inputs + i
            gate_name = getattr(node.gate_type, "value", str(node.gate_type))
            locus = f"node_{node_idx}"
            unit = GeneUnit(
                locus_id=locus,
                kind=gate_name,
                payload=(node.input_a, node.input_b),
                gene_type=GeneType.EXON if node_idx in active_indices else GeneType.INTRON,
                metadata={"grid_pos": i},
            )
            if node_idx in active_indices:
                exons.append(unit)
            else:
                introns.append(unit)

        total = len(cgp.nodes)
        act_cnt = len(exons)
        neutral_ratio = len(introns) / max(1, total)
        sig = hashlib.sha256(f"cgp:{[e.locus_id for e in exons]}".encode()).hexdigest()[:16]

        return DNASequence(
            genome_type="CGPGenome",
            exons=exons,
            introns=introns,
            total_genes=total,
            active_genes=act_cnt,
            neutral_ratio=neutral_ratio,
            signature=sig,
            raw_genome=cgp,
        )

    def _sequence_ast(self, repair: Any) -> DNASequence:
        exons: list[GeneUnit] = []
        introns: list[GeneUnit] = []
        
        for i, edit in enumerate(repair.edits):
            loc = getattr(edit, "locus", lambda: (getattr(edit, "file", ""), getattr(edit, "lineno", 0), getattr(edit, "col_offset", 0)))()
            locus_id = f"{loc[0]}:{loc[1]}:{loc[2]}"
            unit = GeneUnit(
                locus_id=locus_id,
                kind=getattr(edit, "kind", "unknown"),
                payload=getattr(edit, "payload", ()),
                gene_type=GeneType.EXON, # Default pending IntronDetector attribution
                metadata={"file": loc[0], "line": loc[1], "col": loc[2]},
            )
            exons.append(unit)

        total = len(repair.edits)
        sig = getattr(repair, "fingerprint", lambda: hashlib.sha256(str(exons).encode()).hexdigest()[:16])()

        return DNASequence(
            genome_type="RepairGenome",
            exons=exons,
            introns=introns,
            total_genes=total,
            active_genes=total,
            neutral_ratio=0.0,
            signature=str(sig),
            raw_genome=repair,
        )

    def _sequence_float(self, fgenome: Any) -> DNASequence:
        vals = list(getattr(fgenome, "values", fgenome))
        exons: list[GeneUnit] = []
        for i, v in enumerate(vals):
            exons.append(
                GeneUnit(
                    locus_id=f"dim_{i}",
                    kind="float_weight",
                    payload=round(float(v), 6),
                    gene_type=GeneType.EXON,
                )
            )
        total = len(vals)
        sig = hashlib.sha256(f"float:{vals}".encode()).hexdigest()[:16]
        return DNASequence(
            genome_type="FloatGenome",
            exons=exons,
            introns=[],
            total_genes=total,
            active_genes=total,
            neutral_ratio=0.0,
            signature=sig,
            raw_genome=fgenome,
        )

    def _sequence_generic(self, generic: Any) -> DNASequence:
        sig = hashlib.sha256(str(generic).encode()).hexdigest()[:16]
        unit = GeneUnit(locus_id="l0", kind="generic", payload=str(generic), gene_type=GeneType.EXON)
        return DNASequence(
            genome_type=type(generic).__name__,
            exons=[unit],
            introns=[],
            total_genes=1,
            active_genes=1,
            neutral_ratio=0.0,
            signature=sig,
            raw_genome=generic,
        )


# ============================================================================
# Layer 1: Genome Analyzer (Introns, Schemata, Neutrality, Bloat)
# ============================================================================

@dataclass
class IntronAblationReport:
    """Report detailing functional exons vs hitchhiking non-functional introns."""
    original_gene_count: int
    functional_exons_count: int
    hitchhiking_introns_count: int
    pruned_genome: Any
    intron_loci: list[str]
    fitness_preserved: bool
    efficiency_gain_pct: float


class IntronDetector:
    """Detects hitchhiking non-functional code mutations and neutral bloat."""

    def analyze_ast_introns(
        self,
        genome: Any,
        evaluator: Any,
    ) -> IntronAblationReport:
        """Ablation analysis on multi-edit AST genomes to find neutral hitchhikers.
        
        Tests whether removing each edit changes the test fitness. Any edit
        whose omission preserves 100% of the fitness and holdout score is
        diagnosed as an Intron (Hitchhiker) and pruned.
        """
        inner = getattr(genome, "genome", genome)
        if not hasattr(inner, "edits") or len(inner.edits) <= 1:
            return IntronAblationReport(
                original_gene_count=len(getattr(inner, "edits", [])),
                functional_exons_count=len(getattr(inner, "edits", [])),
                hitchhiking_introns_count=0,
                pruned_genome=inner,
                intron_loci=[],
                fitness_preserved=True,
                efficiency_gain_pct=0.0,
            )

        # Baseline evaluation
        base_res = evaluator.evaluate(inner) if hasattr(evaluator, "evaluate") else evaluator(inner)
        base_score = float(getattr(base_res, "score", base_res))
        base_hold = getattr(base_res, "passed_holdout", True)

        active_edits = list(inner.edits)
        intron_loci: list[str] = []

        # Backward single-ablation test
        for i in range(len(inner.edits) - 1, -1, -1):
            candidate_edits = [e for idx, e in enumerate(active_edits) if idx != i]
            trial = inner.clone()
            trial.edits = candidate_edits
            
            res = evaluator.evaluate(trial) if hasattr(evaluator, "evaluate") else evaluator(trial)
            score = float(getattr(res, "score", res))
            hold = getattr(res, "passed_holdout", True)

            # If fitness and holdout are unchanged, the ablated edit was a neutral intron!
            if score >= base_score and (hold is True or hold == base_hold):
                ablated_edit = active_edits[i]
                loc = getattr(ablated_edit, "locus", lambda: ("", ablated_edit.lineno, ablated_edit.col_offset))()
                intron_loci.append(f"{loc[0]}:{loc[1]}:{loc[2]}:{ablated_edit.kind}")
                active_edits = candidate_edits # Keep pruned state

        pruned = inner.clone()
        pruned.edits = active_edits
        orig_len = len(inner.edits)
        pruned_len = len(active_edits)
        gain_pct = ((orig_len - pruned_len) / max(1, orig_len)) * 100.0

        return IntronAblationReport(
            original_gene_count=orig_len,
            functional_exons_count=pruned_len,
            hitchhiking_introns_count=len(intron_loci),
            pruned_genome=pruned,
            intron_loci=intron_loci,
            fitness_preserved=True,
            efficiency_gain_pct=round(gain_pct, 2),
        )


@dataclass
class Schema:
    """Holland Building Block representation: pattern, order, and defining length."""
    pattern_tuple: tuple[str, ...]
    order: int                  # o(H): count of fixed positions
    defining_length: int        # delta(H): distance between first and last fixed position
    observed_count: int
    mean_fitness: float
    is_building_block: bool     # True if low-order, short length, above-average fitness


class SchemaMiner:
    """Mines Holland Building Blocks and recurrent co-adapted schemata across populations."""

    def mine_schemata(
        self,
        population: list[Any],
        min_support: int = 2,
    ) -> list[Schema]:
        """Extracts short, low-order, above-average schemata from a population."""
        if not population:
            return []

        # Calculate population mean fitness
        fit_vals = [float(getattr(ind, "fitness", 0.0)) for ind in population]
        pop_mean_fit = sum(fit_vals) / max(1, len(fit_vals))

        # Extract edit sequences or discrete patterns
        patterns_to_fits: dict[tuple[str, ...], list[float]] = defaultdict(list)
        pattern_loci: dict[tuple[str, ...], list[int]] = defaultdict(list)

        for ind in population:
            inner = getattr(ind, "genome", ind)
            fit = float(getattr(ind, "fitness", 0.0))

            if hasattr(inner, "edits") and inner.edits:
                kinds = [e.kind for e in inner.edits]
                lines = [getattr(e, "lineno", 0) for e in inner.edits]
                # Mine sub-patterns of length 1, 2, 3
                for k in range(1, min(4, len(kinds) + 1)):
                    for start in range(len(kinds) - k + 1):
                        sub_tuple = tuple(kinds[start : start + k])
                        patterns_to_fits[sub_tuple].append(fit)
                        if lines:
                            delta = max(lines[start : start + k]) - min(lines[start : start + k])
                            pattern_loci[sub_tuple].append(delta)

        schemata: list[Schema] = []
        for pat, fits in patterns_to_fits.items():
            if len(fits) >= min_support:
                mean_f = sum(fits) / len(fits)
                order = len(pat)
                avg_delta = int(sum(pattern_loci[pat]) / max(1, len(pattern_loci[pat]))) if pat in pattern_loci else 0
                is_bb = bool(order <= 3 and avg_delta <= 15 and mean_f >= pop_mean_fit)

                schemata.append(
                    Schema(
                        pattern_tuple=pat,
                        order=order,
                        defining_length=avg_delta,
                        observed_count=len(fits),
                        mean_fitness=round(mean_f, 2),
                        is_building_block=is_bb,
                    )
                )

        schemata.sort(key=lambda s: (s.is_building_block, s.mean_fitness), reverse=True)
        return schemata


@dataclass
class NeutralityMetrics:
    """Julian Miller / Ebner neutrality network metrics."""
    total_nodes: int
    active_nodes: int
    neutral_nodes: int
    neutral_ratio: float
    evolvability_buffer_status: str  # NOMINAL, STARVATION_RISK, EXCESSIVE_NEUTRALITY


class NeutralityAnalyzer:
    """Measures neutral genetic networks and reservoir buffering for escaping local minima."""

    def analyze(self, dna_seq: DNASequence) -> NeutralityMetrics:
        tot = dna_seq.total_genes
        act = dna_seq.active_genes
        neut = len(dna_seq.introns)
        ratio = dna_seq.neutral_ratio

        if ratio < 0.05 and tot > 10:
            status = "STARVATION_RISK" # Lack of neutral paths to bypass local optima
        elif ratio > 0.95 and tot > 20:
            status = "EXCESSIVE_NEUTRALITY"
        else:
            status = "NOMINAL"

        return NeutralityMetrics(
            total_nodes=tot,
            active_nodes=act,
            neutral_nodes=neut,
            neutral_ratio=round(ratio, 4),
            evolvability_buffer_status=status,
        )


@dataclass
class BloatDiagnostics:
    """Turner / Koza code and structural bloat analysis."""
    turner_bloat_ratio: float
    structural_size: int
    is_bloated: bool
    verdict: str


class BloatMonitor:
    """Monitors code and circuit growth against fitness progress to flag pathological bloat."""

    def evaluate_bloat(
        self,
        current_size: int,
        initial_size: int,
        current_fitness: float,
        initial_fitness: float,
    ) -> BloatDiagnostics:
        """Turner bloat equation: B(g) = (Size(g)/Size(0)) * (Fit(0) / Fit*(g))."""
        s_ratio = current_size / max(1, initial_size)
        f_ratio = max(1e-4, initial_fitness) / max(1e-4, current_fitness)
        turner_metric = round(s_ratio * f_ratio, 4)

        # Pathological bloat: size doubled with no meaningful fitness increase
        is_bloat = bool(s_ratio >= 2.0 and (current_fitness <= initial_fitness * 1.05))
        verdict = "PATHOLOGICAL_BLOAT" if is_bloat else ("HEALTHY_PARSIMONY" if s_ratio <= 1.2 else "ACCEPTABLE_GROWTH")

        return BloatDiagnostics(
            turner_bloat_ratio=turner_metric,
            structural_size=current_size,
            is_bloated=is_bloat,
            verdict=verdict,
        )


# ============================================================================
# Layer 2: Evolutionary Historian (Dream-RSI & Strategy Genome)
# ============================================================================

@dataclass
class StrategyGenomeReport:
    """Introspective analysis of search history as an evolving strategy genome."""
    total_generations: int
    operator_velocity: dict[str, int]
    governor_acceptance_rate: float
    governor_drift_detected: bool
    dream_replay_readiness: bool
    verdict: str


class EvolutionaryHistorian:
    """Reads discovery trees and search logs as a meta-genome of strategy itself."""

    def analyze_history(
        self,
        history: list[dict[str, Any]],
        decision_log: list[dict[str, Any]] | None = None,
        alpha: float = 0.05,
        epsilon: float = 1e-6,
    ) -> StrategyGenomeReport:
        if not history:
            return StrategyGenomeReport(
                total_generations=0,
                operator_velocity={},
                governor_acceptance_rate=0.0,
                governor_drift_detected=False,
                dream_replay_readiness=False,
                verdict="INSUFFICIENT_HISTORY",
            )

        decisions = decision_log or []
        op_counter: Counter[str] = Counter()
        accept_count = 0
        total_proposals = 0

        for d in decisions:
            event = d.get("event", "")
            delta = float(d.get("delta", 0.0) or d.get("fitness_delta", 0.0) or 0.0)
            if "proposal" in event or "governor" in event:
                total_proposals += 1
                action = d.get("action", "")
                # Enforce epsilon invariant: neutral drift delta <= epsilon cannot be counted as an improvement accept
                if ("accept" in str(d.get("detail", "")).lower() or action == "ACCEPT"):
                    if "delta" in d or "fitness_delta" in d:
                        if delta > epsilon:
                            accept_count += 1
                    else:
                        accept_count += 1
            if "operator" in d:
                op_counter[str(d["operator"])] += 1

        acc_rate = (accept_count / max(1, total_proposals)) if total_proposals else 0.0
        # Governor drift: acceptance rate too permissive (> 30% indicates delusion trap risk) or starved (< 1%)
        drift = bool(total_proposals >= 10 and (acc_rate > 0.30 or acc_rate < 0.01))

        # Dream-RSI readiness: history is long and diverse enough to serve as offline replay simulator
        unique_trajectories = len(set(h.get("best_fitness", 0.0) for h in history))
        dream_ready = bool(len(history) >= 8 and unique_trajectories >= 4)

        verdict = "GOVERNOR_DRIFT_DETECTED" if drift else ("DREAM_RSI_READY" if dream_ready else "STABLE_STRATEGY")

        return StrategyGenomeReport(
            total_generations=len(history),
            operator_velocity=dict(op_counter),
            governor_acceptance_rate=round(acc_rate, 4),
            governor_drift_detected=drift,
            dream_replay_readiness=dream_ready,
            verdict=verdict,
        )


# ============================================================================
# High-Level Interoceptive DNAReader
# ============================================================================

class DNAReader:
    """Unified Interoceptive DNA Reader for Darwin-Evolab.
    
    Reads, diagnoses, and understands the internal structure of digital genomes
    across Software Repair, Cartesian Logic, and Numerical landscapes.
    """

    def __init__(self) -> None:
        self.sequencer = DNASequencer()
        self.intron_detector = IntronDetector()
        self.schema_miner = SchemaMiner()
        self.neutrality_analyzer = NeutralityAnalyzer()
        self.bloat_monitor = BloatMonitor()
        self.historian = EvolutionaryHistorian()

    def read_genome(
        self,
        genome: Any,
        evaluator: Any = None,
        initial_size: int | None = None,
        initial_fitness: float | None = None,
    ) -> dict[str, Any]:
        """Comprehensive multi-level diagnostic scan of an individual genome."""
        seq = self.sequencer.sequence(genome)
        neutral_metrics = self.neutrality_analyzer.analyze(seq)

        intron_report = None
        if evaluator is not None and seq.genome_type == "RepairGenome":
            intron_report = self.intron_detector.analyze_ast_introns(genome, evaluator)

        bloat_diag = None
        curr_fit = float(getattr(genome, "fitness", 0.0))
        if initial_size is not None and initial_fitness is not None:
            bloat_diag = self.bloat_monitor.evaluate_bloat(
                current_size=seq.total_genes,
                initial_size=initial_size,
                current_fitness=curr_fit,
                initial_fitness=initial_fitness,
            )

        return {
            "sequence": seq.describe(),
            "neutrality": {
                "total_nodes": neutral_metrics.total_nodes,
                "active_nodes": neutral_metrics.active_nodes,
                "neutral_nodes": neutral_metrics.neutral_nodes,
                "neutral_ratio": neutral_metrics.neutral_ratio,
                "status": neutral_metrics.evolvability_buffer_status,
            },
            "introns": {
                "functional_exons": intron_report.functional_exons_count,
                "hitchhiking_introns": intron_report.hitchhiking_introns_count,
                "intron_loci": intron_report.intron_loci,
                "efficiency_gain_pct": intron_report.efficiency_gain_pct,
            } if intron_report else None,
            "bloat": {
                "turner_bloat_ratio": bloat_diag.turner_bloat_ratio,
                "is_bloated": bloat_diag.is_bloated,
                "verdict": bloat_diag.verdict,
            } if bloat_diag else None,
        }

    def read_population(
        self,
        population: list[Any],
        min_support: int = 2,
    ) -> dict[str, Any]:
        """Mines population-level building block schemata and genetic diversity."""
        schemata = self.schema_miner.mine_schemata(population, min_support=min_support)
        building_blocks = [s for s in schemata if s.is_building_block]

        return {
            "population_size": len(population),
            "schemata_mined": len(schemata),
            "building_blocks_found": len(building_blocks),
            "top_building_blocks": [
                {
                    "pattern": list(s.pattern_tuple),
                    "order": s.order,
                    "defining_length": s.defining_length,
                    "mean_fitness": s.mean_fitness,
                    "support": s.observed_count,
                }
                for s in building_blocks[:5]
            ],
        }

    def read_history(
        self,
        history: list[dict[str, Any]],
        decision_log: list[dict[str, Any]] | None = None,
        alpha: float = 0.05,
        epsilon: float = 1e-6,
    ) -> dict[str, Any]:
        """Reads search history as an evolving strategy genome for Dream-RSI and Governor."""
        report = self.historian.analyze_history(history, decision_log, alpha=alpha, epsilon=epsilon)
        return {
            "total_generations": report.total_generations,
            "governor_acceptance_rate": report.governor_acceptance_rate,
            "governor_drift_detected": report.governor_drift_detected,
            "dream_replay_readiness": report.dream_replay_readiness,
            "verdict": report.verdict,
        }
