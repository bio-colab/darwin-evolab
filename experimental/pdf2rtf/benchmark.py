"""benchmark.py — Empirical Benchmark Runner for PDF-to-RTF Hardening.

Audits candidate extraction policies against the Golden Corpus, evaluating
text integrity, structural fidelity, table matrices, formatting retention,
and composite scores across all verification gates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .corpus import CorpusItem, create_golden_corpus
from .equivalence import MultiGateVerifier
from .genome import ProfilePolicy
from .pdf_extractor import PDFExtractor
from .rtf_emitter import emit_rtf
from .rtf_parser import parse_rtf


@dataclass
class DocumentBenchmarkResult:
    """Benchmark outcome for an individual document in the corpus."""

    name: str
    passed: bool
    composite_score: float
    gate_scores: dict[str, float]
    gate_pass_status: dict[str, bool]
    elapsed_ms: float
    diff_count: int = 0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "composite_score": round(self.composite_score, 4),
            "gate_scores": {k: round(v, 4) for k, v in self.gate_scores.items()},
            "gate_pass_status": self.gate_pass_status,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "diff_count": self.diff_count,
            "message": self.message,
        }


@dataclass
class GoldenBenchmarkReport:
    """Consolidated benchmark audit across all corpus archetypes."""

    total_documents: int
    text_integrity_pass_rate: float
    overall_pass_rate: float
    average_composite_score: float
    documents: list[DocumentBenchmarkResult] = field(default_factory=list)
    policy_used: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_documents": self.total_documents,
            "text_integrity_pass_rate": round(self.text_integrity_pass_rate, 4),
            "overall_pass_rate": round(self.overall_pass_rate, 4),
            "average_composite_score": round(self.average_composite_score, 4),
            "policy_used": self.policy_used,
            "documents": [d.to_dict() for d in self.documents],
        }

    def save_json(self, path: str | Path) -> Path:
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_p


def run_golden_benchmark(
    policy: ProfilePolicy | None = None,
    corpus: list[CorpusItem] | None = None,
    verifier: MultiGateVerifier | None = None,
    save_report_path: str | Path | None = None,
) -> GoldenBenchmarkReport:
    """Executes the complete Golden Benchmark audit across all corpus archetypes."""
    active_corpus = corpus or create_golden_corpus()
    active_policy = policy or ProfilePolicy()
    active_verifier = verifier or MultiGateVerifier()

    extractor = PDFExtractor(policy=active_policy)
    results: list[DocumentBenchmarkResult] = []

    for item in active_corpus:
        t0 = time.perf_counter()

        # Step 1: Extract IR using candidate policy
        cand_ir = extractor.extract(item.pdf_bytes)

        # Step 2: Emit RTF
        rtf_text = emit_rtf(cand_ir)

        # Step 3: Parse RTF
        recon_ir = parse_rtf(rtf_text)

        # Step 4: Verify via MultiGateVerifier
        report = active_verifier.verify(candidate=recon_ir, reference=item.reference_doc)

        elapsed = (time.perf_counter() - t0) * 1000.0

        gate_scores = {g.name: g.score for g in report.gates}
        gate_pass = {g.name: g.passed for g in report.gates}

        g1 = report.get_gate("Gate 1: Text Integrity")
        diff_count = g1.details.get("diff_count", 0) if g1 else 0

        doc_result = DocumentBenchmarkResult(
            name=item.name,
            passed=report.passed,
            composite_score=report.composite_score,
            gate_scores=gate_scores,
            gate_pass_status=gate_pass,
            elapsed_ms=elapsed,
            diff_count=diff_count,
            message=report.summary,
        )
        results.append(doc_result)

    # Compute aggregate metrics
    total = len(results)
    text_passes = sum(
        1 for r in results if r.gate_pass_status.get("Gate 1: Text Integrity", False)
    )
    overall_passes = sum(1 for r in results if r.passed)
    avg_score = (sum(r.composite_score for r in results) / total) if total > 0 else 0.0

    benchmark_report = GoldenBenchmarkReport(
        total_documents=total,
        text_integrity_pass_rate=(text_passes / total) if total > 0 else 0.0,
        overall_pass_rate=(overall_passes / total) if total > 0 else 0.0,
        average_composite_score=avg_score,
        documents=results,
        policy_used=active_policy.to_dict(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if save_report_path is not None:
        benchmark_report.save_json(save_report_path)

    return benchmark_report
