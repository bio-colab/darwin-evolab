"""test_corpus_benchmark.py — Unit tests for Phase 4 Golden Corpus and Benchmark Runner."""
import json
from pathlib import Path
import pytest

from experimental.pdf2rtf.benchmark import (
    DocumentBenchmarkResult,
    GoldenBenchmarkReport,
    run_golden_benchmark,
)
from experimental.pdf2rtf.calibrate import MultiDocumentCorpusEvaluator
from experimental.pdf2rtf.corpus import HAS_FITZ, CorpusItem, create_golden_corpus
from experimental.pdf2rtf.genome import PARAM_BOUNDS, ProfileGenome, ProfilePolicy
from experimental.pdf2rtf.ir import Table


@pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF is required for Golden Corpus tests.")
class TestGoldenCorpus:
    """Validates programmatic generation of the 5-archetype Golden Benchmark Suite."""

    def test_create_golden_corpus_archetypes(self) -> None:
        corpus = create_golden_corpus()
        assert len(corpus) == 5

        expected_names = [
            "article_standard",
            "financial_table",
            "executive_summary",
            "bulleted_memo",
            "subset_font_stress",
        ]
        actual_names = [item.name for item in corpus]
        assert actual_names == expected_names

        for item in corpus:
            assert isinstance(item, CorpusItem)
            assert item.pdf_bytes.startswith(b"%PDF")
            assert len(item.pdf_bytes) > 200
            assert len(item.reference_doc.pages) >= 1
            assert len(item.reference_doc.pages[0].blocks) >= 3

    def test_financial_table_structure(self) -> None:
        corpus = create_golden_corpus()
        fin_item = next(item for item in corpus if item.name == "financial_table")
        ref_blocks = fin_item.reference_doc.pages[0].blocks

        tables = [b for b in ref_blocks if isinstance(b, Table)]
        assert len(tables) == 1
        tbl = tables[0]
        assert len(tbl.rows) == 3
        assert len(tbl.rows[0].cells) == 3
        assert tbl.rows[0].is_header is True

    def test_executive_summary_alignments(self) -> None:
        corpus = create_golden_corpus()
        exec_item = next(item for item in corpus if item.name == "executive_summary")
        ref_blocks = exec_item.reference_doc.pages[0].blocks

        alignments = [b.alignment for b in ref_blocks]
        assert "center" in alignments
        assert "right" in alignments
        assert "left" in alignments

    def test_subset_font_stress_references(self) -> None:
        corpus = create_golden_corpus()
        font_item = next(item for item in corpus if item.name == "subset_font_stress")
        ref_blocks = font_item.reference_doc.pages[0].blocks

        fonts = [r.font for b in ref_blocks for r in b.runs]
        assert any("BAAAAA+" in f for f in fonts)
        assert any("XYZABC+" in f for f in fonts)


@pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF is required for Golden Benchmark tests.")
class TestGoldenBenchmark:
    """Validates execution of the audited multi-gate benchmark runner."""

    def test_run_golden_benchmark_full_pass(self, tmp_path: Path) -> None:
        report_file = tmp_path / "test_benchmark_report.json"
        report = run_golden_benchmark(save_report_path=report_file)

        assert isinstance(report, GoldenBenchmarkReport)
        assert report.total_documents == 5
        # Strict user gate requirement: 100% text integrity pass rate
        assert report.text_integrity_pass_rate == 1.0
        # All 5 archetypes must pass all gates
        assert report.overall_pass_rate == 1.0
        # Average composite score must exceed 95%
        assert report.average_composite_score >= 0.95

        # Check document details
        for doc_res in report.documents:
            assert isinstance(doc_res, DocumentBenchmarkResult)
            assert doc_res.passed is True
            assert doc_res.diff_count == 0
            assert doc_res.composite_score >= 0.95
            assert doc_res.gate_scores["Gate 1: Text Integrity"] == 1.0
            assert doc_res.gate_pass_status["Gate 1: Text Integrity"] is True

        # Check saved JSON
        assert report_file.exists()
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_documents"] == 5
        assert data["text_integrity_pass_rate"] == 1.0
        assert data["overall_pass_rate"] == 1.0

    def test_calibrated_profile_parameters(self) -> None:
        profile_path = Path(__file__).resolve().parent.parent / "calibrated_word_profile.json"
        assert profile_path.exists(), "Calibrated word profile must exist."

        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "policy" in data
        policy_dict = data["policy"]
        for param, (low, high) in PARAM_BOUNDS.items():
            assert param in policy_dict
            val = policy_dict[param]
            assert low <= val <= high, f"Parameter {param}={val} out of bounds [{low}, {high}]"

    def test_multi_document_corpus_evaluator(self) -> None:
        corpus = create_golden_corpus()
        evaluator = MultiDocumentCorpusEvaluator(corpus)
        assert evaluator.deterministic is True

        genome = ProfileGenome()
        fit_res = evaluator.evaluate(genome)
        assert fit_res.score >= 0.95
        assert "mean_score" in fit_res.artifacts
        assert len(fit_res.artifacts["document_scores"]) == 5
        # Sub scores check
        assert any("Gate 1: Text Integrity" in k for k in fit_res.sub_scores.keys())
