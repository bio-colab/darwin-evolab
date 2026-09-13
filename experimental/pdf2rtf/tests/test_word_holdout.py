"""test_word_holdout.py — Automated Unit Tests for Genuine Microsoft Word Holdout Evaluation.

Audits the independent holdout suite generated directly by Microsoft Word COM automation,
confirming 100% text integrity and multi-gate fidelity exceeding 95%.
"""

from __future__ import annotations

import pytest
from experimental.pdf2rtf.benchmark import run_holdout_benchmark
from experimental.pdf2rtf.word_holdout import load_real_word_holdout


def test_load_real_word_holdout_presence_and_types():
    items = load_real_word_holdout()
    assert len(items) == 12

    expected_names = {
        "word_academic_paper",
        "word_financial_report",
        "word_executive_letter",
        "word_styled_article",
        "word_technical_spec",
        "word_legal_contract",
        "word_medical_summary",
        "word_corporate_newsletter",
        "word_formal_invoice",
        "word_scientific_abstract",
        "word_multi_page_report",
        "word_tabular_matrix",
    }
    actual_names = {item.name for item in items}
    assert actual_names == expected_names

    for item in items:
        assert len(item.pdf_bytes) > 10_000, f"{item.name} PDF bytes too small"
        assert len(item.reference_doc.pages) >= 1
        assert len(item.reference_doc.pages[0].blocks) >= 2


def test_real_word_holdout_benchmark_pass_rate_and_fidelity():
    report = run_holdout_benchmark()

    assert report.total_documents == 12
    # Hard gate: 100% text integrity on genuine Microsoft Word PDFs
    assert report.text_integrity_pass_rate == 1.0
    # Hard gate: 100% overall multi-gate pass rate
    assert report.overall_pass_rate == 1.0
    # Hard gate: High fidelity >= 95%
    assert report.average_composite_score >= 0.95
    # Actual achieved is ~99.11%
    assert report.average_composite_score >= 0.98


def test_individual_word_holdout_gate_fidelities():
    report = run_holdout_benchmark()

    for doc in report.documents:
        assert doc.passed is True, f"Document {doc.name} failed verification: {doc.message}"
        assert doc.composite_score >= 0.95

        # Gate 1: Text Integrity must be flawless
        assert doc.gate_scores["Gate 1: Text Integrity"] == 1.0
        assert doc.gate_pass_status["Gate 1: Text Integrity"] is True
        assert doc.diff_count == 0

        # Gate 2: Structure Integrity must be >= 0.90
        assert doc.gate_scores["Gate 2: Structure Integrity"] >= 0.90
        assert doc.gate_pass_status["Gate 2: Structure Integrity"] is True

        # Gate 2.5: Table Oracle
        assert doc.gate_scores["Gate 2.5: Table Oracle"] >= 0.90
        assert doc.gate_pass_status["Gate 2.5: Table Oracle"] is True

        # Gate 3: Formatting Integrity must be >= 0.90
        assert doc.gate_scores["Gate 3: Formatting Integrity"] >= 0.90
        assert doc.gate_pass_status["Gate 3: Formatting Integrity"] is True

        # Gate 4: Visual / Geometry Oracle must be >= 0.90
        assert doc.gate_scores["Gate 4: Visual / Geometry Oracle"] >= 0.90
        assert doc.gate_pass_status["Gate 4: Visual / Geometry Oracle"] is True


def test_multiseed_evaluation_mini_run():
    from experimental.pdf2rtf.statistical_eval import run_multiseed_evaluation
    # Test a mini 2-seed run to verify statistical pipeline contract
    res = run_multiseed_evaluation(num_seeds=2, population_size=4, generations=1)
    assert res["num_seeds"] == 2
    assert res["summary"]["mean_holdout_composite"] >= 0.95
    assert res["summary"]["text_integrity_pass_rate"] == 1.0


def test_ablation_study_mini_run():
    from experimental.pdf2rtf.ablation import run_ablation_study
    # Test a mini ablation with 3 random samples
    res = run_ablation_study(num_random_samples=3)
    assert "champion_performance" in res
    assert "random_baseline" in res
    assert "systematic_ablations" in res
    assert res["champion_performance"]["composite_score"] >= 0.98

