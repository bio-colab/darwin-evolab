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
    assert len(items) == 4

    expected_names = {
        "word_academic_paper",
        "word_financial_report",
        "word_executive_letter",
        "word_styled_article",
    }
    actual_names = {item.name for item in items}
    assert actual_names == expected_names

    for item in items:
        assert len(item.pdf_bytes) > 10_000, f"{item.name} PDF bytes too small"
        assert len(item.reference_doc.pages) >= 1
        assert len(item.reference_doc.pages[0].blocks) >= 3


def test_real_word_holdout_benchmark_pass_rate_and_fidelity():
    report = run_holdout_benchmark()

    assert report.total_documents == 4
    # Hard gate: 100% text integrity on genuine Microsoft Word PDFs
    assert report.text_integrity_pass_rate == 1.0
    # Hard gate: 100% overall multi-gate pass rate
    assert report.overall_pass_rate == 1.0
    # Hard gate: High fidelity >= 95%
    assert report.average_composite_score >= 0.95
    # Actual achieved is ~98.90%
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
        assert doc.gate_scores["Gate 2.5: Table Oracle"] == 1.0
        assert doc.gate_pass_status["Gate 2.5: Table Oracle"] is True

        # Gate 3: Formatting Integrity must be >= 0.90
        assert doc.gate_scores["Gate 3: Formatting Integrity"] >= 0.90
        assert doc.gate_pass_status["Gate 3: Formatting Integrity"] is True

        # Gate 4: Visual / Geometry Oracle must be >= 0.90
        assert doc.gate_scores["Gate 4: Visual / Geometry Oracle"] >= 0.90
        assert doc.gate_pass_status["Gate 4: Visual / Geometry Oracle"] is True
