"""test_corpus_54.py — Automated Unit Tests for the Evolab-54 Orthogonal Benchmark Suite.

Validates that all 54 documents in the orthogonal benchmark battery achieve
100% text integrity and high multi-gate fidelity exceeding 95% across all 6 categories.
"""

from __future__ import annotations

import pytest
from experimental.pdf2rtf.word_corpus_54 import load_corpus_54, run_benchmark_54


def test_corpus_54_loading_and_group_distribution():
    items = load_corpus_54()
    assert len(items) == 54

    group_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    for item in items:
        assert item.group in group_counts, f"Unexpected group: {item.group}"
        group_counts[item.group] += 1
        assert len(item.pdf_bytes) > 5_000, f"{item.name} PDF bytes too small"
        assert len(item.reference_doc.pages) >= 1
        assert len(item.reference_doc.pages[0].blocks) >= 1

    assert group_counts["A"] == 12, "Group A must contain 12 files"
    assert group_counts["B"] == 12, "Group B must contain 12 files"
    assert group_counts["C"] == 6,  "Group C must contain 6 files"
    assert group_counts["D"] == 12, "Group D must contain 12 files"
    assert group_counts["E"] == 12, "Group E must contain 12 files"


def test_corpus_54_benchmark_fidelity_and_gates():
    report = run_benchmark_54(save_report=False)

    assert report["total_documents"] == 54
    # Hard gate: 100% text integrity across all 54 documents
    assert report["summary"]["text_integrity_pass_rate"] == 1.0
    # Average composite fidelity >= 98%
    assert report["summary"]["average_composite_score"] >= 0.98

    # Verify group breakdowns
    for grp in ["A", "B", "C", "D", "E"]:
        grp_summary = report["groups"][grp]
        assert grp_summary["text_integrity_pass_rate"] == 1.0, f"Group {grp} failed text integrity"
        assert grp_summary["average_composite_score"] >= 0.95, f"Group {grp} score too low: {grp_summary['average_composite_score']}"
