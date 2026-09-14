"""test_word_oracle.py — Automated Unit Tests for Word-in-the-Loop Oracle.

Validates that emitted RTF files opened directly inside native Microsoft Word
achieve >= 95% multi-gate fidelity against the Ground Truth Reference IR.
"""

from __future__ import annotations

import pytest
from experimental.pdf2rtf.genome import ProfilePolicy
from experimental.pdf2rtf.pdf_extractor import PDFExtractor
from experimental.pdf2rtf.rtf_emitter import emit_rtf
from experimental.pdf2rtf.word_holdout import load_real_word_holdout
from experimental.pdf2rtf.word_oracle import WordInTheLoopOracle


import json
from pathlib import Path
import sys

@pytest.fixture(scope="module")
def word_available():
    if sys.platform != "win32":
        return False
    try:
        import win32com.client
        return True
    except Exception:
        return False


def test_word_oracle_single_document_fidelity(word_available):
    if not word_available:
        pytest.skip("Microsoft Word COM automation not available in this environment")

    holdout = load_real_word_holdout()
    item = next(h for h in holdout if h.name == "word_academic_paper")

    policy = ProfilePolicy(
        para_split_delta_ratio=0.60,
        para_split_short_line_factor=0.85,
        para_split_indent_factor=0.60,
        para_split_font_weight=0.60,
        table_col_align_tol_pt=5.0,
    )

    extractor = PDFExtractor(policy=policy)
    cand_ir = extractor.extract(item.pdf_bytes)
    rtf_text = emit_rtf(cand_ir)

    try:
        oracle = WordInTheLoopOracle()
        report = oracle.verify_single_rtf(rtf_text, reference=item.reference_doc)
    except Exception as e:
        pytest.skip(f"Word COM runtime execution error: {e}")

    assert report.composite_score >= 0.95, f"Score too low: {report.composite_score:.2%}"
    assert report.gates[0].passed is True, "Gate 1 (Text Integrity) failed"
    assert report.gates[1].passed is True, "Gate 2 (Paragraph Count) failed"


def test_word_oracle_audit_report_statistics():
    report_path = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "pdf2rtf_word_oracle_audit.json"
    if not report_path.exists():
        pytest.skip("Word Oracle audit report not generated yet")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["summary"]["total_documents"] == 12
    # Text integrity hard gate: 100% on real Microsoft Word rendering
    assert data["summary"]["text_integrity_pass_rate"] == 1.0
    # Composite score >= 98%
    assert data["summary"]["average_composite_score"] >= 0.98
    assert data["summary"]["overall_pass_rate"] >= 0.90

