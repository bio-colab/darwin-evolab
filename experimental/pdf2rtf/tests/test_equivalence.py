"""Tests for Multi-Gate Equivalence Verifier in pdf2rtf."""

from __future__ import annotations

import copy
import pytest

from experimental.pdf2rtf.equivalence import (
    EquivalenceReport,
    FormattingIntegrityGate,
    GateResult,
    MultiGateVerifier,
    StructureIntegrityGate,
    TableOracle,
    TextIntegrityGate,
    VisualDiffOracle,
)
from experimental.pdf2rtf.ir import (
    Cell,
    Color,
    Document,
    Page,
    Paragraph,
    Row,
    Run,
    Table,
)


def _build_sample_document() -> Document:
    """Helper to construct a representative multi-element document."""
    doc = Document()
    page = Page(number=1, width_pts=612.0, height_pts=792.0)

    # Paragraph 1: Title
    p1 = Paragraph(alignment="center", space_after_pt=12.0)
    p1.runs.append(Run(text="Hardened Document Evaluation", font="Calibri", font_size_pt=16.0, bold=True))
    page.blocks.append(p1)

    # Paragraph 2: Formatted text
    p2 = Paragraph(alignment="left", space_before_pt=4.0, space_after_pt=8.0)
    p2.runs.append(Run(text="This is a ", font="Calibri", font_size_pt=11.0))
    p2.runs.append(Run(text="bold test ", font="Calibri", font_size_pt=11.0, bold=True))
    p2.runs.append(Run(text="with italic emphasis.", font="Calibri", font_size_pt=11.0, italic=True))
    page.blocks.append(p2)

    # Table: 2 rows x 2 cols
    t = Table()
    r1 = Row(is_header=True)
    r1.cells.append(Cell([Paragraph([Run("Metric", font="Calibri", font_size_pt=10.0, bold=True)])], width_twips=3000))
    r1.cells.append(Cell([Paragraph([Run("Value", font="Calibri", font_size_pt=10.0, bold=True)])], width_twips=3000))
    t.rows.append(r1)

    r2 = Row()
    r2.cells.append(Cell([Paragraph([Run("Accuracy", font="Calibri", font_size_pt=10.0)])], width_twips=3000))
    r2.cells.append(Cell([Paragraph([Run("99.9%", font="Calibri", font_size_pt=10.0)])], width_twips=3000))
    t.rows.append(r2)

    page.blocks.append(t)
    doc.pages.append(page)
    return doc


def test_identical_documents_pass_all_gates():
    doc1 = _build_sample_document()
    doc2 = _build_sample_document()

    verifier = MultiGateVerifier()
    report = verifier.verify(candidate=doc2, reference=doc1)

    assert report.passed is True
    assert report.cascaded_failure is False
    assert report.composite_score == pytest.approx(1.0, abs=1e-3)
    assert len(report.gates) == 5
    for gate in report.gates:
        assert gate.passed is True
        assert gate.score == pytest.approx(1.0, abs=1e-3)


def test_text_integrity_gate_detects_omissions_and_cascades():
    ref_doc = _build_sample_document()
    # Create candidate with missing text in paragraph 2
    cand_doc = _build_sample_document()
    p2 = cand_doc.pages[0].blocks[1]
    assert isinstance(p2, Paragraph)
    # Remove one of the runs, omitting words
    p2.runs.pop()

    verifier = MultiGateVerifier()
    report = verifier.verify(candidate=cand_doc, reference=ref_doc)

    assert report.passed is False
    assert report.cascaded_failure is True
    # Composite score must be heavily suppressed
    assert report.composite_score < 0.35
    g1 = report.get_gate("Gate 1: Text Integrity")
    assert g1 is not None
    assert g1.passed is False
    assert g1.details["diff_count"] > 0


def test_structure_gate_detects_alignment_and_block_differences():
    ref_doc = _build_sample_document()
    cand_doc = _build_sample_document()

    # Change alignment of first paragraph from center to right
    p1 = cand_doc.pages[0].blocks[0]
    assert isinstance(p1, Paragraph)
    p1.alignment = "right"

    gate = StructureIntegrityGate(pass_threshold=0.95)
    result = gate.evaluate(candidate=cand_doc, reference=ref_doc)

    assert result.details["align_score"] < 1.0
    assert result.score < 1.0


def test_table_oracle_precision_recall_and_grid():
    ref_doc = _build_sample_document()
    cand_doc = _build_sample_document()

    # Mutate cell content in candidate table
    table = cand_doc.pages[0].blocks[2]
    assert isinstance(table, Table)
    table.rows[1].cells[1].paragraphs[0].runs[0].text = "WrongValue"

    oracle = TableOracle(pass_threshold=0.95)
    result = oracle.evaluate(candidate=cand_doc, reference=ref_doc)

    assert result.passed is False
    assert result.details["cell_precision"] < 1.0
    assert result.details["cell_recall"] < 1.0
    assert result.score < 0.95


def test_table_oracle_missing_or_extra_tables():
    doc_with_table = _build_sample_document()
    doc_no_table = _build_sample_document()
    doc_no_table.pages[0].blocks = [b for b in doc_no_table.pages[0].blocks if not isinstance(b, Table)]

    oracle = TableOracle()

    # Missing table when expected
    r_missing = oracle.evaluate(candidate=doc_no_table, reference=doc_with_table)
    assert r_missing.passed is False
    assert r_missing.score == 0.0

    # Spurious table when unexpected
    r_spurious = oracle.evaluate(candidate=doc_with_table, reference=doc_no_table)
    assert r_spurious.passed is False
    assert r_spurious.score == 0.0

    # Neither has table -> passes trivially
    r_empty = oracle.evaluate(candidate=doc_no_table, reference=doc_no_table)
    assert r_empty.passed is True
    assert r_empty.score == 1.0


def test_formatting_gate_font_subset_normalization_and_tolerances():
    ref_doc = Document()
    page1 = Page(1, 612.0, 792.0)
    # Reference font has Word subset prefix
    p1 = Paragraph()
    p1.runs.append(Run(text="Hello", font="BAAAAA+Calibri", font_size_pt=12.0, bold=True))
    page1.blocks.append(p1)
    ref_doc.pages.append(page1)

    cand_doc = Document()
    page2 = Page(1, 612.0, 792.0)
    # Candidate font has clean name, and font size differs by 0.3 pt (within 0.5pt tolerance)
    p2 = Paragraph()
    p2.runs.append(Run(text="Hello", font="Calibri", font_size_pt=12.3, bold=True))
    page2.blocks.append(p2)
    cand_doc.pages.append(page2)

    gate = FormattingIntegrityGate(size_tolerance_pt=0.5)
    result = gate.evaluate(candidate=cand_doc, reference=ref_doc)

    assert result.passed is True
    assert result.details["font_fidelity"] == 1.0
    assert result.details["size_fidelity"] == 1.0
    assert result.details["style_fidelity"] == 1.0
    assert result.score == pytest.approx(1.0, abs=1e-3)


def test_formatting_gate_detects_style_mismatch():
    ref_doc = Document()
    p1 = Page(1, 612.0, 792.0)
    para1 = Paragraph([Run(text="Text", font="Arial", font_size_pt=10.0, bold=True, italic=False)])
    p1.blocks.append(para1)
    ref_doc.pages.append(p1)

    cand_doc = Document()
    p2 = Page(1, 612.0, 792.0)
    # Candidate forgot bold and made it italic
    para2 = Paragraph([Run(text="Text", font="Arial", font_size_pt=10.0, bold=False, italic=True)])
    p2.blocks.append(para2)
    cand_doc.pages.append(p2)

    gate = FormattingIntegrityGate()
    result = gate.evaluate(candidate=cand_doc, reference=ref_doc)

    assert result.details["style_fidelity"] < 0.5
    assert result.score < 0.90


def test_visual_geometry_oracle_calibrated_noise_floor():
    doc1 = Document([Page(1, 612.0, 792.0)])
    # 2pt delta in page height (below 5% noise floor epsilon)
    doc2 = Document([Page(1, 612.0, 790.0)])

    oracle = VisualDiffOracle(epsilon_floor=0.05)
    result = oracle.evaluate(candidate=doc2, reference=doc1)

    assert result.passed is True
    # Calibrated error is zeroed by noise floor
    assert result.score == 1.0


def test_equivalence_report_serialization():
    doc = _build_sample_document()
    verifier = MultiGateVerifier()
    report = verifier.verify(candidate=doc, reference=doc)

    d = report.to_dict()
    assert d["passed"] is True
    assert d["composite_score"] == 1.0
    assert len(d["gates"]) == 5
    assert d["gates"][0]["name"] == "Gate 1: Text Integrity"
    assert "token_similarity" in d["gates"][0]["details"]


def test_end_to_end_rtf_roundtrip_verification():
    from experimental.pdf2rtf.rtf_emitter import emit_rtf
    from experimental.pdf2rtf.rtf_parser import parse_rtf

    orig_doc = _build_sample_document()
    rtf_code = emit_rtf(orig_doc)
    reconstructed_doc = parse_rtf(rtf_code)

    verifier = MultiGateVerifier()
    report = verifier.verify(candidate=reconstructed_doc, reference=orig_doc)

    assert report.passed is True
    assert report.cascaded_failure is False
    # Text and structures must be retained
    g1 = report.get_gate("Gate 1: Text Integrity")
    assert g1 is not None and g1.passed is True
    g2 = report.get_gate("Gate 2: Structure Integrity")
    assert g2 is not None and g2.passed is True
    g25 = report.get_gate("Gate 2.5: Table Oracle")
    assert g25 is not None and g25.passed is True

