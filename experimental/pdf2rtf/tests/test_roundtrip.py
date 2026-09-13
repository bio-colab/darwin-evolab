"""Roundtrip verification tests: IR -> RTF Emitter -> RTF Parser -> IR."""
from __future__ import annotations

import pytest

from experimental.pdf2rtf.ir import Color, Document, Paragraph, Run, Table
from experimental.pdf2rtf.rtf_emitter import emit_rtf
from experimental.pdf2rtf.rtf_parser import parse_rtf


def test_roundtrip_multi_paragraph_and_styles():
    original_doc = Document()
    page = original_doc.add_page()

    p1 = page.add_paragraph(alignment="center", space_after_pt=14.0)
    p1.add_run("Executive Summary", font="Calibri", font_size_pt=16.0, bold=True)

    p2 = page.add_paragraph(alignment="left", space_before_pt=6.0, space_after_pt=6.0)
    p2.add_run("This is a standard paragraph with ", font="Calibri", font_size_pt=11.0)
    p2.add_run("italicized emphasis", font="Calibri", font_size_pt=11.0, italic=True)
    p2.add_run(" and normal text.", font="Calibri", font_size_pt=11.0)

    # 1. Emit to RTF
    rtf_output = emit_rtf(original_doc)

    # 2. Parse back to IR
    restored_doc = parse_rtf(rtf_output)

    # 3. Assert fidelity
    assert len(restored_doc.pages) == 1
    assert len(restored_doc.pages[0].blocks) == 2

    rp1 = restored_doc.pages[0].blocks[0]
    assert isinstance(rp1, Paragraph)
    assert rp1.plain_text.strip() == "Executive Summary"
    assert rp1.alignment == "center"
    assert rp1.runs[0].bold is True
    assert rp1.runs[0].font_size_pt == 16.0

    rp2 = restored_doc.pages[0].blocks[1]
    assert isinstance(rp2, Paragraph)
    assert "This is a standard paragraph with italicized emphasis and normal text." in rp2.plain_text
    italic_runs = [r for r in rp2.runs if r.italic]
    assert len(italic_runs) >= 1
    assert "italicized emphasis" in italic_runs[0].text


def test_roundtrip_table_fidelity():
    original_doc = Document()
    page = original_doc.add_page()

    tbl = page.add_table(alignment="center")
    r1 = tbl.add_row(is_header=True)
    r1.add_cell("Metric", width_twips=1500)
    r1.add_cell("Score", width_twips=1500)

    r2 = tbl.add_row()
    r2.add_cell("Accuracy", width_twips=1500)
    r2.add_cell("99.8%", width_twips=1500)

    rtf_output = emit_rtf(original_doc)
    restored_doc = parse_rtf(rtf_output)

    assert len(restored_doc.all_tables) == 1
    restored_table = restored_doc.all_tables[0]
    assert restored_table.row_count == 2
    assert restored_table.col_count == 2

    # Check cell plain text
    row1 = restored_table.rows[0]
    assert row1.cells[0].plain_text.strip() == "Metric"
    assert row1.cells[1].plain_text.strip() == "Score"

    row2 = restored_table.rows[1]
    assert row2.cells[0].plain_text.strip() == "Accuracy"
    assert row2.cells[1].plain_text.strip() == "99.8%"
