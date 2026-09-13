"""Unit tests for Pure Python RTF Emitter."""
from __future__ import annotations

import pytest

from experimental.pdf2rtf.ir import Color, Document, Paragraph, Run, Table
from experimental.pdf2rtf.rtf_emitter import RTFEmitter, emit_rtf


def test_emit_minimal_document():
    doc = Document()
    page = doc.add_page()
    p = page.add_paragraph("Hello world!")

    rtf_str = emit_rtf(doc)

    assert rtf_str.startswith(r"{\rtf1\ansi")
    assert r"{\fonttbl" in rtf_str
    assert "Calibri" in rtf_str
    assert "Hello world!" in rtf_str
    assert rtf_str.endswith("}\n") or rtf_str.endswith("}")


def test_emit_formatted_runs_and_styles():
    doc = Document()
    page = doc.add_page()
    p = page.add_paragraph(alignment="center", space_after_pt=10.0)
    p.add_run("Bold text ", bold=True)
    p.add_run("Italic text ", italic=True)
    p.add_run("Underlined text", underline=True)

    rtf_str = emit_rtf(doc)

    assert r"\qc" in rtf_str
    assert r"\sa200" in rtf_str  # 10pt * 20 = 200 twips
    assert r"\b" in rtf_str
    assert r"\i" in rtf_str
    assert r"\ul" in rtf_str


def test_emit_colors_and_fonts():
    doc = Document()
    page = doc.add_page()
    p = page.add_paragraph()
    p.add_run("Blue text in Arial", font="Arial", color=Color(0, 0, 255))

    rtf_str = emit_rtf(doc)

    assert r"{\colortbl" in rtf_str
    assert r"\red0\green0\blue255;" in rtf_str
    assert "Arial" in rtf_str
    assert r"\cf1" in rtf_str


def test_emit_unicode_and_escaped_characters():
    doc = Document()
    page = doc.add_page()
    p = page.add_paragraph()
    p.add_run(r"Formula: {x \ y} & café — 100% €")

    rtf_str = emit_rtf(doc)

    # Escaped curly braces and backslashes
    assert r"\{" in rtf_str
    assert r"\}" in rtf_str
    assert r"\\" in rtf_str
    # Unicode é or €
    assert r"\u" in rtf_str


def test_emit_table_grid():
    doc = Document()
    page = doc.add_page()
    tbl = page.add_table(alignment="center")

    r1 = tbl.add_row(is_header=True)
    r1.add_cell("Col A", width_twips=1000)
    r1.add_cell("Col B", width_twips=2000)

    r2 = tbl.add_row()
    r2.add_cell("123", width_twips=1000)
    r2.add_cell("456", width_twips=2000)

    rtf_str = emit_rtf(doc)

    assert r"\trowd" in rtf_str
    assert r"\cellx1000" in rtf_str
    assert r"\cellx3000" in rtf_str
    assert r"\intbl" in rtf_str
    assert r"\cell" in rtf_str
    assert r"\row" in rtf_str
