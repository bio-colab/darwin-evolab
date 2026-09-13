"""Unit tests for Document Intermediate Representation (IR)."""
from __future__ import annotations

import pytest

from experimental.pdf2rtf.ir import (
    Color,
    Document,
    Page,
    Paragraph,
    Row,
    Cell,
    Run,
    Table,
)


def test_color_bounds_and_properties():
    c1 = Color(r=300, g=-10, b=128)
    assert c1.r == 255
    assert c1.g == 0
    assert c1.b == 128
    assert c1.is_black is False

    black = Color(0, 0, 0)
    assert black.is_black is True


def test_run_font_subset_stripping():
    r1 = Run(text="Hello", font="BAAAAA+Calibri", font_size_pt=12.0, bold=True)
    assert r1.clean_font_name() == "Calibri"

    r2 = Run(text="World", font="TimesNewRoman")
    assert r2.clean_font_name() == "TimesNewRoman"


def test_paragraph_and_plain_text():
    p = Paragraph(alignment="center")
    p.add_run("Hello ", bold=True)
    p.add_run("World!", italic=True)

    assert p.plain_text == "Hello World!"
    assert len(p.runs) == 2
    assert p.runs[0].bold is True
    assert p.runs[1].italic is True


def test_table_dimensions_and_cells():
    tbl = Table(alignment="center")
    r1 = tbl.add_row(is_header=True)
    c1 = r1.add_cell("Header 1", width_twips=2000)
    c2 = r1.add_cell("Header 2", width_twips=3000)

    r2 = tbl.add_row()
    r2.add_cell("Data 1", width_twips=2000)
    r2.add_cell("Data 2", width_twips=3000)

    assert tbl.row_count == 2
    assert tbl.col_count == 2
    assert "Header 1 | Header 2" in tbl.plain_text
    assert "Data 1 | Data 2" in tbl.plain_text


def test_document_serialization_roundtrip():
    doc = Document(metadata={"title": "Test Doc"})
    page = doc.add_page(width_pts=612.0, height_pts=792.0)
    p = page.add_paragraph(alignment="justify", space_after_pt=12.0)
    p.add_run("First line of text. ", font="Calibri", font_size_pt=11.0)
    p.add_run("Highlighted in red.", font="Calibri", color=Color(255, 0, 0))

    tbl = page.add_table()
    row = tbl.add_row()
    row.add_cell("Cell 1")
    row.add_cell("Cell 2")

    # Serialize to dict and restore
    d = doc.to_dict()
    restored = Document.from_dict(d)

    assert restored.metadata["title"] == "Test Doc"
    assert len(restored.pages) == 1
    assert len(restored.pages[0].blocks) == 2
    assert restored.plain_text == doc.plain_text
    assert restored.fonts_used == {"Calibri"}
    assert Color(255, 0, 0) in restored.colors_used
