"""Unit tests for PyMuPDF Extractor and Font Normalization."""
from __future__ import annotations

import fitz
import pytest

from experimental.pdf2rtf.pdf_extractor import (
    PDFExtractor,
    clean_base_font_family,
    extract_pdf,
    strip_font_subset_prefix,
)


def test_strip_font_subset_prefix():
    assert strip_font_subset_prefix("BAAAAA+Calibri") == "Calibri"
    assert strip_font_subset_prefix("XYZABC+TimesNewRoman") == "TimesNewRoman"
    assert strip_font_subset_prefix("ABCDEF+Arial-Bold") == "Arial-Bold"
    assert strip_font_subset_prefix("Calibri") == "Calibri"
    assert strip_font_subset_prefix("") == "Calibri"


def test_clean_base_font_family():
    assert clean_base_font_family("BAAAAA+Calibri-Bold") == "Calibri"
    assert clean_base_font_family("XYZABC+TimesNewRoman,Italic") == "TimesNewRoman"
    assert clean_base_font_family("Arial_Black") == "Arial"
    assert clean_base_font_family("Georgia") == "Georgia"


def test_in_memory_pdf_extraction():
    # Build a clean PDF in memory using fitz
    fitz_doc = fitz.open()
    page = fitz_doc.new_page(width=612.0, height=792.0)

    # Insert header and body text
    page.insert_text((72.0, 72.0), "Document Heading", fontname="helv", fontsize=16.0)
    page.insert_text((72.0, 100.0), "First paragraph of the test document.", fontname="times-roman", fontsize=11.0)
    page.insert_text((72.0, 130.0), "Second paragraph with more content.", fontname="times-roman", fontsize=11.0)

    pdf_bytes = fitz_doc.tobytes()
    fitz_doc.close()

    # Extract Document IR
    extractor = PDFExtractor()
    doc_ir = extractor.extract(pdf_bytes)

    assert len(doc_ir.pages) == 1
    page_ir = doc_ir.pages[0]
    assert page_ir.width_pts == 612.0
    assert page_ir.height_pts == 792.0

    plain = doc_ir.plain_text
    assert "Document Heading" in plain
    assert "First paragraph of the test document." in plain
    assert "Second paragraph with more content." in plain
