"""corpus.py — Golden Corpus Benchmark Suite for PDF-to-RTF Hardening.

Provides programmatically generated, deterministic test documents representing
common Microsoft Word document archetypes (articles, financial tables,
mixed-alignment executive summaries, bulleted lists, and font-subset stress tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

from .ir import Cell, Color, Document, Page, Paragraph, Row, Run, Table


@dataclass
class CorpusItem:
    """A test case in the Golden Corpus containing source PDF bytes and matching reference Document IR."""

    name: str
    description: str
    pdf_bytes: bytes
    reference_doc: Document


def _build_article_corpus() -> CorpusItem:
    """Archetype 1: Standard structured article with title, subheadings, and formatted body."""
    pdf = fitz.open()
    page = pdf.new_page(width=612, height=792)

    page.insert_text((72, 72), "An Overview of Evolutionary Computing", fontsize=16, fontname="helv")
    page.insert_text((72, 110), "Section 1: Foundations and Principles", fontsize=13, fontname="helv")
    page.insert_text((72, 140), "Evolutionary algorithms mimic natural selection to discover robust solutions.", fontsize=11, fontname="helv")
    page.insert_text((72, 180), "Through iterative cycles of variation and selection, candidate solutions adapt.", fontsize=11, fontname="helv")

    pdf_bytes = pdf.tobytes()
    pdf.close()

    doc = Document()
    p = Page(1, 612.0, 792.0)
    p.add_paragraph("An Overview of Evolutionary Computing", font="Helvetica", font_size_pt=16.0, bold=True)
    p.add_paragraph("Section 1: Foundations and Principles", font="Helvetica", font_size_pt=13.0, bold=True, space_before_pt=19.0)
    p.add_paragraph("Evolutionary algorithms mimic natural selection to discover robust solutions.", font="Helvetica", font_size_pt=11.0, space_before_pt=14.5)
    p.add_paragraph("Through iterative cycles of variation and selection, candidate solutions adapt.", font="Helvetica", font_size_pt=11.0, space_before_pt=25.0)
    doc.pages.append(p)

    return CorpusItem(
        name="article_standard",
        description="Structured multi-section article with headings and formatted paragraphs.",
        pdf_bytes=pdf_bytes,
        reference_doc=doc,
    )


def _build_financial_table_corpus() -> CorpusItem:
    """Archetype 2: Financial report with text intro and a multi-column data table."""
    pdf = fitz.open()
    page = pdf.new_page(width=612, height=792)

    page.insert_text((72, 72), "Quarterly Financial Performance", fontsize=14, fontname="helv")
    page.insert_text((72, 100), "The following table summarizes revenue and operating margins:", fontsize=11, fontname="helv")

    # Draw table bounding box and internal grid lines
    page.draw_rect(fitz.Rect(72, 120, 540, 210), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(72, 150), fitz.Point(540, 150), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(72, 180), fitz.Point(540, 180), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(220, 120), fitz.Point(220, 210), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(380, 120), fitz.Point(380, 210), color=(0, 0, 0), width=0.5)

    # Header row
    page.insert_text((80, 142), "Quarter", fontsize=10, fontname="helv")
    page.insert_text((230, 142), "Revenue", fontsize=10, fontname="helv")
    page.insert_text((390, 142), "Profit", fontsize=10, fontname="helv")

    # Data row 1
    page.insert_text((80, 172), "Q1", fontsize=10, fontname="helv")
    page.insert_text((230, 172), "USD 1200000", fontsize=10, fontname="helv")
    page.insert_text((390, 172), "USD 340000", fontsize=10, fontname="helv")

    # Data row 2
    page.insert_text((80, 202), "Q2", fontsize=10, fontname="helv")
    page.insert_text((230, 202), "USD 1450000", fontsize=10, fontname="helv")
    page.insert_text((390, 202), "USD 410000", fontsize=10, fontname="helv")

    pdf_bytes = pdf.tobytes()
    pdf.close()

    doc = Document()
    p = Page(1, 612.0, 792.0)
    p.add_paragraph("Quarterly Financial Performance", font="Helvetica", font_size_pt=14.0, bold=True)
    p.add_paragraph("The following table summarizes revenue and operating margins:", font="Helvetica", font_size_pt=11.0, space_before_pt=12.0)

    # Reference Table
    tbl = Table(alignment="left")
    # Row 1 (Header)
    r1 = Row(is_header=True)
    r1.cells.append(Cell([Paragraph([Run("Quarter", font="Helvetica", font_size_pt=10.0, bold=True)])], width_twips=2880))
    r1.cells.append(Cell([Paragraph([Run("Revenue", font="Helvetica", font_size_pt=10.0, bold=True)])], width_twips=2880))
    r1.cells.append(Cell([Paragraph([Run("Profit", font="Helvetica", font_size_pt=10.0, bold=True)])], width_twips=2880))
    tbl.rows.append(r1)

    # Row 2
    r2 = Row()
    r2.cells.append(Cell([Paragraph([Run("Q1", font="Helvetica", font_size_pt=10.0)])], width_twips=2880))
    r2.cells.append(Cell([Paragraph([Run("USD 1200000", font="Helvetica", font_size_pt=10.0)])], width_twips=2880))
    r2.cells.append(Cell([Paragraph([Run("USD 340000", font="Helvetica", font_size_pt=10.0)])], width_twips=2880))
    tbl.rows.append(r2)

    # Row 3
    r3 = Row()
    r3.cells.append(Cell([Paragraph([Run("Q2", font="Helvetica", font_size_pt=10.0)])], width_twips=2880))
    r3.cells.append(Cell([Paragraph([Run("USD 1450000", font="Helvetica", font_size_pt=10.0)])], width_twips=2880))
    r3.cells.append(Cell([Paragraph([Run("USD 410000", font="Helvetica", font_size_pt=10.0)])], width_twips=2880))
    tbl.rows.append(r3)

    p.blocks.append(tbl)
    doc.pages.append(p)

    return CorpusItem(
        name="financial_table",
        description="Business balance document containing text headings and a 3x3 data matrix table.",
        pdf_bytes=pdf_bytes,
        reference_doc=doc,
    )


def _build_executive_summary_corpus() -> CorpusItem:
    """Archetype 3: Executive document with centered title, right-aligned meta date, and body."""
    pdf = fitz.open()
    page = pdf.new_page(width=612, height=792)

    font = fitz.Font("helv")
    title_text = "Executive Strategy Report"
    title_w = font.text_length(title_text, fontsize=16)
    title_x0 = (612.0 - title_w) / 2.0

    date_text = "Date: September 2026"
    date_w = font.text_length(date_text, fontsize=10)
    date_x0 = 540.0 - date_w

    # Centered Title (around midpoint 306)
    page.insert_text((title_x0, 72), title_text, fontsize=16, fontname="helv")
    # Right-aligned date (towards right margin 540)
    page.insert_text((date_x0, 110), date_text, fontsize=10, fontname="helv")
    # Left body
    page.insert_text((72, 150), "This summary outlines operational milestones and risk assessments for the upcoming quarter.", fontsize=11, fontname="helv")
    page.insert_text((72, 190), "All operational initiatives have achieved pre-registered benchmarks with high fidelity.", fontsize=11, fontname="helv")

    pdf_bytes = pdf.tobytes()
    pdf.close()

    doc = Document()
    p = Page(1, 612.0, 792.0)
    p1 = p.add_paragraph("Executive Strategy Report", font="Helvetica", font_size_pt=16.0, bold=True)
    p1.alignment = "center"
    p2 = p.add_paragraph("Date: September 2026", font="Helvetica", font_size_pt=10.0, space_before_pt=22.5)
    p2.alignment = "right"
    p.add_paragraph("This summary outlines operational milestones and risk assessments for the upcoming quarter.", font="Helvetica", font_size_pt=11.0, space_before_pt=25.0)
    p.add_paragraph("All operational initiatives have achieved pre-registered benchmarks with high fidelity.", font="Helvetica", font_size_pt=11.0, space_before_pt=25.0)
    doc.pages.append(p)

    return CorpusItem(
        name="executive_summary",
        description="Executive report featuring mixed alignment (center, right, left) and headers.",
        pdf_bytes=pdf_bytes,
        reference_doc=doc,
    )


def _build_bulleted_memo_corpus() -> CorpusItem:
    """Archetype 4: Memo with bullet points and sequential list items."""
    pdf = fitz.open()
    page = pdf.new_page(width=612, height=792)

    page.insert_text((72, 72), "Action Items and Engineering Directives", fontsize=14, fontname="helv")
    page.insert_text((72, 105), "Please review the following core deliverables for this sprint:", fontsize=11, fontname="helv")

    # Bullets using standard bullet character
    page.insert_text((90, 135), "- Harden extraction kernel against font subset variations.", fontsize=11, fontname="helv")
    page.insert_text((90, 165), "- Calibrate heuristic hyperparameters using quality-diversity search.", fontsize=11, fontname="helv")
    page.insert_text((90, 195), "- Enforce multi-gate verification across all document elements.", fontsize=11, fontname="helv")

    pdf_bytes = pdf.tobytes()
    pdf.close()

    doc = Document()
    p = Page(1, 612.0, 792.0)
    p.add_paragraph("Action Items and Engineering Directives", font="Helvetica", font_size_pt=14.0, bold=True)
    p.add_paragraph("Please review the following core deliverables for this sprint:", font="Helvetica", font_size_pt=11.0, space_before_pt=17.0)
    p.add_paragraph("- Harden extraction kernel against font subset variations.", font="Helvetica", font_size_pt=11.0, space_before_pt=15.0)
    p.add_paragraph("- Calibrate heuristic hyperparameters using quality-diversity search.", font="Helvetica", font_size_pt=11.0, space_before_pt=15.0)
    p.add_paragraph("- Enforce multi-gate verification across all document elements.", font="Helvetica", font_size_pt=11.0, space_before_pt=15.0)
    doc.pages.append(p)

    return CorpusItem(
        name="bulleted_memo",
        description="Engineering memo featuring list items, indentation offsets, and structured text.",
        pdf_bytes=pdf_bytes,
        reference_doc=doc,
    )


def _build_subset_font_showcase_corpus() -> CorpusItem:
    """Archetype 5: Stress test for Word's 6-letter subset prefixes and typography retention."""
    pdf = fitz.open()
    page = pdf.new_page(width=612, height=792)

    page.insert_text((72, 72), "Word Font Subset Normalization Test", fontsize=15, fontname="helv")
    page.insert_text((72, 110), "Verifying that random 6-character subset prefixes are cleaned seamlessly.", fontsize=11, fontname="helv")
    page.insert_text((72, 150), "Font family identity should be preserved in the generated RTF font table.", fontsize=11, fontname="helv")

    pdf_bytes = pdf.tobytes()
    pdf.close()

    doc = Document()
    p = Page(1, 612.0, 792.0)
    # The reference explicitly uses subset prefix to test that normalizer matches it to clean Calibri / Helvetica
    p1 = Paragraph()
    p1.add_run("Word Font Subset Normalization Test", font="BAAAAA+Helvetica", font_size_pt=15.0, bold=True)
    p.blocks.append(p1)

    p2 = Paragraph(space_before_pt=21.5)
    p2.add_run("Verifying that random 6-character subset prefixes are cleaned seamlessly.", font="XYZABC+Helvetica", font_size_pt=11.0)
    p.blocks.append(p2)

    p3 = Paragraph(space_before_pt=25.0)
    p3.add_run("Font family identity should be preserved in the generated RTF font table.", font="Helvetica", font_size_pt=11.0)
    p.blocks.append(p3)

    doc.pages.append(p)

    return CorpusItem(
        name="subset_font_stress",
        description="Document testing font subset prefix stripping (BAAAAA+Font) and style recovery.",
        pdf_bytes=pdf_bytes,
        reference_doc=doc,
    )


def create_golden_corpus() -> list[CorpusItem]:
    """Constructs the synthetic 5-archetype development corpus for unit testing and fast development."""
    if not HAS_FITZ:
        raise ImportError("PyMuPDF (fitz) is required to generate the development corpus.")

    return [
        _build_article_corpus(),
        _build_financial_table_corpus(),
        _build_executive_summary_corpus(),
        _build_bulleted_memo_corpus(),
        _build_subset_font_showcase_corpus(),
    ]


# Alias for explicit dev nomenclature
create_synthetic_dev_corpus = create_golden_corpus
