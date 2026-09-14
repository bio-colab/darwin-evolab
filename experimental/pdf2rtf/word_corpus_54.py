"""word_corpus_54.py — The Evolab-54 Orthogonal Factorial Benchmark Suite.

Implements an exhaustive 54-document ground-truth benchmark suite generated directly
by official Microsoft Word (Office 16.0) via Windows COM automation.

Organized into 6 orthogonal categories:
- Group A: Alignment & Spacing (12 files) [Single-factor Isolation]
- Group B: Typography & Formatting (12 files) [Single-factor Isolation]
- Group C: Lists (6 files) [Indentation & Paragraph Boundary Isolation]
- Group D: Tables (12 files) [Matrix & Cell Alignment Isolation]
- Group E: Realistic Integration (12 files) [Combinatorial Synergy]
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

from .equivalence import MultiGateVerifier
from .genome import ProfilePolicy
from .ir import Document
from .pdf_extractor import PDFExtractor
from .word_holdout import _extract_word_doc_ir

CORPUS_54_DIR = Path(__file__).resolve().parent / "real_word_54"
REPORT_54_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "pdf2rtf_corpus_54_benchmark.json"

GROUP_METADATA = {
    "A": "Alignment & Spacing (12 files)",
    "B": "Typography & Formatting (12 files)",
    "C": "Lists (6 files)",
    "D": "Tables (12 files)",
    "E": "Realistic Integration (12 files)",
}


@dataclass
class HoldoutItem54:
    """Represents a single verified benchmark item in the Evolab-54 suite."""
    name: str
    group: str
    group_name: str
    description: str
    pdf_bytes: bytes
    reference_doc: Document


# ==============================================================================
# 54 Deterministic Programmatic Word COM Builders
# ==============================================================================

def _add_p(
    doc,
    text: str,
    font: str = "Calibri",
    size: float = 11.0,
    bold: bool = False,
    italic: bool = False,
    align: int = 0,
    space_before: float = 0.0,
    space_after: float = 6.0,
    line_spacing_rule: int = 0,
    line_spacing: float = 12.0,
    left_indent: float = 0.0,
    first_line_indent: float = 0.0,
    right_indent: float = 0.0,
):
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p = doc.Paragraphs.Add(rng)
    p.Range.Text = text
    p.Range.Font.Name = font
    p.Range.Font.Size = size
    p.Range.Font.Bold = bold
    p.Range.Font.Italic = italic
    p.Range.Font.Underline = 0
    p.Range.Font.StrikeThrough = False
    p.Range.Font.Superscript = False
    p.Range.Font.Subscript = False
    p.Alignment = align
    p.Format.SpaceBefore = space_before
    p.Format.SpaceAfter = space_after
    p.Format.LineSpacingRule = line_spacing_rule
    p.Format.LineSpacing = line_spacing
    p.Format.LeftIndent = left_indent
    p.Format.FirstLineIndent = first_line_indent
    p.Format.RightIndent = right_indent
    p.Range.InsertParagraphAfter()
    return p


def _fill_tbl(t, data):
    try:
        t.Borders.Enable = True
    except Exception:
        pass
    for r_idx, row in enumerate(data, 1):
        for c_idx, val in enumerate(row, 1):
            cell = t.Cell(r_idx, c_idx)
            cell.Range.Text = str(val)
            cell.Range.Font.Name = "Calibri"
            cell.Range.Font.Size = 10.0
            if r_idx == 1:
                cell.Range.Font.Bold = True


# --- Group A: Alignment & Spacing (12) ---
def build_01(doc):
    _add_p(doc, "Left Aligned Single Spacing Header", size=14, bold=True, space_after=8)
    _add_p(doc, "This is a clean left-aligned paragraph with standard single line spacing. The words flow continuously from the left margin to demonstrate strict left alignment.", align=0, line_spacing_rule=0, line_spacing=12.0)

def build_02(doc):
    _add_p(doc, "Executive Title in Center", size=15, bold=True, align=1, space_after=8)
    _add_p(doc, "This paragraph is centered across the horizontal margins of the page. Centered text is frequently used for titles, invitations, certificates, and declarations.", align=1, line_spacing_rule=0, line_spacing=12.0)

def build_03(doc):
    _add_p(doc, "Right Aligned Document Header", size=14, bold=True, align=2, space_after=8)
    _add_p(doc, "This text is aligned strictly to the right margin of the document. Right alignment is common for dates, signatures, invoice metadata, and page footers.", align=2, line_spacing_rule=0, line_spacing=12.0)

def build_04(doc):
    _add_p(doc, "Justified Text Single Spaced", size=14, bold=True, space_after=8)
    _add_p(doc, "Justified text aligns flush with both the left and right margins of the document page. Word adjusts the inter-word spacing so that every line has equal width from margin to margin, creating a clean formal rectangular reading block.", align=3, line_spacing_rule=0, line_spacing=12.0)

def build_05(doc):
    _add_p(doc, "Default Word Style Justified 1.15", size=14, bold=True, space_after=8)
    _add_p(doc, "In standard modern Microsoft Word versions, the default Normal style uses 1.15 line spacing combined with justified or left alignment. This provides balanced vertical readability across standard letter and A4 pages without excessive white space.", align=3, line_spacing_rule=5, line_spacing=13.8)

def build_06(doc):
    _add_p(doc, "Line Spacing 1.5 with Space After 6pt", size=14, bold=True, space_after=8)
    _add_p(doc, "This paragraph features 1.5 line spacing and a space after of 6 points. Increased line spacing is standard for drafts, manuscripts, and academic reading copies to allow clear visual scanning between consecutive lines.", align=0, line_spacing_rule=1, line_spacing=18.0, space_after=6.0)

def build_07(doc):
    _add_p(doc, "Double Spaced Text Space After 12pt", size=14, bold=True, space_after=8)
    _add_p(doc, "Double line spacing specifies a vertical gap twice the standard line height. Combined with a 12 point space after, it creates distinct, widely separated textual units typical of legal briefs and doctoral thesis submissions.", align=0, line_spacing_rule=2, line_spacing=24.0, space_after=12.0)

def build_08(doc):
    _add_p(doc, "First Line Indent Paragraph", size=14, bold=True, space_after=8)
    _add_p(doc, "This paragraph demonstrates a standard first-line indentation of half an inch or 36 points. Subsequent lines wrap back to the original left margin, which is the classic typesetting convention for prose novels and newspaper articles.", align=0, first_line_indent=36.0, space_after=8)

def build_09(doc):
    _add_p(doc, "Margins with Left and Right Indent", size=14, bold=True, space_after=8)
    _add_p(doc, "This block is indented by 36 points on both the left and right margins. This creates an inset block commonly used for extended quotations, formal citations, or highlighted excerpts within longer articles.", align=3, left_indent=36.0, right_indent=36.0, space_after=8)

def build_10(doc):
    _add_p(doc, "Hanging Indent Reference Entry", size=14, bold=True, space_after=8)
    _add_p(doc, "Sharar, E. (2026). Orthogonal Typographic Benchmarks and Deterministic Conversion Kernels. Journal of Applied Systems, 42(3), 112-128.", align=0, left_indent=36.0, first_line_indent=-18.0, space_after=8)

def build_11(doc):
    _add_p(doc, "Tight Paragraphs Block 1", bold=True, space_after=2)
    _add_p(doc, "This first paragraph has only 2 points of spacing following it.", space_after=2)
    _add_p(doc, "This second paragraph directly follows with 2 points of spacing.", space_after=2)
    _add_p(doc, "This third paragraph closes the tight block verifying paragraph splitting.", space_after=2)

def build_12(doc):
    _add_p(doc, "Wide Paragraphs Block 1", bold=True, space_after=24)
    _add_p(doc, "This paragraph is separated from the next by a generous 24 points space after.", space_after=24)
    _add_p(doc, "This second paragraph has another 24 points space after, verifying wide split calibration.", space_after=24)


# --- Group B: Typography & Formatting (12) ---
def build_13(doc):
    _add_p(doc, "Times New Roman Serif Typestyle", font="Times New Roman", size=14, bold=True, space_after=8)
    _add_p(doc, "This paragraph is rendered entirely in Times New Roman at 12 points. Serif typography features small decorative strokes at the ends of character stems, traditional in books and press.", font="Times New Roman", size=12.0)

def build_14(doc):
    _add_p(doc, "Calibri Sans-Serif Typestyle", font="Calibri", size=14, bold=True, space_after=8)
    _add_p(doc, "Calibri is a contemporary sans-serif typeface designed with subtle rounded corners and clean vertical proportions. It serves as the standard corporate document typeface.", font="Calibri", size=11.0)

def build_15(doc):
    _add_p(doc, "Monospace Code Typestyle", font="Consolas", size=13, bold=True, space_after=8)
    _add_p(doc, "def parse_token(stream: str) -> bool:\n    return len(stream) > 0 and stream[0] != '\\x00'", font="Consolas", size=10.5)

def build_16(doc):
    _add_p(doc, "Complete Bold Paragraph", size=13, bold=True, space_after=6)
    _add_p(doc, "Every single word in this paragraph is styled with bold font weight. Bold text emphasizes crucial warnings, key definitions, and primary findings.", size=11, bold=True)

def build_17(doc):
    _add_p(doc, "Complete Italic Paragraph", size=13, italic=True, space_after=6)
    _add_p(doc, "Every single word in this paragraph is styled with italic font slant. Italic typography is traditionally reserved for foreign terms, book titles, and thoughtful remarks.", size=11, italic=True)

def build_18(doc):
    _add_p(doc, "Mixed Inline Runs", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p2 = doc.Paragraphs.Add(rng)
    p2.Range.Text = "Standard introduction followed by bold emphasis and italic thoughts plus bold italic climax."
    p2.Range.Font.Name = "Calibri"
    p2.Range.Font.Size = 11.0
    p2.Range.Font.Bold = False
    p2.Range.Font.Italic = False
    p2.Format.SpaceAfter = 6
    start = p2.Range.Start
    doc.Range(start + 34, start + 47).Font.Bold = True
    doc.Range(start + 52, start + 67).Font.Italic = True
    bi_rng = doc.Range(start + 73, start + 91)
    bi_rng.Font.Bold = True
    bi_rng.Font.Italic = True
    p2.Range.InsertParagraphAfter()

def build_19(doc):
    _add_p(doc, "Underline and Strikethrough", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p2 = doc.Paragraphs.Add(rng)
    p2.Range.Text = "This text has underlined concepts and superseded strikethrough items."
    p2.Range.Font.Name = "Calibri"
    p2.Range.Font.Size = 11.0
    p2.Format.SpaceAfter = 6
    start = p2.Range.Start
    doc.Range(start + 14, start + 33).Font.Underline = 1
    doc.Range(start + 38, start + 62).Font.StrikeThrough = True
    p2.Range.InsertParagraphAfter()

def build_20(doc):
    _add_p(doc, "Large Title 22pt", size=22, bold=True, space_after=6)
    _add_p(doc, "Medium Subtitle 14pt", size=14, space_after=6)
    _add_p(doc, "Standard body text set at 11 points for regular reading comprehension.", size=11, space_after=6)
    _add_p(doc, "1 Footnote annotation set at small 8 point type at the bottom of the page.", size=8, space_after=4)

def build_21(doc):
    _add_p(doc, "Document Title (20pt)", size=20, bold=True, space_after=6)
    _add_p(doc, "Heading 1: Architecture (16pt)", size=16, bold=True, space_after=4)
    _add_p(doc, "Heading 2: Implementation (13pt)", size=13, bold=True, space_after=4)
    _add_p(doc, "Regular body paragraph following the structured heading hierarchy.", size=11, space_after=6)

def build_22(doc):
    _add_p(doc, "Color and Highlighting", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p2 = doc.Paragraphs.Add(rng)
    p2.Range.Text = "Normal text then blue text and highlighted warning."
    p2.Range.Font.Name = "Calibri"
    p2.Range.Font.Size = 11.0
    p2.Format.SpaceAfter = 6
    start = p2.Range.Start
    doc.Range(start + 17, start + 26).Font.Color = 0xFF0000  # Blue in Word BGR
    doc.Range(start + 31, start + 50).HighlightColorIndex = 7  # wdYellow
    p2.Range.InsertParagraphAfter()

def build_23(doc):
    _add_p(doc, "Formulas with Indices", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p = doc.Paragraphs.Add(rng)
    p.Range.Text = "Einstein formula E = mc2 and water formula H2O."
    p.Range.Font.Name = "Calibri"
    p.Range.Font.Size = 11.0
    p.Format.SpaceAfter = 6
    start = p.Range.Start
    doc.Range(start + 23, start + 24).Font.Superscript = True
    doc.Range(start + 44, start + 45).Font.Subscript = True
    p.Range.InsertParagraphAfter()

def build_24(doc):
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p1 = doc.Paragraphs.Add(rng)
    p1.Range.Text = "Small Caps Heading"
    p1.Range.Font.Name = "Calibri"
    p1.Range.Font.Size = 13.0
    p1.Range.Font.SmallCaps = True
    p1.Format.SpaceAfter = 6
    p1.Range.InsertParagraphAfter()

    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p2 = doc.Paragraphs.Add(rng)
    p2.Range.Text = "All Caps Declaration"
    p2.Range.Font.Name = "Calibri"
    p2.Range.Font.Size = 11.0
    p2.Range.Font.AllCaps = True
    p2.Format.SpaceAfter = 6
    p2.Range.InsertParagraphAfter()


# --- Group C: Lists (6) ---
def build_25(doc):
    _add_p(doc, "Standard Bullet List", size=13, bold=True, space_after=6)
    for item in ["First bulleted item with concise text", "Second bulleted item with detailed notes", "Third bulleted item concluding the list"]:
        _add_p(doc, f"\u2022  {item}", left_indent=24.0, first_line_indent=-12.0, space_after=4)

def build_26(doc):
    _add_p(doc, "Numbered Procedure List", size=13, bold=True, space_after=6)
    for idx, item in enumerate(["Initialize the deterministic runtime engine", "Extract canonical intermediate representation", "Verify equivalence gates through multi-pass oracle"], 1):
        _add_p(doc, f"{idx}.  {item}", left_indent=24.0, first_line_indent=-12.0, space_after=4)

def build_27(doc):
    _add_p(doc, "Multilevel Section List", size=13, bold=True, space_after=6)
    _add_p(doc, "1.  System Core", left_indent=24.0, first_line_indent=-12.0, space_after=3)
    _add_p(doc, "1.1  Memory Controller", left_indent=36.0, first_line_indent=-12.0, space_after=3)
    _add_p(doc, "1.2  Instruction Cache", left_indent=36.0, first_line_indent=-12.0, space_after=3)
    _add_p(doc, "2.  Network Interface", left_indent=24.0, first_line_indent=-12.0, space_after=3)

def build_28(doc):
    _add_p(doc, "Indented Custom Bullet List", size=13, bold=True, space_after=6)
    for item in ["Indented item alpha with custom margin", "Indented item beta with matching indent"]:
        _add_p(doc, f"\u2022  {item}", left_indent=48.0, first_line_indent=-14.0, space_after=6)

def build_29(doc):
    _add_p(doc, "Justified Numbered Paragraphs", size=13, bold=True, space_after=6)
    for idx in range(1, 3):
        _add_p(doc, f"{idx}.  This numbered item contains a complete multi-line paragraph aligned to both margins with justified spacing to verify list wrapping integrity.", align=3, left_indent=24.0, first_line_indent=-12.0, space_after=6)

def build_30(doc):
    _add_p(doc, "Mixed List Structure", size=13, bold=True, space_after=6)
    _add_p(doc, "1.  Numbered requirement one", left_indent=24.0, first_line_indent=-12.0, space_after=3)
    _add_p(doc, "2.  Numbered requirement two", left_indent=24.0, first_line_indent=-12.0, space_after=6)
    _add_p(doc, "\u2022  Subordinate bullet note A", left_indent=36.0, first_line_indent=-12.0, space_after=3)
    _add_p(doc, "\u2022  Subordinate bullet note B", left_indent=36.0, first_line_indent=-12.0, space_after=6)


# --- Group D: Tables (12) ---
def build_31(doc):
    _add_p(doc, "Simple 2x2 Table", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 2, 2)
    _fill_tbl(t, [["Header A", "Header B"], ["Val 1", "Val 2"]])

def build_32(doc):
    _add_p(doc, "Simple 5x4 Table", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 5, 4)
    data = [
        ["ID", "Name", "Category", "Score"],
        ["101", "Alpha", "Metric", "98.5"],
        ["102", "Beta", "System", "99.1"],
        ["103", "Gamma", "Kernel", "97.8"],
        ["104", "Delta", "Oracle", "99.4"],
    ]
    _fill_tbl(t, data)

def build_33(doc):
    _add_p(doc, "Table with Merged Header Cells", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 3, 3)
    t.Borders.Enable = True
    t.Cell(1, 1).Merge(t.Cell(1, 2))
    t.Cell(1, 1).Range.Text = "Merged Header"
    t.Cell(1, 2).Range.Text = "Status"
    t.Cell(2, 1).Range.Text = "Row 1 Col 1"
    t.Cell(2, 2).Range.Text = "Row 1 Col 2"
    t.Cell(2, 3).Range.Text = "Active"
    t.Cell(3, 1).Range.Text = "Row 2 Col 1"
    t.Cell(3, 2).Range.Text = "Row 2 Col 2"
    t.Cell(3, 3).Range.Text = "Verified"

def build_34(doc):
    _add_p(doc, "Table with Shaded Header", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 3, 3)
    _fill_tbl(t, [["Col 1", "Col 2", "Col 3"], ["A", "B", "C"], ["D", "E", "F"]])
    t.Rows(1).Shading.BackgroundPatternColor = 0xD9D9D9

def build_35(doc):
    _add_p(doc, "Table Aligned Left", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 2, 2)
    _fill_tbl(t, [["Left 1", "Left 2"], ["Data A", "Data B"]])
    t.Rows.Alignment = 0

def build_36(doc):
    _add_p(doc, "Table Aligned Center", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 2, 2)
    _fill_tbl(t, [["Center 1", "Center 2"], ["Data A", "Data B"]])
    t.Rows.Alignment = 1

def build_37(doc):
    _add_p(doc, "Table Aligned Right", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 2, 2)
    _fill_tbl(t, [["Right 1", "Right 2"], ["Data A", "Data B"]])
    t.Rows.Alignment = 2

def build_38(doc):
    _add_p(doc, "Numeric Table Right Aligned", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 4, 3)
    t.Borders.Enable = True
    data = [
        ["Item", "Quantity", "Amount"],
        ["Server Unit", "2", "2400.00"],
        ["Backup Drive", "5", "750.00"],
        ["Switch 10Gb", "1", "1250.00"],
    ]
    for r_idx, row in enumerate(data, 1):
        for c_idx, val in enumerate(row, 1):
            cell = t.Cell(r_idx, c_idx)
            cell.Range.Text = val
            cell.Range.Font.Name = "Calibri"
            cell.Range.Font.Size = 10.0
            if r_idx == 1:
                cell.Range.Font.Bold = True
            if c_idx > 1:
                cell.Range.Paragraphs(1).Alignment = 2

def build_39(doc):
    _add_p(doc, "Borderless Table", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 3, 3)
    _fill_tbl(t, [["Item", "Cost", "Code"], ["Core", "100", "CR1"], ["RAM", "200", "RM2"]])
    t.Borders.Enable = False

def build_40(doc):
    _add_p(doc, "Full Border Grid Table", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 3, 3)
    _fill_tbl(t, [["Header 1", "Header 2", "Header 3"], ["10", "20", "30"], ["40", "50", "60"]])
    t.Borders.Enable = True

def build_41(doc):
    _add_p(doc, "Table with Multiline Text in Cells", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 2, 2)
    t.Borders.Enable = True
    t.Cell(1, 1).Range.Text = "Title"
    t.Cell(1, 2).Range.Text = "Summary"
    t.Cell(2, 1).Range.Text = "Multi-line item name"
    t.Cell(2, 2).Range.Text = "First line of description.\nSecond line of description.\nThird line of notes."

def build_42(doc):
    _add_p(doc, "Multiple Tables on Same Page", size=13, bold=True, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t1 = doc.Tables.Add(rng, 2, 2)
    _fill_tbl(t1, [["T1-A", "T1-B"], ["1", "2"]])
    _add_p(doc, "Intervening paragraph separating table one from table two.", space_after=6)
    rng2 = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t2 = doc.Tables.Add(rng2, 2, 2)
    _fill_tbl(t2, [["T2-X", "T2-Y"], ["9", "8"]])


# --- Group E: Realistic Integration (12) ---
def build_43(doc):
    _add_p(doc, "Comparative Study of Evolutionary Verification", size=16, bold=True, align=1, space_after=4)
    _add_p(doc, "Dr. Eylias Sharar & Research Associates", size=11, italic=True, align=1, space_after=8)
    _add_p(doc, "Abstract", size=12, bold=True, space_after=4)
    _add_p(doc, "We present an empirical study of deterministic multi-gate verifiers applied to compiled document representations.", align=3, space_after=6)
    _add_p(doc, "1. Introduction", size=12, bold=True, space_after=4)
    _add_p(doc, "Document reconstruction requires balancing invariant formatting semantics with flexible geometric heuristics.", align=3, space_after=6)

def build_44(doc):
    _add_p(doc, "CONFIDENTIAL SERVICE AGREEMENT", size=14, bold=True, align=1, space_after=8)
    _add_p(doc, "This Service Agreement (the Agreement) is entered into as of the Effective Date by and between Provider and Client.", align=3, space_after=6)
    _add_p(doc, "1. Scope of Services", size=12, bold=True, space_after=4)
    _add_p(doc, "Provider agrees to perform engineering calibration, regression auditing, and equivalence verification in accordance with industry benchmarks.", align=3, left_indent=18.0, space_after=6)

def build_45(doc):
    _add_p(doc, "TAX INVOICE", size=16, bold=True, align=2, space_after=6)
    _add_p(doc, "Invoice Number: INV-2026-0954", size=10, align=2, space_after=6)
    _add_p(doc, "Date: September 14, 2026", size=10, align=2, space_after=8)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 3, 3)
    _fill_tbl(t, [["Description", "Hours", "Total"], ["Calibration Engineering", "40", "4000.00"], ["Oracle Verification", "20", "2000.00"]])
    _add_p(doc, "Total Due: $6,000.00", size=12, bold=True, align=2, space_after=6)

def build_46(doc):
    _add_p(doc, "PATIENT ENCOUNTER SUMMARY", size=14, bold=True, space_after=6)
    _add_p(doc, "Patient ID: 948210", size=10, space_after=6)
    _add_p(doc, "Date of Encounter: 2026-09-14", size=10, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 3, 2)
    _fill_tbl(t, [["Vital Sign", "Measurement"], ["Blood Pressure", "120/80 mmHg"], ["Heart Rate", "72 bpm"]])
    _add_p(doc, "Assessment: Routine wellness examination. Patient displays excellent cardiovascular stability.", space_after=6)

def build_47(doc):
    _add_p(doc, "Weekly Technology Chronicle", size=16, bold=True, space_after=4)
    _add_p(doc, "The evolution of document engines has reached unprecedented fidelity with zero-dependency pipelines.", space_after=6)
    _add_p(doc, "\"True verification occurs when the official runtime application accepts the emitted stream without reservation.\"", size=12, italic=True, align=1, left_indent=36.0, right_indent=36.0, space_after=6)
    _add_p(doc, "This observation continues to guide engineering efforts across the industry.", space_after=6)

def build_48(doc):
    _add_p(doc, "Microkernel IPC Specification", size=15, bold=True, space_after=6)
    _add_p(doc, "The kernel passes messages via zero-copy shared memory descriptors defined as follows:", space_after=4)
    _add_p(doc, "typedef struct {\n    uint32_t channel_id;\n    uint64_t payload_addr;\n    uint32_t flags;\n} ipc_msg_t;", font="Consolas", size=10.0, space_after=6)
    _add_p(doc, "All channels must be initialized before dispatch.", space_after=6)

def build_49(doc):
    _add_p(doc, "Quantum Transport Kinetics", size=14, bold=True, space_after=6)
    _add_p(doc, "The transport parameters \u03b1 = 1.42, \u03b2 = 0.88, and damping factor \u03b3 = 0.05 govern conductivity \u03c3.", space_after=4)
    _add_p(doc, "Variation across bandgap \u0394E: \u0394 = 2.15 eV with mobility \u03bc = 450 cm2/Vs.", space_after=6)

def build_50(doc):
    _add_p(doc, "Multi-Page Report: Page 1", size=15, bold=True, space_after=6)
    _add_p(doc, "This is the initial section situated entirely on page one of the formal report.", space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p_brk1 = doc.Paragraphs.Add(rng)
    p_brk1.Range.InsertBreak(7)  # wdPageBreak = 7
    _add_p(doc, "Multi-Page Report: Page 2", size=15, bold=True, space_after=6)
    _add_p(doc, "This second section follows a genuine physical page break on page two.", space_after=6)
    rng2 = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p_brk2 = doc.Paragraphs.Add(rng2)
    p_brk2.Range.InsertBreak(7)
    _add_p(doc, "Multi-Page Report: Page 3", size=15, bold=True, space_after=6)
    _add_p(doc, "Final conclusions presented on page three of the document.", space_after=6)

def build_51(doc):
    _add_p(doc, "Section Header One", size=13, bold=True, space_after=6)
    _add_p(doc, "Paragraph preceding deliberate blank line.", space_after=6)
    # Deliberate empty paragraph
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    p_empty = doc.Paragraphs.Add(rng)
    p_empty.Range.Text = ""
    p_empty.Format.SpaceAfter = 12
    p_empty.Range.InsertParagraphAfter()
    _add_p(doc, "Paragraph following deliberate blank line.", space_after=6)

def build_52(doc):
    _add_p(doc, "Extended Continuous Paragraph", size=14, bold=True, space_after=6)
    long_text = (
        "In the formal analysis of algorithmic verification, deterministic pipelines ensure that identical "
        "inputs consistently yield identical intermediate representations across distinct execution environments. "
        "When evaluating document converters, this principle demands that formatting primitives including font family, "
        "size, weight, margins, paragraph spacing, and tabular matrices are preserved with sub-point precision. "
        "Heuristic calibration models tuned by quality-diversity evolutionary algorithms discover nonlinear parameter "
        "couplings that human domain experts frequently overlook. Specifically, balancing paragraph split thresholds "
        "with adaptive short-line detection enables the decoder to disambiguate tight headings from multi-line text blocks. "
        "By enforcing multi-gate equivalence checks across text, structure, tables, formatting, and visual geometry, "
        "we guarantee both grammatical fidelity and visual fidelity. Consequently, real-world deployment achieves high "
        "reliability, eliminating artifacts such as dropped words, misaligned columns, and collapsed table cells across "
        "the entire typographic corpus."
    )
    _add_p(doc, long_text, align=3, space_after=6)

def build_53(doc):
    _add_p(doc, "Multi-Currency Portfolio", size=14, bold=True, space_after=6)
    _add_p(doc, "Assets denominated in Dollar: $15,400.00", space_after=4)
    _add_p(doc, "Assets denominated in Euro: \u20ac12,850.50", space_after=4)
    _add_p(doc, "Assets denominated in British Pound: \u00a39,420.25", space_after=4)
    _add_p(doc, "Assets denominated in Japanese Yen: \u00a51,850,000", space_after=6)

def build_54(doc):
    _add_p(doc, "COMPREHENSIVE MULTI-FEATURE BENCHMARK", size=15, bold=True, align=1, space_after=6)
    _add_p(doc, "This document integrates diverse typographical and structural elements into a single composite test.", space_after=4)
    _add_p(doc, "1. Executive Highlights", size=12, bold=True, space_after=3)
    _add_p(doc, "\u2022  High precision calibration achieved across all gates", left_indent=24.0, first_line_indent=-12.0, space_after=3)
    _add_p(doc, "\u2022  Zero dependency pure Python RTF generator", left_indent=24.0, first_line_indent=-12.0, space_after=6)
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    t = doc.Tables.Add(rng, 3, 3)
    _fill_tbl(t, [["Component", "Target", "Status"], ["Text Core", "100%", "Passed"], ["Table Oracle", "100%", "Passed"]])
    _add_p(doc, "Conclusion: All architectural requirements verified.", italic=True, space_after=6)


CORPUS_SPECS: list[tuple[str, str, str, Callable]] = [
    # Group A
    ("word_seed_01_align_left_single", "A", "Left alignment with single line spacing", build_01),
    ("word_seed_02_align_center_single", "A", "Center alignment with single line spacing", build_02),
    ("word_seed_03_align_right_single", "A", "Right alignment with single line spacing", build_03),
    ("word_seed_04_align_justify_single", "A", "Justified alignment with single line spacing", build_04),
    ("word_seed_05_align_justify_1.15", "A", "Default Word style justified with 1.15 spacing", build_05),
    ("word_seed_06_align_left_1.5_space6", "A", "Left alignment with 1.5 line spacing and 6pt space after", build_06),
    ("word_seed_07_align_left_double_space12", "A", "Left alignment with double spacing and 12pt space after", build_07),
    ("word_seed_08_indent_firstline", "A", "Paragraphs with 36pt first-line indentation", build_08),
    ("word_seed_09_indent_left_right", "A", "Paragraph with 36pt left and right indentation", build_09),
    ("word_seed_10_indent_hanging", "A", "Hanging indent reference formatting", build_10),
    ("word_seed_11_para_split_tight", "A", "Tight paragraphs separated by 2pt spacing", build_11),
    ("word_seed_12_para_split_wide", "A", "Wide paragraphs separated by 24pt spacing", build_12),
    # Group B
    ("word_seed_13_font_serif_times", "B", "Times New Roman 12pt serif typography", build_13),
    ("word_seed_14_font_sans_calibri", "B", "Calibri 11pt sans-serif typography", build_14),
    ("word_seed_15_font_mono_courier", "B", "Consolas 10.5pt monospace code typography", build_15),
    ("word_seed_16_style_bold_only", "B", "Uniform bold styled paragraph", build_16),
    ("word_seed_17_style_italic_only", "B", "Uniform italic styled paragraph", build_17),
    ("word_seed_18_style_bold_italic_mixed", "B", "Mixed inline runs (regular, bold, italic, bold italic)", build_18),
    ("word_seed_19_style_underline_strike", "B", "Underline and strikethrough character styles", build_19),
    ("word_seed_20_size_mixed_small_large", "B", "Mixed font sizes from 8pt to 22pt", build_20),
    ("word_seed_21_heading_hierarchy", "B", "Structured heading hierarchy (Title, H1, H2, Body)", build_21),
    ("word_seed_22_color_highlight", "B", "Color font and text highlighting", build_22),
    ("word_seed_23_superscript_subscript", "B", "Mathematical superscript and chemical subscript", build_23),
    ("word_seed_24_allcaps_smallcaps", "B", "SmallCaps and AllCaps typography", build_24),
    # Group C
    ("word_seed_25_list_bullets", "C", "Standard bulleted list", build_25),
    ("word_seed_26_list_numbered", "C", "Standard numbered procedure list", build_26),
    ("word_seed_27_list_multilevel", "C", "Multilevel hierarchical section list", build_27),
    ("word_seed_28_list_bullet_with_indent", "C", "Bullet list with custom generous left indent", build_28),
    ("word_seed_29_list_numbered_justify", "C", "Numbered list with justified multi-line wrapping", build_29),
    ("word_seed_30_list_mixed", "C", "Mixed list with numbered and bulleted sections", build_30),
    # Group D
    ("word_seed_31_table_simple_2x2", "D", "Simple minimal 2x2 table", build_31),
    ("word_seed_32_table_simple_5x4", "D", "5x4 data matrix table", build_32),
    ("word_seed_33_table_merged_cells", "D", "Table with horizontally merged header cells", build_33),
    ("word_seed_34_table_header_shaded", "D", "Table with shaded header row", build_34),
    ("word_seed_35_table_align_left", "D", "Left-aligned table", build_35),
    ("word_seed_36_table_align_center", "D", "Center-aligned table", build_36),
    ("word_seed_37_table_align_right", "D", "Right-aligned table", build_37),
    ("word_seed_38_table_numbers_right", "D", "Numeric table with right-aligned cell amounts", build_38),
    ("word_seed_39_table_borders_none", "D", "Borderless table", build_39),
    ("word_seed_40_table_borders_all", "D", "Full border grid table", build_40),
    ("word_seed_41_table_nested_text", "D", "Table with multi-line paragraphs in cells", build_41),
    ("word_seed_42_table_multi_tables", "D", "Multiple tables on the same page", build_42),
    # Group E
    ("word_seed_43_real_academic", "E", "Academic paper with abstract and introduction", build_43),
    ("word_seed_44_real_legal_justify", "E", "Confidential legal service agreement with clauses", build_44),
    ("word_seed_45_real_invoice", "E", "Commercial tax invoice with line items and totals", build_45),
    ("word_seed_46_real_medical", "E", "Clinical patient encounter summary and vitals", build_46),
    ("word_seed_47_real_newsletter_blockquote", "E", "Editorial newsletter with blockquote pull quote", build_47),
    ("word_seed_48_real_technical_code", "E", "Microkernel IPC specification with Consolas code", build_48),
    ("word_seed_49_real_scientific_greek", "E", "Scientific paper with Greek symbols and formulas", build_49),
    ("word_seed_50_real_multipage_breaks", "E", "Multi-page report spanning 3 pages with real page breaks", build_50),
    ("word_seed_51_real_empty_para", "E", "Document with deliberate blank empty paragraphs", build_51),
    ("word_seed_52_real_long_para", "E", "Extended continuous paragraph exceeding 300 words", build_52),
    ("word_seed_53_real_currency_symbols", "E", "Multi-currency financial portfolio ($, \u20ac, \u00a3, \u00a5)", build_53),
    ("word_seed_54_real_mixed_all", "E", "Comprehensive composite stress test combining all features", build_54),
]


# ==============================================================================
# Corpus Generator & Loader
# ==============================================================================

def generate_corpus_54(out_dir: Path | None = None) -> list[str]:
    """Generates all 54 benchmark documents via Microsoft Word COM automation."""
    import win32com.client

    target_dir = Path(out_dir or CORPUS_54_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False

    generated_names: list[str] = []
    t0 = time.perf_counter()
    print(f"=== Generating Evolab-54 Benchmark Corpus (54 documents) ===")

    try:
        for idx, (doc_name, grp, desc, builder_fn) in enumerate(CORPUS_SPECS, 1):
            t_doc0 = time.perf_counter()
            doc = word.Documents.Add()

            # Set standard margins (1 inch = 72 pt)
            doc.PageSetup.TopMargin = 72.0
            doc.PageSetup.BottomMargin = 72.0
            doc.PageSetup.LeftMargin = 72.0
            doc.PageSetup.RightMargin = 72.0
            doc.PageSetup.PageWidth = 612.0   # Letter width
            doc.PageSetup.PageHeight = 792.0  # Letter height

            # Build document content
            builder_fn(doc)

            docx_path = target_dir / f"{doc_name}.docx"
            pdf_path = target_dir / f"{doc_name}.pdf"
            json_path = target_dir / f"{doc_name}.json"

            # Save DOCX and export PDF
            doc.SaveAs2(str(docx_path.resolve()))
            doc.ExportAsFixedFormat(
                OutputFileName=str(pdf_path.resolve()),
                ExportFormat=17,  # wdExportFormatPDF
                OpenAfterExport=False,
                OptimizeFor=0,     # wdExportOptimizeForPrint
            )

            # Extract ground-truth Reference IR directly from Word DOM
            ref_ir = _extract_word_doc_ir(doc)
            ref_dict = ref_ir.to_dict()
            ref_dict["_metadata"] = {
                "name": doc_name,
                "group": grp,
                "group_name": GROUP_METADATA[grp],
                "description": desc,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(ref_dict, f, indent=2)

            doc.Close(False)
            generated_names.append(doc_name)
            elapsed_ms = (time.perf_counter() - t_doc0) * 1000.0
            print(f"  [{idx:02d}/54] Group {grp}: {doc_name:<40} ({elapsed_ms:.1f}ms)")
    finally:
        word.Quit()

    total_time = time.perf_counter() - t0
    print(f"=== Generated {len(generated_names)} documents in {total_time:.2f}s ===")
    return generated_names


def load_corpus_54(corpus_dir: Path | None = None) -> list[HoldoutItem54]:
    """Loads all 54 benchmark items from disk."""
    src_dir = Path(corpus_dir or CORPUS_54_DIR)
    if not src_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {src_dir}")

    items: list[HoldoutItem54] = []
    for doc_name, grp, desc, _ in CORPUS_SPECS:
        pdf_path = src_dir / f"{doc_name}.pdf"
        json_path = src_dir / f"{doc_name}.json"

        if not pdf_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"Missing benchmark files for {doc_name}")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        with open(json_path, "r", encoding="utf-8") as f:
            ref_dict = json.load(f)

        ref_doc = Document.from_dict(ref_dict)
        item = HoldoutItem54(
            name=doc_name,
            group=grp,
            group_name=GROUP_METADATA[grp],
            description=desc,
            pdf_bytes=pdf_bytes,
            reference_doc=ref_doc,
        )
        items.append(item)

    return items


# ==============================================================================
# Group-by-Group Multi-Gate Benchmark Audit
# ==============================================================================

def run_benchmark_54(
    policy: ProfilePolicy | None = None,
    corpus_items: list[HoldoutItem54] | None = None,
    save_report: bool = True,
) -> dict[str, Any]:
    """Runs MultiGateVerifier across all 54 items with group-by-group breakdowns."""
    items = corpus_items or load_corpus_54()
    pol = policy or ProfilePolicy()

    extractor = PDFExtractor(policy=pol)
    verifier = MultiGateVerifier()

    group_results: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": [], "D": [], "E": []}
    all_scores: list[float] = []
    all_passed_count = 0
    text_passed_count = 0

    print(f"=== Running Evolab-54 Diagnostic Benchmark (N={len(items)}) ===")

    for item in items:
        t0 = time.perf_counter()
        cand_ir = extractor.extract(item.pdf_bytes)
        rep = verifier.verify(candidate=cand_ir, reference=item.reference_doc)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        all_scores.append(rep.composite_score)
        if rep.passed:
            all_passed_count += 1
        if rep.gates[0].passed:
            text_passed_count += 1

        rec = {
            "name": item.name,
            "group": item.group,
            "description": item.description,
            "passed": rep.passed,
            "composite_score": round(rep.composite_score, 4),
            "gate_scores": {g.name: round(g.score, 4) for g in rep.gates},
            "gate_pass_status": {g.name: g.passed for g in rep.gates},
            "elapsed_ms": round(elapsed_ms, 2),
            "summary": rep.summary,
        }
        group_results[item.group].append(rec)

    # Compute group summaries
    group_summaries: dict[str, Any] = {}
    for grp, grp_items in group_results.items():
        grp_scores = [r["composite_score"] for r in grp_items]
        grp_pass = sum(1 for r in grp_items if r["passed"])
        grp_text_pass = sum(1 for r in grp_items if r["gate_pass_status"]["Gate 1: Text Integrity"])
        avg_score = sum(grp_scores) / len(grp_scores) if grp_scores else 0.0

        group_summaries[grp] = {
            "group_code": grp,
            "group_name": GROUP_METADATA[grp],
            "total_documents": len(grp_items),
            "average_composite_score": round(avg_score, 4),
            "overall_pass_rate": round(grp_pass / len(grp_items), 4),
            "text_integrity_pass_rate": round(grp_text_pass / len(grp_items), 4),
        }
        print(f"  Group {grp} ({GROUP_METADATA[grp]:<30}): {avg_score:.2%} composite | Pass Rate: {grp_pass}/{len(grp_items)}")

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    overall_pass_rate = all_passed_count / len(items) if items else 0.0
    text_pass_rate = text_passed_count / len(items) if items else 0.0

    print("-" * 75)
    print(f"Evolab-54 Total Documents:         {len(items)}")
    print(f"Evolab-54 Average Composite Score: {overall_avg:.2%}")
    print(f"Evolab-54 Text Integrity Pass:    {text_pass_rate:.2%}")
    print(f"Evolab-54 Overall Pass Rate:       {overall_pass_rate:.2%}")

    report = {
        "title": "Evolab-54 Orthogonal Factorial Benchmark Audit Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_documents": len(items),
        "summary": {
            "average_composite_score": round(overall_avg, 4),
            "text_integrity_pass_rate": round(text_pass_rate, 4),
            "overall_pass_rate": round(overall_pass_rate, 4),
        },
        "groups": group_summaries,
        "document_results": group_results,
    }

    if save_report:
        REPORT_54_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_54_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {REPORT_54_PATH}")

    return report


if __name__ == "__main__":
    if "--generate" in sys.argv or not CORPUS_54_DIR.exists():
        generate_corpus_54()
    run_benchmark_54()
