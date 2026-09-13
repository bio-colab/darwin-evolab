"""word_dilemma.py — Genuine Microsoft Word Typographic Dilemma Benchmark Corpus.

Generates authentic Microsoft Word documents exhibiting challenging typographic dilemmas
(tight line leading, compact lists with indentation shifts, mixed font size hierarchies,
and dense tables with narrow gutters) where static human heuristic baselines fail.
Ground truth Reference IR is extracted directly from Word COM objects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from .corpus import CorpusItem
from .ir import Cell, Color, Document, Page, Paragraph, Row, Run, Table
from .word_holdout import _extract_word_doc_ir

DILEMMA_DIR = Path(__file__).parent / "real_word_dilemma"


def generate_dilemma_documents(force_recreate: bool = False) -> None:
    """Generates Word documents with typographic dilemmas via Word COM automation."""
    DILEMMA_DIR.mkdir(parents=True, exist_ok=True)

    dilemma_names = [
        "word_dilemma_tight_lead",
        "word_dilemma_compact_list",
        "word_dilemma_mixed_scale",
        "word_dilemma_dense_table",
    ]

    missing = any(
        not (DILEMMA_DIR / f"{name}.pdf").exists()
        or not (DILEMMA_DIR / f"{name}.json").exists()
        for name in dilemma_names
    )

    if not missing and not force_recreate:
        return

    import win32com.client

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False

    try:
        # 1. Dilemma Tight Lead: Executive compliance memo with 2pt spacing and short-line paragraph breaks
        doc1 = word.Documents.Add()
        doc1.PageSetup.TopMargin = 72
        doc1.PageSetup.BottomMargin = 72
        doc1.PageSetup.LeftMargin = 72
        doc1.PageSetup.RightMargin = 72

        p1 = doc1.Paragraphs.Add()
        p1.Range.Text = "EXECUTIVE SUMMARY AND REGULATORY COMPLIANCE MEMORANDUM\n"
        p1.Range.Font.Name = "Calibri"
        p1.Range.Font.Size = 14
        p1.Range.Font.Bold = True
        p1.Range.ParagraphFormat.SpaceAfter = 2.0
        p1.Range.ParagraphFormat.LineSpacingRule = 0

        p2 = doc1.Paragraphs.Add()
        p2.Range.Text = "1. Jurisdictional Scope and Enforcement Priorities\n"
        p2.Range.Font.Name = "Calibri"
        p2.Range.Font.Size = 12
        p2.Range.Font.Bold = True
        p2.Range.ParagraphFormat.SpaceAfter = 2.0

        p3 = doc1.Paragraphs.Add()
        p3.Range.Text = "This memorandum establishes the formal supervisory expectations applicable to designated institutions under Article 4. All regulated entities must maintain audit trails.\n"
        p3.Range.Font.Name = "Calibri"
        p3.Range.Font.Size = 11
        p3.Range.Font.Bold = False
        p3.Range.ParagraphFormat.SpaceAfter = 2.0

        p4 = doc1.Paragraphs.Add()
        p4.Range.Text = "Failure to comply with mandatory reporting schedules will trigger administrative penalties under Section 12. Continued non-compliance warrants revocation of operating authority.\n"
        p4.Range.Font.Name = "Calibri"
        p4.Range.Font.Size = 11
        p4.Range.Font.Bold = False
        p4.Range.ParagraphFormat.SpaceAfter = 2.0

        p5 = doc1.Paragraphs.Add()
        p5.Range.Text = "2. Capital Adequacy and Liquidity Reserve Ratios\n"
        p5.Range.Font.Name = "Calibri"
        p5.Range.Font.Size = 12
        p5.Range.Font.Bold = True
        p5.Range.ParagraphFormat.SpaceAfter = 2.0

        p6 = doc1.Paragraphs.Add()
        p6.Range.Text = "Core capital reserves must exceed the prescribed risk-weighted threshold at all reporting dates. Tier 1 assets shall be maintained in liquid sovereign instruments.\n"
        p6.Range.Font.Name = "Calibri"
        p6.Range.Font.Size = 11
        p6.Range.Font.Bold = False
        p6.Range.ParagraphFormat.SpaceAfter = 2.0

        _save_and_extract_word_doc(doc1, "word_dilemma_tight_lead")
        doc1.Close(False)

        # 2. Dilemma Compact List: Operational checklist with indented list items and zero space between items
        doc2 = word.Documents.Add()
        doc2.PageSetup.TopMargin = 72
        doc2.PageSetup.BottomMargin = 72
        doc2.PageSetup.LeftMargin = 72
        doc2.PageSetup.RightMargin = 72

        lp1 = doc2.Paragraphs.Add()
        lp1.Range.Text = "STANDARD OPERATING PROCEDURE: INCIDENT RESPONSE CHECKLIST\n"
        lp1.Range.Font.Name = "Calibri"
        lp1.Range.Font.Size = 13
        lp1.Range.Font.Bold = True
        lp1.Range.ParagraphFormat.SpaceAfter = 4.0

        lp2 = doc2.Paragraphs.Add()
        lp2.Range.Text = "Upon confirmation of a security anomaly, operations personnel must execute the following protocol steps in immediate sequence:\n"
        lp2.Range.Font.Name = "Calibri"
        lp2.Range.Font.Size = 11
        lp2.Range.Font.Bold = False
        lp2.Range.ParagraphFormat.SpaceAfter = 3.0

        items = [
            "Step 1: Isolate affected network segment and sever outbound gateway routing.",
            "Step 2: Capture volatile memory dump from primary host for forensic examination.",
            "Step 3: Notify incident commander and activate designated escalation channels.",
            "Step 4: Rotate all administrative credentials and revoke active session tokens.",
            "Step 5: Verify integrity of cold offline backups prior to system restoration.",
        ]
        for it in items:
            lp_item = doc2.Paragraphs.Add()
            lp_item.Range.Text = f"{it}\n"
            lp_item.Range.Font.Name = "Calibri"
            lp_item.Range.Font.Size = 10.5
            lp_item.Range.Font.Bold = False
            lp_item.Range.ParagraphFormat.LeftIndent = 18.0  # 18pt indent shift!
            lp_item.Range.ParagraphFormat.SpaceAfter = 1.5  # Very tight space after!
            lp_item.Range.ParagraphFormat.LineSpacingRule = 0

        lp_end = doc2.Paragraphs.Add()
        lp_end.Range.Text = "Formal post-incident review must convene within twenty-four hours of containment.\n"
        lp_end.Range.Font.Name = "Calibri"
        lp_end.Range.Font.Size = 11
        lp_end.Range.Font.Italic = True
        lp_end.Range.ParagraphFormat.LeftIndent = 0.0
        lp_end.Range.ParagraphFormat.SpaceAfter = 4.0

        _save_and_extract_word_doc(doc2, "word_dilemma_compact_list")
        doc2.Close(False)

        # 3. Dilemma Mixed Scale: Academic notice with steep font size transitions and tight vertical gap
        doc3 = word.Documents.Add()
        doc3.PageSetup.TopMargin = 72
        doc3.PageSetup.BottomMargin = 72
        doc3.PageSetup.LeftMargin = 72
        doc3.PageSetup.RightMargin = 72

        mp1 = doc3.Paragraphs.Add()
        mp1.Range.Text = "DEPARTMENT OF QUANTUM INFORMATION & COMPUTING\n"
        mp1.Range.Font.Name = "Times New Roman"
        mp1.Range.Font.Size = 16
        mp1.Range.Font.Bold = True
        mp1.Range.ParagraphFormat.SpaceAfter = 2.0

        mp2 = doc3.Paragraphs.Add()
        mp2.Range.Text = "Doctoral Defense Schedule and Colloquium Notice\n"
        mp2.Range.Font.Name = "Times New Roman"
        mp2.Range.Font.Size = 12
        mp2.Range.Font.Italic = True
        mp2.Range.ParagraphFormat.SpaceAfter = 2.0

        mp3 = doc3.Paragraphs.Add()
        mp3.Range.Text = "Candidate: Dr. Marcus Vance. Title: Non-Equilibrium Phase Transitions in Superconducting Qubits.\n"
        mp3.Range.Font.Name = "Times New Roman"
        mp3.Range.Font.Size = 11
        mp3.Range.ParagraphFormat.SpaceAfter = 2.0

        mp4 = doc3.Paragraphs.Add()
        mp4.Range.Text = "Abstract: We investigate boundary state evolution in open quantum systems driven by dissipative Lindblad operators. The observed critical phenomena demonstrate robust protection against decoherence.\n"
        mp4.Range.Font.Name = "Times New Roman"
        mp4.Range.Font.Size = 11
        mp4.Range.ParagraphFormat.SpaceAfter = 2.0

        mp5 = doc3.Paragraphs.Add()
        mp5.Range.Text = "Notice: Attendance is restricted to credentialed department members. Recording devices prohibited under Faculty Bylaw 8.1.\n"
        mp5.Range.Font.Name = "Times New Roman"
        mp5.Range.Font.Size = 8.5  # Steep font scale drop!
        mp5.Range.Font.Italic = True
        mp5.Range.ParagraphFormat.SpaceAfter = 2.0

        _save_and_extract_word_doc(doc3, "word_dilemma_mixed_scale")
        doc3.Close(False)

        # 4. Dilemma Dense Table: Financial balance ledger with compact columns and tight vertical leading
        doc4 = word.Documents.Add()
        doc4.PageSetup.TopMargin = 72
        doc4.PageSetup.BottomMargin = 72
        doc4.PageSetup.LeftMargin = 72
        doc4.PageSetup.RightMargin = 72

        tp1 = doc4.Paragraphs.Add()
        tp1.Range.Text = "CONSOLIDATED QUARTERLY LIQUIDITY AND ASSET ALLOCATION MATRIX\n"
        tp1.Range.Font.Name = "Calibri"
        tp1.Range.Font.Size = 12
        tp1.Range.Font.Bold = True
        tp1.Range.ParagraphFormat.SpaceAfter = 4.0

        tbl_range = doc4.Paragraphs.Add().Range
        tbl = doc4.Tables.Add(tbl_range, NumRows=5, NumColumns=4)
        tbl.Borders.Enable = True

        headers = ["Asset Class", "Q1 Allocation", "Q2 Allocation", "Variance"]
        for c_idx, h in enumerate(headers, 1):
            cell = tbl.Cell(1, c_idx)
            cell.Range.Text = h
            cell.Range.Font.Name = "Calibri"
            cell.Range.Font.Size = 9.5
            cell.Range.Font.Bold = True

        rows_data = [
            ["Cash & Equiv", "$1,450,000", "$1,620,000", "+11.7%"],
            ["Sovereign Debt", "$4,200,000", "$3,950,000", "-5.9%"],
            ["Corporate Debt", "$2,800,000", "$2,910,000", "+3.9%"],
            ["Total Assets", "$8,450,000", "$8,480,000", "+0.3%"],
        ]
        for r_idx, row_vals in enumerate(rows_data, 2):
            for c_idx, val in enumerate(row_vals, 1):
                cell = tbl.Cell(r_idx, c_idx)
                cell.Range.Text = val
                cell.Range.Font.Name = "Calibri"
                cell.Range.Font.Size = 9.5
                cell.Range.Font.Bold = (r_idx == 5)

        tp_end = doc4.Paragraphs.Add()
        tp_end.Range.Text = "All allocation variances remain within statutory macro-prudential thresholds.\n"
        tp_end.Range.Font.Name = "Calibri"
        tp_end.Range.Font.Size = 10
        tp_end.Range.ParagraphFormat.SpaceAfter = 4.0

        _save_and_extract_word_doc(doc4, "word_dilemma_dense_table")
        doc4.Close(False)

    finally:
        word.Quit()


def _save_and_extract_word_doc(doc, base_name: str) -> None:
    """Saves docx, exports pdf, and writes ground-truth IR json."""
    docx_path = DILEMMA_DIR / f"{base_name}.docx"
    pdf_path = DILEMMA_DIR / f"{base_name}.pdf"
    json_path = DILEMMA_DIR / f"{base_name}.json"

    doc.SaveAs2(str(docx_path.resolve()), FileFormat=16)  # wdFormatXMLDocument = 16
    doc.SaveAs2(str(pdf_path.resolve()), FileFormat=17)  # wdFormatPDF = 17

    ir_doc = _extract_word_doc_ir(doc)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ir_doc.to_dict(), f, indent=2)


def load_word_dilemma_corpus() -> list[CorpusItem]:
    """Loads all Word dilemma benchmark documents as CorpusItems."""
    generate_dilemma_documents(force_recreate=False)

    dilemma_names = [
        "word_dilemma_tight_lead",
        "word_dilemma_compact_list",
        "word_dilemma_mixed_scale",
        "word_dilemma_dense_table",
    ]

    dilemma_descriptions = {
        "word_dilemma_tight_lead": "Executive compliance memo with tight 2pt vertical leading and short-line paragraph breaks.",
        "word_dilemma_compact_list": "Operational procedure checklist with 18pt indented items and compact 1.5pt spacing.",
        "word_dilemma_mixed_scale": "Departmental academic colloquium notice with steep 16pt to 8.5pt font size hierarchy transitions.",
        "word_dilemma_dense_table": "Consolidated financial liquidity matrix with 5x4 grid and compact 60pt column gutters.",
    }

    items: list[CorpusItem] = []
    for name in dilemma_names:
        pdf_file = DILEMMA_DIR / f"{name}.pdf"
        json_file = DILEMMA_DIR / f"{name}.json"

        if not pdf_file.exists() or not json_file.exists():
            generate_dilemma_documents(force_recreate=True)

        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()

        with open(json_file, "r", encoding="utf-8") as f:
            ref_dict = json.load(f)

        ref_doc = Document.from_dict(ref_dict)
        desc = dilemma_descriptions.get(name, "Challenging MS Word typographic dilemma document.")
        items.append(CorpusItem(name=name, description=desc, pdf_bytes=pdf_bytes, reference_doc=ref_doc))

    return items


if __name__ == "__main__":
    generate_dilemma_documents(force_recreate=True)
    items = load_word_dilemma_corpus()
    print(f"Successfully generated and loaded {len(items)} Word dilemma documents.")
