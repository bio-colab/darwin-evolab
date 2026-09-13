"""word_holdout.py — Genuine Microsoft Word-generated Holdout Benchmark Corpus.

Provides a truly independent, out-of-distribution holdout evaluation suite
generated directly by Microsoft Word (Office 16.0) via COM automation.
Ground truth Reference IR is extracted from Word's native object model,
completely independent of PyMuPDF or synthetic generators.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .corpus import CorpusItem
from .ir import Cell, Color, Document, Page, Paragraph, Row, Run, Table

HOLDOUT_DIR = Path(__file__).parent / "real_word_holdout"


def _extract_word_doc_ir(doc) -> Document:
    """Extracts ground truth Document IR directly from a Word COM Document object."""
    ir_doc = Document()

    # Get page setup from first section
    page_w = float(doc.PageSetup.PageWidth)
    page_h = float(doc.PageSetup.PageHeight)
    page = ir_doc.add_page(width_pts=page_w, height_pts=page_h)

    # In Word, table paragraphs also appear in doc.Paragraphs.
    # To preserve layout ordering, iterate through document content ranges
    # or identify which tables and paragraphs exist.
    tables_in_doc = list(doc.Tables)
    table_ranges = [(t.Range.Start, t.Range.End, t) for t in tables_in_doc]

    elements_by_pos: list[tuple[int, Any]] = []

    # Process Tables
    for t_start, t_end, t in table_ranges:
        tbl_ir = Table(alignment="left")
        for r_idx in range(1, t.Rows.Count + 1):
            is_hdr = (r_idx == 1)
            row_ir = Row(is_header=is_hdr)
            row_obj = t.Rows(r_idx)
            for c_idx in range(1, row_obj.Cells.Count + 1):
                cell_obj = row_obj.Cells(c_idx)
                raw_txt = cell_obj.Range.Text.strip("\r\x07 ")
                c_ir = Cell(width_twips=2880)
                if raw_txt:
                    c_ir.add_paragraph(
                        raw_txt,
                        font=str(cell_obj.Range.Font.Name),
                        font_size_pt=float(cell_obj.Range.Font.Size),
                        bold=bool(cell_obj.Range.Font.Bold),
                    )
                row_ir.cells.append(c_ir)
            tbl_ir.rows.append(row_ir)
        elements_by_pos.append((t_start, tbl_ir))

    # Process Paragraphs outside tables
    for p in doc.Paragraphs:
        # Check if inside a table (wdWithInTable = 12)
        try:
            if p.Range.Information(12):
                continue
        except Exception:
            pass

        full_txt = p.Range.Text.strip("\r\x07")
        if not full_txt:
            continue

        p_ir = Paragraph()

        # Alignment mapping
        align_code = p.Alignment
        align_map = {0: "left", 1: "center", 2: "right", 3: "justify"}
        p_ir.alignment = align_map.get(align_code, "left")

        p_ir.space_before_pt = float(p.Format.SpaceBefore)
        p_ir.space_after_pt = float(p.Format.SpaceAfter)
        try:
            p_ir.line_spacing_pt = float(p.Format.LineSpacing)
        except Exception:
            p_ir.line_spacing_pt = None

        # Extract character runs
        start = p.Range.Start
        current_text = []
        current_props = None
        for idx in range(len(full_txt)):
            ch_rng = doc.Range(start + idx, start + idx + 1)
            props = (
                str(ch_rng.Font.Name),
                float(ch_rng.Font.Size),
                bool(ch_rng.Font.Bold),
                bool(ch_rng.Font.Italic),
            )
            if current_props is None:
                current_props = props
                current_text.append(full_txt[idx])
            elif props == current_props:
                current_text.append(full_txt[idx])
            else:
                f_name, f_sz, bld, itl = current_props
                p_ir.add_run("".join(current_text), font=f_name, font_size_pt=f_sz, bold=bld, italic=itl)
                current_props = props
                current_text = [full_txt[idx]]

        if current_text and current_props:
            f_name, f_sz, bld, itl = current_props
            p_ir.add_run("".join(current_text), font=f_name, font_size_pt=f_sz, bold=bld, italic=itl)

        elements_by_pos.append((p.Range.Start, p_ir))

    # Sort elements in document order
    elements_by_pos.sort(key=lambda item: item[0])
    for _, el in elements_by_pos:
        page.blocks.append(el)

    return ir_doc


def generate_real_word_holdout(output_dir: str | Path | None = None) -> list[CorpusItem]:
    """Generates 4 genuine Microsoft Word documents, exports them to PDF via Word COM,
    and extracts ground-truth Reference IR directly from Word's native DOM."""
    import win32com.client

    target_dir = Path(output_dir or HOLDOUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False

    items: list[CorpusItem] = []

    try:
        # -------------------------------------------------------------
        # 1. word_academic_paper
        # -------------------------------------------------------------
        doc1 = word.Documents.Add()
        p1 = doc1.Paragraphs.Add()
        p1.Range.Text = "Evolutionary Architecture of Autonomous Agentic Systems"
        p1.Range.Font.Name = "Arial"
        p1.Range.Font.Size = 16
        p1.Range.Font.Bold = True
        p1.Range.Font.Italic = False
        p1.Alignment = 1  # Center
        p1.Format.SpaceAfter = 6.0
        p1.Range.InsertParagraphAfter()

        p2 = doc1.Paragraphs.Add()
        p2.Range.Text = "Darwin-Evolab Scientific Research Group"
        p2.Range.Font.Name = "Calibri"
        p2.Range.Font.Size = 11
        p2.Range.Font.Bold = False
        p2.Range.Font.Italic = True
        p2.Alignment = 1  # Center
        p2.Format.SpaceAfter = 18.0
        p2.Range.InsertParagraphAfter()

        p3 = doc1.Paragraphs.Add()
        p3.Range.Text = "1. Introduction and Architectural Primitives"
        p3.Range.Font.Name = "Arial"
        p3.Range.Font.Size = 13
        p3.Range.Font.Bold = True
        p3.Range.Font.Italic = False
        p3.Alignment = 0  # Left
        p3.Format.SpaceBefore = 12.0
        p3.Format.SpaceAfter = 6.0
        p3.Range.InsertParagraphAfter()

        p4 = doc1.Paragraphs.Add()
        p4.Range.Text = "Autonomous evolutionary optimization operates by generating bounded hypotheses and verifying them against strict equivalence oracles."
        p4.Range.Font.Name = "Calibri"
        p4.Range.Font.Size = 11
        p4.Range.Font.Bold = False
        p4.Range.Font.Italic = False
        p4.Alignment = 0  # Left
        p4.Format.SpaceAfter = 6.0
        p4.Range.InsertParagraphAfter()

        p5 = doc1.Paragraphs.Add()
        p5.Range.Text = "- Invariant core representation of canonical document trees."
        p5.Range.Font.Name = "Calibri"
        p5.Range.Font.Size = 11
        p5.Range.Font.Bold = False
        p5.Range.Font.Italic = False
        p5.Alignment = 0
        p5.Format.SpaceAfter = 4.0
        p5.Range.InsertParagraphAfter()

        p6 = doc1.Paragraphs.Add()
        p6.Range.Text = "- Multi-gate verification with cascading integrity penalties."
        p6.Range.Font.Name = "Calibri"
        p6.Range.Font.Size = 11
        p6.Range.Font.Bold = False
        p6.Range.Font.Italic = False
        p6.Alignment = 0
        p6.Format.SpaceAfter = 4.0
        p6.Range.InsertParagraphAfter()

        p7 = doc1.Paragraphs.Add()
        p7.Range.Text = "- Empirical calibration against real Word document layouts."
        p7.Range.Font.Name = "Calibri"
        p7.Range.Font.Size = 11
        p7.Range.Font.Bold = False
        p7.Range.Font.Italic = False
        p7.Alignment = 0
        p7.Format.SpaceAfter = 12.0
        p7.Range.InsertParagraphAfter()

        pdf1_path = str(target_dir / "word_academic_paper.pdf")
        docx1_path = str(target_dir / "word_academic_paper.docx")
        doc1.ExportAsFixedFormat(pdf1_path, 17)
        doc1.SaveAs2(docx1_path)

        ir1 = _extract_word_doc_ir(doc1)
        doc1.Close(False)

        with open(pdf1_path, "rb") as f:
            pdf1_bytes = f.read()
        with open(target_dir / "word_academic_paper.json", "w", encoding="utf-8") as f:
            json.dump(ir1.to_dict(), f, indent=2)

        items.append(CorpusItem("word_academic_paper", "Genuine Word academic article with title, author, and bullets", pdf1_bytes, ir1))

        # -------------------------------------------------------------
        # 2. word_financial_report
        # -------------------------------------------------------------
        doc2 = word.Documents.Add()
        p = doc2.Paragraphs.Add()
        p.Range.Text = "Quarterly Audit and Performance Ledger"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 14
        p.Range.Font.Bold = True
        p.Range.Font.Italic = False
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        p = doc2.Paragraphs.Add()
        p.Range.Text = "Summary of verified computational metrics across benchmark suites:"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        # Insert 3x3 Table
        rng = doc2.Range(doc2.Content.End - 1, doc2.Content.End - 1)
        tbl = doc2.Tables.Add(rng, 3, 3)
        tbl.Borders.Enable = True
        
        # Row 1 (Header - Bold)
        for c_i, header_txt in enumerate(["Benchmark", "Pass Rate", "Fidelity"], 1):
            cell_rng = tbl.Cell(1, c_i).Range
            cell_rng.Text = header_txt
            cell_rng.Font.Name = "Calibri"
            cell_rng.Font.Size = 11
            cell_rng.Font.Bold = True
            cell_rng.Font.Italic = False

        # Row 2 (Data - Normal)
        for c_i, data_txt in enumerate(["Text Integrity", "100.0%", "Exact"], 1):
            cell_rng = tbl.Cell(2, c_i).Range
            cell_rng.Text = data_txt
            cell_rng.Font.Name = "Calibri"
            cell_rng.Font.Size = 11
            cell_rng.Font.Bold = False
            cell_rng.Font.Italic = False

        # Row 3 (Data - Normal)
        for c_i, data_txt in enumerate(["Structure Gate", "98.3%", "High"], 1):
            cell_rng = tbl.Cell(3, c_i).Range
            cell_rng.Text = data_txt
            cell_rng.Font.Name = "Calibri"
            cell_rng.Font.Size = 11
            cell_rng.Font.Bold = False
            cell_rng.Font.Italic = False

        p_end = doc2.Paragraphs.Add()
        p_end.Range.Text = "All values verified through deterministic execution."
        p_end.Range.Font.Name = "Calibri"
        p_end.Range.Font.Size = 10
        p_end.Range.Font.Bold = False
        p_end.Range.Font.Italic = True
        p_end.Format.SpaceBefore = 12.0
        p_end.Range.InsertParagraphAfter()

        pdf2_path = str(target_dir / "word_financial_report.pdf")
        docx2_path = str(target_dir / "word_financial_report.docx")
        doc2.ExportAsFixedFormat(pdf2_path, 17)
        doc2.SaveAs2(docx2_path)

        ir2 = _extract_word_doc_ir(doc2)
        doc2.Close(False)

        with open(pdf2_path, "rb") as f:
            pdf2_bytes = f.read()
        with open(target_dir / "word_financial_report.json", "w", encoding="utf-8") as f:
            json.dump(ir2.to_dict(), f, indent=2)

        items.append(CorpusItem("word_financial_report", "Genuine Word financial report with 3x3 data matrix", pdf2_bytes, ir2))

        # -------------------------------------------------------------
        # 3. word_executive_letter
        # -------------------------------------------------------------
        doc3 = word.Documents.Add()
        p = doc3.Paragraphs.Add()
        p.Range.Text = "GLOBAL SYSTEMS RESEARCH DIRECTIVE"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 15
        p.Range.Font.Bold = True
        p.Range.Font.Italic = False
        p.Alignment = 1  # Center
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        p = doc3.Paragraphs.Add()
        p.Range.Text = "Date: September 13, 2026"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 10
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Alignment = 2  # Right
        p.Format.SpaceBefore = 6.0
        p.Format.SpaceAfter = 18.0
        p.Range.InsertParagraphAfter()

        p = doc3.Paragraphs.Add()
        p.Range.Text = "Memorandum for Senior Reviewers"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Range.Font.Bold = True
        p.Range.Font.Italic = False
        p.Alignment = 0  # Left
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        p = doc3.Paragraphs.Add()
        p.Range.Text = "This letter serves as formal notification regarding the completion of Phase 4 hardening benchmarks."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Alignment = 0
        p.Format.SpaceAfter = 8.0
        p.Range.InsertParagraphAfter()

        p = doc3.Paragraphs.Add()
        p.Range.Text = "All empirical criteria established by independent peer audit have been rigorously tested on genuine Word layouts."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Alignment = 0
        p.Format.SpaceAfter = 18.0
        p.Range.InsertParagraphAfter()

        p = doc3.Paragraphs.Add()
        p.Range.Text = "Office of Architecture and Standards"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Range.Font.Bold = True
        p.Range.Font.Italic = False
        p.Alignment = 0
        p.Range.InsertParagraphAfter()

        pdf3_path = str(target_dir / "word_executive_letter.pdf")
        docx3_path = str(target_dir / "word_executive_letter.docx")
        doc3.ExportAsFixedFormat(pdf3_path, 17)
        doc3.SaveAs2(docx3_path)

        ir3 = _extract_word_doc_ir(doc3)
        doc3.Close(False)

        with open(pdf3_path, "rb") as f:
            pdf3_bytes = f.read()
        with open(target_dir / "word_executive_letter.json", "w", encoding="utf-8") as f:
            json.dump(ir3.to_dict(), f, indent=2)

        items.append(CorpusItem("word_executive_letter", "Genuine Word executive letter with mixed alignments", pdf3_bytes, ir3))

        # -------------------------------------------------------------
        # 4. word_styled_article
        # -------------------------------------------------------------
        doc4 = word.Documents.Add()
        p = doc4.Paragraphs.Add()
        p.Range.Text = "Advanced Typographic Specification and Formatting"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 16
        p.Range.Font.Bold = True
        p.Range.Font.Italic = False
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        p = doc4.Paragraphs.Add()
        text_p2 = "This section validates that mixed bold runs, italic expressions, and subset font families are preserved with high fidelity."
        p.Range.Text = text_p2
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 12
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        
        idx_bold_start = text_p2.find("mixed bold runs")
        idx_bold_end = idx_bold_start + len("mixed bold runs")
        idx_ital_start = text_p2.find("italic expressions")
        idx_ital_end = idx_ital_start + len("italic expressions")
        
        rng_bold = doc4.Range(p.Range.Start + idx_bold_start, p.Range.Start + idx_bold_end)
        rng_bold.Bold = True
        rng_bold.Italic = False
        
        rng_ital = doc4.Range(p.Range.Start + idx_ital_start, p.Range.Start + idx_ital_end)
        rng_ital.Bold = False
        rng_ital.Italic = True
        
        p.Format.SpaceAfter = 10.0
        p.Range.InsertParagraphAfter()

        p = doc4.Paragraphs.Add()
        p.Range.Text = "Deterministic extraction ensures that document topology remains invariant under format migration."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Format.SpaceBefore = 14.0
        p.Format.SpaceAfter = 8.0
        p.Range.InsertParagraphAfter()

        pdf4_path = str(target_dir / "word_styled_article.pdf")
        docx4_path = str(target_dir / "word_styled_article.docx")
        doc4.ExportAsFixedFormat(pdf4_path, 17)
        doc4.SaveAs2(docx4_path)

        ir4 = _extract_word_doc_ir(doc4)
        doc4.Close(False)

        with open(pdf4_path, "rb") as f:
            pdf4_bytes = f.read()
        with open(target_dir / "word_styled_article.json", "w", encoding="utf-8") as f:
            json.dump(ir4.to_dict(), f, indent=2)

        items.append(CorpusItem("word_styled_article", "Genuine Word styled article with mixed typography", pdf4_bytes, ir4))

    finally:
        word.Quit()

    return items


def load_real_word_holdout(target_dir: str | Path | None = None) -> list[CorpusItem]:
    """Loads the genuine Microsoft Word Holdout Corpus.
    If the serialized files do not yet exist, generates them via Word COM."""
    dir_path = Path(target_dir or HOLDOUT_DIR)

    names = [
        ("word_academic_paper", "Genuine Word academic article with title, author, and bullets"),
        ("word_financial_report", "Genuine Word financial report with 3x3 data matrix"),
        ("word_executive_letter", "Genuine Word executive letter with mixed alignments"),
        ("word_styled_article", "Genuine Word styled article with mixed typography"),
    ]

    # Check if all files exist
    all_exist = all(
        (dir_path / f"{name}.pdf").exists() and (dir_path / f"{name}.json").exists()
        for name, _ in names
    )

    if not all_exist:
        return generate_real_word_holdout(dir_path)

    items: list[CorpusItem] = []
    for name, desc in names:
        pdf_p = dir_path / f"{name}.pdf"
        json_p = dir_path / f"{name}.json"
        with open(pdf_p, "rb") as f:
            pdf_bytes = f.read()
        with open(json_p, "r", encoding="utf-8") as f:
            ir_dict = json.load(f)
        ref_doc = Document.from_dict(ir_dict)
        items.append(CorpusItem(name, desc, pdf_bytes, ref_doc))

    return items
