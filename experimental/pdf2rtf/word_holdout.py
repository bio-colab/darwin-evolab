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
    try:
        total_pages = max(1, doc.ComputeStatistics(2))  # wdStatisticPages = 2
    except Exception:
        total_pages = 1
    for _ in range(total_pages):
        ir_doc.add_page(width_pts=page_w, height_pts=page_h)

    # In Word, table paragraphs also appear in doc.Paragraphs.
    # To preserve layout ordering, iterate through document content ranges
    # or identify which tables and paragraphs exist.
    tables_in_doc = list(doc.Tables)
    table_ranges = [(t.Range.Start, t.Range.End, t) for t in tables_in_doc]

    elements_by_pos: list[tuple[int, int, Any]] = []

    # Process Tables
    for t_start, t_end, t in table_ranges:
        try:
            pg_num = int(t.Range.Information(3))  # wdActiveEndPageNumber
        except Exception:
            pg_num = 1
        tbl_ir = Table(alignment="left")
        for r_idx in range(1, t.Rows.Count + 1):
            is_hdr = (r_idx == 1)
            row_ir = Row(is_header=is_hdr)
            row_obj = t.Rows(r_idx)
            for c_idx in range(1, row_obj.Cells.Count + 1):
                cell_obj = row_obj.Cells(c_idx)
                try:
                    c_width_twips = int(round(float(cell_obj.Width) * 20.0))
                except Exception:
                    c_width_twips = 2880
                c_ir = Cell(width_twips=max(720, c_width_twips))
                raw_txt = cell_obj.Range.Text.strip("\r\x07\x0c ")
                if raw_txt:
                    c_ir.add_paragraph(
                        raw_txt,
                        font=str(cell_obj.Range.Font.Name),
                        font_size_pt=float(cell_obj.Range.Font.Size),
                        bold=bool(cell_obj.Range.Font.Bold),
                    )
                row_ir.cells.append(c_ir)
            tbl_ir.rows.append(row_ir)
        elements_by_pos.append((pg_num, t_start, tbl_ir))

    # Process Paragraphs outside tables
    for p in doc.Paragraphs:
        # Check if inside a table (wdWithInTable = 12)
        try:
            if p.Range.Information(12):
                continue
        except Exception:
            pass

        full_txt = p.Range.Text.strip("\r\x07\x0c ")
        if not full_txt:
            continue

        try:
            pg_num = int(p.Range.Information(3))  # wdActiveEndPageNumber
        except Exception:
            pg_num = 1

        p_ir = Paragraph()

        # Alignment mapping
        align_code = p.Alignment
        align_map = {0: "left", 1: "center", 2: "right", 3: "justify"}
        p_ir.alignment = align_map.get(align_code, "left")

        p_ir.space_before_pt = float(p.Format.SpaceBefore)
        p_ir.space_after_pt = float(p.Format.SpaceAfter)
        # Compute typographical line spacing for multi-line paragraphs using layout baselines
        try:
            num_lines = p.Range.ComputeStatistics(1)  # wdStatisticLines = 1
            if num_lines > 1:
                y_pos = []
                for c in range(1, p.Range.Characters.Count):
                    y = p.Range.Characters(c).Information(6)  # wdVerticalPositionRelativeToPage
                    if not y_pos or abs(y - y_pos[-1]) > 2.0:
                        y_pos.append(y)
                diffs = [y_pos[k + 1] - y_pos[k] for k in range(len(y_pos) - 1)]
                p_ir.line_spacing_pt = round(sum(diffs) / len(diffs), 2) if diffs else None
            else:
                p_ir.line_spacing_pt = None
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

        elements_by_pos.append((pg_num, p.Range.Start, p_ir))

    # Sort elements in document order by page and start position
    elements_by_pos.sort(key=lambda item: (item[0], item[1]))
    for pg_num, _, el in elements_by_pos:
        page_idx = max(0, min(pg_num - 1, len(ir_doc.pages) - 1))
        ir_doc.pages[page_idx].blocks.append(el)

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

        # -------------------------------------------------------------
        # 5. word_technical_spec
        # -------------------------------------------------------------
        doc5 = word.Documents.Add()
        p = doc5.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Distributed Consensus Engine: Technical Specification v2.4"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 16
        p.Range.Font.Bold = True
        p.Alignment = 1
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        p = doc5.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "1. Architecture Overview"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 13
        p.Range.Font.Bold = True
        p.Alignment = 0
        p.Format.SpaceBefore = 10.0
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        p = doc5.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "The consensus module maintains an invariant Byzantine fault tolerant log across distributed validator nodes."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Alignment = 0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        p = doc5.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "2. Primitive Interface Definition"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 13
        p.Range.Font.Bold = True
        p.Alignment = 0
        p.Format.SpaceBefore = 10.0
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        p = doc5.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "def commit_block(state: NodeState, tx_root: bytes) -> bool:"
        p.Range.Font.Name = "Consolas"
        p.Range.Font.Size = 10
        p.Alignment = 0
        p.Format.SpaceAfter = 2.0
        p.Range.InsertParagraphAfter()

        p = doc5.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "    return state.apply_payload(tx_root, verify_sig=True)"
        p.Range.Font.Name = "Consolas"
        p.Range.Font.Size = 10
        p.Alignment = 0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        p = doc5.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "- High-throughput pipelined verification."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Alignment = 0
        p.Format.SpaceAfter = 3.0
        p.Range.InsertParagraphAfter()

        p = doc5.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "- Zero-knowledge state transition oracles."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Alignment = 0
        p.Format.SpaceAfter = 8.0
        p.Range.InsertParagraphAfter()

        pdf5_path = str(target_dir / "word_technical_spec.pdf")
        docx5_path = str(target_dir / "word_technical_spec.docx")
        doc5.ExportAsFixedFormat(pdf5_path, 17)
        doc5.SaveAs2(docx5_path)
        ir5 = _extract_word_doc_ir(doc5)
        doc5.Close(False)

        with open(pdf5_path, "rb") as f:
            pdf5_bytes = f.read()
        with open(target_dir / "word_technical_spec.json", "w", encoding="utf-8") as f:
            json.dump(ir5.to_dict(), f, indent=2)

        items.append(CorpusItem("word_technical_spec", "Genuine Word technical specification with monospace code and bullets", pdf5_bytes, ir5))

        # -------------------------------------------------------------
        # 6. word_legal_contract
        # -------------------------------------------------------------
        doc6 = word.Documents.Add()
        p = doc6.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "MUTUAL CONFIDENTIALITY AND NON-DISCLOSURE AGREEMENT"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 14
        p.Range.Font.Bold = True
        p.Alignment = 1
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        p = doc6.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "This Agreement is entered into by and between the Participating Parties subject to the covenants set forth herein."
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 11
        p.Alignment = 3  # justify
        p.Format.SpaceAfter = 8.0
        p.Range.InsertParagraphAfter()

        p = doc6.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "1. Definition of Proprietary Information"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 12
        p.Range.Font.Bold = True
        p.Alignment = 0
        p.Format.SpaceBefore = 8.0
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        p = doc6.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "All data, specifications, algorithms, and models disclosed directly or indirectly shall remain strictly confidential."
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 11
        p.Alignment = 3
        p.Format.SpaceAfter = 8.0
        p.Range.InsertParagraphAfter()

        p = doc6.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "2. Term and Invariant Obligations"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 12
        p.Range.Font.Bold = True
        p.Alignment = 0
        p.Format.SpaceBefore = 8.0
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        p = doc6.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "The recipient agrees to protect all proprietary assets with the same standard of care as its own critical infrastructure."
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 11
        p.Alignment = 3
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        p = doc6.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "IN WITNESS WHEREOF, the parties execute this instrument."
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 11
        p.Range.Font.Italic = True
        p.Alignment = 0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        pdf6_path = str(target_dir / "word_legal_contract.pdf")
        docx6_path = str(target_dir / "word_legal_contract.docx")
        doc6.ExportAsFixedFormat(pdf6_path, 17)
        doc6.SaveAs2(docx6_path)
        ir6 = _extract_word_doc_ir(doc6)
        doc6.Close(False)

        with open(pdf6_path, "rb") as f:
            pdf6_bytes = f.read()
        with open(target_dir / "word_legal_contract.json", "w", encoding="utf-8") as f:
            json.dump(ir6.to_dict(), f, indent=2)

        items.append(CorpusItem("word_legal_contract", "Genuine Word legal contract with justified recitals and numbered clauses", pdf6_bytes, ir6))

        # -------------------------------------------------------------
        # 7. word_medical_summary
        # -------------------------------------------------------------
        doc7 = word.Documents.Add()
        p = doc7.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "CLINICAL OBSERVATION AND ADMISSION SUMMARY"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 15
        p.Range.Font.Bold = True
        p.Alignment = 1
        p.Format.SpaceAfter = 10.0
        p.Range.InsertParagraphAfter()

        p = doc7.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Patient Demographics & Triage Record"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 12
        p.Range.Font.Bold = True
        p.Alignment = 0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        t7 = doc7.Tables.Add(doc7.Paragraphs.Add().Range, 4, 3)
        t7.Borders.Enable = True
        med_data = [
            [("Vital Sign", True), ("Observed Value", True), ("Clinical Status", True)],
            [("Resting Heart Rate", False), ("68 bpm", False), ("Normal Range", False)],
            [("Systolic / Diastolic", False), ("118 / 76 mmHg", False), ("Optimal Baseline", False)],
            [("Core Temperature", False), ("36.8 C", False), ("Normothermic", False)],
        ]
        for r_i, row in enumerate(med_data):
            for c_i, (val, bld) in enumerate(row):
                cell = t7.Cell(r_i + 1, c_i + 1)
                cell.Range.Font.Bold = False
                cell.Range.Font.Italic = False
                cell.Range.Text = val
                cell.Range.Font.Name = "Calibri"
                cell.Range.Font.Size = 10
                cell.Range.Font.Bold = bld
        p_after = doc7.Paragraphs.Add()
        p_after.Range.InsertParagraphBefore()

        p = doc7.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Patient exhibits stable post-operative recovery metrics with no hemodynamic abnormalities."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Alignment = 0
        p.Format.SpaceBefore = 8.0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        p = doc7.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Medical Notice: Confidential clinical summary for attending staff only."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 9
        p.Range.Font.Italic = True
        p.Alignment = 0
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        pdf7_path = str(target_dir / "word_medical_summary.pdf")
        docx7_path = str(target_dir / "word_medical_summary.docx")
        doc7.ExportAsFixedFormat(pdf7_path, 17)
        doc7.SaveAs2(docx7_path)
        ir7 = _extract_word_doc_ir(doc7)
        doc7.Close(False)

        with open(pdf7_path, "rb") as f:
            pdf7_bytes = f.read()
        with open(target_dir / "word_medical_summary.json", "w", encoding="utf-8") as f:
            json.dump(ir7.to_dict(), f, indent=2)

        items.append(CorpusItem("word_medical_summary", "Genuine Word clinical summary with 4x3 vitals matrix and notes", pdf7_bytes, ir7))

        # -------------------------------------------------------------
        # 8. word_corporate_newsletter
        # -------------------------------------------------------------
        doc8 = word.Documents.Add()
        p = doc8.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "GLOBAL INNOVATION & RESEARCH DISPATCH"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 16
        p.Range.Font.Bold = True
        p.Alignment = 1
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        p = doc8.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Quarterly Breakthroughs in Autonomous Synthesis"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Range.Font.Italic = True
        p.Alignment = 1
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        p = doc8.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "The laboratory has achieved deterministic convergence across all empirical validation suites this quarter."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Alignment = 0
        p.Format.SpaceAfter = 8.0
        p.Range.InsertParagraphAfter()

        p = doc8.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Deterministic verification anchors computational truth."
        p.Range.Font.Name = "Georgia"
        p.Range.Font.Size = 13
        p.Range.Font.Italic = True
        p.Alignment = 1
        p.Format.SpaceBefore = 8.0
        p.Format.SpaceAfter = 10.0
        p.Range.InsertParagraphAfter()

        p = doc8.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "- High-precision geometric and structural alignment."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Alignment = 0
        p.Format.SpaceAfter = 3.0
        p.Range.InsertParagraphAfter()

        p = doc8.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "- Seamless cross-platform document reconstruction."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Alignment = 0
        p.Format.SpaceAfter = 8.0
        p.Range.InsertParagraphAfter()

        p = doc8.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Published by the Editorial Board of Evolab Systems."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 10
        p.Alignment = 2  # Right align
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        pdf8_path = str(target_dir / "word_corporate_newsletter.pdf")
        docx8_path = str(target_dir / "word_corporate_newsletter.docx")
        doc8.ExportAsFixedFormat(pdf8_path, 17)
        doc8.SaveAs2(docx8_path)
        ir8 = _extract_word_doc_ir(doc8)
        doc8.Close(False)

        with open(pdf8_path, "rb") as f:
            pdf8_bytes = f.read()
        with open(target_dir / "word_corporate_newsletter.json", "w", encoding="utf-8") as f:
            json.dump(ir8.to_dict(), f, indent=2)

        items.append(CorpusItem("word_corporate_newsletter", "Genuine Word corporate newsletter with pull-quote callout and signoff", pdf8_bytes, ir8))

        # -------------------------------------------------------------
        # 9. word_formal_invoice
        # -------------------------------------------------------------
        doc9 = word.Documents.Add()
        p = doc9.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "COMMERCIAL TAX INVOICE"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 16
        p.Range.Font.Bold = True
        p.Alignment = 0
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        p = doc9.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Invoice No: INV-2026-0913 | Issue Date: September 13, 2026"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 10
        p.Alignment = 0
        p.Format.SpaceAfter = 10.0
        p.Range.InsertParagraphAfter()

        p = doc9.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Billed To: Global Systems Architecture & Computing Corp."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Range.Font.Bold = True
        p.Alignment = 0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        t9 = doc9.Tables.Add(doc9.Paragraphs.Add().Range, 4, 4)
        t9.Borders.Enable = True
        inv_data = [
            [("Item Description", True), ("Quantity", True), ("Unit Rate ($)", True), ("Total Amount ($)", True)],
            [("Neural Compiler License", False), ("1", False), ("1,200.00", False), ("1,200.00", False)],
            [("Verification Engine", False), ("2", False), ("850.00", False), ("1,700.00", False)],
            [("Support & Maintenance", False), ("1", False), ("500.00", False), ("500.00", False)],
        ]
        for r_i, row in enumerate(inv_data):
            for c_i, (val, bld) in enumerate(row):
                cell = t9.Cell(r_i + 1, c_i + 1)
                cell.Range.Font.Bold = False
                cell.Range.Font.Italic = False
                cell.Range.Text = val
                cell.Range.Font.Name = "Calibri"
                cell.Range.Font.Size = 10
                cell.Range.Font.Bold = bld
        p_after = doc9.Paragraphs.Add()
        p_after.Range.InsertParagraphBefore()

        p = doc9.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Total Payable: $3,400.00 USD"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 12
        p.Range.Font.Bold = True
        p.Alignment = 2  # Right align
        p.Format.SpaceBefore = 6.0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        p = doc9.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Payment terms: Net 30 days from date of receipt."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 10
        p.Range.Font.Italic = True
        p.Alignment = 0
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        pdf9_path = str(target_dir / "word_formal_invoice.pdf")
        docx9_path = str(target_dir / "word_formal_invoice.docx")
        doc9.ExportAsFixedFormat(pdf9_path, 17)
        doc9.SaveAs2(docx9_path)
        ir9 = _extract_word_doc_ir(doc9)
        doc9.Close(False)

        with open(pdf9_path, "rb") as f:
            pdf9_bytes = f.read()
        with open(target_dir / "word_formal_invoice.json", "w", encoding="utf-8") as f:
            json.dump(ir9.to_dict(), f, indent=2)

        items.append(CorpusItem("word_formal_invoice", "Genuine Word commercial tax invoice with 4x4 itemized table", pdf9_bytes, ir9))

        # -------------------------------------------------------------
        # 10. word_scientific_abstract
        # -------------------------------------------------------------
        doc10 = word.Documents.Add()
        p = doc10.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Stochastic Boundary Analysis in High-Dimensional Search Spaces"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 15
        p.Range.Font.Bold = True
        p.Alignment = 1
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        p = doc10.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "A. Turing, J. von Neumann, and C. Shannon"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 11
        p.Range.Font.Italic = True
        p.Alignment = 1
        p.Format.SpaceAfter = 3.0
        p.Range.InsertParagraphAfter()

        p = doc10.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Department of Theoretical Cybernetics & Advanced Computation"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 10
        p.Alignment = 1
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        p = doc10.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Abstract"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 12
        p.Range.Font.Bold = True
        p.Alignment = 1
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        p = doc10.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "We formalize asymptotic bounds for stochastic parameters alpha, beta, mu, sigma, and Delta. The empirical landscape guarantees convergence under monotonic selection criteria."
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 11
        p.Alignment = 3  # justify
        p.Format.SpaceAfter = 8.0
        p.Range.InsertParagraphAfter()

        p = doc10.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Keywords: Evolutionary Optimization, Multi-Gate Verification, Canonical Equivalence"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 10
        p.Range.Font.Italic = True
        p.Alignment = 0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        pdf10_path = str(target_dir / "word_scientific_abstract.pdf")
        docx10_path = str(target_dir / "word_scientific_abstract.docx")
        doc10.ExportAsFixedFormat(pdf10_path, 17)
        doc10.SaveAs2(docx10_path)
        ir10 = _extract_word_doc_ir(doc10)
        doc10.Close(False)

        with open(pdf10_path, "rb") as f:
            pdf10_bytes = f.read()
        with open(target_dir / "word_scientific_abstract.json", "w", encoding="utf-8") as f:
            json.dump(ir10.to_dict(), f, indent=2)

        items.append(CorpusItem("word_scientific_abstract", "Genuine Word scientific abstract with Greek parameters and keywords", pdf10_bytes, ir10))

        # -------------------------------------------------------------
        # 11. word_multi_page_report
        # -------------------------------------------------------------
        doc11 = word.Documents.Add()
        p = doc11.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "COMPREHENSIVE MULTI-PAGE RESEARCH DIRECTIVE"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 16
        p.Range.Font.Bold = True
        p.Alignment = 1
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        p = doc11.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Section 1: Initial Findings and Empirical Framework"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 13
        p.Range.Font.Bold = True
        p.Alignment = 0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        p = doc11.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "This first volume details the methodological foundation of our multi-gate validation architecture."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Alignment = 0
        p.Format.SpaceAfter = 12.0
        p.Range.InsertParagraphAfter()

        # Page Break
        p_break = doc11.Paragraphs.Add()
        p_break.Range.InsertBreak(7)  # wdPageBreak

        p = doc11.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Section 2: Concluding Recommendations and Future Horizons"
        p.Range.Font.Name = "Arial"
        p.Range.Font.Size = 13
        p.Range.Font.Bold = True
        p.Alignment = 0
        p.Format.SpaceAfter = 6.0
        p.Range.InsertParagraphAfter()

        p = doc11.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "The subsequent phase establishes rigorous cross-seed statistical evaluation and systematic baseline ablation."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Alignment = 0
        p.Format.SpaceAfter = 8.0
        p.Range.InsertParagraphAfter()

        p = doc11.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Approved by the Executive Directorate of Autonomous Systems."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 11
        p.Range.Font.Bold = True
        p.Alignment = 2
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        pdf11_path = str(target_dir / "word_multi_page_report.pdf")
        docx11_path = str(target_dir / "word_multi_page_report.docx")
        doc11.ExportAsFixedFormat(pdf11_path, 17)
        doc11.SaveAs2(docx11_path)
        ir11 = _extract_word_doc_ir(doc11)
        doc11.Close(False)

        with open(pdf11_path, "rb") as f:
            pdf11_bytes = f.read()
        with open(target_dir / "word_multi_page_report.json", "w", encoding="utf-8") as f:
            json.dump(ir11.to_dict(), f, indent=2)

        items.append(CorpusItem("word_multi_page_report", "Genuine Word multi-page directive with page breaks and sections", pdf11_bytes, ir11))

        # -------------------------------------------------------------
        # 12. word_tabular_matrix
        # -------------------------------------------------------------
        doc12 = word.Documents.Add()
        p = doc12.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Computational Matrix and Benchmarking Grid"
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 14
        p.Range.Font.Bold = True
        p.Alignment = 1
        p.Format.SpaceAfter = 10.0
        p.Range.InsertParagraphAfter()

        t12 = doc12.Tables.Add(doc12.Paragraphs.Add().Range, 5, 4)
        t12.Borders.Enable = True
        mat_data = [
            [("Benchmark ID", True), ("Kernel Mode", True), ("Throughput (MOps)", True), ("Validation Status", True)],
            [("BENCH-01", False), ("FP32 SIMD", False), ("452.8", False), ("PASSED", False)],
            [("BENCH-02", False), ("INT8 Tensor", False), ("1280.4", False), ("PASSED", False)],
            [("BENCH-03", False), ("BFloat16 Engine", False), ("890.1", False), ("PASSED", False)],
            [("BENCH-04", False), ("Sparse FP16", False), ("2100.5", False), ("PASSED", False)],
        ]
        for r_i, row in enumerate(mat_data):
            for c_i, (val, bld) in enumerate(row):
                cell = t12.Cell(r_i + 1, c_i + 1)
                cell.Range.Font.Bold = False
                cell.Range.Font.Italic = False
                cell.Range.Text = val
                cell.Range.Font.Name = "Calibri"
                cell.Range.Font.Size = 10
                cell.Range.Font.Bold = bld
        p_after = doc12.Paragraphs.Add()
        p_after.Range.InsertParagraphBefore()

        p = doc12.Paragraphs.Add()
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.Text = "Matrix represents verified hardware performance under deterministic test vectors."
        p.Range.Font.Name = "Calibri"
        p.Range.Font.Size = 10
        p.Range.Font.Italic = True
        p.Alignment = 0
        p.Format.SpaceBefore = 8.0
        p.Format.SpaceAfter = 4.0
        p.Range.InsertParagraphAfter()

        pdf12_path = str(target_dir / "word_tabular_matrix.pdf")
        docx12_path = str(target_dir / "word_tabular_matrix.docx")
        doc12.ExportAsFixedFormat(pdf12_path, 17)
        doc12.SaveAs2(docx12_path)
        ir12 = _extract_word_doc_ir(doc12)
        doc12.Close(False)

        with open(pdf12_path, "rb") as f:
            pdf12_bytes = f.read()
        with open(target_dir / "word_tabular_matrix.json", "w", encoding="utf-8") as f:
            json.dump(ir12.to_dict(), f, indent=2)

        items.append(CorpusItem("word_tabular_matrix", "Genuine Word 5x4 benchmarking data matrix with diverse headers", pdf12_bytes, ir12))

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
        ("word_technical_spec", "Genuine Word technical specification with monospace code and bullets"),
        ("word_legal_contract", "Genuine Word legal contract with justified recitals and numbered clauses"),
        ("word_medical_summary", "Genuine Word clinical summary with 4x3 vitals matrix and notes"),
        ("word_corporate_newsletter", "Genuine Word corporate newsletter with pull-quote callout and signoff"),
        ("word_formal_invoice", "Genuine Word commercial tax invoice with 4x4 itemized table"),
        ("word_scientific_abstract", "Genuine Word scientific abstract with Greek parameters and keywords"),
        ("word_multi_page_report", "Genuine Word multi-page directive with page breaks and sections"),
        ("word_tabular_matrix", "Genuine Word 5x4 benchmarking data matrix with diverse headers"),
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


if __name__ == "__main__":
    print("Regenerating real Word holdout documents (N=12)...")
    res = generate_real_word_holdout()
    print(f"Generated {len(res)} holdout documents successfully.")
