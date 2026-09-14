"""
rtf_parser.py — Pure Python RTF Lexer and AST-to-IR Parser.

Parses standard Microsoft RTF files into the canonical Document IR.
Tokenizes control words, handles group scoping, parses font and color tables,
and reconstructs pages, paragraphs, runs, and table matrices.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .ir import Color, Document, Page, Paragraph, Row, Run, Table


@dataclass
class _FormatState:
    """Formatting state snapshot for RTF group scoping."""
    font_id: int = 0
    font_size_pt: float = 11.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color_idx: int = 0
    alignment: str = "left"
    space_before_pt: float = 0.0
    space_after_pt: float = 0.0
    line_spacing_pt: float | None = None
    slmult: bool = False
    in_table: bool = False
    ignore_group: bool = False


class RTFParser:
    """Pure Python RTF Parser to canonical Document IR."""

    def __init__(self, rtf_text: str) -> None:
        self.raw = rtf_text
        self.font_table: dict[int, str] = {}
        self.color_table: dict[int, Color] = {}

    def parse(self) -> Document:
        """Parses the RTF string into a Document IR."""
        doc = Document()
        current_page = doc.add_page()

        # Extract font table and color table first
        self._parse_headers()

        # State stack
        state = _FormatState()
        stack: list[_FormatState] = []

        current_para = Paragraph()
        current_run_text: list[str] = []

        # Table accumulation state
        current_table: Table | None = None
        current_row: Row | None = None
        current_cell_paras: list[Paragraph] = []

        i = 0
        n = len(self.raw)

        def flush_run() -> None:
            nonlocal current_run_text
            if current_run_text and not state.ignore_group:
                txt = "".join(current_run_text)
                if txt:
                    font_name = self.font_table.get(state.font_id, "Calibri")
                    col = self.color_table.get(state.color_idx, Color())
                    run = Run(
                        text=txt,
                        font=font_name,
                        font_size_pt=state.font_size_pt,
                        bold=state.bold,
                        italic=state.italic,
                        underline=state.underline,
                        color=col,
                    )
                    current_para.runs.append(run)
                current_run_text = []

        def flush_para() -> None:
            nonlocal current_para, current_table, current_row, current_cell_paras
            flush_run()
            if current_para.runs or current_para.space_after_pt > 0:
                current_para.alignment = state.alignment
                current_para.space_before_pt = state.space_before_pt
                current_para.space_after_pt = state.space_after_pt
                current_para.line_spacing_pt = state.line_spacing_pt

                if state.in_table:
                    current_cell_paras.append(current_para)
                else:
                    if current_table is not None and current_table.rows:
                        current_page.blocks.append(current_table)
                        current_table = None
                        current_row = None
                    current_page.blocks.append(current_para)
            current_para = Paragraph()

        while i < n:
            ch = self.raw[i]

            if ch == "{":
                flush_run()
                stack.append(
                    _FormatState(
                        font_id=state.font_id,
                        font_size_pt=state.font_size_pt,
                        bold=state.bold,
                        italic=state.italic,
                        underline=state.underline,
                        color_idx=state.color_idx,
                        alignment=state.alignment,
                        space_before_pt=state.space_before_pt,
                        space_after_pt=state.space_after_pt,
                        line_spacing_pt=state.line_spacing_pt,
                        in_table=state.in_table,
                        ignore_group=state.ignore_group,
                    )
                )
                i += 1
                # Check for destination groups like {\*...} or {\fonttbl...}
                if i < n and self.raw[i] == "\\":
                    # Peek word
                    m = re.match(r"\\(\*|[a-zA-Z]+)", self.raw[i:])
                    if m:
                        word = m.group(1)
                        if word in ("fonttbl", "colortbl", "stylesheet", "info", "*"):
                            state.ignore_group = True

            elif ch == "}":
                flush_run()
                if stack:
                    state = stack.pop()
                i += 1

            elif ch == "\\":
                i += 1
                if i >= n:
                    break

                # Control symbols: \\, \{, \}, \~, \_
                sym = self.raw[i]
                if sym in ("\\", "{", "}"):
                    if not state.ignore_group:
                        current_run_text.append(sym)
                    i += 1
                    continue
                elif sym == "~":
                    if not state.ignore_group:
                        current_run_text.append(" ")
                    i += 1
                    continue
                elif sym == "'":
                    # Hex byte \'hh
                    hex_val = self.raw[i + 1 : i + 3]
                    try:
                        byte_val = int(hex_val, 16)
                        char = bytes([byte_val]).decode("cp1252", errors="replace")
                        if not state.ignore_group:
                            current_run_text.append(char)
                    except Exception:
                        pass
                    i += 3
                    continue

                # Control word: letters + optional digits
                m = re.match(r"([a-zA-Z]+)(-?[0-9]*)", self.raw[i:])
                if not m:
                    i += 1
                    continue

                word = m.group(1)
                arg_str = m.group(2)
                arg = int(arg_str) if arg_str else None
                i += len(m.group(0))

                # Skip single trailing delimiter space
                if i < n and self.raw[i] == " ":
                    i += 1

                if state.ignore_group:
                    continue

                # Process control word
                if word == "b":
                    flush_run()
                    state.bold = (arg is None or arg != 0)
                elif word == "i":
                    flush_run()
                    state.italic = (arg is None or arg != 0)
                elif word == "ul":
                    flush_run()
                    state.underline = True
                elif word == "ulnone":
                    flush_run()
                    state.underline = False
                elif word == "fs":
                    flush_run()
                    if arg is not None:
                        state.font_size_pt = arg / 2.0
                elif word == "f":
                    flush_run()
                    if arg is not None:
                        state.font_id = arg
                elif word == "cf":
                    flush_run()
                    if arg is not None:
                        state.color_idx = arg
                elif word == "par":
                    flush_para()
                elif word == "pard":
                    flush_run()
                    state.in_table = False
                    state.alignment = "left"
                    state.space_before_pt = 0.0
                    state.space_after_pt = 0.0
                    state.line_spacing_pt = None
                    state.slmult = False
                    state.bold = False
                    state.italic = False
                    state.underline = False
                elif word == "ql":
                    state.alignment = "left"
                elif word == "qc":
                    state.alignment = "center"
                elif word == "qr":
                    state.alignment = "right"
                elif word == "qj":
                    state.alignment = "justify"
                elif word == "sa":
                    if arg is not None:
                        state.space_after_pt = arg / 20.0
                elif word == "sb":
                    if arg is not None:
                        state.space_before_pt = arg / 20.0
                elif word == "slmult":
                    state.slmult = (arg is None or arg != 0)
                elif word == "sl":
                    if arg is not None:
                        if state.slmult:
                            state.line_spacing_pt = (abs(arg) / 240.0) * 12.0
                        else:
                            state.line_spacing_pt = abs(arg) / 20.0
                elif word == "page":
                    flush_para()
                    current_page = doc.add_page()
                elif word == "trowd":
                    flush_para()
                    state.in_table = True
                    if current_table is None:
                        current_table = Table()
                    current_row = current_table.add_row()
                elif word == "trql":
                    if current_table is not None:
                        current_table.alignment = "left"
                elif word == "trqc":
                    if current_table is not None:
                        current_table.alignment = "center"
                elif word == "trqr":
                    if current_table is not None:
                        current_table.alignment = "right"
                elif word == "intbl":
                    state.in_table = True
                elif word == "cell":
                    flush_run()
                    if current_para.runs:
                        current_cell_paras.append(current_para)
                        current_para = Paragraph()
                    if current_row is not None:
                        cell = current_row.add_cell()
                        cell.content = list(current_cell_paras)
                    current_cell_paras = []
                elif word == "row":
                    flush_para()
                    if current_table is not None:
                        # Row completed
                        pass
                elif word == "u":
                    # Unicode character \uN?
                    if arg is not None:
                        char_code = arg if arg >= 0 else arg + 65536
                        current_run_text.append(chr(char_code))
                        # Skip replacement char if next
                        if i < n and self.raw[i] == "?":
                            i += 1

            elif ch in ("\r", "\n"):
                # Newlines in RTF are treated as whitespace / formatting separators
                i += 1

            else:
                if not state.ignore_group:
                    current_run_text.append(ch)
                i += 1

        # Final flush
        flush_para()

        # If a table was in progress, add it to page blocks
        if current_table is not None and current_table.rows:
            current_page.blocks.append(current_table)

        # Remove trailing empty paragraphs from empty pages
        for page in doc.pages:
            while page.blocks and isinstance(page.blocks[-1], Paragraph) and not page.blocks[-1].runs:
                page.blocks.pop()

        return doc

    def _parse_headers(self) -> None:
        """Extracts font and color tables from RTF header."""
        # 1. Font table extraction: find all font entries directly
        font_entries = re.findall(r"\{\\f(\d+)[^}]*?\s+([^;{}]+);\}", self.raw)
        for f_idx_str, f_name in font_entries:
            clean_name = re.sub(r"^[A-Z]{6}\+", "", f_name).strip()
            self.font_table[int(f_idx_str)] = clean_name

        # 2. Color table extraction: {\colortbl ;...}
        ct_match = re.search(r"\{\\colortbl\s*(.*?)\}", self.raw, re.DOTALL)
        if ct_match:
            table_content = ct_match.group(1)
            entries = table_content.split(";")
            color_idx = 1
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue
                r_m = re.search(r"\\red(\d+)", entry)
                g_m = re.search(r"\\green(\d+)", entry)
                b_m = re.search(r"\\blue(\d+)", entry)
                if r_m and g_m and b_m:
                    self.color_table[color_idx] = Color(
                        r=int(r_m.group(1)),
                        g=int(g_m.group(1)),
                        b=int(b_m.group(1)),
                    )
                    color_idx += 1


def parse_rtf(rtf_text: str) -> Document:
    """Convenience functional helper to parse an RTF string into a Document IR."""
    return RTFParser(rtf_text).parse()
