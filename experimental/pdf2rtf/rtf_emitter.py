"""
rtf_emitter.py — Pure Python Microsoft RTF Specification 1.9 Emitter.

Translates the canonical Document IR into compliant, standard RTF files
supporting font tables, color tables, twip-accurate page dimensions,
paragraph alignments, styles, and full table matrices.
"""
from __future__ import annotations

import io
from typing import TextIO

from .ir import Color, Document, Page, Paragraph, Run, Table


class RTFEmitter:
    """Zero-dependency Microsoft RTF 1.9 compliant code generator."""

    def __init__(self, doc: Document) -> None:
        self.doc = doc
        self.font_table: dict[str, int] = {}
        self.color_table: dict[Color, int] = {}
        self._build_tables()

    def _build_tables(self) -> None:
        """Pre-indexes unique fonts and non-black colors across the document."""
        # Ensure default font is at index 0
        default_font = "Calibri"
        self.font_table[default_font] = 0

        # Discover all unique fonts
        for font in sorted(self.doc.fonts_used):
            if font not in self.font_table:
                self.font_table[font] = len(self.font_table)

        # Discover all unique colors (0 is auto/black)
        for col in sorted(self.doc.colors_used, key=lambda c: (c.r, c.g, c.b)):
            if not col.is_black and col not in self.color_table:
                # 1-indexed in RTF color table (index 0 is default/auto)
                self.color_table[col] = len(self.color_table) + 1

    @staticmethod
    def escape_text(text: str) -> str:
        """Escapes raw text for RTF including control chars and Unicode surrogates."""
        out = []
        for ch in text:
            code = ord(ch)
            if ch == "\\":
                out.append(r"\\")
            elif ch == "{":
                out.append(r"\{")
            elif ch == "}":
                out.append(r"\}")
            elif ch == "\n":
                out.append(r"\line ")
            elif ch == "\t":
                out.append(r"\tab ")
            elif 32 <= code <= 126:
                out.append(ch)
            else:
                # Standard RTF Unicode encoding \uN?
                if code > 32767:
                    signed_code = code - 65536
                else:
                    signed_code = code
                out.append(f"\\u{signed_code}?")
        return "".join(out)

    def emit(self) -> str:
        """Emits complete RTF string."""
        buf = io.StringIO()
        self.write(buf)
        return buf.getvalue()

    def write(self, out: TextIO) -> None:
        """Streams the RTF document to a file or stream."""
        # 1. Header and Charset
        out.write(r"{\rtf1\ansi\ansicpg1252\deff0\nouicompat" + "\n")

        # 2. Font Table
        out.write(r"{\fonttbl" + "\n")
        for font_name, idx in sorted(self.font_table.items(), key=lambda x: x[1]):
            # Assign font family heuristic
            family = "fswiss" if "arial" in font_name.lower() or "calibri" in font_name.lower() else "froman"
            out.write(f"{{\\f{idx}\\{family}\\fcharset0 {font_name};}}\n")
        out.write("}\n")

        # 3. Color Table
        if self.color_table:
            out.write(r"{\colortbl ;")
            for col, _ in sorted(self.color_table.items(), key=lambda x: x[1]):
                out.write(f"\\red{col.r}\\green{col.g}\\blue{col.b};")
            out.write("}\n")

        # 4. Global View and Page Setup
        first_page = self.doc.pages[0] if self.doc.pages else Page()
        paper_w_twips = int(round(first_page.width_pts * 20.0))
        paper_h_twips = int(round(first_page.height_pts * 20.0))
        # Default 1-inch margins (1440 twips)
        out.write(
            f"\\viewkind4\\uc1\\paperw{paper_w_twips}\\paperh{paper_h_twips}"
            r"\margl1440\margr1440\margt1440\margb1440" + "\n"
        )

        # 5. Emit Pages & Blocks
        for p_idx, page in enumerate(self.doc.pages):
            if p_idx > 0:
                out.write(r"\page" + "\n")

            for block in page.blocks:
                if isinstance(block, Paragraph):
                    self._emit_paragraph(out, block)
                elif isinstance(block, Table):
                    self._emit_table(out, block)

        out.write("}\n")

    def _emit_paragraph(self, out: TextIO, para: Paragraph, in_table: bool = False) -> None:
        """Renders a single paragraph and its styled runs."""
        # Reset paragraph formatting
        out.write(r"\pard")
        if in_table:
            out.write(r"\intbl")

        # Alignment
        align_map = {
            "left": r"\ql",
            "center": r"\qc",
            "right": r"\qr",
            "justify": r"\qj",
        }
        out.write(align_map.get(para.alignment.lower(), r"\ql"))

        # Spacing
        if para.space_before_pt > 0:
            out.write(f"\\sb{int(round(para.space_before_pt * 20.0))}")
        if para.space_after_pt > 0:
            out.write(f"\\sa{int(round(para.space_after_pt * 20.0))}")
        if para.line_spacing_pt is not None and para.line_spacing_pt > 0:
            twips = int(round(para.line_spacing_pt * 20.0))
            out.write(f"\\sl{twips}\\slmult0")

        out.write(" ")

        # Runs
        for run in para.runs:
            self._emit_run(out, run)

        if not in_table:
            out.write(r"\par" + "\n")

    def _emit_run(self, out: TextIO, run: Run) -> None:
        """Renders an inline formatted run."""
        clean_font = run.clean_font_name()
        f_idx = self.font_table.get(clean_font, 0)
        # Font size in half-points: 11pt -> \fs22
        fs = int(round(run.font_size_pt * 2.0))

        out.write(f"{{\\f{f_idx}\\fs{fs}")

        if run.bold:
            out.write(r"\b")
        if run.italic:
            out.write(r"\i")
        if run.underline:
            out.write(r"\ul")
        if not run.color.is_black and run.color in self.color_table:
            c_idx = self.color_table[run.color]
            out.write(f"\\cf{c_idx}")

        out.write(" ")
        out.write(self.escape_text(run.text))
        out.write("}")

    def _emit_table(self, out: TextIO, table: Table) -> None:
        """Renders an RTF table grid row by row."""
        for row in table.rows:
            out.write(r"\trowd\trgaph108")

            # Table alignment
            if table.alignment == "center":
                out.write(r"\trqc")
            elif table.alignment == "right":
                out.write(r"\trqr")
            else:
                out.write(r"\trql")

            if row.is_header:
                out.write(r"\trhdr")

            # Calculate cumulative cell boundaries in twips
            acc_x = 0
            for cell in row.cells:
                acc_x += cell.width_twips
                # Cell borders
                if cell.borders.get("top", True):
                    out.write(r"\clbrdrt\brdrs\brdrw10")
                if cell.borders.get("bottom", True):
                    out.write(r"\clbrdrb\brdrs\brdrw10")
                if cell.borders.get("left", True):
                    out.write(r"\clbrdrl\brdrs\brdrw10")
                if cell.borders.get("right", True):
                    out.write(r"\clbrdrr\brdrs\brdrw10")
                out.write(f"\\cellx{acc_x}")

            out.write("\n")

            # Emit cell content
            for cell in row.cells:
                if not cell.content:
                    # Empty cell still needs \intbl ... \cell
                    out.write(r"\pard\intbl \cell" + "\n")
                else:
                    for p in cell.content:
                        self._emit_paragraph(out, p, in_table=True)
                    out.write(r"\cell" + "\n")

            out.write(r"\row" + "\n")

        # Reset paragraph formatting after table
        out.write(r"\pard" + "\n")


def emit_rtf(doc: Document) -> str:
    """Convenience functional helper to emit an RTF string from a Document IR."""
    return RTFEmitter(doc).emit()
