"""
ir.py — Canonical Intermediate Representation (IR) for Document Conversion.

Defines an immutable, normalized document object model representing documents,
pages, paragraphs, text runs, and tables. Acts as the universal semantic bridge
between PDF extraction, RTF emission, and Ground-Truth verification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union


@dataclass(frozen=True)
class Color:
    """RGB Color representation."""
    r: int = 0
    g: int = 0
    b: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", max(0, min(255, int(self.r))))
        object.__setattr__(self, "g", max(0, min(255, int(self.g))))
        object.__setattr__(self, "b", max(0, min(255, int(self.b))))

    @property
    def is_black(self) -> bool:
        return self.r == 0 and self.g == 0 and self.b == 0

    def to_dict(self) -> dict[str, int]:
        return {"r": self.r, "g": self.g, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Color:
        return cls(r=d.get("r", 0), g=d.get("g", 0), b=d.get("b", 0))


@dataclass
class Run:
    """Atomic formatted text span."""
    text: str
    font: str = "Calibri"
    font_size_pt: float = 11.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color: Color = field(default_factory=Color)

    def clean_font_name(self) -> str:
        """Strips Word subset prefix (e.g. BAAAAA+Calibri -> Calibri)."""
        return re.sub(r"^[A-Z]{6}\+", "", self.font).strip()

    @property
    def is_whitespace(self) -> bool:
        return len(self.text.strip()) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "font": self.clean_font_name(),
            "font_size_pt": round(self.font_size_pt, 2),
            "bold": self.bold,
            "italic": self.italic,
            "underline": self.underline,
            "color": self.color.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Run:
        return cls(
            text=d.get("text", ""),
            font=d.get("font", "Calibri"),
            font_size_pt=float(d.get("font_size_pt", 11.0)),
            bold=bool(d.get("bold", False)),
            italic=bool(d.get("italic", False)),
            underline=bool(d.get("underline", False)),
            color=Color.from_dict(d.get("color", {})),
        )


@dataclass
class Paragraph:
    """Paragraph container composed of formatted runs."""
    runs: list[Run] = field(default_factory=list)
    alignment: str = "left"  # left | center | right | justify
    space_before_pt: float = 0.0
    space_after_pt: float = 0.0
    line_spacing_pt: float | None = None

    @property
    def plain_text(self) -> str:
        return "".join(r.text for r in self.runs)

    def add_run(
        self,
        text: str,
        font: str = "Calibri",
        font_size_pt: float = 11.0,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        color: Color | None = None,
    ) -> Run:
        run = Run(
            text=text,
            font=font,
            font_size_pt=font_size_pt,
            bold=bold,
            italic=italic,
            underline=underline,
            color=color or Color(),
        )
        self.runs.append(run)
        return run

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "paragraph",
            "alignment": self.alignment,
            "space_before_pt": round(self.space_before_pt, 2),
            "space_after_pt": round(self.space_after_pt, 2),
            "line_spacing_pt": round(self.line_spacing_pt, 2) if self.line_spacing_pt is not None else None,
            "runs": [r.to_dict() for r in self.runs],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Paragraph:
        p = cls(
            alignment=d.get("alignment", "left"),
            space_before_pt=float(d.get("space_before_pt", 0.0)),
            space_after_pt=float(d.get("space_after_pt", 0.0)),
            line_spacing_pt=float(d["line_spacing_pt"]) if d.get("line_spacing_pt") is not None else None,
        )
        p.runs = [Run.from_dict(r) for r in d.get("runs", [])]
        return p


@dataclass
class Cell:
    """Table cell containing one or more paragraphs."""
    content: list[Paragraph] = field(default_factory=list)
    width_twips: int = 1440  # 1 inch default (72pt * 20)
    borders: dict[str, bool] = field(
        default_factory=lambda: {"top": True, "bottom": True, "left": True, "right": True}
    )

    @property
    def plain_text(self) -> str:
        return "\n".join(p.plain_text for p in self.content)

    def add_paragraph(self, text: str = "", **kwargs: Any) -> Paragraph:
        p = Paragraph(**kwargs)
        if text:
            p.add_run(text)
        self.content.append(p)
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_twips": self.width_twips,
            "borders": dict(self.borders),
            "content": [p.to_dict() for p in self.content],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Cell:
        c = cls(
            width_twips=int(d.get("width_twips", 1440)),
            borders=d.get("borders", {"top": True, "bottom": True, "left": True, "right": True}),
        )
        c.content = [Paragraph.from_dict(p) for p in d.get("content", [])]
        return c


@dataclass
class Row:
    """Table row containing a sequence of cells."""
    cells: list[Cell] = field(default_factory=list)
    is_header: bool = False
    height_twips: int | None = None

    def add_cell(self, text: str = "", width_twips: int = 1440) -> Cell:
        cell = Cell(width_twips=width_twips)
        if text:
            cell.add_paragraph(text)
        self.cells.append(cell)
        return cell

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_header": self.is_header,
            "height_twips": self.height_twips,
            "cells": [c.to_dict() for c in self.cells],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Row:
        r = cls(
            is_header=bool(d.get("is_header", False)),
            height_twips=int(d["height_twips"]) if d.get("height_twips") is not None else None,
        )
        r.cells = [Cell.from_dict(c) for c in d.get("cells", [])]
        return r


@dataclass
class Table:
    """Table structure composed of rows."""
    rows: list[Row] = field(default_factory=list)
    alignment: str = "center"  # left | center | right

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return max((len(r.cells) for r in self.rows), default=0)

    @property
    def plain_text(self) -> str:
        lines = []
        for r in self.rows:
            lines.append(" | ".join(c.plain_text.strip() for c in r.cells))
        return "\n".join(lines)

    def add_row(self, is_header: bool = False) -> Row:
        row = Row(is_header=is_header)
        self.rows.append(row)
        return row

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "table",
            "alignment": self.alignment,
            "rows": [r.to_dict() for r in self.rows],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Table:
        t = cls(alignment=d.get("alignment", "center"))
        t.rows = [Row.from_dict(r) for r in d.get("rows", [])]
        return t


Block = Union[Paragraph, Table]


@dataclass
class Page:
    """Document page containing structured content blocks."""
    number: int = 1
    width_pts: float = 612.0   # Letter width (8.5 * 72)
    height_pts: float = 792.0  # Letter height (11.0 * 72)
    blocks: list[Block] = field(default_factory=list)

    @property
    def plain_text(self) -> str:
        return "\n\n".join(b.plain_text for b in self.blocks)

    def add_paragraph(self, text: str = "", **kwargs: Any) -> Paragraph:
        p = Paragraph(**kwargs)
        if text:
            p.add_run(text)
        self.blocks.append(p)
        return p

    def add_table(self, alignment: str = "center") -> Table:
        tbl = Table(alignment=alignment)
        self.blocks.append(tbl)
        return tbl

    def to_dict(self) -> dict[str, Any]:
        serialized_blocks = []
        for b in self.blocks:
            if isinstance(b, Paragraph):
                serialized_blocks.append(b.to_dict())
            elif isinstance(b, Table):
                serialized_blocks.append(b.to_dict())
        return {
            "number": self.number,
            "width_pts": round(self.width_pts, 2),
            "height_pts": round(self.height_pts, 2),
            "blocks": serialized_blocks,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Page:
        page = cls(
            number=int(d.get("number", 1)),
            width_pts=float(d.get("width_pts", 612.0)),
            height_pts=float(d.get("height_pts", 792.0)),
        )
        for b in d.get("blocks", []):
            if b.get("type") == "table":
                page.blocks.append(Table.from_dict(b))
            else:
                page.blocks.append(Paragraph.from_dict(b))
        return page


@dataclass
class Document:
    """Root document container representing the universal intermediate representation."""
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plain_text(self) -> str:
        return "\n\n--- PAGE BREAK ---\n\n".join(p.plain_text for p in self.pages)

    @property
    def all_paragraphs(self) -> list[Paragraph]:
        paras: list[Paragraph] = []
        for page in self.pages:
            for b in page.blocks:
                if isinstance(b, Paragraph):
                    paras.append(b)
                elif isinstance(b, Table):
                    for r in b.rows:
                        for c in r.cells:
                            paras.extend(c.content)
        return paras

    @property
    def all_tables(self) -> list[Table]:
        tables: list[Table] = []
        for page in self.pages:
            for b in page.blocks:
                if isinstance(b, Table):
                    tables.append(b)
        return tables

    @property
    def fonts_used(self) -> set[str]:
        fonts = set()
        for p in self.all_paragraphs:
            for r in p.runs:
                clean = r.clean_font_name()
                if clean:
                    fonts.add(clean)
        return fonts

    @property
    def colors_used(self) -> set[Color]:
        colors = set()
        for p in self.all_paragraphs:
            for r in p.runs:
                colors.add(r.color)
        return colors

    def add_page(self, width_pts: float = 612.0, height_pts: float = 792.0) -> Page:
        page = Page(
            number=len(self.pages) + 1,
            width_pts=width_pts,
            height_pts=height_pts,
        )
        self.pages.append(page)
        return page

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "pages": [p.to_dict() for p in self.pages],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        doc = cls(metadata=d.get("metadata", {}))
        doc.pages = [Page.from_dict(p) for p in d.get("pages", [])]
        return doc
