"""Unit tests for Pure Python RTF Parser."""
from __future__ import annotations

import pytest

from experimental.pdf2rtf.ir import Document
from experimental.pdf2rtf.rtf_parser import RTFParser, parse_rtf


def test_parse_simple_rtf():
    rtf_code = (
        r"{\rtf1\ansi\deff0"
        r"{\fonttbl{\f0\fnil Calibri;}}"
        r"\pard\ql Hello World!\par"
        r"}"
    )
    doc = parse_rtf(rtf_code)

    assert len(doc.pages) == 1
    assert len(doc.pages[0].blocks) == 1
    p = doc.pages[0].blocks[0]
    assert p.plain_text.strip() == "Hello World!"
    assert p.alignment == "left"


def test_parse_font_and_color_tables():
    rtf_code = (
        r"{\rtf1\ansi\deff0"
        r"{\fonttbl{\f0\fnil Calibri;}{\f1\fnil Times New Roman;}}"
        r"{\colortbl ;\red255\green0\blue0;\red0\green128\blue0;}"
        r"\pard\f1\fs24\cf1 Red Times New Roman Text\par"
        r"}"
    )
    doc = parse_rtf(rtf_code)

    p = doc.pages[0].blocks[0]
    assert p.plain_text.strip() == "Red Times New Roman Text"
    run = p.runs[0]
    assert run.font == "Times New Roman"
    assert run.font_size_pt == 12.0  # \fs24 -> 12pt
    assert run.color.r == 255
    assert run.color.g == 0
    assert run.color.b == 0


def test_parse_formatting_styles_bold_italic():
    rtf_code = (
        r"{\rtf1\ansi\deff0"
        r"\pard Normal text {\b Bold text} {\i Italic text} {\b\i BoldItalic}\par"
        r"}"
    )
    doc = parse_rtf(rtf_code)
    p = doc.pages[0].blocks[0]

    bold_runs = [r for r in p.runs if r.bold and not r.italic]
    italic_runs = [r for r in p.runs if r.italic and not r.bold]
    bold_italic_runs = [r for r in p.runs if r.bold and r.italic]

    assert any("Bold text" in r.text for r in bold_runs)
    assert any("Italic text" in r.text for r in italic_runs)
    assert any("BoldItalic" in r.text for r in bold_italic_runs)


def test_parse_table():
    rtf_code = (
        r"{\rtf1\ansi\deff0"
        r"\trowd\cellx1000\cellx2000"
        r"\intbl A1\cell\intbl B1\cell\row"
        r"\trowd\cellx1000\cellx2000"
        r"\intbl A2\cell\intbl B2\cell\row"
        r"\pard"
        r"}"
    )
    doc = parse_rtf(rtf_code)

    tables = doc.all_tables
    assert len(tables) == 1
    t = tables[0]
    assert t.row_count == 2
    assert t.col_count == 2
    assert "A1 | B1" in t.plain_text
    assert "A2 | B2" in t.plain_text
