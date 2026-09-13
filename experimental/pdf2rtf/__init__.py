"""
pdf2rtf — Self-Calibrating Hardened PDF-to-RTF Conversion Subsystem.
"""
from __future__ import annotations

from .ir import (
    Block,
    Color,
    Document,
    Page,
    Paragraph,
    Row,
    Cell,
    Run,
    Table,
)
from .rtf_emitter import RTFEmitter, emit_rtf
from .rtf_parser import RTFParser, parse_rtf
from .pdf_extractor import (
    PDFExtractor,
    clean_base_font_family,
    extract_pdf,
    strip_font_subset_prefix,
)
from .equivalence import (
    GateResult,
    EquivalenceReport,
    TextIntegrityGate,
    StructureIntegrityGate,
    TableOracle,
    FormattingIntegrityGate,
    VisualDiffOracle,
    MultiGateVerifier,
)

from .genome import PARAM_BOUNDS, ProfileGenome, ProfilePolicy
from .adapter import PDF2RTFAdapter, PDF2RTFEvaluator, PDF2RTFSpec

__all__ = [
    "Block",
    "Color",
    "Document",
    "Page",
    "Paragraph",
    "Row",
    "Cell",
    "Run",
    "Table",
    "RTFEmitter",
    "emit_rtf",
    "RTFParser",
    "parse_rtf",
    "PDFExtractor",
    "extract_pdf",
    "strip_font_subset_prefix",
    "clean_base_font_family",
    "GateResult",
    "EquivalenceReport",
    "TextIntegrityGate",
    "StructureIntegrityGate",
    "TableOracle",
    "FormattingIntegrityGate",
    "VisualDiffOracle",
    "MultiGateVerifier",
    "PARAM_BOUNDS",
    "ProfileGenome",
    "ProfilePolicy",
    "PDF2RTFAdapter",
    "PDF2RTFEvaluator",
    "PDF2RTFSpec",
]

