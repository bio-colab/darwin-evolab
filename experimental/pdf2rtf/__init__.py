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
from .map_elites_archive import (
    ArchiveCell,
    MAPElitesArchive,
    MAPElitesPolicyDispatcher,
    compute_document_descriptors,
)
from pathlib import Path
from typing import Any

_DEFAULT_ARCHIVE_FILE = Path(__file__).resolve().parent / "calibrated_map_elites_archive.json"


def convert(
    source: str | Path | bytes,
    output: str | Path | None = None,
    mode: str = "auto",
    archive: MAPElitesArchive | str | Path | None = None,
    policy: ProfilePolicy | None = None,
) -> str:
    """High-level Python API: Converts PDF input into standard Microsoft RTF.

    Args:
        source: File path (str/Path) or raw PDF bytes.
        output: Optional destination file path (.rtf) to save the generated output.
        mode: Policy mode: 'auto' (MAP-Elites dynamic dispatch), 'champion', or 'default'.
        archive: Optional custom MAPElitesArchive or path to archive JSON.
        policy: Optional custom ProfilePolicy override.

    Returns:
        str: Emitted RTF document content string.
    """
    loaded_archive: MAPElitesArchive | None = None
    target_policy: ProfilePolicy | None = policy

    if mode in ("auto", "champion"):
        if isinstance(archive, MAPElitesArchive):
            loaded_archive = archive
        elif archive is not None:
            loaded_archive = MAPElitesArchive.load_json(archive)
        elif _DEFAULT_ARCHIVE_FILE.exists():
            try:
                loaded_archive = MAPElitesArchive.load_json(_DEFAULT_ARCHIVE_FILE)
            except Exception:
                loaded_archive = None

        if mode == "champion" and loaded_archive and loaded_archive.get_champion():
            target_policy = loaded_archive.get_champion().policy
            loaded_archive = None  # Use champion directly without re-dispatching

    extractor = PDFExtractor(policy=target_policy, archive=loaded_archive)
    doc_ir = extractor.extract(source)
    rtf_str = emit_rtf(doc_ir)

    if output:
        out_p = Path(output).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(rtf_str)

    return rtf_str


def verify(
    pdf_source: str | Path | bytes,
    rtf_source: str | Path,
    oracle: str = "internal",
) -> EquivalenceReport:
    """High-level Python API: Verifies equivalence between source PDF and candidate RTF.

    Args:
        pdf_source: Reference PDF path or bytes.
        rtf_source: Candidate RTF file path or string.
        oracle: 'internal' (MultiGateVerifier) or 'word' (Microsoft Word COM DOM).

    Returns:
        EquivalenceReport: Comprehensive 5-Gate equivalence audit report.
    """
    ref_ir = PDFExtractor().extract(pdf_source)

    if oracle == "word":
        import sys
        if sys.platform != "win32":
            raise RuntimeError("Word COM Oracle requires Windows OS.")
        from .word_oracle import _extract_word_doc_ir
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        try:
            wdoc = word.Documents.Open(str(Path(rtf_source).resolve()))
            try:
                cand_ir = _extract_word_doc_ir(wdoc)
            finally:
                wdoc.Close(False)
        finally:
            word.Quit()
    else:
        if isinstance(rtf_source, (str, Path)) and Path(rtf_source).exists():
            with open(rtf_source, "r", encoding="utf-8", errors="replace") as f:
                rtf_content = f.read()
        else:
            rtf_content = str(rtf_source)
        cand_ir = parse_rtf(rtf_content)

    verifier = MultiGateVerifier()
    return verifier.verify(candidate=cand_ir, reference=ref_ir)


def inspect_pdf(source: str | Path | bytes) -> dict[str, Any]:
    """High-level Python API: Analyzes PDF geometry, descriptors, and behavioral niche."""
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            pdf_bytes = f.read()
    else:
        pdf_bytes = bytes(source)

    d1, d2 = compute_document_descriptors(pdf_bytes)
    loaded_archive: MAPElitesArchive | None = None
    if _DEFAULT_ARCHIVE_FILE.exists():
        try:
            loaded_archive = MAPElitesArchive.load_json(_DEFAULT_ARCHIVE_FILE)
        except Exception:
            pass

    doc_ir = PDFExtractor(archive=loaded_archive).extract(pdf_bytes)

    niche_info = {}
    if loaded_archive:
        coord = loaded_archive.coord_from_values(d1, d2)
        cell = loaded_archive.get_nearest_elite(d1, d2)
        niche_info = {
            "target_coordinate": coord,
            "matched_coordinate": cell.coord if cell else None,
            "is_exact_match": bool(cell and cell.coord == coord),
            "elite_fitness": round(cell.fitness, 4) if cell else None,
            "policy": cell.policy.to_dict() if cell else None,
        }

    return {
        "byte_size": len(pdf_bytes),
        "page_count": len(doc_ir.pages),
        "paragraph_count": doc_ir.paragraph_count,
        "table_count": doc_ir.table_count,
        "fonts_used": sorted(list(doc_ir.fonts_used)),
        "descriptors": {
            "cluster_density_d1": round(d1, 4),
            "table_sensitivity_d2": round(d2, 4),
        },
        "map_elites": niche_info,
    }


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
    "ArchiveCell",
    "MAPElitesArchive",
    "MAPElitesPolicyDispatcher",
    "compute_document_descriptors",
    "convert",
    "verify",
    "inspect_pdf",
]


