"""
cli.py — Standalone Command-Line Interface for the pdf2rtf Subsystem.

Provides end-user commands for:
  - convert:   Convert PDF documents to high-fidelity RTF using MAP-Elites auto-dispatch.
  - verify:    Audit conversion fidelity via 5-Gate MultiGateVerifier or Word COM Oracle.
  - inspect:   Analyze PDF typographic geometry, behavioral descriptors (D1, D2), and niches.
  - benchmark: Execute built-in validation suites (golden, dilemma, corpus54, holdout).
  - info:      Display environment status, PyMuPDF, Word COM, and MAP-Elites archive health.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, List, Sequence

from .genome import ProfilePolicy
from .ir import Document
from .map_elites_archive import (
    MAPElitesArchive,
    compute_document_descriptors,
)
from .pdf_extractor import HAS_FITZ, PDFExtractor
from .rtf_emitter import emit_rtf
from .rtf_parser import parse_rtf
from .equivalence import MultiGateVerifier

DEFAULT_ARCHIVE_PATH = Path(__file__).resolve().parent / "calibrated_map_elites_archive.json"
VERSION = "0.5.0"

# Configure UTF-8 output with replacement fallback for Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Terminal Formatting & Color Support
# ---------------------------------------------------------------------------

class Color:
    """ANSI terminal styling helpers with graceful fallback."""
    _USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")

    RESET = "\033[0m" if _USE_COLOR else ""
    BOLD = "\033[1m" if _USE_COLOR else ""
    DIM = "\033[2m" if _USE_COLOR else ""

    RED = "\033[31m" if _USE_COLOR else ""
    GREEN = "\033[32m" if _USE_COLOR else ""
    YELLOW = "\033[33m" if _USE_COLOR else ""
    BLUE = "\033[34m" if _USE_COLOR else ""
    MAGENTA = "\033[35m" if _USE_COLOR else ""
    CYAN = "\033[36m" if _USE_COLOR else ""
    WHITE = "\033[37m" if _USE_COLOR else ""

    @classmethod
    def paint(cls, text: str, style: str) -> str:
        if not cls._USE_COLOR:
            return text
        return f"{style}{text}{cls.RESET}"


def print_banner() -> None:
    """Prints a styled CLI header banner."""
    title = f"darwin-evolab | pdf2rtf v{VERSION}"
    subtitle = "Self-Calibrating Evolutionary PDF-to-RTF Engine"
    width = max(len(title), len(subtitle)) + 4
    border = "=" * width
    print(Color.paint(f"\n+{border}+", Color.CYAN))
    print(Color.paint(f"|  {title:<{width-4}}  |", Color.BOLD + Color.CYAN))
    print(Color.paint(f"|  {subtitle:<{width-4}}  |", Color.DIM + Color.WHITE))
    print(Color.paint(f"+{border}+\n", Color.CYAN))


def _load_archive(archive_path: Path | None = None) -> MAPElitesArchive | None:
    """Loads the MAP-Elites behavioral archive from default or custom path."""
    target = archive_path or DEFAULT_ARCHIVE_PATH
    if target.exists():
        try:
            return MAPElitesArchive.load_json(target)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    """Displays subsystem environment status, dependencies, and archive metrics."""
    print_banner()
    print(Color.paint("[*] System & Runtime Environment:", Color.BOLD + Color.WHITE))
    print(f"  * Python:          {sys.version.split()[0]} ({sys.platform})")

    # PyMuPDF Status
    if HAS_FITZ:
        import fitz
        fitz_ver = getattr(fitz, "__version__", "installed")
        print(f"  * PyMuPDF (fitz):  {Color.paint(f'Available (v{fitz_ver})', Color.GREEN)}")
    else:
        print(f"  * PyMuPDF (fitz):  {Color.paint('Missing (install via pip install pymupdf)', Color.RED)}")

    # Microsoft Word COM Status
    word_com_status = "Unavailable (Windows only)"
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Word.Application\CurVer") as k:
                cur_ver, _ = winreg.QueryValueEx(k, "")
            ver_num = cur_ver.split(".")[-1] + ".0"
            word_com_status = Color.paint(f"Available (Microsoft Word {ver_num})", Color.GREEN)
        except Exception:
            try:
                import win32com.client
                word = win32com.client.Dispatch("Word.Application")
                ver = getattr(word, "Version", "16.0")
                word.Quit()
                word_com_status = Color.paint(f"Available (Microsoft Word {ver})", Color.GREEN)
            except Exception:
                word_com_status = Color.paint("Installed but inactive / no Word license", Color.YELLOW)
    print(f"  * Word COM Oracle: {word_com_status}")

    # MAP-Elites Archive Status
    print(Color.paint("\n[*] MAP-Elites Behavioral Archive:", Color.BOLD + Color.WHITE))
    archive = _load_archive(Path(args.archive) if getattr(args, "archive", None) else None)
    if archive:
        champ = archive.get_champion()
        champ_fit = f"{champ.fitness * 100:.2f}% (Niche {champ.coord})" if champ else "N/A"
        print(f"  * Archive File:    {DEFAULT_ARCHIVE_PATH.name}")
        print(f"  * Dimensions:      {archive.grid_x} x {archive.grid_y} ({archive.total_capacity} total niches)")
        print(f"  * Occupied Niches: {Color.paint(f'{archive.occupied_count} / {archive.total_capacity} ({archive.coverage * 100:.1f}%)', Color.GREEN)}")
        print(f"  * QD-Score:        {Color.paint(f'{archive.qd_score:.4f}', Color.CYAN)}")
        print(f"  * Champion Fit:    {Color.paint(champ_fit, Color.GREEN)}")
        print(f"  * Calibrated At:   {archive.calibrated_at}")
    else:
        print(Color.paint("  • Status: Archive not found or corrupted.", Color.YELLOW))

    print("\n" + Color.paint("Ready for conversion.", Color.DIM + Color.WHITE) + "\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: inspect
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    """Analyzes PDF geometry, behavioral descriptors, and matched MAP-Elites niche."""
    pdf_path = Path(args.input).resolve()
    if not pdf_path.exists():
        print(Color.paint(f"Error: input file not found: {pdf_path}", Color.RED), file=sys.stderr)
        return 1

    if not HAS_FITZ:
        print(Color.paint("Error: PyMuPDF is required. Run: pip install pymupdf", Color.RED), file=sys.stderr)
        return 1

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    d1, d2 = compute_document_descriptors(pdf_bytes)
    archive = _load_archive(Path(args.archive) if getattr(args, "archive", None) else None)

    extractor = PDFExtractor(archive=archive) if archive else PDFExtractor()
    t0 = time.perf_counter()
    doc_ir = extractor.extract(pdf_bytes)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print_banner()
    print(Color.paint(f"[*] Document Inspection: {pdf_path.name}", Color.BOLD + Color.WHITE))
    print(f"  * File Size:       {len(pdf_bytes):,} bytes")
    print(f"  * Pages:           {len(doc_ir.pages)}")
    print(f"  * Paragraphs:      {doc_ir.paragraph_count}")
    print(f"  * Tables:          {doc_ir.table_count}")
    print(f"  * Fonts Detected:  {', '.join(sorted(doc_ir.fonts_used)) if doc_ir.fonts_used else 'None'}")
    print(f"  * Parse Duration:  {elapsed_ms:.2f} ms")

    print(Color.paint("\n[*] Behavioral Descriptors & Behavioral Niche:", Color.BOLD + Color.WHITE))
    print(f"  * Cluster Density (D1):    {d1:.4f}")
    print(f"  * Table Sensitivity (D2):  {d2:.4f}")

    if archive:
        target_coord = archive.coord_from_values(d1, d2)
        cell = archive.get_nearest_elite(d1, d2)
        exact_match = (cell is not None and cell.coord == target_coord)
        match_str = Color.paint("EXACT NICHE MATCH", Color.GREEN) if exact_match else Color.paint("NEAREST NEIGHBOR", Color.YELLOW)
        elite_fit = f"{cell.fitness * 100:.2f}%" if cell else "N/A"
        cell_coord = cell.coord if cell else target_coord

        print(f"  * Target Coordinate:       {target_coord}")
        print(f"  * Dispatched Elite Niche:  {cell_coord} ({match_str})")
        print(f"  * Elite Verified Fitness:  {Color.paint(elite_fit, Color.GREEN)}")
        if cell:
            p = cell.policy
            print(Color.paint("\n[*] Dispatched Niche Policy Parameters:", Color.BOLD + Color.WHITE))
            print(f"  * space_gap_ratio:             {p.space_gap_ratio:.4f}")
            print(f"  * para_split_delta_ratio:      {p.para_split_delta_ratio:.4f}")
            print(f"  * align_tolerance_pt:          {p.align_tolerance_pt:.2f} pt")
            print(f"  * table_col_align_tol_pt:      {p.table_col_align_tol_pt:.2f} pt")
            print(f"  * para_split_short_line_factor:{p.para_split_short_line_factor:.4f}")

    print("")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: convert
# ---------------------------------------------------------------------------

def _resolve_input_files(inputs: Sequence[str]) -> List[Path]:
    """Resolves individual files, glob expressions, and directory scans into a unique file list."""
    resolved: list[Path] = []
    for pattern in inputs:
        p = Path(pattern)
        if p.is_dir():
            for f in sorted(p.glob("*.pdf")):
                resolved.append(f.resolve())
        elif "*" in pattern or "?" in pattern:
            for match in glob.glob(pattern):
                mp = Path(match)
                if mp.is_file() and mp.suffix.lower() == ".pdf":
                    resolved.append(mp.resolve())
        elif p.is_file():
            resolved.append(p.resolve())
        else:
            print(Color.paint(f"Warning: file or pattern not found: {pattern}", Color.YELLOW), file=sys.stderr)
    return sorted(list(set(resolved)))


def cmd_convert(args: argparse.Namespace) -> int:
    """Converts one or multiple PDF documents into standard Microsoft RTF files."""
    if not HAS_FITZ:
        print(Color.paint("Error: PyMuPDF is required for conversion. Run: pip install pymupdf", Color.RED), file=sys.stderr)
        return 1

    pdf_files = _resolve_input_files(args.inputs)
    if not pdf_files:
        print(Color.paint("Error: no valid PDF input files specified.", Color.RED), file=sys.stderr)
        return 2

    # Load MAP-Elites Archive if applicable
    archive: MAPElitesArchive | None = None
    policy: ProfilePolicy | None = None

    if args.mode == "auto":
        archive = _load_archive(Path(args.archive) if args.archive else None)
        if archive is None and args.verbose:
            print(Color.paint("Notice: Archive not found, falling back to default heuristic policy.", Color.YELLOW))
    elif args.mode == "champion":
        archive = _load_archive(Path(args.archive) if args.archive else None)
        if archive and archive.get_champion():
            policy = archive.get_champion().policy
        else:
            policy = ProfilePolicy()
    elif args.mode == "default":
        policy = ProfilePolicy()
    elif args.policy:
        pol_path = Path(args.policy)
        if pol_path.exists():
            with open(pol_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            policy = ProfilePolicy.from_dict(data)
        else:
            print(Color.paint(f"Error: custom policy file not found: {pol_path}", Color.RED), file=sys.stderr)
            return 2

    # Determine output destinations
    is_multi = len(pdf_files) > 1
    out_dir: Path | None = None
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    elif is_multi and args.output:
        out_dir = Path(args.output).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    if not args.quiet:
        print_banner()
        mode_label = f"mode={args.mode}"
        if args.mode == "auto" and archive:
            mode_label += " (MAP-Elites 10x10 Specialized Niche Dispatch)"
        elif args.mode == "champion":
            mode_label += " (Monolithic Champion)"
        print(Color.paint(f"Converting {len(pdf_files)} document(s) [{mode_label}]...\n", Color.BOLD + Color.WHITE))

    for idx, pdf_path in enumerate(pdf_files, 1):
        if is_multi or out_dir:
            target_rtf = (out_dir or pdf_path.parent) / f"{pdf_path.stem}.rtf"
        elif args.output:
            target_rtf = Path(args.output).resolve()
        else:
            target_rtf = pdf_path.with_suffix(".rtf")

        if target_rtf.exists() and not args.overwrite:
            print(Color.paint(f"[{idx}/{len(pdf_files)}] SKIPPED: {target_rtf.name} (exists, use -f/--overwrite)", Color.YELLOW))
            continue

        try:
            t0 = time.perf_counter()
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            extractor = PDFExtractor(policy=policy, archive=archive)
            doc_ir = extractor.extract(pdf_bytes)
            rtf_str = emit_rtf(doc_ir)

            target_rtf.parent.mkdir(parents=True, exist_ok=True)
            with open(target_rtf, "w", encoding="utf-8") as f:
                f.write(rtf_str)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Optional Document IR metadata export
            if args.meta:
                meta_path = target_rtf.with_suffix(".meta.json")
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(doc_ir.to_dict(), mf, indent=2)

            dispatched_info = ""
            if extractor.last_dispatched_cell:
                exact_flag = "exact" if extractor.last_dispatched_exact else "nearest"
                dispatched_info = f" [niche: {extractor.last_dispatched_cell.coord} ({exact_flag})]"

            status_tag = Color.paint("[OK]", Color.GREEN)
            rtf_size_kb = len(rtf_str.encode("utf-8")) / 1024.0
            print(f"  {status_tag} [{idx}/{len(pdf_files)}] {pdf_path.name} -> {target_rtf.name} ({rtf_size_kb:.1f} KB, {elapsed_ms:.1f}ms){dispatched_info}")

            if args.verbose:
                print(f"       * Pages: {len(doc_ir.pages)} | Paragraphs: {doc_ir.paragraph_count} | Tables: {doc_ir.table_count}")
                print(f"       * Fonts: {', '.join(sorted(doc_ir.fonts_used))}")

            success_count += 1
        except Exception as exc:
            fail_count += 1
            print(Color.paint(f"  [FAIL] [{idx}/{len(pdf_files)}] {pdf_path.name}: {exc}", Color.RED), file=sys.stderr)

    if not args.quiet:
        print("\n" + "-" * 60)
        summary_color = Color.GREEN if fail_count == 0 else Color.YELLOW
        print(Color.paint(f"Conversion complete: {success_count} succeeded, {fail_count} failed.", summary_color))
        print("")

    return 0 if fail_count == 0 else 1


# ---------------------------------------------------------------------------
# Subcommand: verify
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    """Audits conversion fidelity against source PDF via 5-Gate Verifier or Word COM."""
    pdf_path = Path(args.pdf).resolve()
    rtf_path = Path(args.rtf).resolve()

    if not pdf_path.exists():
        print(Color.paint(f"Error: source PDF not found: {pdf_path}", Color.RED), file=sys.stderr)
        return 1
    if not rtf_path.exists():
        print(Color.paint(f"Error: candidate RTF not found: {rtf_path}", Color.RED), file=sys.stderr)
        return 1

    if not HAS_FITZ:
        print(Color.paint("Error: PyMuPDF is required. Run: pip install pymupdf", Color.RED), file=sys.stderr)
        return 1

    if not args.json:
        print_banner()
        print(Color.paint(f"[*] Multi-Gate Equivalence Verification", Color.BOLD + Color.WHITE))
        print(f"  * Reference (PDF): {pdf_path.name}")
        print(f"  * Candidate (RTF): {rtf_path.name}")
        print(f"  * Oracle Mode:     {args.oracle}")

    t0 = time.perf_counter()
    ref_ir = PDFExtractor().extract(str(pdf_path))

    if args.oracle == "word":
        if sys.platform != "win32":
            print(Color.paint("Error: Word COM Oracle requires Windows OS.", Color.RED), file=sys.stderr)
            return 1
        from .word_oracle import _extract_word_doc_ir
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        try:
            wdoc = word.Documents.Open(str(rtf_path))
            try:
                cand_ir = _extract_word_doc_ir(wdoc)
            finally:
                wdoc.Close(False)
        finally:
            word.Quit()
    else:
        with open(rtf_path, "r", encoding="utf-8", errors="replace") as f:
            rtf_content = f.read()
        cand_ir = parse_rtf(rtf_content)

    verifier = MultiGateVerifier()
    report = verifier.verify(candidate=cand_ir, reference=ref_ir)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.passed else 1

    print(Color.paint("\n[*] Verification Gate Results:", Color.BOLD + Color.WHITE))
    for g in report.gates:
        status_tag = Color.paint("[PASS]", Color.GREEN) if g.passed else Color.paint("[FAIL]", Color.RED)
        score_pct = f"{g.score * 100:6.2f}%"
        print(f"  {status_tag} {g.name:<28} : {score_pct}  ({g.message})")

    print("-" * 60)
    composite_pct = report.composite_score * 100.0
    status_label = Color.paint("PASSED ALL GATES", Color.BOLD + Color.GREEN) if report.passed else Color.paint("GATE FAILURE DETECTED", Color.BOLD + Color.RED)
    print(f"  Composite Fidelity Score: {Color.paint(f'{composite_pct:.2f}%', Color.BOLD + Color.CYAN)}")
    print(f"  Overall Equivalence:      {status_label}")
    print(f"  Verification Elapsed:     {elapsed_ms:.1f} ms\n")

    return 0 if report.passed else 1


# ---------------------------------------------------------------------------
# Subcommand: benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace) -> int:
    """Runs built-in benchmark test suites."""
    print_banner()
    corpus_name = args.corpus
    print(Color.paint(f"[*] Running Benchmark Suite: {corpus_name.upper()}\n", Color.BOLD + Color.WHITE))

    if corpus_name == "corpus54":
        from .word_corpus_54 import run_corpus_54_benchmark
        run_corpus_54_benchmark()
    elif corpus_name == "holdout":
        from .benchmark import run_holdout_benchmark
        run_holdout_benchmark()
    elif corpus_name == "dilemma":
        from .word_dilemma import run_dilemma_benchmark
        run_dilemma_benchmark()
    elif corpus_name == "golden":
        from .benchmark import run_golden_benchmark
        run_golden_benchmark()
    else:
        print(Color.paint(f"Unknown corpus: {corpus_name}", Color.RED), file=sys.stderr)
        return 2

    return 0


# ---------------------------------------------------------------------------
# Main CLI Parser & Dispatcher
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Constructs the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="pdf2rtf",
        description="darwin-evolab pdf2rtf: Self-Calibrating Evolutionary PDF-to-RTF Converter.",
        epilog="Examples:\n"
               "  pdf2rtf convert document.pdf -o output.rtf\n"
               "  pdf2rtf convert ./papers/*.pdf --out-dir ./rtfs/ --mode auto\n"
               "  pdf2rtf inspect document.pdf\n"
               "  pdf2rtf verify document.pdf output.rtf\n"
               "  pdf2rtf info\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"pdf2rtf v{VERSION}")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # --- convert ---
    p_conv = subparsers.add_parser("convert", help="Convert PDF file(s) to RTF.")
    p_conv.add_argument("inputs", nargs="+", help="Input PDF file(s), wildcard pattern, or directory.")
    p_conv.add_argument("-o", "--output", type=str, default=None, help="Target output RTF file path (for single file) or directory.")
    p_conv.add_argument("--out-dir", type=str, default=None, help="Target output directory for batch conversion.")
    p_conv.add_argument(
        "--mode",
        choices=["auto", "champion", "default"],
        default="auto",
        help="Conversion policy mode: 'auto' (dynamic MAP-Elites niche dispatch), 'champion' (monolithic champion), 'default' (baseline heuristic). Default: auto.",
    )
    p_conv.add_argument("--archive", type=str, default=None, help="Custom MAP-Elites archive JSON path.")
    p_conv.add_argument("--policy", type=str, default=None, help="Custom ProfilePolicy JSON path.")
    p_conv.add_argument("-f", "--overwrite", action="store_true", help="Overwrite existing output files.")
    p_conv.add_argument("-v", "--verbose", action="store_true", help="Display detailed extraction metrics and policy dispatch.")
    p_conv.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error terminal output.")
    p_conv.add_argument("--meta", action="store_true", help="Export extracted Document IR as .meta.json alongside RTF.")

    # --- verify ---
    p_ver = subparsers.add_parser("verify", help="Verify equivalence between source PDF and candidate RTF.")
    p_ver.add_argument("pdf", help="Reference source PDF file.")
    p_ver.add_argument("rtf", help="Candidate converted RTF file.")
    p_ver.add_argument(
        "--oracle",
        choices=["internal", "word"],
        default="internal",
        help="Oracle verification mode: 'internal' (5-Gate Verifier) or 'word' (official Microsoft Word COM DOM). Default: internal.",
    )
    p_ver.add_argument("--json", action="store_true", help="Output raw verification report as JSON.")

    # --- inspect ---
    p_insp = subparsers.add_parser("inspect", help="Inspect PDF geometry, behavioral descriptors (D1, D2), and niche.")
    p_insp.add_argument("input", help="Target PDF file to inspect.")
    p_insp.add_argument("--archive", type=str, default=None, help="Custom MAP-Elites archive JSON path.")

    # --- benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="Execute built-in validation test suites.")
    p_bench.add_argument(
        "--corpus",
        choices=["golden", "dilemma", "corpus54", "holdout"],
        default="golden",
        help="Target benchmark corpus suite.",
    )

    # --- info ---
    p_info = subparsers.add_parser("info", help="Display environment, runtime oracles, and MAP-Elites archive health.")
    p_info.add_argument("--archive", type=str, default=None, help="Custom MAP-Elites archive JSON path.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for pdf2rtf CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    handlers = {
        "convert": cmd_convert,
        "verify": cmd_verify,
        "inspect": cmd_inspect,
        "benchmark": cmd_benchmark,
        "info": cmd_info,
    }

    handler = handlers.get(args.subcommand)
    if handler:
        return handler(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
