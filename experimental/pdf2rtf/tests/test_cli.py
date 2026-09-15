"""
test_cli.py — Comprehensive Unit & Integration Tests for pdf2rtf Standalone CLI and Python API.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import pytest

from experimental.pdf2rtf.cli import main as cli_main
import experimental.pdf2rtf as p2r
from evolab.cli import main as evolab_main

TEST_PDF = Path(__file__).resolve().parent.parent / "real_word_54" / "word_seed_01_align_left_single.pdf"


def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    """Verifies that --help prints usage without crashing and exits with 0."""
    with pytest.raises(SystemExit) as exc:
        cli_main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "pdf2rtf" in captured.out
    assert "convert" in captured.out
    assert "verify" in captured.out
    assert "inspect" in captured.out
    assert "info" in captured.out


def test_cli_info(capsys: pytest.CaptureFixture) -> None:
    """Verifies the info command output and status reporting."""
    ret = cli_main(["info"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "System & Runtime Environment" in captured.out
    assert "MAP-Elites Behavioral Archive" in captured.out
    assert "10 x 10" in captured.out


def test_cli_inspect(capsys: pytest.CaptureFixture) -> None:
    """Verifies the inspect command analyzes PDF geometry and behavioral niche."""
    assert TEST_PDF.exists(), f"Missing fixture PDF: {TEST_PDF}"
    ret = cli_main(["inspect", str(TEST_PDF)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Document Inspection:" in captured.out
    assert "Cluster Density (D1):" in captured.out
    assert "Table Sensitivity (D2):" in captured.out
    assert "Dispatched Elite Niche:" in captured.out


def test_cli_convert_auto(tmp_path: Path) -> None:
    """Verifies single-file conversion in auto mode (MAP-Elites dispatch)."""
    assert TEST_PDF.exists()
    out_rtf = tmp_path / "test_out_auto.rtf"
    ret = cli_main(["convert", str(TEST_PDF), "-o", str(out_rtf), "--mode", "auto", "--verbose"])
    assert ret == 0
    assert out_rtf.exists()
    content = out_rtf.read_text(encoding="utf-8")
    assert r"\rtf1" in content
    assert r"\deff0" in content


def test_cli_convert_champion(tmp_path: Path) -> None:
    """Verifies single-file conversion using monolithic champion policy."""
    out_rtf = tmp_path / "test_out_champ.rtf"
    ret = cli_main(["convert", str(TEST_PDF), "-o", str(out_rtf), "--mode", "champion"])
    assert ret == 0
    assert out_rtf.exists()
    content = out_rtf.read_text(encoding="utf-8")
    assert r"\rtf1" in content


def test_cli_convert_default(tmp_path: Path) -> None:
    """Verifies single-file conversion using default heuristic baseline."""
    out_rtf = tmp_path / "test_out_default.rtf"
    ret = cli_main(["convert", str(TEST_PDF), "-o", str(out_rtf), "--mode", "default"])
    assert ret == 0
    assert out_rtf.exists()
    content = out_rtf.read_text(encoding="utf-8")
    assert r"\rtf1" in content


def test_cli_convert_with_metadata(tmp_path: Path) -> None:
    """Verifies --meta flag exports Document IR metadata JSON alongside RTF."""
    out_rtf = tmp_path / "test_out_meta.rtf"
    ret = cli_main(["convert", str(TEST_PDF), "-o", str(out_rtf), "--meta"])
    assert ret == 0
    assert out_rtf.exists()
    meta_file = tmp_path / "test_out_meta.meta.json"
    assert meta_file.exists()
    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
    assert "pages" in meta_data
    assert len(meta_data["pages"]) >= 1


def test_cli_convert_batch(tmp_path: Path) -> None:
    """Verifies batch conversion into an output directory."""
    out_dir = tmp_path / "batch_out"
    ret = cli_main(["convert", str(TEST_PDF), "--out-dir", str(out_dir)])
    assert ret == 0
    expected_file = out_dir / f"{TEST_PDF.stem}.rtf"
    assert expected_file.exists()


def test_cli_verify(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Verifies equivalence check between PDF and candidate RTF via CLI."""
    out_rtf = tmp_path / "verify_target.rtf"
    cli_main(["convert", str(TEST_PDF), "-o", str(out_rtf), "-q"])
    assert out_rtf.exists()

    ret = cli_main(["verify", str(TEST_PDF), str(out_rtf)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Multi-Gate Equivalence Verification" in captured.out
    assert "Gate 1: Text Integrity" in captured.out
    assert "Composite Fidelity Score:" in captured.out
    assert "PASSED ALL GATES" in captured.out


def test_cli_verify_json_output(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Verifies --json flag outputs structured parseable report."""
    out_rtf = tmp_path / "verify_json_target.rtf"
    cli_main(["convert", str(TEST_PDF), "-o", str(out_rtf), "-q"])
    capsys.readouterr()  # Flush convert stdout

    ret = cli_main(["verify", str(TEST_PDF), str(out_rtf), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "composite_score" in data
    assert "passed" in data
    assert data["passed"] is True


def test_cli_error_handling(capsys: pytest.CaptureFixture) -> None:
    """Verifies appropriate error exit codes for invalid inputs."""
    # Non-existent input file
    ret = cli_main(["convert", "non_existent_file.pdf"])
    assert ret != 0

    # Non-existent inspect target
    ret = cli_main(["inspect", "missing.pdf"])
    assert ret != 0

    # Missing verify arguments
    with pytest.raises(SystemExit):
        cli_main(["verify"])


def test_python_api_programmatic_access(tmp_path: Path) -> None:
    """Verifies top-level python API (convert, verify, inspect_pdf)."""
    # 1. Inspect
    info = p2r.inspect_pdf(TEST_PDF)
    assert info["page_count"] >= 1
    assert "descriptors" in info
    assert "map_elites" in info

    # 2. Convert to string
    rtf_str = p2r.convert(TEST_PDF)
    assert isinstance(rtf_str, str)
    assert r"\rtf1" in rtf_str

    # 3. Convert to file
    out_p = tmp_path / "api_out.rtf"
    p2r.convert(TEST_PDF, output=out_p)
    assert out_p.exists()

    # 4. Verify
    rep = p2r.verify(TEST_PDF, out_p)
    assert rep.passed is True
    assert rep.composite_score >= 0.99


def test_evolab_cli_subcommand_forwarding(capsys: pytest.CaptureFixture) -> None:
    """Verifies that 'evolab pdf2rtf info' forwards cleanly to the pdf2rtf CLI."""
    ret = evolab_main(["pdf2rtf", "info"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "darwin-evolab | pdf2rtf v0.5.0" in captured.out
    assert "MAP-Elites Behavioral Archive" in captured.out
