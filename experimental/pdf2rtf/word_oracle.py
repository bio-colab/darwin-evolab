"""word_oracle.py — Word-in-the-Loop Verification Oracle.

Completely eliminates "Mirror Blindness" by using Microsoft Word (Office 16.0)
via COM automation to open emitted RTF files, extract the native runtime Document
Object Model (DOM), and verify it directly against the Ground Truth Reference IR.
"If Microsoft Word cannot tell the difference, no one can."
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any, List

from .equivalence import EquivalenceReport, MultiGateVerifier
from .genome import ProfilePolicy
from .ir import Document
from .pdf_extractor import PDFExtractor
from .rtf_emitter import emit_rtf
from .word_holdout import _extract_word_doc_ir, load_real_word_holdout

REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "pdf2rtf_word_oracle_audit.json"


class WordInTheLoopOracle:
    """End-to-End Verification Oracle using native Microsoft Word runtime rendering."""

    def __init__(self, verifier: MultiGateVerifier | None = None) -> None:
        self.verifier = verifier or MultiGateVerifier()

    def verify_single_rtf(self, rtf_text: str, reference: Document) -> EquivalenceReport:
        """Renders RTF in Microsoft Word COM and compares DOM against Reference IR."""
        import win32com.client

        with tempfile.NamedTemporaryFile(suffix=".rtf", delete=False, mode="w", encoding="utf-8") as f:
            f.write(rtf_text)
            temp_path = Path(f.name).resolve()

        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False

        try:
            doc = word.Documents.Open(str(temp_path))
            try:
                word_rendered_ir = _extract_word_doc_ir(doc)
            finally:
                doc.Close(False)
        finally:
            word.Quit()
            temp_path.unlink(missing_ok=True)

        return self.verifier.verify(candidate=word_rendered_ir, reference=reference)

    def audit_corpus(
        self,
        corpus,
        policy: ProfilePolicy | None = None,
        save_report: bool = True,
    ) -> dict[str, Any]:
        """Audits an entire corpus using a single Word COM instance for maximum speed and fidelity."""
        import win32com.client

        pol = policy or ProfilePolicy()
        extractor = PDFExtractor(policy=pol)

        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False

        doc_results: list[dict[str, Any]] = []
        composite_scores: list[float] = []
        text_pass_count = 0
        overall_pass_count = 0

        print(f"=== Running Word-in-the-Loop Oracle Audit ({len(corpus)} documents) ===")

        try:
            for item in corpus:
                t0 = time.perf_counter()
                cand_ir = extractor.extract(item.pdf_bytes)
                rtf_text = emit_rtf(cand_ir)

                with tempfile.NamedTemporaryFile(suffix=".rtf", delete=False, mode="w", encoding="utf-8") as f:
                    f.write(rtf_text)
                    temp_path = Path(f.name).resolve()

                try:
                    word_doc = word.Documents.Open(str(temp_path))
                    try:
                        word_dom_ir = _extract_word_doc_ir(word_doc)
                    finally:
                        word_doc.Close(False)
                finally:
                    temp_path.unlink(missing_ok=True)

                rep = self.verifier.verify(candidate=word_dom_ir, reference=item.reference_doc)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                composite_scores.append(rep.composite_score)
                if rep.gates[0].passed:
                    text_pass_count += 1
                if rep.passed:
                    overall_pass_count += 1

                gate_scores = {g.name: round(g.score, 4) for g in rep.gates}
                gate_status = {g.name: g.passed for g in rep.gates}

                doc_results.append({
                    "name": item.name,
                    "passed": rep.passed,
                    "composite_score": round(rep.composite_score, 4),
                    "gate_scores": gate_scores,
                    "gate_pass_status": gate_status,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "summary": rep.summary,
                })

                status_str = "PASSED [OK]" if rep.passed else "FAILED [WARN]"
                print(f"  {item.name:<30} -> {rep.composite_score * 100:.2f}% | {status_str} ({elapsed_ms:.1f}ms)")
        finally:
            word.Quit()

        avg_score = sum(composite_scores) / len(composite_scores) if composite_scores else 0.0
        text_pass_rate = text_pass_count / len(corpus) if corpus else 0.0
        overall_pass_rate = overall_pass_count / len(corpus) if corpus else 0.0

        print("-" * 70)
        print(f"Word-in-the-Loop Average Composite: {avg_score * 100:.2f}%")
        print(f"Word-in-the-Loop Text Integrity Pass: {text_pass_rate * 100:.2f}%")
        print(f"Word-in-the-Loop Overall Pass Rate:   {overall_pass_rate * 100:.2f}%")

        audit_report = {
            "title": "Word-in-the-Loop Official Microsoft Word Runtime Oracle Audit",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "oracle_engine": "Microsoft Word 16.0 (Win32 COM Automation)",
            "mirror_blindness_status": "Completely Eliminated (Native Word Rendering DOM compared)",
            "summary": {
                "total_documents": len(corpus),
                "average_composite_score": round(avg_score, 4),
                "text_integrity_pass_rate": round(text_pass_rate, 4),
                "overall_pass_rate": round(overall_pass_rate, 4),
            },
            "documents": doc_results,
        }

        if save_report:
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(audit_report, f, indent=2)
            print(f"Audit report saved to: {REPORT_PATH}")

        return audit_report


if __name__ == "__main__":
    holdout = load_real_word_holdout()
    oracle = WordInTheLoopOracle()
    # Use calibrated breakthrough policy
    policy = ProfilePolicy(
        para_split_delta_ratio=0.60,
        para_split_short_line_factor=0.85,
        para_split_indent_factor=0.60,
        para_split_font_weight=0.60,
        table_col_align_tol_pt=5.0,
    )
    oracle.audit_corpus(holdout, policy=policy)
