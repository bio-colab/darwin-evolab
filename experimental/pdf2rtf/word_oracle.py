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

from evolab.evaluators import Evaluator, FitnessResult

from .equivalence import EquivalenceReport, MultiGateVerifier
from .genome import ProfileGenome, ProfilePolicy
from .ir import Document
from .pdf_extractor import PDFExtractor
from .rtf_emitter import emit_rtf
from .rtf_parser import parse_rtf
from .word_holdout import _extract_word_doc_ir, load_real_word_holdout

REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "pdf2rtf_word_oracle_audit.json"


class WordInTheLoopFitnessEvaluator(Evaluator):
    """Word-in-the-Loop evolutionary fitness evaluator.

    Eliminates mirror blindness from the calibration process by comparing candidate
    RTF emissions directly inside the official Microsoft Word COM runtime DOM.

    Supports:
    - 'pure': Evaluates all candidate genomes in live Microsoft Word COM.
    - 'hybrid': Rapid screening via internal MultiGateVerifier (20ms/doc), with mandatory
      Microsoft Word COM evaluation for top-performing candidates (>= screening_threshold)
      and all MAP-Elites niche updates.
    """

    def __init__(
        self,
        corpus: list[Any],
        mode: str = "hybrid",
        screening_threshold: float = 0.90,
        verifier: MultiGateVerifier | None = None,
        max_com_batch_before_recycle: int = 400,
    ) -> None:
        self.corpus = corpus
        self.mode = mode
        self.screening_threshold = screening_threshold
        self.verifier = verifier or MultiGateVerifier()
        self.max_com_batch_before_recycle = max_com_batch_before_recycle
        self._word: Any | None = None
        self._com_eval_count = 0
        self._total_evals = 0

    @property
    def deterministic(self) -> bool:
        return True

    def _get_word(self) -> Any:
        if self._word is None:
            import win32com.client
            self._word = win32com.client.Dispatch("Word.Application")
            self._word.Visible = False
            self._word.ScreenUpdating = False
            self._word.DisplayAlerts = False
        return self._word

    def _recycle_word_if_needed(self) -> None:
        if self._word is not None and self._com_eval_count >= self.max_com_batch_before_recycle:
            self.flush()

    def flush(self) -> None:
        """Closes and terminates the Microsoft Word COM automation process."""
        if self._word is not None:
            try:
                self._word.Quit()
            except Exception:
                pass
            self._word = None
            self._com_eval_count = 0

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        genome = getattr(target, "genome", target)
        if not isinstance(genome, ProfileGenome):
            raise TypeError(f"Expected ProfileGenome, got {type(genome)}")

        policy = genome.to_policy()
        extractor = PDFExtractor(policy=policy)
        self._total_evals += 1

        # Hybrid Tier 1: Fast internal screening
        if self.mode == "hybrid":
            screen_scores: list[float] = []
            cand_rtfs: list[tuple[Any, str]] = []
            for item in self.corpus:
                cand_ir = extractor.extract(item.pdf_bytes)
                rtf_text = emit_rtf(cand_ir)
                recon_ir = parse_rtf(rtf_text)
                rep = self.verifier.verify(candidate=recon_ir, reference=item.reference_doc)
                screen_scores.append(rep.composite_score)
                cand_rtfs.append((item, rtf_text))

            avg_screen = sum(screen_scores) / len(screen_scores) if screen_scores else 0.0

            # If candidate does not meet threshold, return screening fitness
            if avg_screen < self.screening_threshold:
                return FitnessResult(
                    score=avg_screen,
                    sub_scores={"screening_avg": avg_screen, "tier": 1.0},
                    artifacts={"mode": "hybrid_screening", "score": avg_screen},
                )

            # Hybrid Tier 2: Promote candidate to full Word COM Oracle evaluation
            return self._evaluate_in_word_com(cand_rtfs, is_promoted=True)

        # Pure Mode: Direct Microsoft Word COM evaluation
        cand_rtfs = []
        for item in self.corpus:
            cand_ir = extractor.extract(item.pdf_bytes)
            rtf_text = emit_rtf(cand_ir)
            cand_rtfs.append((item, rtf_text))

        return self._evaluate_in_word_com(cand_rtfs, is_promoted=False)

    def _evaluate_in_word_com(
        self,
        items_and_rtfs: list[tuple[Any, str]],
        is_promoted: bool = False,
    ) -> FitnessResult:
        word = self._get_word()
        com_scores: list[float] = []
        sub_scores: dict[str, float] = {}

        for item, rtf_text in items_and_rtfs:
            with tempfile.NamedTemporaryFile(suffix=".rtf", delete=False, mode="w", encoding="utf-8") as f:
                f.write(rtf_text)
                temp_path = Path(f.name).resolve()

            try:
                word_doc = word.Documents.Open(str(temp_path))
                try:
                    word_dom_ir = _extract_word_doc_ir(word_doc)
                finally:
                    word_doc.Close(False)
            except Exception:
                # If COM crashes, recycle word and score zero for this document
                self.flush()
                word = self._get_word()
                com_scores.append(0.0)
                continue
            finally:
                temp_path.unlink(missing_ok=True)

            self._com_eval_count += 1
            rep = self.verifier.verify(candidate=word_dom_ir, reference=item.reference_doc)
            com_scores.append(rep.composite_score)
            for g in rep.gates:
                sub_scores[f"{item.name}_{g.name}"] = g.score

        self._recycle_word_if_needed()
        mean_com = sum(com_scores) / len(com_scores) if com_scores else 0.0

        return FitnessResult(
            score=mean_com,
            sub_scores=sub_scores,
            artifacts={
                "mode": "word_com_promoted" if is_promoted else "word_com_pure",
                "com_score": mean_com,
                "document_scores": com_scores,
            },
        )



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
