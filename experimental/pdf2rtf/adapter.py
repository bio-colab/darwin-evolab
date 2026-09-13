"""adapter.py — DomainAdapter Implementation for PDF-to-RTF Self-Calibration.

Bridges the universal Darwin-Evolab evolutionary kernel and MAP-Elites archive
to the self-calibrating PDF-to-RTF conversion subsystem.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

from evolab.adapters import DomainAdapter, register_domain_adapter
from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import Individual

from .equivalence import MultiGateVerifier
from .genome import PARAM_BOUNDS, ProfileGenome, ProfilePolicy
from .ir import Cell, Document, Page, Paragraph, Row, Run, Table
from .pdf_extractor import PDFExtractor
from .rtf_emitter import emit_rtf
from .rtf_parser import parse_rtf

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


def _create_synthetic_calibration_dataset() -> tuple[bytes, Document]:
    """Generates an in-memory PDF and matching reference Document IR for self-contained calibration."""
    if not HAS_FITZ:
        raise ImportError("PyMuPDF (fitz) is required to generate synthetic calibration PDF.")

    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=612, height=792)

    # Insert heading
    page.insert_text((72, 72), "Evaluation Benchmark Document", fontsize=16, fontname="helv")
    # Insert body paragraph 1
    page.insert_text((72, 120), "This is the primary calibration text for evolutionary tuning.", fontsize=11, fontname="helv")
    # Insert body paragraph 2 (separated by vertical gap to test paragraph break detection)
    page.insert_text((72, 160), "Second paragraph verifying spatial clustering and style retention.", fontsize=11, fontname="helv")

    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    # Corresponding Reference IR
    ref_doc = Document()
    ref_page = Page(1, 612.0, 792.0)
    ref_page.add_paragraph("Evaluation Benchmark Document", font="Helvetica", font_size_pt=16.0)
    ref_page.add_paragraph("This is the primary calibration text for evolutionary tuning.", font="Helvetica", font_size_pt=11.0)
    ref_page.add_paragraph("Second paragraph verifying spatial clustering and style retention.", font="Helvetica", font_size_pt=11.0)
    ref_doc.pages.append(ref_page)

    return pdf_bytes, ref_doc


@dataclass
class PDF2RTFSpec:
    """Domain specification holding calibration documents and verification expectations."""

    name: str = "pdf2rtf_benchmark"
    pdf_bytes: bytes | None = None
    pdf_path: str | None = None
    reference_doc: Document | None = None
    reference_rtf: str | None = None

    def get_pdf_bytes(self) -> bytes:
        if self.pdf_bytes is not None:
            return self.pdf_bytes
        if self.pdf_path is not None:
            p = Path(self.pdf_path)
            if p.exists() and p.is_file():
                return p.read_bytes()
        # Fallback to synthetic in-memory calibration document
        bytes_data, ref = _create_synthetic_calibration_dataset()
        self.pdf_bytes = bytes_data
        if self.reference_doc is None:
            self.reference_doc = ref
        return self.pdf_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "has_pdf": self.pdf_bytes is not None or self.pdf_path is not None,
            "pdf_path": self.pdf_path,
            "has_reference_doc": self.reference_doc is not None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PDF2RTFSpec:
        spec = cls(
            name=d.get("name", "pdf2rtf_benchmark"),
            pdf_path=d.get("pdf_path"),
        )
        if "reference_doc" in d and isinstance(d["reference_doc"], dict):
            spec.reference_doc = Document.from_dict(d["reference_doc"])
        return spec


class PDF2RTFEvaluator(Evaluator):
    """Evaluates candidate heuristic profiles by executing extraction, emission, parsing, and multi-gate verification."""

    def __init__(self, spec: PDF2RTFSpec, verifier: MultiGateVerifier | None = None) -> None:
        self.spec = spec
        self.verifier = verifier or MultiGateVerifier()
        # Ensure reference doc is materialized
        if self.spec.reference_doc is None:
            _bytes, ref = _create_synthetic_calibration_dataset()
            if self.spec.pdf_bytes is None:
                self.spec.pdf_bytes = _bytes
            self.spec.reference_doc = ref

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        genome = getattr(target, "genome", target)
        if not isinstance(genome, ProfileGenome):
            raise TypeError(f"Expected ProfileGenome, got {type(genome)}")

        policy = genome.to_policy()
        pdf_data = self.spec.get_pdf_bytes()

        # Step 1: Extract candidate IR from PDF with this policy
        extractor = PDFExtractor(policy=policy)
        candidate_ir = extractor.extract(pdf_data)

        # Step 2: Emit RTF from candidate IR
        candidate_rtf = emit_rtf(candidate_ir)

        # Step 3: Parse emitted RTF back to IR
        reconstructed_ir = parse_rtf(candidate_rtf)

        # Step 4: Multi-Gate Verification against reference IR
        report = self.verifier.verify(candidate=reconstructed_ir, reference=self.spec.reference_doc)

        return FitnessResult(
            score=report.composite_score,
            sub_scores={g.name: g.score for g in report.gates},
            artifacts=report.to_dict(),
        )


class PDF2RTFAdapter(DomainAdapter[ProfileGenome, PDF2RTFSpec, dict[str, Any]]):
    """Canonical DomainAdapter connecting PDF2RTF to Darwin-Evolab evolutionary kernel."""

    @property
    def name(self) -> str:
        return "pdf2rtf"

    def parse_spec(self, raw_input: Any) -> PDF2RTFSpec:
        """Parses specification from dict, JSON file path, or defaults to synthetic benchmark."""
        if isinstance(raw_input, PDF2RTFSpec):
            return raw_input
        if isinstance(raw_input, dict):
            return PDF2RTFSpec.from_dict(raw_input)
        if isinstance(raw_input, (str, Path)):
            p = Path(raw_input)
            if p.exists() and p.is_file():
                if p.suffix.lower() == ".json":
                    with open(p, "r", encoding="utf-8") as f:
                        return PDF2RTFSpec.from_dict(json.load(f))
                elif p.suffix.lower() == ".pdf":
                    return PDF2RTFSpec(name=p.stem, pdf_path=str(p))
            try:
                data = json.loads(str(raw_input))
                if isinstance(data, dict):
                    return PDF2RTFSpec.from_dict(data)
            except Exception:
                pass

        # Default synthetic calibration spec
        pdf_bytes, ref_doc = _create_synthetic_calibration_dataset()
        return PDF2RTFSpec(name="synthetic_default", pdf_bytes=pdf_bytes, reference_doc=ref_doc)

    def build_population(self, spec: PDF2RTFSpec, size: int, rng: random.Random) -> list[Individual]:
        """Initializes a diverse population of ProfileGenomes exploring the heuristic space."""
        population: list[Individual] = []

        # Always include the canonical baseline profile as an elite seed
        baseline = ProfileGenome()
        population.append(Individual(genome=baseline, species="spec_pdf_profile"))

        for _ in range(size - 1):
            genome = ProfileGenome(
                space_gap_ratio=rng.uniform(*PARAM_BOUNDS["space_gap_ratio"]),
                para_split_delta_ratio=rng.uniform(*PARAM_BOUNDS["para_split_delta_ratio"]),
                align_tolerance_pt=rng.uniform(*PARAM_BOUNDS["align_tolerance_pt"]),
                line_spacing_round_pt=rng.uniform(*PARAM_BOUNDS["line_spacing_round_pt"]),
                table_col_align_tol_pt=rng.uniform(*PARAM_BOUNDS["table_col_align_tol_pt"]),
                table_min_rows=int(round(rng.uniform(*PARAM_BOUNDS["table_min_rows"]))),
                table_min_cols=int(round(rng.uniform(*PARAM_BOUNDS["table_min_cols"]))),
            )
            population.append(Individual(genome=genome, species="spec_pdf_profile"))

        return population

    def build_evaluator(self, spec: PDF2RTFSpec) -> Evaluator:
        """Constructs a deterministic PDF2RTFEvaluator."""
        return PDF2RTFEvaluator(spec)

    def export_solution(
        self,
        individual: Individual,
        spec: PDF2RTFSpec,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Exports winning profile into a deployable profile.json artifact and summary."""
        genome = getattr(individual, "genome", individual)
        if not isinstance(genome, ProfileGenome):
            raise TypeError(f"Expected ProfileGenome, got {type(genome)}")

        policy = genome.to_policy()
        solution_data = {
            "name": spec.name,
            "fitness": getattr(individual, "fitness", 0.0),
            "fingerprint": genome.fingerprint(),
            "descriptors": genome.describe(),
            "policy": policy.to_dict(),
        }

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(solution_data, f, indent=2)
            solution_data["saved_path"] = str(out_p)

        return solution_data


# Register driver into Darwin-Evolab central registry
register_domain_adapter("pdf2rtf", PDF2RTFAdapter())
register_domain_adapter("pdf_to_rtf", PDF2RTFAdapter())
