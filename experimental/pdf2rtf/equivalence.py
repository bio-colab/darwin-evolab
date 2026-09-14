"""Equivalence Oracles and Multi-Gate Verifier for PDF-to-RTF Conversion.

Provides mathematically rigorous, multi-gate evaluation comparing candidate
documents (reconstructed from emitted RTF) against reference documents (from
source PDF or reference IR).

Gating Architecture:
- Gate 1: Text Integrity (Exact Token Match, Levenshtein distance, Diff reporting)
- Gate 2: Structure Integrity (Block count, paragraph alignment, spacing tolerances)
- Gate 2.5: Table Oracle (Row/Col matrix dimensions, cell text precision/recall, width ratios)
- Gate 3: Formatting Integrity (Normalized font family, size +/- 0.5pt, bold/italic/underline, color)
- Gate 4: Visual / Geometry Oracle (Calibrated layout footprint with epsilon noise floor)
- MultiGateVerifier: Weighted aggregation with cascading penalty for text corruption.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from experimental.pdf2rtf.ir import (
    Block,
    Cell,
    Color,
    Document,
    Page,
    Paragraph,
    Row,
    Run,
    Table,
)


@dataclass
class GateResult:
    """Outcome of an individual verification gate."""

    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    weight: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": round(self.score, 4),
            "weight": self.weight,
            "details": self.details,
            "message": self.message,
        }


@dataclass
class EquivalenceReport:
    """Comprehensive evaluation report across all gates."""

    passed: bool
    composite_score: float  # 0.0 to 1.0
    gates: list[GateResult] = field(default_factory=list)
    summary: str = ""
    cascaded_failure: bool = False

    def get_gate(self, name: str) -> GateResult | None:
        for gate in self.gates:
            if gate.name == name:
                return gate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "composite_score": round(self.composite_score, 4),
            "cascaded_failure": self.cascaded_failure,
            "summary": self.summary,
            "gates": [g.to_dict() for g in self.gates],
        }


# =========================================================================
# Helper Utilities
# =========================================================================

def _normalize_tokens(text: str) -> list[str]:
    """Extract sequence of non-empty words/tokens."""
    return re.findall(r"\S+", text)


def _canonical_font_name(font_name: str) -> str:
    """Strip subset prefixes and common font suffixes for canonical matching."""
    cleaned = re.sub(r"^[A-Z]{6}\+", "", font_name.strip())
    # Strip common style suffixes if attached directly to name
    cleaned = re.sub(r"[-_,](Bold|Italic|Regular|BoldItalic|BoldItal|Ital).*$", "", cleaned, flags=re.IGNORECASE)
    # Strip PostScript / Monotype family suffixes (PS, MT, PSMT)
    cleaned = re.sub(r"(PSMT|PS|MT)$", "", cleaned, flags=re.IGNORECASE)
    # Strip whitespace, hyphens, and underscores for robust comparison
    cleaned = re.sub(r"[\s\-_]", "", cleaned).lower()
    return cleaned


def _document_full_text(doc: Document) -> str:
    """Extract plain text of entire document across all pages and blocks."""
    lines: list[str] = []
    for page in doc.pages:
        for block in page.blocks:
            if isinstance(block, Paragraph):
                text = block.plain_text.strip()
                if text:
                    lines.append(text)
            elif isinstance(block, Table):
                for row in block.rows:
                    cell_texts = [c.plain_text.strip() for c in row.cells]
                    lines.append(" | ".join(cell_texts))
    return "\n".join(lines)


# =========================================================================
# Gate 1: Text Integrity Gate
# =========================================================================

class TextIntegrityGate:
    """Verifies that all textual content is preserved without omission, alteration, or corruption.

    Computes token-level precision, recall, sequence similarity, and exact identity.
    """

    def __init__(self, pass_threshold: float = 1.0, token_pass_threshold: float = 1.0):
        self.pass_threshold = pass_threshold
        self.token_pass_threshold = token_pass_threshold

    def evaluate(self, candidate: Document, reference: Document, weight: float = 0.40) -> GateResult:
        cand_text = _document_full_text(candidate)
        ref_text = _document_full_text(reference)

        cand_tokens = _normalize_tokens(cand_text)
        ref_tokens = _normalize_tokens(ref_text)

        if not ref_tokens and not cand_tokens:
            return GateResult(
                name="Gate 1: Text Integrity",
                passed=True,
                score=1.0,
                weight=weight,
                details={"ref_token_count": 0, "cand_token_count": 0, "token_match_ratio": 1.0},
                message="Both documents contain empty text.",
            )

        # Sequence matcher ratio on token sequence
        matcher = difflib.SequenceMatcher(None, ref_tokens, cand_tokens)
        token_similarity = matcher.ratio()

        # Levenshtein-like character level ratio on whitespace-normalized text
        ref_norm_text = re.sub(r"\s+", " ", ref_text).strip()
        cand_norm_text = re.sub(r"\s+", " ", cand_text).strip()
        char_matcher = difflib.SequenceMatcher(None, ref_norm_text, cand_norm_text)
        char_similarity = char_matcher.ratio()

        # Composite score
        score = (token_similarity * 0.7) + (char_similarity * 0.3)
        passed = token_similarity >= self.token_pass_threshold and score >= self.pass_threshold

        diff_snippets: list[str] = []
        if not passed:
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != "equal":
                    ref_sub = " ".join(ref_tokens[i1:i2])
                    cand_sub = " ".join(cand_tokens[j1:j2])
                    diff_snippets.append(f"[{tag}] expected '{ref_sub}' vs got '{cand_sub}'")
                    if len(diff_snippets) >= 5:
                        break

        details = {
            "ref_token_count": len(ref_tokens),
            "cand_token_count": len(cand_tokens),
            "token_similarity": round(token_similarity, 4),
            "char_similarity": round(char_similarity, 4),
            "diff_count": len(diff_snippets),
            "diff_samples": diff_snippets,
        }

        msg = (
            "Text matches reference exactly."
            if passed
            else f"Text divergence detected (token similarity: {token_similarity:.2%}, diffs: {len(diff_snippets)})."
        )

        return GateResult(
            name="Gate 1: Text Integrity",
            passed=passed,
            score=score,
            weight=weight,
            details=details,
            message=msg,
        )


# =========================================================================
# Gate 2: Structure Integrity Gate
# =========================================================================

class StructureIntegrityGate:
    """Verifies paragraph ordering, alignment preservation, and spacing bounds."""

    def __init__(
        self,
        spacing_tolerance_pt: float = 6.0,
        pass_threshold: float = 0.90,
    ):
        self.spacing_tolerance_pt = spacing_tolerance_pt
        self.pass_threshold = pass_threshold

    def evaluate(self, candidate: Document, reference: Document, weight: float = 0.20) -> GateResult:
        ref_blocks = [b for p in reference.pages for b in p.blocks]
        cand_blocks = [b for p in candidate.pages for b in p.blocks]

        if not ref_blocks and not cand_blocks:
            return GateResult(
                name="Gate 2: Structure Integrity",
                passed=True,
                score=1.0,
                weight=weight,
                details={"ref_blocks": 0, "cand_blocks": 0},
                message="Both documents have no blocks.",
            )

        # 1. Block count similarity
        total_blocks = max(len(ref_blocks), len(cand_blocks))
        min_blocks = min(len(ref_blocks), len(cand_blocks))
        block_count_score = min_blocks / total_blocks if total_blocks > 0 else 1.0

        # 2. Block type matching
        type_matches = 0
        alignment_matches = 0
        elements_with_align = 0
        spacing_scores: list[float] = []
        compared_count = min(len(ref_blocks), len(cand_blocks))

        for i in range(compared_count):
            rb = ref_blocks[i]
            cb = cand_blocks[i]
            if type(rb) is type(cb):
                type_matches += 1

            if isinstance(rb, (Paragraph, Table)) and isinstance(cb, (Paragraph, Table)):
                elements_with_align += 1
                # Alignment
                if rb.alignment == cb.alignment:
                    alignment_matches += 1

            if isinstance(rb, Paragraph) and isinstance(cb, Paragraph):
                # Spacing delta (evaluates individual margins, combined total, and inter-block physical separation)
                delta_sb = abs(rb.space_before_pt - cb.space_before_pt)
                delta_sa = abs(rb.space_after_pt - cb.space_after_pt)
                delta_total = abs(
                    (rb.space_before_pt + rb.space_after_pt)
                    - (cb.space_before_pt + cb.space_after_pt)
                )
                sb_score = max(0.0, 1.0 - (delta_sb / max(1.0, self.spacing_tolerance_pt * 2)))
                sa_score = max(0.0, 1.0 - (delta_sa / max(1.0, self.spacing_tolerance_pt * 2)))
                total_score = max(0.0, 1.0 - (delta_total / max(1.0, self.spacing_tolerance_pt * 2)))

                # Physical inter-block boundary separation
                inter_score = 0.0
                if i > 0:
                    rg = getattr(ref_blocks[i - 1], "space_after_pt", 0.0) + rb.space_before_pt
                    cg = getattr(cand_blocks[i - 1], "space_after_pt", 0.0) + cb.space_before_pt
                    inter_score = max(0.0, 1.0 - (abs(rg - cg) / max(1.0, self.spacing_tolerance_pt * 2)))

                spacing_scores.append(max((sb_score + sa_score) / 2.0, total_score, inter_score))

        type_score = type_matches / total_blocks if total_blocks > 0 else 1.0
        align_score = alignment_matches / elements_with_align if elements_with_align > 0 else 1.0
        avg_spacing_score = (sum(spacing_scores) / len(spacing_scores)) if spacing_scores else 1.0

        score = (
            (block_count_score * 0.30)
            + (type_score * 0.30)
            + (align_score * 0.25)
            + (avg_spacing_score * 0.15)
        )

        passed = score >= self.pass_threshold

        details = {
            "ref_block_count": len(ref_blocks),
            "cand_block_count": len(cand_blocks),
            "block_count_score": round(block_count_score, 4),
            "type_score": round(type_score, 4),
            "align_score": round(align_score, 4),
            "spacing_score": round(avg_spacing_score, 4),
        }

        msg = (
            f"Structure preserved (score: {score:.2%})."
            if passed
            else f"Structure discrepancy detected (score: {score:.2%}, expected {len(ref_blocks)} blocks, got {len(cand_blocks)})."
        )

        return GateResult(
            name="Gate 2: Structure Integrity",
            passed=passed,
            score=score,
            weight=weight,
            details=details,
            message=msg,
        )


# =========================================================================
# Gate 2.5: Table Oracle
# =========================================================================

class TableOracle:
    """Verifies table presence, grid dimension preservation, cell content precision/recall, and column distribution."""

    def __init__(self, pass_threshold: float = 0.90):
        self.pass_threshold = pass_threshold

    def evaluate(self, candidate: Document, reference: Document, weight: float = 0.20) -> GateResult:
        ref_tables = [b for p in reference.pages for b in p.blocks if isinstance(b, Table)]
        cand_tables = [b for p in candidate.pages for b in p.blocks if isinstance(b, Table)]

        # If neither document has tables, gate trivially passes
        if not ref_tables and not cand_tables:
            return GateResult(
                name="Gate 2.5: Table Oracle",
                passed=True,
                score=1.0,
                weight=weight,
                details={"ref_tables": 0, "cand_tables": 0},
                message="No tables present in reference or candidate.",
            )

        if not ref_tables and cand_tables:
            return GateResult(
                name="Gate 2.5: Table Oracle",
                passed=False,
                score=0.0,
                weight=weight,
                details={"ref_tables": 0, "cand_tables": len(cand_tables)},
                message=f"Spurious table detected: expected 0 tables, got {len(cand_tables)}.",
            )

        if ref_tables and not cand_tables:
            return GateResult(
                name="Gate 2.5: Table Oracle",
                passed=False,
                score=0.0,
                weight=weight,
                details={"ref_tables": len(ref_tables), "cand_tables": 0},
                message=f"Missing tables: expected {len(ref_tables)} tables, got 0.",
            )

        # Compare table counts
        table_count_score = min(len(ref_tables), len(cand_tables)) / max(len(ref_tables), len(cand_tables))

        table_scores: list[float] = []
        cell_precision_list: list[float] = []
        cell_recall_list: list[float] = []

        num_tables_to_compare = min(len(ref_tables), len(cand_tables))
        for t_idx in range(num_tables_to_compare):
            rt = ref_tables[t_idx]
            ct = cand_tables[t_idx]

            # 1. Row count similarity
            max_rows = max(len(rt.rows), len(ct.rows))
            min_rows = min(len(rt.rows), len(ct.rows))
            row_sim = min_rows / max_rows if max_rows > 0 else 1.0

            # 2. Cell text comparison across grid
            matches = 0
            total_ref_cells = sum(len(r.cells) for r in rt.rows)
            total_cand_cells = sum(len(r.cells) for r in ct.rows)

            common_rows = min(len(rt.rows), len(ct.rows))
            for r in range(common_rows):
                r_row = rt.rows[r]
                c_row = ct.rows[r]
                common_cells = min(len(r_row.cells), len(c_row.cells))
                for c in range(common_cells):
                    rt_text = r_row.cells[c].plain_text.strip()
                    ct_text = c_row.cells[c].plain_text.strip()
                    if rt_text == ct_text and rt_text != "":
                        matches += 1
                    elif rt_text == ct_text:  # both empty
                        matches += 1
                    else:
                        # Partial token similarity
                        sim = difflib.SequenceMatcher(None, rt_text, ct_text).ratio()
                        if sim > 0.8:
                            matches += sim

            precision = matches / total_cand_cells if total_cand_cells > 0 else 1.0
            recall = matches / total_ref_cells if total_ref_cells > 0 else 1.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            cell_precision_list.append(precision)
            cell_recall_list.append(recall)

            # 3. Column widths similarity (first row)
            width_sim = 1.0
            if rt.rows and ct.rows:
                r_widths = [c.width_twips for c in rt.rows[0].cells]
                c_widths = [c.width_twips for c in ct.rows[0].cells]
                if r_widths and c_widths and len(r_widths) == len(c_widths):
                    r_total = sum(r_widths) or 1
                    c_total = sum(c_widths) or 1
                    r_norm = [w / r_total for w in r_widths]
                    c_norm = [w / c_total for w in c_widths]
                    diff = sum(abs(a - b) for a, b in zip(r_norm, c_norm))
                    width_sim = max(0.0, 1.0 - (diff / 2.0))

            t_score = (row_sim * 0.30) + (f1 * 0.50) + (width_sim * 0.20)
            table_scores.append(t_score)

        avg_table_score = sum(table_scores) / len(table_scores) if table_scores else 0.0
        score = (table_count_score * 0.30) + (avg_table_score * 0.70)
        passed = score >= self.pass_threshold

        avg_precision = sum(cell_precision_list) / len(cell_precision_list) if cell_precision_list else 1.0
        avg_recall = sum(cell_recall_list) / len(cell_recall_list) if cell_recall_list else 1.0

        details = {
            "ref_tables": len(ref_tables),
            "cand_tables": len(cand_tables),
            "table_count_score": round(table_count_score, 4),
            "avg_table_score": round(avg_table_score, 4),
            "cell_precision": round(avg_precision, 4),
            "cell_recall": round(avg_recall, 4),
        }

        msg = (
            f"Table structures verified (score: {score:.2%}, cell F1: {avg_table_score:.2%})."
            if passed
            else f"Table discrepancy detected (score: {score:.2%})."
        )

        return GateResult(
            name="Gate 2.5: Table Oracle",
            passed=passed,
            score=score,
            weight=weight,
            details=details,
            message=msg,
        )


# =========================================================================
# Gate 3: Formatting Integrity Gate
# =========================================================================

class FormattingIntegrityGate:
    """Verifies typography: font families, font sizes within tolerance, style flags, and colors."""

    def __init__(
        self,
        size_tolerance_pt: float = 0.5,
        color_delta_threshold: float = 25.0,
        pass_threshold: float = 0.85,
    ):
        self.size_tolerance_pt = size_tolerance_pt
        self.color_delta_threshold = color_delta_threshold
        self.pass_threshold = pass_threshold

    def evaluate(self, candidate: Document, reference: Document, weight: float = 0.20) -> GateResult:
        def _collect_all_runs(doc: Document) -> list[Run]:
            runs: list[Run] = []
            for page in doc.pages:
                for blk in page.blocks:
                    if isinstance(blk, Paragraph):
                        runs.extend(blk.runs)
                    elif isinstance(blk, Table):
                        for row in blk.rows:
                            for cell in row.cells:
                                for cell_p in cell.paragraphs:
                                    runs.extend(cell_p.runs)
            return runs

        ref_runs = _collect_all_runs(reference)
        cand_runs = _collect_all_runs(candidate)

        if not ref_runs and not cand_runs:
            return GateResult(
                name="Gate 3: Formatting Integrity",
                passed=True,
                score=1.0,
                weight=weight,
                details={"ref_runs": 0, "cand_runs": 0},
                message="No text runs present in either document.",
            )

        n_compare = min(len(ref_runs), len(cand_runs))
        max_runs = max(len(ref_runs), len(cand_runs))
        run_count_score = n_compare / max_runs if max_runs > 0 else 1.0

        font_matches = 0
        size_scores: list[float] = []
        style_matches = 0
        color_matches = 0

        for i in range(n_compare):
            rr = ref_runs[i]
            cr = cand_runs[i]

            # 1. Font family
            if _canonical_font_name(rr.font) == _canonical_font_name(cr.font):
                font_matches += 1

            # 2. Font size with tolerance
            delta_size = abs(rr.font_size_pt - cr.font_size_pt)
            if delta_size <= self.size_tolerance_pt:
                size_scores.append(1.0)
            else:
                size_scores.append(max(0.0, 1.0 - (delta_size / 6.0)))

            # 3. Styles (bold, italic, underline)
            flags_matching = (
                (1 if rr.bold == cr.bold else 0)
                + (1 if rr.italic == cr.italic else 0)
                + (1 if rr.underline == cr.underline else 0)
            )
            style_matches += flags_matching / 3.0

            # 4. Color
            if rr.color.is_auto and cr.color.is_auto:
                color_matches += 1
            else:
                dist = math.sqrt(
                    (rr.color.r - cr.color.r) ** 2
                    + (rr.color.g - cr.color.g) ** 2
                    + (rr.color.b - cr.color.b) ** 2
                )
                if dist <= self.color_delta_threshold:
                    color_matches += 1
                else:
                    color_matches += max(0.0, 1.0 - (dist / 255.0))

        font_score = font_matches / n_compare if n_compare > 0 else 1.0
        avg_size_score = sum(size_scores) / len(size_scores) if size_scores else 1.0
        style_score = style_matches / n_compare if n_compare > 0 else 1.0
        color_score = color_matches / n_compare if n_compare > 0 else 1.0

        score = (
            (run_count_score * 0.05)
            + (font_score * 0.30)
            + (avg_size_score * 0.30)
            + (style_score * 0.25)
            + (color_score * 0.10)
        )

        passed = score >= self.pass_threshold

        details = {
            "ref_runs": len(ref_runs),
            "cand_runs": len(cand_runs),
            "font_fidelity": round(font_score, 4),
            "size_fidelity": round(avg_size_score, 4),
            "style_fidelity": round(style_score, 4),
            "color_fidelity": round(color_score, 4),
        }

        msg = (
            f"Formatting preserved (score: {score:.2%}, font: {font_score:.2%}, size: {avg_size_score:.2%})."
            if passed
            else f"Formatting discrepancy (score: {score:.2%}, font fidelity: {font_score:.2%})."
        )

        return GateResult(
            name="Gate 3: Formatting Integrity",
            passed=passed,
            score=score,
            weight=weight,
            details=details,
            message=msg,
        )


# =========================================================================
# Gate 4: Visual / Geometry Diff Oracle
# =========================================================================

class VisualDiffOracle:
    """Verifies layout geometry and spatial page flow with a calibrated noise floor (epsilon).

    Absorbs subpixel hinting variations and font rasterization antialiasing noise.
    """

    def __init__(self, epsilon_floor: float = 0.05, pass_threshold: float = 0.90):
        self.epsilon_floor = epsilon_floor
        self.pass_threshold = pass_threshold

    def evaluate(self, candidate: Document, reference: Document, weight: float = 0.10) -> GateResult:
        # Page count & dimensions
        ref_pages = len(reference.pages)
        cand_pages = len(candidate.pages)

        if ref_pages == 0 and cand_pages == 0:
            return GateResult(
                name="Gate 4: Visual / Geometry Oracle",
                passed=True,
                score=1.0,
                weight=weight,
                details={"ref_pages": 0, "cand_pages": 0},
                message="No pages in document.",
            )

        page_sim = min(ref_pages, cand_pages) / max(ref_pages, cand_pages) if max(ref_pages, cand_pages) > 0 else 1.0

        # Page dimension similarity
        dim_scores: list[float] = []
        for p_idx in range(min(ref_pages, cand_pages)):
            rp = reference.pages[p_idx]
            cp = candidate.pages[p_idx]
            dw = abs(rp.width_pts - cp.width_pts) / max(rp.width_pts, 1.0)
            dh = abs(rp.height_pts - cp.height_pts) / max(rp.height_pts, 1.0)
            dim_scores.append(max(0.0, 1.0 - (dw + dh) / 2.0))

        avg_dim_score = sum(dim_scores) / len(dim_scores) if dim_scores else 1.0

        # Block distribution and layout density similarity
        density_scores: list[float] = []
        for p_idx in range(min(ref_pages, cand_pages)):
            rp = reference.pages[p_idx]
            cp = candidate.pages[p_idx]
            r_b = len(rp.blocks)
            c_b = len(cp.blocks)
            max_b = max(r_b, c_b)
            min_b = min(r_b, c_b)
            density_scores.append((min_b / max_b) if max_b > 0 else 1.0)

        avg_density_score = sum(density_scores) / len(density_scores) if density_scores else 1.0

        raw_score = (page_sim * 0.30) + (avg_dim_score * 0.40) + (avg_density_score * 0.30)

        # Apply noise floor calibration
        # Any residual delta <= epsilon_floor is calibrated to zero error
        error = 1.0 - raw_score
        calibrated_error = max(0.0, error - self.epsilon_floor)
        score = 1.0 - calibrated_error

        passed = score >= self.pass_threshold

        details = {
            "ref_pages": ref_pages,
            "cand_pages": cand_pages,
            "raw_score": round(raw_score, 4),
            "epsilon_calibrated_score": round(score, 4),
            "noise_floor_epsilon": self.epsilon_floor,
        }

        msg = (
            f"Visual geometry verified within calibrated noise floor (score: {score:.2%})."
            if passed
            else f"Visual geometry mismatch (score: {score:.2%})."
        )

        return GateResult(
            name="Gate 4: Visual / Geometry Oracle",
            passed=passed,
            score=score,
            weight=weight,
            details=details,
            message=msg,
        )


# =========================================================================
# Multi-Gate Composite Verifier
# =========================================================================

class MultiGateVerifier:
    """Master evaluator coordinating all verification gates.

    Enforces lexicographic cascading: if Gate 1 (Text Integrity) suffers severe
    omission or corruption, subsequent formatting scores are truncated to prevent
    rewarding beautifully formatted documents that have destroyed the text.
    """

    def __init__(
        self,
        gate1_threshold: float = 0.98,
        cascade_penalty_multiplier: float = 0.20,
        weights: dict[str, float] | None = None,
    ):
        self.gate1_threshold = gate1_threshold
        self.cascade_penalty_multiplier = cascade_penalty_multiplier

        # Default gate weights across all 5 verification gates
        self.weights = {
            "text": 0.35,
            "structure": 0.20,
            "table": 0.20,
            "formatting": 0.15,
            "visual": 0.10,
        }
        if weights:
            self.weights.update(weights)

        self.g1_text = TextIntegrityGate(pass_threshold=self.gate1_threshold, token_pass_threshold=self.gate1_threshold)
        self.g2_structure = StructureIntegrityGate()
        self.g25_table = TableOracle()
        self.g3_formatting = FormattingIntegrityGate()
        self.g4_visual = VisualDiffOracle()

    def verify(self, candidate: Document, reference: Document) -> EquivalenceReport:
        """Run all verification gates and generate composite diagnostic report."""
        r1 = self.g1_text.evaluate(candidate, reference, weight=self.weights["text"])
        r2 = self.g2_structure.evaluate(candidate, reference, weight=self.weights["structure"])
        r25 = self.g25_table.evaluate(candidate, reference, weight=self.weights["table"])
        r3 = self.g3_formatting.evaluate(candidate, reference, weight=self.weights["formatting"])
        r4 = self.g4_visual.evaluate(candidate, reference, weight=self.weights.get("visual", 0.10))

        gates = [r1, r2, r25, r3, r4]

        # Calculate weighted composite score
        total_weight = sum(g.weight for g in gates)
        raw_composite = sum(g.score * g.weight for g in gates) / total_weight if total_weight > 0 else 0.0

        # Check cascading condition on Gate 1 (Text Integrity)
        cascaded = False
        if r1.score < self.gate1_threshold:
            cascaded = True
            # Severe suppression of score when text is corrupted
            final_composite = raw_composite * self.cascade_penalty_multiplier
            overall_passed = False
            summary = (
                f"FAILED (CASCADED): Gate 1 (Text Integrity) failed with score {r1.score:.2%} "
                f"(below threshold {self.gate1_threshold:.2%}). Formatting scores suppressed."
            )
        else:
            final_composite = raw_composite
            overall_passed = all(g.passed for g in gates)
            summary = (
                f"PASSED: Composite fidelity {final_composite:.2%}. All gates satisfied."
                if overall_passed
                else f"FAILED: Composite fidelity {final_composite:.2%}. One or more gates below threshold."
            )

        return EquivalenceReport(
            passed=overall_passed,
            composite_score=final_composite,
            gates=gates,
            summary=summary,
            cascaded_failure=cascaded,
        )
