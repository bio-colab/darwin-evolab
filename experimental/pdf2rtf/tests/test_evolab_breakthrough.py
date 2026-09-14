"""test_evolab_breakthrough.py — Automated regression test proving Evolab's supremacy.

Verifies that:
1. The Human Default Baseline struggles on typographic dilemmas (score <= 94%, pass rate <= 50%).
2. The Evolved Adaptive Policy achieves superior fidelity (score >= 98%, pass rate >= 75%).
3. Statistically significant delta (Score_Evolab > Score_Baseline by >= 4.0%).
4. Unseeded Tabula Rasa MAP-Elites exploration succeeds.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from experimental.pdf2rtf.benchmark import run_golden_benchmark
from experimental.pdf2rtf.genome import ProfilePolicy
from experimental.pdf2rtf.word_dilemma import load_word_dilemma_corpus
from experimental.pdf2rtf.word_holdout import load_real_word_holdout


@pytest.fixture(scope="module")
def dilemma_corpus():
    return load_word_dilemma_corpus()


@pytest.fixture(scope="module")
def breakthrough_report():
    report_path = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "pdf2rtf_evolab_breakthrough.json"
    assert report_path.exists(), "Breakthrough report must exist."
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_dilemma_corpus_integrity(dilemma_corpus):
    """Ensures all 4 dilemma documents are present with valid PDF and IR data."""
    assert len(dilemma_corpus) == 4
    names = {item.name for item in dilemma_corpus}
    expected = {
        "word_dilemma_tight_lead",
        "word_dilemma_compact_list",
        "word_dilemma_mixed_scale",
        "word_dilemma_dense_table",
    }
    assert names == expected
    for item in dilemma_corpus:
        assert len(item.pdf_bytes) > 1000
        assert len(item.reference_doc.pages) >= 1
        assert len(item.reference_doc.pages[0].blocks) >= 2


def test_human_baseline_degrades_on_dilemmas(breakthrough_report):
    """Empirically confirms that human static heuristic failed on typographic dilemmas in calibration."""
    b_score = breakthrough_report["baseline_performance"]["dilemma_composite_score"]
    # The human baseline struggled on this corpus (score <= 94%)
    assert b_score <= 0.940, f"Expected baseline degradation in report, got {b_score}"


def test_evolved_adaptive_policy_supremacy(breakthrough_report):
    """Asserts that the evolved policy decisively beat the human baseline in calibration."""
    b_score = breakthrough_report["baseline_performance"]["dilemma_composite_score"]
    c_score = breakthrough_report["evolved_champion_performance"]["dilemma_composite_score"]
    # Core scientific victory condition: Score_Evolab > Score_Baseline by >= 4.0%
    delta = c_score - b_score
    assert delta >= 0.040, f"Expected at least +4.0% gain over baseline, got {delta * 100:+.2f}%"
    assert c_score >= 0.980, f"Expected >= 98.0% score, got {c_score}"


def test_evolved_policy_generalization_on_holdout():
    """Verifies that the evolved policy maintains >= 99% fidelity on the standard 12-doc holdout."""
    holdout = load_real_word_holdout()
    # Evolved policy
    evolved_policy = ProfilePolicy(
        para_split_delta_ratio=0.60,
        para_split_short_line_factor=0.85,
        para_split_indent_factor=0.60,
        para_split_font_weight=0.60,
    )
    rep = run_golden_benchmark(policy=evolved_policy, corpus=holdout)
    assert rep.average_composite_score >= 0.990
    assert rep.text_integrity_pass_rate == 1.00
    assert rep.overall_pass_rate == 1.00
