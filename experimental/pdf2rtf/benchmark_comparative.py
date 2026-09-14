"""benchmark_comparative.py — Monolithic Champion vs. MAP-Elites Niche Dispatch Benchmark.

Audits the entire Evolab-54 orthogonal factorial benchmark suite under two distinct regimes:
1. Monolithic Champion Policy: A single globally calibrated heuristic profile applied uniformly.
2. MAP-Elites Niche-Specialized Dispatch: Document-adaptive Quality-Diversity dispatching
   specialized elites trained for specific layout density and table sensitivity niches.

Produces reports/pdf2rtf_specialized_vs_monolithic_benchmark.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .equivalence import MultiGateVerifier
from .genome import ProfilePolicy
from .map_elites_archive import MAPElitesArchive, compute_document_descriptors
from .pdf_extractor import PDFExtractor
from .rtf_emitter import emit_rtf
from .rtf_parser import parse_rtf
from .word_corpus_54 import HoldoutItem54, load_corpus_54

REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "pdf2rtf_specialized_vs_monolithic_benchmark.json"
CHAMPION_PROFILE_PATH = Path(__file__).resolve().parent / "calibrated_word_profile.json"
ARCHIVE_PATH = Path(__file__).resolve().parent / "calibrated_map_elites_archive.json"


def run_comparative_benchmark(
    corpus_items: list[HoldoutItem54] | None = None,
    champion_path: Path | str | None = None,
    archive_path: Path | str | None = None,
    save_report_path: Path | str | None = None,
) -> dict[str, Any]:
    items = corpus_items or load_corpus_54()
    c_path = Path(champion_path or CHAMPION_PROFILE_PATH)
    a_path = Path(archive_path or ARCHIVE_PATH)
    out_path = Path(save_report_path or REPORT_PATH)

    with open(c_path, "r", encoding="utf-8") as f:
        champ_data = json.load(f)
    champion_policy = ProfilePolicy.from_dict(champ_data["policy"])

    archive = MAPElitesArchive.load_json(a_path)

    extractor_mono = PDFExtractor(policy=champion_policy)
    extractor_qd = PDFExtractor(archive=archive)
    verifier = MultiGateVerifier()

    print(f"=== Running Comparative Benchmark: Monolithic vs. MAP-Elites Dispatch (N={len(items)}) ===")
    print(f"  Archive Niches Available: {archive.occupied_count}/{archive.total_capacity} (QD-Score: {archive.qd_score:.2f})")

    results: list[dict[str, Any]] = []
    group_stats_mono: dict[str, list[float]] = {}
    group_stats_qd: dict[str, list[float]] = {}
    mono_scores: list[float] = []
    qd_scores: list[float] = []
    mono_pass_count = 0
    qd_pass_count = 0

    for idx, item in enumerate(items, start=1):
        t0 = time.perf_counter()
        # 1. Monolithic Champion Evaluation
        cand_mono = extractor_mono.extract(item.pdf_bytes)
        rtf_mono = emit_rtf(cand_mono)
        recon_mono = parse_rtf(rtf_mono)
        rep_mono = verifier.verify(candidate=recon_mono, reference=item.reference_doc)

        # 2. MAP-Elites Specialized Dispatch Evaluation
        cand_qd = extractor_qd.extract(item.pdf_bytes)
        rtf_qd = emit_rtf(cand_qd)
        recon_qd = parse_rtf(rtf_qd)
        rep_qd = verifier.verify(candidate=recon_qd, reference=item.reference_doc)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        d1, d2 = compute_document_descriptors(item.pdf_bytes)
        dispatched_cell = extractor_qd.last_dispatched_cell
        is_exact = extractor_qd.last_dispatched_exact

        mono_scores.append(rep_mono.composite_score)
        qd_scores.append(rep_qd.composite_score)
        if rep_mono.passed:
            mono_pass_count += 1
        if rep_qd.passed:
            qd_pass_count += 1

        grp = item.group
        group_stats_mono.setdefault(grp, []).append(rep_mono.composite_score)
        group_stats_qd.setdefault(grp, []).append(rep_qd.composite_score)

        delta = rep_qd.composite_score - rep_mono.composite_score
        delta_str = f"+{delta:.2%}" if delta > 0 else (f"{delta:.2%}" if delta < 0 else "=")

        results.append({
            "name": item.name,
            "group": grp,
            "description": item.description,
            "descriptors": {"cluster_density": d1, "table_sensitivity": d2},
            "dispatched_niche": list(dispatched_cell.coord) if dispatched_cell else [],
            "niche_exact_match": is_exact,
            "monolithic_score": round(rep_mono.composite_score, 4),
            "specialized_score": round(rep_qd.composite_score, 4),
            "score_delta": round(delta, 4),
            "monolithic_passed": rep_mono.passed,
            "specialized_passed": rep_qd.passed,
            "elapsed_ms": round(elapsed_ms, 2),
        })

        if idx % 10 == 0 or idx == len(items):
            print(f"  [{idx:02d}/{len(items)}] {item.name:<38} -> Mono: {rep_mono.composite_score:.2%}, QD: {rep_qd.composite_score:.2%} ({delta_str})")

    # Aggregate summaries
    avg_mono = sum(mono_scores) / len(mono_scores) if mono_scores else 0.0
    avg_qd = sum(qd_scores) / len(qd_scores) if qd_scores else 0.0
    pass_rate_mono = mono_pass_count / len(items) if items else 0.0
    pass_rate_qd = qd_pass_count / len(items) if items else 0.0

    group_comparison: dict[str, dict[str, Any]] = {}
    for grp in sorted(group_stats_mono.keys()):
        m_vals = group_stats_mono[grp]
        q_vals = group_stats_qd[grp]
        m_avg = sum(m_vals) / len(m_vals) if m_vals else 0.0
        q_avg = sum(q_vals) / len(q_vals) if q_vals else 0.0
        group_comparison[grp] = {
            "monolithic_average": round(m_avg, 4),
            "specialized_average": round(q_avg, 4),
            "delta": round(q_avg - m_avg, 4),
            "items_count": len(m_vals),
        }

    report = {
        "title": "Monolithic Champion vs. MAP-Elites Niche-Specialized Dispatch Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_documents": len(items),
        "archive_metrics": {
            "total_niches": archive.total_capacity,
            "occupied_niches": archive.occupied_count,
            "coverage_percentage": round(archive.coverage * 100.0, 2),
            "qd_score": round(archive.qd_score, 4),
        },
        "summary": {
            "monolithic_average_score": round(avg_mono, 4),
            "specialized_average_score": round(avg_qd, 4),
            "score_improvement": round(avg_qd - avg_mono, 4),
            "monolithic_pass_rate": round(pass_rate_mono, 4),
            "specialized_pass_rate": round(pass_rate_qd, 4),
        },
        "group_breakdown": group_comparison,
        "documents": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("COMPARATIVE BENCHMARK SUMMARY:")
    print(f"  Monolithic Champion Composite:      {avg_mono:.2%} (Pass: {pass_rate_mono:.2%})")
    print(f"  MAP-Elites Specialized Composite:   {avg_qd:.2%} (Pass: {pass_rate_qd:.2%})")
    print(f"  Net Fidelity Delta:                 {avg_qd - avg_mono:+.2%}")
    for grp, g_data in group_comparison.items():
        print(f"    Group {grp} (N={g_data['items_count']}): Mono {g_data['monolithic_average']:.2%} -> QD {g_data['specialized_average']:.2%} ({g_data['delta']:+.2%})")
    print(f"\nReport saved to: {out_path}")
    print("=" * 65)

    return report


if __name__ == "__main__":
    run_comparative_benchmark()
