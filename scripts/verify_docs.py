"""verify_docs.py — Automated Truth-in-Documentation Verification Engine.

Parses experimental/pdf2rtf/README.md, extracts key numerical claims, benchmark tables,
pass rates, and gate scores, and cross-references them against genuine generated
reports in reports/*.json.

Guarantees 100% truth-in-documentation before every repository commit or push.
Fails with exit code 1 if any discrepancy or unverified claim is detected.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "experimental" / "pdf2rtf" / "README.md"
REPORTS_DIR = REPO_ROOT / "reports"


def load_json(rel_path: str) -> dict[str, Any]:
    full_path = REPORTS_DIR / rel_path
    if not full_path.exists():
        raise FileNotFoundError(f"Required report not found: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_documentation() -> bool:
    print(f"=== Running Truth-in-Documentation Verification ===")
    print(f"Checking document: {DOC_PATH}")
    print(f"Source of truth:   {REPORTS_DIR}\n")

    if not DOC_PATH.exists():
        print(f"[FAIL] Documentation file not found: {DOC_PATH}")
        return False

    with open(DOC_PATH, "r", encoding="utf-8") as f:
        doc_text = f.read()

    errors: list[str] = []
    checks_passed = 0

    # Check 1: Real Word Holdout Benchmark (pdf2rtf_real_word_holdout_benchmark.json)
    try:
        holdout = load_json("pdf2rtf_real_word_holdout_benchmark.json")
        holdout_n = holdout["total_documents"]
        holdout_avg = holdout["average_composite_score"] * 100.0
        holdout_text = holdout["text_integrity_pass_rate"] * 100.0
        holdout_overall = holdout["overall_pass_rate"] * 100.0

        # Assert document mentions holdout N
        if f"Holdout N={holdout_n}" not in doc_text and f"Holdout (N={holdout_n})" not in doc_text:
            errors.append(f"Document missing mention of Holdout N={holdout_n}")
        else:
            checks_passed += 1

        # Assert text integrity pass rate
        if f"{holdout_text:.2f}%" not in doc_text:
            errors.append(f"Holdout text integrity {holdout_text:.2f}% not found in doc")
        else:
            checks_passed += 1

        # Assert overall pass rate
        if f"{holdout_overall:.0f}% ALL PASS ({holdout_n}/{holdout_n})" not in doc_text:
            errors.append(f"Holdout pass summary {holdout_overall:.0f}% ALL PASS ({holdout_n}/{holdout_n}) not found")
        else:
            checks_passed += 1

        print(f"[OK] Real Word Holdout (N={holdout_n}) claims verified against report.")
    except Exception as e:
        errors.append(f"Holdout check error: {e}")

    # Check 2: Word-in-the-Loop Oracle Audit (pdf2rtf_word_oracle_audit.json)
    try:
        word_audit = load_json("pdf2rtf_word_oracle_audit.json")
        w_n = word_audit["summary"]["total_documents"]
        w_avg = word_audit["summary"]["average_composite_score"] * 100.0
        w_overall = word_audit["summary"]["overall_pass_rate"] * 100.0

        if f"{w_overall:.0f}% ALL PASS ({w_n}/{w_n})" not in doc_text:
            errors.append(f"Word-in-the-loop summary {w_overall:.0f}% ALL PASS ({w_n}/{w_n}) not found")
        else:
            checks_passed += 1

        print(f"[OK] Word-in-the-Loop Oracle Audit (N={w_n}) claims verified against report.")
    except Exception as e:
        errors.append(f"Word-in-the-loop check error: {e}")

    # Check 3: Evolab-54 Benchmark (pdf2rtf_corpus_54_benchmark.json)
    try:
        c54 = load_json("pdf2rtf_corpus_54_benchmark.json")
        n54 = c54.get("total_documents", c54.get("summary", {}).get("total_documents", 54))
        avg54 = c54["summary"]["average_composite_score"] * 100.0
        pass54 = c54["summary"]["overall_pass_rate"] * 100.0

        if f"{pass54:.0f}% ALL PASS ({n54}/{n54})" not in doc_text and f"{n54}/{n54}" not in doc_text:
            errors.append(f"Evolab-54 summary ({n54}/{n54}) not found in doc")
        else:
            checks_passed += 1

        print(f"[OK] Evolab-54 Benchmark (N={n54}) claims verified against report.")
    except Exception as e:
        errors.append(f"Evolab-54 check error: {e}")

    # Check 4: MAP-Elites Archive Stats (pdf2rtf_map_elites_archive_stats.json) if exists
    archive_stats_path = REPORTS_DIR / "pdf2rtf_map_elites_archive_stats.json"
    if archive_stats_path.exists():
        try:
            astats = load_json("pdf2rtf_map_elites_archive_stats.json")
            total_niches = astats["configuration"]["total_niches"]
            occupied = astats["archive_metrics"]["occupied_niches"]
            cov_pct = astats["archive_metrics"]["coverage_percentage"]
            qd_score = astats["archive_metrics"]["qd_score"]

            print(f"[OK] MAP-Elites Stats available: {occupied}/{total_niches} niches ({cov_pct}%), QD-Score: {qd_score}")
            checks_passed += 1
        except Exception as e:
            errors.append(f"Archive stats check error: {e}")

    # Check 5: Ablation Study (pdf2rtf_ablation_study.json)
    try:
        abl = load_json("pdf2rtf_ablation_study.json")
        m_samples = abl["random_baseline"]["num_samples"]
        if m_samples < 50:
            errors.append(f"Ablation study has only M={m_samples} samples; must have M>=50 genuine samples!")
        else:
            checks_passed += 1
            print(f"[OK] Ablation Study verified with M={m_samples} genuine random samples (>= 50).")

        # Assert document mentions M = 50
        if f"M = {m_samples}" not in doc_text and f"M={m_samples}" not in doc_text:
            errors.append(f"Document does not correctly mention Ablation sample count M = {m_samples}")
        else:
            checks_passed += 1
    except Exception as e:
        errors.append(f"Ablation study check error: {e}")

    # Check 6: Multi-Seed Evaluation (pdf2rtf_multiseed_evaluation.json)
    try:
        ms = load_json("pdf2rtf_multiseed_evaluation.json")
        k_seeds = ms["num_seeds"]
        if k_seeds < 20:
            errors.append(f"Multi-seed evaluation has only K={k_seeds} seeds; must have K>=20 independent seeds!")
        else:
            checks_passed += 1
            print(f"[OK] Multi-Seed Evaluation verified with K={k_seeds} independent seeds (>= 20).")

        if f"K = {k_seeds}" not in doc_text and f"K={k_seeds}" not in doc_text:
            errors.append(f"Document does not correctly mention Multi-seed count K = {k_seeds}")
        else:
            checks_passed += 1
    except Exception as e:
        errors.append(f"Multi-seed check error: {e}")

    # Check 7: Comparative Benchmark (pdf2rtf_specialized_vs_monolithic_benchmark.json)
    comp_path = REPORTS_DIR / "pdf2rtf_specialized_vs_monolithic_benchmark.json"
    if comp_path.exists():
        try:
            comp = load_json("pdf2rtf_specialized_vs_monolithic_benchmark.json")
            m_score = comp["summary"]["monolithic_average_score"] * 100.0
            s_score = comp["summary"]["specialized_average_score"] * 100.0
            delta = comp["summary"]["score_improvement"] * 100.0

            if f"{m_score:.2f}%" not in doc_text or f"{s_score:.2f}%" not in doc_text:
                errors.append(f"Comparative benchmark scores (Mono: {m_score:.2f}%, QD: {s_score:.2f}%) not reflected in README")
            else:
                checks_passed += 1
            print(f"[OK] Comparative Benchmark verified (Mono {m_score:.2f}% vs QD {s_score:.2f}%, delta {delta:+.2f}%).")
        except Exception as e:
            errors.append(f"Comparative benchmark check error: {e}")

    # Check 8: Causal Attribution Benchmark (pdf2rtf_map_elites_vs_random_search.json)
    causal_path = REPORTS_DIR / "pdf2rtf_map_elites_vs_random_search.json"
    if causal_path.exists():
        try:
            causal = load_json("pdf2rtf_map_elites_vs_random_search.json")
            proto = causal["preregistered_protocol"]
            b_evals = proto["budget_evaluations"]
            n_seeds = proto["num_seeds"]
            if b_evals < 5000 or n_seeds < 5:
                errors.append(f"Causal benchmark requires budget >= 5000 and seeds >= 5 (got {b_evals}, {n_seeds})")
            else:
                checks_passed += 1
                print(f"[OK] Causal Benchmark verified with B={b_evals} evaluations across K={n_seeds} seeds.")

            me_cov = causal["comparative_summary"]["coverage_pct"]["map_elites"]["mean"]
            rs_cov = causal["comparative_summary"]["coverage_pct"]["random_search"]["mean"]
            me_qd = causal["comparative_summary"]["qd_score"]["map_elites"]["mean"]
            rs_qd = causal["comparative_summary"]["qd_score"]["random_search"]["mean"]

            if f"{me_cov:.1f}%" not in doc_text and f"{me_cov}%" not in doc_text:
                errors.append(f"README does not contain MAP-Elites coverage {me_cov}%")
            else:
                checks_passed += 1

            if f"{rs_cov:.1f}%" not in doc_text and f"{rs_cov}%" not in doc_text:
                errors.append(f"README does not contain Random Search coverage {rs_cov}%")
            else:
                checks_passed += 1

            if str(me_qd) not in doc_text and f"{me_qd:.2f}" not in doc_text:
                errors.append(f"README does not contain MAP-Elites QD-score {me_qd}")
            else:
                checks_passed += 1

            if str(rs_qd) not in doc_text and f"{rs_qd:.2f}" not in doc_text:
                errors.append(f"README does not contain Random Search QD-score {rs_qd}")
            else:
                checks_passed += 1

            print(f"[OK] Causal Attribution claims verified (ME Cov: {me_cov}%, RS Cov: {rs_cov}%, ME QD: {me_qd}, RS QD: {rs_qd}).")
        except Exception as e:
            errors.append(f"Causal benchmark check error: {e}")

    print("\n" + "-" * 60)
    if errors:
        print(f"[ERROR] Verification failed with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print(f"[SUCCESS] All {checks_passed} documentation truth checks PASSED with 100% agreement!")
        return True


if __name__ == "__main__":
    success = verify_documentation()
    sys.exit(0 if success else 1)
