"""verify_recursive_rsi.py — Empirical Multi-Stage Recursive Policy Improvement Verification.

Executes the tripartite recursive learning pipeline:
D_0 -> π_1 -> D_1 -> π_2-op / π_2-context -> D_2 (Fresh Unseen Holdouts)
Across 3 independent seeds (100, 2026, 42) yielding 75 strictly unseen holdout tasks.

Generates reports/recursive_rsi_evaluation.json and verifies:
- Strict cohort disjointness: D_0 ∩ D_1 ∩ D_2 = ∅ for each seed
- Monotonic progression: evals(π_2-context) < evals(π_2-op) < evals(π_1) < evals(π_0)
- Statistical significance: paired Student's t-test p < 0.05
- Cohen's d effect size > 0.8
- 0 regressions on holdout tasks
- Statistical Governor ACCEPT verdict
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Configure UTF-8 encoding for standard output if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evolab.dream.recursive_rsi import RecursiveRSIConfig, run_recursive_rsi_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = REPO_ROOT / "src" / "evolab" / "fixtures" / "swe_bench_300"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "recursive_rsi_evaluation.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Stage Recursive Policy Improvement Verification")
    parser.add_argument("--n-per-stage", type=int, default=25, help="Number of instances per stage cohort (default: 25)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 2026, 42], help="Random seeds (default: 100 2026 42)")
    parser.add_argument("--fixtures-dir", type=str, default=str(DEFAULT_FIXTURES), help="Fixtures directory")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--governor-alpha", type=float, default=0.05, help="Governor alpha threshold")
    parser.add_argument("--markdown-out", type=str, default=None, help="Optional path to write summary markdown")
    args = parser.parse_args()

    config = RecursiveRSIConfig(
        n_per_stage=args.n_per_stage,
        seeds=args.seeds,
        governor_alpha=args.governor_alpha,
    )

    print("=== Starting Multi-Stage Recursive Search Policy Verification ===")
    print(f"Dataset:       {args.fixtures_dir}")
    print(f"Cohort size:   {args.n_per_stage} instances per stage (D_0, D_1, D_2)")
    print(f"Seeds:         {args.seeds}")
    print(f"Target report: {args.output}\n")

    result = run_recursive_rsi_pipeline(
        config=config,
        fixtures_dir=args.fixtures_dir,
        output_report_path=args.output,
    )

    md_summary = result.summary_markdown()
    print(md_summary)

    if args.markdown_out:
        Path(args.markdown_out).write_text(md_summary, encoding="utf-8")

    pm = result.pooled_metrics
    verdict = result.governor_verdict.get("decision")
    monotonic = pm.get("monotonic_progression", False)
    regressions = pm.get("total_regressions", 0)

    if verdict == "ACCEPT" and monotonic and regressions == 0:
        print("\n[SUCCESS] Statistical Governor ACCEPTED recursive policy π_2-context!")
        print(f"Monotonic progression verified: evals(π_2-ctx) < evals(π_2-op) < evals(π_1) < evals(π_0)")
        print(f"Evaluations saved: {pm['pi2_ctx_saved_percent']:.2f}% | p-value: {pm['p_val_0_vs_2ctx']:.4e} | Regressions: {regressions}")
        return 0
    else:
        print(f"\n[FAILURE] Criteria not fully satisfied: Governor={verdict}, Monotonic={monotonic}, Regressions={regressions}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
