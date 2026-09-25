"""verify_live_rsi.py — Empirical Live Search Rollout Verification for Dream-RSI.

Runs head-to-head live search rollouts comparing baseline policy π_0 vs evolved
policy π* on strictly disjoint, unseen SWE-bench instances (D_train ∩ D_test = ∅).

Generates reports/live_rsi_generalization.json and verifies:
- evaluations consumed to verified resolution (FAIL_TO_PASS & PASS_TO_PASS clean)
- paired Student's t-test p < 0.05
- Cohen's d effect size > 0.5
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

from evolab.dream.live_rollout import LiveRSIConfig, run_live_rsi_rollout

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = REPO_ROOT / "src" / "evolab" / "fixtures" / "swe_bench_300"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "live_rsi_generalization.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Search Rollout Verification for Dream-RSI")
    parser.add_argument("--n-train", type=int, default=25, help="Number of instances for D_train (default: 25)")
    parser.add_argument("--n-test", type=int, default=25, help="Number of instances for unseen D_test (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split (default: 42)")
    parser.add_argument("--fixtures-dir", type=str, default=str(DEFAULT_FIXTURES), help="Fixtures directory")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--governor-alpha", type=float, default=0.05, help="Governor alpha significance threshold")
    parser.add_argument("--markdown-out", type=str, default=None, help="Optional path to write summary markdown")
    args = parser.parse_args()

    config = LiveRSIConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        governor_alpha=args.governor_alpha,
    )

    print(f"=== Starting Dream-RSI Live Online Search Rollout ===")
    print(f"Dataset:       {args.fixtures_dir}")
    print(f"D_train size:  {args.n_train} instances")
    print(f"D_test size:   {args.n_test} strictly unseen holdout instances")
    print(f"Random seed:   {args.seed}")
    print(f"Target report: {args.output}\n")

    result = run_live_rsi_rollout(
        config=config,
        fixtures_dir=args.fixtures_dir,
        output_report_path=args.output,
    )

    md_summary = result.summary_markdown()
    print(md_summary)

    if args.markdown_out:
        Path(args.markdown_out).write_text(md_summary, encoding="utf-8")

    verdict = result.governor_verdict.get("decision")
    if verdict == "ACCEPT":
        print(f"\n[SUCCESS] Statistical Governor ACCEPTED live evolved meta-policy π*!")
        print(f"Evaluations saved: {result.mean_evaluations_saved_percent:.2f}% | p-value: {result.p_value:.6f} | Regressions: {result.regressions_count}")
        return 0
    else:
        print(f"\n[FAILURE] Statistical Governor REJECTED meta-policy π*: {result.governor_verdict.get('reasons')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
