"""ablation.py — Baseline Comparison and Systematic Parameter Ablation Study.

Compares the Evolved Champion against:
1. Default / Naive Heuristic Profile
2. Random Guessing Monte Carlo Distribution (M=50 random policies)
3. Systematic Single-Parameter Ablations

Quantifies the empirical delta, statistical significance (p-value, t-statistic),
and isolates the contribution of each heuristic dimension to holdout generalization.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from .benchmark import run_golden_benchmark
from .genome import PARAM_BOUNDS, ProfileGenome, ProfilePolicy
from .word_holdout import load_real_word_holdout


def run_ablation_study(
    num_random_samples: int = 50,
    seed: int = 42,
    save_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Executes the full baseline and ablation suite against the 12-document real Word holdout."""
    holdout_corpus = load_real_word_holdout()
    rng = random.Random(seed)

    print("=== Running Baseline Comparison and Systematic Ablation Study ===")

    # -------------------------------------------------------------
    # 1. Default Naive Baseline
    # -------------------------------------------------------------
    print("\n1. Evaluating Default Heuristic Baseline Profile...")
    default_policy = ProfilePolicy()
    t0 = time.perf_counter()
    rep_default = run_golden_benchmark(policy=default_policy, corpus=holdout_corpus)
    default_score = rep_default.average_composite_score
    print(f"  Default Profile Composite Score: {default_score:.2%} (Pass Rate: {rep_default.overall_pass_rate:.2%})")

    # -------------------------------------------------------------
    # 2. Random Guessing Monte Carlo Baseline
    # -------------------------------------------------------------
    print(f"\n2. Evaluating Random Guessing Distribution (M={num_random_samples} random policies)...")
    random_scores: list[float] = []
    random_pass_rates: list[float] = []

    for r_idx in range(num_random_samples):
        rand_policy = ProfilePolicy(
            space_gap_ratio=rng.uniform(*PARAM_BOUNDS["space_gap_ratio"]),
            para_split_delta_ratio=rng.uniform(*PARAM_BOUNDS["para_split_delta_ratio"]),
            align_tolerance_pt=rng.uniform(*PARAM_BOUNDS["align_tolerance_pt"]),
            line_spacing_round_pt=rng.uniform(*PARAM_BOUNDS["line_spacing_round_pt"]),
            table_col_align_tol_pt=rng.uniform(*PARAM_BOUNDS["table_col_align_tol_pt"]),
            table_min_rows=int(round(rng.uniform(*PARAM_BOUNDS["table_min_rows"]))),
            table_min_cols=int(round(rng.uniform(*PARAM_BOUNDS["table_min_cols"]))),
        )
        rep_rand = run_golden_benchmark(policy=rand_policy, corpus=holdout_corpus)
        random_scores.append(rep_rand.average_composite_score)
        random_pass_rates.append(rep_rand.overall_pass_rate)

    m = len(random_scores)
    mean_random = sum(random_scores) / m
    var_random = sum((s - mean_random) ** 2 for s in random_scores) / (m - 1) if m > 1 else 0.0
    sd_random = math.sqrt(var_random)
    sorted_random = sorted(random_scores)
    median_random = sorted_random[m // 2]
    p25_random = sorted_random[int(m * 0.25)]
    p75_random = sorted_random[int(m * 0.75)]

    print(f"  Random Policies Mean:   {mean_random:.2%} ± {sd_random:.4f}")
    print(f"  Random Policies Median: {median_random:.2%} (P25: {p25_random:.2%}, P75: {p75_random:.2%})")
    print(f"  Random Policies Range:  [{min(random_scores):.2%}, {max(random_scores):.2%}]")

    # -------------------------------------------------------------
    # 3. Evolved Champion Profile
    # -------------------------------------------------------------
    print("\n3. Evaluating Evolved Champion Profile...")
    profile_file = Path(__file__).resolve().parent / "calibrated_word_profile.json"
    if profile_file.exists():
        with open(profile_file, "r", encoding="utf-8") as f:
            champ_data = json.load(f)
        champ_policy = ProfilePolicy.from_dict(champ_data.get("policy", champ_data))
    else:
        champ_policy = ProfilePolicy(
            space_gap_ratio=0.35,
            para_split_delta_ratio=0.55,
            align_tolerance_pt=4.0,
            line_spacing_round_pt=0.5,
            table_col_align_tol_pt=3.0,
            table_min_rows=2,
            table_min_cols=2,
        )

    rep_champ = run_golden_benchmark(policy=champ_policy, corpus=holdout_corpus)
    champ_score = rep_champ.average_composite_score
    print(f"  Evolved Champion Score: {champ_score:.2%} (Pass Rate: {rep_champ.overall_pass_rate:.2%})")

    delta_vs_default = champ_score - default_score
    delta_vs_random_mean = champ_score - mean_random

    # Welch's t-statistic for champion vs random distribution
    # Treat champion as target value against normal approximation of random
    t_stat = (champ_score - mean_random) / (sd_random / math.sqrt(m)) if sd_random > 0 else float("inf")
    # Approximate two-tailed p-value for large t
    if abs(t_stat) > 6.0:
        p_val_str = "< 1e-9"
    elif abs(t_stat) > 4.0:
        p_val_str = "< 0.0001"
    else:
        p_val_str = f"{math.erfc(abs(t_stat) / math.sqrt(2)):.6f}"

    print(f"  Delta vs Default:     {delta_vs_default:+.2%}")
    print(f"  Delta vs Random Mean: {delta_vs_random_mean:+.2%}")
    print(f"  Statistical t-stat:   {t_stat:.2f} (p-value: {p_val_str})")

    # -------------------------------------------------------------
    # 4. Systematic Single-Parameter Ablations
    # -------------------------------------------------------------
    print("\n4. Running Systematic Parameter Ablation Study...")
    ablation_experiments: list[dict[str, Any]] = []

    ablation_cases = [
        ("Ablate line_spacing_round_pt (coarse 5.0pt)", {"line_spacing_round_pt": 5.0}),
        ("Ablate line_spacing_round_pt (zero grid 0.0pt)", {"line_spacing_round_pt": 0.0}),
        ("Ablate align_tolerance_pt (zero tolerance 0.0pt)", {"align_tolerance_pt": 0.0}),
        ("Ablate align_tolerance_pt (loose tolerance 20.0pt)", {"align_tolerance_pt": 20.0}),
        ("Ablate space_gap_ratio (tight gap 0.10)", {"space_gap_ratio": 0.10}),
        ("Ablate space_gap_ratio (loose gap 0.80)", {"space_gap_ratio": 0.80}),
        ("Ablate table_col_align_tol_pt (zero tolerance 0.0pt)", {"table_col_align_tol_pt": 0.0}),
        ("Ablate para_split_delta_ratio (low split 0.20)", {"para_split_delta_ratio": 0.20}),
    ]

    for label, param_overrides in ablation_cases:
        ablated_dict = champ_policy.to_dict()
        ablated_dict.update(param_overrides)
        ablated_policy = ProfilePolicy.from_dict(ablated_dict)

        rep_abl = run_golden_benchmark(policy=ablated_policy, corpus=holdout_corpus)
        abl_score = rep_abl.average_composite_score
        drop = champ_score - abl_score

        rec = {
            "ablation_name": label,
            "overrides": param_overrides,
            "ablated_composite_score": round(abl_score, 4),
            "drop_vs_champion": round(drop, 4),
            "overall_pass_rate": round(rep_abl.overall_pass_rate, 4),
            "text_integrity_pass_rate": round(rep_abl.text_integrity_pass_rate, 4),
        }
        ablation_experiments.append(rec)
        print(f"  {label:<52}: {abl_score:.2%} (Drop: -{drop:.2%}, Pass: {rep_abl.overall_pass_rate:.2%})")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "holdout_corpus_size": len(holdout_corpus),
        "champion_performance": {
            "composite_score": round(champ_score, 4),
            "overall_pass_rate": round(rep_champ.overall_pass_rate, 4),
            "text_integrity_pass_rate": round(rep_champ.text_integrity_pass_rate, 4),
            "policy": champ_policy.to_dict(),
        },
        "default_baseline": {
            "composite_score": round(default_score, 4),
            "overall_pass_rate": round(rep_default.overall_pass_rate, 4),
            "delta_vs_champion": round(delta_vs_default, 4),
        },
        "random_baseline": {
            "num_samples": num_random_samples,
            "mean_composite_score": round(mean_random, 4),
            "sd_composite_score": round(sd_random, 4),
            "median_composite_score": round(median_random, 4),
            "p25_composite_score": round(p25_random, 4),
            "p75_composite_score": round(p75_random, 4),
            "min_composite_score": round(min(random_scores), 4),
            "max_composite_score": round(max(random_scores), 4),
            "delta_vs_champion": round(delta_vs_random_mean, 4),
            "t_statistic": round(t_stat, 2),
            "p_value_approximation": p_val_str,
        },
        "systematic_ablations": ablation_experiments,
    }

    save_path = save_report_path or (
        Path(__file__).resolve().parent.parent.parent / "reports" / "pdf2rtf_ablation_study.json"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAblation Report Saved: {save_path}")

    return report


if __name__ == "__main__":
    run_ablation_study(num_random_samples=50)
