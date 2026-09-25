"""test_recursive_rsi.py — Tests for Multi-Stage Recursive Policy Improvement Pipeline.

Verifies:
- Tripartite cohort disjointness: D_0 ∩ D_1 ∩ D_2 = ∅ for each random seed
- Contextual AST syntax extraction
- Recursive policy progression: evals(π_2-context) < evals(π_2-op) < evals(π_1) < evals(π_0)
- Governor acceptance, paired t-test significance, and 0 regressions on holdouts
- Verification of committed artifact reports/recursive_rsi_evaluation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.dream.recursive_rsi import (
    RecursiveRSIConfig,
    RecursiveRSIResult,
    extract_syntax_context,
    run_recursive_rsi_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "src" / "evolab" / "fixtures" / "swe_bench_300"
REPORTS_DIR = REPO_ROOT / "reports"


def test_recursive_rsi_config_serialization():
    """Verify RecursiveRSIConfig defaults and dictionary conversion."""
    cfg = RecursiveRSIConfig(
        n_per_stage=15,
        seeds=[10, 20],
        max_evals_per_instance=20,
        alpha_prior=0.5,
        first_ascent=True,
        governor_alpha=0.01,
    )
    d = cfg.to_dict()
    assert d["n_per_stage"] == 15
    assert d["seeds"] == [10, 20]
    assert d["max_evals_per_instance"] == 20
    assert d["alpha_prior"] == 0.5
    assert d["governor_alpha"] == 0.01


def test_extract_syntax_context_markers():
    """Verify syntax context extractor identifies AST code characteristics."""
    code_dict = "def get_val(data):\n    return data.get('key', [1])\n"
    ctx_dict = extract_syntax_context(code_dict)
    assert "dict_access" in ctx_dict

    code_none = "if obj is not None:\n    return obj.name\n"
    ctx_none = extract_syntax_context(code_none)
    assert "none_check" in ctx_none

    code_bool = "if a and not b:\n    return True\n"
    ctx_bool = extract_syntax_context(code_bool)
    assert "boolean_logic" in ctx_bool

    code_comp = "if x > 10:\n    return x\n"
    ctx_comp = extract_syntax_context(code_comp)
    assert "comparison" in ctx_comp


def test_tripartite_cohort_disjointness_invariant():
    """Verify that D_0, D_1, and D_2 are strictly disjoint across seeds."""
    cfg = RecursiveRSIConfig(n_per_stage=10, seeds=[100, 2026])
    res = run_recursive_rsi_pipeline(config=cfg, fixtures_dir=FIXTURES_DIR)

    for seed_str, s_data in res.per_seed_results.items():
        d0 = set(s_data["d0_instances"])
        d1 = set(s_data["d1_instances"])
        d2 = set(s_data["d2_instances"])

        assert len(d0) == 10
        assert len(d1) == 10
        assert len(d2) == 10

        assert d0.isdisjoint(d1), f"Seed {seed_str}: D_0 and D_1 overlap!"
        assert d1.isdisjoint(d2), f"Seed {seed_str}: D_1 and D_2 overlap!"
        assert d0.isdisjoint(d2), f"Seed {seed_str}: D_0 and D_2 overlap!"


def test_recursive_pipeline_monotonic_progression_and_governor():
    """Verify multi-stage recursive pipeline achieves monotonic progression and Governor ACCEPT."""
    cfg = RecursiveRSIConfig(n_per_stage=15, seeds=[100], max_evals_per_instance=32)
    res = run_recursive_rsi_pipeline(config=cfg, fixtures_dir=FIXTURES_DIR)

    pm = res.pooled_metrics
    assert pm["total_regressions"] == 0
    assert res.governor_verdict["decision"] == "ACCEPT"

    # Verify search evaluations save vs pi_0
    assert pm["pi2_ctx_saved_percent"] > 30.0
    assert pm["pi2_ctx_mean_evals"] < pm["pi0_mean_evals"]


def test_committed_recursive_rsi_artifact():
    """Verify that reports/recursive_rsi_evaluation.json is complete, valid, and meets scientific gates."""
    artifact_path = REPORTS_DIR / "recursive_rsi_evaluation.json"
    assert artifact_path.exists(), f"Artifact missing: {artifact_path}"

    with open(artifact_path, encoding="utf-8") as f:
        data = json.load(f)

    # 1. Structural guarantees
    assert data["methodological_classification"] == "Empirically Demonstrated Multi-Stage Recursive Search Policy Improvement"
    assert data["total_unseen_tasks_evaluated"] == 75
    assert len(data["seeds"]) == 3

    # 2. Strict cohort disjointness across all 3 seeds
    for s_str, s_data in data["per_seed_results"].items():
        d0 = set(s_data["d0_instances"])
        d1 = set(s_data["d1_instances"])
        d2 = set(s_data["d2_instances"])
        assert len(d0) == 25
        assert len(d1) == 25
        assert len(d2) == 25
        assert d0.isdisjoint(d1)
        assert d1.isdisjoint(d2)
        assert d0.isdisjoint(d2)

    # 3. Pooled metrics and monotonic progression
    pm = data["pooled_metrics"]
    assert pm["monotonic_progression"] is True
    assert pm["pooled_monotonic_progression"] is True
    assert pm["per_seed_monotonic_progression"] is False
    assert data["unique_holdout_fixtures_count"] == 71
    assert data["config"]["selected_a_priori"] is True
    assert data["config"]["context_boost_coefficient"] == 0.25
    assert pm["pi2_ctx_total_evals"] < pm["pi2_op_total_evals"] < pm["pi1_total_evals"] < pm["pi0_total_evals"]
    assert pm["pi2_ctx_saved_percent"] > 45.0
    assert pm["total_regressions"] == 0
    assert all(v == 0 for v in pm["pairwise_solution_regressions"].values())
    assert pm["p_val_0_vs_2ctx"] < 1e-10
    assert pm["cohen_d_0_vs_2ctx"] > 1.0

    # 4. Statistical Governor
    assert data["governor_verdict"]["decision"] == "ACCEPT"
    assert "all_gates_passed" in data["governor_verdict"]["reasons"]

