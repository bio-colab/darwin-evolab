"""test_live_rsi_generalization.py — Tests for Live Online Search Rollout Verification.

Verifies that meta-policies evolved via Dream-RSI achieve statistically significant,
zero-regression speedups on strictly unseen, disjoint real-world holdout tasks
(D_train ∩ D_test = ∅) in head-to-head live AST search execution.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.dream.live_rollout import (
    LiveInstanceRecord,
    LiveRSIConfig,
    LiveRSIResult,
    learn_operator_yields,
    run_live_rsi_rollout,
)
from evolab.swe_bench import SWEBenchAdapter


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "src" / "evolab" / "fixtures" / "swe_bench_300"
REPORTS_DIR = REPO_ROOT / "reports"


def test_live_rsi_config_serialization():
    """Verify LiveRSIConfig serialization and defaults."""
    cfg = LiveRSIConfig(
        n_train=20,
        n_test=20,
        seed=123,
        max_evals_per_instance=24,
        alpha_prior=0.5,
        first_ascent=True,
        governor_alpha=0.01,
        min_effect_size=0.4,
        tolerance_regressions=0,
    )
    d = cfg.to_dict()
    assert d["n_train"] == 20
    assert d["n_test"] == 20
    assert d["seed"] == 123
    assert d["alpha_prior"] == 0.5
    assert d["governor_alpha"] == 0.01

    restored = LiveRSIConfig.from_dict(d)
    assert restored.n_train == 20
    assert restored.n_test == 20
    assert restored.seed == 123
    assert restored.alpha_prior == 0.5


def test_disjoint_train_test_split_guarantee():
    """Verify that training and test instances are guaranteed strictly disjoint."""
    cfg = LiveRSIConfig(n_train=15, n_test=15, seed=42)
    res = run_live_rsi_rollout(config=cfg, fixtures_dir=FIXTURES_DIR)

    train_set = set(res.train_instances)
    test_set = set(res.test_instances)
    assert len(train_set) == 15
    assert len(test_set) == 15
    assert train_set.isdisjoint(test_set), "D_train and D_test must never overlap!"


def test_learn_operator_yields_prior():
    """Verify that learn_operator_yields produces valid probability distribution."""
    fixtures = sorted(FIXTURES_DIR.glob("*.json"))[:10]
    adapter = SWEBenchAdapter()
    cfg = LiveRSIConfig(alpha_prior=1.0)
    yields, default_yield = learn_operator_yields(fixtures, adapter, cfg)

    assert isinstance(yields, dict)
    assert len(yields) >= 1
    assert default_yield > 0.0
    for op, prob in yields.items():
        assert 0.0 < prob < 1.0


def test_live_rsi_rollout_governor_acceptance():
    """Verify that Live Online Rollout achieves Governor ACCEPT with p < 0.05 and 0 regressions."""
    cfg = LiveRSIConfig(n_train=25, n_test=25, seed=42)
    result = run_live_rsi_rollout(config=cfg, fixtures_dir=FIXTURES_DIR)

    # 1. Governor Decision Verification
    assert result.governor_verdict["decision"] == "ACCEPT"
    assert "all_gates_passed" in result.governor_verdict["reasons"]
    assert result.regressions_count == 0

    # 2. Statistical Invariants on Unseen Holdouts
    assert result.p_value < 0.05, f"Expected p < 0.05, got {result.p_value}"
    assert result.cohen_d >= 0.5, f"Expected Cohen's d >= 0.5, got {result.cohen_d}"
    assert result.mean_evaluations_saved_percent > 30.0
    assert result.evolved_mean_evals < result.baseline_mean_evals

    # 3. Preservation of Full Solve Rate
    assert result.baseline_solve_rate == 1.0
    assert result.evolved_solve_rate == 1.0
    assert result.baseline_total_evals > result.evolved_total_evals


def test_live_rsi_artifact_export_and_schema(tmp_path: Path):
    """Verify that run_live_rsi_rollout writes a complete, compliant JSON report."""
    out_file = tmp_path / "live_rsi_report.json"
    cfg = LiveRSIConfig(n_train=10, n_test=10, seed=99)

    res = run_live_rsi_rollout(
        config=cfg,
        fixtures_dir=FIXTURES_DIR,
        output_report_path=out_file,
    )

    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))

    assert "config" in data
    assert "train_instances" in data and len(data["train_instances"]) == 10
    assert "test_instances" in data and len(data["test_instances"]) == 10
    assert "learned_operator_yields" in data
    assert "summary_metrics" in data
    assert "governor_verdict" in data
    assert "records" in data and len(data["records"]) == 10

    # Markdown summary check
    md = res.summary_markdown()
    assert "Live Online Search Rollout" in md
    assert "Governor Verdict" in md


def test_live_rsi_multi_seed_aggregation():
    """Verify multi-seed aggregation computes pooled metrics across independent random splits."""
    cfg = LiveRSIConfig(
        n_train=10,
        n_test=10,
        seed=42,
        multi_seeds=[42, 123],
        evaluate_multi_seed=True,
    )
    res = run_live_rsi_rollout(config=cfg, fixtures_dir=FIXTURES_DIR)
    assert res.multi_seed_aggregate is not None
    assert res.multi_seed_aggregate["total_trials"] == 20
    pm = res.multi_seed_aggregate["pooled_metrics"]
    assert pm["mean_evaluations_saved_percent"] > 0.0
    assert pm["p_value"] < 0.05
    assert pm["total_regressions"] == 0
    assert "42" in res.multi_seed_aggregate["per_seed_results"]
    assert "123" in res.multi_seed_aggregate["per_seed_results"]

