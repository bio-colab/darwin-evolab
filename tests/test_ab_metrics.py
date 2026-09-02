"""Phase 3 A/B harness: metric helpers and the pre-registered decision rule."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ab_memory_value.py"
_spec = importlib.util.spec_from_file_location("ab_memory_value", _SCRIPT)
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def _pair(seed, ctrl_success, ctrl_total, mem_success, mem_total, ctrl_eval=None, mem_eval=None):
    return {
        "seed": seed,
        "control": {
            "evals_total": ctrl_total,
            "first_success_eval": ctrl_eval if ctrl_success else None,
            "first_success_score": 100.0 if ctrl_success else None,
        },
        "memory": {
            "evals_total": mem_total,
            "first_success_eval": mem_eval if mem_success else None,
            "first_success_score": 100.0 if mem_success else None,
        },
    }


# ---------- sign test ----------

def test_sign_p_edges():
    assert ab.sign_pvalue(0, 0) is None
    assert ab.sign_pvalue(3, 0) == 0.25  # 2 * (1/8)
    assert ab.sign_pvalue(2, 2) == 1.0
    assert ab.sign_pvalue(1, 0) == 1.0
    assert 0.0 < ab.sign_pvalue(4, 0) < 0.25


# ---------- arm cost & censoring ----------

def test_arm_cost_censors_at_budget():
    assert ab.arm_cost({"evals_total": 90, "first_success_eval": 12}) == 12
    assert ab.arm_cost({"evals_total": 90, "first_success_eval": None}) == 90


# ---------- summarize_pairs ----------

def test_summarize_pairs_basic_gain():
    pairs = [
        _pair(1, True, 90, True, 90, ctrl_eval=10, mem_eval=5),
        _pair(2, True, 90, True, 90, ctrl_eval=20, mem_eval=10),
    ]
    s = ab.summarize_pairs(pairs)
    assert s["N_mean_evals"] == 15.0
    assert s["M_mean_evals"] == 7.5
    assert s["search_efficiency_gain"] == 0.5
    assert s["successes"] == {"control": 2, "memory": 2}
    assert s["paired"] == {"memory_wins": 2, "control_wins": 0, "ties": 0}
    assert s["sign_test_p"] == 0.5


def test_summarize_pairs_censored_runs_count_at_budget():
    pairs = [
        _pair(1, True, 90, False, 90, ctrl_eval=10),          # memory failed -> cost 90
        _pair(2, True, 90, True, 90, ctrl_eval=30, mem_eval=30),  # tie
    ]
    s = ab.summarize_pairs(pairs)
    assert s["N_mean_evals"] == 20.0
    assert s["M_mean_evals"] == 60.0
    assert s["search_efficiency_gain"] == -2.0  # honest negative result
    assert s["successes"] == {"control": 2, "memory": 1}
    assert s["paired"]["control_wins"] == 1 and s["paired"]["ties"] == 1


def test_summarize_pairs_all_ties_gain_zero():
    pairs = [_pair(i, True, 50, True, 50, ctrl_eval=7, mem_eval=7) for i in range(3)]
    s = ab.summarize_pairs(pairs)
    assert s["search_efficiency_gain"] == 0.0
    assert s["paired"] == {"memory_wins": 0, "control_wins": 0, "ties": 3}
    assert s["sign_test_p"] is None


# ---------- the pre-registered decision rule ----------

def test_keeps_default_on_rule():
    good = {"search_efficiency_gain": 0.3, "paired": {"memory_wins": 6, "control_wins": 2, "ties": 2}}
    negative = {"search_efficiency_gain": -0.1, "paired": {"memory_wins": 2, "control_wins": 6, "ties": 2}}
    unpopular = {"search_efficiency_gain": 0.3, "paired": {"memory_wins": 1, "control_wins": 5, "ties": 4}}
    no_gain = {"search_efficiency_gain": None, "paired": {"memory_wins": 0, "control_wins": 0, "ties": 5}}
    zero_gain = {"search_efficiency_gain": 0.0, "paired": {"memory_wins": 0, "control_wins": 0, "ties": 10}}
    assert ab.keeps_default_on(good) is True
    assert ab.keeps_default_on(negative) is False
    assert ab.keeps_default_on(unpopular) is False  # gain alone is not enough
    assert ab.keeps_default_on(no_gain) is False
    assert ab.keeps_default_on(zero_gain) is False


def test_mean_empty():
    assert ab.mean([]) == 0.0
