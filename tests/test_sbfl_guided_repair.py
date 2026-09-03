"""Tests for SBFL fault-localization guided greedy search."""
from __future__ import annotations

from evolab.repair import RepairEdit, RepairGenome, greedy_repair
from evolab.suspicion import SuspicionMap


class DummyEvaluator:
    def __init__(self, target_line: int):
        self.target_line = target_line
        self.last_suspicion_map = SuspicionMap(
            line_scores={target_line: 0.95, 1: 0.05},
            total_passed=2,
            total_failed=1,
        )
        self.evaluated_edits: list[RepairEdit] = []

    def evaluate(self, target):
        from evolab.evaluators import FitnessResult
        if hasattr(target, "edits") and target.edits:
            last_edit = target.edits[-1]
            self.evaluated_edits.append(last_edit)
            if last_edit.lineno == self.target_line:
                return FitnessResult(score=100.0, passed_holdout=True)
            return FitnessResult(score=40.0, passed_holdout=False)
        return FitnessResult(score=20.0, passed_holdout=False)


def test_sbfl_prioritization_orders_suspicious_lines_first():
    # Multi-line code with candidates across lines 1, 2, 3
    code = (
        "def func(a, b, c):\n"
        "    flag = False\n"          # line 2 (bool_flip)
        "    return a < 10\n"         # line 3 (boundary_cmp)
    )
    sources = {"app.py": code}
    ev = DummyEvaluator(target_line=3)

    repaired, history, n_eval = greedy_repair(
        sources,
        "app.py",
        ev,
        prioritize_by_suspicion=True,
    )

    assert repaired.edits
    # The first evaluated edit should be on the suspicious line 3!
    first_tested = ev.evaluated_edits[0]
    assert first_tested.lineno == 3
    assert repaired.edits[0].lineno == 3


def test_sbfl_prioritization_disabled_preserves_legacy_order():
    code = (
        "def func(a, b, c):\n"
        "    flag = False\n"          # line 2
        "    return a < 10\n"         # line 3
    )
    sources = {"app.py": code}
    ev = DummyEvaluator(target_line=3)

    repaired, history, n_eval = greedy_repair(
        sources,
        "app.py",
        ev,
        prioritize_by_suspicion=False,
    )

    # Without suspicion prioritization, line 2 comes before line 3 in natural AST order
    first_tested = ev.evaluated_edits[0]
    assert first_tested.lineno == 2
