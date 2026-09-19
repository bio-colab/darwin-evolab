"""tests/test_compositional_repair.py — Tests for Multi-Hunk Compositional Repair Engine."""

from __future__ import annotations

import pytest

from evolab.compositional import (
    SuspicionCluster,
    cluster_edits_by_proximity,
    compositional_repair,
    generate_dual_hunk_candidates,
)
from evolab.repair import RepairEdit, RepairGenome, catalog_sources


def test_cluster_edits_by_proximity():
    """Verify that edits within max_line_distance are clustered together."""
    edits = [
        RepairEdit(kind="bool_flip", file="main.py", lineno=10, col_offset=4),
        RepairEdit(kind="compare_flip", file="main.py", lineno=12, col_offset=8),
        RepairEdit(kind="binop_flip", file="main.py", lineno=45, col_offset=4),
        RepairEdit(kind="bool_flip", file="other.py", lineno=10, col_offset=4),
    ]

    clusters = cluster_edits_by_proximity(edits, max_line_distance=5)
    # Expect 3 clusters: main.py (lines 10-12), main.py (line 45), other.py (line 10)
    assert len(clusters) == 3

    main_cluster = next(c for c in clusters if c.file == "main.py" and c.start_line == 10)
    assert len(main_cluster.edits) == 2
    assert main_cluster.span == 3


def test_generate_dual_hunk_candidates_prioritizes_lead_edits():
    """Verify that dual-hunk candidate generator anchors on lead edits."""
    e1 = RepairEdit(kind="bool_flip", file="app.py", lineno=5, col_offset=2)
    e2 = RepairEdit(kind="compare_flip", file="app.py", lineno=6, col_offset=4)
    e3 = RepairEdit(kind="binop_flip", file="app.py", lineno=20, col_offset=0)
    catalog = [e1, e2, e3]

    pairs = generate_dual_hunk_candidates(catalog, lead_edits=[e3], max_pairs=10)
    assert len(pairs) >= 2
    # First generated pairs must include lead edit e3
    assert any(e3 in p for p in pairs)


class MockDualHunkEvaluator:
    """Evaluator that simulates a bug plateau:
    - Baseline: 40.0%
    - Single edit 1 alone: 60.0% (fails holdout)
    - Single edit 2 alone: 40.0% (fails holdout)
    - Dual edit {1, 2} together: 100.0% (passes holdout!)
    """
    def __init__(self, target_line_1: int, target_line_2: int):
        self.t1 = target_line_1
        self.t2 = target_line_2
        self.last_suspicion_map = None

    def evaluate(self, genome: RepairGenome):
        class Result:
            def __init__(self, score: float, holdout: bool):
                self.score = score
                self.passed_holdout = holdout

        has_1 = any(e.lineno == self.t1 for e in genome.edits)
        has_2 = any(e.lineno == self.t2 for e in genome.edits)

        if has_1 and has_2:
            return Result(100.0, True)
        elif has_1:
            return Result(60.0, False)
        elif has_2:
            return Result(40.0, False)
        return Result(40.0, False)


def test_compositional_repair_solves_dual_hunk_plateau():
    """Verify that compositional_repair overcomes single-edit plateau via dual-hunk exploration."""
    code = (
        "def process(x, y):\n"
        "    flag = False\n"     # line 2 (bool_flip)
        "    if x < 10:\n"       # line 3 (compare_flip)
        "        return x + y\n"
        "    return 0\n"
    )
    sources = {"app.py": code}
    evaluator = MockDualHunkEvaluator(target_line_1=2, target_line_2=3)

    winning_genome, history, evals = compositional_repair(
        sources=sources,
        target_file="app.py",
        evaluator=evaluator,
        max_evals=32,
        lookahead_depth=2,
    )

    # Must find the 100% solution with 2 edits
    res = evaluator.evaluate(winning_genome)
    assert res.score == 100.0
    assert res.passed_holdout is True
    assert len(winning_genome.edits) == 2
    assert any(h["stage"] in ("dual_hunk_breakthrough", "dual_hunk_ascent") for h in history)
