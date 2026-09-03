"""Tests for universal AST repair patterns."""
from __future__ import annotations

import pytest
from evolab.repair import (
    RepairEdit,
    RepairGenome,
    apply_edits,
    catalog_edits,
    greedy_repair,
)
from evolab.evaluators import FunctionTestEvaluator


def test_logical_flip_catalog_and_apply():
    code = "def check(a, b):\n    return a and b\n"
    edits = catalog_edits(code, "test.py")
    flip = [e for e in edits if e.kind == "logical_flip"]
    assert len(flip) == 1
    repaired = apply_edits(code, flip)
    assert "return a or b" in repaired


def test_boundary_cmp_catalog_and_apply():
    code = "def is_valid(x):\n    return x < 10\n"
    edits = catalog_edits(code, "test.py")
    b_edits = [e for e in edits if e.kind == "boundary_cmp"]
    assert len(b_edits) == 1
    repaired = apply_edits(code, b_edits)
    assert "return x <= 10" in repaired


def test_none_check_flip_catalog_and_apply():
    code = "def get_val(x):\n    if x is None:\n        return 'default'\n    return x\n"
    edits = catalog_edits(code, "test.py")
    n_edits = [e for e in edits if e.kind == "none_check_flip"]
    assert len(n_edits) == 1
    repaired = apply_edits(code, n_edits)
    assert "if x is not None:" in repaired


def test_binop_flip_catalog_and_apply():
    code = "def calc(a, b):\n    return a + b\n"
    edits = catalog_edits(code, "test.py")
    bin_edits = [e for e in edits if e.kind == "binop_flip"]
    assert len(bin_edits) == 1
    repaired = apply_edits(code, bin_edits)
    assert "return a - b" in repaired


def test_off_by_one_catalog_and_apply():
    code = "def count(n):\n    return n + 1\n"
    edits = catalog_edits(code, "test.py")
    inc_edits = [e for e in edits if e.kind == "off_by_one_inc"]
    dec_edits = [e for e in edits if e.kind == "off_by_one_dec"]
    assert len(inc_edits) == 1
    assert len(dec_edits) == 1

    rep_inc = apply_edits(code, inc_edits)
    assert "return n + 2" in rep_inc

    rep_dec = apply_edits(code, dec_edits)
    assert "return n + 0" in rep_dec


def test_greedy_repair_fixes_off_by_one_bug():
    buggy = "def get_len(lst):\n    return len(lst) + 1\n"
    ev = FunctionTestEvaluator(
        base_sources={"mod.py": buggy},
        target_file="mod.py",
        func_name="get_len",
        test_cases=[
            (([1, 2, 3],), 3),
            (([],), 0),
        ],
        holdout_cases=[(([10, 20],), 2)],
    )
    repaired_genome, history, n_eval = greedy_repair(
        {"mod.py": buggy},
        "mod.py",
        ev,
    )
    res = ev.evaluate(repaired_genome)
    assert res.score == 100.0
    assert res.passed_holdout is True
    assert "return len(lst) + 0" in repaired_genome.to_code() or "len(lst)" in repaired_genome.to_code()
