"""Tests for code evolution: EvolabGenome, PatchGenome, Evaluator contract, and live code repair."""
from __future__ import annotations

import random
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.genome import EvolabGenome, FloatGenome, Individual
from evolab.patch import (
    Hunk,
    PatchGenome,
    PatchApplyError,
    apply_patch,
    create_patch_from_diff,
    patch_distance,
    mutate_patch,
)
from evolab.evaluators import (
    Evaluator,
    FitnessResult,
    NumericEvaluator,
    CompileCheckEvaluator,
    FunctionTestEvaluator,
)


def test_float_genome_contract():
    g1 = FloatGenome(values=[1.0, 2.0, 3.0, 4.0])
    g2 = g1.clone()
    assert g1.fingerprint() == g2.fingerprint()
    assert g1.distance_to(g2) == 0.0

    g3 = FloatGenome(values=[6.0, 7.0, 8.0, 9.0])
    d = g1.distance_to(g3)
    assert d > 0.0
    assert g3.distance_to(g1) == pytest.approx(d)

    desc = g1.describe()
    assert "mean" in desc
    assert "std" in desc
    assert desc["mean"] == 2.5


def test_hunk_serialization():
    h = Hunk(
        file_path="src/utils.py",
        start_line=10,
        num_lines=2,
        old_text="x = 1\ny = 2\n",
        new_text="x = 10\ny = 20\n",
    )
    s = h.serialize()
    h2 = Hunk.from_dict(s)
    assert h == h2


def test_patch_genome_diff_and_apply():
    old_code = "def calc(x):\n    return x - 5\n"
    new_code = "def calc(x):\n    return x + 5\n"

    sources = {"calc.py": old_code}
    patch = create_patch_from_diff("calc.py", old_code, new_code)

    assert len(patch.hunks) == 1
    applied = patch.apply_to(sources)
    assert applied["calc.py"] == new_code


def test_patch_genome_multi_hunk():
    old_code = "a = 1\nb = 2\nc = 3\nd = 4\n"
    new_code = "a = 10\nb = 2\nc = 30\nd = 4\n"

    sources = {"file.py": old_code}
    patch = create_patch_from_diff("file.py", old_code, new_code)
    applied = patch.apply_to(sources)
    assert applied["file.py"] == new_code


def test_patch_distance_properties():
    p1 = PatchGenome(
        hunks=[
            Hunk("a.py", 0, 1, "x = 1\n", "x = 2\n"),
        ]
    )
    p2 = p1.clone()
    assert p1.fingerprint() == p2.fingerprint()
    assert patch_distance(p1, p2) == 0.0

    p3 = PatchGenome(
        hunks=[
            Hunk("b.py", 0, 1, "y = 1\n", "y = 9\n"),
        ]
    )
    d = patch_distance(p1, p3)
    assert 0.0 < d <= 1.0
    assert patch_distance(p3, p1) == pytest.approx(d)


def test_compile_check_evaluator():
    valid_patch = PatchGenome(
        hunks=[
            Hunk("math_ops.py", 0, 0, "", "def add(a, b):\n    return a + b\n"),
        ]
    )
    invalid_patch = PatchGenome(
        hunks=[
            Hunk("math_ops.py", 0, 0, "", "def add(a, b)\n    return a +\n"),
        ]
    )

    base = {"math_ops.py": ""}
    evaluator = CompileCheckEvaluator(base)
    assert evaluator.deterministic is True

    res_valid = evaluator.evaluate(valid_patch)
    assert res_valid.score == 100.0
    assert res_valid.sub_scores["compiled"] == 100.0

    res_invalid = evaluator.evaluate(invalid_patch)
    assert res_invalid.score == 0.0
    assert "SyntaxError" in str(res_invalid.artifacts["errors"])


def test_function_test_evaluator_with_holdout():
    base_sources = {
        "solution.py": "def multiply(a, b):\n    return a + b\n" # Buggy implementation
    }

    test_cases = [
        ((2, 3), 6),
        ((0, 5), 0),
        ((4, 5), 20),
    ]
    holdout_cases = [
        ((10, 10), 100),
    ]

    evaluator = FunctionTestEvaluator(
        base_sources=base_sources,
        target_file="solution.py",
        func_name="multiply",
        test_cases=test_cases,
        holdout_cases=holdout_cases,
    )

    # Evaluate buggy initial patch (empty patch)
    empty_patch = PatchGenome()
    res_initial = evaluator.evaluate(empty_patch)
    # 20 points for compile, 0 tests pass (2+3=5!=6, 0+5=5!=0, 4+5=9!=20)
    assert res_initial.score == 20.0
    assert res_initial.passed_holdout is False

    # Create repair patch
    repair_patch = PatchGenome(
        hunks=[
            Hunk("solution.py", 1, 1, "    return a + b\n", "    return a * b\n")
        ]
    )
    res_repair = evaluator.evaluate(repair_patch)
    assert res_repair.score == 100.0
    assert res_repair.passed_holdout is True
    assert res_repair.artifacts["passed_tests"] == 3


def test_lethal_mutation_guard():
    base_sources = {
        "algo.py": "def compute(x):\n    return x + 1\n"
    }
    patch = PatchGenome()
    rng = random.Random(42)

    for _ in range(20):
        mutated = mutate_patch(patch, base_sources, rng)
        applied = mutated.apply_to(base_sources)
        # Verify that mutated code is always syntactically valid (non-lethal)
        compile(applied["algo.py"], "algo.py", "exec")


def test_live_code_evolution_loop():
    """Demonstrates live evolutionary code repair on a buggy function."""
    base_sources = {
        "target.py": "def solve(x):\n    return x - 2\n"  # Buggy: goal is return x + 2
    }
    test_cases = [
        ((0,), 2),
        ((5,), 7),
        ((10,), 12),
        ((-2,), 0),
    ]
    evaluator = FunctionTestEvaluator(
        base_sources=base_sources,
        target_file="target.py",
        func_name="solve",
        test_cases=test_cases,
    )

    # Initial population of PatchGenomes
    rng = random.Random(42)
    pop = [
        Individual(genome=PatchGenome(), species="spec_patch")
        for _ in range(10)
    ]

    best_fitness = 0.0
    solved = False

    for gen in range(25):
        for ind in pop:
            res = evaluator.evaluate(ind.genome)
            ind.fitness = res.score
            if ind.fitness == 100.0:
                solved = True
                best_fitness = 100.0
                break
        if solved:
            break

        # Sort by fitness
        pop.sort(key=lambda x: x.fitness, reverse=True)
        elites = [ind.genome.clone() for ind in pop[:2]]

        new_pop = [Individual(genome=g, species="spec_patch") for g in elites]
        while len(new_pop) < 10:
            parent = rng.choice(elites)
            mutated_g = mutate_patch(parent, base_sources, rng)
            new_pop.append(Individual(genome=mutated_g, species="spec_patch"))
        pop = new_pop

    assert solved is True
    assert best_fitness == 100.0


def test_patch_distance_exact_symmetry():
    """Verify that patch_distance satisfies the strict mathematical symmetry metric axiom."""
    p1 = PatchGenome(hunks=[
        Hunk(file_path="a.py", start_line=3, num_lines=2, old_text="line3\nline4\n", new_text="line3_mod\nline4_mod\n"),
        Hunk(file_path="a.py", start_line=0, num_lines=1, old_text="line0\n", new_text="line0_mod\n"),
        Hunk(file_path="b.py", start_line=1, num_lines=1, old_text="    pass\n", new_text="    return 42\n"),
    ])
    p2 = PatchGenome(hunks=[
        Hunk(file_path="c.py", start_line=0, num_lines=0, old_text="", new_text="# new file\n")
    ])

    d12 = patch_distance(p1, p2)
    d21 = patch_distance(p2, p1)
    assert d12 == d21, f"Metric asymmetry detected: d(p1,p2)={d12} != d(p2,p1)={d21}"
    assert 0.0 <= d12 <= 1.0


def test_patch_genome_fast_fingerprint_determinism():
    """Verify that fingerprint is fast, deterministic, and identical across clones and orderings."""
    p1 = PatchGenome(hunks=[
        Hunk(file_path="x.py", start_line=1, num_lines=1, old_text="a = 1\n", new_text="a = 2\n"),
        Hunk(file_path="y.py", start_line=5, num_lines=2, old_text="b = 3\nc = 4\n", new_text="b = 30\n"),
    ])
    p2 = PatchGenome(hunks=[
        Hunk(file_path="y.py", start_line=5, num_lines=2, old_text="b = 3\nc = 4\n", new_text="b = 30\n"),
        Hunk(file_path="x.py", start_line=1, num_lines=1, old_text="a = 1\n", new_text="a = 2\n"),
    ])
    assert p1.fingerprint() == p2.fingerprint()
    assert len(p1.fingerprint()) == 16


