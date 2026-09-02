"""Tests for ASTGenome, structural distance metrics, and AST-aware mutations."""
from __future__ import annotations

import random
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.ast_genome import (
    ASTGenome,
    ast_distance,
    mutate_ast,
    get_ast_metrics,
    ConstantTransformer,
    BinOpTransformer,
    CompareTransformer,
)
from evolab.evaluators import FunctionTestEvaluator, SandboxFunctionTestEvaluator
from evolab.genome import Individual


def test_ast_genome_basics():
    code = "def greet(name):\n    return f'Hello, {name}!'\n"
    genome = ASTGenome.from_code(code)

    assert "def greet(name):" in genome.to_code()
    assert len(genome.fingerprint()) == 16

    clone = genome.clone()
    assert clone.fingerprint() == genome.fingerprint()
    assert clone.distance_to(genome) == 0.0

    serialized = genome.serialize()
    assert serialized["type"] == "ASTGenome"
    assert "fingerprint" in serialized


def test_ast_metrics_and_describe():
    code = (
        "def compute(a, b):\n"
        "    x = a + b\n"
        "    if x > 10:\n"
        "        return x * 2\n"
        "    return x\n"
    )
    genome = ASTGenome.from_code(code)
    desc = genome.describe()

    assert desc["func_count"] == 1
    assert desc["stmt_count"] >= 3
    assert desc["node_count"] > 10
    assert desc["max_depth"] >= 3


def test_ast_distance_metric():
    code1 = "def f(x):\n    return x + 1\n"
    code2 = "def f(x):\n    return x + 1\n"
    code3 = "def f(x):\n    return x * 5\n"
    code4 = "def g(a, b, c):\n    for i in range(a):\n        print(b, c)\n"

    g1 = ASTGenome.from_code(code1)
    g2 = ASTGenome.from_code(code2)
    g3 = ASTGenome.from_code(code3)
    g4 = ASTGenome.from_code(code4)

    assert g1.distance_to(g2) == 0.0
    dist_minor = g1.distance_to(g3)
    dist_major = g1.distance_to(g4)

    assert 0.0 < dist_minor < 1.0
    assert 0.0 < dist_major <= 1.0
    assert dist_major > dist_minor


def test_ast_binop_and_compare_mutations():
    code = "def check(x, y):\n    return (x + y) > 10\n"
    genome = ASTGenome.from_code(code)
    rng = random.Random(42)

    mutated = mutate_ast(genome, rng=rng)
    assert isinstance(mutated, ASTGenome)
    # The code must remain syntactically valid python
    compile(mutated.to_code(), "<test>", "exec")


def test_ast_evaluator_integration():
    code_buggy = "def multiply_add(a, b, c):\n    return (a - b) + c\n" # Goal: (a * b) + c
    code_correct = "def multiply_add(a, b, c):\n    return (a * b) + c\n"

    test_cases = [
        ((2, 3, 4), 10),
        ((0, 5, 2), 2),
        ((3, 3, 1), 10),
    ]

    evaluator = FunctionTestEvaluator(
        base_sources={"math.py": code_buggy},
        target_file="math.py",
        func_name="multiply_add",
        test_cases=test_cases,
    )

    g_buggy = ASTGenome.from_code(code_buggy)
    g_correct = ASTGenome.from_code(code_correct)

    res_buggy = evaluator.evaluate(g_buggy)
    res_correct = evaluator.evaluate(g_correct)

    assert res_buggy.score < res_correct.score
    assert res_correct.score == 100.0


def test_live_ast_evolution_loop():
    """Repairs buggy AST tree using AST-level mutation operators."""
    buggy_code = "def calc(x):\n    return x - 10\n" # Goal: return x + 10
    test_cases = [
        ((0,), 10),
        ((5,), 15),
        ((-10,), 0),
    ]

    evaluator = FunctionTestEvaluator(
        base_sources={"calc.py": buggy_code},
        target_file="calc.py",
        func_name="calc",
        test_cases=test_cases,
    )

    rng = random.Random(123)
    pop = [Individual(genome=ASTGenome.from_code(buggy_code), species="ast_spec") for _ in range(8)]

    solved = False
    for gen in range(25):
        for ind in pop:
            ind.fitness = evaluator.evaluate(ind.genome).score
            if ind.fitness >= 100.0:
                solved = True
                break
        if solved:
            break

        pop.sort(key=lambda x: x.fitness, reverse=True)
        elites = [ind.genome.clone() for ind in pop[:2]]
        new_pop = [Individual(genome=g, species="ast_spec") for g in elites]
        while len(new_pop) < 8:
            parent = rng.choice(elites)
            child_g = mutate_ast(parent, rng=rng)
            new_pop.append(Individual(genome=child_g, species="ast_spec"))
        pop = new_pop

    assert solved is True
