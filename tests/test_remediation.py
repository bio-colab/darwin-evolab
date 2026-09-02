"""Tests for Undeclared Gaps Remediation: AST Crossover, Multi-Hunk Mutations, Code Causal Memory, and Code MAP-Elites."""
from __future__ import annotations

import random
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.ast_genome import ASTGenome, crossover_ast
from evolab.patch import PatchGenome, Hunk, mutate_multi_hunk
from evolab.causal import (
    CausalModel,
    TrapSignatureLibrary,
    discretize_context,
    discretize_code_context,
)
from evolab.engine import _desc_mean, _desc_std, EvolutionEngine
from evolab.genome import Individual


def test_ast_subtree_crossover():
    code_a = "def func_a(x, y):\n    a = x + 10\n    return a * 2\n"
    code_b = "def func_b(x, y):\n    b = y - 50\n    return b // 3\n"

    parent_a = ASTGenome.from_code(code_a)
    parent_b = ASTGenome.from_code(code_b)

    rng = random.Random(42)
    child_a, child_b = crossover_ast(parent_a, parent_b, rng=rng)

    assert isinstance(child_a, ASTGenome)
    assert isinstance(child_b, ASTGenome)

    # Both children must compile cleanly
    compile(child_a.to_code(), "<child_a>", "exec")
    compile(child_b.to_code(), "<child_b>", "exec")


def test_multi_hunk_mutation():
    sources = {
        "multi_bug.py": (
            "def process(a, b):\n"
            "    x = a - 10\n"
            "    y = b - 20\n"
            "    return x + y\n"
        )
    }
    initial_patch = PatchGenome()
    rng = random.Random(123)

    mutated = mutate_multi_hunk(initial_patch, sources, rng=rng, num_mutations=2)
    assert isinstance(mutated, PatchGenome)
    applied = mutated.apply_to(sources)
    compile(applied["multi_bug.py"], "multi_bug.py", "exec")


def test_code_causal_context_and_traps():
    code = "def algo(n):\n    return n * 2\n"
    g = ASTGenome.from_code(code)

    ctx_ok = discretize_context(g, parent_fitness=95.0)
    assert "fit:high" in ctx_ok
    assert "status:ok" in ctx_ok

    ctx_err = discretize_code_context(g, parent_fitness=20.0, last_error="SyntaxError")
    assert "fit:low" in ctx_err
    assert "status:err" in ctx_err

    model = CausalModel()
    # Record repeated failures in this code context
    for _ in range(6):
        model.observe("ast_constant", ctx_err, delta=-10.0)

    lib = TrapSignatureLibrary(min_failures=5, fail_rate_threshold=0.7)
    traps = lib.scan(model)
    assert len(traps) >= 1
    assert lib.is_known_trap(f"ast_constant|{ctx_err}") is True


def test_code_descriptors_map_elites():
    code = "def sample(x):\n    if x > 0:\n        return x + 1\n    return 0\n"
    ast_g = ASTGenome.from_code(code)

    m = _desc_mean(ast_g)
    s = _desc_std(ast_g)

    assert m > 0.0 # node count
    assert s > 0.0 # max depth

    patch_g = PatchGenome(hunks=[Hunk("test.py", 1, 1, "a = 1\n", "a = 2\n")])
    pm = _desc_mean(patch_g)
    ps = _desc_std(patch_g)
    assert pm == 1.0 # 1 hunk
    assert ps == 1.0 # 1 line added


def test_assign_species_finds_closest_representative():
    """Verifies that assign_species finds existing matching species instead of creating new ones."""
    engine = EvolutionEngine(speciation_enabled=True, speciation_threshold=2.0)
    engine._representatives["spec_01"] = [1.0] * 16
    engine._representatives["spec_02"] = [4.0] * 16

    child = Individual(genome=[1.05] * 16, species="spec_01")
    assigned = engine.assign_species(child)
    assert assigned == "spec_01"


def test_engine_run_with_ast_population():
    """Verifies that EvolutionEngine.run executes cleanly with an initial population of ASTGenomes."""
    code = "def f(x):\n    return x + 1\n"
    pop_size = 4
    init_pop = [Individual(genome=ASTGenome.from_code(code), species=f"spec_{i:02d}") for i in range(pop_size)]

    def ast_evaluator(ind: Individual) -> float:
        return 50.0 + len(ind.genome.to_code())

    engine = EvolutionEngine(
        population_size=pop_size,
        fitness_fn=ast_evaluator,
        seed=42,
    )
    report = engine.run(generations=2, initial_population=init_pop)
    assert "best_individual" in report
    assert report["best_individual"]["fitness"] > 50.0
    assert len(report["history"]) >= 2
    assert "map_elites" in report
    assert "pareto_front" in report


def test_cyclomatic_complexity_calculation():
    import ast
    from evolab.ast_genome import compute_cyclomatic_complexity

    # Straight line code -> M = 1
    code_linear = "def linear(x):\n    a = x + 1\n    return a * 2\n"
    tree_linear = ast.parse(code_linear)
    assert compute_cyclomatic_complexity(tree_linear) == 1

    # Code with If, For, While, BoolOp -> M = 1 + 1 (if) + 1 (for) + 1 (while) + 1 (and) = 5
    code_branches = (
        "def branchy(x, items):\n"
        "    if x > 0:\n"
        "        for item in items:\n"
        "            while x > 10:\n"
        "                if item > 0 and x < 100:\n"
        "                    return True\n"
        "    return False\n"
    )
    tree_branches = ast.parse(code_branches)
    cc = compute_cyclomatic_complexity(tree_branches)
    assert cc >= 5


def test_multi_file_ast_genome():
    from evolab.ast_genome import MultiFileASTGenome

    sources = {
        "mod_a.py": "def helper(x):\n    return x * 2\n",
        "mod_b.py": "from mod_a import helper\ndef run(n):\n    return helper(n) + 1\n",
    }

    genome = MultiFileASTGenome.from_sources(sources)
    assert len(genome.files) == 2
    assert "mod_a.py" in genome.files
    assert "mod_b.py" in genome.files

    desc = genome.describe()
    assert desc["files_count"] == 2
    assert desc["node_count"] > 0
    assert desc["cyclomatic_complexity"] >= 2

    clone = genome.clone()
    assert clone.distance_to(genome) == 0.0


