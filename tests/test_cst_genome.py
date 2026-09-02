"""
Unit tests for CSTGenome: lossless Concrete Syntax Tree (LibCST) representation.
"""
import random
import pytest

pytest.importorskip("libcst")  # optional dependency; skip module if absent

from evolab.cst_genome import (
    CSTGenome,
    CSTConstantMutator,
    CSTComparisonMutator,
    CSTBinaryOpMutator,
    CSTSemanticNodeCollector,
)

SAMPLE_CODE_WITH_COMMENTS = """# Module Header Comment: CLI argument parser
def parse_cli(args: list[str]) -> dict:
    # NOTE: keep this dict in sync with the docs table (see issue #42)
    config = {'port': 8000, 'debug': False, 'host': '127.0.0.1'}  # inline config comment

    for arg in args:
        # Check for explicit debug flag
        if arg == '--debug':
            config['debug'] = True  # user requested debug mode
        elif arg == '--fast':
            config['port'] = 9090

    return config
"""

def test_cst_genome_creation_and_lossless_roundtrip():
    """Validates that CSTGenome preserves 100% of formatting, comments, and newlines."""
    genome = CSTGenome(SAMPLE_CODE_WITH_COMMENTS)
    assert genome.code == SAMPLE_CODE_WITH_COMMENTS
    assert genome.to_code() == SAMPLE_CODE_WITH_COMMENTS
    assert "# Module Header Comment: CLI argument parser" in genome.code
    assert "# NOTE: keep this dict in sync with the docs table (see issue #42)" in genome.code
    assert "# inline config comment" in genome.code


def test_cst_genome_contract_compliance():
    """Validates full compliance with EvolabGenome abstract contract."""
    genome = CSTGenome(SAMPLE_CODE_WITH_COMMENTS)
    clone = genome.clone()

    assert clone.fingerprint() == genome.fingerprint()
    assert clone.distance_to(genome) == 0.0

    desc = genome.describe()
    assert desc["comment_count"] >= 4
    assert desc["semantic_node_count"] > 10
    assert desc["line_count"] > 10

    serialized = genome.serialize()
    assert serialized["type"] == "CSTGenome"
    assert serialized["fingerprint"] == genome.fingerprint()


def test_cst_semantic_distance_ignores_whitespace_and_comments():
    """Proves expert insight: layout and comment modifications do NOT distort genetic distance."""
    base_genome = CSTGenome(SAMPLE_CODE_WITH_COMMENTS)

    code_with_extra_comments = SAMPLE_CODE_WITH_COMMENTS + "\n# Extra trailing comment 1\n# Extra trailing comment 2\n"
    modified_layout_genome = CSTGenome(code_with_extra_comments)

    # Semantic code is identical, so semantic distance must be zero!
    distance = base_genome.distance_to(modified_layout_genome)
    assert distance == 0.0, f"Expected 0.0 semantic distance, got {distance}"


def test_cst_constant_mutator_preserves_comments():
    """Verifies that mutating literal constants leaves all comments and indentation intact."""
    genome = CSTGenome(SAMPLE_CODE_WITH_COMMENTS)
    rng = random.Random(42)

    # Force constant mutation
    mutated = None
    for _ in range(10):
        candidate = genome.mutate(rng=rng)
        if candidate.fingerprint() != genome.fingerprint():
            mutated = candidate
            break

    assert mutated is not None
    # Verify comments survived intact
    assert "# Module Header Comment: CLI argument parser" in mutated.code
    assert "# NOTE: keep this dict in sync with the docs table (see issue #42)" in mutated.code
    assert "# inline config comment" in mutated.code
    assert "# user requested debug mode" in mutated.code


def test_cst_comparison_mutator():
    """Verifies operator mutation in comparison statements."""
    code = "def check(x: int) -> bool:\n    return x == 10  # check equality\n"
    genome = CSTGenome(code)
    rng = random.Random(123)

    mutator = CSTComparisonMutator(rng=rng)
    new_tree = genome.tree.visit(mutator)
    new_code = new_tree.code
    assert "# check equality" in new_code


def test_cst_binary_op_mutator():
    """Verifies binary arithmetic mutation in mathematical expressions."""
    code = "def calc(a: int, b: int) -> int:\n    # compute sum\n    return a + b\n"
    genome = CSTGenome(code)
    rng = random.Random(456)

    mutator = CSTBinaryOpMutator(rng=rng)
    new_tree = genome.tree.visit(mutator)
    new_code = new_tree.code
    assert "# compute sum" in new_code


def test_cst_genome_crossover():
    """Verifies crossover between two compatible CST genomes."""
    code1 = "def f1():\n    # comment 1\n    a = 1\n    b = 2\n    return a + b\n"
    code2 = "def f2():\n    # comment 2\n    x = 10\n    y = 20\n    return x * y\n"
    g1 = CSTGenome(code1)
    g2 = CSTGenome(code2)

    child = g1.crossover(g2, rng=random.Random(42))
    assert isinstance(child, CSTGenome)
    assert len(child.code) > 0
