"""Tests for RealASTGenome, SymbolTable, TypeInfo, Intelligent Mutation, Crossover, and Optimization."""
from __future__ import annotations

import ast
import random
import pytest

from evolab import (
    RealASTGenome,
    SymbolTable,
    Symbol,
    TypeInfo,
    DeepAnalyzer,
    NodeType,
    CriticalityLevel,
    IntelligentMutator,
    IntelligentCrossover,
    ASTOptimizer,
)


def test_symbol_table_lexical_scoping_and_shadowing():
    table = SymbolTable()
    # Global variable x
    table.add_symbol(Symbol(name="x", scope="global", symbol_type="variable", inferred_type="int"))
    # Function scope func:compute child of global
    table.set_parent_scope("func:compute", "global")
    # Shadowing variable x inside func:compute
    table.add_symbol(Symbol(name="x", scope="func:compute", symbol_type="variable", inferred_type="float"))
    table.add_symbol(Symbol(name="y", scope="func:compute", symbol_type="parameter", inferred_type="int"))

    # Lookup x from func:compute should return the shadowed float
    sym_local = table.lookup("x", "func:compute")
    assert sym_local is not None
    assert sym_local.scope == "func:compute"
    assert sym_local.inferred_type == "float"

    # Lookup x from global should return the global int
    sym_global = table.lookup("x", "global")
    assert sym_global is not None
    assert sym_global.scope == "global"
    assert sym_global.inferred_type == "int"

    # Lookup y from func:compute succeeds; from global fails
    assert table.lookup("y", "func:compute") is not None
    assert table.lookup("y", "global") is None


def test_type_inference_and_compatibility():
    type_info = TypeInfo()
    table = SymbolTable()
    table.add_symbol(Symbol(name="a", scope="global", symbol_type="variable", inferred_type="int"))
    table.add_symbol(Symbol(name="b", scope="global", symbol_type="variable", inferred_type="int"))

    # Int literal
    node_const = ast.Constant(value=42)
    assert type_info.infer_type(node_const) == "int"

    # Float literal
    node_float = ast.Constant(value=3.14)
    assert type_info.infer_type(node_float) == "float"

    # BinOp int + int
    node_bin = ast.BinOp(left=ast.Constant(value=10), op=ast.Add(), right=ast.Constant(value=20))
    assert type_info.infer_type(node_bin) == "int"

    # BinOp int / int -> float
    node_div = ast.BinOp(left=ast.Constant(value=10), op=ast.Div(), right=ast.Constant(value=2))
    assert type_info.infer_type(node_div) == "float"

    # Type compatibility
    assert type_info.is_type_compatible(node_const, node_float) is True
    node_str = ast.Constant(value="hello")
    type_info.infer_type(node_str)
    assert type_info.is_type_compatible(node_const, node_str) is False


def test_deep_analyzer_metadata_extraction():
    code = (
        "def process(x: int, y: int) -> int:\n"
        "    res = x + y\n"
        "    if res > 10:\n"
        "        return res * 2\n"
        "    return res\n"
    )
    tree = ast.parse(code)
    analyzer = DeepAnalyzer()
    metadata, symbol_table, type_info = analyzer.analyze(tree)

    # FunctionDef should be marked critical
    func_node = tree.body[0]
    meta_func = metadata[id(func_node)]
    assert meta_func.node_type == NodeType.FUNCTION_DEF
    assert meta_func.is_critical is True
    assert meta_func.criticality == CriticalityLevel.IMMUTABLE
    assert meta_func.mutation_probability <= 0.1

    # Parameters registered
    assert symbol_table.lookup("x", "func:process") is not None
    assert symbol_table.lookup("y", "func:process") is not None
    assert symbol_table.lookup("res", "func:process") is not None

    # Returns marked critical
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    assert len(returns) == 2
    for r in returns:
        assert metadata[id(r)].node_type == NodeType.RETURN
        assert metadata[id(r)].is_critical is True


def test_intelligent_mutator_semantic_safety():
    code = (
        "def compute(x: int, y: int):\n"
        "    base = 10\n"
        "    factor = 2\n"
        "    if x > y:\n"
        "        return (x + base) * factor\n"
        "    return y - base\n"
    )
    genome = RealASTGenome.from_code(code)
    rng = random.Random(42)

    # Perform 50 sequential mutations, verifying 100% syntactic and compilation validity
    current = genome
    for _ in range(50):
        mutated = current.mutate(rng=rng)
        mutated_code = mutated.to_code()
        # Must compile cleanly
        compiled = compile(mutated_code, "<test_mutate>", "exec")
        assert compiled is not None
        # Execute safely
        env = {}
        exec(compiled, env)
        func = env.get("compute")
        assert callable(func)
        # Verify it runs without NameError or TypeError
        val = func(5, 3)
        assert isinstance(val, (int, float, bool))
        current = mutated


def test_intelligent_crossover_type_and_scope_safety():
    code_a = (
        "def compute(x, y):\n"
        "    alpha = x + 1\n"
        "    return alpha * 2\n"
    )
    code_b = (
        "def compute(x, y):\n"
        "    beta = y * 3\n"
        "    return beta + 5\n"
    )
    parent_a = RealASTGenome.from_code(code_a)
    parent_b = RealASTGenome.from_code(code_b)
    rng = random.Random(42)

    crossover_op = IntelligentCrossover(rng=rng)
    child_a, child_b = crossover_op.crossover(parent_a, parent_b)

    for child in (child_a, child_b):
        code = child.to_code()
        compiled = compile(code, "<test_crossover>", "exec")
        assert compiled is not None
        env = {}
        exec(compiled, env)
        fn = env["compute"]
        assert isinstance(fn(4, 2), (int, float))


def test_ast_optimizer_passes():
    # Code with constant folding opportunity, dead code, and redundant branches
    unoptimized_code = (
        "def calculate(x):\n"
        "    val = 10 + 5\n"
        "    flag = not False\n"
        "    if True:\n"
        "        return val * 2\n"
        "        unreachable = 999\n"
        "    return 0\n"
    )
    genome = RealASTGenome.from_code(unoptimized_code)
    optimized = genome.optimize()
    code = optimized.to_code()

    # Constant folding: 10 + 5 -> 15
    assert "15" in code
    # Dead code: unreachable statement pruned
    assert "unreachable" not in code

    # Execute to confirm semantic fidelity
    env = {}
    exec(compile(code, "<opt_exec>", "exec"), env)
    assert env["calculate"](1) == 30


def test_real_ast_genome_evolab_protocol():
    code = "def f(x):\n    return x + 1\n"
    g1 = RealASTGenome.from_code(code)
    g2 = g1.clone()

    assert g1.fingerprint() == g2.fingerprint()
    assert g1.distance_to(g2) == 0.0

    desc = g1.describe()
    assert desc["node_count"] > 0
    assert "func:f" in desc["scopes"]
    assert desc["critical_nodes"] >= 1

    serialized = g1.serialize()
    assert serialized["type"] == "RealASTGenome"
    assert "f" in serialized["symbols"]
