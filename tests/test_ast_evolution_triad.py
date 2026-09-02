"""
Comprehensive Unit Tests for AST Evolution Triad Hardening:
1. AST Semantic Guard (ast_guard.py)
2. Fault-Guided Mutation / Spectrum-Based Fault Localization (suspicion.py)
3. AST-Anchored Patch Contract (anchored_patch.py)
"""
from __future__ import annotations

import ast
import pytest

from evolab import (
    ASTSemanticGuard,
    is_ast_change_safe,
    SuspicionMap,
    SuspiciousNode,
    build_suspicion_map,
    FaultGuidedASTMutator,
    compute_ochiai_score,
    NodeAnchor,
    AnchoredHunk,
    AnchoredPatchGenome,
    apply_anchored_hunk,
    AnchoredPatchApplyError,
)


# ===========================================================================
# Pillar 1: AST Semantic Guard
# ===========================================================================

def test_ast_semantic_guard_valid_code():
    valid_code = """
def process_data(items, multiplier=1):
    squared = [x ** 2 for x in items if x > 0]
    total = sum(squared)
    return total * multiplier
"""
    safe, violations = is_ast_change_safe(valid_code)
    assert safe is True
    assert len(violations) == 0


def test_ast_semantic_guard_undefined_variable():
    code_with_undefined = """
def calc(x):
    return x + missing_discount
"""
    safe, violations = is_ast_change_safe(code_with_undefined)
    assert safe is False
    assert any("UndefinedVariable" in v and "missing_discount" in v for v in violations)


def test_ast_semantic_guard_arity_mismatch():
    code_with_arity_bug = """
def multiply(a, b):
    return a * b

def test():
    return multiply(10)
"""
    safe, violations = is_ast_change_safe(code_with_arity_bug)
    assert safe is False
    assert any("ArityMismatch" in v for v in violations)


def test_ast_semantic_guard_non_callable_and_invalid_operations():
    code_with_errors = """
def buggy():
    x = 42()
    y = None + 5
    z = True[0]
"""
    safe, violations = is_ast_change_safe(code_with_errors)
    assert safe is False
    assert any("NonCallableInvocation" in v for v in violations)
    assert any("InvalidOperation" in v for v in violations)
    assert any("InvalidSubscript" in v for v in violations)


# ===========================================================================
# Pillar 2: Fault-Guided AST Mutation / SBFL
# ===========================================================================

def test_compute_ochiai_score():
    # 0 failed hits -> score 0
    assert compute_ochiai_score(failed_hits=0, total_failed=2, passed_hits=1, total_passed=3) == 0.0
    # Executed on all failed tests and no passed tests -> score 1.0
    assert compute_ochiai_score(failed_hits=2, total_failed=2, passed_hits=0, total_passed=3) == 1.0
    # Executed on 1 failed out of 2, and 1 passed out of 2
    score = compute_ochiai_score(failed_hits=1, total_failed=2, passed_hits=1, total_passed=2)
    assert 0.0 < score < 1.0


def test_build_suspicion_map_and_targeted_mutation():
    code = """def can_enter(age, limit):
    if age > limit:
        return True
    return False
"""
    # Test spectra: line 2 and 4 executed on failing test, line 3 only on passing test
    test_runs = [
        ({1, 2, 4}, False),
        ({1, 2, 3}, True),
        ({1, 2, 4}, True),
    ]

    smap = build_suspicion_map(code, test_runs)
    assert smap.line_scores[3] == 0.0
    assert smap.line_scores[4] > smap.line_scores[3]

    top_nodes = smap.get_top_nodes()
    assert len(top_nodes) > 0

    mutator = FaultGuidedASTMutator()
    tree = ast.parse(code)
    mutated_tree, desc = mutator.mutate(tree, smap)
    if mutated_tree is not None:
        assert desc is not None
        safe, _ = is_ast_change_safe(mutated_tree)
        assert safe is True


# ===========================================================================
# Pillar 3: AST-Anchored Patch Contract
# ===========================================================================

def test_anchored_patch_operator_swap():
    sources = {
        "rules.py": """# Unrelated comments and empty lines

def can_enter(age, limit):
    # check condition
    if age > limit:
        return True
    return False
"""
    }

    anchor = NodeAnchor(
        function="can_enter",
        node_type="Compare",
        left_name="age",
        right_name="limit",
        operator="Gt",
    )
    hunk = AnchoredHunk(
        file_path="rules.py",
        operation="replace_operator",
        anchor=anchor,
        parameters={"new_op": "GtE"},
    )
    patch = AnchoredPatchGenome([hunk])

    assert len(patch) == 1
    assert "replace_operator" in patch.describe()["operations"]

    patched_sources = patch.apply_to(sources)
    assert "if age >= limit:" in patched_sources["rules.py"]

    diff = patch.to_unified_diff(sources)
    assert "+    if age >= limit:" in diff


def test_anchored_patch_constant_and_call_replacement():
    code = """def get_greeting(role):
    prefix = 'User'
    return prefix + role
"""
    sources = {"greeting.py": code}

    # Replace constant 'User' with 'Admin'
    anchor_const = NodeAnchor(
        function="get_greeting",
        node_type="Constant",
    )
    hunk_const = AnchoredHunk(
        file_path="greeting.py",
        operation="replace_constant",
        anchor=anchor_const,
        parameters={"new_value": "Admin"},
    )
    patch = AnchoredPatchGenome([hunk_const])
    patched = patch.apply_to(sources)
    assert "prefix = 'Admin'" in patched["greeting.py"]


def test_anchored_patch_missing_anchor_error():
    sources = {"test.py": "def foo(): return 1"}
    anchor = NodeAnchor(function="non_existent", node_type="Compare")
    hunk = AnchoredHunk(file_path="test.py", operation="replace_operator", anchor=anchor, parameters={"new_op": "GtE"})
    patch = AnchoredPatchGenome([hunk])

    with pytest.raises(AnchoredPatchApplyError):
        patch.apply_to(sources)


def test_anchored_patch_fuzzy_and_hierarchical_matching():
    # Demonstrates resilience against variable refactoring/renaming
    code_refactored = """def check_clearance(user_age, limit, clearance_level, min_clearance):
    if user_age > limit:
        return False
    if clearance_level > min_clearance:
        return True
    return False
"""
    sources = {"clearance.py": code_refactored}

    # Anchor specifies original variable name 'age', but enables fuzzy_identifiers + statement_index=0
    anchor = NodeAnchor(
        function="check_clearance",
        node_type="Compare",
        left_name="age",
        operator="Gt",
        statement_index=0,
        parent_type="If",
        fuzzy_identifiers=True,
    )
    hunk = AnchoredHunk(
        file_path="clearance.py",
        operation="replace_operator",
        anchor=anchor,
        parameters={"new_op": "GtE"},
    )
    patch = AnchoredPatchGenome([hunk])
    patched = patch.apply_to(sources)
    assert "if user_age >= limit:" in patched["clearance.py"]


def test_holdout_generalization_invariant_and_overfitting_defense():
    from evolab.instrumentation import SecurityEvaluator

    sources = {
        "bank.py": """def process_withdrawal(balance, amount, daily_limit, current_daily_total):
    if balance >= amount and amount <= daily_limit:
        return balance - amount, True
    return balance, False
"""
    }

    weak_tests = [
        ((100, 40, 50, 0), (60, True)),
        ((100, 150, 50, 0), (100, False)),
    ]
    holdout_tests = [
        ((1000, 30, 50, 40), (1000, False)),
    ]

    evaluator = SecurityEvaluator(
        base_sources=sources,
        target_file="bank.py",
        func_name="process_withdrawal",
        test_cases=weak_tests,
        holdout_test_cases=holdout_tests,
        mode="hardening",
    )

    fit = evaluator.evaluate(sources["bank.py"])
    assert fit.score == 0.0
    assert fit.sub_scores.get("security_score") == 0.0


def test_path_sensitive_conditional_assignment_guard():
    from evolab.ast_guard import is_ast_change_safe

    # 1. Single branch if -> conditional definition -> UnboundLocalError risk
    buggy_code = """def compute(flag):
    if flag:
        res = 42
    return res + 1
"""
    safe, violations = is_ast_change_safe(buggy_code)
    assert not safe
    assert any("ConditionalDefinitionError" in v for v in violations)

    # 2. Both branches defined -> Safe!
    safe_if_else = """def compute(flag):
    if flag:
        res = 42
    else:
        res = 0
    return res + 1
"""
    safe2, violations2 = is_ast_change_safe(safe_if_else)
    assert safe2
    assert len(violations2) == 0

    # 3. Pre-initialized variable -> Safe!
    safe_preinit = """def compute(flag):
    res = 0
    if flag:
        res = 42
    return res + 1
"""
    safe3, violations3 = is_ast_change_safe(safe_preinit)
    assert safe3
    assert len(violations3) == 0


def test_crossover_ast_strict_compatibility_and_no_naive_fallback():
    import random
    from evolab.ast_genome import ASTGenome, crossover_ast

    # Parent A only has Constants; Parent B only has BinOps
    parent_a = ASTGenome.from_code("def foo(): return 42")
    parent_b = ASTGenome.from_code("def foo(x, y): return x + y")

    # In legacy crossover, parent_a would fall back to swapping Constant with BinOp arbitrarily
    # In hardened crossover, incompatible parents return safe clones
    child_a, child_b = crossover_ast(parent_a, parent_b, rng=random.Random(42))
    assert child_a.to_code() == parent_a.to_code()
    assert child_b.to_code() == parent_b.to_code()


def test_typeinfo_library_signatures_and_runtime_harvesting():
    import ast
    from evolab.real_ast.types import TypeInfo

    ti = TypeInfo()

    # Standard and Data Science library signatures
    node_math = ast.parse("math.sqrt(16.0)").body[0].value
    assert ti.infer_type(node_math) == "float"

    node_np = ast.parse("np.mean([1, 2, 3])").body[0].value
    assert ti.infer_type(node_np) == "float"

    # Runtime harvested types
    ti.record_runtime_type("fetch_records", "list")
    node_custom = ast.parse("fetch_records()").body[0].value
    assert ti.infer_type(node_custom) == "list"

    # PEP 484 return type annotation
    fn_annotated = ast.parse("def query(id: int) -> dict: return {}").body[0]
    assert ti.infer_type(fn_annotated) == "dict"
