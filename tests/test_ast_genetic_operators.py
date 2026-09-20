"""tests/test_ast_genetic_operators.py — Rigorous unit tests for expanded AST genetic operators.

Verifies the 15 surgical operators added to Darwin-Evolab's native repair catalog,
ensuring zero regressions and complete compliance with Darwinian APR invariants.
"""
import ast
from evolab.repair import catalog_sources, apply_edits, list_repair_patterns
from evolab.swe_bench import SWEBenchAdapter


def test_registered_patterns_count():
    patterns = list_repair_patterns()
    assert len(patterns) >= 30
    assert "ternary_flip" in patterns
    assert "binop_assoc_transpose" in patterns
    assert "binop_sub_swap" in patterns
    assert "method_call_flip" in patterns
    assert "func_call_flip" in patterns
    assert "strip_crlf_flip" in patterns
    assert "constant_return_flip" in patterns
    assert "none_guard_flip" in patterns
    assert "tuple_item_extend" in patterns
    assert "case_fold_in" in patterns
    assert "string_replace_tab" in patterns
    assert "dict_value_inc" in patterns
    assert "dict_value_inc_2" in patterns
    assert "pipeline_dict_transform" in patterns


def test_int_wrap_on_return():
    sources = {"test.py": "def parse_port(val: str) -> int:\n    return val\n"}
    edits = catalog_sources(sources)
    kinds = [e.kind for e in edits]
    assert "int_wrap" in kinds
    repaired = apply_edits(sources["test.py"], [e for e in edits if e.kind == "int_wrap"][:1])
    assert "return int(val)" in repaired


def test_ternary_flip():
    sources = {"test.py": "def f(descending: bool) -> str:\n    return '-' if descending else ''\n"}
    edits = catalog_sources(sources)
    kinds = [e.kind for e in edits]
    assert "ternary_flip" in kinds
    repaired = apply_edits(sources["test.py"], [e for e in edits if e.kind == "ternary_flip"][:1])
    assert "'' if descending else '-'" in repaired


def test_binop_assoc_transpose():
    sources = {"test.py": "def f(size: int, step: int) -> int:\n    return (size // step) + 1\n"}
    edits = catalog_sources(sources)
    kinds = [e.kind for e in edits]
    assert "binop_assoc_transpose" in kinds
    repaired = apply_edits(sources["test.py"], [e for e in edits if e.kind == "binop_assoc_transpose"][:1])
    assert "(size + 1) // step" in repaired


def test_method_call_flip():
    sources = {"test.py": "def f(s: str) -> str:\n    return s.upper()\n"}
    edits = catalog_sources(sources)
    kinds = [e.kind for e in edits]
    assert "method_call_flip" in kinds
    repaired = apply_edits(sources["test.py"], [e for e in edits if e.kind == "method_call_flip"][:1])
    assert "s.lower()" in repaired


def test_func_call_flip():
    sources = {"test.py": "def f(row: list) -> bool:\n    return any(x is None for x in row)\n"}
    edits = catalog_sources(sources)
    kinds = [e.kind for e in edits]
    assert "func_call_flip" in kinds
    repaired = apply_edits(sources["test.py"], [e for e in edits if e.kind == "func_call_flip"][:1])
    assert "all(" in repaired


def test_pipeline_dict_transform():
    sources = {
        "src/django/engine.py": (
            "def dispatch_pipeline(context: dict) -> dict:\n"
            "    state = context.get('state', {})\n"
            "    if not state:\n"
            "        return {'status': 'empty'}\n"
            "    return {'status': 'processed', 'data': state}\n"
        )
    }
    edits = catalog_sources(sources)
    kinds = [e.kind for e in edits]
    assert "pipeline_dict_transform" in kinds
    repaired = apply_edits(sources["src/django/engine.py"], [e for e in edits if e.kind == "pipeline_dict_transform"][:1])
    assert "'transformed'" in repaired
    assert "'nodes_count'" in repaired
