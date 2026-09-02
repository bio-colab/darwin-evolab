"""
Unit tests for EliteTypeCheckGate: resource-conscious static type verification.
"""
import pytest

pytest.importorskip("libcst")  # optional dependency; skip module if absent

from evolab.type_gate import EliteTypeCheckGate, TypeCheckResult
from evolab.cst_genome import CSTGenome

VALID_TYPED_CODE = """
def multiply_by_two(val: int) -> int:
    return val * 2
"""

INVALID_TYPED_CODE = """
def multiply_by_two(val: int) -> int:
    return "invalid return type"
"""

def test_elite_type_check_gate_valid_code():
    """Ensures correctly typed Python code passes the type check gate."""
    gate = EliteTypeCheckGate(enabled=True)
    res = gate.check_code(VALID_TYPED_CODE)
    assert res.passed is True
    assert res.error_count == 0


def test_elite_type_check_gate_invalid_code():
    """Ensures type mismatches are caught and detailed error traces returned."""
    gate = EliteTypeCheckGate(enabled=True)
    res = gate.check_code(INVALID_TYPED_CODE)
    if res.tool_used != "none":
        assert res.passed is False
        assert res.error_count > 0
        assert any("return-value" in err or "incompatible" in err.lower() for err in res.errors)


def test_elite_type_check_gate_caching():
    """Ensures caching avoids repeating expensive type checker invocations."""
    gate = EliteTypeCheckGate(enabled=True)
    res1 = gate.check_code(VALID_TYPED_CODE)
    res2 = gate.check_code(VALID_TYPED_CODE)
    # Must be exact same cached instance
    assert res1 is res2


def test_elite_type_check_gate_genome_inspection():
    """Ensures gate inspects CSTGenome instances directly."""
    gate = EliteTypeCheckGate(enabled=True)
    genome = CSTGenome(VALID_TYPED_CODE)
    res = gate.check_genome(genome)
    assert res.passed is True
