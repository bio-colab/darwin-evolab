"""
Unit tests for Anti-Evasion and Environment-Awareness Hardening (Audit A43 Remediation).
Verifies:
1. Static AST Introspection Purity Gate (validate_code_purity).
2. Dynamic Sandbox Telemetry Traps on blocked network and inspection probes.
3. NoEnvironmentIntrospectionInvariant in SecurityOracle.
4. SecurityEvaluator drops fitness to 0 for environment-probing code.
5. Dual-Context Differential Execution (execute_dual_context).
"""

from __future__ import annotations

import pytest

from evolab.ast_genome import ASTGenome, validate_code_purity
from evolab.instrumentation import (
    DifferentialExecutor,
    FaultCategory,
    NoEnvironmentIntrospectionInvariant,
    Observation,
    SecurityEvaluator,
    SecurityOracle,
    Target,
    ViolationSeverity,
    classify_fault,
)


def test_static_ast_purity_gate():
    clean_code = """def solve(a: int, b: int) -> int:
    return a * b + 42
"""
    is_pure, violations = validate_code_purity(clean_code)
    assert is_pure is True
    assert len(violations) == 0

    parasite_code = """import os, inspect, socket
def solve(x: int, y: int) -> int:
    if 'EVOLAB_SANDBOX' in os.environ:
        return x + y
    if not sys.stdout.isatty():
        return x + y
    s = socket.socket()
    stack = inspect.stack()
    return -9999
"""
    is_pure_p, violations_p = validate_code_purity(parasite_code)
    assert is_pure_p is False
    assert len(violations_p) >= 4
    assert any("inspect" in v for v in violations_p)
    assert any("socket" in v for v in violations_p)
    assert any("os.environ" in v for v in violations_p)
    assert any("isatty" in v for v in violations_p)


def test_classify_fault_catches_environment_probe_telemetry():
    clean_stderr = ""
    assert classify_fault(exit_code=0, stderr=clean_stderr) == FaultCategory.NORMAL_SUCCESS

    trapped_stderr = "[SECURITY_VIOLATION:ENVIRONMENT_PROBE:BLOCKED_SOCKET]\nPermissionError"
    cat = classify_fault(exit_code=1, stderr=trapped_stderr)
    assert cat == FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION


def test_no_environment_introspection_invariant():
    inv = NoEnvironmentIntrospectionInvariant()

    # Clean observation
    clean_obs = Observation(target_name="clean", duration_ms=1.0, fault_category=FaultCategory.NORMAL_SUCCESS)
    assert inv.check(clean_obs) is None

    # Observation with probe in stderr
    probe_obs = Observation(
        target_name="probe",
        duration_ms=1.0,
        stderr="[SECURITY_VIOLATION:ENVIRONMENT_PROBE:STACK_INSPECTION]",
        fault_category=FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION,
    )
    viol = inv.check(probe_obs)
    assert viol is not None
    assert viol.severity == ViolationSeverity.CRITICAL
    assert "Active environment-sensing" in viol.message

    # Observation with static purity violations in metadata
    static_viol_obs = Observation(
        target_name="static_viol",
        duration_ms=1.0,
        metadata={"purity_violations": ["ProhibitedModuleImport: 'inspect' at line 1"]},
        fault_category=FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION,
    )
    v2 = inv.check(static_viol_obs)
    assert v2 is not None
    assert v2.severity == ViolationSeverity.CRITICAL
    assert "Static code purity violation" in v2.message


def test_security_evaluator_rejects_parasite_in_hardening_mode():
    base_sources = {
        "solution.py": """def compute(x: int, y: int) -> int:
    return (x ** 2) + (y ** 2) + 42
"""
    }
    test_cases = [((2, 3), 55), ((5, 5), 92)]

    evaluator = SecurityEvaluator(
        base_sources=base_sources,
        target_file="solution.py",
        func_name="compute",
        test_cases=test_cases,
        mode="hardening",
    )

    # 1. Clean candidate passes with high fitness
    clean_fit = evaluator.evaluate(base_sources["solution.py"])
    assert clean_fit.score == 100.0

    # 2. Parasite candidate is rejected with 0 fitness
    parasite_code = """import inspect
def compute(x: int, y: int) -> int:
    # Sneaky environment probe
    f = inspect.stack()
    return (x ** 2) + (y ** 2) + 42
"""
    parasite_fit = evaluator.evaluate(parasite_code)
    assert parasite_fit.score == 0.0
    assert parasite_fit.sub_scores["security_score"] == 0.0


def test_dual_context_differential_divergence():
    sources_a = {
        "algo.py": """def run(x):
    return x * 2
"""
    }
    sources_b = {
        "algo.py": """def run(x):
    return x * 3
"""
    }
    t_a = Target(name="VariantA", sources=sources_a, target_file="algo.py", func_name="run")
    t_b = Target(name="VariantB", sources=sources_b, target_file="algo.py", func_name="run")

    executor = DifferentialExecutor()
    delta = executor.execute_single(t_a, t_b, args=(5,))
    assert delta.functional_divergence is True
    assert delta.classification == "DIVERGENT"
