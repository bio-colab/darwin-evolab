"""
Unit tests for post-adversarial hardening (Z-Hardening 5.0).
Verifies the 5 defensive pillars developed after experimental red-teaming:
1. Semantic & Complexity-Aware AST Speciation Distance.
2. Proactive Trap Blacklisting overrides a poisoned success rate
   (gradual ``apply_decay`` was removed by design — memory-hygiene M4).
3. Sub-Threshold Latency Dilation Invariant in SecurityOracle.
4. Semantic Output Consistency Digest in BehavioralSecurityDescriptor.
5. Cross-File Contract Preservation Validator in CrossFileDependencyGraph.
"""

from __future__ import annotations

import ast
import random
import pytest

from evolab.ast_genome import ASTGenome, ast_distance
from evolab.causal import CausalModel, StrategicMutationSelector, TrapSignatureLibrary, TrapSignature
from evolab.cross_file import CrossFileDependencyGraph
from evolab.instrumentation import (
    BehavioralSecurityDescriptor,
    FaultCategory,
    LatencyDilationInvariant,
    Observation,
    SecurityOracle,
    ViolationSeverity,
)


# ---------------------------------------------------------------------------
# Pillar 1: Semantic & Complexity Distance
# ---------------------------------------------------------------------------

def test_ast_distance_complexity_sensitivity():
    code_base = """def f(x: int) -> int:
    return x * 2
"""
    # Injects conditional logic bomb
    code_bomb = """def f(x: int) -> int:
    if x == 13:
        return -999
    return x * 2
"""
    g_base = ASTGenome.from_code(code_base)
    g_bomb = ASTGenome.from_code(code_bomb)

    code_clean_edit = """def f(x: int) -> int:
    y = x * 2
    return y
"""
    g_clean = ASTGenome.from_code(code_clean_edit)

    d_clean = ast_distance(g_base, g_clean)
    d_bomb = ast_distance(g_base, g_bomb)

    # Injected control-flow branch yields significantly higher distance
    assert d_bomb > d_clean
    assert d_bomb >= 0.35


# ---------------------------------------------------------------------------
# Pillar 2: Proactive Trap Blacklisting (poisoned-rate override)
# ---------------------------------------------------------------------------

def test_proactive_trap_blacklisting_overrides_poisoned_rate():
    """The real defense against a poisoned success rate is the trap penalty.

    Memory-hygiene M4 note: gradual ``apply_decay`` used to be called here
    as if it cleaned the poison — it cannot. Uniform multiplicative count
    shrinkage is scale-invariant on ``success_rate``, so a poisoned 1.0
    rate survives any amount of decay exactly. The poisoned rate below is
    left untouched on purpose: the assertion chain proves the trap library
    (negative memory) — not decay — is what neutralizes the poisoning.
    """
    model = CausalModel()
    context = "cplx:simple|fit:high|status:ok"

    # Attacker injects 20 fake successes
    for _ in range(20):
        model.observe(mtype="semantic", context_bin=context, delta=10.0)

    # Clean operator has 10 observations with 50% success
    for _ in range(5):
        model.observe(mtype="light", context_bin=context, delta=1.0)
    for _ in range(5):
        model.observe(mtype="light", context_bin=context, delta=-1.0)

    assert model.success_rate("semantic", context) == 1.0

    # The poison persists in the model's beliefs — nothing decays it away
    # (and counts stay integers; the int contract of _stats is intact).
    assert model.success_rate("semantic", context) == 1.0
    assert all(type(v) is int for v in model._stats[f"semantic|{context}"].values())

    # Trap Library integration
    trap_lib = TrapSignatureLibrary(min_failures=3, fail_rate_threshold=0.60)
    # Register trap manually or via scan
    trap_lib.signatures[f"semantic|{context}"] = TrapSignature(
        context_bin=f"semantic|{context}",
        failure_count=5,
        mean_negative_delta=-50.0,
        confidence=0.9,
        last_seen_generation=1,
    )

    selector = StrategicMutationSelector(
        model=model,
        epsilon=0.0,  # Pure greedy to test trap penalty
        trap_library=trap_lib,
    )
    rng = random.Random(42)
    # Even though semantic has 100% success rate, the proactive trap penalty downweights it
    chosen = selector.select(context_bin=context, rng=rng)
    assert chosen == "light"


def test_gradual_decay_removed_by_design():
    """Memory-hygiene M4 decision lock.

    ``CausalModel.apply_decay`` was removed: it had zero production callers
    (the causal layer itself is opt-in), and its documented purpose —
    "mitigate historical poisoning" — is mathematically impossible for
    uniform multiplicative decay, which is scale-invariant on
    ``success_rate``. Forgetting in the causal layer is reset-based
    (inertia breaker + trap TTL). This test exists so that re-adding a
    silent decay must confront this recorded decision explicitly.
    """
    assert not hasattr(CausalModel(), "apply_decay")


# ---------------------------------------------------------------------------
# Pillar 3: Latency Dilation Invariant
# ---------------------------------------------------------------------------

def test_latency_dilation_invariant_penalizes_sub_threshold_bloat():
    inv = LatencyDilationInvariant(max_allowed_ms=500.0, max_latency_ratio=2.0)

    # Observation with high latency ratio (3.3x slower)
    obs_dilated = Observation(
        target_name="target",
        duration_ms=350.0,
        metadata={"latency_ratio": 3.3},
        fault_category=FaultCategory.NORMAL_SUCCESS,
    )

    v = inv.check(obs_dilated)
    assert v is not None
    assert v.severity in (ViolationSeverity.LOW, ViolationSeverity.MEDIUM)
    assert "Sub-threshold latency dilation detected" in v.message

    # Test in SecurityOracle
    oracle = SecurityOracle()
    verdict = oracle.evaluate(obs_dilated)
    assert verdict.security_score < 100.0


# ---------------------------------------------------------------------------
# Pillar 4: Output Digest in BehavioralSecurityDescriptor
# ---------------------------------------------------------------------------

def test_behavioral_descriptor_output_digest():
    obs_true = Observation(target_name="calc", return_value=140, duration_ms=2.0, fault_category=FaultCategory.NORMAL_SUCCESS)
    obs_spoof = Observation(target_name="calc", return_value=-999800, duration_ms=2.0, fault_category=FaultCategory.NORMAL_SUCCESS)

    d_true = BehavioralSecurityDescriptor.from_trajectory([obs_true])
    d_spoof = BehavioralSecurityDescriptor.from_trajectory([obs_spoof])

    assert d_true.output_digest != ""
    assert d_spoof.output_digest != ""
    assert d_true.output_digest != d_spoof.output_digest

    # Standard representation-blind distance
    dist_blind = d_true.distance(d_spoof, check_functional_digest=False)
    # Functional integrity checked distance
    dist_checked = d_true.distance(d_spoof, check_functional_digest=True)

    assert dist_checked > dist_blind
    assert dist_checked >= 0.15


# ---------------------------------------------------------------------------
# Pillar 5: Cross-File Contract Preservation
# ---------------------------------------------------------------------------

def test_cross_file_contract_preservation_validator():
    sources = {
        "service.py": """def execute_request(req_id: int, options: dict) -> dict:
    return {"status": "OK", "data": [req_id]}
""",
        "client.py": """from service import execute_request
def call_api(id: int) -> str:
    res = execute_request(id, {})
    return res["status"]
""",
    }

    graph = CrossFileDependencyGraph.build(sources)

    # Valid mutation: preserves signature and keys
    code_valid = """def execute_request(req_id: int, options: dict) -> dict:
    return {"status": "OK", "data": [req_id * 2], "extra": True}
"""
    valid, violations = graph.validate_contract_preservation("service.py", code_valid)
    assert valid is True
    assert len(violations) == 0

    # Invalid mutation 1: removes key 'status' accessed by client.py
    code_missing_key = """def execute_request(req_id: int, options: dict) -> dict:
    return {"result": "OK", "data": [req_id]}
"""
    valid_key, violations_key = graph.validate_contract_preservation("service.py", code_missing_key)
    assert valid_key is False
    assert any("SchemaContractViolation" in v and "'status'" in v for v in violations_key)

    # Invalid mutation 2: removes exported function
    code_missing_func = """def another_function(x: int) -> int:
    return x
"""
    valid_fn, violations_fn = graph.validate_contract_preservation("service.py", code_missing_func)
    assert valid_fn is False
    assert any("ExportViolation" in v for v in violations_fn)
