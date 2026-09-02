import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.instrumentation import (
    FaultCategory,
    classify_fault,
    Target,
    Observation,
    Invariant,
    Violation,
    ViolationSeverity,
    ExpectedBehaviorInvariant,
    NoPrivilegeBoundaryViolationInvariant,
    NoStateCorruptionInvariant,
    NoResourceExhaustionInvariant,
    BoundedExecutionTimeInvariant,
    SecurityOracle,
    OracleVerdict,
    DifferentialExecutor,
    BehavioralDelta,
)
from evolab.sandbox import SandboxConfig, SandboxRunner, ExecutionResult


# ===========================================================================
# Pillar D: Fault Taxonomy Tests
# ===========================================================================

def test_classify_fault_normal_success():
    fault = classify_fault(exit_code=0, error=None)
    assert fault == FaultCategory.NORMAL_SUCCESS


def test_classify_fault_memory_safety_posix_and_win32():
    # POSIX SIGSEGV
    assert classify_fault(exit_code=139) == FaultCategory.MEMORY_SAFETY_SIGNAL
    # Windows STATUS_ACCESS_VIOLATION (0xC0000005)
    assert classify_fault(exit_code=3221225477) == FaultCategory.MEMORY_SAFETY_SIGNAL
    assert classify_fault(exit_code=-1073741819) == FaultCategory.MEMORY_SAFETY_SIGNAL
    # Explicit string in stderr
    assert classify_fault(error="Segmentation fault (core dumped)") == FaultCategory.MEMORY_SAFETY_SIGNAL


def test_classify_fault_privilege_boundary_violation():
    assert classify_fault(error="PermissionError: Network access is blocked") == FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION
    # Via telemetry
    tele = [{"cat": "socket.connect", "target": "127.0.0.1"}]
    assert classify_fault(telemetry_events=tele) == FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION


def test_classify_fault_resource_exhaustion():
    assert classify_fault(timeout_triggered=True) == FaultCategory.RESOURCE_EXHAUSTION
    assert classify_fault(memory_limit_triggered=True) == FaultCategory.RESOURCE_EXHAUSTION
    assert classify_fault(error="MemoryError: unable to allocate array") == FaultCategory.RESOURCE_EXHAUSTION


def test_classify_fault_logic_deviation():
    assert classify_fault(error="ZeroDivisionError: division by zero") == FaultCategory.LOGIC_DEVIATION
    assert classify_fault(error="KeyError: 'token_id'") == FaultCategory.LOGIC_DEVIATION
    assert classify_fault(error="TypeError: unsupported operand type") == FaultCategory.LOGIC_DEVIATION


def test_sandbox_runner_records_fault_category():
    runner = SandboxRunner(SandboxConfig(timeout_seconds=2.0))
    sources = {"target.py": "def solve(x):\n    return x * 2\n"}
    res = runner.run_function(sources, "target.py", "solve", (5,))
    assert res.success is True
    assert res.return_value == 10
    assert res.fault_category == FaultCategory.NORMAL_SUCCESS.value

    # Crash function
    crash_sources = {"target.py": "def solve(x):\n    return 1 / 0\n"}
    res_crash = runner.run_function(crash_sources, "target.py", "solve", (5,))
    assert res_crash.success is False
    assert res_crash.fault_category == FaultCategory.LOGIC_DEVIATION.value


# ===========================================================================
# Pillar A: Target Instrumentation Abstraction Tests
# ===========================================================================

def test_target_creation_and_retrieval():
    code = "def solve(x): return x + 1"
    target = Target.from_code(code, name="adder_v1")
    assert target.name == "adder_v1"
    assert target.target_file == "target.py"
    assert target.get_source() == code


def test_observation_lifecycle():
    res = ExecutionResult(
        success=True,
        return_value=42,
        stdout="computed\n",
        duration_ms=12.5,
        exit_code=0,
    )
    obs = Observation.from_execution_result("calc_v1", res)
    assert obs.target_name == "calc_v1"
    assert obs.return_value == 42
    assert obs.is_clean() is True
    assert obs.fault_category == FaultCategory.NORMAL_SUCCESS


# ===========================================================================
# Pillar B: Security-Oriented Invariant Oracle Tests
# ===========================================================================

def test_security_oracle_clean_pass():
    oracle = SecurityOracle()
    obs = Observation(
        target_name="clean_fn",
        return_value=10,
        fault_category=FaultCategory.NORMAL_SUCCESS,
        duration_ms=5.0,
    )
    verdict = oracle.evaluate(obs, expected=10)
    assert verdict.passed is True
    assert verdict.security_score == 100.0
    assert verdict.max_severity == ViolationSeverity.NONE
    assert len(verdict.violations) == 0


def test_security_oracle_output_mismatch():
    oracle = SecurityOracle()
    obs = Observation(
        target_name="bad_calc",
        return_value=999,
        fault_category=FaultCategory.NORMAL_SUCCESS,
        duration_ms=5.0,
    )
    verdict = oracle.evaluate(obs, expected=10)
    assert verdict.passed is False
    assert verdict.max_severity == ViolationSeverity.MEDIUM
    assert verdict.security_score < 100.0
    assert any(v.invariant_name == "FunctionalCorrectness" for v in verdict.violations)


def test_security_oracle_privilege_violation_escalation():
    oracle = SecurityOracle()
    obs = Observation(
        target_name="escaped_fn",
        return_value=10,
        fault_category=FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION,
        stderr="PermissionError: Network access is blocked by Sandbox policy",
        telemetry_events=[{"cat": "socket.create_connection", "target": "8.8.8.8"}],
    )
    verdict = oracle.evaluate(obs, expected=10)
    assert verdict.passed is False
    assert verdict.max_severity == ViolationSeverity.CRITICAL
    assert verdict.security_score == 0.0
    assert any(v.severity == ViolationSeverity.CRITICAL for v in verdict.violations)


def test_security_oracle_state_corruption():
    oracle = SecurityOracle()
    obs = Observation(
        target_name="polluter_fn",
        return_value=10,
        state_diff={"added": ["unauthorized_dump.txt"], "changed": []},
    )
    verdict = oracle.evaluate(obs, expected=10)
    assert verdict.passed is False
    assert verdict.max_severity == ViolationSeverity.HIGH
    assert verdict.security_score <= 50.0
    assert any(v.invariant_name == "NoStateCorruption" for v in verdict.violations)


# ===========================================================================
# Pillar C: Differential Execution Engine Tests
# ===========================================================================

def test_differential_executor_identical():
    diff_exec = DifferentialExecutor()
    obs1 = Observation(target_name="base", return_value=10, duration_ms=10.0)
    obs2 = Observation(target_name="cand", return_value=10, duration_ms=10.2)

    delta = diff_exec.compare_observations(obs1, obs2)
    assert delta.functional_divergence is False
    assert delta.output_distance == 0.0
    assert delta.classification == "IDENTICAL"
    assert delta.is_benign() is True


def test_differential_executor_functional_divergence():
    diff_exec = DifferentialExecutor()
    obs1 = Observation(target_name="base", return_value=10, duration_ms=10.0)
    obs2 = Observation(target_name="cand", return_value=15, duration_ms=10.0)

    delta = diff_exec.compare_observations(obs1, obs2)
    assert delta.functional_divergence is True
    assert delta.output_distance == 5.0
    assert delta.classification == "DIVERGENT"
    assert delta.is_benign() is False


def test_differential_executor_crash_and_security_anomaly():
    diff_exec = DifferentialExecutor()
    obs_base = Observation(target_name="base", return_value=10, duration_ms=10.0)

    # Crash candidate
    obs_crash = Observation(
        target_name="cand",
        return_value=None,
        fault_category=FaultCategory.MEMORY_SAFETY_SIGNAL,
        error="Segmentation fault",
    )
    delta_crash = diff_exec.compare_observations(obs_base, obs_crash)
    assert delta_crash.fault_divergence is True
    assert delta_crash.classification == "CRASH_DIFFERENTIAL"

    # Security anomaly candidate
    obs_sec = Observation(
        target_name="cand",
        return_value=10,
        fault_category=FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION,
    )
    delta_sec = diff_exec.compare_observations(obs_base, obs_sec)
    assert delta_sec.classification == "SECURITY_ANOMALY"


def test_differential_executor_live_sandbox_execution():
    runner = SandboxRunner(SandboxConfig(timeout_seconds=2.0))
    diff_exec = DifferentialExecutor()

    base_target = Target.from_code(
        "def solve(x):\n    return x + 2\n",
        name="baseline_addition"
    )
    cand_target = Target.from_code(
        "def solve(x):\n    return x * 2\n",
        name="mutated_multiplication"
    )

    delta = diff_exec.execute_single(base_target, cand_target, args=(5,), runner=runner)
    assert delta.baseline_obs.return_value == 7
    assert delta.candidate_obs.return_value == 10
    assert delta.functional_divergence is True
    assert delta.output_distance == 3.0
    assert delta.classification == "DIVERGENT"


def test_security_evaluator_modes():
    from evolab.instrumentation import SecurityEvaluator

    sources = {"target.py": "def solve(x):\n    return x + 10\n"}
    eval_disc = SecurityEvaluator(base_sources=sources, test_cases=[((5,), 15)], mode="discovery")
    res_disc = eval_disc.evaluate("def solve(x):\n    return x + 10\n")
    # Clean execution has 0 severity -> 0 discovery score
    assert res_disc.score == 0.0

    eval_hard = SecurityEvaluator(base_sources=sources, test_cases=[((5,), 15)], mode="hardening")
    res_hard = eval_hard.evaluate("def solve(x):\n    return x + 10\n")
    assert res_hard.score == 100.0


def test_differential_evaluator_modes():
    from evolab.instrumentation import DifferentialEvaluator, Target

    base = Target.from_code("def solve(x):\n    return x + 1\n", name="base")
    eval_diff_disc = DifferentialEvaluator(baseline_target=base, test_cases=[((2,), 3)], mode="discovery")
    # Divergent code gives discovery score
    res = eval_diff_disc.evaluate("def solve(x):\n    return x * 10\n")
    assert res.score > 0.0

    eval_diff_reg = DifferentialEvaluator(baseline_target=base, test_cases=[((2,), 3)], mode="regression")
    # Identical code gives 100 regression fidelity
    res_reg = eval_diff_reg.evaluate("def solve(x):\n    return x + 1\n")
    assert res_reg.score == 100.0


def test_z_protocol_breakthrough_probe_promotion():
    from evolab.z_protocol import ZProtocolConfig, ZProtocolEngine
    from evolab.genome import Individual, FloatGenome

    # Peak target at [1.0, 1.0] giving 100.0, everywhere else giving 50.0
    def fitness_fn(ind: Individual) -> float:
        vals = ind.genome.values
        if all(0.95 <= v <= 1.05 for v in vals):
            return 100.0
        return 50.0

    cfg = ZProtocolConfig(
        frontier_pioneer_enabled=True,
        causal_momentum_enabled=True,
        internal_robustness_verification=True,
        perturbation_epsilon=0.05,
    )
    engine = ZProtocolEngine(
        fitness_fn=fitness_fn,
        population_size=10,
        genome_size=2,
        seed=42,
        z_config=cfg,
    )
    # Seed an individual right next to the boundary [0.92, 0.92] -> probe at +0.05 hits 0.97 (100.0)
    near_ind = Individual(genome=FloatGenome(values=[0.92, 0.92]), species="spec_default", fitness=50.0)
    engine.evaluate([near_ind])

    # The probe at +eps (0.92 + 0.05 = 0.97) hits 100.0 and should be promoted
    assert near_ind.fitness == 100.0
    assert 0.95 <= near_ind.genome.values[0] <= 1.05


def test_memory_arena_integrity_and_overflow():
    from evolab.instrumentation import MemoryArena

    arena = MemoryArena(capacity=16)
    ok, status = arena.check_canaries()
    assert ok is True
    assert status == "INTEGRITY_VERIFIED"

    # Benign write of 10 bytes inside capacity 16
    arena.write(b"1234567890", offset=0)
    ok, status = arena.check_canaries()
    assert ok is True

    # Buffer overrun of 10 bytes starting at offset 10 (total 20 > 16, within 24 allocated including canaries)
    arena.write(b"OVERRUN_CL", offset=10)
    ok, status = arena.check_canaries()
    assert ok is False
    assert "EPILOGUE_CANARY_OVERWRITTEN" in status

    # Extreme overflow out of mapped arena triggers hardware MemoryError
    import pytest
    with pytest.raises(MemoryError):
        arena.write(b"EXTREME_OVERFLOW_PAST_PAGE" * 10, offset=10)


def test_no_memory_safety_invariant_trigger():
    from evolab.instrumentation import (
        NoMemorySafetyViolationInvariant,
        Observation,
        FaultCategory,
        ViolationSeverity,
    )

    inv = NoMemorySafetyViolationInvariant()

    # Clean observation
    obs_clean = Observation(target_name="clean_mem", return_value=0)
    assert inv.check(obs_clean) is None

    # Memory corruption signal
    obs_fault = Observation(
        target_name="corrupt_mem",
        fault_category=FaultCategory.MEMORY_SAFETY_SIGNAL,
        error="Access violation reading 0x00000000",
    )
    v = inv.check(obs_fault)
    assert v is not None
    assert v.severity == ViolationSeverity.CRITICAL
    assert v.fault_category == FaultCategory.MEMORY_SAFETY_SIGNAL


def test_cross_target_transfer_memory():
    from evolab.instrumentation import (
        CrossTargetTransferMemory,
        FaultCategory,
        ViolationSeverity,
    )
    from evolab.engine import EvolutionEngine

    mem = CrossTargetTransferMemory()
    # Learned on Target A (e.g. protocol traversal)
    mem.record_exploit_motif(
        name="traversal_depth_4",
        genome_values=[3.2, 4.8, 4.9, 4.7, 4.8],
        fault=FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION,
        severity=ViolationSeverity.CRITICAL,
        source_target="Target_Protocol",
    )
    assert len(mem.archive) == 1

    # New engine on Target B
    engine = EvolutionEngine(
        fitness_fn=lambda ind: 10.0,
        population_size=8,
        genome_size=5,
        seed=42,
    )
    injected = mem.seed_target_population(engine, target_name="Target_Service", max_injections=2)
    assert injected == 1
    # Check that the injected pioneer matches the transferred motif
    pioneer = engine.population[-1]
    assert pioneer.species == "spec_transfer_Target_Protocol"
    assert pioneer.genome.values[0] == 3.2
    assert pioneer.genome.values[1] == 4.8


def test_behavioral_security_descriptor_from_trajectory():
    from evolab.instrumentation import (
        BehavioralSecurityDescriptor,
        Observation,
        FaultCategory,
        ViolationSeverity,
    )

    t1 = Observation(
        target_name="ServiceA",
        return_value="init",
        duration_ms=2.0,
        fault_category=FaultCategory.NORMAL_SUCCESS,
    )
    t2 = Observation(
        target_name="ServiceA",
        return_value="drift",
        duration_ms=4.0,
        fault_category=FaultCategory.LOGIC_DEVIATION,
        state_diff={"canary": "mutated"},
    )
    t3 = Observation(
        target_name="ServiceA",
        return_value="crash",
        duration_ms=10.0,
        fault_category=FaultCategory.STATE_CORRUPTION,
        state_diff={"canary": "smashed", "arena": "bleed"},
    )

    desc = BehavioralSecurityDescriptor.from_trajectory([t1, t2, t3], baseline_duration_ms=2.0)
    assert desc.fault_transitions == [
        ("NORMAL_SUCCESS", "LOGIC_DEVIATION"),
        ("LOGIC_DEVIATION", "STATE_CORRUPTION"),
    ]
    assert desc.max_severity == ViolationSeverity.HIGH
    assert "NoStateCorruptionInvariant" in desc.violated_invariants
    assert sorted(desc.state_diff_keys) == ["arena", "canary"]
    assert desc.temporal_expansion_ratio > 1.5

    # Roundtrip serialization
    d_dict = desc.serialize()
    recovered = BehavioralSecurityDescriptor.from_dict(d_dict)
    assert recovered.fault_transitions == desc.fault_transitions
    assert recovered.state_diff_keys == desc.state_diff_keys
    assert recovered.max_severity == desc.max_severity


def test_behavioral_security_descriptor_coordinate_blindness():
    from evolab.instrumentation import (
        BehavioralSecurityDescriptor,
        Observation,
        FaultCategory,
    )

    # Two completely distinct genomes / representation spaces
    # Genome 1: Float vector (numeric)
    # Genome 2: AST code snippet (symbolic)
    # Both execute and produce identical telemetry
    obs1 = [
        Observation(
            target_name="Target_Numeric",
            fault_category=FaultCategory.STATE_CORRUPTION,
            state_diff={"isolated_state": [1, 2]},
            duration_ms=5.0,
        )
    ]
    obs2 = [
        Observation(
            target_name="Target_SymbolicAST",
            fault_category=FaultCategory.STATE_CORRUPTION,
            state_diff={"isolated_state": ["alpha", "beta"]},
            duration_ms=5.0,
        )
    ]

    d1 = BehavioralSecurityDescriptor.from_trajectory(obs1, baseline_duration_ms=5.0)
    d2 = BehavioralSecurityDescriptor.from_trajectory(obs2, baseline_duration_ms=5.0)

    # Distance must be 0: pure behavioral identity, totally blind to genome representations
    assert d1.distance(d2) == 0.0


def test_cross_target_transfer_memory_behavioral_motif():
    from evolab.instrumentation import (
        CrossTargetTransferMemory,
        Observation,
        FaultCategory,
    )

    mem = CrossTargetTransferMemory()

    # Learn on Target A (numeric target)
    traj_a = [
        Observation(target_name="A", fault_category=FaultCategory.NORMAL_SUCCESS),
        Observation(
            target_name="A",
            fault_category=FaultCategory.STATE_CORRUPTION,
            state_diff={"memory_guard": "clobbered"},
        ),
    ]
    desc_a = mem.record_behavioral_motif(
        name="canary_clobber_motif",
        trajectory=traj_a,
        source_target="Target_Numeric_A",
    )
    assert desc_a is not None
    assert len(mem.behavioral_motifs) == 1

    # Candidate in Target B (symbolic instructions) triggers matching behavioral transition
    traj_b_match = [
        Observation(target_name="B", fault_category=FaultCategory.NORMAL_SUCCESS),
        Observation(
            target_name="B",
            fault_category=FaultCategory.STATE_CORRUPTION,
            state_diff={"memory_guard": "clobbered"},
        ),
    ]
    bonus_match = mem.compute_behavioral_affinity(traj_b_match)
    assert bonus_match > 10.0  # High affinity bonus

    # Candidate in Target B with completely unrelated behavior
    traj_b_clean = [
        Observation(target_name="B", fault_category=FaultCategory.NORMAL_SUCCESS),
        Observation(target_name="B", fault_category=FaultCategory.NORMAL_SUCCESS),
    ]
    bonus_clean = mem.compute_behavioral_affinity(traj_b_clean)
    assert bonus_clean == 0.0  # Zero bonus



