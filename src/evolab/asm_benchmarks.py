"""Low-Level Assembly Superoptimization Benchmark Scenarios."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .asm_evaluators import AssemblyHardwareEvaluator


@dataclass
class AssemblyScenario:
    """Benchmark scenario definition for assembly superoptimization."""
    name: str
    description: str
    evaluator: AssemblyHardwareEvaluator
    reference_solution_cycles: int
    optimal_size_bytes: int


def scenario_branchless_min() -> AssemblyScenario:
    """Calculates min(a, b) across signed integer inputs."""
    test_cases = [
        ((10, 20), 10),
        ((50, 5), 5),
        ((0, 0), 0),
        ((100, 100), 100),
        ((12, 45), 12),
        ((99, 1), 1),
    ]
    holdout_cases = [
        ((7, 9), 7),
        ((88, 33), 33),
        ((250, 150), 150),
    ]
    return AssemblyScenario(
        name="branchless_min",
        description="Calculate min(R0, R1) in minimal clock cycles and code size",
        evaluator=AssemblyHardwareEvaluator(test_cases=test_cases, holdout_cases=holdout_cases),
        reference_solution_cycles=4,
        optimal_size_bytes=6,
    )


def scenario_branchless_abs() -> AssemblyScenario:
    """Calculates abs(x) for integer x in R0."""
    test_cases = [
        ((5,), 5),
        ((0,), 0),
        ((120,), 120),
        ((42,), 42),
        ((1,), 1),
    ]
    holdout_cases = [
        ((77,), 77),
        ((999,), 999),
    ]
    return AssemblyScenario(
        name="branchless_abs",
        description="Compute absolute value of R0",
        evaluator=AssemblyHardwareEvaluator(test_cases=test_cases, holdout_cases=holdout_cases),
        reference_solution_cycles=3,
        optimal_size_bytes=4,
    )


def scenario_fast_popcount() -> AssemblyScenario:
    """Counts set bits of 32-bit integer in R0."""
    test_cases = [
        ((0,), 0),
        ((1,), 1),
        ((3,), 2),
        ((7,), 3),
        ((15,), 4),
        ((255,), 8),
        ((1023,), 10),
    ]
    holdout_cases = [
        ((31,), 5),
        ((63,), 6),
        ((4095,), 12),
    ]
    return AssemblyScenario(
        name="fast_popcount",
        description="Count number of set bits (popcount) in R0",
        evaluator=AssemblyHardwareEvaluator(test_cases=test_cases, holdout_cases=holdout_cases),
        reference_solution_cycles=2,
        optimal_size_bytes=3,
    )


def scenario_euclidean_gcd() -> AssemblyScenario:
    """Computes greatest common divisor gcd(R0, R1)."""
    test_cases = [
        ((12, 8), 4),
        ((15, 5), 5),
        ((100, 25), 25),
        ((17, 13), 1),
        ((48, 18), 6),
    ]
    holdout_cases = [
        ((60, 24), 12),
        ((81, 27), 27),
    ]
    return AssemblyScenario(
        name="euclidean_gcd",
        description="Compute GCD of R0 and R1 via Euclidean reduction",
        evaluator=AssemblyHardwareEvaluator(test_cases=test_cases, holdout_cases=holdout_cases),
        reference_solution_cycles=15,
        optimal_size_bytes=10,
    )


def scenario_fibonacci_n() -> AssemblyScenario:
    """Computes n-th Fibonacci number."""
    test_cases = [
        ((0,), 0),
        ((1,), 1),
        ((2,), 1),
        ((3,), 2),
        ((4,), 3),
        ((5,), 5),
        ((6,), 8),
    ]
    holdout_cases = [
        ((7,), 13),
        ((8,), 21),
    ]
    return AssemblyScenario(
        name="fibonacci_n",
        description="Compute n-th Fibonacci number in R0",
        evaluator=AssemblyHardwareEvaluator(test_cases=test_cases, holdout_cases=holdout_cases),
        reference_solution_cycles=20,
        optimal_size_bytes=12,
    )


ASM_SCENARIOS: dict[str, Callable[[], AssemblyScenario]] = {
    "branchless_min": scenario_branchless_min,
    "branchless_abs": scenario_branchless_abs,
    "fast_popcount": scenario_fast_popcount,
    "euclidean_gcd": scenario_euclidean_gcd,
    "fibonacci_n": scenario_fibonacci_n,
}


def get_all_asm_scenarios() -> list[AssemblyScenario]:
    return [factory() for factory in ASM_SCENARIOS.values()]
