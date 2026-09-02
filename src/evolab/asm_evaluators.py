"""Low-Level Assembly and Hardware Superoptimization Evaluators."""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .asm_vm import VirtualMachine, VMExecutionResult
from .assembly_genome import AssemblyGenome
from .evaluators import Evaluator, FitnessResult
from .genome import EvolabGenome, Individual


@dataclass
class AssemblyHardwareEvaluator(Evaluator):
    """Evaluates AssemblyGenome programs using low-level hardware metrics.

    Fitness calculation balances functional correctness against physical resource utilization:
    Fitness = Correctness_Score - (w_cycles * Clock_Cycles) - (w_size * Byte_Size) - (w_reg * Registers_Used)
    """

    test_cases: Sequence[tuple[Sequence[int], int]]
    holdout_cases: Sequence[tuple[Sequence[int], int]] = field(default_factory=list)
    w_correctness: float = 80.0
    w_cycles: float = 0.05
    w_code_size: float = 0.15
    w_registers: float = 0.50
    max_cycles: int = 1500

    def __post_init__(self) -> None:
        if not self.test_cases:
            raise ValueError("AssemblyHardwareEvaluator requires at least one test case")

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def evaluate(
        self,
        target: EvolabGenome | Individual | list[float],
        context: dict[str, Any] | None = None,
    ) -> FitnessResult:
        t0 = time.perf_counter()
        genome: Any = target.genome if isinstance(target, Individual) else target

        if not isinstance(genome, AssemblyGenome):
            return FitnessResult(
                score=0.0,
                passed_holdout=False,
                sub_scores={},
                artifacts={"error": f"Expected AssemblyGenome, got {type(genome)}"},
            )

        vm = VirtualMachine(max_cycles=self.max_cycles)
        total_tests = len(self.test_cases)
        passed_tests = 0
        total_cycles = 0
        total_memory_accesses = 0
        all_registers_used: set[str] = set()
        details: list[str] = []

        # Run primary test cases
        for args, expected in self.test_cases:
            res: VMExecutionResult = vm.execute(genome.instructions, initial_args=args)
            total_cycles += res.clock_cycles
            total_memory_accesses += res.memory_access_count
            all_registers_used.update(res.registers_used)

            if not res.success or res.timeout_triggered:
                details.append(f"Args {args}: VM Error/Timeout ({res.error})")
                continue

            # Compare 32-bit integer result
            expected_32 = expected & 0xFFFFFFFF
            actual_32 = res.return_value & 0xFFFFFFFF
            if actual_32 == expected_32:
                passed_tests += 1
            else:
                details.append(f"Args {args}: Expected {expected_32}, got {actual_32}")

        correctness_ratio = passed_tests / total_tests
        mean_cycles = total_cycles / max(1, total_tests)
        code_size_bytes = sum(ins.byte_size() for ins in genome.instructions)
        reg_pressure = len(all_registers_used)

        # Correctness base score (0.0 to 80.0)
        base_score = correctness_ratio * self.w_correctness

        # Low-level hardware penalties
        cycle_penalty = min(20.0, mean_cycles * self.w_cycles)
        size_penalty = min(15.0, code_size_bytes * self.w_code_size)
        reg_penalty = min(10.0, reg_pressure * self.w_registers)

        # Bonus for passing 100% of test cases
        solved_bonus = 20.0 if passed_tests == total_tests else 0.0

        raw_fitness = base_score + solved_bonus - cycle_penalty - size_penalty - reg_penalty
        final_fitness = max(0.0, min(100.0, round(raw_fitness, 4)))

        # Validate holdout generalization
        holdout_passed = True
        if self.holdout_cases and passed_tests == total_tests:
            for h_args, h_expected in self.holdout_cases:
                h_res = vm.execute(genome.instructions, initial_args=h_args)
                if not h_res.success or (h_res.return_value & 0xFFFFFFFF) != (h_expected & 0xFFFFFFFF):
                    holdout_passed = False
                    break

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return FitnessResult(
            score=final_fitness,
            passed_holdout=holdout_passed if self.holdout_cases else (passed_tests == total_tests),
            sub_scores={
                "correctness": round(correctness_ratio, 4),
                "mean_cycles": round(mean_cycles, 2),
                "code_size_bytes": float(code_size_bytes),
                "registers_used": float(reg_pressure),
            },
            artifacts={
                "tests_passed": passed_tests,
                "total_tests": total_tests,
                "failure_details": details[:3],
            },
            evaluation_time_ms=duration_ms,
        )
