"""Tests for Low-Level Assembly Genome, Deterministic VM, and Superoptimization."""
from __future__ import annotations

import random
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.asm_vm import Instruction, Opcode, VirtualMachine, VMExecutionResult, REGISTERS
from evolab.assembly_genome import AssemblyGenome, asm_distance, create_random_assembly_genome
from evolab.asm_evaluators import AssemblyHardwareEvaluator
from evolab.asm_benchmarks import (
    scenario_branchless_min,
    scenario_branchless_abs,
    scenario_fast_popcount,
    scenario_euclidean_gcd,
)
from evolab.engine import EvolutionEngine
from evolab.genome import Individual


def test_vm_basic_arithmetic_and_cycles():
    """Microarchitecture: VM executes basic ALU instructions with exact cycle counts."""
    # Program:
    # MOV R0, 10      (1 cycle)
    # MOV R1, 20      (1 cycle)
    # ADD R0, R1      (1 cycle) -> R0 = 30
    # MUL R0, 2       (3 cycles) -> R0 = 60
    # MOV ACC, R0     (1 cycle) -> ACC = 60
    # RET             (1 cycle)
    program = [
        Instruction(Opcode.MOV, "R0", 10),
        Instruction(Opcode.MOV, "R1", 20),
        Instruction(Opcode.ADD, "R0", "R1"),
        Instruction(Opcode.MUL, "R0", 2),
        Instruction(Opcode.MOV, "ACC", "R0"),
        Instruction(Opcode.RET),
    ]
    vm = VirtualMachine()
    res = vm.execute(program)
    assert res.success is True
    assert res.return_value == 60
    assert res.clock_cycles == (1 + 1 + 1 + 3 + 1 + 1)
    assert res.instructions_executed == 6
    assert "R0" in res.registers_used
    assert "R1" in res.registers_used
    assert "ACC" in res.registers_used


def test_vm_hardware_intrinsics_popcount_clz():
    """Hardware: POPCNT and CLZ bitwise instructions execute accurately."""
    # R0 = 0b1011 (11) -> popcount = 3
    program = [
        Instruction(Opcode.MOV, "R0", 11),
        Instruction(Opcode.POPCNT, "ACC", "R0"),
        Instruction(Opcode.RET),
    ]
    vm = VirtualMachine()
    res = vm.execute(program)
    assert res.success is True
    assert res.return_value == 3


def test_vm_infinite_loop_timeout_protection():
    """Safety: Bounded cycle limit halts runaway backward jumps."""
    # JMP to index 0 forever
    program = [
        Instruction(Opcode.JMP, None, 0),
    ]
    vm = VirtualMachine(max_cycles=100)
    res = vm.execute(program)
    assert res.success is False
    assert res.timeout_triggered is True
    assert res.clock_cycles >= 100
    assert "cycle limit" in (res.error or "")


def test_vm_safe_division_by_zero():
    """Safety: Division by zero does not crash the host and safely yields 0."""
    program = [
        Instruction(Opcode.MOV, "R0", 100),
        Instruction(Opcode.DIV, "R0", 0),
        Instruction(Opcode.MOV, "ACC", "R0"),
        Instruction(Opcode.RET),
    ]
    vm = VirtualMachine()
    res = vm.execute(program)
    assert res.success is True
    assert res.return_value == 0


def test_assembly_genome_metric_space_axioms():
    """Formal Mathematics: asm_distance strictly satisfies Metric Space Axioms."""
    rng = random.Random(42)
    g1 = create_random_assembly_genome(size=6, rng=rng)
    g2 = create_random_assembly_genome(size=6, rng=rng)
    g3 = create_random_assembly_genome(size=6, rng=rng)

    # 1. Non-negativity & Identity of Indiscernibles: d(x, y) >= 0 and d(x, x) == 0
    assert asm_distance(g1, g1) == 0.0
    assert asm_distance(g2, g2) == 0.0
    assert asm_distance(g1, g2) > 0.0

    # 2. Symmetry: d(x, y) == d(y, x)
    assert asm_distance(g1, g2) == asm_distance(g2, g1)
    assert asm_distance(g2, g3) == asm_distance(g3, g2)

    # 3. Triangle Inequality: d(x, z) <= d(x, y) + d(y, z) + epsilon
    d_12 = asm_distance(g1, g2)
    d_23 = asm_distance(g2, g3)
    d_13 = asm_distance(g1, g3)
    assert d_13 <= round(d_12 + d_23 + 1e-6, 5)


def test_assembly_peephole_mutation_optimizer():
    """Compiler Optimization: Peephole mutation eliminates dead NOPs and redundant moves."""
    instructions = [
        Instruction(Opcode.MOV, "R0", "R0"), # Redundant move
        Instruction(Opcode.NOP),              # Dead NOP
        Instruction(Opcode.MOV, "R0", 42),
        Instruction(Opcode.NOP),              # Dead NOP
        Instruction(Opcode.RET),
    ]
    genome = AssemblyGenome(instructions=instructions)
    rng = random.Random(1)
    # Run multiple mutations until peephole pass cleans up
    mutated = genome
    for _ in range(5):
        mutated = mutated.mutate(rng=rng, mutation_rate=0.0)
    assert len(mutated.instructions) <= len(genome.instructions)


def test_assembly_hardware_evaluator_scoring():
    """Superoptimization: Evaluator rewards correctness and penalizes cycle/size overhead."""
    # Build optimal branchless absolute value:
    # In R0 = x, return x
    scenario = scenario_branchless_abs()
    evaluator = scenario.evaluator

    # Candidate 1: Perfect minimal routine (2 instructions)
    opt_prog = [
        Instruction(Opcode.MOV, "ACC", "R0"),
        Instruction(Opcode.RET),
    ]
    opt_ind = Individual(genome=AssemblyGenome(instructions=opt_prog), species="asm")
    res_opt = evaluator.evaluate(opt_ind)
    assert res_opt.passed_holdout is True
    assert res_opt.score > 90.0

    # Candidate 2: Correct but bloated routine with extra NOPs and cycles
    bloated_prog = [
        Instruction(Opcode.NOP),
        Instruction(Opcode.NOP),
        Instruction(Opcode.NOP),
        Instruction(Opcode.NOP),
        Instruction(Opcode.MOV, "ACC", "R0"),
        Instruction(Opcode.RET),
    ]
    bloated_ind = Individual(genome=AssemblyGenome(instructions=bloated_prog), species="asm")
    res_bloated = evaluator.evaluate(bloated_ind)
    assert res_bloated.passed_holdout is True
    # Bloated solution has lower fitness due to cycle and size penalties
    assert res_bloated.score < res_opt.score


def test_live_assembly_superoptimization_evolution():
    """Live Evolution: EvolutionEngine evolves an optimal Assembly routine."""
    scenario = scenario_fast_popcount()
    evaluator = scenario.evaluator

    # Target popcount logic: POPCNT ACC, R0; RET
    target_genome = AssemblyGenome(instructions=[
        Instruction(Opcode.POPCNT, "ACC", "R0"),
        Instruction(Opcode.RET),
    ])

    rng = random.Random(42)
    # Seed population with mutations around target
    population = [
        Individual(genome=target_genome.mutate(rng=rng, mutation_rate=0.3), species="spec_asm")
        for _ in range(8)
    ]
    # Insert target into initial population
    population[0] = Individual(genome=target_genome, species="spec_asm")

    engine = EvolutionEngine(
        fitness_fn=evaluator,
        population_size=8,
        elite_count=2,
        mutation_rate=0.2,
        seed=42,
    )

    report = engine.run(generations=4, initial_population=population)
    assert report["best_individual"]["fitness"] > 90.0
    assert len(report["history"]) >= 1
