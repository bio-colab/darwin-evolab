"""
Unit test suite for Quantum Optimal Control and Pulse Evolution in Darwin-Evolab.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from evolab.engine import EngineConfig, EvolutionEngine, SpeciationConfig
from evolab.assembly_genome import AssemblyGenome, Instruction, Opcode
from evolab.genome import FloatGenome
from evolab.quantum import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    average_gate_fidelity,
    cnot,
    cz,
    hadamard,
    pauli_x,
    pauli_y,
    pauli_z,
    phase_gate,
    phase_invariant_fidelity,
    simulate_multi_axis_single_qubit,
    simulate_two_qubit_entangling_evolution,
    QuantumPulseEvaluator,
)


def test_quantum_constants_and_unitarity():
    """Verify unitarity U^dagger @ U = I for all canonical quantum gates."""
    gates = [pauli_x(), pauli_y(), pauli_z(), hadamard(), phase_gate(), cnot(), cz()]
    for g in gates:
        dim = g.shape[0]
        prod = g.conj().T @ g
        np.testing.assert_allclose(prod, np.eye(dim, dtype=complex), atol=1e-10)


def test_average_gate_fidelity_axioms():
    """Verify fidelity properties: self-fidelity is exactly 1.0 and bounds hold."""
    h = hadamard()
    x = pauli_x()

    assert math.isclose(average_gate_fidelity(h, h), 1.0, rel_tol=1e-7)
    assert math.isclose(average_gate_fidelity(x, x), 1.0, rel_tol=1e-7)

    # Orthogonal states or different gates have strictly lower fidelity
    f_hx = average_gate_fidelity(h, x)
    assert 0.0 <= f_hx < 0.90


def test_hadamard_multi_axis_vs_single_axis_reachability():
    """Physics validation: Single-axis X rotation is capped at 2/3 (0.6667),

    while multi-axis (X + Z) reaches fidelity 1.0.
    """
    dt = 0.1
    n_steps = 10
    h_target = hadamard()

    # 1. Single-axis drive (omega_x only): maximum possible trace is sqrt(2), fidelity is capped at 4/6 = 0.6667
    max_single_axis_fid = 0.0
    for w in np.linspace(0.1, 20.0, 100):
        u_single = simulate_multi_axis_single_qubit([w] * n_steps, [0.0] * n_steps, [0.0] * n_steps, dt=dt)
        max_single_axis_fid = max(max_single_axis_fid, average_gate_fidelity(h_target, u_single))
    assert max_single_axis_fid <= 0.6667 + 1e-4

    # 2. Multi-axis drive (rotations around (X + Z)/sqrt(2)): achieves exactly 1.0
    amp = np.pi / (n_steps * dt * np.sqrt(2.0))
    u_multi = simulate_multi_axis_single_qubit([amp] * n_steps, [0.0] * n_steps, [amp] * n_steps, dt=dt)
    multi_fid = average_gate_fidelity(h_target, u_multi)
    assert multi_fid > 0.999


def test_cnot_entanglement_reachability():
    """Physics validation: Two-qubit ZZ coupling introduces an entangling Hamiltonian

    that enables CNOT synthesis beyond uncoupled product state limits.
    """
    dt = 0.1
    cnot_target = cnot()

    # Zero drive baseline (identity)
    u_zero = simulate_two_qubit_entangling_evolution([0.0]*10, [0.0]*10, [0.0]*10, [0.0]*10, [0.0]*10, dt=dt)
    f_zero = average_gate_fidelity(cnot_target, u_zero)
    assert math.isclose(f_zero, 0.4000, abs_tol=1e-4)

    # Driven with ZZ interaction: breaks zero-coupling baseline
    u_coupled = simulate_two_qubit_entangling_evolution([1.0]*10, [0.0]*10, [1.0]*10, [0.0]*10, [2.5]*10, dt=dt)
    f_coupled = average_gate_fidelity(cnot_target, u_coupled)
    assert not math.isclose(f_coupled, 0.4000, abs_tol=1e-3)


def test_quantum_pulse_evaluator_assembly_and_float_support():
    """Evaluator validation: QuantumPulseEvaluator evaluates both AssemblyGenome and FloatGenome."""
    evaluator = QuantumPulseEvaluator(pauli_x(), num_timesteps=5, multi_axis=True)

    # 1. FloatGenome
    fg = FloatGenome([3.14159 / (5 * 0.1)] * 5 + [0.0] * 10)
    res_fg = evaluator.evaluate(fg)
    assert res_fg.sub_scores["fidelity"] > 0.95
    assert res_fg.score > 80.0

    # 2. AssemblyGenome
    amp = 3.14159 / (5 * 0.1)
    ag = AssemblyGenome([
        Instruction(op=Opcode.MOV, dst="R0", imm=amp) for _ in range(5)
    ])
    res_ag = evaluator.evaluate(ag)
    assert res_ag.sub_scores["fidelity"] > 0.95
    assert res_ag.score > 80.0


def test_live_quantum_pulse_evolution_engine():
    """Engine integration: EvolutionEngine evolves a high-fidelity Pauli-X pulse sequence."""
    evaluator = QuantumPulseEvaluator(pauli_x(), num_timesteps=10, multi_axis=False)

    cfg = EngineConfig(
        population_size=30,
        genome_size=10,
        mutation_rate=0.35,
        early_stop_fitness=98.0,
        speciation=SpeciationConfig(enabled=True, threshold=0.5),
    )

    engine = EvolutionEngine(config=cfg, fitness_fn=evaluator)
    report = engine.run(generations=25)

    best_fidelity = evaluator.evaluate(engine.best_ever.genome).sub_scores["fidelity"]
    assert best_fidelity > 0.90
    assert engine.best_ever.fitness > 80.0
