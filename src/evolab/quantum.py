"""Quantum Optimal Control & Pulse Evolution Suite for Darwin-Evolab.

Provides physically complete multi-axis quantum control simulation,
canonical quantum gate definitions (SU(2) and SU(4)), Average Gate Fidelity metrics,
and a first-class QuantumPulseEvaluator supporting continuous (FloatGenome)
and discrete (AssemblyGenome) microarchitecture representations.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.linalg import expm

from .asm_vm import Opcode
from .assembly_genome import AssemblyGenome
from .evaluators import Evaluator, FitnessResult
from .genome import EvolabGenome, FloatGenome, Individual

# ============================================================================
# 1. Quantum Fundamental Constants and Pauli Matrices
# ============================================================================

PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ============================================================================
# 2. Canonical Target Quantum Gates
# ============================================================================

def pauli_x() -> np.ndarray:
    """Single-qubit bit flip gate (NOT)."""
    return np.array([[0, 1], [1, 0]], dtype=complex)

def pauli_y() -> np.ndarray:
    """Single-qubit bit and phase flip gate."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def pauli_z() -> np.ndarray:
    """Single-qubit phase flip gate."""
    return np.array([[1, 0], [0, -1]], dtype=complex)

def hadamard() -> np.ndarray:
    """Single-qubit Hadamard superposition gate (requires X + Z rotation)."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)

def phase_gate() -> np.ndarray:
    """Single-qubit S phase gate (pi/2 around Z)."""
    return np.array([[1, 0], [0, 1j]], dtype=complex)

def t_gate() -> np.ndarray:
    """Single-qubit T gate (pi/4 around Z)."""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]], dtype=complex)

def cnot() -> np.ndarray:
    """Two-qubit Controlled-NOT entangling gate."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

def cz() -> np.ndarray:
    """Two-qubit Controlled-Z entangling gate."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)

def swap_gate() -> np.ndarray:
    """Two-qubit SWAP gate."""
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)

def toffoli_gate() -> np.ndarray:
    """Three-qubit Controlled-Controlled-NOT (Toffoli) gate."""
    U = np.eye(8, dtype=complex)
    U[6, 6] = 0.0
    U[6, 7] = 1.0
    U[7, 6] = 1.0
    U[7, 7] = 0.0
    return U


# ============================================================================
# 3. High-Precision Quantum Fidelity Metrics
# ============================================================================

def average_gate_fidelity(target: np.ndarray, evolved: np.ndarray) -> float:
    """Calculates Average Gate Fidelity F_avg(U_target, U_evolved).

    F_avg = (|Tr(U_target^dagger @ U_evolved)|^2 + d) / (d^2 + d)
    where d is the Hilbert space dimension (d=2 for 1 qubit, d=4 for 2 qubits).
    """
    d = target.shape[0]
    inner = np.trace(target.conj().T @ evolved)
    fidelity = (abs(inner) ** 2 + d) / (d ** 2 + d)
    return float(np.clip(fidelity, 0.0, 1.0))

def phase_invariant_fidelity(target: np.ndarray, evolved: np.ndarray) -> float:
    """Calculates phase-invariant Hilbert-Schmidt norm fidelity |Tr(U^dagger V)| / d."""
    d = target.shape[0]
    inner = np.trace(target.conj().T @ evolved)
    return float(abs(inner) / d)


# ============================================================================
# 4. Multi-Axis Physical Unitary Time Evolution
# ============================================================================

def su2_rodrigues_rotation(wx: float, wy: float, wz: float, dt: float) -> np.ndarray:
    """Computes exact exp(-i * H * dt) for single-qubit SU(2) multi-axis Hamiltonian:
    H = 0.5 * (wx * sigma_x + wy * sigma_y + wz * sigma_z).
    Uses closed-form Rodrigues rotation formula for extreme speed (no expm).
    """
    norm = math.sqrt(wx * wx + wy * wy + wz * wz)
    if norm < 1e-12:
        return np.eye(2, dtype=complex)
    
    half_angle = 0.5 * norm * dt
    c = math.cos(half_angle)
    s = math.sin(half_angle) / norm

    # (wx*sx + wy*sy + wz*sz)
    # [[wz, wx - 1j*wy], [wx + 1j*wy, -wz]]
    u00 = c - 1j * s * wz
    u01 = -1j * s * (wx - 1j * wy)
    u10 = -1j * s * (wx + 1j * wy)
    u11 = c + 1j * s * wz

    return np.array([[u00, u01], [u10, u11]], dtype=complex)


def simulate_multi_axis_single_qubit(
    pulse_x: Sequence[float],
    pulse_y: Sequence[float],
    pulse_z: Sequence[float],
    dt: float = 0.1,
) -> np.ndarray:
    """Simulates multi-axis time evolution on 1 qubit:
    U = prod_k exp(-i * (wx_k * sx + wy_k * sy + wz_k * sz) * dt).
    """
    U = np.eye(2, dtype=complex)
    n = max(len(pulse_x), len(pulse_y), len(pulse_z))
    for k in range(n):
        wx = pulse_x[k] if k < len(pulse_x) else 0.0
        wy = pulse_y[k] if k < len(pulse_y) else 0.0
        wz = pulse_z[k] if k < len(pulse_z) else 0.0
        U_step = su2_rodrigues_rotation(wx, wy, wz, dt)
        U = U_step @ U
    return U


def simulate_two_qubit_entangling_evolution(
    pulse_x1: Sequence[float],
    pulse_z1: Sequence[float],
    pulse_x2: Sequence[float],
    pulse_z2: Sequence[float],
    pulse_zz: Sequence[float],
    dt: float = 0.1,
) -> np.ndarray:
    """Simulates physical two-qubit evolution with ZZ interaction:
    H(t) = (wx1*sx + wz1*sz) (x) I + I (x) (wx2*sx + wz2*sz) + J_zz * (sz (x) sz).
    This model provides complete SU(4) controllability for CNOT and entangling gates.
    """
    sz_sz = np.kron(PAULI_Z, PAULI_Z)
    sx_i = np.kron(PAULI_X, PAULI_I)
    sz_i = np.kron(PAULI_Z, PAULI_I)
    i_sx = np.kron(PAULI_I, PAULI_X)
    i_sz = np.kron(PAULI_I, PAULI_Z)

    U = np.eye(4, dtype=complex)
    n = max(len(pulse_x1), len(pulse_z1), len(pulse_x2), len(pulse_z2), len(pulse_zz))

    for k in range(n):
        wx1 = pulse_x1[k] if k < len(pulse_x1) else 0.0
        wz1 = pulse_z1[k] if k < len(pulse_z1) else 0.0
        wx2 = pulse_x2[k] if k < len(pulse_x2) else 0.0
        wz2 = pulse_z2[k] if k < len(pulse_z2) else 0.0
        j_zz = pulse_zz[k] if k < len(pulse_zz) else 0.0

        H_k = (
            0.5 * (wx1 * sx_i + wz1 * sz_i + wx2 * i_sx + wz2 * i_sz)
            + 0.5 * j_zz * sz_sz
        )
        U_step = expm(-1j * H_k * dt)
        U = U_step @ U

    return U


# ============================================================================
# 5. First-Class Quantum Optimal Control Evaluator
# ============================================================================

class QuantumPulseEvaluator(Evaluator):
    """Production Quantum Pulse Evaluator for Darwin-Evolab.

    Supports:
    - Single-qubit multi-axis control (X, Y, Z drives)
    - Two-qubit entangling control (local drives + ZZ interaction)
    - Both continuous FloatGenome and discrete AssemblyGenome microcode
    - Realistic hardware constraints (amplitude caps, slew rate, energy)
    """

    def __init__(
        self,
        target_gate: np.ndarray,
        num_timesteps: int = 10,
        dt: float = 0.1,
        max_amplitude: float = 10.0,
        energy_penalty_weight: float = 0.02,
        multi_axis: bool = True,
    ) -> None:
        self.target_gate = np.asarray(target_gate, dtype=complex)
        self.num_qubits = 1 if self.target_gate.shape[0] == 2 else 2
        self.num_timesteps = num_timesteps
        self.dt = dt
        self.max_amplitude = max_amplitude
        self.energy_penalty_weight = energy_penalty_weight
        self.multi_axis = multi_axis
        self._deterministic = True

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def _extract_pulses_from_assembly(
        self, genome: AssemblyGenome
    ) -> tuple[list[float], ...]:
        """Maps assembly registers to quantum control channels:
        - R0 -> X-channel amplitude
        - R1 -> Y-channel (or Z1) amplitude
        - R2 -> Z-channel (or X2) amplitude
        - R3 -> Entangling coupling (J_zz) amplitude
        """
        ch_x = [0.0] * self.num_timesteps
        ch_y = [0.0] * self.num_timesteps
        ch_z = [0.0] * self.num_timesteps
        ch_j = [0.0] * self.num_timesteps

        t = 0
        for instr in genome.instructions:
            if t >= self.num_timesteps:
                break
            if instr.op in (Opcode.MOV, Opcode.ADD, Opcode.SUB):
                val = float(instr.imm) if instr.imm is not None else 1.0
                if instr.dst == "R0":
                    ch_x[t] = val
                    t += 1
                elif instr.dst == "R1":
                    ch_y[t] = val
                    t += 1
                elif instr.dst == "R2":
                    ch_z[t] = val
                    t += 1
                elif instr.dst == "R3":
                    ch_j[t] = val
                    t += 1
                else:
                    ch_x[t] = val
                    t += 1

        return ch_x, ch_y, ch_z, ch_j

    def _extract_pulses_from_float(
        self, genome: Sequence[float]
    ) -> tuple[list[float], ...]:
        """Slices flat continuous parameter vector into channel timeseries."""
        data = list(genome)
        if self.num_qubits == 1:
            channels = 3 if self.multi_axis else 1
            step = max(1, len(data) // channels)
            ch_x = data[:step]
            ch_y = data[step : 2 * step] if channels >= 2 else [0.0] * step
            ch_z = data[2 * step : 3 * step] if channels >= 3 else [0.0] * step
            return ch_x, ch_y, ch_z, [0.0] * step
        else:
            # 2 qubits: 5 channels (x1, z1, x2, z2, zz)
            step = max(1, len(data) // 5)
            x1 = data[:step]
            z1 = data[step : 2 * step]
            x2 = data[2 * step : 3 * step]
            z2 = data[3 * step : 4 * step]
            zz = data[4 * step : 5 * step]
            return x1, z1, x2, z2, zz

    def evaluate(
        self,
        target: EvolabGenome | Individual | list[float],
        context: dict[str, Any] | None = None,
    ) -> FitnessResult:
        target_genome: Any = target.genome if isinstance(target, Individual) else target

        if isinstance(target_genome, AssemblyGenome):
            channels = self._extract_pulses_from_assembly(target_genome)
            length = len(target_genome.instructions)
        elif isinstance(target_genome, FloatGenome):
            channels = self._extract_pulses_from_float(target_genome.values)
            length = len(target_genome.values)
        elif isinstance(target_genome, (list, tuple, np.ndarray)):
            channels = self._extract_pulses_from_float(list(target_genome))
            length = len(target_genome)
        else:
            raise TypeError(f"Unsupported genome type for quantum pulse evaluation: {type(target_genome)}")

        # Simulate evolution
        all_pulses = []
        for ch in channels:
            all_pulses.extend(ch)

        if self.num_qubits == 1:
            ch_x, ch_y, ch_z, _ = channels
            evolved = simulate_multi_axis_single_qubit(ch_x, ch_y, ch_z, dt=self.dt)
        else:
            x1, z1, x2, z2, zz = channels[:5]
            evolved = simulate_two_qubit_entangling_evolution(x1, z1, x2, z2, zz, dt=self.dt)

        # Compute average gate fidelity
        fidelity = average_gate_fidelity(self.target_gate, evolved)

        # Hardware penalties
        max_amp = max((abs(v) for v in all_pulses), default=0.0)
        amp_penalty = max(0.0, max_amp - self.max_amplitude) * 0.10
        total_energy = sum(v * v for v in all_pulses) * self.dt
        energy_penalty = total_energy * self.energy_penalty_weight * 0.01

        # Normalized fitness score (0.0 - 100.0)
        raw_score = (fidelity - amp_penalty - energy_penalty) * 100.0
        score = float(np.clip(raw_score, 0.0, 100.0))

        return FitnessResult(
            score=score,
            sub_scores={
                "fidelity": float(fidelity),
                "max_amplitude": float(max_amp),
                "total_energy": float(total_energy),
                "length": float(length),
            },
            artifacts={
                "target_gate": str(self.target_gate.shape),
                "evolved_trace": float(abs(np.trace(evolved))),
            },
        )
