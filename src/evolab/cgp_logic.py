"""
cgp_logic.py — Cartesian Genetic Programming (CGP) & Digital Logic Circuit Synthesis.
Designed for CMOS logic gate synthesis (Half Adder, Full Adder, Comparator, Ripple Adder).

Features:
  - Directed Acyclic Graph (DAG) Netlist representation with standard CMOS cell libraries.
  - Active subgraph extraction (dead-code/neutral gene isolation).
  - Accurate silicon metrics: Equivalent CMOS transistor count and Critical Path Delay (FO4).
  - Truth table verification for ALU arithmetic/logic blocks (Half Adder, Full Adder, Comparator, ALU Slice).
  - Hardware export to Verilog-2001 gate-level netlist for EDA tool compatibility.
  - Multi-objective Pareto frontier extraction (Area vs Delay).
"""
from __future__ import annotations

import enum
import random
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .genome import EvolabGenome


class GateType(str, enum.Enum):
    """Supported logic gate primitives in CMOS standard cell library."""
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NAND = "NAND"
    NOR = "NOR"
    NOT = "NOT"
    XNOR = "XNOR"
    WIRE = "WIRE"


# Equivalent CMOS transistor counts per gate (typical static CMOS layout)
GATE_TRANSISTORS: dict[GateType, int] = {
    GateType.NOT: 2,       # 1 PMOS + 1 NMOS
    GateType.NAND: 4,      # 2 PMOS + 2 NMOS
    GateType.NOR: 4,       # 2 PMOS + 2 NMOS
    GateType.AND: 6,       # NAND (4) + Inverter (2)
    GateType.OR: 6,        # NOR (4) + Inverter (2)
    GateType.XOR: 8,       # Transmission-gate or complex CMOS XOR
    GateType.XNOR: 8,      # Transmission-gate or complex CMOS XNOR
    GateType.WIRE: 0,
}

# Propagation delays normalized to Fan-Out-of-4 (FO4) inverter delay
GATE_DELAYS: dict[GateType, float] = {
    GateType.NOT: 1.0,
    GateType.NAND: 1.0,
    GateType.NOR: 1.2,
    GateType.AND: 1.6,
    GateType.OR: 1.6,
    GateType.XOR: 2.0,
    GateType.XNOR: 2.0,
    GateType.WIRE: 0.0,
}


def eval_gate(gtype: GateType, a: int, b: int) -> int:
    """Evaluates the Boolean output of a gate primitive given binary inputs."""
    if gtype == GateType.AND:
        return a & b
    elif gtype == GateType.OR:
        return a | b
    elif gtype == GateType.XOR:
        return a ^ b
    elif gtype == GateType.NAND:
        return 0 if (a & b) else 1
    elif gtype == GateType.NOR:
        return 0 if (a | b) else 1
    elif gtype == GateType.NOT:
        return 0 if a else 1
    elif gtype == GateType.XNOR:
        return 1 if (a == b) else 0
    elif gtype == GateType.WIRE:
        return a
    return 0


@dataclass
class CGPNode:
    """A single digital logic cell inside the Cartesian grid."""
    gate_type: GateType
    input_a: int
    input_b: int

    def clone(self) -> CGPNode:
        return CGPNode(gate_type=self.gate_type, input_a=self.input_a, input_b=self.input_b)


@dataclass
class CircuitMetrics:
    """Physical hardware and verification metrics for a synthesized circuit."""
    truth_table_accuracy: float
    active_gate_count: int
    transistor_count: int
    critical_path_delay: float
    total_nodes: int
    is_fully_functional: bool


@dataclass
class SwitchingMetrics:
    """Dynamic switching activity and power dissipation metrics for a synthesized circuit."""
    total_toggles: int
    average_wire_activity: float
    dynamic_power_factor: float
    active_wire_count: int
    transitions_measured: int


class CGPGenome(EvolabGenome):
    """Cartesian Genetic Programming (CGP) Netlist Genome."""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        nodes: list[CGPNode],
        output_connections: list[int],
        allowed_gates: Sequence[GateType] | None = None,
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nodes = nodes
        self.output_connections = list(output_connections)
        self.allowed_gates = list(allowed_gates) if allowed_gates is not None else [
            GateType.NAND, GateType.NOR, GateType.AND, GateType.OR, GateType.XOR, GateType.NOT
        ]

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def clone(self) -> CGPGenome:
        return CGPGenome(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            nodes=[n.clone() for n in self.nodes],
            output_connections=list(self.output_connections),
            allowed_gates=list(self.allowed_gates),
        )

    def get_active_nodes(self) -> set[int]:
        """Traces backward from primary outputs to identify all active gates (pruning neutral nodes)."""
        active: set[int] = set()
        queue = list(self.output_connections)
        while queue:
            idx = queue.pop()
            if idx >= self.num_inputs and idx not in active:
                active.add(idx)
                node_idx = idx - self.num_inputs
                node = self.nodes[node_idx]
                queue.append(node.input_a)
                if node.gate_type not in (GateType.NOT, GateType.WIRE):
                    queue.append(node.input_b)
        return active

    def simulate_internal(self, inputs: Sequence[int]) -> list[int]:
        """Simulates circuit propagation and returns state values across all inputs and nodes."""
        values = list(inputs)
        for node in self.nodes:
            val_a = values[node.input_a]
            val_b = values[node.input_b]
            out = eval_gate(node.gate_type, val_a, val_b)
            values.append(out)
        return values

    def simulate(self, inputs: Sequence[int]) -> list[int]:
        """Simulates circuit propagation for a single binary input vector."""
        values = self.simulate_internal(inputs)
        return [values[out_idx] for out_idx in self.output_connections]

    def evaluate_switching_activity(self, input_sequence: Sequence[Sequence[int]]) -> SwitchingMetrics:
        """Evaluates wire transition activity (0->1, 1->0 toggling) and dynamic power consumption."""
        if not input_sequence:
            return SwitchingMetrics(0, 0.0, 0.0, 0, 0)

        active = self.get_active_nodes()
        active_wires = set(range(self.num_inputs)) | active

        fanout = {w: 0 for w in active_wires}
        for idx in active:
            node = self.nodes[idx - self.num_inputs]
            if node.input_a in fanout:
                fanout[node.input_a] += 1
            if node.gate_type not in (GateType.NOT, GateType.WIRE) and node.input_b in fanout:
                fanout[node.input_b] += 1
        for out_conn in self.output_connections:
            if out_conn in fanout:
                fanout[out_conn] += 1

        total_toggles = 0
        weighted_power = 0.0
        prev_state: list[int] | None = None

        for vec in input_sequence:
            curr_state = self.simulate_internal(vec)
            if prev_state is not None:
                for w in active_wires:
                    if curr_state[w] != prev_state[w]:
                        total_toggles += 1
                        load = 1.0 + 0.5 * fanout.get(w, 1)
                        weighted_power += load
            prev_state = curr_state

        transitions_count = max(len(input_sequence) - 1, 1)
        wire_count = max(len(active_wires), 1)
        avg_activity = total_toggles / (transitions_count * wire_count)

        return SwitchingMetrics(
            total_toggles=total_toggles,
            average_wire_activity=round(avg_activity, 4),
            dynamic_power_factor=round(weighted_power, 2),
            active_wire_count=len(active_wires),
            transitions_measured=transitions_count,
        )

    def evaluate_truth_table(self, truth_table: Sequence[tuple[Sequence[int], Sequence[int]]]) -> CircuitMetrics:
        """Evaluates Boolean correctness against a specification truth table and measures hardware cost."""
        total_bits = len(truth_table) * self.num_outputs
        matching_bits = 0

        for inps, expected in truth_table:
            actual = self.simulate(inps)
            for act_bit, exp_bit in zip(actual, expected):
                if act_bit == exp_bit:
                    matching_bits += 1

        accuracy = matching_bits / max(total_bits, 1)
        active = self.get_active_nodes()
        active_gates = 0
        transistor_count = 0
        node_delays = [0.0] * (self.num_inputs + len(self.nodes))

        for i, node in enumerate(self.nodes):
            global_idx = self.num_inputs + i
            if global_idx in active:
                if node.gate_type != GateType.WIRE:
                    active_gates += 1
                transistor_count += GATE_TRANSISTORS[node.gate_type]
                d_in = max(node_delays[node.input_a], node_delays[node.input_b])
                node_delays[global_idx] = d_in + GATE_DELAYS[node.gate_type]

        critical_path_delay = max((node_delays[out_idx] for out_idx in self.output_connections), default=0.0)

        return CircuitMetrics(
            truth_table_accuracy=accuracy,
            active_gate_count=active_gates,
            transistor_count=transistor_count,
            critical_path_delay=critical_path_delay,
            total_nodes=len(self.nodes),
            is_fully_functional=(accuracy == 1.0),
        )

    def to_verilog(self, module_name: str = "cgp_logic_circuit") -> str:
        """Exports the active circuit subgraph into standard synthesizable Verilog-2001 code."""
        active = self.get_active_nodes()
        lines = [
            "// Synthesized by darwin-evolab CGP for Digital Logic Sub-module",
            f"module {module_name} (",
            f"    input  wire [{self.num_inputs-1}:0] in,",
            f"    output wire [{self.num_outputs-1}:0] out",
            ");",
            "",
        ]

        # Declare intermediate wires
        for idx in sorted(active):
            lines.append(f"    wire w_{idx};")
        lines.append("")

        # Instantiate gates
        for idx in sorted(active):
            node = self.nodes[idx - self.num_inputs]
            in_a_name = f"in[{node.input_a}]" if node.input_a < self.num_inputs else f"w_{node.input_a}"
            in_b_name = f"in[{node.input_b}]" if node.input_b < self.num_inputs else f"w_{node.input_b}"

            gt = node.gate_type
            if gt == GateType.NOT:
                lines.append(f"    not g_{idx} (w_{idx}, {in_a_name});")
            elif gt == GateType.AND:
                lines.append(f"    and g_{idx} (w_{idx}, {in_a_name}, {in_b_name});")
            elif gt == GateType.OR:
                lines.append(f"    or g_{idx} (w_{idx}, {in_a_name}, {in_b_name});")
            elif gt == GateType.XOR:
                lines.append(f"    xor g_{idx} (w_{idx}, {in_a_name}, {in_b_name});")
            elif gt == GateType.NAND:
                lines.append(f"    nand g_{idx} (w_{idx}, {in_a_name}, {in_b_name});")
            elif gt == GateType.NOR:
                lines.append(f"    nor g_{idx} (w_{idx}, {in_a_name}, {in_b_name});")
            elif gt == GateType.XNOR:
                lines.append(f"    xnor g_{idx} (w_{idx}, {in_a_name}, {in_b_name});")
            elif gt == GateType.WIRE:
                lines.append(f"    assign w_{idx} = {in_a_name};")

        lines.append("")
        for out_idx, conn in enumerate(self.output_connections):
            src = f"in[{conn}]" if conn < self.num_inputs else f"w_{conn}"
            lines.append(f"    assign out[{out_idx}] = {src};")

        lines.append("endmodule")
        return "\n".join(lines)

    def fingerprint(self) -> str:
        """Stable hash identifying identity of the active circuit subgraph."""
        active = sorted(self.get_active_nodes())
        content = f"{self.num_inputs}:{self.num_outputs}:" + ";".join(
            f"{idx}:{self.nodes[idx - self.num_inputs].gate_type.value}:{self.nodes[idx - self.num_inputs].input_a}:{self.nodes[idx - self.num_inputs].input_b}"
            for idx in active
        ) + ":" + ",".join(str(c) for c in self.output_connections)
        import hashlib
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        """Non-negative symmetric distance metric implementing EvolabGenome contract."""
        return self.distance(other)

    def mutate(self, rng: random.Random | None = None, **kwargs: Any) -> EvolabGenome:
        """Applies point and neutral mutations to the CGP netlist."""
        rate = kwargs.get("mutation_rate", 0.15)
        return mutate_cgp_genome(self, mutation_rate=rate, rng=rng)

    def distance(self, other: EvolabGenome) -> float:
        """Calculates phenotypic distance based on output responses and active topology."""
        if not isinstance(other, CGPGenome):
            return 1.0
        diff = 0
        min_nodes = min(len(self.nodes), len(other.nodes))
        for i in range(min_nodes):
            n1 = self.nodes[i]
            n2 = other.nodes[i]
            if n1.gate_type != n2.gate_type:
                diff += 1
            if n1.input_a != n2.input_a:
                diff += 1
            if n1.input_b != n2.input_b:
                diff += 1
        for o1, o2 in zip(self.output_connections, other.output_connections):
            if o1 != o2:
                diff += 1
        return float(diff)

    def serialize(self) -> dict[str, Any]:
        return {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "active_nodes": list(self.get_active_nodes()),
            "output_connections": list(self.output_connections),
            "node_count": len(self.nodes),
            "nodes": [
                {"gate_type": n.gate_type.value, "input_a": n.input_a, "input_b": n.input_b}
                for n in self.nodes
            ],
        }

    def describe(self) -> dict[str, Any]:
        active = self.get_active_nodes()
        return {
            "node_count": len(active),
            "total_nodes": len(self.nodes),
            "hunk_count": len(self.output_connections),
        }


def create_random_cgp_genome(
    num_inputs: int,
    num_outputs: int,
    num_nodes: int = 15,
    allowed_gates: Sequence[GateType] | None = None,
    rng: random.Random | None = None,
) -> CGPGenome:
    """Generates a random feedforward CGP netlist genome."""
    r = rng or random.Random()
    gates = list(allowed_gates) if allowed_gates else [
        GateType.NAND, GateType.NOR, GateType.AND, GateType.OR, GateType.XOR, GateType.NOT
    ]
    nodes: list[CGPNode] = []
    for i in range(num_nodes):
        max_source = num_inputs + i
        gtype = r.choice(gates)
        in_a = r.randint(0, max_source - 1)
        in_b = r.randint(0, max_source - 1)
        nodes.append(CGPNode(gate_type=gtype, input_a=in_a, input_b=in_b))

    total_sources = num_inputs + num_nodes
    outputs = [r.randint(num_inputs, total_sources - 1) for _ in range(num_outputs)]
    return CGPGenome(num_inputs, num_outputs, nodes, outputs, allowed_gates=gates)


def mutate_cgp_genome(
    genome: CGPGenome,
    mutation_rate: float = 0.15,
    rng: random.Random | None = None,
) -> CGPGenome:
    """Applies point and neutral structural mutations across the CGP DAG."""
    r = rng or random.Random()
    child = genome.clone()
    num_nodes = len(child.nodes)

    for i in range(num_nodes):
        max_source = child.num_inputs + i
        if r.random() < mutation_rate:
            child.nodes[i].gate_type = r.choice(child.allowed_gates)
        if r.random() < mutation_rate:
            child.nodes[i].input_a = r.randint(0, max_source - 1)
        if r.random() < mutation_rate:
            child.nodes[i].input_b = r.randint(0, max_source - 1)

    total_sources = child.num_inputs + num_nodes
    for out_idx in range(child.num_outputs):
        if r.random() < mutation_rate:
            child.output_connections[out_idx] = r.randint(child.num_inputs, total_sources - 1)

    return child


# ===========================================================================
# CMOS Logic Reference Specifications & Truth Tables
# ===========================================================================

# 1. Half Adder: (A, B) -> (Sum, Carry)
HALF_ADDER_TRUTH_TABLE = [
    ((0, 0), (0, 0)),
    ((0, 1), (1, 0)),
    ((1, 0), (1, 0)),
    ((1, 1), (0, 1)),
]

# 2. 1-bit Full Adder with Carry: (A, B, Cin) -> (Sum, Cout)
FULL_ADDER_TRUTH_TABLE = [
    ((0, 0, 0), (0, 0)),
    ((0, 0, 1), (1, 0)),
    ((0, 1, 0), (1, 0)),
    ((0, 1, 1), (0, 1)),
    ((1, 0, 0), (1, 0)),
    ((1, 0, 1), (0, 1)),
    ((1, 1, 0), (0, 1)),
    ((1, 1, 1), (1, 1)),
]

# 3. 2-bit Magnitude Comparator: (A1, A0, B1, B0) -> (A > B, A == B, A < B)
def _generate_comparator_truth_table() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    table = []
    for a in range(4):
        for b in range(4):
            a_bits = ((a >> 1) & 1, a & 1)
            b_bits = ((b >> 1) & 1, b & 1)
            inps = (a_bits[0], a_bits[1], b_bits[0], b_bits[1])
            gt = 1 if a > b else 0
            eq = 1 if a == b else 0
            lt = 1 if a < b else 0
            table.append((inps, (gt, eq, lt)))
    return table

COMPARATOR_TRUTH_TABLE = _generate_comparator_truth_table()


class ALUEvaluator:
    """Fitness evaluator for CMOS ALU sub-module synthesis."""

    def __init__(
        self,
        truth_table: Sequence[tuple[Sequence[int], Sequence[int]]],
        target_name: str = "CMOS_ALU",
        area_weight: float = 15.0,
        delay_weight: float = 15.0,
    ):
        self.truth_table = truth_table
        self.target_name = target_name
        self.area_weight = area_weight
        self.delay_weight = delay_weight

    def evaluate(self, genome: Any) -> float:
        if not isinstance(genome, CGPGenome):
            return 0.0

        metrics = genome.evaluate_truth_table(self.truth_table)
        
        # 1. Primary Objective: Functional Correctness (Truth Table Match)
        if not metrics.is_fully_functional:
            # Harsh penalty for broken logic: scaled between 0 and 70.0
            return metrics.truth_table_accuracy * 70.0 - (metrics.active_gate_count * 0.05)

        # 2. Secondary Objectives (Silicon Optimization): Only granted once 100% correct
        # Max area bonus if gate count is small (e.g. 5 gates for Half Adder or 9 for Full Adder)
        area_bonus = self.area_weight / (1.0 + metrics.active_gate_count)
        delay_bonus = self.delay_weight / (1.0 + metrics.critical_path_delay)
        
        return 70.0 + area_bonus + delay_bonus

    def __call__(self, ind_or_genome: Any) -> float:
        g = getattr(ind_or_genome, "genome", ind_or_genome)
        return self.evaluate(g)


class LowPowerALUEvaluator:
    """Multi-objective fitness evaluator for low-power ALU synthesis minimizing dynamic switching activity."""

    def __init__(
        self,
        truth_table: Sequence[tuple[Sequence[int], Sequence[int]]],
        switching_stream: Sequence[Sequence[int]] | None = None,
        target_name: str = "CMOS_LowPowerALU",
        area_weight: float = 10.0,
        delay_weight: float = 10.0,
        power_weight: float = 15.0,
        max_baseline_toggles: int = 150,
    ):
        self.truth_table = truth_table
        self.target_name = target_name
        self.area_weight = area_weight
        self.delay_weight = delay_weight
        self.power_weight = power_weight
        self.max_baseline_toggles = max_baseline_toggles

        if switching_stream is None:
            r = random.Random(42)
            stream = [vec for vec, _ in truth_table]
            for _ in range(32):
                stream.append(r.choice([vec for vec, _ in truth_table]))
            self.switching_stream = stream
        else:
            self.switching_stream = list(switching_stream)

    def evaluate(self, genome: Any) -> float:
        if not isinstance(genome, CGPGenome):
            return 0.0

        metrics = genome.evaluate_truth_table(self.truth_table)
        
        # 1. Primary Objective: Functional Correctness
        if not metrics.is_fully_functional:
            return max(0.0, metrics.truth_table_accuracy * 70.0 - (metrics.active_gate_count * 0.05))

        # 2. Secondary Objectives: Area, Delay, and Low-Power Switching Activity
        area_bonus = self.area_weight / (1.0 + metrics.active_gate_count)
        delay_bonus = self.delay_weight / (1.0 + metrics.critical_path_delay)
        
        sw_metrics = genome.evaluate_switching_activity(self.switching_stream)
        toggle_ratio = min(sw_metrics.total_toggles / max(self.max_baseline_toggles, 1), 1.0)
        power_bonus = self.power_weight * (1.0 - toggle_ratio)

        return 70.0 + area_bonus + delay_bonus + power_bonus

    def __call__(self, ind_or_genome: Any) -> float:
        g = getattr(ind_or_genome, "genome", ind_or_genome)
        return self.evaluate(g)


@dataclass
class HierarchicalALUMetrics:
    """Aggregated silicon area, transistor count, and latency for multi-bit composed ALUs."""
    bit_width: int
    total_active_gates: int
    total_transistors: int
    critical_path_delay_fo4: float
    is_100_percent_functional: bool
    verified_vector_count: int


class HierarchicalAdder8Bit:
    """8-bit Ripple-Carry Adder composed hierarchically from an evolved or canonical 1-bit Full Adder."""

    def __init__(self, full_adder_cell: CGPGenome):
        if full_adder_cell.num_inputs != 3 or full_adder_cell.num_outputs != 2:
            raise ValueError("full_adder_cell must have exactly 3 inputs (A, B, Cin) and 2 outputs (Sum, Cout)")
        self.cell = full_adder_cell
        self.cell_metrics = full_adder_cell.evaluate_truth_table(FULL_ADDER_TRUTH_TABLE)
        if not self.cell_metrics.is_fully_functional:
            raise ValueError("full_adder_cell must be 100% functional before hierarchical composition")

    @classmethod
    def create_canonical(cls) -> HierarchicalAdder8Bit:
        """Creates an 8-bit adder using the canonical CMOS 5-gate Full Adder cell."""
        # 3 inputs: 0=A, 1=B, 2=Cin
        # Node 3: XOR(0, 1) -> P
        # Node 4: XOR(3, 2) -> Sum
        # Node 5: AND(0, 1) -> G
        # Node 6: AND(3, 2) -> P & Cin
        # Node 7: OR(5, 6)  -> Cout
        nodes = [
            CGPNode(GateType.XOR, 0, 1),
            CGPNode(GateType.XOR, 3, 2),
            CGPNode(GateType.AND, 0, 1),
            CGPNode(GateType.AND, 3, 2),
            CGPNode(GateType.OR, 5, 6),
        ]
        cell = CGPGenome(num_inputs=3, num_outputs=2, nodes=nodes, output_connections=[4, 7])
        return cls(cell)

    def simulate(self, a_byte: int, b_byte: int, cin: int = 0) -> tuple[int, int]:
        """Simulates 8-bit addition (A + B + Cin) through the chained 8 full adder slices.
        Returns: (sum_byte: int [0..255], cout: int [0 or 1]).
        """
        carry = cin & 1
        sum_bits = []
        for bit in range(8):
            a_bit = (a_byte >> bit) & 1
            b_bit = (b_byte >> bit) & 1
            out = self.cell.simulate([a_bit, b_bit, carry])
            sum_bits.append(out[0])
            carry = out[1]

        sum_val = 0
        for bit, val in enumerate(sum_bits):
            sum_val |= (val << bit)

        return sum_val, carry

    def compute_metrics(self) -> HierarchicalALUMetrics:
        """Computes accurate total CMOS transistor count and cumulative ripple-carry delay."""
        total_gates = self.cell_metrics.active_gate_count * 8
        total_transistors = self.cell_metrics.transistor_count * 8
        ripple_delay = self.cell_metrics.critical_path_delay * 8
        return HierarchicalALUMetrics(
            bit_width=8,
            total_active_gates=total_gates,
            total_transistors=total_transistors,
            critical_path_delay_fo4=round(ripple_delay, 2),
            is_100_percent_functional=True,
            verified_vector_count=0,
        )

    def verify_exhaustive(self, max_cases: int = 1000, rng: random.Random | None = None) -> tuple[bool, int]:
        """Verifies accuracy against canonical arithmetic addition over deterministic and random vectors."""
        r = rng or random.Random(42)
        edge_cases = [
            (0, 0, 0), (0, 0, 1),
            (255, 0, 0), (0, 255, 0),
            (255, 1, 0), (255, 255, 0), (255, 255, 1),
            (127, 128, 0), (128, 128, 0),
            (85, 170, 0), (85, 170, 1),
            (15, 1, 0), (16, 16, 0),
        ]
        count = 0
        for a, b, cin in edge_cases:
            expected_sum = (a + b + cin) & 0xFF
            expected_cout = 1 if (a + b + cin) > 0xFF else 0
            actual_sum, actual_cout = self.simulate(a, b, cin)
            if actual_sum != expected_sum or actual_cout != expected_cout:
                return False, count
            count += 1

        while count < max_cases:
            a = r.randint(0, 255)
            b = r.randint(0, 255)
            cin = r.randint(0, 1)
            expected_sum = (a + b + cin) & 0xFF
            expected_cout = 1 if (a + b + cin) > 0xFF else 0
            actual_sum, actual_cout = self.simulate(a, b, cin)
            if actual_sum != expected_sum or actual_cout != expected_cout:
                return False, count
            count += 1

        return True, count

    def to_verilog(self, module_name: str = "hierarchical_8bit_adder") -> str:
        """Exports a full hierarchical synthesizable Verilog module connecting 8 evolved FA instances."""
        cell_verilog = self.cell.to_verilog("cgp_fa_cell")
        lines = [
            "// Hierarchical 8-Bit Ripple-Carry Adder",
            "// Synthesized and composed by darwin-evolab",
            "",
            cell_verilog,
            "",
            f"module {module_name} (",
            "    input  wire [7:0] a,",
            "    input  wire [7:0] b,",
            "    input  wire       cin,",
            "    output wire [7:0] sum,",
            "    output wire       cout",
            ");",
            "    wire [8:0] c;",
            "    assign c[0] = cin;",
            "    assign cout = c[8];",
            "",
        ]
        for bit in range(8):
            lines.append(f"    // Bit {bit} Stage")
            lines.append(f"    cgp_fa_cell fa_{bit} (")
            lines.append(f"        .in({{c[{bit}], b[{bit}], a[{bit}]}}),")
            lines.append(f"        .out({{c[{bit+1}], sum[{bit}]}})")
            lines.append("    );")
            lines.append("")
        lines.append("endmodule")
        return "\n".join(lines)


@dataclass(frozen=True)
class FPGABoardSpec:
    """Hardware specification and physical resource budget for an FPGA board."""
    preset_name: str
    board_name: str
    vendor: str
    family: str
    part: str
    total_luts: int
    total_ffs: int
    total_brams: int
    total_ios: int
    lut_input_size: int = 4
    core_voltage: float = 1.2
    typical_gate_delay_ns: float = 0.55
    interconnect_delay_factor: float = 1.3


FPGA_PRESETS: dict[str, FPGABoardSpec] = {
    "ice40_hx1k": FPGABoardSpec(
        preset_name="ice40_hx1k",
        board_name="Lattice iCEstick (iCE40-HX1K)",
        vendor="Lattice",
        family="iCE40-HX",
        part="iCE40HX1K-TQ144",
        total_luts=1280,
        total_ffs=1280,
        total_brams=16,
        total_ios=96,
        lut_input_size=4,
        core_voltage=1.2,
        typical_gate_delay_ns=0.55,
    ),
    "ice40_up5k": FPGABoardSpec(
        preset_name="ice40_up5k",
        board_name="iCEBreaker / Lattice iCE40-UP5K",
        vendor="Lattice",
        family="iCE40-UltraPlus",
        part="iCE40UP5K-SG48",
        total_luts=5280,
        total_ffs=5280,
        total_brams=30,
        total_ios=39,
        lut_input_size=4,
        core_voltage=1.2,
        typical_gate_delay_ns=0.50,
    ),
    "ecp5_25k": FPGABoardSpec(
        preset_name="ecp5_25k",
        board_name="Colorlight 5A-75B / Lattice ECP5-25F",
        vendor="Lattice",
        family="ECP5",
        part="LFE5U-25F-6BG381C",
        total_luts=24192,
        total_ffs=24192,
        total_brams=56,
        total_ios=197,
        lut_input_size=4,
        core_voltage=1.1,
        typical_gate_delay_ns=0.35,
    ),
    "artix7_35t": FPGABoardSpec(
        preset_name="artix7_35t",
        board_name="Digilent Basys 3 / AMD Artix-7 35T",
        vendor="AMD/Xilinx",
        family="Artix-7",
        part="XC7A35T-1CPG236C",
        total_luts=20800,
        total_ffs=41600,
        total_brams=50,
        total_ios=106,
        lut_input_size=6,
        core_voltage=1.0,
        typical_gate_delay_ns=0.22,
    ),
}


@dataclass
class FPGAResourceReport:
    """Static pre-synthesis hardware resource and timing estimation report."""
    target_preset: str
    board_name: str
    vendor: str
    estimated_luts: int
    total_luts: int
    lut_utilization_pct: float
    estimated_ffs: int
    total_ffs: int
    total_pins_used: int
    total_ios_available: int
    pin_utilization_pct: float
    critical_path_depth: int
    estimated_delay_ns: float
    estimated_fmax_mhz: float
    estimated_dynamic_power_uw: float
    fits_on_target: bool


def estimate_fpga_resources(
    genome: CGPGenome,
    target: str | FPGABoardSpec = "ice40_hx1k",
) -> FPGAResourceReport:
    """Computes static FPGA resource utilization, timing, and dynamic power estimates."""
    if isinstance(target, str):
        board = FPGA_PRESETS.get(target.lower())
        if board is None:
            board = FPGA_PRESETS.get("ice40_hx1k")
    else:
        board = target

    assert board is not None

    active_nodes = sorted(genome.get_active_nodes())
    active_gates = 0
    lut_reduction = 0.65 if board.lut_input_size >= 6 else 0.85

    node_depths = [0] * (genome.num_inputs + len(genome.nodes))
    for idx in active_nodes:
        node = genome.nodes[idx - genome.num_inputs]
        if node.gate_type != GateType.WIRE:
            active_gates += 1
        d_in = max(node_depths[node.input_a], node_depths[node.input_b])
        depth = d_in + (0 if node.gate_type == GateType.WIRE else 1)
        node_depths[idx] = depth

    max_depth = max((node_depths[out] for out in genome.output_connections), default=1)
    max_depth = max(max_depth, 1)

    estimated_luts = max(1, int(round(active_gates * lut_reduction))) if active_gates > 0 else 0
    total_pins_used = genome.num_inputs + genome.num_outputs

    delay_per_stage_ns = board.typical_gate_delay_ns * board.interconnect_delay_factor
    io_pad_delay_ns = 1.2
    est_delay_ns = round(max_depth * delay_per_stage_ns + io_pad_delay_ns, 2)
    est_fmax_mhz = round(min(1000.0 / max(est_delay_ns, 0.1), 400.0), 1)

    c_eff_pf = 1.5
    voltage = board.core_voltage
    activity = 0.125
    power_uw = round(c_eff_pf * (voltage ** 2) * (est_fmax_mhz * 1e6) * max(estimated_luts, 1) * activity * 1e-6, 2)

    lut_util = round((estimated_luts / max(board.total_luts, 1)) * 100.0, 3)
    pin_util = round((total_pins_used / max(board.total_ios, 1)) * 100.0, 3)
    fits = (estimated_luts <= board.total_luts) and (total_pins_used <= board.total_ios)

    return FPGAResourceReport(
        target_preset=board.preset_name,
        board_name=board.board_name,
        vendor=board.vendor,
        estimated_luts=estimated_luts,
        total_luts=board.total_luts,
        lut_utilization_pct=lut_util,
        estimated_ffs=0,
        total_ffs=board.total_ffs,
        total_pins_used=total_pins_used,
        total_ios_available=board.total_ios,
        pin_utilization_pct=pin_util,
        critical_path_depth=max_depth,
        estimated_delay_ns=est_delay_ns,
        estimated_fmax_mhz=est_fmax_mhz,
        estimated_dynamic_power_uw=power_uw,
        fits_on_target=fits,
    )


@dataclass
class EDABundleReport:
    """Artifact and compilation metadata produced by the open-source EDA packager."""
    target_fpga: str
    top_module: str
    bundle_directory: str
    verilog_file: str
    yosys_script_file: str
    constraints_file: str
    yosys_installed: bool
    synthesis_executed: bool
    synthesis_stdout: str = ""
    lut_count: int | None = None
    wire_count: int | None = None
    resource_estimate: FPGAResourceReport | None = None


class EDAPackager:
    """Generates complete open-source EDA synthesis bundles (Yosys + nextpnr) for FPGA deployment."""

    def __init__(self, target_fpga: str = "ice40", package: str = "hx1k-tq144"):
        self.raw_target = target_fpga
        # Resolve target architecture family
        lower = target_fpga.lower()
        if lower in FPGA_PRESETS:
            self.board_spec: FPGABoardSpec | None = FPGA_PRESETS[lower]
            if "ice40" in lower:
                self.target_fpga = "ice40"
            elif "ecp5" in lower:
                self.target_fpga = "ecp5"
            elif "artix" in lower:
                self.target_fpga = "xilinx"
            else:
                self.target_fpga = "ice40"
        elif "ecp5" in lower:
            self.board_spec = FPGA_PRESETS.get("ecp5_25k")
            self.target_fpga = "ecp5"
        elif "xilinx" in lower or "artix" in lower:
            self.board_spec = FPGA_PRESETS.get("artix7_35t")
            self.target_fpga = "xilinx"
        else:
            self.board_spec = FPGA_PRESETS.get("ice40_hx1k")
            self.target_fpga = "ice40"

        self.package = package

    def is_yosys_available(self) -> bool:
        """Checks if yosys CLI is installed in the system PATH."""
        return shutil.which("yosys") is not None

    def generate_yosys_script(self, verilog_filename: str, top_module: str, json_output: str) -> str:
        """Generates standard TCL synthesis script for Yosys."""
        synth_cmd = f"synth_{self.target_fpga} -top {top_module} -json {json_output}"
        return (
            f"# Yosys Open-Source Synthesis Script for {top_module}\n"
            f"# Target Architecture: {self.target_fpga.upper()}\n"
            f"read_verilog {verilog_filename}\n"
            f"hierarchy -check -top {top_module}\n"
            f"proc; opt; fsm; opt; memory; opt\n"
            f"techmap; opt\n"
            f"{synth_cmd}\n"
            f"stat\n"
        )

    def generate_pcf_constraints(self, top_module: str, num_inputs: int, num_outputs: int) -> str:
        """Generates Physical Constraint File (.pcf) for Lattice iCE40 development boards."""
        lines = [
            f"# Physical Pin Constraints for {top_module}",
            "# Compatible with Lattice iCE40 packages (e.g. iCEstick / iCEBreaker)",
            "",
        ]
        for i in range(num_inputs):
            pin_num = 10 + i
            lines.append(f"set_io in[{i}] {pin_num}")
        for o in range(num_outputs):
            pin_num = 30 + o
            lines.append(f"set_io out[{o}] {pin_num}")
        return "\n".join(lines)

    def generate_lpf_constraints(self, top_module: str, num_inputs: int, num_outputs: int) -> str:
        """Generates Logical Preference File (.lpf) for Lattice ECP5 development boards."""
        lines = [
            f"# Physical Pin Constraints for {top_module}",
            "# Compatible with Lattice ECP5 boards (e.g. Colorlight 5A-75B)",
            "",
        ]
        for i in range(num_inputs):
            lines.append(f'LOCATE COMP "in[{i}]" SITE "P{i + 1}";')
            lines.append(f'IOBUF PORT "in[{i}]" IO_TYPE=LVCMOS33;')
        for o in range(num_outputs):
            lines.append(f'LOCATE COMP "out[{o}]" SITE "T{o + 1}";')
            lines.append(f'IOBUF PORT "out[{o}]" IO_TYPE=LVCMOS33;')
        return "\n".join(lines)

    def generate_xdc_constraints(self, top_module: str, num_inputs: int, num_outputs: int) -> str:
        """Generates Xilinx Design Constraints (.xdc) for AMD/Xilinx Artix-7 boards."""
        lines = [
            f"# Physical Pin Constraints for {top_module}",
            "# Compatible with AMD Artix-7 boards (e.g. Basys 3)",
            "",
        ]
        for i in range(num_inputs):
            lines.append(f"set_property -dict {{ PACKAGE_PIN V{17 - i} IOSTANDARD LVCMOS33 }} [get_ports {{ in[{i}] }}];")
        for o in range(num_outputs):
            lines.append(f"set_property -dict {{ PACKAGE_PIN U{16 - o} IOSTANDARD LVCMOS33 }} [get_ports {{ out[{o}] }}];")
        return "\n".join(lines)

    def generate_constraints(self, top_module: str, num_inputs: int, num_outputs: int) -> tuple[str, str]:
        """Returns constraint filename and content tailored to target architecture."""
        if self.target_fpga == "ecp5":
            return f"{top_module}.lpf", self.generate_lpf_constraints(top_module, num_inputs, num_outputs)
        elif self.target_fpga == "xilinx":
            return f"{top_module}.xdc", self.generate_xdc_constraints(top_module, num_inputs, num_outputs)
        else:
            return f"{top_module}.pcf", self.generate_pcf_constraints(top_module, num_inputs, num_outputs)

    def package_bundle(
        self,
        verilog_code: str,
        top_module: str,
        num_inputs: int,
        num_outputs: int,
        output_dir: str | Path,
        run_synthesis_if_available: bool = True,
        genome: CGPGenome | None = None,
    ) -> EDABundleReport:
        """Creates complete synthesis bundle files and optionally runs Yosys logic synthesis."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        v_file = out_path / f"{top_module}.v"
        ys_file = out_path / f"synth_{self.target_fpga}.ys"
        c_filename, c_content = self.generate_constraints(top_module, num_inputs, num_outputs)
        constraints_file = out_path / c_filename
        bat_file = out_path / "run_synth.bat"
        sh_file = out_path / "run_synth.sh"

        v_file.write_text(verilog_code, encoding="utf-8")
        yosys_script = self.generate_yosys_script(v_file.name, top_module, f"{top_module}.json")
        ys_file.write_text(yosys_script, encoding="utf-8")
        constraints_file.write_text(c_content, encoding="utf-8")
        bat_file.write_text(f"@echo off\nyosys -s {ys_file.name}\n", encoding="utf-8")
        sh_file.write_text(f"#!/usr/bin/env bash\nyosys -s {ys_file.name}\n", encoding="utf-8")

        yosys_installed = self.is_yosys_available()
        synthesis_executed = False
        stdout_result = ""
        luts = None
        wires = None

        if yosys_installed and run_synthesis_if_available:
            try:
                proc = subprocess.run(
                    ["yosys", "-s", str(ys_file.name)],
                    cwd=str(out_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                stdout_result = proc.stdout
                synthesis_executed = (proc.returncode == 0)
                for line in stdout_result.splitlines():
                    if "SB_LUT4" in line or "LUT4" in line or "LUT6" in line or "Number of cells:" in line:
                        parts = line.strip().split()
                        if parts and parts[-1].isdigit():
                            luts = int(parts[-1])
                    if "Number of wires:" in line:
                        parts = line.strip().split()
                        if parts and parts[-1].isdigit():
                            wires = int(parts[-1])
            except Exception as e:
                stdout_result = f"Synthesis execution failed: {e}"

        resource_est = None
        if genome is not None and self.board_spec is not None:
            resource_est = estimate_fpga_resources(genome, self.board_spec)

        return EDABundleReport(
            target_fpga=self.target_fpga,
            top_module=top_module,
            bundle_directory=str(out_path),
            verilog_file=str(v_file),
            yosys_script_file=str(ys_file),
            constraints_file=str(constraints_file),
            yosys_installed=yosys_installed,
            synthesis_executed=synthesis_executed,
            synthesis_stdout=stdout_result,
            lut_count=luts,
            wire_count=wires,
            resource_estimate=resource_est,
        )


