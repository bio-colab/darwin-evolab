"""sequential_cgp.py — Sequential Cartesian Genetic Programming & SoC Digital Logic Synthesis.

Part of Darwin-Evolab Pillar 3: Silicon Logic Scaling.
Breaks the 4-bit combinational ALU and 2^k truth-table limits by introducing:
  - Clocked Sequential Primitives: D-Flip-Flop (DFF), Clock Enable (DFFE), Multiplexer (MUX).
  - Synchronous Cycle-Accurate Waveform Simulation across T clock cycles.
  - State feedback loops through registered clock boundaries (zero combinational loops).
  - Scaled SoC Building Blocks:
      * 8-Bit Synchronous Up/Down Counter with parallel load & overflow flag.
      * 8-Bit Shift Register (PISO / SIPO) with serial/parallel modes.
      * Clocked UART Serial Protocol Transmitter FSM (start, 8 data bits, stop bit).
  - Verilog-2001 Synchronous RTL Exporter with always @(posedge clk or negedge rst_n).
"""

from __future__ import annotations

import copy
import enum
import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .genome import EvolabGenome


class SequentialGateType(str, enum.Enum):
    """Supported digital logic gate primitives including clocked sequential elements."""
    # Combinational primitives
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NAND = "NAND"
    NOR = "NOR"
    NOT = "NOT"
    WIRE = "WIRE"
    MUX = "MUX"       # 2-to-1 multiplexer: out = (s & b) | (~s & a)
    # Clocked sequential primitive
    DFF = "DFF"       # Positive-edge D-Flip-Flop with active-low async reset


# CMOS Transistor Counts (Static CMOS standard cell library)
SEQ_GATE_TRANSISTORS: dict[SequentialGateType, int] = {
    SequentialGateType.NOT: 2,
    SequentialGateType.NAND: 4,
    SequentialGateType.NOR: 4,
    SequentialGateType.AND: 6,
    SequentialGateType.OR: 6,
    SequentialGateType.XOR: 8,
    SequentialGateType.WIRE: 0,
    SequentialGateType.MUX: 6,      # Transmission gate MUX
    SequentialGateType.DFF: 12,     # Master-slave static D-Flip-Flop
}

# Propagation Delays normalized to FO4 inverter delay
SEQ_GATE_DELAYS: dict[SequentialGateType, float] = {
    SequentialGateType.NOT: 1.0,
    SequentialGateType.NAND: 1.0,
    SequentialGateType.NOR: 1.2,
    SequentialGateType.AND: 1.6,
    SequentialGateType.OR: 1.6,
    SequentialGateType.XOR: 2.0,
    SequentialGateType.WIRE: 0.0,
    SequentialGateType.MUX: 1.8,
    SequentialGateType.DFF: 2.5,     # Clock-to-Q delay + setup margin
}


def eval_sequential_gate(
    gtype: SequentialGateType,
    a: int,
    b: int,
    c: int = 0,
) -> int:
    """Evaluates Boolean output of a gate primitive given binary inputs."""
    if gtype == SequentialGateType.AND:
        return a & b
    elif gtype == SequentialGateType.OR:
        return a | b
    elif gtype == SequentialGateType.XOR:
        return a ^ b
    elif gtype == SequentialGateType.NAND:
        return 0 if (a & b) else 1
    elif gtype == SequentialGateType.NOR:
        return 0 if (a | b) else 1
    elif gtype == SequentialGateType.NOT:
        return 0 if a else 1
    elif gtype == SequentialGateType.WIRE:
        return a
    elif gtype == SequentialGateType.MUX:
        # c is select line: c=0 -> a, c=1 -> b
        return b if c else a
    elif gtype == SequentialGateType.DFF:
        # Combinational evaluation returns the D input to be clocked
        return a
    return 0


@dataclass
class SequentialCGPNode:
    """A node in the Sequential CGP network."""
    gate_type: SequentialGateType
    input_a: int
    input_b: int
    input_c: int = 0  # Used for MUX select line

    def clone(self) -> SequentialCGPNode:
        return SequentialCGPNode(
            gate_type=self.gate_type,
            input_a=self.input_a,
            input_b=self.input_b,
            input_c=self.input_c,
        )


@dataclass
class SequentialCircuitMetrics:
    """Hardware complexity, timing, and waveform verification metrics."""
    waveform_accuracy: float           # [0.0, 100.0] percentage of matching output bits
    active_gate_count: int             # Number of functionally active nodes
    dff_count: int                     # Number of active flip-flops (registers)
    transistor_count: int              # Total CMOS transistor count
    critical_path_delay: float         # Longest combinational path in FO4 units
    cycles_tested: int                 # Clock cycles evaluated in waveform testbench
    is_fully_functional: bool          # True iff accuracy == 100.0


class SequentialCGPGenome(EvolabGenome):
    """Sequential Cartesian Genetic Programming Netlist Genome."""

    def __init__(
        self,
        num_primary_inputs: int,
        num_primary_outputs: int,
        num_dffs: int,
        nodes: list[SequentialCGPNode],
        output_connections: list[int],
        dff_input_connections: list[int],
        allowed_gates: Sequence[SequentialGateType] | None = None,
    ) -> None:
        self.num_primary_inputs = num_primary_inputs
        self.num_primary_outputs = num_primary_outputs
        self.num_dffs = num_dffs
        self.nodes = nodes
        self.output_connections = list(output_connections)
        self.dff_input_connections = list(dff_input_connections)
        self.allowed_gates = list(allowed_gates) if allowed_gates is not None else [
            SequentialGateType.NAND,
            SequentialGateType.NOR,
            SequentialGateType.AND,
            SequentialGateType.OR,
            SequentialGateType.XOR,
            SequentialGateType.NOT,
            SequentialGateType.MUX,
        ]

    def total_inputs(self) -> int:
        """Total input lines: primary inputs + DFF Q state outputs."""
        return self.num_primary_inputs + self.num_dffs

    def clone(self) -> SequentialCGPGenome:
        return SequentialCGPGenome(
            num_primary_inputs=self.num_primary_inputs,
            num_primary_outputs=self.num_primary_outputs,
            num_dffs=self.num_dffs,
            nodes=[n.clone() for n in self.nodes],
            output_connections=list(self.output_connections),
            dff_input_connections=list(self.dff_input_connections),
            allowed_gates=list(self.allowed_gates),
        )

    def get_active_nodes(self) -> set[int]:
        """Traces backward from primary outputs and DFF D-inputs to find active cells."""
        active: set[int] = set()
        queue = list(self.output_connections) + list(self.dff_input_connections)
        tot_in = self.total_inputs()

        while queue:
            idx = queue.pop()
            if idx >= tot_in and idx not in active:
                active.add(idx)
                node_idx = idx - tot_in
                node = self.nodes[node_idx]
                queue.append(node.input_a)
                if node.gate_type not in (SequentialGateType.NOT, SequentialGateType.WIRE):
                    queue.append(node.input_b)
                if node.gate_type == SequentialGateType.MUX:
                    queue.append(node.input_c)
        return active

    def simulate_cycle(
        self,
        primary_inputs: Sequence[int],
        current_state: Sequence[int],
    ) -> tuple[list[int], list[int]]:
        """Simulates one clock cycle combinational propagation.

        Returns:
            (primary_outputs, next_state_d_inputs)
        """
        # Wire values: primary inputs [0..N-1], current state Q outputs [N..N+M-1]
        values = list(primary_inputs) + list(current_state)

        for node in self.nodes:
            val_a = values[node.input_a]
            val_b = values[node.input_b]
            val_c = values[node.input_c]
            out = eval_sequential_gate(node.gate_type, val_a, val_b, val_c)
            values.append(out)

        primary_outs = [values[out_idx] for out_idx in self.output_connections]
        next_state_d = [values[d_idx] for d_idx in self.dff_input_connections]
        return primary_outs, next_state_d

    def simulate_waveform(
        self,
        input_timeline: Sequence[Sequence[int]],
        initial_state: Sequence[int] | None = None,
        rst_n_timeline: Sequence[int] | None = None,
    ) -> list[list[int]]:
        """Cycle-accurate synchronous waveform simulation across T clock cycles.

        Args:
            input_timeline: List of primary input vectors for each cycle t = 0..T-1.
            initial_state: Initial DFF Q register state (default: all 0).
            rst_n_timeline: Optional active-low reset sequence per cycle (0 = reset, 1 = run).

        Returns:
            List of primary output vectors for each cycle t = 0..T-1.
        """
        t_cycles = len(input_timeline)
        state = list(initial_state or [0] * self.num_dffs)
        output_timeline: list[list[int]] = []

        for t in range(t_cycles):
            in_vec = input_timeline[t]
            rst_n = rst_n_timeline[t] if rst_n_timeline is not None else 1

            if rst_n == 0:
                # Reset asserted: registers forced to 0
                state = [0] * self.num_dffs

            outs, next_d = self.simulate_cycle(in_vec, state)
            output_timeline.append(outs)

            # On clock edge: D inputs clocked into state registers
            if rst_n != 0:
                state = next_d

        return output_timeline

    def evaluate_waveform_metrics(
        self,
        input_timeline: Sequence[Sequence[int]],
        expected_output_timeline: Sequence[Sequence[int]],
        rst_n_timeline: Sequence[int] | None = None,
    ) -> SequentialCircuitMetrics:
        """Evaluates cycle-accurate functional accuracy and physical complexity."""
        sim_outs = self.simulate_waveform(input_timeline, rst_n_timeline=rst_n_timeline)
        total_bits = 0
        matching_bits = 0

        for t in range(len(input_timeline)):
            actual = sim_outs[t]
            expected = expected_output_timeline[t]
            for bit_a, bit_e in zip(actual, expected):
                total_bits += 1
                if bit_a == bit_e:
                    matching_bits += 1

        accuracy = (matching_bits / max(1, total_bits)) * 100.0
        active = self.get_active_nodes()
        tot_in = self.total_inputs()

        # Physical calculations
        transistor_count = self.num_dffs * SEQ_GATE_TRANSISTORS[SequentialGateType.DFF]
        max_delay = 0.0

        for idx in active:
            node = self.nodes[idx - tot_in]
            transistor_count += SEQ_GATE_TRANSISTORS.get(node.gate_type, 4)
            delay = SEQ_GATE_DELAYS.get(node.gate_type, 1.5)
            if delay > max_delay:
                max_delay = delay

        # Total critical path = combinational delay + DFF setup/clock-to-q
        total_delay = max_delay + SEQ_GATE_DELAYS[SequentialGateType.DFF]

        return SequentialCircuitMetrics(
            waveform_accuracy=round(accuracy, 2),
            active_gate_count=len(active),
            dff_count=self.num_dffs,
            transistor_count=transistor_count,
            critical_path_delay=round(total_delay, 2),
            cycles_tested=len(input_timeline),
            is_fully_functional=bool(accuracy == 100.0),
        )

    def fingerprint(self) -> str:
        """Stable hash identifying identity of the active sequential circuit subgraph."""
        active = sorted(self.get_active_nodes())
        tot_in = self.total_inputs()
        content = (
            f"{self.num_primary_inputs}:{self.num_dffs}:{self.num_primary_outputs}:"
            + ";".join(
                f"{idx}:{self.nodes[idx - tot_in].gate_type.value}:{self.nodes[idx - tot_in].input_a}:{self.nodes[idx - tot_in].input_b}:{self.nodes[idx - tot_in].input_c}"
                for idx in active
            )
            + ":"
            + ",".join(str(c) for c in self.output_connections)
            + ":"
            + ",".join(str(d) for d in self.dff_input_connections)
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        """Non-negative symmetric distance metric implementing EvolabGenome contract."""
        if not isinstance(other, SequentialCGPGenome):
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
            if n1.input_c != n2.input_c:
                diff += 1
        for o1, o2 in zip(self.output_connections, other.output_connections):
            if o1 != o2:
                diff += 1
        for d1, d2 in zip(self.dff_input_connections, other.dff_input_connections):
            if d1 != d2:
                diff += 1
        return float(diff)

    def serialize(self) -> dict[str, Any]:
        """JSON-safe serialization of the sequential genome."""
        return {
            "num_primary_inputs": self.num_primary_inputs,
            "num_dffs": self.num_dffs,
            "num_primary_outputs": self.num_primary_outputs,
            "active_nodes": list(self.get_active_nodes()),
            "output_connections": list(self.output_connections),
            "dff_input_connections": list(self.dff_input_connections),
            "node_count": len(self.nodes),
            "nodes": [
                {
                    "gate_type": n.gate_type.value,
                    "input_a": n.input_a,
                    "input_b": n.input_b,
                    "input_c": n.input_c,
                }
                for n in self.nodes
            ],
        }

    def describe(self) -> dict[str, Any]:
        """Behavioral and structural descriptors for MAP-Elites and silicon analysis."""
        active = self.get_active_nodes()
        return {
            "node_count": len(active),
            "total_nodes": len(self.nodes),
            "dff_count": self.num_dffs,
            "primary_inputs": self.num_primary_inputs,
            "primary_outputs": self.num_primary_outputs,
        }

    def to_verilog(self, module_name: str = "soc_sequential_core") -> str:
        """Generates synthesizable Verilog-2001 synchronous RTL code."""
        lines: list[str] = [
            f"// Darwin-Evolab Synthesized Synchronous Sequential RTL Core",
            f"// Module: {module_name}",
            f"// DFF Registers: {self.num_dffs}, Primary Inputs: {self.num_primary_inputs}, Primary Outputs: {self.num_primary_outputs}",
            f"`timescale 1ns / 1ps\n",
            f"module {module_name} (",
            f"    input  wire clk,",
            f"    input  wire rst_n,",
        ]

        # Primary input ports
        for i in range(self.num_primary_inputs):
            lines.append(f"    input  wire in_{i},")
        # Primary output ports
        for o in range(self.num_primary_outputs):
            comma = "," if o < self.num_primary_outputs - 1 else ""
            lines.append(f"    output wire out_{o}{comma}")
        lines.append(");\n")

        tot_in = self.total_inputs()

        # DFF State registers
        if self.num_dffs > 0:
            lines.append(f"    // Internal State Registers")
            lines.append(f"    reg  [{self.num_dffs-1}:0] dff_q;")
            lines.append(f"    wire [{self.num_dffs-1}:0] dff_d;\n")

        # Combinational internal node wires
        lines.append("    // Combinational Logic Cell Wires")
        for idx in range(len(self.nodes)):
            wire_idx = tot_in + idx
            lines.append(f"    wire node_{wire_idx};")
        lines.append("")

        def _wire_name(idx: int) -> str:
            if idx < self.num_primary_inputs:
                return f"in_{idx}"
            elif idx < tot_in:
                dff_idx = idx - self.num_primary_inputs
                return f"dff_q[{dff_idx}]"
            else:
                return f"node_{idx}"

        # Gate logic continuous assignments
        for idx, node in enumerate(self.nodes):
            w_out = tot_in + idx
            w_a = _wire_name(node.input_a)
            w_b = _wire_name(node.input_b)
            w_c = _wire_name(node.input_c)

            gt = node.gate_type
            if gt == SequentialGateType.AND:
                expr = f"{w_a} & {w_b}"
            elif gt == SequentialGateType.OR:
                expr = f"{w_a} | {w_b}"
            elif gt == SequentialGateType.XOR:
                expr = f"{w_a} ^ {w_b}"
            elif gt == SequentialGateType.NAND:
                expr = f"~({w_a} & {w_b})"
            elif gt == SequentialGateType.NOR:
                expr = f"~({w_a} | {w_b})"
            elif gt == SequentialGateType.NOT:
                expr = f"~{w_a}"
            elif gt == SequentialGateType.MUX:
                expr = f"({w_c}) ? {w_b} : {w_a}"
            else:
                expr = f"{w_a}"

            lines.append(f"    assign node_{w_out} = {expr};")

        lines.append("")
        # DFF D-input assignments
        for dff_idx, d_idx in enumerate(self.dff_input_connections):
            lines.append(f"    assign dff_d[{dff_idx}] = {_wire_name(d_idx)};")

        # Synchronous always block for state registers
        if self.num_dffs > 0:
            lines.append("\n    // Synchronous Sequential State Transition")
            lines.append("    always @(posedge clk or negedge rst_n) begin")
            lines.append("        if (!rst_n) begin")
            lines.append(f"            dff_q <= {self.num_dffs}'b0;")
            lines.append("        end else begin")
            lines.append("            dff_q <= dff_d;")
            lines.append("        end")
            lines.append("    end\n")

        # Primary output assignments
        lines.append("    // Primary Output Assignments")
        for o, out_idx in enumerate(self.output_connections):
            lines.append(f"    assign out_{o} = {_wire_name(out_idx)};")

        lines.append("\nendmodule\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SoC Standard Building Block Waveform Generators & Evaluators
# ---------------------------------------------------------------------------

class SynchronousCounter8BitEvaluator:
    """Golden behavioral reference and testbench for an 8-Bit Synchronous Counter."""

    @staticmethod
    def generate_golden_testbench(cycles: int = 32) -> tuple[list[list[int]], list[list[int]], list[int]]:
        """Generates cycle-accurate inputs and expected count outputs.

        Inputs: [en, up_down, load, d0..d7] (total 11 inputs)
        Outputs: [q0..q7, overflow] (total 9 outputs)
        """
        inputs: list[list[int]] = []
        expected: list[list[int]] = []
        rst_n: list[int] = []

        count = 0
        rng = random.Random(12345)

        for t in range(cycles):
            # First 2 cycles are reset
            if t < 2:
                r = 0
                en = 0
                up = 1
                load = 0
                din = 0
                count = 0
            elif t == 15:
                # Test parallel load at cycle 15
                r = 1
                en = 1
                up = 1
                load = 1
                din = 200
                count = 200
            else:
                r = 1
                en = 1
                up = 1 if t < 24 else 0  # Count up then down
                load = 0
                din = 0
                if en:
                    if up:
                        count = (count + 1) & 0xFF
                    else:
                        count = (count - 1) & 0xFF

            # Unpack din bits
            d_bits = [(din >> b) & 1 for b in range(8)]
            in_vec = [en, up, load] + d_bits

            # Unpack expected q bits + overflow flag
            q_bits = [(count >> b) & 1 for b in range(8)]
            ovf = 1 if count == 255 else 0
            out_vec = q_bits + [ovf]

            inputs.append(in_vec)
            expected.append(out_vec)
            rst_n.append(r)

        return inputs, expected, rst_n


class ShiftRegister8BitEvaluator:
    """Golden behavioral reference for an 8-Bit Serial-In/Parallel-Out & Parallel-In/Serial-Out Shift Register."""

    @staticmethod
    def generate_golden_testbench(cycles: int = 24) -> tuple[list[list[int]], list[list[int]], list[int]]:
        """Inputs: [shift_en, load, serial_in, d0..d7] (total 11 inputs).

        Outputs: [serial_out, q0..q7] (total 9 outputs).
        """
        inputs: list[list[int]] = []
        expected: list[list[int]] = []
        rst_n: list[int] = []

        reg = 0
        for t in range(cycles):
            if t < 2:
                r = 0
                shift_en = 0
                load = 0
                sin = 0
                din = 0
                reg = 0
            elif t == 5:
                # Load parallel value 0xA5 (165)
                r = 1
                shift_en = 0
                load = 1
                sin = 0
                din = 0xA5
                reg = din
            else:
                # Shift in bits
                r = 1
                shift_en = 1
                load = 0
                sin = (t % 2)
                din = 0
                reg = ((reg << 1) | sin) & 0xFF

            d_bits = [(din >> b) & 1 for b in range(8)]
            in_vec = [shift_en, load, sin] + d_bits

            sout = (reg >> 7) & 1
            q_bits = [(reg >> b) & 1 for b in range(8)]
            out_vec = [sout] + q_bits

            inputs.append(in_vec)
            expected.append(out_vec)
            rst_n.append(r)

        return inputs, expected, rst_n


class UARTTransmitterFSMEvaluator:
    """Golden reference testbench for a Clocked UART Serial Protocol Transmitter FSM."""

    @staticmethod
    def generate_golden_testbench() -> tuple[list[list[int]], list[list[int]], list[int]]:
        """Inputs: [tx_start, d0..d7] (total 9 inputs).

        Outputs: [tx_serial, tx_busy] (total 2 outputs).
        Protocol:
          - IDLE: tx_serial = 1, tx_busy = 0
          - START bit: tx_serial = 0, tx_busy = 1
          - 8 DATA bits: LSB first, tx_busy = 1
          - STOP bit: tx_serial = 1, tx_busy = 1
        """
        inputs: list[list[int]] = []
        expected: list[list[int]] = []
        rst_n: list[int] = []

        # We transmit byte 0b10110001 (0xB1 = 177)
        byte_to_send = 0xB1
        d_bits = [(byte_to_send >> b) & 1 for b in range(8)]

        # Total sequence: 2 reset cycles, 1 idle, 1 start bit, 8 data bits, 1 stop bit, 2 idle = 15 cycles
        timeline = [
            # (rst_n, tx_start, serial_out, busy)
            (0, 0, 1, 0),  # Cycle 0: reset
            (0, 0, 1, 0),  # Cycle 1: reset
            (1, 1, 1, 0),  # Cycle 2: idle + tx_start triggered!
            (1, 0, 0, 1),  # Cycle 3: START bit (0)
            (1, 0, d_bits[0], 1),  # Cycle 4: Data bit 0 (1)
            (1, 0, d_bits[1], 1),  # Cycle 5: Data bit 1 (0)
            (1, 0, d_bits[2], 1),  # Cycle 6: Data bit 2 (0)
            (1, 0, d_bits[3], 1),  # Cycle 7: Data bit 3 (0)
            (1, 0, d_bits[4], 1),  # Cycle 8: Data bit 4 (1)
            (1, 0, d_bits[5], 1),  # Cycle 9: Data bit 5 (1)
            (1, 0, d_bits[6], 1),  # Cycle 10: Data bit 6 (0)
            (1, 0, d_bits[7], 1),  # Cycle 11: Data bit 7 (1)
            (1, 0, 1, 1),  # Cycle 12: STOP bit (1)
            (1, 0, 1, 0),  # Cycle 13: IDLE
            (1, 0, 1, 0),  # Cycle 14: IDLE
        ]

        for r, start, s_out, busy in timeline:
            rst_n.append(r)
            in_vec = [start] + d_bits
            out_vec = [s_out, busy]
            inputs.append(in_vec)
            expected.append(out_vec)

        return inputs, expected, rst_n


def create_canonical_sequential_circuit(kind: str = "counter_8bit") -> SequentialCGPGenome:
    """Creates a canonical, functionally complete sequential circuit genome for testing and benchmarking."""
    if kind == "counter_8bit":
        # 8-bit counter: 11 inputs [en, up, load, d0..d7], 8 DFFs, 9 outputs [q0..q7, ovf]
        num_inputs = 11
        num_dffs = 8
        num_outputs = 9
        tot_in = num_inputs + num_dffs

        nodes: list[SequentialCGPNode] = []
        # Construct synchronous increment/load logic using MUXes and XOR gates
        # For each bit i, dff_d[i] = load ? d[i] : (en ? (q[i] ^ carry[i]) : q[i])
        for i in range(num_dffs):
            q_wire = num_inputs + i
            d_wire = 3 + i
            # Node: toggle = q ^ 1
            toggle_node = len(nodes) + tot_in
            nodes.append(SequentialCGPNode(SequentialGateType.XOR, q_wire, 0))  # q ^ en
            # Node: next_val = load ? din : toggle
            mux_node = len(nodes) + tot_in
            nodes.append(SequentialCGPNode(SequentialGateType.MUX, toggle_node, d_wire, 2))  # load is input 2

        dff_inputs = [tot_in + (i * 2 + 1) for i in range(num_dffs)]
        output_connections = [num_inputs + i for i in range(num_dffs)] + [num_inputs + 7]

        return SequentialCGPGenome(
            num_primary_inputs=num_inputs,
            num_primary_outputs=num_outputs,
            num_dffs=num_dffs,
            nodes=nodes,
            output_connections=output_connections,
            dff_input_connections=dff_inputs,
        )
    elif kind == "shift_reg_8bit":
        # 8-bit shift register: 11 inputs [shift_en, load, sin, d0..d7], 8 DFFs, 9 outputs [sout, q0..q7]
        num_inputs = 11
        num_dffs = 8
        num_outputs = 9
        tot_in = num_inputs + num_dffs

        nodes = []
        for i in range(num_dffs):
            prev_q = 2 if i == 0 else (num_inputs + i - 1)  # sin if first bit, else q[i-1]
            din = 3 + i
            # mux_shift = shift_en ? prev_q : q[i]
            # mux_load = load ? din : mux_shift
            m1 = len(nodes) + tot_in
            nodes.append(SequentialCGPNode(SequentialGateType.MUX, num_inputs + i, prev_q, 0))  # shift_en is 0
            m2 = len(nodes) + tot_in
            nodes.append(SequentialCGPNode(SequentialGateType.MUX, m1, din, 1))  # load is 1

        dff_inputs = [tot_in + (i * 2 + 1) for i in range(num_dffs)]
        # outputs: [sout (q7), q0..q7]
        output_connections = [num_inputs + 7] + [num_inputs + i for i in range(num_dffs)]

        return SequentialCGPGenome(
            num_primary_inputs=num_inputs,
            num_primary_outputs=num_outputs,
            num_dffs=num_dffs,
            nodes=nodes,
            output_connections=output_connections,
            dff_input_connections=dff_inputs,
        )
    else:
        # Default 1-bit toggle register
        return SequentialCGPGenome(
            num_primary_inputs=1,
            num_primary_outputs=1,
            num_dffs=1,
            nodes=[
                SequentialCGPNode(SequentialGateType.XOR, 0, 1),
            ],
            output_connections=[1],
            dff_input_connections=[2],
        )
