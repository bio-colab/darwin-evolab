"""
exporter.py — Hardware Verilog RTL & ASCII graph exporter for evolved neural circuits.

Converts evolved neuromorphic DAGs into synthesizable Verilog-2001 RTL modules
suitable for FPGA compilation (Yosys / Vivado) or ASIC design.
"""
from __future__ import annotations

from typing import Any
from .genome import NeuromorphicCircuitGenome, NeuromorphicOp
from .spec import NeuromorphicSpec


class NeuromorphicExporter:
    """Exports evolved neural motifs to visual ASCII and synthesizable Verilog RTL."""

    @staticmethod
    def export_ascii(genome: NeuromorphicCircuitGenome, spec: NeuromorphicSpec) -> str:
        lines = [
            f"=== Evolved Drosophila Neuromorphic Circuit: {spec.motif_name} ===",
            f"Inputs: {genome.num_inputs} | Outputs: {genome.num_outputs} | Total Transistors: {genome.transistor_count()}",
            "----------------------------------------------------------------",
        ]

        active_indices = genome.active_node_indices()
        for node in genome.nodes:
            status = "ACTIVE" if node.index in active_indices else "neutral"
            in_a_name = f"IN[{node.input_a}]" if node.input_a < genome.num_inputs else f"N{node.input_a}"
            in_b_name = f"IN[{node.input_b}]" if node.input_b < genome.num_inputs else f"N{node.input_b}"

            if node.op == NeuromorphicOp.NOT:
                op_desc = f"NOT({in_a_name})"
            elif node.op == NeuromorphicOp.DELAY:
                op_desc = f"DELAY[Z^-1]({in_a_name})"
            elif node.op == NeuromorphicOp.WIRE:
                op_desc = f"WIRE({in_a_name})"
            elif node.op == NeuromorphicOp.INHIBIT:
                op_desc = f"{in_a_name} INHIBIT_BY {in_b_name}"
            else:
                op_desc = f"{in_a_name} {node.op.value} {in_b_name}"

            lines.append(f"  [{status:7s}] Node N{node.index:02d} = {op_desc}")

        lines.append("----------------------------------------------------------------")
        for i, out_idx in enumerate(genome.output_indices):
            out_src = f"IN[{out_idx}]" if out_idx < genome.num_inputs else f"N{out_idx}"
            lines.append(f"  OUTPUT[{i}] <= {out_src}")
        lines.append("================================================================")
        return "\n".join(lines)

    @staticmethod
    def export_verilog(
        genome: NeuromorphicCircuitGenome,
        module_name: str = "drosophila_motion_detector",
    ) -> str:
        """
        Generates synthesizable Verilog-2001 RTL for the evolved neuromorphic motif.
        Clocked DELAY nodes are generated as synchronous D-Flip-Flop registers.
        """
        active_nodes = [n for n in genome.nodes if n.index in genome.active_node_indices()]
        delay_nodes = [n for n in active_nodes if n.op == NeuromorphicOp.DELAY]
        comb_nodes = [n for n in active_nodes if n.op != NeuromorphicOp.DELAY]

        lines = [
            "// ============================================================================",
            f"// Module: {module_name}",
            "// Synthesized by Darwin-Evolab Neuromorphic CGP Core",
            "// Biological Provenance: Drosophila Connectome Neural Motif",
            "// Physical Claim: False (Exploratory Biological Digital Model)",
            "// ============================================================================",
            "`default_nettype none",
            "",
            f"module {module_name} (",
            "    input  wire        clk,        // Master clock for temporal delay registers",
            "    input  wire        rst,        // Asynchronous reset",
            f"    input  wire [{genome.num_inputs - 1}:0] in_spikes,  // Sensory inputs",
            f"    output wire [{genome.num_outputs - 1}:0] out_spikes  // Decision outputs",
            ");",
            "",
            "    // --- Internal Wires & Registers ---",
        ]

        # Declare registers for delay elements
        for d in delay_nodes:
            lines.append(f"    reg  w_{d.index}_reg;")
            lines.append(f"    wire w_{d.index} = w_{d.index}_reg;")

        # Declare wires for combinational nodes
        for c in comb_nodes:
            lines.append(f"    wire w_{c.index};")

        lines.append("")
        lines.append("    // --- Sequential Clocked Delay Elements (Synaptic Delay) ---")
        if delay_nodes:
            lines.extend([
                "    always @(posedge clk or posedge rst) begin",
                "        if (rst) begin",
            ])
            for d in delay_nodes:
                lines.append(f"            w_{d.index}_reg <= 1'b0;")
            lines.extend([
                "        end else begin",
            ])
            for d in delay_nodes:
                src = f"in_spikes[{d.input_a}]" if d.input_a < genome.num_inputs else f"w_{d.input_a}"
                lines.append(f"            w_{d.index}_reg <= {src};")
            lines.extend([
                "        end",
                "    end",
                "",
            ])
        else:
            lines.append("    // (No clocked delay elements active in this subcircuit)")
            lines.append("")

        lines.append("    // --- Combinational Logic & Synaptic Interactions ---")

        def wire_ref(idx: int) -> str:
            if idx < genome.num_inputs:
                return f"in_spikes[{idx}]"
            return f"w_{idx}"

        for c in comb_nodes:
            wa = wire_ref(c.input_a)
            wb = wire_ref(c.input_b)
            if c.op == NeuromorphicOp.AND:
                expr = f"{wa} & {wb}"
            elif c.op == NeuromorphicOp.OR:
                expr = f"{wa} | {wb}"
            elif c.op == NeuromorphicOp.XOR:
                expr = f"{wa} ^ {wb}"
            elif c.op == NeuromorphicOp.NOT:
                expr = f"~{wa}"
            elif c.op == NeuromorphicOp.INHIBIT:
                expr = f"{wa} & (~{wb})"
            elif c.op == NeuromorphicOp.WIRE:
                expr = f"{wa}"
            else:
                expr = f"{wa}"
            lines.append(f"    assign w_{c.index} = {expr};")

        lines.append("")
        lines.append("    // --- Circuit Outputs ---")
        for i, out_idx in enumerate(genome.output_indices):
            src = wire_ref(out_idx)
            lines.append(f"    assign out_spikes[{i}] = {src};")

        lines.extend([
            "",
            "endmodule",
            "`default_nettype wire",
        ])

        return "\n".join(lines)

    @staticmethod
    def export_json(
        genome: NeuromorphicCircuitGenome,
        spec: NeuromorphicSpec,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "motif_name": spec.motif_name,
            "provenance": "Drosophila_Connectome_CGP",
            "physical_claim": False,
            "genome": genome.serialize(),
            "metrics": metrics or {},
            "active_nodes_count": len(genome.active_nodes()),
            "transistor_count": genome.transistor_count(),
        }
