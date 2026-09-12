// ============================================================================
// Module: drosophila_hassenstein_reichardt_emd_motif
// Synthesized by Darwin-Evolab Neuromorphic CGP Core
// Biological Provenance: Drosophila Connectome Neural Motif
// Physical Claim: False (Exploratory Biological Digital Model)
// ============================================================================
`default_nettype none

module drosophila_hassenstein_reichardt_emd_motif (
    input  wire        clk,        // Master clock for temporal delay registers
    input  wire        rst,        // Asynchronous reset
    input  wire [1:0] in_spikes,  // Sensory inputs
    output wire [1:0] out_spikes  // Decision outputs
);

    // --- Internal Wires & Registers ---
    wire w_6;

    // --- Sequential Clocked Delay Elements (Synaptic Delay) ---
    // (No clocked delay elements active in this subcircuit)

    // --- Combinational Logic & Synaptic Interactions ---
    assign w_6 = in_spikes[1] & (~in_spikes[0]);

    // --- Circuit Outputs ---
    assign out_spikes[0] = w_6;
    assign out_spikes[1] = in_spikes[0];

endmodule
`default_nettype wire