# Drosophila Connectome Neuromorphic CGP (`experimental/neuromorphic/`)

> **Status: FROZEN / EXPLORATORY RESEARCH TRACK (`physical_claim: false`)**  
> *This exploratory module is preserved in a frozen state under `experimental/` for academic reproducibility. Core development is focused on Pillar 1 (Software APR) and Pillar 2 (Digital CGP).*

---

## 1. Overview
`experimental/neuromorphic` synthesizes temporal neural computation motifs inspired by the Google / Janelia Drosophila fruit fly connectome:
- **Temporal CGP**: Incorporates clocked delay registers ($Z^{-1}$ D-Flip-Flops) and inhibitory synapses ($A \wedge \neg B$) to process temporal spike trains.
- **Connectome Motifs**: Evaluates against biological motion selectivity (T4/T5 Hassenstein-Reichardt Elementary Motion Detectors) and antennal lobe lateral inhibition.
- **FPGA Export**: Emits synthesizable Verilog-2001 RTL modules and ASCII schematics.
