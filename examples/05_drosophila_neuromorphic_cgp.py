"""
Example 05: Drosophila Neuromorphic CGP — Biological Connectome Circuit Synthesis.

Demonstrates how Darwin-Evolab synthesizes biological neural computation motifs
inspired by the Google/Janelia Drosophila Fruit Fly Connectome (T4/T5 motion detectors)
into temporal neuromorphic circuits and synthesizable Verilog-2001 RTL for FPGA deployment.
"""
from __future__ import annotations

from pathlib import Path
import random

from evolab.adapters import get_domain_adapter
from evolab.engine import EvolutionEngine


def main():
    print("================================================================================")
    print(" Darwin-Evolab: Drosophila Connectome Neuromorphic Circuit Synthesis")
    print(" Biological Benchmark: Hassenstein-Reichardt Elementary Motion Detector (EMD)")
    print(" Provenance: Google/Janelia Drosophila Connectome T4/T5 Visual Selectivity Motif")
    print("================================================================================\n")

    # 1. Retrieve the Neuromorphic Domain Adapter
    adapter = get_domain_adapter("neuromorphic")

    # 2. Configure biological motif specification
    spec = adapter.parse_spec({
        "motif_name": "hassenstein_reichardt_emd",
        "num_inputs": 2,          # Photoreceptors R1 and R2
        "num_outputs": 2,         # Directional responses: [Rightward, Leftward]
        "sequence_length": 32,    # Temporal spike train steps
        "max_nodes": 12,          # Max gates in CGP DAG
        "target_accuracy": 95.0,
        "clocked": True,
    })

    print(f"Motif Name        : {spec.motif_name}")
    print(f"Sensory Inputs    : {spec.num_inputs} (Photoreceptors R1, R2)")
    print(f"Decision Outputs  : {spec.num_outputs} (Right, Left Direction Selectivity)")
    print(f"Spike Sequence    : {spec.sequence_length} timesteps")
    print(f"Max DAG Nodes     : {spec.max_nodes}")
    print(f"Clocked Registers : {spec.clocked} (Z^-1 Synaptic Delay)\n")

    # 3. Build initial population and biological benchmark evaluator
    pop_size = 32
    generations = 20
    seed = 42

    rng = random.Random(seed)
    population = adapter.build_population(spec, pop_size, rng)
    evaluator = adapter.build_evaluator(spec)

    print(f"Synthesizing circuit with EvolutionEngine (pop={pop_size}, gen={generations})...")

    # 4. Run Darwin-Evolab Universal Evolutionary Engine
    engine = EvolutionEngine(
        population_size=pop_size,
        seed=seed,
        fitness_fn=evaluator,
    )

    report = engine.run(generations, initial_population=population)
    best_ind = engine.best_ever or engine.population[0]

    # 5. Export solution artifacts
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_v = out_dir / "drosophila_emd.v"

    solution = adapter.export_solution(best_ind, spec)
    out_v.write_text(solution["verilog_rtl"], encoding="utf-8")

    print("\n" + "=" * 70)
    print("EVOLVED NEUROMORPHIC CIRCUIT SCHEMATIC (ASCII)")
    print("=" * 70)
    print(solution["ascii_diagram"])
    print("=" * 70)
    print(f"Best Fitness Score  : {solution['fitness']:.2f}%")
    print(f"Active Gates (DAG)  : {solution['active_nodes']}")
    print(f"Estimated Transistors: {solution['transistor_count']}")
    print(f"Physical Claim      : {solution['physical_claim']} (Exploratory Biological Digital Model)")
    print(f"Saved Verilog RTL   : {out_v}")
    print("=" * 70 + "\n")

    print("Synthesizable Verilog-2001 Module Snippet:")
    print("--------------------------------------------------------------------------------")
    for line in solution["verilog_rtl"].splitlines()[:25]:
        print(line)
    print("    ... [full module saved to reports/drosophila_emd.v] ...")
    print("--------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
