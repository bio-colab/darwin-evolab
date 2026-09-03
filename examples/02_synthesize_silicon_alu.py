"""
Example 02: Synthesizing a Silicon Hardware ALU & Verilog Export.

Demonstrates Darwin-Evolab's Discrete Logic driver synthesizing a 1-bit full adder
circuit via Cartesian Genetic Programming (CGP) and exporting ready-to-fabricate
Verilog-2001 hardware description code.
"""
from pathlib import Path
import random

from evolab.adapters import get_domain_adapter
from evolab.engine import EvolutionEngine, EngineConfig


def main():
    print("=== Darwin-Evolab: Discrete Logic & Silicon ALU Synthesis ===")

    # 1. Acquire the Discrete Logic driver
    driver = get_domain_adapter("discrete_logic")

    # 2. Define the Boolean Logic specification (1-bit Full Adder)
    spec = driver.parse_spec("Sum = A ^ B ^ Cin; Cout = (A & B) | (Cin & (A ^ B))")
    print(f"Driver Name   : {driver.name}")
    print(f"Inputs/Outputs: {spec.num_inputs} Inputs, {spec.num_outputs} Outputs")
    print(f"Truth Table   : {len(spec.truth_table)} rows")

    # 3. Build population and multi-objective evaluator
    rng = random.Random(42)
    pop = driver.build_population(spec, size=16, rng=rng)
    evaluator = driver.build_evaluator(spec)

    # 4. Run evolutionary optimization
    cfg = EngineConfig(
        population_size=16,
        generations=12,
        seed=42,
        early_stop_fitness=95.0,
    )
    engine = EvolutionEngine(fitness_fn=evaluator, config=cfg)
    result = engine.run(12, initial_population=pop)

    best = engine.best_ever
    score = result.get("best_individual", {}).get("fitness", 0.0)
    print(f"\n[Synthesis Complete]")
    print(f"Best Fitness  : {score:.2f}%")

    # 5. Export Verilog-2001 Silicon Code
    v_path = Path("synthesized_adder.v")
    exported = driver.export_solution(best, spec, output_path=v_path)
    print(f"Active Gates  : {exported['active_gates']}")
    first_few_lines = "\n".join(exported['verilog_code'].splitlines()[:10])
    print(f"Verilog Module:\n{first_few_lines}\n    ...\nendmodule")
    if v_path.is_file():
        v_path.unlink()  # Clean up demo output file


if __name__ == "__main__":
    main()
