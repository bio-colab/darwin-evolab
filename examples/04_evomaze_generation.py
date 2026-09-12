"""
Example 04: EvoMaze — Search-Based Procedural Content Generation (SBPCG).

Demonstrates how Darwin-Evolab optimizes discrete spatial graphs and tilemaps
to synthesize challenging, 100% solvable mazes targeting specific path lengths,
branching ratios, and dead-end distributions.
"""
import random
from pathlib import Path

from evolab.adapters import get_domain_adapter
from evolab.engine import EvolutionEngine


def main():
    print("=== Darwin-Evolab: Search-Based Maze Generation (EvoMaze) ===\n")

    # 1. Acquire the EvoMaze domain adapter driver
    driver = get_domain_adapter("evomaze")

    # 2. Define spatial optimization targets
    spec = driver.parse_spec({
        "width": 15,
        "height": 15,
        "start": [0, 0],
        "exit": [14, 14],
        "target_path_length": 55.0,  # Require a tortuous, winding path
        "target_dead_ends": 14,     # Require multiple misleading blind alleys
        "target_loops": 1,          # Inject 1 shortcut cycle
        "population_size": 24,
        "generations": 25,
        "seed": 42,
    })

    print(f"Driver Name        : {driver.name}")
    print(f"Grid Dimensions    : {spec.width}x{spec.height} cells")
    print(f"Target Path Length : {spec.target_path_length} steps")
    print(f"Target Dead Ends   : {spec.target_dead_ends}")
    print(f"Target Cycles      : {spec.target_loops}\n")

    # 3. Build evaluator and initial population
    rng = random.Random(spec.seed)
    population = driver.build_population(spec, spec.population_size, rng)
    evaluator = driver.build_evaluator(spec)

    # 4. Run Darwin-Evolab Universal Evolutionary Engine
    engine = EvolutionEngine(
        population_size=spec.population_size,
        seed=spec.seed,
        fitness_fn=evaluator,
    )

    print("Running evolutionary optimization...")
    report = engine.run(spec.generations, initial_population=population)
    best_ind = engine.best_ever or engine.population[0]

    # 5. Export solution artifacts
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "evomaze_sample.json"

    artifacts = driver.export_solution(best_ind, spec, output_path=out_json)
    metrics = artifacts["metrics"]

    print("\n" + "=" * 50)
    print("EVOLVED MAZE VISUALIZATION (S=Start, E=Exit, ..=Path)")
    print("=" * 50)
    print(artifacts["ascii_preview"])
    print("=" * 50)
    print(f"Best Fitness Score  : {best_ind.fitness:.2f}%")
    print(f"Actual Path Length  : {metrics['path_length']} cells")
    print(f"Decision Junctions  : {metrics['junction_count']}")
    print(f"Dead-End Alleys     : {metrics['dead_end_count']} (avg depth: {metrics['avg_dead_end_depth']})")
    print(f"Cycles (Braiding)   : {metrics['cycle_count']}")
    print(f"Guaranteed Solvable : {metrics['solvable']}")
    print(f"Saved Tilemap JSON  : {out_json}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
