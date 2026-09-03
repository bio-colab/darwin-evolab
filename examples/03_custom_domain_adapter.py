"""
Example 03: Building a Custom Domain Adapter Driver in 15 Minutes.

Demonstrates how external researchers can plug any custom engineering or scientific
domain into Darwin-Evolab by subclassing the standard `DomainAdapter` interface.
"""
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from evolab.adapters import DomainAdapter, register_domain_adapter, get_domain_adapter
from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import FloatGenome, Individual
from evolab.engine import EvolutionEngine, EngineConfig


# 1. Define your domain's specification model
@dataclass(frozen=True)
class ThermalCoolingSpec:
    target_temperature_c: float
    ambient_temperature_c: float
    max_power_watts: float


# 2. Implement the DomainAdapter driver contract
class ThermalCoolingAdapter(DomainAdapter):
    """Domain driver for optimizing heatsink and thermal cooling parameters."""

    @property
    def name(self) -> str:
        return "thermal_cooling"

    def parse_spec(self, raw_input: Any) -> ThermalCoolingSpec:
        if isinstance(raw_input, dict):
            return ThermalCoolingSpec(
                target_temperature_c=float(raw_input.get("target_temp", 45.0)),
                ambient_temperature_c=float(raw_input.get("ambient_temp", 25.0)),
                max_power_watts=float(raw_input.get("max_power", 65.0)),
            )
        return ThermalCoolingSpec(45.0, 25.0, 65.0)

    def build_population(self, spec: ThermalCoolingSpec, size: int, rng: random.Random) -> list[Individual]:
        # Genome represents: [fin_count, fin_thickness_mm, fan_rpm_thousands]
        pop = [
            Individual(
                genome=FloatGenome([rng.uniform(10, 60), rng.uniform(0.5, 3.0), rng.uniform(1.0, 5.0)]),
                species="spec_thermal",
            )
            for _ in range(size)
        ]
        return pop

    def build_evaluator(self, spec: ThermalCoolingSpec) -> Evaluator:
        class ThermalPhysicsEvaluator(Evaluator):
            @property
            def deterministic(self) -> bool:
                return True

            def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
                genome = getattr(target, "genome", target)
                fins, thickness, rpm = genome.values

                # Simplified physical heat dissipation model
                airflow_cfm = rpm * 12.0
                surface_area_cm2 = fins * 25.0 * (1.0 + thickness * 0.1)
                thermal_resistance = 80.0 / (surface_area_cm2 * (airflow_cfm ** 0.5) + 1e-6)
                steady_state_temp = spec.ambient_temperature_c + (spec.max_power_watts * thermal_resistance)

                # Fitness: how closely it hits target temp without over-cooling
                error = abs(steady_state_temp - spec.target_temperature_c)
                score = max(0.0, 100.0 - error * 2.0)
                return FitnessResult(
                    score=score,
                    sub_scores={"temp_c": steady_state_temp, "thermal_resistance": thermal_resistance},
                )

        return ThermalPhysicsEvaluator()

    def export_solution(
        self, individual: Individual, spec: ThermalCoolingSpec, output_path: str | Path | None = None
    ) -> dict[str, Any]:
        fins, thickness, rpm = individual.genome.values
        report = {
            "optimal_fin_count": int(round(fins)),
            "optimal_thickness_mm": round(thickness, 2),
            "optimal_fan_rpm": int(round(rpm * 1000)),
            "target_temperature_c": spec.target_temperature_c,
        }
        return report


def main():
    print("=== Darwin-Evolab: Custom Domain Driver Tutorial ===")

    # 1. Register the custom domain driver
    adapter = ThermalCoolingAdapter()
    register_domain_adapter("thermal_cooling", adapter)

    # 2. Retrieve driver from central registry
    driver = get_domain_adapter("thermal_cooling")
    spec = driver.parse_spec({"target_temp": 42.0, "ambient_temp": 24.0, "max_power": 95.0})
    print(f"Driver Name: {driver.name}")
    print(f"Spec Target: {spec.target_temperature_c}°C under {spec.max_power_watts}W load")

    # 3. Build population and evaluator
    rng = random.Random(123)
    pop = driver.build_population(spec, size=16, rng=rng)
    evaluator = driver.build_evaluator(spec)

    # 4. Run optimization with Darwin-Evolab's core engine
    cfg = EngineConfig(population_size=16, generations=15, seed=123, genome_size=3, early_stop_fitness=98.0)
    engine = EvolutionEngine(fitness_fn=evaluator, config=cfg)
    result = engine.run(15, initial_population=pop)

    best = engine.best_ever
    print(f"\n[Optimization Complete]")
    print(f"Best Fitness: {result['best_individual']['fitness']:.2f}%")

    # 5. Export solution
    solution = driver.export_solution(best, spec)
    print(f"Optimal Design: {solution}")


if __name__ == "__main__":
    main()
