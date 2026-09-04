"""
adapters.py — Standard DomainAdapter Contract and Canonical Domain Drivers.

Provides the universal driver interface connecting Darwin-Evolab's core evolutionary
engine to domain-specific representations (Software Repair, Silicon & Circuit Design,
Discrete Logic & CGP, and Numerical Optimization).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
from typing import Any, Generic, TypeVar

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .evaluators import Evaluator, FitnessResult, NumericEvaluator
from .genome import EvolabGenome, FloatGenome, Individual

G = TypeVar("G", bound=EvolabGenome)
TSpec = TypeVar("TSpec")
TResult = TypeVar("TResult")


class EvaluatorWrapper(Evaluator):
    """Wraps any duck-typed evaluator or callable into a formal Evaluator instance."""

    def __init__(self, inner: Any, name: str = "wrapped_evaluator") -> None:
        self.inner = inner
        self.name = name

    @property
    def deterministic(self) -> bool:
        return getattr(self.inner, "deterministic", True)

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        if hasattr(self.inner, "evaluate"):
            try:
                res = self.inner.evaluate(target, context) if context is not None else self.inner.evaluate(target)
            except (TypeError, AttributeError):
                inner_target = getattr(target, "genome", target)
                res = self.inner.evaluate(inner_target)
            else:
                if res == 0.0 and hasattr(target, "genome"):
                    # Retry with unwrapped genome in case inner checks type strictly
                    retry_res = self.inner.evaluate(target.genome)
                    if retry_res != 0.0:
                        res = retry_res

            if isinstance(res, FitnessResult):
                return res
            return FitnessResult(score=float(res))
        score = float(self.inner(target))
        return FitnessResult(score=score)


class DomainAdapter(ABC, Generic[G, TSpec, TResult]):
    """The canonical driver interface that every Darwin-Evolab domain adapter must fulfill.

    Acts like an operating system device driver: translates raw real-world specifications
    into genomes and evaluators, and translates winning genomes back into deployable domain artifacts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique domain identifier (e.g., 'software_repair', 'electronics', 'discrete_logic', 'numerical_math')."""

    @abstractmethod
    def parse_spec(self, raw_input: Any) -> TSpec:
        """Parses and validates domain-specific input data into a structured specification."""

    @abstractmethod
    def build_population(self, spec: TSpec, size: int, rng: random.Random) -> list[Individual]:
        """Initializes a valid population of domain genomes conforming to the specification."""

    @abstractmethod
    def build_evaluator(self, spec: TSpec) -> Evaluator:
        """Constructs a deterministic or physics-grounded evaluator for this domain."""

    @abstractmethod
    def export_solution(
        self,
        individual: Individual,
        spec: TSpec,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Exports the winning genome into domain-specific deployable artifacts (e.g. patch, Verilog, SPICE, report)."""


# =========================================================================== #
# Canonical Domain Driver 1: Software Repair Adapter
# =========================================================================== #

@dataclass(frozen=True)
class SoftwareRepairSpec:
    sources: dict[str, str]
    target_file: str
    tests: list[tuple[Any, Any]]
    func_name: str
    use_sandbox: bool = True


class SoftwareRepairAdapter(DomainAdapter):
    """Domain driver for Automated Program Repair (APR) on Python source code via AST."""

    @property
    def name(self) -> str:
        return "software_repair"

    def parse_spec(self, raw_input: Any) -> SoftwareRepairSpec:
        from .code_fixtures import CodeScenario, load_scenario_file

        if isinstance(raw_input, SoftwareRepairSpec):
            return raw_input
        elif isinstance(raw_input, CodeScenario):
            return SoftwareRepairSpec(
                sources=dict(raw_input.sources),
                target_file=raw_input.target_file,
                tests=list(raw_input.tests),
                func_name=raw_input.func_name,
            )
        elif isinstance(raw_input, (str, Path)) and Path(raw_input).is_file():
            sc = load_scenario_file(raw_input)
            return self.parse_spec(sc)
        elif isinstance(raw_input, dict):
            return SoftwareRepairSpec(
                sources=dict(raw_input.get("sources", {"main.py": ""})),
                target_file=str(raw_input.get("target_file", "main.py")),
                tests=list(raw_input.get("tests", [])),
                func_name=str(raw_input.get("func_name", "solve")),
                use_sandbox=bool(raw_input.get("use_sandbox", True)),
            )
        raise TypeError(f"Cannot parse software repair spec from {type(raw_input)}")

    def build_population(self, spec: SoftwareRepairSpec, size: int, rng: random.Random) -> list[Individual]:
        from .repair import RepairGenome, catalog_sources

        edits = catalog_sources(spec.sources)
        pop: list[Individual] = [
            Individual(
                genome=RepairGenome(sources=dict(spec.sources), target_file=spec.target_file, edits=[]),
                species="spec_software_repair",
            )
        ]
        if edits:
            for _ in range(size - 1):
                sample_n = min(len(edits), rng.randint(1, 3))
                chosen = rng.sample(edits, sample_n)
                pop.append(
                    Individual(
                        genome=RepairGenome(sources=dict(spec.sources), target_file=spec.target_file, edits=chosen),
                        species="spec_software_repair",
                    )
                )
        else:
            while len(pop) < size:
                pop.append(pop[0].clone())
        return pop

    def build_evaluator(self, spec: SoftwareRepairSpec) -> Evaluator:
        from .evaluators import FunctionTestEvaluator, SandboxFunctionTestEvaluator

        if spec.use_sandbox:
            return SandboxFunctionTestEvaluator(
                base_sources=spec.sources,
                target_file=spec.target_file,
                func_name=spec.func_name,
                test_cases=spec.tests,
            )
        return FunctionTestEvaluator(
            base_sources=spec.sources,
            target_file=spec.target_file,
            func_name=spec.func_name,
            test_cases=spec.tests,
        )

    def export_solution(
        self,
        individual: Individual,
        spec: SoftwareRepairSpec,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        from .reporters import format_git_patch

        genome = individual.genome
        repaired_code = genome.to_code() if hasattr(genome, "to_code") else ""
        repaired_sources = dict(spec.sources)
        repaired_sources[spec.target_file] = repaired_code

        diff_patch = format_git_patch(spec, repaired_sources)
        if output_path:
            p = Path(output_path)
            if p.parent and str(p.parent):
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(diff_patch, encoding="utf-8")

        return {
            "repaired_code": repaired_code,
            "git_patch": diff_patch,
            "target_file": spec.target_file,
            "edits_count": len(getattr(genome, "edits", [])),
        }


# =========================================================================== #
# Canonical Domain Driver 2: Discrete Logic Adapter (CGP)
# =========================================================================== #

@dataclass(frozen=True)
class DiscreteLogicSpec:
    truth_table: list[tuple[tuple[int, ...], tuple[int, ...]]]
    num_inputs: int
    num_outputs: int
    name: str = "SynthesizedLogic"
    objective: str = "balanced"


class DiscreteLogicAdapter(DomainAdapter):
    """Domain driver for synthesizable Cartesian Genetic Programming (CGP) and Verilog."""

    @property
    def name(self) -> str:
        return "discrete_logic"

    def parse_spec(self, raw_input: Any) -> DiscreteLogicSpec:
        from experimental.electronics.inputs.boolean_expr import parse_boolean_spec

        if isinstance(raw_input, DiscreteLogicSpec):
            return raw_input
        elif isinstance(raw_input, str):
            res = parse_boolean_spec(raw_input)
            return DiscreteLogicSpec(
                truth_table=res.truth_table,
                num_inputs=res.num_inputs,
                num_outputs=res.num_outputs,
                name="ParsedLogic",
            )
        elif isinstance(raw_input, dict) and "truth_table" in raw_input:
            tt = raw_input["truth_table"]
            return DiscreteLogicSpec(
                truth_table=tt,
                num_inputs=int(raw_input.get("num_inputs", len(tt[0][0]))),
                num_outputs=int(raw_input.get("num_outputs", len(tt[0][1]))),
                name=str(raw_input.get("name", "CustomLogic")),
                objective=str(raw_input.get("objective", "balanced")),
            )
        raise TypeError(f"Cannot parse discrete logic spec from {type(raw_input)}")

    def build_population(self, spec: DiscreteLogicSpec, size: int, rng: random.Random) -> list[Individual]:
        from .cgp_logic import create_random_cgp_genome

        pop = [
            Individual(
                genome=create_random_cgp_genome(
                    num_inputs=spec.num_inputs,
                    num_outputs=spec.num_outputs,
                    num_nodes=max(10, spec.num_inputs * 4),
                    rng=rng,
                ),
                species="spec_logic",
            )
            for _ in range(size)
        ]
        return pop

    def build_evaluator(self, spec: DiscreteLogicSpec) -> Evaluator:
        from .cgp_logic import LowPowerALUEvaluator

        raw_evaluator = LowPowerALUEvaluator(truth_table=spec.truth_table, target_name=spec.name)
        return EvaluatorWrapper(raw_evaluator, name="DiscreteLogicEvaluator")

    def export_solution(
        self,
        individual: Individual,
        spec: DiscreteLogicSpec,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        genome = individual.genome
        verilog_code = ""
        if hasattr(genome, "to_verilog"):
            verilog_code = genome.to_verilog(module_name=spec.name)
            if output_path:
                p = Path(output_path)
                if p.parent and str(p.parent):
                    p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(verilog_code, encoding="utf-8")

        return {
            "verilog_code": verilog_code,
            "active_gates": len(genome.get_active_nodes()) if hasattr(genome, "get_active_nodes") else 0,
            "module_name": spec.name,
        }


# =========================================================================== #
# Canonical Domain Driver 3: Numerical Math Optimization Adapter
# =========================================================================== #

@dataclass(frozen=True)
class NumericalMathSpec:
    target_function: str = "rastrigin"
    dimensions: int = 4
    bounds: tuple[float, float] = (-5.12, 5.12)


class NumericalMathAdapter(DomainAdapter):
    """Domain driver for continuous numerical optimization on standard mathematical landscapes."""

    @property
    def name(self) -> str:
        return "numerical_math"

    def parse_spec(self, raw_input: Any) -> NumericalMathSpec:
        if isinstance(raw_input, NumericalMathSpec):
            return raw_input
        elif isinstance(raw_input, str):
            return NumericalMathSpec(target_function=raw_input.lower().strip())
        elif isinstance(raw_input, dict):
            return NumericalMathSpec(
                target_function=str(raw_input.get("target_function", "rastrigin")),
                dimensions=int(raw_input.get("dimensions", 4)),
                bounds=tuple(raw_input.get("bounds", (-5.12, 5.12))),
            )
        return NumericalMathSpec()

    def build_population(self, spec: NumericalMathSpec, size: int, rng: random.Random) -> list[Individual]:
        low, high = spec.bounds
        pop = [
            Individual(
                genome=FloatGenome([rng.uniform(low, high) for _ in range(spec.dimensions)]),
                species="spec_math",
            )
            for _ in range(size)
        ]
        return pop

    def build_evaluator(self, spec: NumericalMathSpec) -> Evaluator:
        def rastrigin(target: Any) -> float:
            coords = list(getattr(target, "genes", getattr(getattr(target, "genome", None), "genes", target)))
            cost = 10.0 * len(coords) + sum(x**2 - 10.0 * math.cos(2.0 * math.pi * x) for x in coords)
            return max(0.0, 100.0 - cost)

        def rosenbrock(target: Any) -> float:
            coords = list(getattr(target, "genes", getattr(getattr(target, "genome", None), "genes", target)))
            if len(coords) < 2:
                return 100.0
            cost = sum(100.0 * (coords[i+1] - coords[i]**2)**2 + (1.0 - coords[i])**2 for i in range(len(coords) - 1))
            return max(0.0, 100.0 / (1.0 + cost))

        fn_map = {
            "rastrigin": rastrigin,
            "rosenbrock": rosenbrock,
        }
        chosen_fn = fn_map.get(spec.target_function.lower(), rastrigin)
        return NumericEvaluator(fn=chosen_fn, name=f"numeric_{spec.target_function}")

    def build_vectorized_evaluator(self, spec: NumericalMathSpec, use_jax: bool = False):
        """Build high-performance vectorized evaluator supporting batch evaluation on SIMD/GPU."""
        from .vectorized import VectorizedLandscapeEvaluator
        return VectorizedLandscapeEvaluator(landscape=spec.target_function, use_jax=use_jax)

    def export_solution(
        self,
        individual: Individual,
        spec: NumericalMathSpec,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        genome = individual.genome
        coords = list(getattr(genome, "genes", []))
        data = {
            "target_function": spec.target_function,
            "optimal_coordinates": coords,
            "dimensions": len(coords),
        }
        if output_path:
            p = Path(output_path)
            if p.parent and str(p.parent):
                p.parent.mkdir(parents=True, exist_ok=True)
            import json
            p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return data


# =========================================================================== #
# Canonical Domain Driver 4: Electronics & Silicon Adapter
# =========================================================================== #

@dataclass(frozen=True)
class ElectronicsSpec:
    scenario_or_input: str | dict[str, Any]
    population_size: int = 16
    seed: int | None = None


class ElectronicsAdapter(DomainAdapter):
    """Domain driver for physical and discrete electronics design (SPICE, Breadboards, CGP)."""

    @property
    def name(self) -> str:
        return "electronics"

    def parse_spec(self, raw_input: Any) -> ElectronicsSpec:
        if isinstance(raw_input, ElectronicsSpec):
            return raw_input
        return ElectronicsSpec(scenario_or_input=raw_input)

    def build_population(self, spec: ElectronicsSpec, size: int, rng: random.Random) -> list[Individual]:
        from experimental.electronics.scenarios import prepare_electronics_run, prepare_custom_electronics_run, list_electronics_scenarios

        target = spec.scenario_or_input
        if isinstance(target, str) and target in list_electronics_scenarios():
            _, pop, _ = prepare_electronics_run(target, size, spec.seed)
            return pop
        elif isinstance(target, str):
            _, pop, _ = prepare_custom_electronics_run(expr=target, population_size=size, seed=spec.seed)
            return pop
        elif isinstance(target, dict):
            _, pop, _ = prepare_custom_electronics_run(expr=target.get("expr"), population_size=size, seed=spec.seed)
            return pop
        # Fallback default
        _, pop, _ = prepare_electronics_run("half_adder", size, spec.seed)
        return pop

    def build_evaluator(self, spec: ElectronicsSpec) -> Evaluator:
        from experimental.electronics.scenarios import prepare_electronics_run, prepare_custom_electronics_run, list_electronics_scenarios

        target = spec.scenario_or_input
        ev: Any = None
        if isinstance(target, str) and target in list_electronics_scenarios():
            ev, _, _ = prepare_electronics_run(target, spec.population_size, spec.seed)
        elif isinstance(target, str):
            ev, _, _ = prepare_custom_electronics_run(expr=target, population_size=spec.population_size, seed=spec.seed)
        elif isinstance(target, dict):
            ev, _, _ = prepare_custom_electronics_run(expr=target.get("expr"), population_size=spec.population_size, seed=spec.seed)
        else:
            ev, _, _ = prepare_electronics_run("half_adder", spec.population_size, spec.seed)

        if isinstance(ev, Evaluator):
            return ev
        return EvaluatorWrapper(ev, name="ElectronicsEvaluator")

    def export_solution(
        self,
        individual: Individual,
        spec: ElectronicsSpec,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        from experimental.electronics.instruments.schematic import circuit_to_svg

        genome = individual.genome
        svg_str = ""
        verilog_str = ""
        if hasattr(genome, "to_verilog"):
            verilog_str = genome.to_verilog()
        try:
            svg_str = circuit_to_svg(genome)
        except Exception:
            pass

        return {
            "verilog_code": verilog_str,
            "has_schematic": bool(svg_str),
            "genome_type": type(genome).__name__,
        }


# =========================================================================== #
# Central Domain Adapter Registry
# =========================================================================== #

def _init_registry() -> dict[str, DomainAdapter]:
    from .swe_bench import SWEBenchAdapter

    return {
        "software_repair": SoftwareRepairAdapter(),
        "discrete_logic": DiscreteLogicAdapter(),
        "numerical_math": NumericalMathAdapter(),
        "electronics": ElectronicsAdapter(),
        "swe_bench": SWEBenchAdapter(),
    }

_ADAPTER_REGISTRY: dict[str, DomainAdapter] = _init_registry()


def register_domain_adapter(name: str, adapter: DomainAdapter) -> None:
    """Registers a custom domain adapter driver into Darwin-Evolab."""
    if not isinstance(adapter, DomainAdapter):
        raise TypeError(f"Expected DomainAdapter instance, got {type(adapter)}")
    _ADAPTER_REGISTRY[name.lower().strip()] = adapter


def get_domain_adapter(name: str) -> DomainAdapter:
    """Retrieves a registered domain adapter by name."""
    key = name.lower().strip()
    if key not in _ADAPTER_REGISTRY:
        raise KeyError(f"Unknown domain adapter {name!r}. Available: {list(_ADAPTER_REGISTRY.keys())}")
    return _ADAPTER_REGISTRY[key]


def list_domain_adapters() -> list[str]:
    """Returns the list of all registered domain adapter names."""
    return sorted(_ADAPTER_REGISTRY.keys())
