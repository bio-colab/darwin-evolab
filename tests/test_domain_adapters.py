"""
Tests for DomainAdapter architecture and canonical domain drivers.
"""
from __future__ import annotations

import random
from pathlib import Path
import pytest

from evolab.adapters import (
    DomainAdapter,
    SoftwareRepairAdapter,
    DiscreteLogicAdapter,
    NumericalMathAdapter,
    ElectronicsAdapter,
    get_domain_adapter,
    list_domain_adapters,
    register_domain_adapter,
)
from evolab.genome import Individual, FloatGenome
from evolab.evaluators import Evaluator, FitnessResult


def test_domain_adapter_registry():
    adapters = list_domain_adapters()
    assert "software_repair" in adapters
    assert "discrete_logic" in adapters
    assert "numerical_math" in adapters
    assert "electronics" in adapters

    drv = get_domain_adapter("software_repair")
    assert isinstance(drv, SoftwareRepairAdapter)

    # Unknown domain raises KeyError
    with pytest.raises(KeyError):
        get_domain_adapter("unknown_domain")


def test_software_repair_adapter_lifecycle(tmp_path):
    adapter = get_domain_adapter("software_repair")
    raw_spec = {
        "sources": {"calc.py": "def add(a, b): return a - b\n"},
        "target_file": "calc.py",
        "tests": [((1, 2), 3), ((5, 5), 10)],
        "func_name": "add",
        "use_sandbox": False,
    }
    spec = adapter.parse_spec(raw_spec)
    assert spec.target_file == "calc.py"

    rng = random.Random(42)
    pop = adapter.build_population(spec, size=4, rng=rng)
    assert len(pop) == 4
    assert pop[0].species == "spec_software_repair"

    evaluator = adapter.build_evaluator(spec)
    assert isinstance(evaluator, Evaluator)

    res = evaluator.evaluate(pop[0])
    assert isinstance(res, FitnessResult)

    out_patch = tmp_path / "fix.patch"
    exported = adapter.export_solution(pop[0], spec, output_path=out_patch)
    assert "repaired_code" in exported
    assert out_patch.is_file()


def test_discrete_logic_adapter_lifecycle(tmp_path):
    adapter = get_domain_adapter("discrete_logic")
    # Half-adder equations
    spec = adapter.parse_spec("Sum = A ^ B; Cout = A & B")
    assert spec.num_inputs == 2
    assert spec.num_outputs == 2

    rng = random.Random(42)
    pop = adapter.build_population(spec, size=4, rng=rng)
    assert len(pop) == 4
    assert pop[0].species == "spec_logic"

    evaluator = adapter.build_evaluator(spec)
    assert isinstance(evaluator, Evaluator)

    res = evaluator.evaluate(pop[0])
    assert isinstance(res, FitnessResult)

    v_file = tmp_path / "logic.v"
    exported = adapter.export_solution(pop[0], spec, output_path=v_file)
    assert "verilog_code" in exported
    assert v_file.is_file()


def test_numerical_math_adapter_lifecycle(tmp_path):
    adapter = get_domain_adapter("numerical_math")
    spec = adapter.parse_spec({"target_function": "rastrigin", "dimensions": 3})
    assert spec.dimensions == 3

    rng = random.Random(42)
    pop = adapter.build_population(spec, size=4, rng=rng)
    assert len(pop) == 4
    assert pop[0].species == "spec_math"

    evaluator = adapter.build_evaluator(spec)
    res = evaluator.evaluate(pop[0])
    assert isinstance(res, FitnessResult)

    res_json = tmp_path / "coords.json"
    exported = adapter.export_solution(pop[0], spec, output_path=res_json)
    assert len(exported["optimal_coordinates"]) == 3
    assert res_json.is_file()


def test_electronics_adapter_lifecycle():
    adapter = get_domain_adapter("electronics")
    spec = adapter.parse_spec("half_adder")
    assert spec.scenario_or_input == "half_adder"

    rng = random.Random(42)
    pop = adapter.build_population(spec, size=2, rng=rng)
    assert len(pop) == 2

    evaluator = adapter.build_evaluator(spec)
    assert isinstance(evaluator, Evaluator)

    res = evaluator.evaluate(pop[0])
    assert isinstance(res, FitnessResult)

    exported = adapter.export_solution(pop[0], spec)
    assert "genome_type" in exported


def test_custom_domain_adapter_registration():
    class DummyGenome(FloatGenome):
        pass

    class DummyAdapter(DomainAdapter):
        @property
        def name(self) -> str:
            return "dummy_domain"

        def parse_spec(self, raw_input: Any) -> str:
            return str(raw_input)

        def build_population(self, spec: str, size: int, rng: random.Random) -> list[Individual]:
            return [Individual(genome=DummyGenome([0.0]), species="dummy") for _ in range(size)]

        def build_evaluator(self, spec: str) -> Evaluator:
            class DummyEval(Evaluator):
                def evaluate(self, ind: Individual) -> FitnessResult:
                    return FitnessResult(score=100.0)
            return DummyEval()

        def export_solution(self, ind: Individual, spec: str, output_path: str | Path | None = None) -> dict[str, Any]:
            return {"status": "ok"}

    register_domain_adapter("dummy_domain", DummyAdapter())
    assert "dummy_domain" in list_domain_adapters()
    retrieved = get_domain_adapter("dummy_domain")
    assert isinstance(retrieved, DummyAdapter)
