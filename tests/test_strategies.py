"""Tests for the unified SearchStrategy abstraction layer."""
from __future__ import annotations

import pytest
from evolab.strategies import (
    SearchStrategy,
    GeneticAlgorithmStrategy,
    GreedySearchStrategy,
    get_search_strategy,
)
from evolab.genome import Individual, FloatGenome
from evolab.evaluators import Evaluator, FitnessResult


class MockSphereEvaluator(Evaluator):
    """Simple continuous sphere evaluator for testing strategy execution."""

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, individual: Individual) -> FitnessResult:
        genes = individual.genome.genes
        # Maximize 100 - sum(g^2)
        score = max(0.0, 100.0 - sum(x * x for x in genes))
        return FitnessResult(score=score)


class MockRepairEvaluator:
    """Mock evaluator for greedy repair."""

    def evaluate(self, genome):
        code = genome.to_code() if hasattr(genome, "to_code") else str(genome)
        if "==" in code:
            return FitnessResult(score=100.0, passed_holdout=True)
        return FitnessResult(score=50.0, passed_holdout=True)


def test_search_strategy_factory():
    ga = get_search_strategy("ga", population_size=10)
    assert isinstance(ga, GeneticAlgorithmStrategy)
    assert isinstance(ga, SearchStrategy)
    assert ga.population_size == 10

    greedy = get_search_strategy("greedy", target_file="main.py")
    assert isinstance(greedy, GreedySearchStrategy)
    assert isinstance(greedy, SearchStrategy)
    assert greedy.target_file == "main.py"

    with pytest.raises(ValueError, match="Unknown search strategy"):
        get_search_strategy("quantum_superposition")


def test_ga_search_strategy_execution():
    evaluator = MockSphereEvaluator()
    strategy = GeneticAlgorithmStrategy(
        population_size=6,
        generations=3,
        seed=42,
    )
    initial_pop = [
        Individual(genome=FloatGenome([1.0, 2.0]), species="spec_mock")
        for _ in range(6)
    ]

    result = strategy.search(evaluator=evaluator, initial_population=initial_pop)

    assert "best_individual" in result
    assert "history" in result
    assert result["total_generations"] == 3
    assert result["total_candidates_evaluated"] > 0
    assert result["best_individual"]["fitness"] >= 0.0


def test_greedy_search_strategy_execution():
    sources = {"app.py": "def test(x):\n    if x != 1:\n        return True\n    return False\n"}
    evaluator = MockRepairEvaluator()

    strategy = GreedySearchStrategy(
        sources=sources,
        target_file="app.py",
        scenario_name="test_greedy",
        max_evals=5,
    )

    result = strategy.search(evaluator=evaluator)

    assert "best_individual" in result
    assert result["config"]["search"] == "greedy_forward"
    assert result["config"]["scenario"] == "test_greedy"
    assert result["best_individual"]["fitness"] >= 50.0
