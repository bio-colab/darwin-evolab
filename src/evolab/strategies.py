"""Unified Search Strategy Abstraction Layer for Darwin-Evolab.

Provides a pluggable architectural interface for diverse optimization paradigms:
- GeneticAlgorithmStrategy: Population-based, speciation-aware, niching evolutionary search.
- GreedySearchStrategy: Forward-greedy single-candidate discrete catalog exploration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .engine import EvolutionEngine
from .genome import Individual
from .repair import greedy_run_report


class SearchStrategy(ABC):
    """Abstract base class establishing the contract for all optimization strategies."""

    name: str = "base"

    @abstractmethod
    def search(
        self,
        evaluator: Any,
        initial_population: list[Individual] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the search procedure and return a standardized report dictionary."""


class GeneticAlgorithmStrategy(SearchStrategy):
    """Population-based Genetic Algorithm search with speciation and archive."""

    name: str = "ga"

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 50,
        mutation_rate: float = 0.05,
        stagnation_patience: int = 15,
        sharing_mode: str = "dynamic",
        seed: int | None = None,
        early_stop_fitness: float = 99.7,
        engine: EvolutionEngine | None = None,
        **engine_kwargs: Any,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.stagnation_patience = stagnation_patience
        self.sharing_mode = sharing_mode
        self.seed = seed
        self.early_stop_fitness = early_stop_fitness
        self.engine = engine
        self.engine_kwargs = engine_kwargs

    def search(
        self,
        evaluator: Any,
        initial_population: list[Individual] | None = None,
        generations: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        gens = generations if generations is not None else self.generations
        if self.engine is not None:
            engine = self.engine
        else:
            engine_args = dict(self.engine_kwargs)
            if initial_population and "genome_size" not in engine_args:
                try:
                    engine_args["genome_size"] = len(initial_population[0].genome)
                except Exception:
                    pass
            engine = EvolutionEngine(
                fitness_fn=evaluator,
                population_size=self.population_size,
                mutation_rate=self.mutation_rate,
                stagnation_patience=self.stagnation_patience,
                sharing_mode=self.sharing_mode,
                seed=self.seed,
                early_stop_fitness=self.early_stop_fitness,
                **engine_args,
            )

        result = engine.run(gens, initial_population=initial_population)
        return result


class GreedySearchStrategy(SearchStrategy):
    """Greedy stepwise catalog search for discrete program repair and local exploration."""

    name: str = "greedy"

    def __init__(
        self,
        sources: dict[str, str] | None = None,
        target_file: str = "target.py",
        scenario_name: str = "",
        max_evals: int | None = None,
        prioritize_by_suspicion: bool = True,
    ) -> None:
        self.sources = sources or {}
        self.target_file = target_file
        self.scenario_name = scenario_name
        self.max_evals = max_evals
        self.prioritize_by_suspicion = prioritize_by_suspicion

    def search(
        self,
        evaluator: Any,
        initial_population: list[Individual] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        sources = kwargs.get("sources", self.sources)
        target_file = kwargs.get("target_file", self.target_file)
        scenario_name = kwargs.get("scenario_name", self.scenario_name)
        max_evals = kwargs.get("max_evals", self.max_evals)
        prioritize = kwargs.get("prioritize_by_suspicion", self.prioritize_by_suspicion)

        return greedy_run_report(
            sources=sources,
            target_file=target_file,
            evaluator=evaluator,
            scenario_name=scenario_name,
            max_evals=max_evals,
            prioritize_by_suspicion=prioritize,
        )


def get_search_strategy(name: str, **kwargs: Any) -> SearchStrategy:
    """Factory helper to instantiate a search strategy by identifier."""
    name_norm = name.lower().strip()
    if name_norm in ("ga", "genetic", "evolutionary"):
        return GeneticAlgorithmStrategy(**kwargs)
    elif name_norm in ("greedy", "forward_greedy"):
        return GreedySearchStrategy(**kwargs)
    raise ValueError(f"Unknown search strategy: {name!r}. Supported: 'ga', 'greedy'")
