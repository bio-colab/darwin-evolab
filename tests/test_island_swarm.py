"""test_island_swarm.py — Verification of Deliberate Swarm Scaling & Burden-Gated Utility Invariant.

Verifies:
1. Tagging contract (PROBATIONARY_BURDEN_GATED).
2. Multi-island collaborative co-evolution reaching convergence.
3. Blackboard elite sharing and negative genetic memory (avoid_loci) distribution.
4. Governor gatekeeping rejecting sub-par candidate proposals.
5. Burden-gated automatic eviction whenever the swarm turns into a net compute burden.
"""
from __future__ import annotations

import math
import random
import time
import pytest

from evolab import (
    BurdenDecision,
    BurdenGatedIslandSwarmEngine,
    BurdenMonitor,
    EngineConfig,
    EvolutionEngine,
    Individual,
    SwarmBlackboard,
    SwarmStatus,
)
from evolab.genome import FloatGenome


def test_swarm_contract_tagging():
    """Verify that the engine and its status explicitly bear the probationary burden-gated tag."""
    engine = BurdenGatedIslandSwarmEngine(
        fitness_fn=lambda ind: sum(ind.genome.values),
        island_count=3,
    )
    assert engine.TAG == "PROBATIONARY_BURDEN_GATED"
    assert engine.CONTRACT_STATUS == SwarmStatus.PROBATIONARY
    assert engine.status == SwarmStatus.PROBATIONARY


def test_swarm_blackboard_elite_sharing_and_governor():
    """Verify that blackboard shares top elites and filters out sub-par candidates."""
    board = SwarmBlackboard(max_elites=3)

    # 1. First elite accepted
    ind1 = Individual(genome=FloatGenome(values=[1.0, 2.0]), species="spec_0")
    ind1.fitness = 80.0
    accepted = board.submit_candidate(ind1, baseline_fitness=50.0, island_id=0)
    assert accepted is True
    assert len(board.global_elites) == 1
    assert board.global_elites[0].fitness == 80.0

    # 2. Sub-par candidate (<= baseline) rejected
    ind_bad = Individual(genome=FloatGenome(values=[0.0, 0.0]), species="spec_0")
    ind_bad.fitness = 40.0
    accepted = board.submit_candidate(ind_bad, baseline_fitness=50.0, island_id=1)
    assert accepted is False
    assert board.rejected_proposals == 1

    # 3. Superior elite replaces or tops the pool
    ind2 = Individual(genome=FloatGenome(values=[3.0, 4.0]), species="spec_0")
    ind2.fitness = 95.0
    accepted = board.submit_candidate(ind2, baseline_fitness=80.0, island_id=2)
    assert accepted is True
    assert len(board.global_elites) == 2
    assert board.global_elites[0].fitness == 95.0


def test_swarm_blackboard_negative_genetic_memory_sharing():
    """Verify dead door loci published by one island are shared across all islands."""
    board = SwarmBlackboard()
    
    dead_door_1 = ("main.py", 10, 4, "int_wrap")
    dead_door_2 = ("utils.py", 42, 8, "bool_flip")

    added = board.publish_avoid_loci({dead_door_1})
    assert added == 1
    assert dead_door_1 in board.get_avoid_loci()

    # Second publication with overlap
    added = board.publish_avoid_loci({dead_door_1, dead_door_2})
    assert added == 1
    assert len(board.get_avoid_loci()) == 2
    assert dead_door_2 in board.get_avoid_loci()


def test_burden_monitor_eviction_on_excessive_sync_overhead():
    """Verify burden monitor evicts if synchronization / IPC overhead exceeds budget."""
    monitor = BurdenMonitor(island_count=4, max_sync_overhead_pct=30.0)

    # Simulated run: total 1.0s, but sync took 0.45s (45% overhead)
    decision = monitor.evaluate_burden(
        current_best_fitness=85.0,
        baseline_best_fitness=80.0,
        swarm_elapsed_sec=1.0,
        baseline_expected_sec=2.0,
        sync_time_sec=0.45,
        total_evaluations=100,
        baseline_evaluations=100,
    )
    assert decision.should_evict is True
    assert "Excessive synchronization overhead" in decision.reason
    assert decision.sync_overhead_pct == 45.0


def test_burden_monitor_eviction_on_zero_marginal_utility_and_excess_evals():
    """Verify burden monitor evicts when swarm consumes >1.5x evals with zero improvement over baseline."""
    monitor = BurdenMonitor(island_count=4, stagnation_patience_rounds=2)

    # Round 1 stagnant
    monitor.evaluate_burden(
        current_best_fitness=70.0,
        baseline_best_fitness=70.0,
        swarm_elapsed_sec=0.5,
        baseline_expected_sec=0.5,
        sync_time_sec=0.05,
        total_evaluations=80,
        baseline_evaluations=50,
    )

    # Round 2 stagnant with excess compute
    decision = monitor.evaluate_burden(
        current_best_fitness=70.0,
        baseline_best_fitness=70.0,
        swarm_elapsed_sec=1.0,
        baseline_expected_sec=1.0,
        sync_time_sec=0.05,
        total_evaluations=160,
        baseline_evaluations=100,
    )
    assert decision.should_evict is True
    assert "Zero marginal utility" in decision.reason


def test_burden_monitor_eviction_on_sublinear_scaling_collapse():
    """Verify burden monitor evicts when empirical alpha drops below threshold."""
    # With 4 islands, speedup = 1.0 means alpha = ln(1)/ln(4) = 0.0 < 0.40
    monitor = BurdenMonitor(island_count=4, alpha_min=0.40)

    decision = monitor.evaluate_burden(
        current_best_fitness=85.0,
        baseline_best_fitness=80.0,
        swarm_elapsed_sec=1.5,
        baseline_expected_sec=1.2, # Speedup = 0.8 (< 1.0)
        sync_time_sec=0.02,
        total_evaluations=100,
        baseline_evaluations=100,
    )
    assert decision.should_evict is True
    assert "Sublinear scaling collapse" in decision.reason
    assert decision.empirical_alpha < 0.40


def test_island_swarm_convergence_on_optimization():
    """Verify end-to-end swarm convergence on an optimization landscape."""
    target = [0.75] * 8

    def fitness_fn(ind):
        # 100 - sum squared distance
        vals = ind.genome.values if hasattr(ind.genome, "values") else list(ind.genome)
        dist = sum((v - t) ** 2 for v, t in zip(vals, target))
        return max(0.0, 100.0 / (1.0 + dist))

    cfg = EngineConfig(
        population_size=12,
        genome_size=8,
        elite_count=1,
    )

    swarm = BurdenGatedIslandSwarmEngine(
        fitness_fn=fitness_fn,
        config=cfg,
        island_count=3,
        generations_per_round=3,
        max_rounds=6,
        early_stop_fitness=98.0,
        base_seed=42,
    )

    result = swarm.run()
    assert result["contract_tag"] == "PROBATIONARY_BURDEN_GATED"
    assert result["total_rounds"] >= 1
    assert result["best_fitness"] > 5.0
    # Must either converge or remain active (not prematurely evicted if scaling is healthy)
    assert result["swarm_status"] in (SwarmStatus.CONVERGED.value, SwarmStatus.PROBATIONARY.value)


def test_island_swarm_eviction_in_action():
    """Verify that if an artificial burden is injected, the engine halts rounds and flags eviction."""
    def dummy_fitness(ind):
        return 50.0 # flat stagnation

    cfg = EngineConfig(population_size=8, genome_size=4)
    swarm = BurdenGatedIslandSwarmEngine(
        fitness_fn=dummy_fitness,
        config=cfg,
        island_count=4,
        generations_per_round=2,
        max_rounds=5,
        alpha_threshold=0.50,
        max_sync_overhead_pct=0.0001, # Trigger immediate eviction via sync burden
    )

    result = swarm.run()
    assert result["evicted"] is True
    assert result["swarm_status"] == SwarmStatus.EVICTED.value
    assert result["eviction_reason"] is not None
    assert "Excessive synchronization overhead" in result["eviction_reason"]
    # Verified: Swarm evicted itself early and did not waste all 5 rounds!
    assert result["total_rounds"] < 5


def test_island_swarm_on_code_repair():
    """Verify that multi-island swarm operates smoothly on real code repair scenarios."""
    from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population
    scenario = SCENARIO_REGISTRY["requests_http_helper"]()
    evaluator = scenario.create_evaluator()

    cfg = EngineConfig(population_size=12, elite_count=1)
    swarm = BurdenGatedIslandSwarmEngine(
        fitness_fn=evaluator,
        config=cfg,
        island_count=3,
        generations_per_round=3,
        max_rounds=4,
        early_stop_fitness=100.0,
        base_seed=7,
    )
    pops = [
        make_code_population(scenario, 12, random.Random(7 + i * 10))
        for i in range(3)
    ]
    result = swarm.run(initial_populations=pops)
    assert result["contract_tag"] == "PROBATIONARY_BURDEN_GATED"
    assert result["total_rounds"] >= 1
    assert result["best_fitness"] >= 46.0
