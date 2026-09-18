"""
Unit tests for Genesis & Foundation Model Evolutionary Kernel Bridge under experimental/.
"""
from __future__ import annotations

import ast
import random
import pytest

from experimental.genesis_bridge import (
    GenesisBridge,
    GenesisRewardVector,
    GenesisEnvironment,
    MockGenesisSimulator,
    serialize_for_foundation_model,
    deserialize_from_tensor,
    FoundationModelPrior,
)
from evolab.cgp_logic import CGPGenome, CGPNode, GateType
from evolab.genome import FloatGenome, Individual
from evolab.engine import EvolutionEngine


def test_genesis_reward_vector_pareto_mapping():
    """Validates multi-channel reward vector projection into Pareto objectives."""
    rewards = {
        "stability": 92.5,
        "energy_eff": 88.0,
        "speed": 75.2,
    }
    vec = GenesisRewardVector(
        primary_fitness=85.2,
        task_success=True,
        channel_rewards=rewards,
        metrics={"test": 1},
        simulation_steps=200,
        latency_ms=1.5,
    )
    assert vec.primary_fitness == 85.2
    assert vec.task_success is True
    pareto_tuple = vec.to_pareto_objectives(["stability", "energy_eff", "missing"])
    assert pareto_tuple == [92.5, 88.0, 0.0]


def test_genesis_mock_simulator_evaluations():
    """Validates multi-domain evaluation in MockGenesisSimulator."""
    sim = MockGenesisSimulator(domain="physics_and_silicon")
    sim.reset(seed=42)
    assert sim.reset_count == 1

    # 1. CGP Silicon Genome
    nodes = [CGPNode(GateType.XOR, 0, 1), CGPNode(GateType.AND, 0, 1)]
    cgp = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])
    res_cgp = sim.evaluate_candidate(cgp)
    assert res_cgp.primary_fitness > 0
    assert "correctness" in res_cgp.channel_rewards
    assert "delay_eff" in res_cgp.channel_rewards
    assert "power_eff" in res_cgp.channel_rewards
    assert res_cgp.metrics["physical_claim"] is False
    assert res_cgp.metrics["is_mock"] is True

    # 2. Continuous FloatGenome
    flt = FloatGenome(values=[1.0, -0.5, 0.25])
    res_flt = sim.evaluate_candidate(flt)
    assert "stability" in res_flt.channel_rewards
    assert "energy" in res_flt.channel_rewards
    assert res_flt.simulation_steps == 100


def test_genesis_bridge_batch_and_engine_attachment():
    """Verifies batched parallel execution and integration with EvolutionEngine."""
    sim = MockGenesisSimulator()
    bridge = GenesisBridge(environment=sim, batch_size=4)

    pop = [Individual(genome=FloatGenome(values=[random.uniform(-1, 1) for _ in range(3)]), species="spec_float") for _ in range(6)]
    results = bridge.evaluate_population(pop)
    assert len(results) == 6
    assert bridge.total_evaluations == 6
    assert len(bridge.get_reward_history()) == 6

    # Attach to EvolutionEngine
    engine = EvolutionEngine(
        population_size=6,
        genome_size=3,
        early_stop_fitness=99.0,
        fitness_fn=bridge.attach_to_engine(engine=None, objective_channel="stability"),
    )
    rep = engine.run(generations=2, initial_population=pop)
    assert rep["total_generations"] >= 1
    assert rep["best_individual"]["fitness"] > 0.0


def test_serialize_and_deserialize_foundation_model():
    """Validates tensor and GNN graph serialization/deserialization for foundation models."""
    # 1. CGP Silicon Graph
    nodes = [CGPNode(GateType.XOR, 0, 1), CGPNode(GateType.AND, 0, 1)]
    cgp = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])
    cgp_tensor = serialize_for_foundation_model(cgp)

    assert cgp_tensor["domain"] == "silicon_cgp"
    assert cgp_tensor["num_inputs"] == 2
    assert cgp_tensor["num_outputs"] == 2
    assert len(cgp_tensor["node_features"]) == 2
    assert len(cgp_tensor["edge_index"][0]) == 4  # 2 inputs * 2 nodes

    # Reconstruct from tensor
    reconstructed_cgp = deserialize_from_tensor(cgp_tensor)
    assert isinstance(reconstructed_cgp, CGPGenome)
    assert reconstructed_cgp.num_inputs == 2
    assert reconstructed_cgp.num_outputs == 2
    assert len(reconstructed_cgp.nodes) == 2
    assert reconstructed_cgp.nodes[0].gate_type == GateType.XOR

    # 2. FloatGenome
    flt = FloatGenome(values=[0.1, 0.2, 0.3])
    flt_tensor = serialize_for_foundation_model(flt)
    assert flt_tensor["domain"] == "continuous_tensor"
    assert flt_tensor["vector"] == [0.1, 0.2, 0.3]
    reconstructed_flt = deserialize_from_tensor(flt_tensor)
    assert isinstance(reconstructed_flt, FloatGenome)
    assert list(reconstructed_flt.genes) == [0.1, 0.2, 0.3]


def test_foundation_model_prior_seeding():
    """Tests population seeding from foundation model priors."""
    prior = FoundationModelPrior(
        suggested_candidates=[
            FloatGenome(values=[0.5, 0.5]),
            FloatGenome(values=[1.0, -1.0]),
        ],
        rationale="Heuristic warm-start from foundational GNN.",
    )
    seeds = prior.sample_seed_population(count=2)
    assert len(seeds) == 2
    assert seeds[0].species == "foundation_prior"
    assert isinstance(seeds[0].genome, FloatGenome)
