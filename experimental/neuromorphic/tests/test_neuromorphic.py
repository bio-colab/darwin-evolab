"""
test_neuromorphic.py — Unit and integration tests for Drosophila neuromorphic CGP track.

Verifies:
1. NeuromorphicSpec configuration and bounds checking.
2. Drosophila biological motifs ground truth (EMD and Olfactory).
3. NeuromorphicCircuitGenome DAG generation, mutation, crossover, active nodes, and distance.
4. Stateful multi-step temporal delay simulation (Z^-1 register behavior).
5. NeuromorphicEvaluator scoring accuracy and parsimony regularization.
6. Synthesizable Verilog-2001 and ASCII export.
7. NeuromorphicAdapter registry and end-to-end evolution engine integration.
"""
from __future__ import annotations

import random
import pytest

from evolab.adapters import get_domain_adapter, list_domain_adapters
from evolab.engine import EvolutionEngine
from evolab.genome import Individual

from experimental.neuromorphic import (
    NeuromorphicAdapter,
    NeuromorphicCircuitGenome,
    NeuromorphicEvaluator,
    NeuromorphicExporter,
    NeuromorphicNode,
    NeuromorphicOp,
    NeuromorphicSpec,
    generate_emd_dataset,
    generate_olfactory_dataset,
)


def test_spec_validation() -> None:
    """Verify NeuromorphicSpec default settings and parameter validations."""
    spec = NeuromorphicSpec()
    assert spec.motif_name == "hassenstein_reichardt_emd"
    assert spec.num_inputs == 2
    assert spec.num_outputs == 2
    assert spec.sequence_length == 24
    assert spec.max_nodes == 16
    assert spec.clocked is True

    # Valid olfactory spec
    olf_spec = NeuromorphicSpec(motif_name="olfactory_lateral_inhibition")
    assert olf_spec.motif_name == "olfactory_lateral_inhibition"

    # Unsupported motif
    with pytest.raises(ValueError, match="Unsupported motif_name"):
        NeuromorphicSpec(motif_name="invalid_motif")

    # Invalid input/output counts
    with pytest.raises(ValueError, match="num_inputs must be >= 1"):
        NeuromorphicSpec(num_inputs=0)

    with pytest.raises(ValueError, match="num_outputs must be >= 1"):
        NeuromorphicSpec(num_outputs=0)

    with pytest.raises(ValueError, match="max_nodes must be >= 1"):
        NeuromorphicSpec(max_nodes=0)

    with pytest.raises(ValueError, match="sequence_length must be >= 4"):
        NeuromorphicSpec(sequence_length=2)


def test_emd_dataset_generation() -> None:
    """Verify Hassenstein-Reichardt visual motion dataset contract."""
    ds = generate_emd_dataset(length=30, seed=123)
    assert "inputs" in ds
    assert "expected_outputs" in ds
    assert "metadata" in ds
    assert ds.length == 30
    assert len(ds) == 30
    assert len(ds.inputs) == 30
    assert len(ds.expected_outputs) == 30

    meta = ds.metadata
    assert meta["motif"] == "hassenstein_reichardt_emd"
    assert meta["total_timesteps"] == 30
    assert meta["provenance"] == "Drosophila_T4_T5_EMD"

    # Verify input format: each element is [R1, R2]
    for inp in ds.inputs:
        assert len(inp) == 2
        assert all(val in (0, 1) for val in inp)

    # Verify output format: each element is [out_right, out_left]
    for out in ds.expected_outputs:
        assert len(out) == 2
        assert all(val in (0, 1) for val in out)


def test_olfactory_dataset_generation() -> None:
    """Verify Olfactory Glomerular lateral inhibition dataset contract."""
    ds = generate_olfactory_dataset(length=25, seed=456)
    assert ds.length == 25
    assert len(ds) == 25
    assert len(ds.inputs) == 25
    assert len(ds.expected_outputs) == 25

    meta = ds.metadata
    assert meta["motif"] == "olfactory_lateral_inhibition"
    assert meta["total_timesteps"] == 25
    assert meta["provenance"] == "Drosophila_AntennalLobe_LN"

    # Verify lateral inhibition logic:
    # Dominant receptor wins and suppresses secondary receptors
    for inp, exp in zip(ds.inputs, ds.expected_outputs):
        assert len(inp) == 3
        assert len(exp) == 3
        assert exp[0] == inp[0]
        assert exp[1] == (inp[1] if not inp[0] else 0)
        assert exp[2] == (inp[2] if (not inp[0] and not inp[1]) else 0)


def test_genome_creation_and_serialization() -> None:
    """Test NeuromorphicCircuitGenome instantiation, serialization and mutation."""
    rng = random.Random(42)
    genome = NeuromorphicCircuitGenome.random(
        num_inputs=2,
        num_outputs=2,
        num_nodes=8,
        rng=rng,
    )

    assert genome.num_inputs == 2
    assert genome.num_outputs == 2
    assert len(genome.nodes) == 8
    assert len(genome.output_indices) == 2

    # Roundtrip serialization
    d = genome.to_dict()
    restored = NeuromorphicCircuitGenome.from_dict(d)
    assert restored.num_inputs == genome.num_inputs
    assert restored.num_outputs == genome.num_outputs
    assert restored.output_indices == genome.output_indices
    assert len(restored.nodes) == len(genome.nodes)
    for orig_node, rest_node in zip(genome.nodes, restored.nodes):
        assert orig_node.index == rest_node.index
        assert orig_node.op == rest_node.op
        assert orig_node.input_a == rest_node.input_a
        assert orig_node.input_b == rest_node.input_b

    # Distance metric
    assert genome.distance(restored) == 0.0
    other = NeuromorphicCircuitGenome.random(num_inputs=2, num_outputs=2, num_nodes=8, rng=rng)
    assert genome.distance(other) > 0.0

    # Mutation produces valid DAG
    mutated = genome.mutate(rng=rng, rate=0.4)
    assert len(mutated.nodes) == 8
    for n in mutated.nodes:
        assert n.input_a < n.index
        assert n.input_b < n.index

    # Crossover produces valid DAG
    child = genome.crossover(other, rng=rng)
    assert len(child.nodes) == 8
    for n in child.nodes:
        assert n.input_a < n.index
        assert n.input_b < n.index


def test_stateful_delay_simulation() -> None:
    """Test stateful simulation of DELAY (Z^-1) and INHIBIT operations."""
    # Construct an explicit circuit:
    # Inputs: IN[0], IN[1] (indices 0, 1)
    # Node 2: DELAY(IN[0]) -> stores IN[0] from previous timestep
    # Node 3: AND(Node 2, IN[1]) -> (IN[0] at t-1) AND (IN[1] at t)
    # Outputs: [Node 3, Node 2]
    node2 = NeuromorphicNode(index=2, op=NeuromorphicOp.DELAY, input_a=0, input_b=0)
    node3 = NeuromorphicNode(index=3, op=NeuromorphicOp.AND, input_a=2, input_b=1)

    circuit = NeuromorphicCircuitGenome(
        num_inputs=2,
        num_outputs=2,
        nodes=[node2, node3],
        output_indices=[3, 2],
    )

    # Sequence of inputs:
    # t=0: [1, 0] -> delay register stores 1 at end of cycle, output node2=0 (initially 0), node3=0
    # t=1: [0, 1] -> delay output is 1 (stored from t=0), IN[1] is 1 -> node3 = 1 & 1 = 1
    # t=2: [0, 0] -> delay output is 0 (stored from t=1), node3 = 0
    seq_inputs = [
        [1, 0],
        [0, 1],
        [0, 0],
    ]

    outputs = circuit.simulate_sequence(seq_inputs)
    assert len(outputs) == 3

    # At t=0: node 2 was initialized to 0, node 3 = 0 & 0 = 0
    assert outputs[0] == [0, 0]

    # At t=1: node 2 reads stored value 1, node 3 = 1 & 1 = 1
    assert outputs[1] == [1, 1]

    # At t=2: node 2 reads stored value 0 (from t=1), node 3 = 0 & 0 = 0
    assert outputs[2] == [0, 0]


def test_evaluator_scoring() -> None:
    """Verify NeuromorphicEvaluator deterministic scoring and parsimony."""
    spec = NeuromorphicSpec(sequence_length=16)
    dataset = generate_emd_dataset(length=16, seed=42)
    evaluator = NeuromorphicEvaluator(spec, dataset)

    rng = random.Random(10)
    ind = Individual(
        genome=NeuromorphicCircuitGenome.random(2, 2, 8, rng),
        species="neuromorphic",
    )

    score1 = evaluator.evaluate(ind)
    score2 = evaluator.evaluate(ind)
    assert score1.score == score2.score
    assert 0.0 <= score1.score <= 100.0

    # Test __call__ protocol on evaluator
    call_score = evaluator(ind)
    assert call_score == score1.score


def test_exporter_verilog_and_ascii() -> None:
    """Verify ASCII schematic and synthesizable Verilog-2001 export."""
    spec = NeuromorphicSpec(motif_name="hassenstein_reichardt_emd")
    node2 = NeuromorphicNode(index=2, op=NeuromorphicOp.DELAY, input_a=0, input_b=0)
    node3 = NeuromorphicNode(index=3, op=NeuromorphicOp.AND, input_a=2, input_b=1)
    genome = NeuromorphicCircuitGenome(
        num_inputs=2,
        num_outputs=2,
        nodes=[node2, node3],
        output_indices=[3, 0],
    )

    # ASCII diagram
    ascii_out = NeuromorphicExporter.export_ascii(genome, spec)
    assert "=== Evolved Drosophila Neuromorphic Circuit: hassenstein_reichardt_emd ===" in ascii_out
    assert "DELAY[Z^-1]" in ascii_out
    assert "OUTPUT[0] <= N3" in ascii_out

    # Verilog RTL
    verilog = NeuromorphicExporter.export_verilog(genome, module_name="test_emd_module")
    assert "`default_nettype none" in verilog
    assert "module test_emd_module (" in verilog
    assert "input  wire        clk," in verilog
    assert "input  wire        rst," in verilog
    assert "input  wire [1:0] in_spikes," in verilog
    assert "output wire [1:0] out_spikes" in verilog
    assert "always @(posedge clk or posedge rst) begin" in verilog
    assert "w_2_reg <= 1'b0;" in verilog
    assert "endmodule" in verilog


def test_adapter_registry_and_evolution() -> None:
    """Verify adapter registration and end-to-end evolution execution."""
    assert "neuromorphic" in list_domain_adapters()
    adapter = get_domain_adapter("neuromorphic")

    spec = adapter.parse_spec({
        "motif_name": "hassenstein_reichardt_emd",
        "num_inputs": 2,
        "num_outputs": 2,
        "sequence_length": 16,
        "max_nodes": 6,
    })

    pop = adapter.build_population(spec, size=6, rng=random.Random(42))
    assert len(pop) == 6

    evaluator = adapter.build_evaluator(spec)
    for ind in pop:
        ind.fitness = evaluator(ind)
        assert ind.fitness is not None

    solution = adapter.export_solution(pop[0], spec)
    assert solution["motif_name"] == "hassenstein_reichardt_emd"
    assert "verilog_rtl" in solution
    assert "ascii_diagram" in solution
    assert solution["physical_claim"] is False

    # Short 2-generation evolution engine run
    engine = EvolutionEngine(
        population_size=6,
        seed=100,
        fitness_fn=evaluator,
    )
    report = engine.run(2, initial_population=pop)
    assert report["total_generations"] == 2
    best = report["best_individual"]
    assert best is not None
    assert best["fitness"] >= 0.0
