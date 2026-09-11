"""
tests/test_cgp_miller_drift.py
Validation of Julian Miller's (1+lambda)-ES and Neutral Genetic Drift dynamics
in Cartesian Genetic Programming (CGP) — Milestone M2.
"""
import random
import pytest

from evolab.cgp_logic import (
    CGPGenome,
    CGPNode,
    GateType,
    HALF_ADDER_TRUTH_TABLE,
    ALUEvaluator,
    create_random_cgp_genome,
    mutate_cgp_genome,
    measure_neutrality,
    classify_mutation,
    evolve_cgp,
)


def test_measure_neutrality():
    """Verifies that measure_neutrality accurately computes the fraction of silent gates."""
    # Circuit with 2 inputs (0, 1) and 3 nodes (2, 3, 4)
    # Node 2: AND(0, 1) -> active (connected to output 0)
    # Node 3: OR(0, 1)  -> active (connected to output 1)
    # Node 4: XOR(0, 1) -> INACTIVE / NEUTRAL (not connected)
    nodes = [
        CGPNode(GateType.AND, 0, 1),
        CGPNode(GateType.OR, 0, 1),
        CGPNode(GateType.XOR, 0, 1),
    ]
    genome = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])
    
    assert genome.get_active_nodes() == {2, 3}
    # 1 inactive node out of 3 total = 1/3 ~ 0.3333
    neutrality = measure_neutrality(genome)
    assert pytest.approx(neutrality, 0.001) == 0.3333

    # Fully active genome
    genome_active = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes[:2], output_connections=[2, 3])
    assert measure_neutrality(genome_active) == 0.0

    # Empty genome edge case
    empty_genome = CGPGenome(num_inputs=2, num_outputs=1, nodes=[], output_connections=[0])
    assert measure_neutrality(empty_genome) == 0.0


def test_classify_mutation():
    """Verifies distinguishing phenotypic mutations from silent neutral mutations."""
    # 2 inputs, 3 nodes: Node 2 is active, Node 3 and 4 are inactive
    nodes = [
        CGPNode(GateType.AND, 0, 1),
        CGPNode(GateType.OR, 0, 1),
        CGPNode(GateType.XOR, 0, 1),
    ]
    parent = CGPGenome(num_inputs=2, num_outputs=1, nodes=nodes, output_connections=[2])
    assert parent.get_active_nodes() == {2}

    # Case 1: Mutate inactive node 3 (node index 1)
    child_neutral = parent.clone()
    child_neutral.nodes[1].gate_type = GateType.NAND
    assert classify_mutation(parent, child_neutral) == "neutral"

    # Case 2: Mutate active node 2 (node index 0)
    child_phenotypic = parent.clone()
    child_phenotypic.nodes[0].gate_type = GateType.NOR
    assert classify_mutation(parent, child_phenotypic) == "phenotypic"

    # Case 3: Mutate output connection
    child_conn = parent.clone()
    child_conn.output_connections[0] = 3
    assert classify_mutation(parent, child_conn) == "phenotypic"


def test_neutral_drift_plateau_traversal():
    """Demonstrates Julian Miller's theorem: neutral drift permits exploration of flat fitness plateaus."""
    rng = random.Random(123)
    parent = create_random_cgp_genome(num_inputs=2, num_outputs=2, num_nodes=10, rng=rng)

    # Constant flat evaluator: every candidate achieves exactly score 50.0
    def flat_evaluator(g):
        return 50.0

    # 1. Run WITH neutral drift: parent must drift and evolve across generations
    evolved_drift, history_drift, evals_drift = evolve_cgp(
        num_inputs=2,
        num_outputs=2,
        evaluator=flat_evaluator,
        lambda_offspring=4,
        max_evaluations=25,
        target_fitness=100.0,
        seed_genome=parent,
        allow_neutral_drift=True,
        rng=random.Random(123),
    )

    # Check that neutral drift replacements occurred
    drift_events = [h for h in history_drift if h.get("action") == "neutral_drift"]
    assert len(drift_events) > 0, "Neutral drift should accept equal-fitness offspring"
    # The parent should have changed in structure
    assert (
        [n.gate_type for n in evolved_drift.nodes] != [n.gate_type for n in parent.nodes]
        or evolved_drift.output_connections != parent.output_connections
        or [n.input_a for n in evolved_drift.nodes] != [n.input_a for n in parent.nodes]
    )

    # 2. Run WITHOUT neutral drift: parent must stay completely frozen on the plateau
    evolved_nodrift, history_nodrift, evals_nodrift = evolve_cgp(
        num_inputs=2,
        num_outputs=2,
        evaluator=flat_evaluator,
        lambda_offspring=4,
        max_evaluations=25,
        target_fitness=100.0,
        seed_genome=parent,
        allow_neutral_drift=False,
        rng=random.Random(123),
    )

    # Parent remains identical to the original seed
    assert [n.gate_type for n in evolved_nodrift.nodes] == [n.gate_type for n in parent.nodes]
    assert [n.input_a for n in evolved_nodrift.nodes] == [n.input_a for n in parent.nodes]
    assert [n.input_b for n in evolved_nodrift.nodes] == [n.input_b for n in parent.nodes]
    assert evolved_nodrift.output_connections == parent.output_connections


def test_cgp_evolve_half_adder():
    """Verifies that canonical (1+4)-ES synthesizes a 100% functionally correct Half Adder."""
    evaluator = ALUEvaluator(HALF_ADDER_TRUTH_TABLE, area_weight=5.0, delay_weight=5.0)

    # We use a deterministic seed where Half Adder synthesizes rapidly
    best_circuit = None
    best_metrics = None
    
    # Try a couple seeds to ensure robust synthesis within a compact evaluation budget
    for seed in [42, 101, 777]:
        circuit, history, evals = evolve_cgp(
            num_inputs=2,
            num_outputs=2,
            evaluator=evaluator,
            lambda_offspring=4,
            max_evaluations=400,
            target_fitness=70.0,  # Functional correctness in ALUEvaluator is >= 70.0
            num_nodes=8,
            mutation_rate=0.2,
            allow_neutral_drift=True,
            rng=random.Random(seed),
        )
        metrics = circuit.evaluate_truth_table(HALF_ADDER_TRUTH_TABLE)
        if metrics.is_fully_functional:
            best_circuit = circuit
            best_metrics = metrics
            break

    assert best_circuit is not None, "CGP (1+4)-ES should synthesize a functional Half Adder"
    assert best_metrics.is_fully_functional is True
    assert best_metrics.truth_table_accuracy == 1.0
    assert best_metrics.active_gate_count > 0

    # Verify simulation against truth table directly
    for inps, expected in HALF_ADDER_TRUTH_TABLE:
        actual = best_circuit.simulate(inps)
        assert actual == list(expected)


def test_cgp_seed_continuation():
    """Verifies that seeding an existing valid genome into evolve_cgp preserves correctness."""
    # Seed with canonical Half Adder: Node 2: XOR(0, 1) -> Sum, Node 3: AND(0, 1) -> Carry
    nodes = [
        CGPNode(GateType.XOR, 0, 1),
        CGPNode(GateType.AND, 0, 1),
    ]
    canonical_ha = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])
    evaluator = ALUEvaluator(HALF_ADDER_TRUTH_TABLE, area_weight=10.0, delay_weight=10.0)

    evolved, history, evals = evolve_cgp(
        num_inputs=2,
        num_outputs=2,
        evaluator=evaluator,
        lambda_offspring=2,
        max_evaluations=10,
        seed_genome=canonical_ha,
        rng=random.Random(42),
    )

    # Initial and final fitness should be >= 70.0 (fully functional)
    assert history[0]["fitness"] >= 70.0
    metrics = evolved.evaluate_truth_table(HALF_ADDER_TRUTH_TABLE)
    assert metrics.is_fully_functional is True


def test_cgp_telemetry_and_budget_bounds():
    """Verifies that evaluation budget and history schema are strictly respected."""
    evaluator = ALUEvaluator(HALF_ADDER_TRUTH_TABLE)
    max_budget = 35

    circuit, history, evals = evolve_cgp(
        num_inputs=2,
        num_outputs=2,
        evaluator=evaluator,
        lambda_offspring=4,
        max_evaluations=max_budget,
        target_fitness=100.0,
        rng=random.Random(99),
    )

    assert evals <= max_budget
    assert len(history) >= 1
    assert "generation" in history[0]
    assert "evaluations" in history[0]
    assert "neutrality" in history[0]
    assert "active_nodes" in history[0]
