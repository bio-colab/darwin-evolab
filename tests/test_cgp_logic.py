"""
Unit tests for Cartesian Genetic Programming (CGP) & Digital Logic circuit synthesis.
"""
from __future__ import annotations

import random
import pytest

from evolab.cgp_logic import (
    GateType,
    GATE_TRANSISTORS,
    GATE_DELAYS,
    CGPNode,
    CircuitMetrics,
    CGPGenome,
    eval_gate,
    create_random_cgp_genome,
    mutate_cgp_genome,
    ALUEvaluator,
    HALF_ADDER_TRUTH_TABLE,
    FULL_ADDER_TRUTH_TABLE,
    COMPARATOR_TRUTH_TABLE,
)
from evolab.engine import EvolutionEngine, Individual
from evolab.config import EngineConfig


def test_gate_evaluations():
    """Validates Boolean truth values for all primitive gates."""
    assert eval_gate(GateType.AND, 1, 1) == 1
    assert eval_gate(GateType.AND, 1, 0) == 0
    assert eval_gate(GateType.OR, 0, 1) == 1
    assert eval_gate(GateType.OR, 0, 0) == 0
    assert eval_gate(GateType.XOR, 1, 0) == 1
    assert eval_gate(GateType.XOR, 1, 1) == 0
    assert eval_gate(GateType.NAND, 1, 1) == 0
    assert eval_gate(GateType.NAND, 0, 1) == 1
    assert eval_gate(GateType.NOR, 0, 0) == 1
    assert eval_gate(GateType.NOR, 0, 1) == 0
    assert eval_gate(GateType.NOT, 1, 0) == 0
    assert eval_gate(GateType.NOT, 0, 0) == 1
    assert eval_gate(GateType.XNOR, 1, 1) == 1
    assert eval_gate(GateType.XNOR, 1, 0) == 0
    assert eval_gate(GateType.WIRE, 1, 0) == 1


def test_active_node_extraction_and_pruning():
    """Verifies that unreferenced/dead-code gates are pruned from active topology."""
    # 2 inputs: 0, 1
    # Node 2: XOR(0, 1) -> ACTIVE (output 0)
    # Node 3: AND(0, 1) -> ACTIVE (output 1)
    # Node 4: NOR(2, 3) -> INACTIVE (dead code)
    nodes = [
        CGPNode(GateType.XOR, 0, 1),
        CGPNode(GateType.AND, 0, 1),
        CGPNode(GateType.NOR, 2, 3),
    ]
    genome = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])
    active = genome.get_active_nodes()
    assert active == {2, 3}
    assert 4 not in active


def test_ideal_half_adder_metrics():
    """Validates metrics of an ideal human-designed Half Adder."""
    nodes = [
        CGPNode(GateType.XOR, 0, 1),  # Node 2: Sum
        CGPNode(GateType.AND, 0, 1),  # Node 3: Carry
    ]
    genome = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])
    metrics = genome.evaluate_truth_table(HALF_ADDER_TRUTH_TABLE)
    assert metrics.is_fully_functional is True
    assert metrics.truth_table_accuracy == 1.0
    assert metrics.active_gate_count == 2
    assert metrics.transistor_count == GATE_TRANSISTORS[GateType.XOR] + GATE_TRANSISTORS[GateType.AND]  # 8 + 6 = 14
    assert metrics.critical_path_delay == 2.0  # max(XOR delay 2.0, AND delay 1.6)


def test_verilog_export():
    """Ensures synthesizable Verilog code generation matches circuit topology."""
    nodes = [
        CGPNode(GateType.XOR, 0, 1),
        CGPNode(GateType.AND, 0, 1),
    ]
    genome = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])
    verilog = genome.to_verilog("cgp_half_adder")
    assert "module cgp_half_adder" in verilog
    assert "input  wire [1:0] in," in verilog
    assert "output wire [1:0] out" in verilog
    assert "xor g_2 (w_2, in[0], in[1]);" in verilog
    assert "and g_3 (w_3, in[0], in[1]);" in verilog
    assert "assign out[0] = w_2;" in verilog
    assert "assign out[1] = w_3;" in verilog
    assert "endmodule" in verilog


def test_cgp_genome_abstraction_contract():
    """Validates EvolabGenome protocol (clone, distance, serialize, describe)."""
    genome = create_random_cgp_genome(num_inputs=2, num_outputs=2, num_nodes=5, rng=random.Random(42))
    cloned = genome.clone()
    assert len(cloned.nodes) == len(genome.nodes)
    assert cloned.output_connections == genome.output_connections

    mutated = mutate_cgp_genome(genome, mutation_rate=0.5, rng=random.Random(42))
    dist = genome.distance(mutated)
    assert dist > 0.0

    desc = genome.describe()
    assert "node_count" in desc
    assert "total_nodes" in desc

    ser = genome.serialize()
    assert ser["num_inputs"] == 2
    assert ser["num_outputs"] == 2


def test_alu_evaluator_scoring():
    """Validates that fully functional circuits get rewarded over broken ones."""
    evaluator = ALUEvaluator(HALF_ADDER_TRUTH_TABLE)

    # Ideal circuit
    ideal = CGPGenome(
        num_inputs=2, num_outputs=2,
        nodes=[CGPNode(GateType.XOR, 0, 1), CGPNode(GateType.AND, 0, 1)],
        output_connections=[2, 3]
    )
    score_ideal = evaluator.evaluate(ideal)
    assert score_ideal > 70.0  # 70 + area_bonus + delay_bonus

    # Broken circuit (all zeros)
    broken = CGPGenome(
        num_inputs=2, num_outputs=2,
        nodes=[CGPNode(GateType.AND, 0, 0), CGPNode(GateType.AND, 1, 1)],
        output_connections=[0, 0]
    )
    score_broken = evaluator.evaluate(broken)
    assert score_broken < 70.0
    assert score_ideal > score_broken


def test_evolutionary_discovery_of_half_adder():
    """Verifies that an evolutionary run synthesizes a 100% functional Half Adder in <= 100 generations."""
    evaluator = ALUEvaluator(HALF_ADDER_TRUTH_TABLE)
    rng = random.Random(42)
    allowed = [GateType.NAND, GateType.NOR, GateType.AND, GateType.OR, GateType.XOR, GateType.NOT]

    pop_size = 20
    pop = [
        Individual(genome=create_random_cgp_genome(2, 2, num_nodes=10, allowed_gates=allowed, rng=rng), species="cgp")
        for _ in range(pop_size)
    ]

    discovered_optimal = False
    for gen in range(1, 101):
        for ind in pop:
            ind.fitness = evaluator(ind)

        pop.sort(key=lambda x: x.fitness, reverse=True)
        best = pop[0]
        metrics = best.genome.evaluate_truth_table(HALF_ADDER_TRUTH_TABLE)

        if metrics.is_fully_functional and metrics.active_gate_count <= 2:
            discovered_optimal = True
            break

        elites = pop[:4]
        next_gen = [Individual(genome=e.genome.clone(), species="cgp") for e in elites]
        while len(next_gen) < pop_size:
            parent = rng.choice(elites)
            child_g = mutate_cgp_genome(parent.genome, mutation_rate=0.15, rng=rng)
            next_gen.append(Individual(genome=child_g, species="cgp"))
        pop = next_gen

    assert discovered_optimal is True
    assert best.genome.evaluate_truth_table(HALF_ADDER_TRUTH_TABLE).is_fully_functional is True


def test_full_adder_and_comparator_truth_tables():
    """Validates truth table dimensions and formats for ALU benchmarks."""
    assert len(FULL_ADDER_TRUTH_TABLE) == 8
    assert len(COMPARATOR_TRUTH_TABLE) == 16
    for inps, outs in FULL_ADDER_TRUTH_TABLE:
        assert len(inps) == 3
        assert len(outs) == 2
    for inps, outs in COMPARATOR_TRUTH_TABLE:
        assert len(inps) == 4
        assert len(outs) == 3


def test_hierarchical_8bit_adder_canonical_simulation():
    """Verifies that an 8-bit ripple-carry adder achieves 100% arithmetic accuracy on 1000 vectors."""
    from evolab.cgp_logic import HierarchicalAdder8Bit
    adder8 = HierarchicalAdder8Bit.create_canonical()

    passed, tested = adder8.verify_exhaustive(max_cases=1000)
    assert passed is True
    assert tested >= 1000

    metrics = adder8.compute_metrics()
    assert metrics.bit_width == 8
    assert metrics.total_active_gates == 40
    assert metrics.total_transistors > 150
    assert metrics.is_100_percent_functional is True


def test_hierarchical_8bit_adder_verilog_generation():
    """Ensures hierarchical Verilog generation creates top-level module and 8 chained instances."""
    from evolab.cgp_logic import HierarchicalAdder8Bit
    adder8 = HierarchicalAdder8Bit.create_canonical()
    verilog = adder8.to_verilog("hierarchical_8bit_adder")

    assert "module hierarchical_8bit_adder" in verilog
    assert "input  wire [7:0] a," in verilog
    assert "input  wire [7:0] b," in verilog
    assert "output wire [7:0] sum," in verilog
    assert "output wire       cout" in verilog
    assert "cgp_fa_cell fa_0" in verilog
    assert "cgp_fa_cell fa_7" in verilog
    assert "assign cout = c[8];" in verilog


def test_hierarchical_8bit_adder_invalid_cell_rejection():
    """Ensures invalid or broken cells are rejected upon composition."""
    from evolab.cgp_logic import HierarchicalAdder8Bit, CGPGenome, CGPNode, GateType
    bad_cell = CGPGenome(num_inputs=2, num_outputs=2, nodes=[CGPNode(GateType.AND, 0, 1)], output_connections=[2, 2])
    with pytest.raises(ValueError, match="full_adder_cell must have exactly 3 inputs"):
        HierarchicalAdder8Bit(bad_cell)


def test_switching_activity_metrics():
    """Validates dynamic wire transition and toggling counting."""
    from evolab.cgp_logic import CGPGenome, CGPNode, GateType
    nodes = [
        CGPNode(GateType.XOR, 0, 1),
        CGPNode(GateType.AND, 0, 1),
    ]
    genome = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])
    stream = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
    metrics = genome.evaluate_switching_activity(stream)

    assert metrics.total_toggles > 0
    assert metrics.transitions_measured == 4
    assert metrics.active_wire_count == 4
    assert metrics.dynamic_power_factor > 0.0


def test_low_power_evaluator_rewards_lower_toggles():
    """Validates that circuits with fewer transitions receive higher fitness bonuses."""
    from evolab.cgp_logic import CGPGenome, CGPNode, GateType, LowPowerALUEvaluator, HALF_ADDER_TRUTH_TABLE
    evaluator = LowPowerALUEvaluator(HALF_ADDER_TRUTH_TABLE)

    # 1. Optimal 2-gate Half Adder
    optimal = CGPGenome(num_inputs=2, num_outputs=2, nodes=[
        CGPNode(GateType.XOR, 0, 1),
        CGPNode(GateType.AND, 0, 1),
    ], output_connections=[2, 3])

    # 2. Redundant 4-gate Half Adder (adds buffer/NOT-NOT dead switching)
    redundant = CGPGenome(num_inputs=2, num_outputs=2, nodes=[
        CGPNode(GateType.NOT, 0, 0),
        CGPNode(GateType.NOT, 2, 2), # wire 3 is in[0]
        CGPNode(GateType.XOR, 3, 1), # wire 4 is Sum
        CGPNode(GateType.AND, 3, 1), # wire 5 is Carry
    ], output_connections=[4, 5])

    score_opt = evaluator.evaluate(optimal)
    score_red = evaluator.evaluate(redundant)

    assert score_opt > score_red
    sw_opt = optimal.evaluate_switching_activity(evaluator.switching_stream)
    sw_red = redundant.evaluate_switching_activity(evaluator.switching_stream)
    assert sw_opt.total_toggles < sw_red.total_toggles


def test_eda_packager_script_and_constraint_generation():
    """Validates Yosys script generation and physical constraints formatting."""
    from evolab.cgp_logic import EDAPackager
    packager = EDAPackager(target_fpga="ice40")

    ys_script = packager.generate_yosys_script("circuit.v", "top_mod", "out.json")
    assert "read_verilog circuit.v" in ys_script
    assert "synth_ice40 -top top_mod -json out.json" in ys_script
    assert "stat" in ys_script

    pcf = packager.generate_pcf_constraints("top_mod", num_inputs=3, num_outputs=2)
    assert "set_io in[0] 10" in pcf
    assert "set_io in[2] 12" in pcf
    assert "set_io out[0] 30" in pcf
    assert "set_io out[1] 31" in pcf


def test_eda_packager_bundle_creation_and_graceful_handling(tmp_path):
    """Ensures complete EDA synthesis bundle is emitted with proper cross-platform runners."""
    from evolab.cgp_logic import EDAPackager
    packager = EDAPackager(target_fpga="ice40")

    verilog_code = "// Simple test\nmodule test_mod (input in, output out); assign out = in; endmodule"
    report = packager.package_bundle(
        verilog_code=verilog_code,
        top_module="test_mod",
        num_inputs=1,
        num_outputs=1,
        output_dir=tmp_path,
        run_synthesis_if_available=False,
    )

    assert (tmp_path / "test_mod.v").exists()
    assert (tmp_path / "synth_ice40.ys").exists()
    assert (tmp_path / "test_mod.pcf").exists()
    assert (tmp_path / "run_synth.bat").exists()
    assert (tmp_path / "run_synth.sh").exists()
    assert report.top_module == "test_mod"
    assert report.target_fpga == "ice40"
    assert report.synthesis_executed is False



