"""tests/test_silicon_scaling_pillar3.py — Tests for Silicon CGP Scaling (Pillar 3)."""

from __future__ import annotations

import pytest

from evolab.cgp_logic import (
    ALU4BitMultiOp,
    GateType,
    PriorityEncoder4Bit,
)


def test_priority_encoder_4bit_truth_table_and_netlist():
    """Verify that 4-to-2 Priority Encoder achieves 100% truth table accuracy and valid bit."""
    tt = PriorityEncoder4Bit.generate_truth_table()
    assert len(tt) == 16

    genome = PriorityEncoder4Bit.create_canonical()
    metrics = genome.evaluate_truth_table(tt)

    assert metrics.is_fully_functional is True
    assert metrics.truth_table_accuracy == 1.0
    assert metrics.transistor_count > 0
    assert metrics.critical_path_delay > 0.0

    verilog = genome.to_verilog("priority_encoder_4to2")
    assert "module priority_encoder_4to2" in verilog
    assert "input" in verilog and "output" in verilog


def test_alu_4bit_multi_op_operations_and_flags():
    """Verify arithmetic and logical operations, carry out, and zero flag for ALU4BitMultiOp."""
    # 1. AND (OP=0)
    res, cout, zero = ALU4BitMultiOp.evaluate_operation(0b1100, 0b1010, ALU4BitMultiOp.OP_AND)
    assert res == 0b1000
    assert cout == 0
    assert zero == 0

    # 2. OR (OP=1)
    res, cout, zero = ALU4BitMultiOp.evaluate_operation(0b1100, 0b0011, ALU4BitMultiOp.OP_OR)
    assert res == 0b1111
    assert cout == 0
    assert zero == 0

    # 3. XOR (OP=2) Zero Flag Check
    res, cout, zero = ALU4BitMultiOp.evaluate_operation(0b1010, 0b1010, ALU4BitMultiOp.OP_XOR)
    assert res == 0b0000
    assert cout == 0
    assert zero == 1

    # 4. ADD with Carry Out (OP=3)
    res, cout, zero = ALU4BitMultiOp.evaluate_operation(0b1111, 0b0001, ALU4BitMultiOp.OP_ADD)
    assert res == 0b0000  # 15 + 1 = 16 -> 0 with carry
    assert cout == 1
    assert zero == 1


def test_alu_4bit_multi_op_vector_suite_and_verilog():
    """Verify statistical 1000-vector suite and RTL generation for 4-bit ALU."""
    verification = ALU4BitMultiOp.verify_vector_suite(num_vectors=1000, seed=123)
    assert verification["is_fully_functional"] is True
    assert verification["accuracy"] == 1.0
    assert verification["vectors_verified"] == 1000
    assert verification["estimated_transistors"] > 100

    rtl = ALU4BitMultiOp.to_verilog("alu_4bit_multi_op")
    assert "module alu_4bit_multi_op" in rtl
    assert "output wire       ZERO" in rtl
    assert "output reg        COUT" in rtl
