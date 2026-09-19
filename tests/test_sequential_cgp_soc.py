"""Tests for Sequential CGP & SoC Digital Logic Synthesis (Pillar 3)."""

from __future__ import annotations

import pytest

from evolab import (
    SequentialCGPGenome,
    SequentialCGPNode,
    SequentialGateType,
    SequentialCircuitMetrics,
    SynchronousCounter8BitEvaluator,
    ShiftRegister8BitEvaluator,
    UARTTransmitterFSMEvaluator,
    create_canonical_sequential_circuit,
)


def test_sequential_cgp_dff_clocking_and_reset():
    """Verify D-Flip-Flop resets to 0 when rst_n=0 and clocks D-input on clock edge when rst_n=1."""
    # 1 input: d_in (0), 1 DFF: q (1), 1 output: q (1)
    # Total inputs = 2 (in_0, q_0)
    # Gate node 2: WIRE(in_0) -> feeds dff_d[0]
    genome = SequentialCGPGenome(
        num_primary_inputs=1,
        num_primary_outputs=1,
        num_dffs=1,
        nodes=[SequentialCGPNode(SequentialGateType.WIRE, 0, 0)],
        output_connections=[1],      # output is q (wire 1)
        dff_input_connections=[2],   # dff_d is wire 2 (node 2)
    )

    # 4 clock cycles:
    # Cycle 0: in=1, rst_n=0 -> q should be 0 (reset)
    # Cycle 1: in=1, rst_n=1 -> q should be 0 (previous state), D=1 clocked
    # Cycle 2: in=0, rst_n=1 -> q should be 1 (clocked from cycle 1), D=0 clocked
    # Cycle 3: in=0, rst_n=1 -> q should be 0 (clocked from cycle 2)
    inputs = [[1], [1], [0], [0]]
    rst_n = [0, 1, 1, 1]

    sim_outs = genome.simulate_waveform(inputs, rst_n_timeline=rst_n)
    assert sim_outs[0] == [0]
    assert sim_outs[1] == [0]
    assert sim_outs[2] == [1]
    assert sim_outs[3] == [0]


def test_synchronous_counter_8bit_waveform():
    """Verify 8-Bit Synchronous Counter generates valid cycle-accurate counting and load waveform."""
    inputs, expected, rst_n = SynchronousCounter8BitEvaluator.generate_golden_testbench(cycles=32)
    assert len(inputs) == 32
    assert len(expected) == 32
    assert len(rst_n) == 32

    # Check input dimensions: 11 inputs, 9 outputs
    assert len(inputs[0]) == 11
    assert len(expected[0]) == 9

    # Verify reset behavior in cycles 0 and 1
    assert expected[0] == [0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert expected[1] == [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Verify count increments at cycle 2
    assert expected[2] == [1, 0, 0, 0, 0, 0, 0, 0, 0]  # count=1

    # Verify parallel load at cycle 15 (value 200 = 0b11001000)
    assert expected[15] == [0, 0, 0, 1, 0, 0, 1, 1, 0]


def test_shift_register_8bit_waveform():
    """Verify 8-Bit Shift Register shift and parallel load waveform."""
    inputs, expected, rst_n = ShiftRegister8BitEvaluator.generate_golden_testbench(cycles=24)
    assert len(inputs) == 24
    assert len(expected) == 24

    # Check parallel load at cycle 5 (value 0xA5 = 165 = 0b10100101)
    # output: [sout (bit 7), q0..q7]
    assert expected[5][0] == 1  # MSB is 1
    assert expected[5][1:] == [1, 0, 1, 0, 0, 1, 0, 1]


def test_uart_transmitter_fsm_protocol():
    """Verify UART Transmitter FSM emits standard RS-232 serial frame."""
    inputs, expected, rst_n = UARTTransmitterFSMEvaluator.generate_golden_testbench()
    assert len(inputs) == 15
    assert len(expected) == 15

    # Cycle 0-1: Reset -> Serial=1 (idle line), Busy=0
    assert expected[0] == [1, 0]
    assert expected[1] == [1, 0]

    # Cycle 2: Idle with tx_start triggered -> Serial=1, Busy=0
    assert expected[2] == [1, 0]

    # Cycle 3: START BIT asserted -> Serial=0, Busy=1
    assert expected[3] == [0, 1]

    # Cycle 4: DATA bit 0 of 0xB1 (0b10110001 -> LSB is 1)
    assert expected[4] == [1, 1]

    # Cycle 12: STOP BIT asserted -> Serial=1, Busy=1
    assert expected[12] == [1, 1]

    # Cycle 13: Back to IDLE -> Serial=1, Busy=0
    assert expected[13] == [1, 0]


def test_canonical_sequential_circuits_and_metrics():
    """Verify canonical sequential circuit generation and physical metrics calculation."""
    counter = create_canonical_sequential_circuit("counter_8bit")
    assert counter.num_dffs == 8
    assert counter.num_primary_inputs == 11
    assert counter.num_primary_outputs == 9

    # Check metrics
    inputs, expected, rst_n = SynchronousCounter8BitEvaluator.generate_golden_testbench(cycles=10)
    metrics = counter.evaluate_waveform_metrics(inputs, expected, rst_n_timeline=rst_n)

    assert isinstance(metrics, SequentialCircuitMetrics)
    assert metrics.dff_count == 8
    assert metrics.transistor_count >= 8 * 12  # At least 96 transistors for 8 DFFs
    assert metrics.critical_path_delay > 0.0
    assert metrics.cycles_tested == 10


def test_verilog_2001_synchronous_rtl_export():
    """Verify to_verilog outputs valid synchronous Verilog with always @(posedge clk)."""
    counter = create_canonical_sequential_circuit("counter_8bit")
    verilog_code = counter.to_verilog(module_name="soc_counter_8bit")

    assert "module soc_counter_8bit (" in verilog_code
    assert "input  wire clk," in verilog_code
    assert "input  wire rst_n," in verilog_code
    assert "reg  [7:0] dff_q;" in verilog_code
    assert "wire [7:0] dff_d;" in verilog_code
    assert "always @(posedge clk or negedge rst_n) begin" in verilog_code
    assert "dff_q <= 8'b0;" in verilog_code
    assert "dff_q <= dff_d;" in verilog_code
    assert "endmodule" in verilog_code
