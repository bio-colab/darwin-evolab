"""
independent_verifier.py — Layer 2: Independent Reference Calculator.
Verifies truth tables and theoretical propagation delays independently of the circuit model.
"""
from __future__ import annotations

from collections.abc import Sequence


class IndependentDigitalVerifier:
    """Mathematical reference verifier using analytical Boolean equations."""

    @staticmethod
    def half_adder_reference(a: int, b: int) -> tuple[int, int]:
        """Ground truth: Sum = A ^ B, Cout = A & B."""
        return ((a ^ b) & 1, (a & b) & 1)

    @staticmethod
    def full_adder_reference(a: int, b: int, cin: int) -> tuple[int, int]:
        """Ground truth: Sum = A ^ B ^ Cin, Cout = (A & B) | (Cin & (A ^ B))."""
        sum_bit = (a ^ b ^ cin) & 1
        cout_bit = 1 if ((a + b + cin) > 1) else 0
        return (sum_bit, cout_bit)

    @staticmethod
    def multiplexer_4to1_reference(inputs: Sequence[int], select: tuple[int, int]) -> int:
        """Ground truth: Mux 4-to-1."""
        sel_idx = (select[1] << 1) | select[0]
        return inputs[sel_idx] & 1

    @staticmethod
    def decoder_3to8_reference(a0: int, a1: int, a2: int) -> tuple[int, ...]:
        """74HC138-style active-low one-hot."""
        idx = ((a2 & 1) << 2) | ((a1 & 1) << 1) | (a0 & 1)
        out = [1] * 8
        out[idx] = 0
        return tuple(out)

    @staticmethod
    def mux_8to1_reference(data: Sequence[int], select: tuple[int, int, int]) -> int:
        sel = (select[0] & 1) | ((select[1] & 1) << 1) | ((select[2] & 1) << 2)
        padded = list(data) + [0] * 8
        return padded[sel] & 1
