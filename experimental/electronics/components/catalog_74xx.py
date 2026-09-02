"""
catalog_74xx.py — Ground-Truth Catalog of 74HC Series Logic Integrated Circuits.
Extracted from Texas Instruments standard DIP packages (SCLSxxx datasheets).
"""
from __future__ import annotations

from .specs import (
    ElectricalLimits,
    GateUnit,
    ICPackageSpec,
    LogicFunction,
    TimingSpec,
)


def _create_74hc00() -> ICPackageSpec:
    """74HC00: Quad 2-Input NAND Gate (TI SCLS081E)."""
    gates = (
        GateUnit(1, LogicFunction.NAND, (1, 2), (3,)),
        GateUnit(2, LogicFunction.NAND, (4, 5), (6,)),
        GateUnit(3, LogicFunction.NAND, (9, 10), (8,)),
        GateUnit(4, LogicFunction.NAND, (12, 13), (11,)),
    )
    return ICPackageSpec(
        part_number="74HC00",
        description="Quad 2-Input Positive-NAND Gates",
        datasheet_id="TI-SCLS081E",
        pin_count=14,
        vcc_pin=14,
        gnd_pin=7,
        electrical=ElectricalLimits(icc_quiescent_max_ua=20.0),
        timing=TimingSpec(
            tpd_typ_ns_at_4_5v=9.0,
            tpd_max_ns_at_4_5v_25c=18.0,
            tpd_max_ns_at_4_5v_industrial=23.0,
            tpd_max_ns_at_2_0v_25c=90.0,
            tpd_max_ns_at_6_0v_25c=15.0,
        ),
        gates=gates,
    )


def _create_74hc02() -> ICPackageSpec:
    """74HC02: Quad 2-Input NOR Gate (TI SCLS083D)."""
    gates = (
        GateUnit(1, LogicFunction.NOR, (2, 3), (1,)),
        GateUnit(2, LogicFunction.NOR, (5, 6), (4,)),
        GateUnit(3, LogicFunction.NOR, (8, 9), (10,)),
        GateUnit(4, LogicFunction.NOR, (11, 12), (13,)),
    )
    return ICPackageSpec(
        part_number="74HC02",
        description="Quad 2-Input Positive-NOR Gates",
        datasheet_id="TI-SCLS083D",
        pin_count=14,
        vcc_pin=14,
        gnd_pin=7,
        electrical=ElectricalLimits(icc_quiescent_max_ua=20.0),
        timing=TimingSpec(
            tpd_typ_ns_at_4_5v=9.0,
            tpd_max_ns_at_4_5v_25c=19.0,
            tpd_max_ns_at_4_5v_industrial=24.0,
            tpd_max_ns_at_2_0v_25c=95.0,
            tpd_max_ns_at_6_0v_25c=16.0,
        ),
        gates=gates,
    )


def _create_74hc04() -> ICPackageSpec:
    """74HC04: Hex Inverter (TI SCLS085E)."""
    gates = (
        GateUnit(1, LogicFunction.NOT, (1,), (2,)),
        GateUnit(2, LogicFunction.NOT, (3,), (4,)),
        GateUnit(3, LogicFunction.NOT, (5,), (6,)),
        GateUnit(4, LogicFunction.NOT, (9,), (8,)),
        GateUnit(5, LogicFunction.NOT, (11,), (10,)),
        GateUnit(6, LogicFunction.NOT, (13,), (12,)),
    )
    return ICPackageSpec(
        part_number="74HC04",
        description="Hex Inverters",
        datasheet_id="TI-SCLS085E",
        pin_count=14,
        vcc_pin=14,
        gnd_pin=7,
        electrical=ElectricalLimits(icc_quiescent_max_ua=20.0),
        timing=TimingSpec(
            tpd_typ_ns_at_4_5v=7.0,
            tpd_max_ns_at_4_5v_25c=15.0,
            tpd_max_ns_at_4_5v_industrial=19.0,
            tpd_max_ns_at_2_0v_25c=85.0,
            tpd_max_ns_at_6_0v_25c=14.0,
        ),
        gates=gates,
    )


def _create_74hc08() -> ICPackageSpec:
    """74HC08: Quad 2-Input AND Gate (TI SCLS089D)."""
    gates = (
        GateUnit(1, LogicFunction.AND, (1, 2), (3,)),
        GateUnit(2, LogicFunction.AND, (4, 5), (6,)),
        GateUnit(3, LogicFunction.AND, (9, 10), (8,)),
        GateUnit(4, LogicFunction.AND, (12, 13), (11,)),
    )
    return ICPackageSpec(
        part_number="74HC08",
        description="Quad 2-Input Positive-AND Gates",
        datasheet_id="TI-SCLS089D",
        pin_count=14,
        vcc_pin=14,
        gnd_pin=7,
        electrical=ElectricalLimits(icc_quiescent_max_ua=20.0),
        timing=TimingSpec(
            tpd_typ_ns_at_4_5v=10.0,
            tpd_max_ns_at_4_5v_25c=20.0,
            tpd_max_ns_at_4_5v_industrial=25.0,
            tpd_max_ns_at_2_0v_25c=100.0,
            tpd_max_ns_at_6_0v_25c=17.0,
        ),
        gates=gates,
    )


def _create_74hc32() -> ICPackageSpec:
    """74HC32: Quad 2-Input OR Gate (TI SCLS093D)."""
    gates = (
        GateUnit(1, LogicFunction.OR, (1, 2), (3,)),
        GateUnit(2, LogicFunction.OR, (4, 5), (6,)),
        GateUnit(3, LogicFunction.OR, (9, 10), (8,)),
        GateUnit(4, LogicFunction.OR, (12, 13), (11,)),
    )
    return ICPackageSpec(
        part_number="74HC32",
        description="Quad 2-Input Positive-OR Gates",
        datasheet_id="TI-SCLS093D",
        pin_count=14,
        vcc_pin=14,
        gnd_pin=7,
        electrical=ElectricalLimits(icc_quiescent_max_ua=20.0),
        timing=TimingSpec(
            tpd_typ_ns_at_4_5v=10.0,
            tpd_max_ns_at_4_5v_25c=20.0,
            tpd_max_ns_at_4_5v_industrial=25.0,
            tpd_max_ns_at_2_0v_25c=100.0,
            tpd_max_ns_at_6_0v_25c=17.0,
        ),
        gates=gates,
    )


def _create_74hc86() -> ICPackageSpec:
    """74HC86: Quad 2-Input XOR Gate (TI SCLS099E)."""
    gates = (
        GateUnit(1, LogicFunction.XOR, (1, 2), (3,)),
        GateUnit(2, LogicFunction.XOR, (4, 5), (6,)),
        GateUnit(3, LogicFunction.XOR, (9, 10), (8,)),
        GateUnit(4, LogicFunction.XOR, (12, 13), (11,)),
    )
    return ICPackageSpec(
        part_number="74HC86",
        description="Quad 2-Input Exclusive-OR Gates",
        datasheet_id="TI-SCLS099E",
        pin_count=14,
        vcc_pin=14,
        gnd_pin=7,
        electrical=ElectricalLimits(icc_quiescent_max_ua=20.0),
        timing=TimingSpec(
            tpd_typ_ns_at_4_5v=11.0,
            tpd_max_ns_at_4_5v_25c=22.0,
            tpd_max_ns_at_4_5v_industrial=28.0,
            tpd_max_ns_at_2_0v_25c=115.0,
            tpd_max_ns_at_6_0v_25c=19.0,
        ),
        gates=gates,
    )


def _create_74hc138() -> ICPackageSpec:
    """74HC138: 3-to-8 Line Decoder/Demultiplexer (TI SCLS134D)."""
    gates = (
        GateUnit(1, LogicFunction.DECODER_3TO8, (1, 2, 3), (15, 14, 13, 12, 11, 10, 9, 7)),
    )
    return ICPackageSpec(
        part_number="74HC138",
        description="3-Line To 8-Line Decoders/Demultiplexers",
        datasheet_id="TI-SCLS134D",
        pin_count=16,
        vcc_pin=16,
        gnd_pin=8,
        electrical=ElectricalLimits(icc_quiescent_max_ua=80.0),
        timing=TimingSpec(
            tpd_typ_ns_at_4_5v=15.0,
            tpd_max_ns_at_4_5v_25c=30.0,
            tpd_max_ns_at_4_5v_industrial=38.0,
            tpd_max_ns_at_2_0v_25c=150.0,
            tpd_max_ns_at_6_0v_25c=26.0,
        ),
        gates=gates,
    )


def _create_74hc151() -> ICPackageSpec:
    """74HC151: 8-Line to 1-Line Data Selector/Multiplexer (TI SCLS139C).

    Data pins 1-4,6-9 then select S0-S2 on 11,12,13. Y on pin 5.
    """
    gates = (
        GateUnit(1, LogicFunction.MUX_8TO1, (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13), (5,)),
    )
    return ICPackageSpec(
        part_number="74HC151",
        description="8-Line To 1-Line Data Selectors/Multiplexers",
        datasheet_id="TI-SCLS139C",
        pin_count=16,
        vcc_pin=16,
        gnd_pin=8,
        electrical=ElectricalLimits(icc_quiescent_max_ua=80.0),
        timing=TimingSpec(
            tpd_typ_ns_at_4_5v=14.0,
            tpd_max_ns_at_4_5v_25c=28.0,
            tpd_max_ns_at_4_5v_industrial=35.0,
            tpd_max_ns_at_2_0v_25c=140.0,
            tpd_max_ns_at_6_0v_25c=24.0,
        ),
        gates=gates,
    )


def _create_74hc283() -> ICPackageSpec:
    """74HC283: 4-Bit Binary Full Adder (TI SCLS149C)."""
    gates = (
        GateUnit(1, LogicFunction.ADDER_4BIT, (1, 2, 3, 4, 5, 6, 7, 9), (10, 11, 12, 13, 14)),
    )
    return ICPackageSpec(
        part_number="74HC283",
        description="4-Bit Binary Full Adders With Fast Carry",
        datasheet_id="TI-SCLS149C",
        pin_count=16,
        vcc_pin=16,
        gnd_pin=8,
        electrical=ElectricalLimits(icc_quiescent_max_ua=80.0),
        timing=TimingSpec(
            tpd_typ_ns_at_4_5v=18.0,
            tpd_max_ns_at_4_5v_25c=36.0,
            tpd_max_ns_at_4_5v_industrial=45.0,
            tpd_max_ns_at_2_0v_25c=180.0,
            tpd_max_ns_at_6_0v_25c=30.0,
        ),
        gates=gates,
    )


# Catalog registry mapping part number to full datasheet spec
CATALOG_74XX: dict[str, ICPackageSpec] = {
    "74HC00": _create_74hc00(),
    "74HC02": _create_74hc02(),
    "74HC04": _create_74hc04(),
    "74HC08": _create_74hc08(),
    "74HC32": _create_74hc32(),
    "74HC86": _create_74hc86(),
    "74HC138": _create_74hc138(),
    "74HC151": _create_74hc151(),
    "74HC283": _create_74hc283(),
}


def get_ic_spec(part_number: str) -> ICPackageSpec:
    """Retrieves the datasheet specification for a 74xx IC."""
    if part_number not in CATALOG_74XX:
        raise KeyError(f"IC {part_number} not found in 74xx catalog. Available: {list(CATALOG_74XX.keys())}")
    return CATALOG_74XX[part_number]
