"""
specs.py — Standard Datasheet Electrical, Timing, and Functional Contracts.
Extracted from official Texas Instruments and Nexperia 74HC-family datasheets.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LogicFunction(str, Enum):
    NAND = "NAND"
    NOR = "NOR"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    DECODER_3TO8 = "DECODER_3TO8"
    MUX_8TO1 = "MUX_8TO1"
    ADDER_4BIT = "ADDER_4BIT"


@dataclass(frozen=True)
class ElectricalLimits:
    """Operating DC electrical limits extracted from manufacturer datasheets."""

    vcc_min_v: float = 2.0
    vcc_max_v: float = 6.0
    vcc_nominal_v: float = 5.0
    icc_quiescent_max_ua: float = 20.0  # Max quiescent supply current per package @ 25C
    v_ih_min_v: float = 3.15            # High-level input voltage threshold @ Vcc=4.5V
    v_il_max_v: float = 1.35            # Low-level input voltage threshold @ Vcc=4.5V
    v_oh_min_v: float = 4.4             # High-level output voltage @ Vcc=4.5V, Ioh=-20uA
    v_ol_max_v: float = 0.1             # Low-level output voltage @ Vcc=4.5V, Iol=20uA


@dataclass(frozen=True)
class TimingCorner:
    """One published datasheet timing point. Not an interpolated model."""

    vcc_v: float
    temp_c: float
    tpd_max_ns: float
    cl_pf: float = 50.0


@dataclass(frozen=True)
class TimingSpec:
    """Propagation delay parameters (t_pd) in nanoseconds at various supply voltages and temperatures."""

    # Values at Vcc = 4.5V, CL = 50pF
    tpd_typ_ns_at_4_5v: float = 9.0
    tpd_max_ns_at_4_5v_25c: float = 18.0
    tpd_max_ns_at_4_5v_industrial: float = 23.0  # -40C to +85C

    # Values at Vcc = 2.0V, CL = 50pF
    tpd_max_ns_at_2_0v_25c: float = 90.0

    # Values at Vcc = 6.0V, CL = 50pF
    tpd_max_ns_at_6_0v_25c: float = 15.0

    DATASHEET_CL_PF = 50.0
    INTRINSIC_C_PF = 10.0

    def published_corners(self) -> tuple[TimingCorner, ...]:
        return (
            TimingCorner(4.5, 25.0, self.tpd_max_ns_at_4_5v_25c, 50.0),
            TimingCorner(4.5, 85.0, self.tpd_max_ns_at_4_5v_industrial, 50.0),
            TimingCorner(2.0, 25.0, self.tpd_max_ns_at_2_0v_25c, 50.0),
            TimingCorner(6.0, 25.0, self.tpd_max_ns_at_6_0v_25c, 50.0),
        )

    def lookup_published_tpd_ns(self, vcc: float, temp_c: float) -> float:
        """Exact published point if present; otherwise nearest published point. No derating."""
        corners = self.published_corners()
        for c in corners:
            if abs(c.vcc_v - vcc) < 0.06 and abs(c.temp_c - temp_c) < 1.0:
                return c.tpd_max_ns
        return min(
            corners,
            key=lambda c: abs(c.vcc_v - vcc) + 0.02 * abs(c.temp_c - temp_c),
        ).tpd_max_ns

    def get_delay_ns(self, vcc: float, temp_c: float = 25.0, cl_pf: float = 50.0) -> float:
        """Published t_pd at the closest datasheet corner, scaled only for CL."""
        base = self.lookup_published_tpd_ns(vcc, temp_c)
        load = (self.INTRINSIC_C_PF + max(cl_pf, 0.0)) / (self.INTRINSIC_C_PF + self.DATASHEET_CL_PF)
        return round(base * load, 2)

    def get_transition_ns(self, vcc: float, temp_c: float = 25.0, cl_pf: float = 50.0) -> float:
        """10–90 transition estimate; scales with the same load as t_pd."""
        return round(0.35 * self.get_delay_ns(vcc, temp_c, cl_pf), 2)


@dataclass(frozen=True)
class GateUnit:
    """A single functional sub-gate inside a multi-gate IC package."""

    unit_id: int
    logic_fn: LogicFunction
    input_pins: tuple[int, ...]
    output_pins: tuple[int, ...]


@dataclass(frozen=True)
class ICPackageSpec:
    """Full architectural specification of a commercial 74xx IC DIP package."""

    part_number: str
    description: str
    datasheet_id: str
    pin_count: int
    vcc_pin: int
    gnd_pin: int
    electrical: ElectricalLimits
    timing: TimingSpec
    gates: tuple[GateUnit, ...]
