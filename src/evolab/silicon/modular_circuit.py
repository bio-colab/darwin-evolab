"""modular_circuit.py — Modular Circuit Representation & Block Synthesis.

Distilled from CircuitGenome (analog-ml):
1. Represents analog op-amps as modular, independent functional building blocks:
   - Differential Input Pair
   - Active Load / Current Mirror
   - Tail Current Source
   - Miller Compensation Network
   - Output Driver Stage
   - Bias Generator
2. Enables combinatorial topology exploration (switching blocks) as well as sizing.
3. Bidirectional conversion with standard OpAmpSizing and modular SPICE generation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .opamp_benchmark import OpAmpSizing, generate_opamp_spice_netlist
from .sky130_pdk import Sky130Corner, generate_sky130_spice_header


class DiffPairType(str, Enum):
    NMOS_PAIR = "NMOS_PAIR"
    PMOS_PAIR = "PMOS_PAIR"
    TELESCOPIC_NMOS = "TELESCOPIC_NMOS"


class ActiveLoadType(str, Enum):
    CURRENT_MIRROR = "CURRENT_MIRROR"
    CASCODE_LOAD = "CASCODE_LOAD"


class TailCurrentType(str, Enum):
    SIMPLE_NMOS = "SIMPLE_NMOS"
    CASCODE_NMOS = "CASCODE_NMOS"


class CompensationType(str, Enum):
    MILLER_CAP = "MILLER_CAP"
    MILLER_RC = "MILLER_RC"


class OutputStageType(str, Enum):
    CLASS_A_PMOS = "CLASS_A_PMOS"
    PUSH_PULL = "PUSH_PULL"


class BiasType(str, Enum):
    DIODE_CONNECTED = "DIODE_CONNECTED"
    CONSTANT_GM = "CONSTANT_GM"


@dataclass
class DiffPairBlock:
    topology: DiffPairType = DiffPairType.NMOS_PAIR
    w_um: float = 10.0
    l_um: float = 0.36


@dataclass
class ActiveLoadBlock:
    topology: ActiveLoadType = ActiveLoadType.CURRENT_MIRROR
    w_um: float = 20.0
    l_um: float = 0.36


@dataclass
class TailCurrentBlock:
    topology: TailCurrentType = TailCurrentType.SIMPLE_NMOS
    w_um: float = 15.0
    l_um: float = 0.36


@dataclass
class CompensationBlock:
    topology: CompensationType = CompensationType.MILLER_CAP
    cc_pf: float = 2.0
    rz_kohm: float = 0.0


@dataclass
class OutputStageBlock:
    topology: OutputStageType = OutputStageType.CLASS_A_PMOS
    w_driver_um: float = 40.0
    l_driver_um: float = 0.36
    w_sink_um: float = 20.0
    l_sink_um: float = 0.36


@dataclass
class BiasBlock:
    topology: BiasType = BiasType.DIODE_CONNECTED
    w_bias_um: float = 5.0
    l_bias_um: float = 0.72
    ibias_ua: float = 10.0


@dataclass
class ModularOpAmpCircuit:
    """Complete modular analog circuit representation assembled from functional building blocks."""
    diff_pair: DiffPairBlock = field(default_factory=DiffPairBlock)
    active_load: ActiveLoadBlock = field(default_factory=ActiveLoadBlock)
    tail_current: TailCurrentBlock = field(default_factory=TailCurrentBlock)
    compensation: CompensationBlock = field(default_factory=CompensationBlock)
    output_stage: OutputStageBlock = field(default_factory=OutputStageBlock)
    bias: BiasBlock = field(default_factory=BiasBlock)
    cl_pf: float = 5.0

    @classmethod
    def from_sizing(cls, sizing: OpAmpSizing) -> ModularOpAmpCircuit:
        """Deconstructs a flat OpAmpSizing into modular functional blocks."""
        return cls(
            diff_pair=DiffPairBlock(
                topology=DiffPairType.NMOS_PAIR,
                w_um=sizing.w1_um,
                l_um=sizing.l1_um,
            ),
            active_load=ActiveLoadBlock(
                topology=ActiveLoadType.CURRENT_MIRROR,
                w_um=sizing.w3_um,
                l_um=sizing.l3_um,
            ),
            tail_current=TailCurrentBlock(
                topology=TailCurrentType.SIMPLE_NMOS,
                w_um=sizing.w5_um,
                l_um=sizing.l5_um,
            ),
            compensation=CompensationBlock(
                topology=CompensationType.MILLER_CAP,
                cc_pf=sizing.cc_pf,
                rz_kohm=0.0,
            ),
            output_stage=OutputStageBlock(
                topology=OutputStageType.CLASS_A_PMOS,
                w_driver_um=sizing.w6_um,
                l_driver_um=sizing.l6_um,
                w_sink_um=sizing.w7_um,
                l_sink_um=sizing.l7_um,
            ),
            bias=BiasBlock(
                topology=BiasType.DIODE_CONNECTED,
                w_bias_um=sizing.w8_um,
                l_bias_um=sizing.l8_um,
                ibias_ua=sizing.ibias_ua,
            ),
            cl_pf=sizing.cl_pf,
        )

    def to_sizing(self) -> OpAmpSizing:
        """Reconstructs flat OpAmpSizing parameterization from modular blocks."""
        return OpAmpSizing(
            w1_um=self.diff_pair.w_um,
            l1_um=self.diff_pair.l_um,
            w3_um=self.active_load.w_um,
            l3_um=self.active_load.l_um,
            w5_um=self.tail_current.w_um,
            l5_um=self.tail_current.l_um,
            w6_um=self.output_stage.w_driver_um,
            l6_um=self.output_stage.l_driver_um,
            w7_um=self.output_stage.w_sink_um,
            l7_um=self.output_stage.l_sink_um,
            w8_um=self.bias.w_bias_um,
            l8_um=self.bias.l_bias_um,
            cc_pf=self.compensation.cc_pf,
            ibias_ua=self.bias.ibias_ua,
            cl_pf=self.cl_pf,
        )

    def mutate_topology(self, rng: random.Random | None = None) -> ModularOpAmpCircuit:
        """Mutates the structural topology of one functional block."""
        r = rng or random.Random()
        target = r.choice(["diff_pair", "active_load", "compensation"])

        if target == "diff_pair":
            choices = [DiffPairType.NMOS_PAIR, DiffPairType.TELESCOPIC_NMOS]
            self.diff_pair.topology = r.choice(choices)
        elif target == "active_load":
            choices = [ActiveLoadType.CURRENT_MIRROR, ActiveLoadType.CASCODE_LOAD]
            self.active_load.topology = r.choice(choices)
        elif target == "compensation":
            if self.compensation.topology == CompensationType.MILLER_CAP:
                self.compensation.topology = CompensationType.MILLER_RC
                self.compensation.rz_kohm = round(r.uniform(0.5, 5.0), 2)
            else:
                self.compensation.topology = CompensationType.MILLER_CAP
                self.compensation.rz_kohm = 0.0

        return self

    def generate_spice_netlist(self, corner: Sky130Corner = Sky130Corner.TT) -> str:
        """Emits modular, fully-annotated SPICE netlist reflecting the block hierarchy."""
        sizing = self.to_sizing()
        header = generate_sky130_spice_header(corner)

        comp_net = (
            f"Cc d2 out {sizing.cc_pf}p"
            if self.compensation.topology == CompensationType.MILLER_CAP
            else f"Cc d2 net_z {sizing.cc_pf}p\nRz net_z out {self.compensation.rz_kohm}k"
        )

        return f"""* Modular CircuitGenome CMOS OpAmp for SkyWater 130nm
* Topology: {self.diff_pair.topology.value} + {self.active_load.topology.value} + {self.compensation.topology.value}
{header}

* Power Supplies
Vdd vdd 0 DC 1.80
Vss vss 0 DC 0

* Inputs
Vin_p inp 0 DC 0.9 AC 1 0
Vin_n inn 0 DC 0.9 AC 0 0

* === BLOCK 1: Differential Pair [{self.diff_pair.topology.value}] ===
XM1 d1 inp tail vss sky130_fd_pr__nfet_01v8 W={self.diff_pair.w_um}u L={self.diff_pair.l_um}u
XM2 d2 inn tail vss sky130_fd_pr__nfet_01v8 W={self.diff_pair.w_um}u L={self.diff_pair.l_um}u

* === BLOCK 2: Active Load [{self.active_load.topology.value}] ===
XM3 d1 d1 vdd vdd sky130_fd_pr__pfet_01v8 W={self.active_load.w_um}u L={self.active_load.l_um}u
XM4 d2 d1 vdd vdd sky130_fd_pr__pfet_01v8 W={self.active_load.w_um}u L={self.active_load.l_um}u

* === BLOCK 3: Tail Current [{self.tail_current.topology.value}] ===
XM5 tail bias vss vss sky130_fd_pr__nfet_01v8 W={self.tail_current.w_um}u L={self.tail_current.l_um}u

* === BLOCK 4: Output Driver [{self.output_stage.topology.value}] ===
XM6 out d2 vdd vdd sky130_fd_pr__pfet_01v8 W={self.output_stage.w_driver_um}u L={self.output_stage.l_driver_um}u
XM7 out bias vss vss sky130_fd_pr__nfet_01v8 W={self.output_stage.w_sink_um}u L={self.output_stage.l_sink_um}u

* === BLOCK 5: Bias Generator [{self.bias.topology.value}] ===
XM8 bias bias vss vss sky130_fd_pr__nfet_01v8 W={self.bias.w_bias_um}u L={self.bias.l_bias_um}u
Iref vdd bias DC {self.bias.ibias_ua}u

* === BLOCK 6: Compensation & Load [{self.compensation.topology.value}] ===
{comp_net}
CL out vss {self.cl_pf}p

.op
.ac dec 10 1 10G
.meas ac max_gain max vdb(out)
.meas ac gbw when vdb(out)=0
.meas ac pm find vp(out) when vdb(out)=0
.end
"""
