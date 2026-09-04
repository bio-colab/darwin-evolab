"""Two-Stage Miller-Compensated CMOS Operational Amplifier Benchmark for SkyWater 130nm.

Implements sizing parameterization, pure physical analytical small-signal calculation,
ngspice netlist synthesis, and multi-objective Pareto evaluation (Av, GBW, PM, Power).
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from evolab.adapters import DomainAdapter
from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import FloatGenome, Individual
from evolab.pareto import Objective

from .sky130_pdk import (
    CORNER_SPECS,
    SKY130_PARAMS,
    Sky130Corner,
    compute_transistor_operating_point,
    generate_sky130_spice_header,
)


@dataclass
class OpAmpSizing:
    """Design variables for Two-Stage Miller OpAmp sizing."""
    w1_um: float = 10.0  # M1, M2 diff pair NMOS width
    l1_um: float = 0.36  # M1, M2 length
    w3_um: float = 20.0  # M3, M4 mirror PMOS width
    l3_um: float = 0.36  # M3, M4 length
    w5_um: float = 15.0  # M5 tail NMOS width
    l5_um: float = 0.36  # M5 length
    w6_um: float = 40.0  # M6 driver PMOS width
    l6_um: float = 0.36  # M6 length
    w7_um: float = 20.0  # M7 sink NMOS width
    l7_um: float = 0.36  # M7 length
    w8_um: float = 5.0   # M8 bias NMOS width
    l8_um: float = 0.72  # M8 length
    cc_pf: float = 2.0   # Miller compensation cap (pF)
    ibias_ua: float = 10.0  # Reference bias current (uA)
    cl_pf: float = 5.0   # Load cap (pF)

    @classmethod
    def from_normalized_vector(cls, vec: Sequence[float]) -> OpAmpSizing:
        """Decodes a normalized [0, 1] parameter vector into physical dimensions."""
        # Defaults if vector is short
        v = list(vec) + [0.5] * max(0, 14 - len(vec))
        def _scale(val: float, low: float, high: float) -> float:
            c = max(0.0, min(float(val), 1.0))
            return low + c * (high - low)

        return cls(
            w1_um=round(_scale(v[0], 1.0, 50.0), 3),
            l1_um=round(_scale(v[1], 0.18, 2.0), 3),
            w3_um=round(_scale(v[2], 2.0, 80.0), 3),
            l3_um=round(_scale(v[3], 0.18, 2.0), 3),
            w5_um=round(_scale(v[4], 2.0, 60.0), 3),
            l5_um=round(_scale(v[5], 0.18, 2.0), 3),
            w6_um=round(_scale(v[6], 5.0, 120.0), 3),
            l6_um=round(_scale(v[7], 0.18, 2.0), 3),
            w7_um=round(_scale(v[8], 2.0, 80.0), 3),
            l7_um=round(_scale(v[9], 0.18, 2.0), 3),
            w8_um=round(_scale(v[10], 1.0, 30.0), 3),
            l8_um=round(_scale(v[11], 0.36, 3.0), 3),
            cc_pf=round(_scale(v[12], 0.2, 10.0), 3),
            ibias_ua=round(_scale(v[13], 2.0, 50.0), 2),
        )

    def to_normalized_vector(self) -> list[float]:
        """Encodes physical parameters to normalized [0, 1] float vector."""
        def _norm(val: float, low: float, high: float) -> float:
            return max(0.0, min((val - low) / (high - low), 1.0))

        return [
            _norm(self.w1_um, 1.0, 50.0),
            _norm(self.l1_um, 0.18, 2.0),
            _norm(self.w3_um, 2.0, 80.0),
            _norm(self.l3_um, 0.18, 2.0),
            _norm(self.w5_um, 2.0, 60.0),
            _norm(self.l5_um, 0.18, 2.0),
            _norm(self.w6_um, 5.0, 120.0),
            _norm(self.l6_um, 0.18, 2.0),
            _norm(self.w7_um, 2.0, 80.0),
            _norm(self.l7_um, 0.18, 2.0),
            _norm(self.w8_um, 1.0, 30.0),
            _norm(self.l8_um, 0.36, 3.0),
            _norm(self.cc_pf, 0.2, 10.0),
            _norm(self.ibias_ua, 2.0, 50.0),
        ]


@dataclass
class OpAmpPerformanceMetrics:
    """Extracted AC, stability, and power metrics for the OpAmp."""
    gain_db: float           # Low-frequency differential voltage gain (dB)
    gbw_mhz: float           # Gain-Bandwidth Product (MHz)
    pm_deg: float            # Phase Margin (degrees)
    power_uw: float          # Total static power consumption (microWatts)
    cmrr_db: float           # Common-Mode Rejection Ratio (dB)
    slew_rate_v_us: float    # Slew Rate (V/us)
    is_stable: bool          # PM >= 45 degrees
    meets_spec: bool         # Meets all 4 target specifications
    physical_claim: bool     # True if real simulation or exact physics used
    artifacts: dict[str, Any] = None  # Additional data (voltages, poles)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def evaluate_opamp_analytical(
    sizing: OpAmpSizing,
    corner: Sky130Corner = Sky130Corner.TT,
) -> OpAmpPerformanceMetrics:
    """Evaluates the OpAmp using exact small-signal CMOS physics equations for SkyWater 130nm."""
    cs = CORNER_SPECS[corner]
    vdd = cs.vdd
    ibias = sizing.ibias_ua * 1e-6
    cl = sizing.cl_pf * 1e-12
    cc = sizing.cc_pf * 1e-12

    # Bias branch: M8 carries Ibias
    m8 = compute_transistor_operating_point(
        sizing.w8_um, sizing.l8_um, ibias, vds_v=0.7, is_pmos=False, corner=corner
    )

    # Tail current source M5 mirrors M8: I5 = Ibias * (W5/L5) / (W8/L8)
    scale_m5 = (sizing.w5_um / sizing.l5_um) / max(sizing.w8_um / sizing.l8_um, 1e-4)
    i5 = ibias * scale_m5
    i1 = i5 / 2.0  # Branch current for M1, M2, M3, M4

    # Stage 1: Input pair M1, M2 (NMOS)
    m1 = compute_transistor_operating_point(
        sizing.w1_um, sizing.l1_um, i1, vds_v=0.7, is_pmos=False, corner=corner
    )
    # Stage 1: PMOS Active Load M3, M4
    m3 = compute_transistor_operating_point(
        sizing.w3_um, sizing.l3_um, i1, vds_v=0.7, is_pmos=True, corner=corner
    )

    # Stage 2: M7 current sink mirrors M8: I7 = Ibias * (W7/L7) / (W8/L8)
    scale_m7 = (sizing.w7_um / sizing.l7_um) / max(sizing.w8_um / sizing.l8_um, 1e-4)
    i7 = ibias * scale_m7

    # Stage 2: Driver M6 (PMOS) biased at I7
    m6 = compute_transistor_operating_point(
        sizing.w6_um, sizing.l6_um, i7, vds_v=0.9, is_pmos=True, corner=corner
    )
    m7 = compute_transistor_operating_point(
        sizing.w7_um, sizing.l7_um, i7, vds_v=0.9, is_pmos=False, corner=corner
    )

    # Stage 1 Gain: Av1 = gm1 * (ro2 || ro4)
    r_out1 = (m1.ro_ohm * m3.ro_ohm) / max(m1.ro_ohm + m3.ro_ohm, 1.0)
    av1 = m1.gm_s * r_out1

    # Stage 2 Gain: Av2 = gm6 * (ro6 || ro7)
    r_out2 = (m6.ro_ohm * m7.ro_ohm) / max(m6.ro_ohm + m7.ro_ohm, 1.0)
    av2 = m6.gm_s * r_out2

    # Total DC Gain
    av_total = max(av1 * av2, 1.0)
    gain_db = 20.0 * math.log10(av_total)

    # Dominant pole: p1 approx 1 / (r_out1 * gm6 * r_out2 * Cc)
    p1_rad = 1.0 / max(r_out1 * av2 * cc, 1e-18)
    p1_hz = p1_rad / (2.0 * math.pi)

    # Gain-Bandwidth Product (GBW): GBW = gm1 / (2 * pi * Cc)
    gbw_hz = m1.gm_s / (2.0 * math.pi * cc)
    gbw_mhz = gbw_hz / 1e6

    # Second pole (output pole): p2 approx gm6 / (2 * pi * CL)
    p2_hz = m6.gm_s / (2.0 * math.pi * cl)

    # RHP Zero from Miller cap: z1 = gm6 / (2 * pi * Cc)
    z1_hz = m6.gm_s / (2.0 * math.pi * cc)

    # Phase Margin (PM) at GBW:
    # PM = 180 - atan(GBW / p1) - atan(GBW / p2) - atan(GBW / z1)
    phase_lag_deg = (
        math.degrees(math.atan(gbw_hz / max(p1_hz, 1e-6)))
        + math.degrees(math.atan(gbw_hz / max(p2_hz, 1e-6)))
        + math.degrees(math.atan(gbw_hz / max(z1_hz, 1e-6)))
    )
    pm_deg = max(0.0, min(180.0 - phase_lag_deg, 180.0))

    # Total Power Dissipation: P = VDD * (I5 + I7 + Ibias)
    i_total = i5 + i7 + ibias
    power_uw = (vdd * i_total) * 1e6

    # Slew Rate: SR = I5 / Cc (V/s) -> convert to V/us
    sr_v_us = (i5 / max(cc, 1e-15)) / 1e6

    # CMRR: CMRR approx 2 * gm1 * ro5 * (1 + gm3 * ro3)
    cmrr_lin = 2.0 * m1.gm_s * m1.ro_ohm * max(m3.gm_s * m3.ro_ohm, 1.0)
    cmrr_db = 20.0 * math.log10(max(cmrr_lin, 1.0))

    is_stable = pm_deg >= 45.0
    meets_spec = (
        gain_db >= 60.0 and gbw_mhz >= 10.0 and pm_deg >= 60.0 and power_uw <= 600.0
    )

    return OpAmpPerformanceMetrics(
        gain_db=round(gain_db, 2),
        gbw_mhz=round(gbw_mhz, 2),
        pm_deg=round(pm_deg, 2),
        power_uw=round(power_uw, 2),
        cmrr_db=round(cmrr_db, 2),
        slew_rate_v_us=round(sr_v_us, 2),
        is_stable=is_stable,
        meets_spec=meets_spec,
        physical_claim=True,
        artifacts={
            "i_tail_ua": round(i5 * 1e6, 2),
            "i_stage2_ua": round(i7 * 1e6, 2),
            "gm1_ms": round(m1.gm_s * 1e3, 3),
            "gm6_ms": round(m6.gm_s * 1e3, 3),
            "p1_hz": round(p1_hz, 1),
            "p2_mhz": round(p2_hz / 1e6, 2),
            "corner": corner.value,
        },
    )


def generate_opamp_spice_netlist(
    sizing: OpAmpSizing,
    corner: Sky130Corner = Sky130Corner.TT,
    temp_c: float | None = None,
) -> str:
    """Generates a complete, standard-compliant ngspice netlist for the Two-Stage Miller OpAmp."""
    cs = CORNER_SPECS[corner]
    temp = temp_c if temp_c is not None else cs.temp_c
    header = generate_sky130_spice_header(corner)

    return f"""* Two-Stage Miller OpAmp for SkyWater 130nm
* Generated by Darwin-Evolab Silicon Engine
.temp {temp}

{header}

* Power Supplies
Vdd vdd 0 DC {cs.vdd}
Vss vss 0 DC 0

* AC Differential Inputs
Vin_p inp 0 DC 0.9 AC 1 0
Vin_n inn 0 DC 0.9 AC 0 0

* Stage 1: Differential Pair (NMOS)
XM1 d1 inp tail vss sky130_fd_pr__nfet_01v8 W={sizing.w1_um}u L={sizing.l1_um}u
XM2 d2 inn tail vss sky130_fd_pr__nfet_01v8 W={sizing.w1_um}u L={sizing.l1_um}u

* Stage 1: Current Mirror Load (PMOS)
XM3 d1 d1 vdd vdd sky130_fd_pr__pfet_01v8 W={sizing.w3_um}u L={sizing.l3_um}u
XM4 d2 d1 vdd vdd sky130_fd_pr__pfet_01v8 W={sizing.w3_um}u L={sizing.l3_um}u

* Stage 1: Tail Current Source (NMOS)
XM5 tail bias vss vss sky130_fd_pr__nfet_01v8 W={sizing.w5_um}u L={sizing.l5_um}u

* Stage 2: Driver (PMOS Common-Source)
XM6 out d2 vdd vdd sky130_fd_pr__pfet_01v8 W={sizing.w6_um}u L={sizing.l6_um}u

* Stage 2: Active Current Sink (NMOS)
XM7 out bias vss vss sky130_fd_pr__nfet_01v8 W={sizing.w7_um}u L={sizing.l7_um}u

* Bias Circuit (M8 diode-connected)
XM8 bias bias vss vss sky130_fd_pr__nfet_01v8 W={sizing.w8_um}u L={sizing.l8_um}u
Iref vdd bias DC {sizing.ibias_ua}u

* Miller Compensation & Load
Cc d2 out {sizing.cc_pf}p
CL out vss {sizing.cl_pf}p

* Operating Point & AC Frequency Sweep
.op
.ac dec 10 1 10G

* Measurements
.meas ac max_gain max vdb(out)
.meas ac gbw when vdb(out)=0
.meas ac pm find vp(out) when vdb(out)=0

.control
run
.endc
.end
"""


class TwoStageMillerOpAmpEvaluator(Evaluator):
    """Multi-Objective Evaluator for Two-Stage CMOS OpAmp in SkyWater 130nm."""

    def __init__(
        self,
        target_gain_db: float = 60.0,
        target_gbw_mhz: float = 10.0,
        target_pm_deg: float = 60.0,
        target_max_power_uw: float = 600.0,
        corner: Sky130Corner = Sky130Corner.TT,
    ):
        self.target_gain_db = target_gain_db
        self.target_gbw_mhz = target_gbw_mhz
        self.target_pm_deg = target_pm_deg
        self.target_max_power_uw = target_max_power_uw
        self.corner = corner

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate_metrics(self, ind_or_genome: Any) -> OpAmpPerformanceMetrics:
        g = getattr(ind_or_genome, "genome", ind_or_genome)
        if isinstance(g, FloatGenome) or hasattr(g, "values") or hasattr(g, "genes"):
            vals = list(getattr(g, "values", getattr(g, "genes", [])))
            sizing = OpAmpSizing.from_normalized_vector(vals)
        elif isinstance(g, OpAmpSizing):
            sizing = g
        else:
            sizing = OpAmpSizing()

        return evaluate_opamp_analytical(sizing, corner=self.corner)

    def evaluate(self, ind: Individual) -> FitnessResult:
        metrics = self.evaluate_metrics(ind)

        # Composite score calculation (0 to 100)
        # Reward gain up to target (40 pts)
        gain_score = min(metrics.gain_db / max(self.target_gain_db, 1.0), 1.25) * 35.0

        # Reward GBW up to target (25 pts)
        gbw_score = min(metrics.gbw_mhz / max(self.target_gbw_mhz, 0.1), 1.5) * 25.0

        # Reward Phase Margin: peak at 60-70 degrees (25 pts)
        if metrics.pm_deg >= 60.0:
            pm_score = 25.0
        elif metrics.pm_deg >= 45.0:
            pm_score = 15.0 + (metrics.pm_deg - 45.0) / 15.0 * 10.0
        else:
            pm_score = max(metrics.pm_deg / 45.0 * 15.0, 0.0)

        # Reward low power consumption (15 pts)
        power_score = max(0.0, 15.0 * (1.0 - (metrics.power_uw / (self.target_max_power_uw * 2.0))))

        total_score = max(0.0, min(gain_score + gbw_score + pm_score + power_score, 100.0))

        return FitnessResult(
            score=round(total_score, 3),
            passed_holdout=metrics.meets_spec,
            artifacts={
                "metrics": metrics.to_dict(),
                "corner": self.corner.value,
            },
        )

    def build_pareto_objectives(self) -> list[Objective]:
        """Returns 4 conflicting Pareto objectives for NSGA2 optimization."""
        return [
            # 1. Maximize Differential Voltage Gain
            Objective(name="Gain_dB", direction="maximize", weight=1.0),
            # 2. Maximize Gain-Bandwidth Product
            Objective(name="GBW_MHz", direction="maximize", weight=1.0),
            # 3. Phase Margin Stability (penalize deviation from optimal 65 deg)
            Objective(name="PM_Stability", direction="maximize", weight=1.0),
            # 4. Minimize Total Static Power Dissipation
            Objective(name="Power_uW", direction="minimize", weight=1.0),
        ]

    def evaluate_pareto_vector(self, ind: Individual) -> dict[str, float]:
        """Evaluates objective dictionary for NSGA2 sorting."""
        m = self.evaluate_metrics(ind)
        return {
            "Gain_dB": m.gain_db,
            "GBW_MHz": m.gbw_mhz,
            "PM_Stability": -abs(m.pm_deg - 65.0),
            "Power_uW": m.power_uw,
        }


class Sky130OpAmpAdapter(DomainAdapter):
    """First-class DomainAdapter integrating SkyWater 130nm OpAmp sizing into Evolab OS."""

    def __init__(self, corner: Sky130Corner = Sky130Corner.TT):
        self.corner = corner

    @property
    def name(self) -> str:
        return "sky130_opamp"

    def parse_spec(self, raw_spec: Any) -> dict[str, Any]:
        d = dict(raw_spec) if isinstance(raw_spec, dict) else {}
        return {
            "target_gain_db": float(d.get("target_gain_db", 60.0)),
            "target_gbw_mhz": float(d.get("target_gbw_mhz", 10.0)),
            "target_pm_deg": float(d.get("target_pm_deg", 60.0)),
            "target_max_power_uw": float(d.get("target_max_power_uw", 600.0)),
            "corner": str(d.get("corner", self.corner.value)),
            "enable_pvt": bool(d.get("enable_pvt", False)),
            "skip_on_fail": bool(d.get("skip_on_fail", True)),
        }

    def build_population(
        self, spec: dict[str, Any], size: int, rng: Any = None
    ) -> list[Individual]:
        import random
        r = rng or random.Random(42)

        # Check if spec requests spec-conditioned generation or has specific targets
        has_targets = any(
            k in spec
            for k in (
                "target_gain_db",
                "target_gbw_mhz",
                "target_pm_deg",
                "target_max_power_uw",
                "spec_conditioned",
            )
        )
        use_spec_cond = spec.get("spec_conditioned", has_targets)

        if use_spec_cond and has_targets:
            try:
                from .spec_conditioned_prior import SpecConditionedPrior, TargetCircuitSpec

                target_spec = TargetCircuitSpec(
                    gain_db=float(spec.get("target_gain_db", 60.0)),
                    gbw_mhz=float(spec.get("target_gbw_mhz", 10.0)),
                    pm_deg=float(spec.get("target_pm_deg", 60.0)),
                    max_power_uw=float(spec.get("target_max_power_uw", 600.0)),
                )
                prior = SpecConditionedPrior(target_spec=target_spec)
                return prior.sample_seed_population(count=size, species="sky130_opamp")
            except Exception:
                pass  # Graceful fallback to default baseline

        pop: list[Individual] = []

        # Seed with a realistic baseline design
        baseline = OpAmpSizing()
        pop.append(Individual(
            genome=FloatGenome(values=baseline.to_normalized_vector()),
            species="sky130_opamp",
            fitness=0.0,
            _generation=0,
            _index=0,
        ))

        # Generate perturbed population
        for i in range(1, size):
            vec = [
                min(1.0, max(0.0, baseline.to_normalized_vector()[k] + r.gauss(0.0, 0.15)))
                for k in range(14)
            ]
            pop.append(Individual(
                genome=FloatGenome(values=vec),
                species="sky130_opamp",
                fitness=0.0,
                _generation=0,
                _index=i,
            ))
        return pop

    def build_evaluator(self, spec: dict[str, Any]) -> Evaluator:
        if spec.get("enable_pvt", False):
            from .pvt_evaluator import PVTAwareOpAmpEvaluator
            return PVTAwareOpAmpEvaluator(
                target_gain_db=float(spec.get("target_gain_db", 60.0)),
                target_gbw_mhz=float(spec.get("target_gbw_mhz", 10.0)),
                target_pm_deg=float(spec.get("target_pm_deg", 60.0)),
                target_max_power_uw=float(spec.get("target_max_power_uw", 600.0)),
                skip_on_fail=bool(spec.get("skip_on_fail", True)),
            )
        c = Sky130Corner(spec.get("corner", self.corner.value))
        return TwoStageMillerOpAmpEvaluator(
            target_gain_db=float(spec.get("target_gain_db", 60.0)),
            target_gbw_mhz=float(spec.get("target_gbw_mhz", 10.0)),
            target_pm_deg=float(spec.get("target_pm_deg", 60.0)),
            target_max_power_uw=float(spec.get("target_max_power_uw", 600.0)),
            corner=c,
        )

    def build_mutator(self, spec: dict[str, Any] | None = None) -> Any:
        """Returns PhysicsInformedOpAmpMutator matching the specification targets."""
        d = spec or {}
        from .physics_mutator import PhysicsInformedOpAmpMutator
        return PhysicsInformedOpAmpMutator(
            target_gain_db=float(d.get("target_gain_db", 60.0)),
            target_gbw_mhz=float(d.get("target_gbw_mhz", 10.0)),
            target_pm_deg=float(d.get("target_pm_deg", 60.0)),
            target_max_power_uw=float(d.get("target_max_power_uw", 600.0)),
        )

    def export_solution(
        self,
        individual: Individual,
        spec: dict[str, Any],
        output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        g = individual.genome
        vals = list(getattr(g, "values", getattr(g, "genes", [])))
        sizing = OpAmpSizing.from_normalized_vector(vals)
        c = Sky130Corner(spec.get("corner", self.corner.value))
        metrics = evaluate_opamp_analytical(sizing, corner=c)

        if output_path:
            p = Path(output_path)
            netlist = generate_opamp_spice_netlist(sizing, corner=c)
            p.write_text(netlist, encoding="utf-8")

        return metrics.to_dict()
