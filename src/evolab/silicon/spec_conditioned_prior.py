"""spec_conditioned_prior.py — Spec-Conditioned Generative Prior for Analog Circuits.

Promoted to production core: Translates high-level circuit specifications directly
into physically realizable transistor sizing seeds using closed-form small-signal
inversion, latent manifold perturbation, and ARCS grammar guards.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from evolab.genome import FloatGenome, Individual

from .grammar_guard import PhysicalGrammarGuard
from .opamp_benchmark import OpAmpPerformanceMetrics, OpAmpSizing, evaluate_opamp_analytical
from .sky130_pdk import SKY130_PARAMS, Sky130Corner


@dataclass
class SpecTolerance:
    """Allowed tolerance margins for target specifications."""
    gain_rel_tol: float = 0.05    # 5% relative tolerance
    gbw_rel_tol: float = 0.10     # 10% relative tolerance
    pm_abs_deg_tol: float = 5.0   # 5 degrees tolerance
    power_rel_slack: float = 0.10 # Allow up to 10% above max power
    sr_rel_slack: float = 0.10    # Allow up to 10% below min slew rate


@dataclass
class TargetCircuitSpec:
    """Target performance requirements for analog circuit synthesis."""
    name: str = "custom_opamp"
    gain_db: float = 60.0          # Target low-frequency gain (dB)
    gbw_mhz: float = 10.0          # Target Gain-Bandwidth Product (MHz)
    pm_deg: float = 60.0           # Target Phase Margin (degrees)
    max_power_uw: float = 600.0    # Maximum allowable power dissipation (uW)
    min_slew_rate_v_us: float = 8.0  # Minimum Slew Rate (V/us)
    min_cmrr_db: float = 50.0      # Minimum CMRR (dB)

    def to_spec_vector(self) -> list[float]:
        """Returns normalized target vector [gain, gbw, pm, power, sr]."""
        return [
            self.gain_db / 100.0,
            self.gbw_mhz / 50.0,
            self.pm_deg / 90.0,
            self.max_power_uw / 2000.0,
            self.min_slew_rate_v_us / 50.0,
        ]

    def evaluate_satisfaction(
        self,
        metrics: OpAmpPerformanceMetrics,
        tol: SpecTolerance | None = None,
    ) -> dict[str, Any]:
        """Evaluates whether the circuit metrics satisfy all specifications."""
        tol = tol or SpecTolerance()

        gain_ok = metrics.gain_db >= self.gain_db * (1.0 - tol.gain_rel_tol)
        gbw_ok = metrics.gbw_mhz >= self.gbw_mhz * (1.0 - tol.gbw_rel_tol)
        pm_ok = metrics.pm_deg >= (self.pm_deg - tol.pm_abs_deg_tol)
        power_ok = metrics.power_uw <= self.max_power_uw * (1.0 + tol.power_rel_slack)
        sr_ok = metrics.slew_rate_v_us >= self.min_slew_rate_v_us * (1.0 - tol.sr_rel_slack)
        stable_ok = metrics.is_stable

        all_ok = all([gain_ok, gbw_ok, pm_ok, power_ok, sr_ok, stable_ok])

        score_gain = min(1.0, metrics.gain_db / max(self.gain_db, 1.0))
        score_gbw = min(1.0, metrics.gbw_mhz / max(self.gbw_mhz, 0.1))
        score_pm = min(1.0, metrics.pm_deg / max(self.pm_deg, 1.0)) if metrics.is_stable else 0.2
        score_pwr = min(1.0, self.max_power_uw / max(metrics.power_uw, 1.0))
        score_sr = min(1.0, metrics.slew_rate_v_us / max(self.min_slew_rate_v_us, 0.1))

        composite_score = 0.25 * score_gain + 0.25 * score_gbw + 0.20 * score_pm + 0.15 * score_pwr + 0.15 * score_sr
        if not metrics.is_stable:
            composite_score *= 0.5

        return {
            "all_satisfied": all_ok,
            "composite_score": round(composite_score, 4),
            "gain_ok": gain_ok,
            "gbw_ok": gbw_ok,
            "pm_ok": pm_ok,
            "power_ok": power_ok,
            "sr_ok": sr_ok,
            "stable_ok": stable_ok,
            "metrics": metrics.to_dict(),
        }

    def compute_fitness(self, metrics: OpAmpPerformanceMetrics) -> float:
        """Computes evolutionary scalar fitness from performance metrics against this spec."""
        res = self.evaluate_satisfaction(metrics)
        score = float(res["composite_score"])
        if res["all_satisfied"]:
            score += 0.5
        return score


# CktBench industry benchmark targets
CKTBENCH_LOW_POWER = TargetCircuitSpec(
    name="CktBench_LowPower",
    gain_db=55.0,
    gbw_mhz=6.0,
    pm_deg=65.0,
    max_power_uw=300.0,
    min_slew_rate_v_us=5.0,
    min_cmrr_db=50.0,
)

CKTBENCH_BALANCED = TargetCircuitSpec(
    name="CktBench_Balanced",
    gain_db=65.0,
    gbw_mhz=12.0,
    pm_deg=60.0,
    max_power_uw=600.0,
    min_slew_rate_v_us=10.0,
    min_cmrr_db=55.0,
)

CKTBENCH_HIGH_GAIN = TargetCircuitSpec(
    name="CktBench_HighGain",
    gain_db=75.0,
    gbw_mhz=15.0,
    pm_deg=55.0,
    max_power_uw=850.0,
    min_slew_rate_v_us=12.0,
    min_cmrr_db=60.0,
)

CKTBENCH_HIGH_SPEED = TargetCircuitSpec(
    name="CktBench_HighSpeed",
    gain_db=60.0,
    gbw_mhz=22.0,
    pm_deg=52.0,
    max_power_uw=1100.0,
    min_slew_rate_v_us=18.0,
    min_cmrr_db=50.0,
)

BENCHMARK_SPECS: dict[str, TargetCircuitSpec] = {
    "low_power": CKTBENCH_LOW_POWER,
    "balanced": CKTBENCH_BALANCED,
    "high_gain": CKTBENCH_HIGH_GAIN,
    "high_speed": CKTBENCH_HIGH_SPEED,
}


class AnalyticalSpecInverter:
    """Inverts SkyWater 130nm CMOS small-signal equations to compute baseline transistor sizing."""

    def __init__(self, pmos_mobility_ratio: float = 0.35):
        self.un_cox = SKY130_PARAMS.mu_n0 * SKY130_PARAMS.cox
        self.up_cox = SKY130_PARAMS.mu_p0 * SKY130_PARAMS.cox
        self.vdd = SKY130_PARAMS.vdd_nominal
        self.guard = PhysicalGrammarGuard()

    def invert(self, spec: TargetCircuitSpec) -> OpAmpSizing:
        """Computes deterministic central sizing that meets target specifications."""
        i_total_max = (spec.max_power_uw * 1e-6) / self.vdd
        i_tail = max(2.0e-6, min(0.40 * i_total_max, 60.0e-6))
        i_stage2 = max(3.0e-6, min(0.50 * i_total_max, 80.0e-6))
        i_bias = max(1.0e-6, min(0.10 * i_total_max, 20.0e-6))
        i1 = i_tail / 2.0

        sr_target = max(spec.min_slew_rate_v_us * 1e6, 1.0e6)
        cc_sr = i_tail / sr_target
        cc_f = max(1.5e-12, min(max(cc_sr, 1.5e-12), 3.5e-12))
        cc_pf = cc_f * 1e12

        gbw_hz = spec.gbw_mhz * 1e6
        gm1_req = 2.0 * math.pi * gbw_hz * cc_f
        w1_over_l1 = (gm1_req ** 2) / (2.0 * self.un_cox * max(i1, 1e-7))
        w1_over_l1 = max(2.0, min(w1_over_l1, 150.0))

        gain_lin = 10.0 ** (spec.gain_db / 20.0)
        l_scale = max(0.18, min(0.18 * (gain_lin / 300.0) ** 0.5, 1.0))
        l1 = round(max(0.18, min(l_scale, 0.8)), 3)
        w1 = round(max(1.0, min(w1_over_l1 * l1, 50.0)), 3)

        w3_over_l3 = (gm1_req ** 2) / (2.0 * self.up_cox * max(i1, 1e-7))
        w3_over_l3 = max(2.0, min(w3_over_l3, 150.0))
        l3 = round(max(0.18, min(l_scale, 0.8)), 3)
        w3 = round(max(2.0, min(w3_over_l3 * l3, 60.0)), 3)

        cl_f = 5.0e-12
        p2_factor = 3.5 if spec.pm_deg >= 55.0 else 2.5
        gm6_req = p2_factor * (2.0 * math.pi * gbw_hz * cl_f)
        w6_over_l6 = (gm6_req ** 2) / (2.0 * self.up_cox * max(i_stage2, 1e-7))
        w6_over_l6 = max(15.0, min(w6_over_l6, 300.0))
        l6 = round(max(0.18, min(l_scale * 0.7, 0.5)), 3)
        w6 = round(max(10.0, min(w6_over_l6 * l6, 120.0)), 3)

        l_mirrors = round(max(0.36, min(l_scale * 1.2, 1.0)), 3)
        w8 = round(max(2.0, min(4.0 * (i_bias / 10e-6), 15.0)), 3)
        w8_over_l8 = w8 / l_mirrors

        w5_over_l5 = w8_over_l8 * (i_tail / max(i_bias, 1e-7))
        w5 = round(max(2.0, min(w5_over_l5 * l_mirrors, 60.0)), 3)

        w7_over_l7 = w8_over_l8 * (i_stage2 / max(i_bias, 1e-7))
        w7 = round(max(2.0, min(w7_over_l7 * l_mirrors, 80.0)), 3)

        raw_sizing = OpAmpSizing(
            w1_um=w1,
            l1_um=l1,
            w3_um=w3,
            l3_um=l3,
            w5_um=w5,
            l5_um=l_mirrors,
            w6_um=w6,
            l6_um=l6,
            w7_um=w7,
            l7_um=l_mirrors,
            w8_um=w8,
            l8_um=l_mirrors,
            cc_pf=round(cc_pf, 3),
            ibias_ua=round(i_bias * 1e6, 2),
            cl_pf=5.0,
        )
        return self.guard.repair_and_project(raw_sizing)


class SpecConditionedGenerator:
    """Generates warm, diverse population distributions conditioned on target performance specifications."""

    def __init__(self, seed: int = 42):
        self.inverter = AnalyticalSpecInverter()
        self.guard = PhysicalGrammarGuard()
        self.rng = random.Random(seed)

    def generate_center_candidate(self, spec: TargetCircuitSpec) -> OpAmpSizing:
        """Returns the optimal central sizing candidate for the given spec."""
        return self.inverter.invert(spec)

    def sample_conditioned_population(
        self,
        spec: TargetCircuitSpec,
        count: int = 30,
        perturbation_sigma: float = 0.08,
    ) -> list[OpAmpSizing]:
        """Samples a diverse population of candidates clustered around the target manifold."""
        center_sizing = self.generate_center_candidate(spec)
        center_norm = center_sizing.to_normalized_vector()

        candidates: list[OpAmpSizing] = [center_sizing]

        for _ in range(count - 1):
            perturbed_vec = []
            for val in center_norm:
                noise = self.rng.gauss(0.0, perturbation_sigma)
                perturbed_vec.append(max(0.0, min(val + noise, 1.0)))

            raw = OpAmpSizing.from_normalized_vector(perturbed_vec)
            repaired = self.guard.repair_and_project(raw)
            candidates.append(repaired)

        return candidates


@dataclass
class SpecConditionedPrior:
    """Plugs into Darwin's FoundationModelPrior for seamless kernel seeding."""
    target_spec: TargetCircuitSpec
    generator: SpecConditionedGenerator = field(default_factory=SpecConditionedGenerator)
    suggested_sizings: list[OpAmpSizing] = field(default_factory=list)

    def __post_init__(self):
        if not self.suggested_sizings:
            self.suggested_sizings = self.generator.sample_conditioned_population(
                self.target_spec, count=30
            )

    def sample_seed_population(
        self,
        count: int,
        species: str = "spec_conditioned_prior",
    ) -> list[Individual]:
        """Produces Darwin Individual instances containing normalized FloatGenomes."""
        inds = []
        sizings = self.suggested_sizings
        if len(sizings) < count:
            extra = self.generator.sample_conditioned_population(
                self.target_spec, count=count - len(sizings)
            )
            sizings = sizings + extra

        for i, sizing in enumerate(sizings[:count]):
            norm_vec = sizing.to_normalized_vector()
            genome = FloatGenome(values=norm_vec)
            metrics = evaluate_opamp_analytical(sizing, Sky130Corner.TT)
            fitness = self.target_spec.compute_fitness(metrics)
            ind = Individual(
                genome=genome,
                species=species,
                fitness=fitness,
                _generation=0,
                _index=i,
            )
            inds.append(ind)

        return inds
