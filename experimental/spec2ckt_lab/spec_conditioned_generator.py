"""spec_conditioned_generator.py — CktGen-style Spec-Conditioned Prior for Analog Circuits.

Translates high-level circuit specifications directly into physically-grounded
transistor sizing candidates without cold-start evolutionary trial and error.

Combines:
1. Analytical CMOS physics inversion (closed-form small-signal equations).
2. Latent manifold perturbation for generating diverse populations of valid seeds.
3. Native integration with Darwin's FoundationModelPrior and EvolutionEngine.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from evolab.genome import FloatGenome, Individual
from evolab.silicon.opamp_benchmark import OpAmpSizing, evaluate_opamp_analytical
from evolab.silicon.sky130_pdk import SKY130_PARAMS, Sky130Corner

from .grammar_guard import PhysicalGrammarGuard
from .spec_types import TargetCircuitSpec


class AnalyticalSpecInverter:
    """Inverts SkyWater 130nm CMOS small-signal equations to compute baseline transistor sizing."""

    def __init__(self, pmos_mobility_ratio: float = 0.35):
        # Sky130 nominal parameters
        self.un_cox = SKY130_PARAMS.mu_n0 * SKY130_PARAMS.cox
        self.up_cox = SKY130_PARAMS.mu_p0 * SKY130_PARAMS.cox
        self.vdd = SKY130_PARAMS.vdd_nominal
        self.guard = PhysicalGrammarGuard()

    def invert(self, spec: TargetCircuitSpec) -> OpAmpSizing:
        """Computes deterministic central sizing that meets target specifications."""
        # 1. Total current budget from max power
        i_total_max = (spec.max_power_uw * 1e-6) / self.vdd
        # Distribute currents: Tail ~ 40%, Stage2 driver ~ 50%, Bias branch ~ 10%
        i_tail = max(2.0e-6, min(0.40 * i_total_max, 60.0e-6))
        i_stage2 = max(3.0e-6, min(0.50 * i_total_max, 80.0e-6))
        i_bias = max(1.0e-6, min(0.10 * i_total_max, 20.0e-6))
        i1 = i_tail / 2.0

        # 2. Compensation capacitor Cc
        # Choose Cc between 1.5pF and 3.5pF for optimal phase margin stability
        sr_target = max(spec.min_slew_rate_v_us * 1e6, 1.0e6)
        cc_sr = i_tail / sr_target
        cc_f = max(1.5e-12, min(max(cc_sr, 1.5e-12), 3.5e-12))
        cc_pf = cc_f * 1e12

        # 3. Input pair transconductance gm1
        gbw_hz = spec.gbw_mhz * 1e6
        gm1_req = 2.0 * math.pi * gbw_hz * cc_f
        w1_over_l1 = (gm1_req ** 2) / (2.0 * self.un_cox * max(i1, 1e-7))
        w1_over_l1 = max(2.0, min(w1_over_l1, 150.0))

        # Channel lengths scaled to support Gain target
        gain_lin = 10.0 ** (spec.gain_db / 20.0)
        l_scale = max(0.18, min(0.18 * (gain_lin / 300.0) ** 0.5, 1.0))
        l1 = round(max(0.18, min(l_scale, 0.8)), 3)
        w1 = round(max(1.0, min(w1_over_l1 * l1, 50.0)), 3)

        # 4. Active load M3, M4 (PMOS)
        w3_over_l3 = (gm1_req ** 2) / (2.0 * self.up_cox * max(i1, 1e-7))
        w3_over_l3 = max(2.0, min(w3_over_l3, 150.0))
        l3 = round(max(0.18, min(l_scale, 0.8)), 3)
        w3 = round(max(2.0, min(w3_over_l3 * l3, 60.0)), 3)

        # 5. Output stage driver M6 (PMOS) & sink M7 (NMOS)
        # To guarantee Phase Margin >= spec.pm_deg, second pole p2 = gm6 / (2*pi*CL)
        # must satisfy p2 >= 3.5 * GBW (for PM > 60 deg)
        cl_f = 5.0e-12
        p2_factor = 3.5 if spec.pm_deg >= 55.0 else 2.5
        gm6_req = p2_factor * (2.0 * math.pi * gbw_hz * cl_f)
        w6_over_l6 = (gm6_req ** 2) / (2.0 * self.up_cox * max(i_stage2, 1e-7))
        w6_over_l6 = max(15.0, min(w6_over_l6, 300.0))
        l6 = round(max(0.18, min(l_scale * 0.7, 0.5)), 3)
        w6 = round(max(10.0, min(w6_over_l6 * l6, 120.0)), 3)

        # 6. Current mirrors M5, M7, M8
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

        # Pass through ARCS PhysicalGrammarGuard to enforce hard PDK bounds
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
            # Perturb in normalized [0, 1] parameter space
            perturbed_vec = []
            for val in center_norm:
                noise = self.rng.gauss(0.0, perturbation_sigma)
                perturbed_val = max(0.0, min(val + noise, 1.0))
                perturbed_vec.append(perturbed_val)

            # Decode to sizing and project to physically valid manifold
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
            # Sample more if needed
            extra = self.generator.sample_conditioned_population(
                self.target_spec, count=count - len(sizings)
            )
            sizings = sizings + extra

        for i, sizing in enumerate(sizings[:count]):
            norm_vec = sizing.to_normalized_vector()
            genome = FloatGenome(values=norm_vec)
            # Pre-evaluate fitness under the target spec
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
