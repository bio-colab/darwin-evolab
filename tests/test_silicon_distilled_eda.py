"""tests/test_silicon_distilled_eda.py — Test suite for distilled EDA techniques.

Covers:
1. PPAAS: PVTAwareOpAmpEvaluator with Skip-on-Fail acceleration & Conservative Hindsight Replay.
2. AnalogCoder-Pro: PhysicsInformedOpAmpMutator with closed-loop diagnostic repair.
3. CircuitGenome: ModularOpAmpCircuit block representation & topology mutation.
"""
from __future__ import annotations

import random
import pytest

from evolab.genome import FloatGenome, Individual
from evolab.silicon import (
    ActiveLoadType,
    AnalyticalSpecInverter,
    CKTBENCH_BALANCED,
    CompensationType,
    ConservativeHindsightReplay,
    DiffPairType,
    ModularOpAmpCircuit,
    OpAmpPerformanceMetrics,
    OpAmpSizing,
    PVTAwareOpAmpEvaluator,
    PhysicsInformedOpAmpMutator,
    Sky130Corner,
    evaluate_opamp_analytical,
)


def test_pvt_evaluator_all_corners():
    evaluator = PVTAwareOpAmpEvaluator(
        target_gain_db=60.0,
        target_gbw_mhz=10.0,
        target_pm_deg=60.0,
        target_max_power_uw=600.0,
        corners=[Sky130Corner.TT, Sky130Corner.SS, Sky130Corner.FF],
        skip_on_fail=True,
    )

    # Valid balanced design
    balanced = AnalyticalSpecInverter().invert(CKTBENCH_BALANCED)
    res = evaluator.evaluate_multi_corner(balanced)

    assert res.passed_nominal is True
    assert res.skipped_corners is False
    assert len(res.corner_metrics) == 3
    assert Sky130Corner.TT in res.corner_metrics
    assert Sky130Corner.SS in res.corner_metrics
    assert Sky130Corner.FF in res.corner_metrics
    assert res.worst_gain_db > 40.0
    assert res.worst_pm_deg > 50.0
    assert res.execution_time_savings_pct == 0.0

    ind = Individual(genome=FloatGenome(values=balanced.to_normalized_vector()), species="sky130_opamp")
    fit = evaluator.evaluate(ind)
    assert fit.score > 50.0
    assert "multi_corner" in fit.artifacts

    pareto_vec = evaluator.evaluate_pareto_vector(ind)
    assert "Worst_Gain_dB" in pareto_vec
    assert "Max_Power_uW" in pareto_vec


def test_pvt_evaluator_skip_on_fail():
    evaluator = PVTAwareOpAmpEvaluator(
        target_gain_db=60.0,
        target_gbw_mhz=10.0,
        target_pm_deg=60.0,
        target_max_power_uw=600.0,
        corners=[Sky130Corner.TT, Sky130Corner.SS, Sky130Corner.FF],
        skip_on_fail=True,
        min_nominal_pm_deg=15.0,
    )

    # Unstable sizing with collapsed phase margin (Cc=0.2pF, W6=5um -> PM=0 deg)
    poor_sizing = OpAmpSizing(cc_pf=0.2, w6_um=5.0)
    res = evaluator.evaluate_multi_corner(poor_sizing)

    assert res.passed_nominal is False
    assert res.skipped_corners is True
    assert len(res.corner_metrics) == 1
    assert Sky130Corner.SS not in res.corner_metrics
    assert Sky130Corner.FF not in res.corner_metrics
    assert res.execution_time_savings_pct > 60.0

    ind = Individual(genome=FloatGenome(values=poor_sizing.to_normalized_vector()), species="sky130_opamp")
    fit = evaluator.evaluate(ind)
    assert fit.passed_holdout is False
    assert "skip_on_fail" in fit.artifacts["note"]


def test_conservative_hindsight_replay():
    replay = ConservativeHindsightReplay(max_capacity=10)

    sizing1 = AnalyticalSpecInverter().invert(CKTBENCH_BALANCED)
    m1 = evaluate_opamp_analytical(sizing1)
    added1 = replay.add(sizing1, m1, generation=0)
    assert added1 is True
    assert len(replay) == 1

    # Candidate with strictly worse metrics across all dimensions is rejected
    m_worse = OpAmpPerformanceMetrics(
        gain_db=m1.gain_db - 10.0,
        gbw_mhz=m1.gbw_mhz - 5.0,
        pm_deg=m1.pm_deg - 10.0,
        power_uw=m1.power_uw + 200.0,
        cmrr_db=40.0,
        slew_rate_v_us=5.0,
        is_stable=True,
        meets_spec=False,
        physical_claim=True,
    )
    added_worse = replay.add(sizing1, m_worse, generation=1)
    assert added_worse is False
    assert len(replay) == 1

    # Candidate with trade-off (higher gain, though slightly higher power) is preserved
    m_tradeoff = OpAmpPerformanceMetrics(
        gain_db=m1.gain_db + 12.0,
        gbw_mhz=m1.gbw_mhz - 1.0,
        pm_deg=m1.pm_deg,
        power_uw=m1.power_uw + 50.0,
        cmrr_db=50.0,
        slew_rate_v_us=8.0,
        is_stable=True,
        meets_spec=False,
        physical_claim=True,
    )
    added_tradeoff = replay.add(sizing1, m_tradeoff, generation=2)
    assert added_tradeoff is True
    assert len(replay) == 2

    seeds = replay.sample_seeds(count=2)
    assert len(seeds) == 2
    assert isinstance(seeds[0], OpAmpSizing)


def test_physics_informed_mutator():
    mutator = PhysicsInformedOpAmpMutator(
        target_gain_db=65.0,
        target_gbw_mhz=12.0,
        target_pm_deg=65.0,
    )

    # Diagnose instability
    poor_pm_metrics = OpAmpPerformanceMetrics(
        gain_db=62.0,
        gbw_mhz=15.0,
        pm_deg=35.0,  # Below 45 deg -> critical instability
        power_uw=300.0,
        cmrr_db=60.0,
        slew_rate_v_us=15.0,
        is_stable=False,
        meets_spec=False,
        physical_claim=True,
    )
    diag = mutator.diagnose(poor_pm_metrics)
    assert diag.pm_deficit is True
    assert diag.dominant_symptom == "critical_instability"

    # Mutate sizing to repair instability
    rng = random.Random(42)
    base_s = OpAmpSizing(cc_pf=1.0, w6_um=30.0)
    mutated_s = mutator.mutate_sizing(base_s, metrics=poor_pm_metrics, rng=rng)

    # Miller cap and driver width must have increased to restore phase margin
    assert mutated_s.cc_pf > base_s.cc_pf
    assert mutated_s.w6_um > base_s.w6_um

    # Mutate individual directly
    ind = Individual(genome=FloatGenome(values=base_s.to_normalized_vector()), species="sky130_opamp")
    new_ind = mutator.mutate_individual(ind, rng=rng)
    assert isinstance(new_ind.genome, FloatGenome)
    assert new_ind.genome.values != ind.genome.values


def test_modular_opamp_circuit():
    sizing = OpAmpSizing(
        w1_um=12.0,
        l1_um=0.5,
        w3_um=25.0,
        l3_um=0.5,
        w6_um=50.0,
        cc_pf=3.0,
    )

    # 1. From sizing to modular circuit
    mod_ckt = ModularOpAmpCircuit.from_sizing(sizing)
    assert mod_ckt.diff_pair.w_um == 12.0
    assert mod_ckt.diff_pair.topology == DiffPairType.NMOS_PAIR
    assert mod_ckt.compensation.cc_pf == 3.0
    assert mod_ckt.compensation.topology == CompensationType.MILLER_CAP

    # 2. Back to sizing (roundtrip)
    recovered_sizing = mod_ckt.to_sizing()
    assert recovered_sizing.w1_um == sizing.w1_um
    assert recovered_sizing.w6_um == sizing.w6_um
    assert recovered_sizing.cc_pf == sizing.cc_pf

    # 3. Topology mutation
    rng = random.Random(123)
    mod_ckt.mutate_topology(rng=rng)
    assert mod_ckt.compensation.topology in [CompensationType.MILLER_CAP, CompensationType.MILLER_RC]

    # 4. Generate modular SPICE netlist
    netlist = mod_ckt.generate_spice_netlist(Sky130Corner.TT)
    assert "* Modular CircuitGenome CMOS OpAmp" in netlist
    assert "XM1 d1 inp tail" in netlist
    assert "XM6 out d2 vdd" in netlist
    assert ".meas ac max_gain" in netlist
