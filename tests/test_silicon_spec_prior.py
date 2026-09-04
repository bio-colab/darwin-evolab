"""test_silicon_spec_prior.py — Tests for Production Spec2Ckt & ARCS Silicon Hardening.

Verifies:
1. PhysicalGrammarGuard CMOS design rules and stability enforcement.
2. SpecConditionedPrior zero-shot candidate seeding.
3. BanditLatentOptimizer test-time refinement.
4. Sky130OpAmpAdapter automatic spec-conditioned population generation.
5. High-level synthesis via experimental.electronics.spec2ckt.
"""
from __future__ import annotations

import pytest

from evolab.silicon import (
    BENCHMARK_SPECS,
    CKTBENCH_BALANCED,
    CKTBENCH_HIGH_GAIN,
    CKTBENCH_HIGH_SPEED,
    CKTBENCH_LOW_POWER,
    AnalyticalSpecInverter,
    BanditLatentOptimizer,
    GrammarCheckResult,
    OpAmpSizing,
    PhysicalGrammarGuard,
    Sky130Corner,
    Sky130OpAmpAdapter,
    SpecConditionedGenerator,
    SpecConditionedPrior,
    SpecTolerance,
    TargetCircuitSpec,
    evaluate_opamp_analytical,
)
from experimental.electronics.spec2ckt import synthesize_spec_conditioned_opamp


def test_production_grammar_guard_aspect_ratios_and_repairs():
    guard = PhysicalGrammarGuard()

    # Create unphysical transistor geometries
    unphysical = OpAmpSizing(
        w1_um=0.2,   # Under min width
        l1_um=2.5,   # Inverse aspect ratio W/L < 1
        w5_um=1.0,
        l5_um=2.0,
        w8_um=25.0,  # Tail mirror starvation
        l8_um=0.36,
        cc_pf=0.05,  # Zero compensation
        ibias_ua=0.5,
    )

    check = guard.validate(unphysical)
    assert check.is_valid is False
    assert len(check.violations) >= 2

    # Project to physically realizable manifold
    repaired = guard.repair_and_project(unphysical)
    check_repaired = guard.validate(repaired)
    assert check_repaired.is_valid is True
    assert repaired.w1_um >= 1.0
    assert repaired.cc_pf >= 0.5


def test_production_spec_conditioned_prior_diversity():
    target = CKTBENCH_BALANCED
    prior = SpecConditionedPrior(target_spec=target)

    # Sample population for Darwin kernel
    inds = prior.sample_seed_population(count=12, species="sky130_opamp")
    assert len(inds) == 12
    assert inds[0].species == "sky130_opamp"
    # Seed population should already possess high fitness at gen 0
    assert inds[0].fitness >= 0.70


def test_production_bandit_latent_optimizer():
    bandit = BanditLatentOptimizer(seed=42)
    target = CKTBENCH_LOW_POWER
    gen = SpecConditionedGenerator(seed=42)

    initial_sizings = gen.sample_conditioned_population(target, count=6)
    refined = bandit.optimize_seeds(initial_sizings, target_spec=target, budget_trials=10)
    assert len(refined) == 6


def test_sky130_adapter_build_population_spec_conditioned():
    adapter = Sky130OpAmpAdapter()

    # 1. With target specifications -> triggers spec-conditioned prior
    spec_with_targets = {
        "target_gain_db": 70.0,
        "target_gbw_mhz": 15.0,
        "target_pm_deg": 60.0,
        "target_max_power_uw": 700.0,
    }
    pop_warm = adapter.build_population(spec_with_targets, size=8)
    assert len(pop_warm) == 8
    # High fitness initial individuals
    assert pop_warm[0].fitness >= 0.60

    # 2. Without target specifications -> 100% backward compatible baseline
    pop_cold = adapter.build_population({}, size=8)
    assert len(pop_cold) == 8
    assert pop_cold[0].fitness == 0.0  # Default initial fitness


def test_electronics_spec2ckt_high_level_api():
    result = synthesize_spec_conditioned_opamp(
        target_gain_db=65.0,
        target_gbw_mhz=12.0,
        target_pm_deg=60.0,
        max_power_uw=600.0,
    )

    assert result["satisfaction"]["all_satisfied"] is True
    assert result["pvt_passes"] >= 4
    assert ".subckt" in result["spice_netlist"] or "M1" in result["spice_netlist"]
    assert result["achieved_metrics"]["gain_db"] >= 60.0
    assert result["achieved_metrics"]["is_stable"] is True
