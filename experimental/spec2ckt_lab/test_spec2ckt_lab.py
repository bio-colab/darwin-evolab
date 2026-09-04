"""test_spec2ckt_lab.py — Test Suite for Spec-Conditioned Generative Evolution Lab.

Verifies:
1. Physical grammar guards and projection repairs (ARCS distillation).
2. Spec-conditioned prior generation and analytical inversion (CktGen distillation).
3. Test-time bandit latent optimization.
4. Hybrid evolutionary convergence and PVT corner verification.
"""
from __future__ import annotations

import pytest

from evolab.silicon.opamp_benchmark import (
    OpAmpPerformanceMetrics,
    OpAmpSizing,
    evaluate_opamp_analytical,
)
from evolab.silicon.sky130_pdk import Sky130Corner

from experimental.spec2ckt_lab.bandit_latent_optimizer import BanditLatentOptimizer
from experimental.spec2ckt_lab.grammar_guard import GrammarCheckResult, PhysicalGrammarGuard
from experimental.spec2ckt_lab.hybrid_evolution_engine import (
    HybridSpecEvolutionEngine,
    OptimizationResult,
)
from experimental.spec2ckt_lab.spec_conditioned_generator import (
    AnalyticalSpecInverter,
    SpecConditionedGenerator,
    SpecConditionedPrior,
)
from experimental.spec2ckt_lab.spec_types import (
    CKTBENCH_BALANCED,
    CKTBENCH_LOW_POWER,
    SpecTolerance,
    TargetCircuitSpec,
)


def test_spec_types_and_scoring():
    spec = TargetCircuitSpec(
        gain_db=60.0,
        gbw_mhz=10.0,
        pm_deg=60.0,
        max_power_uw=600.0,
        min_slew_rate_v_us=8.0,
    )
    # Good metrics
    good_m = OpAmpPerformanceMetrics(
        gain_db=65.0,
        gbw_mhz=12.0,
        pm_deg=62.0,
        power_uw=450.0,
        cmrr_db=70.0,
        slew_rate_v_us=10.0,
        is_stable=True,
        meets_spec=True,
        physical_claim=True,
    )
    res = spec.evaluate_satisfaction(good_m)
    assert res["all_satisfied"] is True
    assert res["composite_score"] >= 0.90
    assert spec.compute_fitness(good_m) > 1.0

    # Failing metrics
    bad_m = OpAmpPerformanceMetrics(
        gain_db=35.0,
        gbw_mhz=2.0,
        pm_deg=30.0,
        power_uw=1200.0,
        cmrr_db=40.0,
        slew_rate_v_us=2.0,
        is_stable=False,
        meets_spec=False,
        physical_claim=True,
    )
    bad_res = spec.evaluate_satisfaction(bad_m)
    assert bad_res["all_satisfied"] is False
    assert bad_res["composite_score"] < 0.5


def test_physical_grammar_guard_violations_and_repair():
    guard = PhysicalGrammarGuard()

    # Invalid sizing: tiny W/L aspect ratio, tiny tail, inverted mirror
    unphysical = OpAmpSizing(
        w1_um=0.2,   # Under min width
        l1_um=3.0,   # Large L, W/L << 1.0
        w5_um=1.0,
        l5_um=2.0,
        w8_um=20.0,  # Tail mirror starvation
        l8_um=0.36,
        cc_pf=0.01,  # Compensation cap collapsed
        ibias_ua=0.1,# Sub-microamp starvation
    )

    check = guard.validate(unphysical)
    assert check.is_valid is False
    assert len(check.violations) >= 2

    # Repair and project
    repaired = guard.repair_and_project(unphysical)
    check_repaired = guard.validate(repaired)
    assert check_repaired.is_valid is True
    assert repaired.w1_um >= 1.0
    assert repaired.cc_pf >= 0.5
    assert repaired.ibias_ua >= 2.0


def test_analytical_spec_inverter():
    inverter = AnalyticalSpecInverter()
    target = CKTBENCH_BALANCED  # 65 dB, 12 MHz, 60 deg, 600 uW

    sizing = inverter.invert(target)
    assert isinstance(sizing, OpAmpSizing)

    # Evaluate physics
    metrics = evaluate_opamp_analytical(sizing, Sky130Corner.TT)
    assert metrics.is_stable is True
    assert metrics.pm_deg >= 50.0
    assert metrics.gain_db >= 55.0
    assert metrics.power_uw <= 800.0


def test_spec_conditioned_generator_and_prior():
    gen = SpecConditionedGenerator(seed=123)
    target = CKTBENCH_LOW_POWER

    pop = gen.sample_conditioned_population(target, count=10)
    assert len(pop) == 10

    # Ensure diversity among candidates
    w1_vals = [cand.w1_um for cand in pop]
    assert len(set(w1_vals)) > 1

    # Check prior integration with Darwin Individual
    prior = SpecConditionedPrior(target_spec=target, generator=gen)
    inds = prior.sample_seed_population(count=8)
    assert len(inds) == 8
    assert inds[0].species == "spec_conditioned_prior"
    assert inds[0].fitness > 0.6  # High initial fitness at generation 0!


def test_bandit_latent_optimizer():
    bandit = BanditLatentOptimizer(seed=42)
    target = CKTBENCH_BALANCED
    gen = SpecConditionedGenerator(seed=42)

    initial_pop = gen.sample_conditioned_population(target, count=8)
    initial_fits = [
        target.compute_fitness(evaluate_opamp_analytical(s, Sky130Corner.TT))
        for s in initial_pop
    ]
    initial_best = max(initial_fits)

    refined_pop = bandit.optimize_seeds(initial_pop, target_spec=target, budget_trials=15)
    assert len(refined_pop) == len(initial_pop)

    refined_fits = [
        target.compute_fitness(evaluate_opamp_analytical(s, Sky130Corner.TT))
        for s in refined_pop
    ]
    refined_best = max(refined_fits)

    # Bandit should maintain or improve the best fitness
    assert refined_best >= initial_best


def test_hybrid_evolution_engine_convergence_and_pvt():
    engine = HybridSpecEvolutionEngine(use_surrogate=True, seed=42)
    target = CKTBENCH_LOW_POWER

    # Run Spec2Ckt Generative Darwin
    res_spec = engine.run_spec2ckt_pipeline(
        target, pop_size=12, max_generations=6, bandit_budget=10
    )

    assert isinstance(res_spec, OptimizationResult)
    assert res_spec.strategy_name == "Spec2Ckt_Generative_Darwin"
    assert res_spec.nominal_metrics.is_stable is True
    assert res_spec.nominal_metrics.pm_deg >= 50.0
    assert res_spec.converged_generation is not None
    # Converged very early (zero-shot or within 3 generations)
    assert res_spec.converged_generation <= 3

    # PVT signoff should verify all 5 corners
    assert len(res_spec.pvt_metrics) == 5
    assert res_spec.pvt_pass_count >= 3  # High PVT yield


def test_cold_start_vs_spec2ckt_head_to_head():
    """Head-to-head comparison proving the distilled advantage of Spec2Ckt over Cold-Start."""
    engine = HybridSpecEvolutionEngine(use_surrogate=True, seed=99)
    target = CKTBENCH_BALANCED

    # Run cold-start baseline (random unconditioned start)
    res_cold = engine.run_cold_start_baseline(target, pop_size=16, max_generations=10)

    # Run spec-conditioned generative evolution
    res_warm = engine.run_spec2ckt_pipeline(
        target, pop_size=16, max_generations=10, bandit_budget=8
    )

    # 1. Spec-conditioned Darwin should converge faster or have higher fitness at gen 0
    gen0_cold_fitness = res_cold.history[0]["best_fitness"]
    gen0_warm_fitness = res_warm.history[0]["best_fitness"]
    assert gen0_warm_fitness > gen0_cold_fitness

    # 2. Spec-conditioned Darwin should reach satisfaction in early generations
    assert res_warm.converged_generation is not None
    assert res_warm.converged_generation <= 3

    # 3. Final nominal satisfaction
    assert res_warm.all_specs_satisfied is True
    assert res_warm.nominal_metrics.is_stable is True


def test_all_cktbench_targets_inversion():
    """Verifies that all 4 standardized CktBench targets produce physically stable sizings."""
    inverter = AnalyticalSpecInverter()
    from experimental.spec2ckt_lab.spec_types import BENCHMARK_SPECS

    for name, spec in BENCHMARK_SPECS.items():
        sizing = inverter.invert(spec)
        metrics = evaluate_opamp_analytical(sizing, Sky130Corner.TT)
        assert metrics.is_stable is True, f"Failed stability on {name}"
        assert metrics.pm_deg >= 45.0, f"Low PM on {name}: {metrics.pm_deg}"
        assert metrics.gain_db >= 50.0, f"Low Gain on {name}: {metrics.gain_db}"
