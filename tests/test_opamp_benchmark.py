"""Unit test suite for SkyWater 130nm Two-Stage Miller OpAmp benchmark and adapter."""
from __future__ import annotations

import pytest

from evolab.genome import FloatGenome, Individual
from evolab.silicon.opamp_benchmark import (
    OpAmpPerformanceMetrics,
    OpAmpSizing,
    Sky130OpAmpAdapter,
    TwoStageMillerOpAmpEvaluator,
    evaluate_opamp_analytical,
    generate_opamp_spice_netlist,
)
from evolab.silicon.sky130_pdk import Sky130Corner


def test_opamp_sizing_roundtrip_vector():
    s1 = OpAmpSizing(w1_um=15.5, l1_um=0.5, cc_pf=3.5, ibias_ua=25.0)
    vec = s1.to_normalized_vector()
    assert len(vec) == 14
    for x in vec:
        assert 0.0 <= x <= 1.0

    s2 = OpAmpSizing.from_normalized_vector(vec)
    assert abs(s2.w1_um - s1.w1_um) < 0.1
    assert abs(s2.l1_um - s1.l1_um) < 0.05
    assert abs(s2.cc_pf - s1.cc_pf) < 0.1
    assert abs(s2.ibias_ua - s1.ibias_ua) < 0.1


def test_evaluate_opamp_analytical_metrics():
    sizing = OpAmpSizing(
        w1_um=15.0, l1_um=0.36,
        w3_um=30.0, l3_um=0.36,
        w5_um=20.0, l5_um=0.36,
        w6_um=60.0, l6_um=0.36,
        w7_um=30.0, l7_um=0.36,
        w8_um=5.0,  l8_um=0.72,
        cc_pf=2.5,  ibias_ua=15.0,
    )
    metrics = evaluate_opamp_analytical(sizing, corner=Sky130Corner.TT)

    assert isinstance(metrics, OpAmpPerformanceMetrics)
    assert metrics.gain_db > 40.0  # Real two-stage CMOS amp achieves 50-80dB
    assert metrics.gbw_mhz > 1.0   # MHz range
    assert metrics.pm_deg > 10.0   # Stability degrees
    assert metrics.power_uw > 0.0  # MicroWatts
    assert metrics.physical_claim is True
    assert "i_tail_ua" in metrics.artifacts


def test_generate_opamp_spice_netlist():
    sizing = OpAmpSizing()
    netlist = generate_opamp_spice_netlist(sizing, corner=Sky130Corner.TT)

    assert "XM1 d1 inp tail vss sky130_fd_pr__nfet_01v8" in netlist
    assert "XM6 out d2 vdd vdd sky130_fd_pr__pfet_01v8" in netlist
    assert ".ac dec 10 1 10G" in netlist
    assert ".meas ac max_gain" in netlist


def test_two_stage_opamp_evaluator_and_pareto():
    evaluator = TwoStageMillerOpAmpEvaluator(
        target_gain_db=60.0, target_gbw_mhz=10.0, target_pm_deg=60.0, target_max_power_uw=600.0
    )
    sizing = OpAmpSizing()
    ind = Individual(
        genome=FloatGenome(values=sizing.to_normalized_vector()),
        species="opamp",
        fitness=0.0,
        _generation=0,
        _index=0,
    )
    res = evaluator.evaluate(ind)
    assert 0.0 <= res.score <= 100.0
    assert "metrics" in res.artifacts

    # Pareto objectives
    objectives = evaluator.build_pareto_objectives()
    assert len(objectives) == 4
    obj_names = [o.name for o in objectives]
    assert "Gain_dB" in obj_names
    assert "GBW_MHz" in obj_names
    assert "PM_Stability" in obj_names
    assert "Power_uW" in obj_names

    # Pareto vector
    scores = evaluator.evaluate_pareto_vector(ind)
    assert "Gain_dB" in scores
    assert "Power_uW" in scores


def test_sky130_opamp_adapter_workflow():
    adapter = Sky130OpAmpAdapter()
    assert adapter.name == "sky130_opamp"

    spec = adapter.parse_spec({})
    assert spec["target_gain_db"] == 60.0

    pop = adapter.build_population(spec, size=6)
    assert len(pop) == 6
    assert all(isinstance(ind.genome, FloatGenome) for ind in pop)

    ev = adapter.build_evaluator(spec)
    for ind in pop:
        fit = ev.evaluate(ind)
        assert fit.score > 0.0

    best = pop[0]
    exported = adapter.export_solution(best, spec)
    assert isinstance(exported, dict)
    assert "gain_db" in exported
