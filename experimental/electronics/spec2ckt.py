"""spec2ckt.py — High-Level Spec2Ckt Synthesis API for experimental/electronics.

Bridges the spec-conditioned generative analog design and physical grammar guard
directly into the electronics track.
"""
from __future__ import annotations

from typing import Any

from evolab.silicon.grammar_guard import PhysicalGrammarGuard
from evolab.silicon.opamp_benchmark import (
    OpAmpPerformanceMetrics,
    OpAmpSizing,
    evaluate_opamp_analytical,
    generate_opamp_spice_netlist,
)
from evolab.silicon.sky130_pdk import Sky130Corner
from evolab.silicon.spec_conditioned_prior import (
    AnalyticalSpecInverter,
    SpecConditionedGenerator,
    TargetCircuitSpec,
)


def synthesize_spec_conditioned_opamp(
    target_gain_db: float = 65.0,
    target_gbw_mhz: float = 12.0,
    target_pm_deg: float = 60.0,
    max_power_uw: float = 600.0,
    min_slew_rate_v_us: float = 10.0,
    corner: Sky130Corner = Sky130Corner.TT,
) -> dict[str, Any]:
    """Directly synthesizes an analog CMOS OpAmp matching performance specifications."""
    spec = TargetCircuitSpec(
        name="direct_spec2ckt",
        gain_db=target_gain_db,
        gbw_mhz=target_gbw_mhz,
        pm_deg=target_pm_deg,
        max_power_uw=max_power_uw,
        min_slew_rate_v_us=min_slew_rate_v_us,
    )
    generator = SpecConditionedGenerator()
    sizing = generator.generate_center_candidate(spec)
    metrics = evaluate_opamp_analytical(sizing, corner=corner)
    netlist = generate_opamp_spice_netlist(sizing, corner=corner)
    sat = spec.evaluate_satisfaction(metrics)

    # PVT 5-Corner Verification
    pvt_evals = {}
    pvt_passes = 0
    for c in Sky130Corner:
        m_c = evaluate_opamp_analytical(sizing, corner=c)
        pvt_evals[c.value] = m_c.to_dict()
        if spec.evaluate_satisfaction(m_c)["all_satisfied"]:
            pvt_passes += 1

    return {
        "target_spec": {
            "gain_db": target_gain_db,
            "gbw_mhz": target_gbw_mhz,
            "pm_deg": target_pm_deg,
            "max_power_uw": max_power_uw,
        },
        "achieved_metrics": metrics.to_dict(),
        "satisfaction": sat,
        "pvt_passes": pvt_passes,
        "pvt_total": len(Sky130Corner),
        "pvt_pass_rate": f"{pvt_passes / len(Sky130Corner) * 100:.0f}%",
        "pvt_corners": pvt_evals,
        "sizing": sizing.__dict__,
        "spice_netlist": netlist,
    }
