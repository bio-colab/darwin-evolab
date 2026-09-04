"""
Unit tests for FPGA hardware target presets, static resource estimation, and multi-architecture EDA packaging.
"""
from __future__ import annotations

import random
from pathlib import Path
import pytest

from evolab.cgp_logic import (
    GateType,
    CGPNode,
    CGPGenome,
    FPGABoardSpec,
    FPGA_PRESETS,
    FPGAResourceReport,
    estimate_fpga_resources,
    EDAPackager,
    create_random_cgp_genome,
)


def test_fpga_presets_catalog():
    """Validates existence and invariants of built-in FPGA board presets."""
    expected_presets = {"ice40_hx1k", "ice40_up5k", "ecp5_25k", "artix7_35t"}
    assert expected_presets.issubset(set(FPGA_PRESETS.keys()))

    for name, spec in FPGA_PRESETS.items():
        assert spec.preset_name == name
        assert spec.total_luts > 0
        assert spec.total_ios > 0
        assert spec.core_voltage > 0.5
        assert spec.lut_input_size in (4, 6)
        assert spec.typical_gate_delay_ns > 0.0


def test_estimate_fpga_resources_half_adder():
    """Verifies static synthesis estimation for a 2-input, 2-output half adder."""
    nodes = [
        CGPNode(GateType.XOR, 0, 1),  # Sum
        CGPNode(GateType.AND, 0, 1),  # Carry
    ]
    genome = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])

    # 1. Test Lattice iCE40-HX1K
    report_ice40 = estimate_fpga_resources(genome, "ice40_hx1k")
    assert isinstance(report_ice40, FPGAResourceReport)
    assert report_ice40.target_preset == "ice40_hx1k"
    assert report_ice40.vendor == "Lattice"
    assert report_ice40.estimated_luts >= 1
    assert report_ice40.total_pins_used == 4  # 2 in + 2 out
    assert report_ice40.fits_on_target is True
    assert 0.0 < report_ice40.lut_utilization_pct < 1.0  # tiny fraction of 1280
    assert report_ice40.estimated_fmax_mhz > 100.0
    assert report_ice40.estimated_dynamic_power_uw > 0.0

    # 2. Test AMD/Xilinx Artix-7 35T (6-input LUTs)
    report_artix = estimate_fpga_resources(genome, "artix7_35t")
    assert report_artix.target_preset == "artix7_35t"
    assert report_artix.vendor == "AMD/Xilinx"
    assert report_artix.fits_on_target is True
    assert report_artix.estimated_delay_ns < report_ice40.estimated_delay_ns  # Artix-7 is faster cell delay


def test_estimate_fpga_resources_large_random_circuit():
    """Verifies estimation on a complex multi-stage DAG with inactive pruning."""
    rng = random.Random(42)
    genome = create_random_cgp_genome(num_inputs=4, num_outputs=3, num_nodes=25, rng=rng)

    report = estimate_fpga_resources(genome, "ecp5_25k")
    assert report.total_luts == 24192
    assert report.total_pins_used == 7  # 4 in + 3 out
    assert report.critical_path_depth >= 1
    assert report.fits_on_target is True


def test_eda_packager_multi_target_constraints(tmp_path: Path):
    """Verifies constraint generation across Lattice iCE40, Lattice ECP5, and AMD Artix-7."""
    nodes = [
        CGPNode(GateType.XOR, 0, 1),
        CGPNode(GateType.AND, 0, 1),
    ]
    genome = CGPGenome(num_inputs=2, num_outputs=2, nodes=nodes, output_connections=[2, 3])
    verilog = genome.to_verilog("top_half_adder")

    # 1. iCE40 -> .pcf
    pack_ice40 = EDAPackager(target_fpga="ice40_up5k")
    bundle_ice40 = pack_ice40.package_bundle(
        verilog_code=verilog,
        top_module="top_half_adder",
        num_inputs=2,
        num_outputs=2,
        output_dir=tmp_path / "ice40",
        run_synthesis_if_available=False,
        genome=genome,
    )
    assert bundle_ice40.constraints_file.endswith(".pcf")
    pcf_content = Path(bundle_ice40.constraints_file).read_text(encoding="utf-8")
    assert "set_io in[0]" in pcf_content
    assert "synth_ice40" in Path(bundle_ice40.yosys_script_file).read_text(encoding="utf-8")
    assert bundle_ice40.resource_estimate is not None
    assert bundle_ice40.resource_estimate.target_preset == "ice40_up5k"

    # 2. ECP5 -> .lpf
    pack_ecp5 = EDAPackager(target_fpga="ecp5_25k")
    bundle_ecp5 = pack_ecp5.package_bundle(
        verilog_code=verilog,
        top_module="top_half_adder",
        num_inputs=2,
        num_outputs=2,
        output_dir=tmp_path / "ecp5",
        run_synthesis_if_available=False,
        genome=genome,
    )
    assert bundle_ecp5.constraints_file.endswith(".lpf")
    lpf_content = Path(bundle_ecp5.constraints_file).read_text(encoding="utf-8")
    assert 'LOCATE COMP "in[0]"' in lpf_content
    assert "synth_ecp5" in Path(bundle_ecp5.yosys_script_file).read_text(encoding="utf-8")

    # 3. Artix-7 -> .xdc
    pack_artix = EDAPackager(target_fpga="artix7_35t")
    bundle_artix = pack_artix.package_bundle(
        verilog_code=verilog,
        top_module="top_half_adder",
        num_inputs=2,
        num_outputs=2,
        output_dir=tmp_path / "artix",
        run_synthesis_if_available=False,
        genome=genome,
    )
    assert bundle_artix.constraints_file.endswith(".xdc")
    xdc_content = Path(bundle_artix.constraints_file).read_text(encoding="utf-8")
    assert "set_property -dict { PACKAGE_PIN" in xdc_content
    assert "synth_xilinx" in Path(bundle_artix.yosys_script_file).read_text(encoding="utf-8")
