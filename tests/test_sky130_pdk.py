"""Unit test suite for SkyWater 130nm PDK models and device physics equations."""
from __future__ import annotations

import math
import pytest

from evolab.silicon.sky130_pdk import (
    CORNER_SPECS,
    SKY130_PARAMS,
    Sky130Corner,
    compute_transistor_operating_point,
    generate_sky130_spice_header,
)


def test_sky130_parameters_and_invariants():
    p = SKY130_PARAMS
    assert p.l_min_um == 0.15
    assert p.w_min_um == 0.42
    assert p.vdd_nominal == 1.80
    assert p.vth0_n > 0.0
    assert p.vth0_p < 0.0
    assert p.cox > 0.0
    assert p.mu_n0 > p.mu_p0  # Electron mobility > Hole mobility in silicon


def test_sky130_pvt_corners():
    for c in Sky130Corner:
        assert c in CORNER_SPECS
        spec = CORNER_SPECS[c]
        assert spec.vdd > 0.0
        assert spec.mobility_scale > 0.0

    # Slow-Slow corner: lower VDD, high temp, lower mobility
    ss = CORNER_SPECS[Sky130Corner.SS]
    assert ss.temp_c == 125.0
    assert ss.vdd < SKY130_PARAMS.vdd_nominal
    assert ss.mobility_scale < 1.0

    # Fast-Fast corner: higher VDD, cold temp, higher mobility
    ff = CORNER_SPECS[Sky130Corner.FF]
    assert ff.temp_c == -40.0
    assert ff.vdd > SKY130_PARAMS.vdd_nominal
    assert ff.mobility_scale > 1.0


def test_compute_transistor_operating_point_nmos():
    # Sized NMOS with 10uA current
    op = compute_transistor_operating_point(
        w_um=10.0,
        l_um=0.36,
        id_target_a=10e-6,
        vds_v=0.9,
        is_pmos=False,
        corner=Sky130Corner.TT,
    )
    assert not op.is_pmos
    assert op.w_um == 10.0
    assert op.l_um == 0.36
    assert op.id_a == 10e-6
    assert op.v_overdrive > 0.0
    assert op.gm_s > 0.0
    assert op.ro_ohm > 0.0
    # ro approx 1 / (lambda * Id)
    assert op.ro_ohm > 100e3  # Greater than 100k ohms


def test_compute_transistor_operating_point_pmos():
    op = compute_transistor_operating_point(
        w_um=20.0,
        l_um=0.36,
        id_target_a=10e-6,
        vds_v=0.9,
        is_pmos=True,
        corner=Sky130Corner.TT,
    )
    assert op.is_pmos
    assert op.gm_s > 0.0
    assert op.ro_ohm > 0.0


def test_generate_sky130_spice_header():
    header = generate_sky130_spice_header(corner=Sky130Corner.TT)
    assert "sky130_fd_pr__nfet_01v8" in header
    assert "sky130_fd_pr__pfet_01v8" in header
    assert ".param sky130_vdd=1.8" in header
