"""
Tests for 74HC IC component datasheet specifications.
"""
from experimental.electronics.components.catalog_74xx import CATALOG_74XX, get_ic_spec


def test_catalog_completeness():
    expected_ics = ["74HC00", "74HC02", "74HC04", "74HC08", "74HC32", "74HC86"]
    for ic in expected_ics:
        assert ic in CATALOG_74XX
        spec = get_ic_spec(ic)
        assert spec.pin_count == 14
        assert spec.vcc_pin == 14
        assert spec.gnd_pin == 7
        assert spec.electrical.icc_quiescent_max_ua == 20.0


def test_published_timing_corners_not_derated():
    spec = get_ic_spec("74HC00")
    timing = spec.timing
    assert timing.get_delay_ns(4.5, 25.0) == timing.tpd_max_ns_at_4_5v_25c
    assert timing.get_delay_ns(4.5, 85.0) == timing.tpd_max_ns_at_4_5v_industrial
    assert timing.get_delay_ns(6.0, 25.0) == timing.tpd_max_ns_at_6_0v_25c
    assert timing.get_delay_ns(2.0, 25.0) == timing.tpd_max_ns_at_2_0v_25c
    assert timing.get_delay_ns(4.5, 85.0) > timing.get_delay_ns(4.5, 25.0) > timing.get_delay_ns(6.0, 25.0)
    light = timing.get_delay_ns(4.5, 25.0, cl_pf=15.0)
    heavy = timing.get_delay_ns(4.5, 25.0, cl_pf=50.0)
    assert light < heavy
    assert timing.get_transition_ns(4.5, 25.0, 15.0) < timing.get_transition_ns(4.5, 25.0, 50.0)


def test_datasheet_constraint_verifier_pass_fail_unknown():
    from experimental.electronics.components.catalog_74xx import get_ic_spec
    from experimental.electronics.models.datasheet_constraints import DatasheetConstraintVerifier

    elec = get_ic_spec("74HC00").electrical
    v = DatasheetConstraintVerifier.from_electrical(elec)
    assert v.check({"vmax": 4.82})["verdict"] in ("PASS", "UNKNOWN")
    assert any(c["name"] == "voh_min" and c["verdict"] == "PASS" for c in v.check({"vmax": 4.82})["checks"])
    assert any(c["name"] == "voh_min" and c["verdict"] == "FAIL" for c in v.check({"vmax": 3.0})["checks"])
    assert v.check({})["verdict"] == "UNKNOWN"
    assert v.check({"vmax": 3.0})["verdict"] == "FAIL"
