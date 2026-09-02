"""Regression: the behavioral 555 benchmark circuit must actually oscillate.

History: astable_behavioral.cir shipped with (a) comparator pull-ups wired to
node 0 so the SR latch could never flip, (b) Qdisch with base/emitter swapped
so the timing cap never discharged, and (c) a .print flood (~161 MB stdout per
eval) that made every evaluation time out at score 0.0. The circuit was also
measured through the wrong observable (first printed column = cap node) and
ngspice's paginated .print output corrupted the parsed waveform.

This test pins the whole chain: circuit -> ngspice -> parse -> scope, using
the datasheet astable formula as the independent oracle.
"""
from experimental.electronics.evaluators.tran_evaluator import BEHAVIORAL_CIR
from experimental.electronics.instruments.oscilloscope import measure_transient
from experimental.electronics.models.ngspice_bridge import NGSpiceBridge


def test_behavioral_555_oscillates_near_datasheet_formula():
    bridge = NGSpiceBridge()
    if not bridge.is_ngspice_available():
        # No simulator: nothing may be claimed, but the circuit file must at
        # least carry the repaired topology markers.
        text = BEHAVIORAL_CIR.read_text(encoding="utf-8", errors="ignore")
        assert "Qdisch 7 discharge_ctrl 0" in text  # correct BJT pin order
        assert "Bcmp1" in text and "Bcmp2" in text  # continuous comparators
        return

    r1, r2, c1 = 10000.0, 10000.0, 1e-7
    art = bridge.run_transient_file(
        BEHAVIORAL_CIR,
        {"R1": r1, "R2": r2, "C1": c1, "VCC": 5.0},
        timeout_sec=60.0,
    )
    assert art.success is True
    assert art.tool_used == "ngspice"

    meas = measure_transient(art, signal="v(out)")
    assert meas.get("error") is None
    assert meas["cycles_used"] >= 10            # real oscillation, not a blip
    assert meas["frequency_confidence"] >= 0.5  # stable period estimate
    assert meas["vpp"] > 3.0                    # rail-to-rail-ish output swing

    expected = 1.44 / ((r1 + 2.0 * r2) * c1)    # datasheet astable formula: 480 Hz
    err = abs(meas["frequency_hz"] - expected) / expected
    assert err < 0.10, f"measured {meas['frequency_hz']:.1f} Hz vs formula {expected:.1f} Hz"

    # Charge phase (R1+R2) must dominate discharge (R2): duty > 0.55.
    assert meas["duty_cycle"] > 0.55
