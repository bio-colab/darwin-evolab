from experimental.electronics.instruments.oscilloscope import attach_scope, measure_waveform


def test_ideal_scope_on_1khz_square():
    fs = 100_000.0
    period = 0.001
    n = int(fs * 0.02)
    high = int(fs * period / 2)
    v = [5.0 if (i % int(fs * period)) < high else 0.0 for i in range(n)]
    m = measure_waveform(v, fs)
    assert m["mode"] == "ideal"
    assert abs(m["frequency_hz"] - 1000.0) < 5.0
    assert abs(m["vpp"] - 5.0) < 1e-6
    assert abs(m["duty_cycle"] - 0.5) < 0.02
    assert m["zero_crossings"] >= 18


def test_attach_scope_only_when_waveform_present():
    assert "scope" not in attach_scope({"gain_db": 40})
    out = attach_scope({"waveform": [0.0, 1.0, 0.0, 1.0], "sample_rate_hz": 4.0})
    assert "scope" in out
    assert out["gain_db"] if False else out["scope"]["n"] == 4
