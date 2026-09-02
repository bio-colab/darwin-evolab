from experimental.electronics.instruments.oscilloscope import measure_transient
from experimental.electronics.models.ngspice_bridge import NGSpiceBridge, parse_tran_table


TRAN_LOG = """
Index   time            v(6)            v(out)
0	0.000000e+00	0.000000e+00	5.000000e+00
1	2.500000e-04	0.000000e+00	5.000000e+00
2	5.000000e-04	5.000000e+00	0.000000e+00
3	7.500000e-04	5.000000e+00	0.000000e+00
4	1.000000e+00	0.000000e+00	5.000000e+00
"""
# time step in rows 0-3 is 0.25ms; last row is messy on purpose — parser should keep numeric rows only


def test_parse_tran_table_and_scope():
    # uniform 1 kHz-ish square on v(out): 5,5,0,0,5 at 0.25ms until we only use first 4? 
    log = """
Index   time            v(out)
0	0.000000e+00	5.000000e+00
1	5.000000e-04	5.000000e+00
2	1.000000e-03	0.000000e+00
3	1.500000e-03	0.000000e+00
4	2.000000e-03	5.000000e+00
5	2.500000e-03	5.000000e+00
6	3.000000e-03	0.000000e+00
7	3.500000e-03	0.000000e+00
"""
    art = parse_tran_table(log)
    assert art.success is True
    assert len(art.t) == 8
    assert "v(out)" in art.signals
    m = measure_transient(art)
    assert m.get("error") is None
    assert abs(m["vpp"] - 5.0) < 1e-6
    assert m["signal"] == "v(out)"


def test_run_transient_without_ngspice_does_not_invent_wave():
    art = NGSpiceBridge().run_transient_file(
        "experimental/electronics/circuits/timer_555_astable/circuit.cir"
    )
    if art.tool_used == "none":
        assert art.t == ()
        assert art.signals == {}
        m = measure_transient(art)
        assert m["error"] == "no_waveform"
    else:
        assert art.success is True
        assert len(art.t) > 2
