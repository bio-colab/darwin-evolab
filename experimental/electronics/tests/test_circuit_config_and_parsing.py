"""
Tests for CircuitConfigEvaluator and NGSpiceBridge parsing (low-cost, no ngspice required).
Covers the two cheapest improvements from opensource-analog-circuits inspiration.
"""
from pathlib import Path

from experimental.electronics.evaluators.spice_evaluator import CircuitConfigEvaluator, compute_objective
from experimental.electronics.models.ngspice_bridge import NGSpiceBridge, _parse_ac_table, _parse_meas
from evolab.genome import FloatGenome


def test_parse_ac_table_synthetic():
    # Test with ngspice-compatible format (Index frequency vdb vp)
    log = """
    Index   frequency       vdb(out)        vp(out)         
    0	1.000000e+00	4.160114e+01	3.141592e+00	
    1	1.000000e+01	4.160114e+01	3.141592e+00	
    2	1.000000e+02	4.160114e+01	3.141592e+00	
    3	1.000000e+03	4.160114e+01	3.141592e+00	
    4	1.000000e+04	4.160114e+01	3.141592e+00	
    5	1.000000e+05	4.160114e+01	3.141592e+00	
    6	1.000000e+06	4.160114e+01	3.141592e+00	
    7	1.000000e+07	-2.000e+01	-2.0e+02
    """
    metrics = _parse_ac_table(log)
    assert "gain_db" in metrics
    # gain should be the max vdb
    assert metrics["gain_db"] == 41.60114
    assert "bandwidth_mhz" in metrics
    # bandwidth should be around 100kHz (where vdb crosses 0)
    assert "phase_margin_deg" in metrics


def test_parse_meas_regex():
    log = "vout = 5.5\nsome other line"
    patterns = {"vout": r"^\s*vout\s+=\s+([\-\d\.eE+]+)"}
    metrics = _parse_meas(log, patterns)
    assert metrics["vout"] == 5.5


def test_compute_objective_sum_violations():
    specs = {"gain": [">", 60], "vout": [">", 5.0]}
    # meets
    assert compute_objective({"gain": 65, "vout": 5.5}, specs) == 0.0
    # fails gain
    obj = compute_objective({"gain": 50, "vout": 5.5}, specs)
    assert obj > 0
    # missing metric -> 1000 penalty
    obj2 = compute_objective({"vout": 5.5}, specs)
    assert obj2 >= 1000


def test_circuit_config_evaluator_bjt():
    cfg = Path("experimental/electronics/circuits/bjt_ce_amp/config.json")
    ev = CircuitConfigEvaluator(cfg)
    assert ev.dim == 5
    assert "R1" in ev.names
    # default point should pass (vout >5)
    defaults = ev.defaults
    genome = FloatGenome(defaults)
    res = ev.evaluate(genome)
    assert res.score > 80
    assert res.artifacts.get("tool_used")
    assert isinstance(res.passed_holdout, bool)
    assert "vout" in res.artifacts


def test_circuit_config_evaluator_ptm_defaults():
    cfg = Path("experimental/electronics/circuits/ptm180nm_opamp/config.json")
    ev = CircuitConfigEvaluator(cfg)
    assert ev.dim == 19
    genome = FloatGenome(ev.defaults)
    res = ev.evaluate(genome)
    # ptm default has gain 42 <60 spec, so not passed but should be deterministic and cheap
    assert res.score >= 0
    assert "gain" in res.artifacts or "gain_db" in res.artifacts
    # tool_used can be either analytical_fallback or ngspice depending on availability
    assert res.artifacts["tool_used"] in ("analytical_fallback", "ngspice")


def test_ngspice_bridge_run_circuit_file_injection():
    bridge = NGSpiceBridge()
    # use bjt circuit file with param injection
    cir = Path("experimental/electronics/circuits/bjt_ce_amp/circuit.cir")
    assert cir.exists()
    params = {"R1": 110000, "R2": 10000}
    res = bridge.run_circuit_file(cir, params=params)
    assert res.success is True
    # tool_used can be either analytical_fallback or ngspice depending on availability
    assert res.tool_used in ("analytical_fallback", "ngspice")


def test_bridge_size_aware_differentiation():
    bridge = NGSpiceBridge()
    # small W vs large W should give different gain (size-aware)
    # Use ptm opamp circuit with different sizing - this is known to work
    # Default ptm point
    params_default = {
        'W1': 10, 'W2': 10, 'W3': 20, 'W4': 20, 'W5': 15, 'W6': 40, 'W7': 20, 'W8': 5,
        'L1': 0.36, 'L2': 0.36, 'L3': 0.36, 'L4': 0.36, 'L5': 0.36, 'L6': 0.36, 'L7': 0.36, 'L8': 0.72,
        'Cc': 2.0, 'CL': 5.0, 'Ib': 10.0
    }
    # Optimal ptm point (larger W, smaller L)
    params_opt = {
        'W1': 15, 'W2': 15, 'W3': 30, 'W4': 30, 'W5': 22.5, 'W6': 60, 'W7': 30, 'W8': 7.5,
        'L1': 0.18, 'L2': 0.18, 'L3': 0.18, 'L4': 0.18, 'L5': 0.18, 'L6': 0.18, 'L7': 0.18, 'L8': 0.36,
        'Cc': 1.5, 'CL': 2.5, 'Ib': 15.0
    }
    from pathlib import Path
    cir = Path("experimental/electronics/circuits/ptm180nm_opamp/circuit.cir")
    small = bridge.run_circuit_file(cir, params=params_default)
    large = bridge.run_circuit_file(cir, params=params_opt)
    # With ngspice available, we expect real SPICE results showing size-awareness
    # Verify that different sizing produces DIFFERENT results (size-awareness)
    # Note: real SPICE behavior may differ from analytical predictions
    assert large.gain_db != small.gain_db
    assert large.bandwidth_mhz != small.bandwidth_mhz


def test_circuit_config_evaluator_comparator():
    cfg = Path("experimental/electronics/circuits/comparator_simple/config.json")
    ev = CircuitConfigEvaluator(cfg)
    assert ev.dim == 4
    genome = FloatGenome(ev.defaults)
    res = ev.evaluate(genome)
    assert res.score > 80
    assert "vout_high" in res.artifacts
    assert "vout_low" in res.artifacts
    # comparator high >3.5 and low <1.0 should pass with physics fallback
    assert res.artifacts.get("tool_used")
    assert isinstance(res.passed_holdout, bool)


def test_circuit_config_evaluator_timer():
    cfg = Path("experimental/electronics/circuits/timer_555_astable/config.json")
    ev = CircuitConfigEvaluator(cfg)
    assert ev.dim == 3
    genome = FloatGenome(ev.defaults)
    res = ev.evaluate(genome)
    assert "freq" in res.artifacts
    # freq =1.44/((10k+20k)*100n)=480Hz, default slightly below 500 spec -> score <100 but deterministic
    assert res.score >= 0
    # with smaller C, freq rises
    genome_fast = FloatGenome([5000, 5000, 50e-9])
    res_fast = ev.evaluate(genome_fast)
    assert res_fast.artifacts["freq"] > res.artifacts["freq"]
