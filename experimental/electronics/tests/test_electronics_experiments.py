"""
Tests verifying execution of laboratory experiments 1 and 2.
"""
from experimental.electronics.experiments.exp1_datasheet_74xx_synthesis import run_half_adder_synthesis_lab
from experimental.electronics.experiments.exp2_analog_sizing_ngspice import run_comparative_analog_experiment


def test_exp1_half_adder_lab():
    rep = run_half_adder_synthesis_lab()
    assert rep["experiment"] == "Exp1_Datasheet_74xx_Half_Adder_Synthesis"
    assert rep["passed"] is True
    assert rep["fitness_score"] > 80.0
    assert len(rep["ic_packages_used"]) == 2
    assert "layer_1_datasheet_specs" in rep["verification_layers"]


def test_exp2_comparative_analog_lab():
    rep = run_comparative_analog_experiment()
    assert rep["experiment"] == "Exp2_Comparative_Analog_Sizing_Benchmark"
    assert len(rep["strategies"]) == 3
    for s in rep["strategies"]:
        assert s["best_fitness"] > 80.0
