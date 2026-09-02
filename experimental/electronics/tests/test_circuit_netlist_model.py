"""
Tests for BreadboardCircuit netlist model and CircuitNetlistGenome contract.
"""
import random
from experimental.electronics.models.circuit_netlist import (
    BreadboardCircuit,
    CircuitNetlistGenome,
    Connection,
    PinRef,
)


def test_breadboard_circuit_simulation():
    # Construct a simple inverter circuit using 74HC04:
    # Pin -1, 0 (Input) -> Pin 1 (IC 0)
    # Pin 2 (IC 0) -> Pin -1, 100 (Output)
    conns = [
        Connection(PinRef(-1, 0), PinRef(0, 1)),
        Connection(PinRef(0, 2), PinRef(-1, 100)),
    ]
    circuit = BreadboardCircuit(
        ic_packages=["74HC04"],
        connections=conns,
        num_inputs=1,
        num_outputs=1,
    )

    out0, stable0 = circuit.simulate([0])
    assert stable0 is True
    assert out0 == [1]

    out1, stable1 = circuit.simulate([1])
    assert stable1 is True
    assert out1 == [0]


def test_circuit_netlist_genome_contract():
    conns = [
        Connection(PinRef(-1, 0), PinRef(0, 1)),
        Connection(PinRef(0, 2), PinRef(-1, 100)),
    ]
    genome = CircuitNetlistGenome(
        ic_packages=["74HC04"],
        connections=conns,
        num_inputs=1,
        num_outputs=1,
    )

    clone = genome.clone()
    assert clone.fingerprint() == genome.fingerprint()
    assert clone.distance_to(genome) == 0.0

    desc = genome.describe()
    assert desc["ic_count"] == 1
    assert desc["wire_count"] == 2
    assert desc["quiescent_ua"] == 20.0

    ser = genome.serialize()
    assert ser["type"] == "CircuitNetlistGenome"
    assert len(ser["connections"]) == 2

    # Mutation test
    rng = random.Random(42)
    mutated = genome.mutate(rng=rng)
    assert isinstance(mutated, CircuitNetlistGenome)


def test_bridge_half_adder_population_seeds_one_known_solution():
    from experimental.electronics.bridge import prepare_electronics_run
    from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef

    ev, pop, name = prepare_electronics_run("half_adder", 8, seed=0)
    assert name == "half_adder"
    known = {
        (c.source.ic_index, c.source.pin, c.destination.ic_index, c.destination.pin)
        for c in [
            Connection(PinRef(-1, 0), PinRef(0, 1)),
            Connection(PinRef(-1, 1), PinRef(0, 2)),
            Connection(PinRef(0, 3), PinRef(-1, 100)),
            Connection(PinRef(-1, 0), PinRef(1, 1)),
            Connection(PinRef(-1, 1), PinRef(1, 2)),
            Connection(PinRef(1, 3), PinRef(-1, 101)),
        ]
    }
    assert all(isinstance(ind.genome, CircuitNetlistGenome) for ind in pop)
    matching = 0
    for ind in pop:
        got = {
            (c.source.ic_index, c.source.pin, c.destination.ic_index, c.destination.pin)
            for c in ind.genome.connections
        }
        if got == known and list(ind.genome.ic_packages) == ["74HC86", "74HC08"]:
            matching += 1
    assert matching == 1


def test_scenario_registry_includes_circuits():
    from experimental.electronics.scenarios import list_electronics_scenarios, prepare_electronics_run
    names = list_electronics_scenarios()
    assert names[0] == "half_adder"
    assert "bjt_ce_amp" in names
    ev, pop, name = prepare_electronics_run("bjt_ce_amp", 3, seed=1)
    assert name == "bjt_ce_amp"
    assert len(pop) == 3
    assert ev.evaluate(pop[0]).artifacts.get("tool_used")


def test_logic_primitives_and_msi():
    from experimental.electronics.components.specs import LogicFunction
    from experimental.electronics.models.logic import eval_gate
    from experimental.electronics.models.independent_verifier import IndependentDigitalVerifier as V

    assert eval_gate(LogicFunction.NAND, [1, 1]) == 0
    assert eval_gate(LogicFunction.NAND, [1, 0]) == 1
    assert eval_gate(LogicFunction.XOR, [1, 0]) == 1
    dec = eval_gate(LogicFunction.DECODER_3TO8, [1, 0, 0])
    assert dec == list(V.decoder_3to8_reference(1, 0, 0))
    assert dec[1] == 0 and dec[0] == 1
    mux = eval_gate(LogicFunction.MUX_8TO1, [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    assert mux == V.mux_8to1_reference([0, 1, 0, 0, 0, 0, 0, 0], (1, 0, 0)) == 1


def test_equivalence_layer_not_fitness():
    import importlib.util

    from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
    from experimental.electronics.models.equivalence import encode_half_adder_ref, verify_equivalent
    from experimental.electronics.models.independent_verifier import IndependentDigitalVerifier

    good = CircuitNetlistGenome(
        ["74HC86", "74HC08"],
        [
            Connection(PinRef(-1, 0), PinRef(0, 1)),
            Connection(PinRef(-1, 1), PinRef(0, 2)),
            Connection(PinRef(0, 3), PinRef(-1, 100)),
            Connection(PinRef(-1, 0), PinRef(1, 1)),
            Connection(PinRef(-1, 1), PinRef(1, 2)),
            Connection(PinRef(1, 3), PinRef(-1, 101)),
        ],
        2, 2,
    )
    ref = IndependentDigitalVerifier.half_adder_reference
    ok = verify_equivalent(good, ref, encode_ref=encode_half_adder_ref)
    assert ok["equivalent"] is True
    # z3 is an optional dependency: when absent the scan method proves
    # equivalence exhaustively and must NOT fail the suite.
    if importlib.util.find_spec("z3") is not None:
        assert ok["method"] == "z3"
    else:
        assert ok["method"] == "scan"
        assert ok.get("z3") == "unavailable"
    bad = CircuitNetlistGenome(["74HC00", "74HC00"], [], 2, 2)
    no = verify_equivalent(bad, ref, encode_ref=encode_half_adder_ref)
    assert no["equivalent"] is False
    assert no.get("counterexample")


def _half():
    from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
    return CircuitNetlistGenome(
        ["74HC86", "74HC08"],
        [
            Connection(PinRef(-1, 0), PinRef(0, 1)),
            Connection(PinRef(-1, 1), PinRef(0, 2)),
            Connection(PinRef(0, 3), PinRef(-1, 100)),
            Connection(PinRef(-1, 0), PinRef(1, 1)),
            Connection(PinRef(-1, 1), PinRef(1, 2)),
            Connection(PinRef(1, 3), PinRef(-1, 101)),
        ],
        2, 2,
    )


def test_validity_accepts_known_half_adder():
    from experimental.electronics.models.validity import electrical_validity
    assert electrical_validity(_half())["valid"] is True


def test_validity_rejects_rail_and_multi_driver():
    from experimental.electronics.evaluators.digital_evaluator import MultiCorner74xxEvaluator
    from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
    from experimental.electronics.models.validity import electrical_validity

    rail = CircuitNetlistGenome(
        ["74HC00"],
        [Connection(PinRef(-1, 0), PinRef(0, 14))],
        2, 1,
    )
    assert electrical_validity(rail)["valid"] is False
    clash = CircuitNetlistGenome(
        ["74HC08"],
        [
            Connection(PinRef(-1, 0), PinRef(0, 1)),
            Connection(PinRef(-1, 1), PinRef(0, 1)),
        ],
        2, 1,
    )
    assert any("multi_driver" in v for v in electrical_validity(clash)["violations"])
    ev = MultiCorner74xxEvaluator([((0, 0), (0,))], max_delay_ns=150.0)
    res = ev.evaluate(rail)
    assert res.score == 0.0
    assert res.artifacts.get("invalid") is True
