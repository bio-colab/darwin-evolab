from evolab.genome import Individual
from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
from experimental.electronics.proposal import Proposal, apply_proposal, inject_proposals, on_stagnation, rule_proposals


def _wired(ics):
    return CircuitNetlistGenome(
        ics,
        [
            Connection(PinRef(-1, 0), PinRef(0, 1)),
            Connection(PinRef(-1, 1), PinRef(0, 2)),
            Connection(PinRef(0, 3), PinRef(-1, 100)),
            Connection(PinRef(-1, 0), PinRef(1, 1)),
            Connection(PinRef(-1, 1), PinRef(1, 2)),
            Connection(PinRef(1, 3), PinRef(-1, 101)),
        ],
        2, 2, functions_needed=("XOR", "AND"),
    )


def test_rule_proposal_covers_missing_xor_and_guard_rejects_rail():
    g = _wired(["74HC00", "74HC08"])
    props = rule_proposals(g)
    assert any(p.action == "swap_ic" and p.payload.get("part") == "74HC86" for p in props)
    kids = inject_proposals(g, props)
    assert any(list(k.ic_packages) == ["74HC86", "74HC08"] for k in kids)
    bad = Proposal("add_wire", {"src_ic": -1, "src_pin": 0, "dst_ic": 0, "dst_pin": 14})
    assert apply_proposal(g, bad) is not None
    assert inject_proposals(g, [bad]) == []


def test_stagnation_injects_individuals():
    ind = Individual(genome=_wired(["74HC00", "74HC08"]), species="spec_electronics")
    kids = on_stagnation(ind)
    assert kids
    assert all(isinstance(k, Individual) for k in kids)


def test_seed_matches_half_adder_controller():
    from experimental.electronics.proposal import seed_for_controller
    from experimental.electronics.models.validity import electrical_validity
    g = seed_for_controller(("XOR", "AND"), 2, 2)
    assert list(g.ic_packages) == ["74HC86", "74HC08"]
    assert electrical_validity(g)["valid"] is True
