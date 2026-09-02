"""Proposal operator: structured edits only. Not an oracle, not an LLM client."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evolab.genome import FloatGenome, Individual

from .models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
from .models.logic import PARTS_FOR_FUNCTION, missing_functions, parts_for_functions
from .models.validity import electrical_validity


ALLOWED = frozenset({"swap_ic", "add_wire", "remove_wire", "change_value"})


@dataclass(frozen=True)
class Proposal:
    action: str
    payload: dict[str, Any]
    reason: str = ""


def apply_proposal(genome: Any, proposal: Proposal) -> Any | None:
    if proposal.action not in ALLOWED:
        return None
    if isinstance(genome, CircuitNetlistGenome):
        return _apply_netlist(genome, proposal)
    if isinstance(genome, FloatGenome) and proposal.action == "change_value":
        return _apply_float(genome, proposal)
    return None


def _apply_float(genome: FloatGenome, proposal: Proposal) -> FloatGenome | None:
    idx = int(proposal.payload.get("index", -1))
    if idx < 0 or idx >= len(genome.values):
        return None
    vals = list(genome.values)
    vals[idx] = float(proposal.payload["value"])
    return FloatGenome(vals)


def _apply_netlist(genome: CircuitNetlistGenome, proposal: Proposal) -> CircuitNetlistGenome | None:
    ics = list(genome.ic_packages)
    conns = list(genome.connections)
    act = proposal.action
    p = proposal.payload
    if act == "swap_ic":
        idx = int(p.get("ic_index", 0))
        part = str(p.get("part", ""))
        if idx < 0 or idx >= len(ics) or not part:
            return None
        ics[idx] = part
    elif act == "remove_wire":
        idx = int(p.get("wire_index", -1))
        if idx < 0 or idx >= len(conns):
            return None
        conns.pop(idx)
    elif act == "add_wire":
        try:
            conns.append(
                Connection(
                    PinRef(int(p["src_ic"]), int(p["src_pin"])),
                    PinRef(int(p["dst_ic"]), int(p["dst_pin"])),
                )
            )
        except (KeyError, TypeError, ValueError):
            return None
    else:
        return None
    return CircuitNetlistGenome(
        ics, conns, genome.num_inputs, genome.num_outputs,
        functions_needed=genome.functions_needed,
    )


def accept(genome: Any) -> bool:
    if isinstance(genome, CircuitNetlistGenome):
        return electrical_validity(genome)["valid"]
    return genome is not None


def inject_proposals(base: Any, proposals: list[Proposal], limit: int = 20) -> list[Any]:
    out: list[Any] = []
    for prop in proposals[:limit]:
        nxt = apply_proposal(base, prop)
        if nxt is not None and accept(nxt):
            out.append(nxt)
    return out


def rule_proposals(genome: Any, max_n: int = 8) -> list[Proposal]:
    """Cheap stand-in for an LLM: cover missing logic parts. No model call."""
    if not isinstance(genome, CircuitNetlistGenome):
        return []
    used = genome.circuit.functions_used()
    miss = missing_functions(used, genome.functions_needed)
    props: list[Proposal] = []
    for fn in miss:
        part = PARTS_FOR_FUNCTION.get(fn.upper())
        if not part:
            continue
        for i in range(len(genome.ic_packages)):
            if genome.ic_packages[i] == part:
                continue
            props.append(
                Proposal(
                    "swap_ic",
                    {"ic_index": i, "part": part},
                    reason=f"cover_missing_{fn}",
                )
            )
            if len(props) >= max_n:
                return props
    return props


def seed_for_controller(
    functions_needed: tuple[str, ...] | list[str],
    num_inputs: int,
    num_outputs: int,
) -> CircuitNetlistGenome:
    """Gen-1 seed from the evaluator contract (needed functions + ports). No LLM."""
    parts = parts_for_functions(functions_needed)
    if not parts:
        parts = ["74HC00"]
    n_ics = 2 if num_inputs <= 2 else min(3, max(2, len(parts)))
    while len(parts) < n_ics:
        parts.append(parts[0])
    ics = parts[:n_ics]
    conns: list[Connection] = []
    for ic in range(len(ics)):
        conns.append(Connection(PinRef(-1, 0), PinRef(ic, 1)))
        if num_inputs > 1:
            conns.append(Connection(PinRef(-1, 1), PinRef(ic, 2)))
    if num_inputs > 2 and len(ics) > 1:
        conns.append(Connection(PinRef(-1, 2), PinRef(1, 2)))
    conns.append(Connection(PinRef(0, 3), PinRef(-1, 100)))
    if num_outputs > 1 and len(ics) > 1:
        conns.append(Connection(PinRef(1, 3), PinRef(-1, 101)))
    return CircuitNetlistGenome(
        ics, conns, num_inputs, num_outputs, functions_needed=tuple(functions_needed)
    )


def on_stagnation(individual: Individual, max_n: int = 8) -> list[Individual]:
    """Call when the engine reports a plateau. Injects valid proposal children."""
    props = rule_proposals(individual.genome, max_n=max_n)
    kids = inject_proposals(individual.genome, props, limit=max_n)
    return [Individual(genome=g, species=individual.species) for g in kids]
