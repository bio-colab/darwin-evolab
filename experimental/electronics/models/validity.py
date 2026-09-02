"""ElectricalValidityGuard — reject illegal netlists before simulation."""
from __future__ import annotations

from typing import Any

from ..components.catalog_74xx import get_ic_spec
from .circuit_netlist import CircuitNetlistGenome, PinRef


def classify_pin(genome: CircuitNetlistGenome, ref: PinRef) -> str:
    if ref.ic_index == -1:
        if 0 <= ref.pin < genome.num_inputs:
            return "primary_in"
        if 100 <= ref.pin < 100 + genome.num_outputs:
            return "primary_out"
        return "illegal"
    if ref.ic_index < 0 or ref.ic_index >= len(genome.ic_packages):
        return "illegal"
    try:
        spec = get_ic_spec(genome.ic_packages[ref.ic_index])
    except KeyError:
        return "illegal"
    if ref.pin == spec.vcc_pin:
        return "vcc"
    if ref.pin == spec.gnd_pin:
        return "gnd"
    if ref.pin < 1 or ref.pin > spec.pin_count:
        return "illegal"
    for gate in spec.gates:
        if ref.pin in gate.output_pins:
            return "output"
        if ref.pin in gate.input_pins:
            return "input"
    return "unused"


def is_driver(kind: str) -> bool:
    return kind in ("primary_in", "output")


def is_load(kind: str) -> bool:
    return kind in ("primary_out", "input")


def electrical_validity(target: Any) -> dict[str, Any]:
    genome = target.genome if hasattr(target, "genome") and not isinstance(target, CircuitNetlistGenome) else target
    if not isinstance(genome, CircuitNetlistGenome):
        return {"valid": False, "violations": ["not_a_netlist"]}

    violations: list[str] = []
    drivers: dict[tuple[int, int], int] = {}

    for i, conn in enumerate(genome.connections):
        sk = classify_pin(genome, conn.source)
        dk = classify_pin(genome, conn.destination)
        if sk in ("vcc", "gnd") or dk in ("vcc", "gnd"):
            violations.append(f"wire_{i}:rail")
        if sk == "unused" or dk == "unused":
            violations.append(f"wire_{i}:unused_pin")
        if sk == "illegal" or dk == "illegal":
            violations.append(f"wire_{i}:illegal_pin")
        if sk == "input" and dk == "input":
            violations.append(f"wire_{i}:input_to_input")
        if sk == "output" and dk == "output":
            violations.append(f"wire_{i}:output_to_output")
        if sk == "primary_out":
            violations.append(f"wire_{i}:driven_from_port")
        if dk == "primary_in":
            violations.append(f"wire_{i}:drive_into_port")
        if not is_driver(sk):
            violations.append(f"wire_{i}:src_not_driver:{sk}")
        if not is_load(dk):
            violations.append(f"wire_{i}:dst_not_load:{dk}")
        dest = (conn.destination.ic_index, conn.destination.pin)
        drivers[dest] = drivers.get(dest, 0) + 1

    for dest, n in drivers.items():
        if n > 1:
            violations.append(f"multi_driver:{dest}:{n}")

    return {"valid": not violations, "violations": violations}


class ElectricalValidityGuard:
    def check(self, target: Any) -> dict[str, Any]:
        return electrical_validity(target)

    def reject(self, target: Any) -> bool:
        return not self.check(target)["valid"]
