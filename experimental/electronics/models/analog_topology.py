"""
analog_topology.py — Free-Form Analog Circuit Topology Synthesis and Genome.

Enables generative synthesis of continuous-time analog networks (filters, attenuators,
amplifiers) from scratch via topological mutations (component addition, deletion,
value retuning, and node rewiring) rather than fixed-parameter sizing.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import enum
import hashlib
import math
import random
from typing import Any

from evolab.genome import EvolabGenome


class AnalogComponentKind(str, enum.Enum):
    RESISTOR = "R"
    CAPACITOR = "C"
    INDUCTOR = "L"
    BJT = "Q"
    DIODE = "D"


@dataclass(frozen=True)
class AnalogComponent:
    """Represents a discrete analog component connected between circuit nodes."""
    kind: AnalogComponentKind
    name: str
    nodes: tuple[str, ...]
    value: float
    model: str = ""

    def to_spice_line(self) -> str:
        nodes_str = " ".join(self.nodes)
        if self.kind == AnalogComponentKind.RESISTOR:
            return f"{self.name} {nodes_str} {self.value:.6g}"
        elif self.kind == AnalogComponentKind.CAPACITOR:
            return f"{self.name} {nodes_str} {self.value:.6e}"
        elif self.kind == AnalogComponentKind.INDUCTOR:
            return f"{self.name} {nodes_str} {self.value:.6e}"
        elif self.kind == AnalogComponentKind.BJT:
            mod = self.model or "NPN_GENERIC"
            return f"{self.name} {nodes_str} {mod}"
        elif self.kind == AnalogComponentKind.DIODE:
            mod = self.model or "D_GENERIC"
            return f"{self.name} {nodes_str} {mod}"
        return f"* Unknown component {self.name}"


class AnalogTopologyGenome(EvolabGenome):
    """Evolutionary genome representing an arbitrary interconnected analog circuit topology."""

    def __init__(
        self,
        components: Sequence[AnalogComponent],
        input_node: str = "in",
        output_node: str = "out",
        ground_node: str = "0",
        vcc_node: str = "vcc",
        max_internal_nodes: int = 5,
    ) -> None:
        self.components = tuple(components)
        self.input_node = input_node
        self.output_node = output_node
        self.ground_node = ground_node
        self.vcc_node = vcc_node
        self.max_internal_nodes = max_internal_nodes

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def clone(self) -> AnalogTopologyGenome:
        return AnalogTopologyGenome(
            components=list(self.components),
            input_node=self.input_node,
            output_node=self.output_node,
            ground_node=self.ground_node,
            vcc_node=self.vcc_node,
            max_internal_nodes=self.max_internal_nodes,
        )

    def fingerprint(self) -> str:
        rep = ";".join(c.to_spice_line() for c in sorted(self.components, key=lambda c: c.name))
        return hashlib.sha256(rep.encode("utf-8")).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, AnalogTopologyGenome):
            return 1.0
        s1 = {c.name: (c.kind, c.nodes, round(math.log10(max(c.value, 1e-15)), 2)) for c in self.components}
        s2 = {c.name: (c.kind, c.nodes, round(math.log10(max(c.value, 1e-15)), 2)) for c in other.components}
        all_names = set(s1.keys()) | set(s2.keys())
        if not all_names:
            return 0.0
        diff = sum(1 for k in all_names if s1.get(k) != s2.get(k))
        return diff / len(all_names)

    def describe(self) -> dict[str, Any]:
        kinds = [c.kind.value for c in self.components]
        return {
            "parts": len(self.components),
            "kinds": dict((k, kinds.count(k)) for k in set(kinds)),
            "max_depth": 1,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "components": [
                {
                    "kind": c.kind.value,
                    "name": c.name,
                    "nodes": list(c.nodes),
                    "value": c.value,
                    "model": c.model,
                }
                for c in self.components
            ],
            "input_node": self.input_node,
            "output_node": self.output_node,
        }

    def get_all_nodes(self) -> set[str]:
        nodes = {self.input_node, self.output_node, self.ground_node, self.vcc_node}
        for c in self.components:
            nodes.update(c.nodes)
        return nodes

    def to_spice_netlist(self, title: str = "Synthesized Analog Topology", vcc: float = 5.0) -> str:
        lines = [
            f"* {title}",
            f"Vcc {self.vcc_node} 0 DC {vcc}",
            f"Vin {self.input_node} 0 AC 1.0",
            "",
            "* Circuit Components",
        ]
        for c in self.components:
            lines.append(c.to_spice_line())
        lines.extend([
            "",
            "* Models",
            ".model NPN_GENERIC NPN(IS=1e-14 BF=100)",
            ".model D_GENERIC D(IS=1e-14 RS=1)",
            ".ac dec 10 10 100k",
            ".end",
        ])
        return "\n".join(lines) + "\n"

    def mutate(
        self,
        rng: random.Random | None = None,
        sigma: float = 0.1,
        kind: str = "light",
        **kwargs: Any,
    ) -> AnalogTopologyGenome:
        r = rng or random.Random()
        comps = list(self.components)
        nodes = sorted(self.get_all_nodes())

        mutation_type = r.choice(["modify_value", "add_component", "rewire", "remove_component"])

        if mutation_type == "modify_value" and comps:
            idx = r.randint(0, len(comps) - 1)
            target_comp = comps[idx]
            factor = r.choice([0.5, 0.8, 1.25, 2.0, 10.0, 0.1])
            new_val = max(1e-12, min(1e7, target_comp.value * factor))
            comps[idx] = AnalogComponent(
                kind=target_comp.kind,
                name=target_comp.name,
                nodes=target_comp.nodes,
                value=new_val,
                model=target_comp.model,
            )

        elif mutation_type == "add_component" and len(comps) < 15:
            kind = r.choice([AnalogComponentKind.RESISTOR, AnalogComponentKind.CAPACITOR])
            name_prefix = kind.value
            existing_indices = [
                int(c.name[len(name_prefix):])
                for c in comps if c.name.startswith(name_prefix) and c.name[len(name_prefix):].isdigit()
            ]
            next_idx = (max(existing_indices) + 1) if existing_indices else 1
            comp_name = f"{name_prefix}{next_idx}"

            # Pick 2 different nodes
            n1 = r.choice(nodes)
            other_nodes = [n for n in nodes if n != n1] or [self.ground_node]
            n2 = r.choice(other_nodes)

            val = r.choice([100.0, 1000.0, 10000.0, 47000.0]) if kind == AnalogComponentKind.RESISTOR else r.choice([1e-9, 1e-8, 1e-7, 1e-6])
            comps.append(AnalogComponent(kind=kind, name=comp_name, nodes=(n1, n2), value=val))

        elif mutation_type == "rewire" and comps:
            idx = r.randint(0, len(comps) - 1)
            target_comp = comps[idx]
            if len(target_comp.nodes) == 2:
                n1, n2 = target_comp.nodes
                if r.random() < 0.5:
                    n1 = r.choice(nodes)
                else:
                    n2 = r.choice(nodes)
                comps[idx] = AnalogComponent(
                    kind=target_comp.kind,
                    name=target_comp.name,
                    nodes=(n1, n2),
                    value=target_comp.value,
                    model=target_comp.model,
                )

        elif mutation_type == "remove_component" and len(comps) > 2:
            idx = r.randint(0, len(comps) - 1)
            comps.pop(idx)

        return AnalogTopologyGenome(
            components=comps,
            input_node=self.input_node,
            output_node=self.output_node,
            ground_node=self.ground_node,
            vcc_node=self.vcc_node,
            max_internal_nodes=self.max_internal_nodes,
        )

    def crossover(self, other: EvolabGenome, rng: random.Random | None = None) -> AnalogTopologyGenome:
        if not isinstance(other, AnalogTopologyGenome):
            return self.clone()
        r = rng or random.Random()
        # Uniform component mixing
        all_comps = list(self.components) + list(other.components)
        selected_comps = []
        names_seen = set()
        for c in all_comps:
            if c.name not in names_seen and r.random() < 0.6:
                selected_comps.append(c)
                names_seen.add(c.name)
        if len(selected_comps) < 2:
            selected_comps = list(self.components)
        return AnalogTopologyGenome(
            components=selected_comps,
            input_node=self.input_node,
            output_node=self.output_node,
            ground_node=self.ground_node,
            vcc_node=self.vcc_node,
            max_internal_nodes=self.max_internal_nodes,
        )
