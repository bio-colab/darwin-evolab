"""
genome.py — Cartesian Genetic Programming (CGP) Neuromorphic Circuit Genome.

Represents a directed acyclic graph (DAG) of logic, synaptic inhibition, and
clocked delay register nodes suitable for temporal neural signal processing.
"""
from __future__ import annotations

import enum
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

from evolab.genome import EvolabGenome


class NeuromorphicOp(str, enum.Enum):
    """Supported computational and synaptic primitives."""

    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"
    INHIBIT = "INHIBIT"  # A AND (NOT B) — synaptic inhibition
    DELAY = "DELAY"      # Z^-1 clocked register (D-Flip-Flop)
    WIRE = "WIRE"        # Pass-through input A


OP_TRANSISTORS: dict[NeuromorphicOp, int] = {
    NeuromorphicOp.WIRE: 0,
    NeuromorphicOp.NOT: 2,
    NeuromorphicOp.AND: 6,
    NeuromorphicOp.OR: 6,
    NeuromorphicOp.XOR: 8,
    NeuromorphicOp.INHIBIT: 6,  # NOT + AND
    NeuromorphicOp.DELAY: 12,   # Static D-Flip-Flop register
}


@dataclass
class NeuromorphicNode:
    """Individual node within the Cartesian neuromorphic DAG."""

    index: int
    op: NeuromorphicOp
    input_a: int
    input_b: int

    def serialize(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "op": self.op.value,
            "input_a": self.input_a,
            "input_b": self.input_b,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> NeuromorphicNode:
        return cls(
            index=int(data["index"]),
            op=NeuromorphicOp(data["op"]),
            input_a=int(data["input_a"]),
            input_b=int(data["input_b"]),
        )


@dataclass
class NeuromorphicCircuitGenome(EvolabGenome):
    """
    Evolvable Cartesian circuit genome with stateful temporal delay elements.
    """

    num_inputs: int
    num_outputs: int
    nodes: list[NeuromorphicNode] = field(default_factory=list)
    output_indices: list[int] = field(default_factory=list)

    @classmethod
    def random(
        cls,
        num_inputs: int,
        num_outputs: int,
        num_nodes: int,
        rng: random.Random,
    ) -> NeuromorphicCircuitGenome:
        """Constructs a random initial circuit DAG."""
        ops = list(NeuromorphicOp)
        nodes: list[NeuromorphicNode] = []
        for i in range(num_nodes):
            node_idx = num_inputs + i
            # Inputs must come from sensory inputs or preceding nodes (feed-forward DAG)
            valid_sources = list(range(node_idx))
            op = rng.choice(ops)
            in_a = rng.choice(valid_sources)
            in_b = rng.choice(valid_sources)
            nodes.append(NeuromorphicNode(index=node_idx, op=op, input_a=in_a, input_b=in_b))

        total_nodes = num_inputs + num_nodes
        outputs = [rng.randint(0, total_nodes - 1) for _ in range(num_outputs)]
        return cls(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            nodes=nodes,
            output_indices=outputs,
        )

    def clone(self) -> NeuromorphicCircuitGenome:
        return NeuromorphicCircuitGenome(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            nodes=[
                NeuromorphicNode(n.index, n.op, n.input_a, n.input_b)
                for n in self.nodes
            ],
            output_indices=list(self.output_indices),
        )

    def fingerprint(self) -> str:
        active = self.active_node_indices()
        repr_str = (
            f"in:{self.num_inputs}|out:{self.output_indices}|"
            + "|".join(
                f"{n.index}:{n.op.value}:{n.input_a}:{n.input_b}"
                for n in self.nodes
                if n.index in active
            )
        )
        return hashlib.sha256(repr_str.encode("utf-8")).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, NeuromorphicCircuitGenome):
            return 1.0
        if self.num_inputs != other.num_inputs or self.num_outputs != other.num_outputs:
            return 1.0
        if not self.nodes or not other.nodes:
            return 0.0

        diffs = 0
        total = min(len(self.nodes), len(other.nodes))
        for na, nb in zip(self.nodes[:total], other.nodes[:total]):
            if na.op != nb.op:
                diffs += 1
            if na.input_a != nb.input_a:
                diffs += 1
            if na.input_b != nb.input_b:
                diffs += 1

        for oa, ob in zip(self.output_indices, other.output_indices):
            if oa != ob:
                diffs += 1

        max_diffs = (total * 3) + len(self.output_indices)
        return round(diffs / max(max_diffs, 1), 6)

    def distance(self, other: EvolabGenome) -> float:
        """Alias for distance_to."""
        return self.distance_to(other)

    def __len__(self) -> int:
        """Returns total number of nodes in genome."""
        return len(self.nodes)

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "NeuromorphicCircuitGenome",
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "nodes": [n.serialize() for n in self.nodes],
            "output_indices": list(self.output_indices),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> NeuromorphicCircuitGenome:
        nodes = [NeuromorphicNode.deserialize(n) for n in data.get("nodes", [])]
        return cls(
            num_inputs=int(data["num_inputs"]),
            num_outputs=int(data["num_outputs"]),
            nodes=nodes,
            output_indices=[int(x) for x in data.get("output_indices", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return self.serialize()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeuromorphicCircuitGenome:
        return cls.deserialize(data)

    def describe(self) -> dict[str, float | int | str]:
        active = self.active_nodes()
        delays = sum(1 for n in active if n.op == NeuromorphicOp.DELAY)
        inhibits = sum(1 for n in active if n.op == NeuromorphicOp.INHIBIT)
        return {
            "active_nodes": len(active),
            "delay_registers": delays,
            "inhibitory_synapses": inhibits,
            "transistor_count": self.transistor_count(),
        }

    def active_node_indices(self) -> set[int]:
        """Finds all node indices connected to outputs via backward graph traversal."""
        active: set[int] = set()
        queue = [out for out in self.output_indices if out >= self.num_inputs]
        nodes_by_idx = {n.index: n for n in self.nodes}

        while queue:
            idx = queue.pop()
            if idx in active or idx < self.num_inputs:
                continue
            active.add(idx)
            node = nodes_by_idx.get(idx)
            if node is not None:
                if node.input_a >= self.num_inputs:
                    queue.append(node.input_a)
                if node.input_b >= self.num_inputs:
                    queue.append(node.input_b)

        return active

    def active_nodes(self) -> list[NeuromorphicNode]:
        active_set = self.active_node_indices()
        return [n for n in self.nodes if n.index in active_set]

    def transistor_count(self) -> int:
        return sum(OP_TRANSISTORS.get(n.op, 4) for n in self.active_nodes())

    def mutate(
        self,
        rng: random.Random | None = None,
        mutation_rate: float = 0.15,
        **kwargs: Any,
    ) -> NeuromorphicCircuitGenome:
        """Point mutations on node operators, input wiring, or output taps."""
        r = rng or random.Random()
        child = self.clone()
        ops = list(NeuromorphicOp)

        for node in child.nodes:
            if r.random() < mutation_rate:
                node.op = r.choice(ops)
            valid_sources = list(range(node.index))
            if r.random() < mutation_rate:
                node.input_a = r.choice(valid_sources)
            if r.random() < mutation_rate:
                node.input_b = r.choice(valid_sources)

        total_nodes = child.num_inputs + len(child.nodes)
        for i in range(len(child.output_indices)):
            if r.random() < mutation_rate:
                child.output_indices[i] = r.randint(0, total_nodes - 1)

        return child

    def crossover(
        self,
        other: EvolabGenome,
        rng: random.Random | None = None,
    ) -> NeuromorphicCircuitGenome:
        if not isinstance(other, NeuromorphicCircuitGenome) or len(self.nodes) != len(other.nodes):
            return self.clone()

        r = rng or random.Random()
        child = self.clone()
        # Uniform node-level crossover
        for i in range(len(child.nodes)):
            if r.random() < 0.5:
                n = other.nodes[i]
                child.nodes[i] = NeuromorphicNode(n.index, n.op, n.input_a, n.input_b)

        for i in range(len(child.output_indices)):
            if r.random() < 0.5:
                child.output_indices[i] = other.output_indices[i]

        return child

    def simulate_sequence(
        self,
        inputs_over_time: list[list[int]],
    ) -> list[list[int]]:
        """
        Executes the temporal stateful simulation of the circuit across time steps.
        Delays maintain internal state across steps (t -> t+1).
        """
        # Internal state registers for DELAY nodes: node_index -> registered bit
        delay_states: dict[int, int] = {
            n.index: 0 for n in self.nodes if n.op == NeuromorphicOp.DELAY
        }

        outputs_over_time: list[list[int]] = []

        for current_inputs in inputs_over_time:
            # Wire up sensory inputs
            signals: dict[int, int] = {i: int(current_inputs[i]) for i in range(self.num_inputs)}
            next_delay_states: dict[int, int] = {}

            # Evaluate nodes in topological order (index order is topological by construction)
            for node in self.nodes:
                val_a = signals.get(node.input_a, 0)
                val_b = signals.get(node.input_b, 0)

                if node.op == NeuromorphicOp.AND:
                    out = val_a & val_b
                elif node.op == NeuromorphicOp.OR:
                    out = val_a | val_b
                elif node.op == NeuromorphicOp.XOR:
                    out = val_a ^ val_b
                elif node.op == NeuromorphicOp.NOT:
                    out = 1 if val_a == 0 else 0
                elif node.op == NeuromorphicOp.INHIBIT:
                    out = val_a & (1 if val_b == 0 else 0)
                elif node.op == NeuromorphicOp.DELAY:
                    # Current output is current state of the flip-flop
                    out = delay_states.get(node.index, 0)
                    # Next state will be current input_a
                    next_delay_states[node.index] = val_a
                elif node.op == NeuromorphicOp.WIRE:
                    out = val_a
                else:
                    out = val_a

                signals[node.index] = out

            # Read outputs for this timestep
            step_outputs = [signals.get(out_idx, 0) for out_idx in self.output_indices]
            outputs_over_time.append(step_outputs)

            # Update delay registers for the next timestep
            delay_states.update(next_delay_states)

        return outputs_over_time
