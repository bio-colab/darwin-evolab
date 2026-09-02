"""Optional formal layer: netlist ≡ reference. Not used as fitness."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..components.specs import LogicFunction
from .circuit_netlist import BreadboardCircuit, CircuitNetlistGenome


def _circuit_of(target: Any) -> BreadboardCircuit:
    if isinstance(target, CircuitNetlistGenome):
        return target.circuit
    return target


def _scan_equivalent(
    circuit: BreadboardCircuit,
    reference_fn: Callable[..., tuple[int, ...]],
) -> dict[str, Any]:
    n = circuit.num_inputs
    if n > 16:
        return {
            "equivalent": False,
            "method": "scan_skipped",
            "reason": "too many inputs for exhaustive scan",
            "counterexample": None,
        }
    for mask in range(1 << n):
        vec = tuple((mask >> i) & 1 for i in range(n))
        got, stable = circuit.simulate(vec)
        if not stable:
            return {
                "equivalent": False,
                "method": "scan",
                "counterexample": {"inputs": vec, "reason": "unstable"},
            }
        exp = tuple(reference_fn(*vec))
        if tuple(got) != exp:
            return {
                "equivalent": False,
                "method": "scan",
                "counterexample": {"inputs": vec, "got": tuple(got), "expected": exp},
            }
    return {"equivalent": True, "method": "scan", "counterexample": None}


def _z3_gate(fn: LogicFunction, ins: list[Any]) -> Any:
    import z3

    def b(i: int) -> Any:
        return ins[i] if i < len(ins) else z3.BoolVal(False)

    if fn == LogicFunction.NOT:
        return z3.Not(b(0))
    if fn == LogicFunction.AND:
        return z3.And(*ins) if ins else z3.BoolVal(False)
    if fn == LogicFunction.NAND:
        return z3.Not(z3.And(*ins)) if ins else z3.BoolVal(True)
    if fn == LogicFunction.OR:
        return z3.Or(*ins) if ins else z3.BoolVal(False)
    if fn == LogicFunction.NOR:
        return z3.Not(z3.Or(*ins)) if ins else z3.BoolVal(True)
    if fn == LogicFunction.XOR:
        if not ins:
            return z3.BoolVal(False)
        acc = ins[0]
        for x in ins[1:]:
            acc = z3.Xor(acc, x)
        return acc
    if fn == LogicFunction.DECODER_3TO8:
        a0, a1, a2 = b(0), b(1), b(2)
        idx = z3.Concat(z3.If(a2, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                        z3.If(a1, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                        z3.If(a0, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
        # active-low one-hot without Concat complexity: explicit
        outs = []
        for k in range(8):
            sel = z3.And(
                a0 if (k & 1) else z3.Not(a0),
                a1 if (k & 2) else z3.Not(a1),
                a2 if (k & 4) else z3.Not(a2),
            )
            outs.append(z3.Not(sel))
        return outs
    if fn == LogicFunction.MUX_8TO1:
        data = [b(i) for i in range(8)]
        s0, s1, s2 = b(8), b(9), b(10)
        acc: Any = data[0]
        for k in range(8):
            sel = z3.And(
                s0 if (k & 1) else z3.Not(s0),
                s1 if (k & 2) else z3.Not(s1),
                s2 if (k & 4) else z3.Not(s2),
            )
            acc = z3.If(sel, data[k], acc)
        return acc
    if fn == LogicFunction.ADDER_4BIT:
        def nibble(start: int) -> Any:
            v = z3.BitVecVal(0, 5)
            for i in range(4):
                v = v + z3.If(b(start + i), z3.BitVecVal(1 << i, 5), z3.BitVecVal(0, 5))
            return v
        s = nibble(0) + nibble(4)
        return [s & 1 == 1, (s >> 1) & 1 == 1, (s >> 2) & 1 == 1, (s >> 3) & 1 == 1, (s >> 4) & 1 == 1]
    return z3.BoolVal(False)


def _z3_equivalent(
    circuit: BreadboardCircuit,
    encode_ref: Callable[..., tuple[Any, ...]],
) -> dict[str, Any]:
    import z3

    pins: dict[tuple[int, int], Any] = {}

    def node(ic: int, pin: int) -> Any:
        key = (ic, pin)
        if key not in pins:
            pins[key] = z3.Bool(f"p_{ic}_{pin}")
        return pins[key]

    s = z3.Solver()
    inputs = [z3.Bool(f"in_{i}") for i in range(circuit.num_inputs)]
    for i, var in enumerate(inputs):
        s.add(node(-1, i) == var)

    for conn in circuit.connections:
        s.add(node(conn.destination.ic_index, conn.destination.pin) == node(conn.source.ic_index, conn.source.pin))

    for ic_idx, spec in enumerate(circuit.ic_specs):
        for gate in spec.gates:
            ins = [node(ic_idx, p) for p in gate.input_pins]
            out = _z3_gate(gate.logic_fn, ins)
            if isinstance(out, list):
                for pin, expr in zip(gate.output_pins, out):
                    s.add(node(ic_idx, pin) == expr)
            else:
                for pin in gate.output_pins:
                    s.add(node(ic_idx, pin) == out)

    outs = [node(-1, 100 + i) for i in range(circuit.num_outputs)]
    ref = encode_ref(*inputs)
    if len(ref) != len(outs):
        return {
            "equivalent": False,
            "method": "z3",
            "reason": "output arity mismatch",
            "counterexample": None,
        }
    s.add(z3.Or(*[outs[i] != ref[i] for i in range(len(outs))]))
    result = s.check()
    if result == z3.unsat:
        return {"equivalent": True, "method": "z3", "counterexample": None}
    if result == z3.sat:
        m = s.model()
        cex = tuple(1 if z3.is_true(m.eval(v, model_completion=True)) else 0 for v in inputs)
        return {
            "equivalent": False,
            "method": "z3",
            "counterexample": {"inputs": cex},
        }
    return {"equivalent": False, "method": "z3", "reason": str(result), "counterexample": None}


def encode_half_adder_ref(*bits: Any) -> tuple[Any, ...]:
    import z3
    a, b = bits[0], bits[1]
    return (z3.Xor(a, b), z3.And(a, b))


def encode_full_adder_ref(*bits: Any) -> tuple[Any, ...]:
    import z3
    a, b, c = bits[0], bits[1], bits[2]
    s = z3.Xor(z3.Xor(a, b), c)
    cout = z3.Or(z3.And(a, b), z3.And(c, z3.Xor(a, b)))
    return (s, cout)


def verify_equivalent(
    target: Any,
    reference_fn: Callable[..., tuple[int, ...]],
    encode_ref: Callable[..., tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    """Prove or refute functional equivalence. Never called from fitness."""
    circuit = _circuit_of(target)
    scan = _scan_equivalent(circuit, reference_fn)
    try:
        import z3  # noqa: F401
    except ImportError:
        scan["z3"] = "unavailable"
        return scan
    if encode_ref is None:
        scan["z3"] = "no_encoder"
        return scan
    proved = _z3_equivalent(circuit, encode_ref)
    proved["scan_agrees"] = scan["equivalent"] == proved["equivalent"]
    proved["scan"] = scan
    return proved
