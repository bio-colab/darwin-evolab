"""Combinational helpers used by the breadboard simulator."""
from __future__ import annotations

from collections.abc import Sequence

from ..components.specs import LogicFunction


def bit(v: int) -> int:
    return 1 if v else 0


def eval_gate(fn: LogicFunction, inputs: Sequence[int]) -> int | list[int]:
    bits = [bit(x) for x in inputs]
    if fn == LogicFunction.NOT:
        return 0 if bits[0] else 1
    if fn == LogicFunction.AND:
        return 1 if bits and all(bits) else 0
    if fn == LogicFunction.NAND:
        return 0 if bits and all(bits) else 1
    if fn == LogicFunction.OR:
        return 1 if any(bits) else 0
    if fn == LogicFunction.NOR:
        return 0 if any(bits) else 1
    if fn == LogicFunction.XOR:
        return sum(bits) & 1
    if fn == LogicFunction.DECODER_3TO8:
        a0 = bits[0] if bits else 0
        a1 = bits[1] if len(bits) > 1 else 0
        a2 = bits[2] if len(bits) > 2 else 0
        idx = (a2 << 2) | (a1 << 1) | a0
        # 74HC138 outputs are active-low
        out = [1] * 8
        if 0 <= idx < 8:
            out[idx] = 0
        return out
    if fn == LogicFunction.MUX_8TO1:
        data = bits[:8]
        while len(data) < 8:
            data.append(0)
        if len(bits) >= 11:
            sel = bits[8] | (bits[9] << 1) | (bits[10] << 2)
        else:
            sel = 0
        return data[sel & 7]
    if fn == LogicFunction.ADDER_4BIT:
        a = sum(bits[i] << i for i in range(min(4, len(bits))))
        b = sum(bits[i] << (i - 4) for i in range(4, min(8, len(bits))))
        s = a + b
        return [(s >> i) & 1 for i in range(5)]
    return 0


PARTS_FOR_FUNCTION = {
    "NAND": "74HC00",
    "NOR": "74HC02",
    "NOT": "74HC04",
    "AND": "74HC08",
    "OR": "74HC32",
    "XOR": "74HC86",
    "DECODER_3TO8": "74HC138",
    "MUX_8TO1": "74HC151",
    "ADDER_4BIT": "74HC283",
}


def missing_functions(used: Sequence[str], needed: Sequence[str]) -> list[str]:
    have = {u.upper() for u in used}
    return [n for n in needed if str(n).upper() not in have]


def parts_for_functions(needed: Sequence[str]) -> list[str]:
    out: list[str] = []
    for name in needed:
        part = PARTS_FOR_FUNCTION.get(str(name).upper())
        if part and part not in out:
            out.append(part)
    return out


def functions_in_packages(names: Sequence[str]) -> list[str]:
    from ..components.catalog_74xx import get_ic_spec

    seen: list[str] = []
    for name in names:
        try:
            spec = get_ic_spec(name)
        except Exception:
            continue
        for gate in spec.gates:
            tag = gate.logic_fn.value
            if tag not in seen:
                seen.append(tag)
    return seen
