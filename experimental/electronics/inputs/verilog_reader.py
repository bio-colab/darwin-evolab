"""
verilog_reader.py — Synthesizable Verilog RTL Behavioral Reader.

Parses continuous assignment and combinational Verilog-2001 modules,
extracts input/output ports, and derives verification truth tables for logic synthesis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .boolean_expr import BooleanExpressionParser, BooleanParseResult


@dataclass(frozen=True)
class VerilogModuleSpec:
    module_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    assignments: dict[str, str]
    parse_result: BooleanParseResult


class VerilogRTLReader:
    """Extracts combinational logic semantics from Verilog-2001 source files."""

    def __init__(self, source_code: str) -> None:
        self.raw_code = source_code
        self.module_name = ""
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.assignments: dict[str, str] = {}
        self._parse()

    @classmethod
    def from_file(cls, filepath: str | Path) -> VerilogRTLReader:
        p = Path(filepath)
        return cls(p.read_text(encoding="utf-8", errors="ignore"))

    def _parse(self) -> None:
        # Strip single-line and multi-line comments
        text = re.sub(r"//.*", "", self.raw_code)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

        # 1. Module name
        m_mod = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", text)
        if m_mod:
            self.module_name = m_mod.group(1)
        else:
            self.module_name = "verilog_module"

        # 2. Extract inputs
        # Match input declarations: input wire [1:0] sel; or input a, b;
        for m in re.finditer(r"\binput\s+(?:wire\s+|reg\s+)?(?:\[(\d+):(\d+)\]\s*)?([A-Za-z0-9_,\s]+);", text):
            msb_str, lsb_str, names_str = m.groups()
            names = [n.strip() for n in names_str.split(",") if n.strip()]
            if msb_str is not None and lsb_str is not None:
                msb, lsb = int(msb_str), int(lsb_str)
                step = 1 if msb >= lsb else -1
                for name in names:
                    for bit in range(msb, lsb - step, -step):
                        self.inputs.append(f"{name}_{bit}")
            else:
                self.inputs.extend(names)

        # 3. Extract outputs
        for m in re.finditer(r"\boutput\s+(?:wire\s+|reg\s+)?(?:\[(\d+):(\d+)\]\s*)?([A-Za-z0-9_,\s]+);", text):
            msb_str, lsb_str, names_str = m.groups()
            names = [n.strip() for n in names_str.split(",") if n.strip()]
            if msb_str is not None and lsb_str is not None:
                msb, lsb = int(msb_str), int(lsb_str)
                step = 1 if msb >= lsb else -1
                for name in names:
                    for bit in range(msb, lsb - step, -step):
                        self.outputs.append(f"{name}_{bit}")
            else:
                self.outputs.extend(names)

        # 4. Extract assign statements
        for m in re.finditer(r"\bassign\s+([A-Za-z0-9_]+)\s*=\s*([^;]+);", text):
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            # Convert ternary: cond ? a : b -> (cond & a) | (~cond & b)
            rhs_converted = self._convert_ternary(rhs)
            self.assignments[lhs] = rhs_converted

        if not self.outputs and self.assignments:
            self.outputs = list(self.assignments.keys())

    def _convert_ternary(self, expr: str) -> str:
        # Matches: cond ? a : b
        ternary_pat = re.compile(r"([A-Za-z0-9_~&|^!]+)\s*\?\s*([A-Za-z0-9_~&|^!]+)\s*:\s*([A-Za-z0-9_~&|^!]+)")
        while True:
            m = ternary_pat.search(expr)
            if not m:
                break
            cond, if_t, if_f = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            sub = f"(({cond} & {if_t}) | (~({cond}) & {if_f}))"
            expr = expr[:m.start()] + sub + expr[m.end():]
        return expr

    def to_spec(self) -> VerilogModuleSpec:
        if not self.assignments:
            raise ValueError(f"No combinational assign statements found in module {self.module_name}")

        parser = BooleanExpressionParser(self.assignments)
        parse_res = parser.generate_truth_table()

        return VerilogModuleSpec(
            module_name=self.module_name,
            inputs=tuple(self.inputs or parse_res.inputs),
            outputs=tuple(self.outputs or parse_res.outputs),
            assignments=dict(self.assignments),
            parse_result=parse_res,
        )


def parse_verilog_spec(filepath_or_code: str | Path) -> VerilogModuleSpec:
    """Convenience helper to read Verilog RTL and return its complete specification."""
    p = Path(filepath_or_code)
    if p.is_file():
        return VerilogRTLReader.from_file(p).to_spec()
    return VerilogRTLReader(str(filepath_or_code)).to_spec()
