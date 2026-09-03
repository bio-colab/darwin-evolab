"""
boolean_expr.py — Safe AST-based Boolean Expression Parser and Truth Table Generator.

Parses arbitrary Boolean logic equations without unsafe eval() calls, extracts variables,
and synthesizes verification truth tables for digital logic evolutionary synthesis.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class BooleanParseResult:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    expressions: dict[str, str]
    truth_table: list[tuple[tuple[int, ...], tuple[int, ...]]]
    num_inputs: int
    num_outputs: int


class BooleanExpressionParser:
    """Safely parses Boolean algebra equations into executable AST evaluators and truth tables."""

    ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Constant,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Invert,
        ast.Not,
        ast.Load,
    )

    def __init__(self, raw_input: str | dict[str, str]) -> None:
        self.raw_input = raw_input
        self.equations: dict[str, str] = self._normalize_input(raw_input)
        self.parsed_ast: dict[str, ast.Expression] = {}
        self.input_vars: list[str] = []
        self._parse_and_validate()

    def _normalize_input(self, raw: str | dict[str, str]) -> dict[str, str]:
        if isinstance(raw, dict):
            return {k.strip(): self._clean_expression(v) for k, v in raw.items()}

        eqs: dict[str, str] = {}
        # Split by ';' or newline
        lines = [line.strip() for line in re.split(r"[;\n]", str(raw)) if line.strip()]
        for idx, line in enumerate(lines):
            if "=" in line:
                lhs, rhs = line.split("=", 1)
                eqs[lhs.strip()] = self._clean_expression(rhs)
            else:
                out_name = f"out{idx}" if len(lines) > 1 else "out"
                eqs[out_name] = self._clean_expression(line)
        return eqs

    @staticmethod
    def _clean_expression(expr: str) -> str:
        s = expr.strip()
        # Word replacements
        s = re.sub(r"\bAND\b", "&", s, flags=re.IGNORECASE)
        s = re.sub(r"\bOR\b", "|", s, flags=re.IGNORECASE)
        s = re.sub(r"\bXOR\b", "^", s, flags=re.IGNORECASE)
        s = re.sub(r"\bNOT\b", "~", s, flags=re.IGNORECASE)
        # Symbol replacements
        s = s.replace("!", "~")
        return s

    def _parse_and_validate(self) -> None:
        variables_seen: set[str] = set()
        for out_name, expr in self.equations.items():
            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if not isinstance(node, self.ALLOWED_NODES):
                    raise ValueError(f"Disallowed syntax node {type(node).__name__} in expression: {expr!r}")
                if isinstance(node, ast.Name):
                    variables_seen.add(node.id)
            self.parsed_ast[out_name] = tree

        # Sort input variables alphabetically for deterministic column ordering
        self.input_vars = sorted(variables_seen)

    def _evaluate_node(self, node: ast.AST, env: dict[str, int]) -> int:
        if isinstance(node, ast.Expression):
            return self._evaluate_node(node.body, env)
        elif isinstance(node, ast.Name):
            return env.get(node.id, 0) & 1
        elif isinstance(node, ast.Constant):
            return 1 if bool(node.value) else 0
        elif isinstance(node, ast.UnaryOp):
            val = self._evaluate_node(node.operand, env)
            return (1 - val) if isinstance(node.op, (ast.Invert, ast.Not)) else val
        elif isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left, env)
            right = self._evaluate_node(node.right, env)
            if isinstance(node.op, ast.BitAnd):
                return left & right
            elif isinstance(node.op, ast.BitOr):
                return left | right
            elif isinstance(node.op, ast.BitXor):
                return left ^ right
        raise ValueError(f"Cannot evaluate node {type(node).__name__}")

    def evaluate_vector(self, inputs: dict[str, int]) -> dict[str, int]:
        return {out: self._evaluate_node(tree, inputs) for out, tree in self.parsed_ast.items()}

    def generate_truth_table(self, max_exhaustive_inputs: int = 12) -> BooleanParseResult:
        n_in = len(self.input_vars)
        outputs_list = list(self.equations.keys())
        table: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        if n_in <= max_exhaustive_inputs:
            # Full exhaustive evaluation: 2^N rows
            for mask in range(1 << n_in):
                in_vec = tuple((mask >> (n_in - 1 - i)) & 1 for i in range(n_in))
                env = dict(zip(self.input_vars, in_vec))
                out_dict = self.evaluate_vector(env)
                out_vec = tuple(out_dict[out_name] for out_name in outputs_list)
                table.append((in_vec, out_vec))
        else:
            # Sub-sampled deterministically (corners + random walk)
            import random
            rng = random.Random(42)
            # All zeros and all ones
            for corner in (0, (1 << n_in) - 1):
                in_vec = tuple((corner >> (n_in - 1 - i)) & 1 for i in range(n_in))
                env = dict(zip(self.input_vars, in_vec))
                out_dict = self.evaluate_vector(env)
                table.append((in_vec, tuple(out_dict[o] for o in outputs_list)))
            # Sample 256 representative vectors
            for _ in range(256):
                in_vec = tuple(rng.randint(0, 1) for _ in range(n_in))
                env = dict(zip(self.input_vars, in_vec))
                out_dict = self.evaluate_vector(env)
                table.append((in_vec, tuple(out_dict[o] for o in outputs_list)))

        return BooleanParseResult(
            inputs=tuple(self.input_vars),
            outputs=tuple(outputs_list),
            expressions=dict(self.equations),
            truth_table=table,
            num_inputs=n_in,
            num_outputs=len(outputs_list),
        )


def parse_boolean_spec(expr_input: str | dict[str, str]) -> BooleanParseResult:
    """Convenience helper to parse Boolean expressions and return the full specification."""
    parser = BooleanExpressionParser(expr_input)
    return parser.generate_truth_table()
