"""
Fault-Guided AST Mutation & Spectrum-Based Fault Localization (SBFL).
Traces line execution across passing and failing tests, maps execution traces to AST nodes,
and focuses evolutionary mutations on suspicious failure-inducing code locations.
"""
from __future__ import annotations

import ast
import copy
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SuspiciousNode:
    """Represents an AST node tagged with its fault suspicion score and context."""

    node_id: int
    node_type: str
    line_no: int
    suspicion_score: float  # [0.0, 1.0]
    func_name: str = ""
    reason: str = ""
    node: ast.AST | None = None


@dataclass
class SuspicionMap:
    """Collection of suspicious lines and AST nodes computed from test coverage spectra."""

    line_scores: dict[int, float] = field(default_factory=dict)
    suspicious_nodes: list[SuspiciousNode] = field(default_factory=list)
    total_passed: int = 0
    total_failed: int = 0

    def get_top_nodes(self, top_k: int = 5, min_score: float = 0.1) -> list[SuspiciousNode]:
        """Returns the top K suspicious nodes sorted by suspicion score descending."""
        filtered = [n for n in self.suspicious_nodes if n.suspicion_score >= min_score]
        return sorted(filtered, key=lambda n: n.suspicion_score, reverse=True)[:top_k]


class LineCoverageTracer:
    """Lightweight tracer that captures executed line numbers for a specific target file/function."""

    def __init__(self, target_filename: str | None = None):
        self.target_filename = target_filename
        self.executed_lines: set[int] = set()

    def _trace_func(self, frame: Any, event: str, arg: Any) -> Any:
        if event == "line":
            filename = frame.f_code.co_filename
            if self.target_filename is None or self.target_filename in filename:
                self.executed_lines.add(frame.f_lineno)
        return self._trace_func

    def __enter__(self) -> LineCoverageTracer:
        self.executed_lines.clear()
        sys.settrace(self._trace_func)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        sys.settrace(None)


def compute_ochiai_score(
    failed_hits: int, total_failed: int, passed_hits: int, total_passed: int
) -> float:
    """Computes standard Ochiai fault localization score in [0.0, 1.0]."""
    if total_failed == 0 or failed_hits == 0:
        return 0.0
    denominator = math.sqrt(total_failed * (failed_hits + passed_hits))
    if denominator == 0:
        return 0.0
    return min(1.0, failed_hits / denominator)


def build_suspicion_map(
    tree_or_code: ast.AST | str,
    test_runs: Sequence[tuple[set[int], bool]],
    target_func: str | None = None,
) -> SuspicionMap:
    """Builds a SuspicionMap from test spectra (set of lines executed, passed boolean)."""
    if isinstance(tree_or_code, str):
        tree = ast.parse(tree_or_code)
    else:
        tree = tree_or_code

    total_passed = sum(1 for _, passed in test_runs if passed)
    total_failed = sum(1 for _, passed in test_runs if not passed)

    # Gather all unique executed lines
    all_lines: set[int] = set()
    for lines, _ in test_runs:
        all_lines.update(lines)

    # Compute line suspicion
    line_scores: dict[int, float] = {}
    for line in all_lines:
        failed_hits = sum(1 for lines, passed in test_runs if (line in lines and not passed))
        passed_hits = sum(1 for lines, passed in test_runs if (line in lines and passed))
        score = compute_ochiai_score(failed_hits, total_failed, passed_hits, total_passed)
        line_scores[line] = round(score, 4)

    # Walk AST to map suspicious lines to specific modifiable AST nodes
    suspicious_nodes: list[SuspiciousNode] = []

    class SuspicionNodeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_func = ""

        def visit_FunctionDef(self, node: ast.FunctionDef):
            prev_func = self.current_func
            self.current_func = node.name
            self.generic_visit(node)
            self.current_func = prev_func

        def visit_Compare(self, node: ast.Compare):
            self._tag_node(node, "Compare")
            self.generic_visit(node)

        def visit_BinOp(self, node: ast.BinOp):
            self._tag_node(node, "BinOp")
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                self._tag_node(node, "Constant")
            self.generic_visit(node)

        def visit_If(self, node: ast.If):
            self._tag_node(node.test, "IfCondition")
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return):
            if node.value:
                self._tag_node(node, "Return")
            self.generic_visit(node)

        def visit_Pass(self, node: ast.Pass):
            self._tag_node(node, "Pass")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            self._tag_node(node, "Call")
            self.generic_visit(node)

        def visit_UnaryOp(self, node: ast.UnaryOp):
            self._tag_node(node, "UnaryOp")
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript):
            self._tag_node(node, "Subscript")
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign):
            self._tag_node(node, "Assign")
            self.generic_visit(node)

        def _tag_node(self, node: ast.AST, node_type: str):
            line = getattr(node, "lineno", None)
            if line is not None and line in line_scores:
                score = line_scores[line]
                if score > 0.0:
                    reason = f"Executed in failing tests (Ochiai score: {score})"
                    suspicious_nodes.append(
                        SuspiciousNode(
                            node_id=id(node),
                            node_type=node_type,
                            line_no=line,
                            suspicion_score=score,
                            func_name=self.current_func,
                            reason=reason,
                            node=node,
                        )
                    )

    visitor = SuspicionNodeVisitor()
    visitor.visit(tree)

    return SuspicionMap(
        line_scores=line_scores,
        suspicious_nodes=suspicious_nodes,
        total_passed=total_passed,
        total_failed=total_failed,
    )


class FaultGuidedASTMutator:
    """Mutates ASTs by focusing changes on high-suspicion nodes identified by test failures."""

    COMPARE_SWAPS = {
        ast.Gt: [ast.GtE, ast.Lt, ast.Eq],
        ast.GtE: [ast.Gt, ast.Eq, ast.LtE],
        ast.Lt: [ast.LtE, ast.Gt, ast.Eq],
        ast.LtE: [ast.Lt, ast.Eq, ast.GtE],
        ast.Eq: [ast.NotEq, ast.GtE, ast.LtE],
        ast.NotEq: [ast.Eq],
    }

    BINOP_SWAPS = {
        ast.Add: [ast.Sub, ast.Mult],
        ast.Sub: [ast.Add, ast.Mult],
        ast.Mult: [ast.Add, ast.FloorDiv, ast.Div],
        ast.FloorDiv: [ast.Mult, ast.Div],
        ast.Div: [ast.Mult, ast.FloorDiv],
    }

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def mutate(
        self,
        tree: ast.AST,
        suspicion_map: SuspicionMap,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> tuple[ast.AST | None, str | None]:
        """Selects a top-suspicious node and applies a domain-targeted mutation."""
        top_candidates = suspicion_map.get_top_nodes(top_k=top_k, min_score=min_score)
        if not top_candidates:
            return None, None

        # Weighted selection based on suspicion score
        weights = [c.suspicion_score for c in top_candidates]
        target_candidate = self.rng.choices(top_candidates, weights=weights, k=1)[0]

        mutated_tree = copy.deepcopy(tree)
        mutation_desc: str | None = None

        # Locate corresponding node in cloned tree by line and type
        target_node = None
        for n in ast.walk(mutated_tree):
            if getattr(n, "lineno", None) == target_candidate.line_no:
                if type(n).__name__ == target_candidate.node_type or (
                    target_candidate.node_type == "IfCondition" and isinstance(n, (ast.Compare, ast.Name, ast.UnaryOp))
                ):
                    target_node = n
                    break

        if target_node is None:
            return None, None

        # Apply targeted mutation based on node type
        if isinstance(target_node, ast.Compare) and target_node.ops:
            old_op_cls = type(target_node.ops[0])
            if old_op_cls in self.COMPARE_SWAPS:
                new_op_cls = self.rng.choice(self.COMPARE_SWAPS[old_op_cls])
                target_node.ops[0] = new_op_cls()
                mutation_desc = f"CompareOperatorSwap: {old_op_cls.__name__} -> {new_op_cls.__name__} at line {target_candidate.line_no}"

        elif isinstance(target_node, ast.BinOp):
            old_op_cls = type(target_node.op)
            if old_op_cls in self.BINOP_SWAPS:
                new_op_cls = self.rng.choice(self.BINOP_SWAPS[old_op_cls])
                target_node.op = new_op_cls()
                mutation_desc = f"BinOpOperatorSwap: {old_op_cls.__name__} -> {new_op_cls.__name__} at line {target_candidate.line_no}"

        elif isinstance(target_node, ast.Constant) and isinstance(target_node.value, bool):
            old_val = target_node.value
            target_node.value = not old_val
            mutation_desc = f"BoolFlip: {old_val} -> {target_node.value} at line {target_candidate.line_no}"

        elif isinstance(target_node, ast.Constant) and isinstance(target_node.value, int):
            delta = self.rng.choice([-1, 1])
            old_val = target_node.value
            target_node.value = old_val + delta
            mutation_desc = f"ConstantOffset: {old_val} -> {target_node.value} at line {target_candidate.line_no}"

        elif isinstance(target_node, ast.Subscript):
            sl = target_node.slice
            if isinstance(sl, ast.Constant) and sl.value in (0, 1):
                old_val = sl.value
                sl.value = 1 - old_val
                mutation_desc = f"IndexFlip: {old_val} -> {sl.value} at line {target_candidate.line_no}"

        elif isinstance(target_node, ast.UnaryOp) and isinstance(target_node.op, ast.USub):
            if isinstance(target_node.operand, ast.Constant) and target_node.operand.value == 1:
                target_node.op = ast.UAdd()
                target_node.operand.value = 0
                mutation_desc = f"NegOneToZero at line {target_candidate.line_no}"

        elif isinstance(target_node, ast.Call):
            fname = ""
            if isinstance(target_node.func, ast.Attribute):
                fname = target_node.func.attr
            elif isinstance(target_node.func, ast.Name):
                fname = target_node.func.id
            if fname == "pop" and target_node.args:
                target_node.args[0] = ast.Constant(value=0)
                mutation_desc = f"PopToFront at line {target_candidate.line_no}"

        elif isinstance(target_node, ast.Pass):
            for parent in ast.walk(mutated_tree):
                if isinstance(parent, ast.If) and parent.body and parent.body[0] is target_node:
                    if (
                        isinstance(parent.test, ast.Compare)
                        and parent.test.ops
                        and isinstance(parent.test.ops[0], ast.In)
                    ):
                        key = parent.test.left
                        seq = parent.test.comparators[0]
                        parent.body = [
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(value=seq, attr="remove", ctx=ast.Load()),
                                    args=[key],
                                    keywords=[],
                                )
                            ),
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(value=seq, attr="append", ctx=ast.Load()),
                                    args=[key],
                                    keywords=[],
                                )
                            ),
                        ]
                        mutation_desc = f"HitMoveToEnd at line {target_candidate.line_no}"
                    break

        elif isinstance(target_node, ast.Assign) and target_node.value is not None:
            if isinstance(target_node.value, ast.Subscript):
                target_node.value = ast.Call(
                    func=ast.Name(id="int", ctx=ast.Load()),
                    args=[target_node.value],
                    keywords=[],
                )
                mutation_desc = f"IntWrap at line {target_candidate.line_no}"

        elif isinstance(target_node, ast.Return) and target_node.value:
            # Null guard / fallback wrap
            old_val = target_node.value
            target_node.value = ast.BinOp(
                left=old_val,
                op=ast.Add(),
                right=ast.Constant(value=1),
            )
            mutation_desc = f"ReturnAdjustment: +1 offset at line {target_candidate.line_no}"

        if mutation_desc:
            ast.fix_missing_locations(mutated_tree)
            from .ast_guard import is_ast_change_safe
            safe, _ = is_ast_change_safe(mutated_tree)
            if safe:
                return mutated_tree, mutation_desc

        return None, None
