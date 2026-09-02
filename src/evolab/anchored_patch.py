"""
AST-Anchored Patch Contract: Structurally-anchored semantic patches for darwin-evolab.
Immune to line shifts, comment additions, and formatting changes.
Represents edits as verifiable AST contracts targeting semantic anchors.
"""
from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .ast_guard import is_ast_change_safe
from .genome import EvolabGenome


@dataclass
class NodeAnchor:
    """Semantic anchor identifying a precise target node within an AST function body."""

    function: str
    node_type: str
    left_name: str | None = None
    right_name: str | None = None
    operator: str | None = None
    occurrence_index: int = 0  # 0-indexed if multiple identical nodes exist
    statement_index: int | None = None  # 0-indexed statement position in function body
    parent_type: str | None = None  # Parent node type (e.g. 'If', 'While', 'Return')
    fuzzy_identifiers: bool = False  # If True, matches even if variables were refactored/renamed

    def serialize(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeAnchor:
        return cls(
            function=str(data.get("function", "")),
            node_type=str(data.get("node_type", "")),
            left_name=data.get("left_name"),
            right_name=data.get("right_name"),
            operator=data.get("operator"),
            occurrence_index=int(data.get("occurrence_index", 0)),
            statement_index=data.get("statement_index"),
            parent_type=data.get("parent_type"),
            fuzzy_identifiers=bool(data.get("fuzzy_identifiers", False)),
        )


@dataclass
class AnchoredHunk:
    """A semantic AST transformation targeting a specific anchor."""

    file_path: str
    operation: str  # replace_operator, replace_constant, insert_guard, wrap_with_try_except, replace_call_name, delete_statement
    anchor: NodeAnchor
    parameters: dict[str, Any] = field(default_factory=dict)
    hunk_id: str = ""

    def __post_init__(self):
        if not self.hunk_id:
            raw = f"{self.file_path}:{self.anchor.function}:{self.anchor.node_type}:{self.operation}:{json.dumps(self.parameters, sort_keys=True)}"
            self.hunk_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def serialize(self) -> dict[str, Any]:
        return {
            "hunk_id": self.hunk_id,
            "file_path": self.file_path,
            "operation": self.operation,
            "anchor": self.anchor.serialize(),
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnchoredHunk:
        return cls(
            hunk_id=data.get("hunk_id", ""),
            file_path=data["file_path"],
            operation=data["operation"],
            anchor=NodeAnchor.from_dict(data["anchor"]),
            parameters=dict(data.get("parameters", {})),
        )


class AnchoredPatchApplyError(Exception):
    """Raised when an anchored patch cannot find its target anchor or produces invalid code."""


class AnchoredASTApplier(ast.NodeTransformer):
    """Finds and transforms target AST node based on semantic anchor and operation."""

    def __init__(self, hunk: AnchoredHunk):
        self.hunk = hunk
        self.matched_count = 0
        self.applied = False
        self.in_target_func = False
        self.current_stmt_idx: int | None = None
        self.parent_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        was_in = self.in_target_func
        if node.name == self.hunk.anchor.function or not self.hunk.anchor.function:
            self.in_target_func = True
            for idx, stmt in enumerate(node.body):
                self.current_stmt_idx = idx
                self.visit(stmt)
            self.in_target_func = was_in
            self.current_stmt_idx = None
            return node
        self.generic_visit(node)
        return node

    def _matches_context(self) -> bool:
        if self.hunk.anchor.statement_index is not None and self.current_stmt_idx is not None:
            if self.current_stmt_idx != self.hunk.anchor.statement_index:
                return False
        if self.hunk.anchor.parent_type is not None:
            if self.hunk.anchor.parent_type not in self.parent_stack:
                return False
        return True

    def visit_If(self, node: ast.If):
        self.parent_stack.append("If")
        self.generic_visit(node)
        self.parent_stack.pop()
        return node

    def visit_While(self, node: ast.While):
        self.parent_stack.append("While")
        self.generic_visit(node)
        self.parent_stack.pop()
        return node

    def visit_Return(self, node: ast.Return):
        self.parent_stack.append("Return")
        self.generic_visit(node)
        self.parent_stack.pop()
        return node

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if not self.in_target_func or self.applied:
            return node

        if self.hunk.anchor.node_type == "Compare" and self._matches_context():
            left_str = ast.unparse(node.left).strip()
            right_str = ast.unparse(node.comparators[0]).strip() if node.comparators else ""
            op_name = type(node.ops[0]).__name__ if node.ops else ""

            if self.hunk.anchor.operator and self.hunk.anchor.operator != op_name:
                return node

            # Check identifier constraints unless fuzzy matching is enabled
            if not self.hunk.anchor.fuzzy_identifiers:
                if self.hunk.anchor.left_name and self.hunk.anchor.left_name != left_str:
                    return node
                if self.hunk.anchor.right_name and self.hunk.anchor.right_name != right_str:
                    return node

            if self.matched_count == self.hunk.anchor.occurrence_index:
                return self._apply_operation(node)
            self.matched_count += 1
        return node

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if not self.in_target_func or self.applied:
            return node

        if self.hunk.anchor.node_type == "BinOp" and self._matches_context():
            left_str = ast.unparse(node.left).strip()
            right_str = ast.unparse(node.right).strip()
            op_name = type(node.op).__name__

            if self.hunk.anchor.operator and self.hunk.anchor.operator != op_name:
                return node

            if not self.hunk.anchor.fuzzy_identifiers:
                if self.hunk.anchor.left_name and self.hunk.anchor.left_name != left_str:
                    return node
                if self.hunk.anchor.right_name and self.hunk.anchor.right_name != right_str:
                    return node

            if self.matched_count == self.hunk.anchor.occurrence_index:
                return self._apply_operation(node)
            self.matched_count += 1
        return node

    def visit_Constant(self, node: ast.Constant):
        self.generic_visit(node)
        if not self.in_target_func or self.applied:
            return node

        if self.hunk.anchor.node_type == "Constant" and self._matches_context():
            if self.matched_count == self.hunk.anchor.occurrence_index:
                return self._apply_operation(node)
            self.matched_count += 1
        return node

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not self.in_target_func or self.applied:
            return node

        if self.hunk.anchor.node_type == "Call" and self._matches_context():
            func_name = ast.unparse(node.func).strip()
            if not self.hunk.anchor.fuzzy_identifiers and self.hunk.anchor.left_name and self.hunk.anchor.left_name != func_name:
                return node
            if self.matched_count == self.hunk.anchor.occurrence_index:
                return self._apply_operation(node)
            self.matched_count += 1
        return node

    def _apply_operation(self, node: ast.AST) -> ast.AST:
        op = self.hunk.operation
        params = self.hunk.parameters

        if op == "replace_operator":
            new_op_name = params.get("new_op", "")
            op_cls = getattr(ast, new_op_name, None)
            if op_cls is None:
                raise AnchoredPatchApplyError(f"Unknown operator class: {new_op_name}")
            if isinstance(node, ast.Compare) and node.ops:
                node.ops[0] = op_cls()
                self.applied = True
                return node
            elif isinstance(node, ast.BinOp):
                node.op = op_cls()
                self.applied = True
                return node

        elif op == "replace_constant":
            new_val = params.get("new_value")
            node.value = new_val
            self.applied = True
            return node

        elif op == "replace_call_name":
            new_name = params.get("new_name", "")
            node.func = ast.Name(id=new_name, ctx=ast.Load())
            self.applied = True
            return node

        raise AnchoredPatchApplyError(f"Unsupported anchored operation: {op}")


def apply_anchored_hunk(tree_or_code: ast.AST | str, hunk: AnchoredHunk) -> tuple[ast.AST, str]:
    """Applies an AnchoredHunk to an AST, returns (modified_ast, modified_code)."""
    if isinstance(tree_or_code, str):
        tree = ast.parse(tree_or_code)
    else:
        tree = copy.deepcopy(tree_or_code)

    applier = AnchoredASTApplier(hunk)
    new_tree = applier.visit(tree)
    if not applier.applied:
        raise AnchoredPatchApplyError(
            f"Anchor not found: {hunk.anchor.function}::{hunk.anchor.node_type} in {hunk.file_path}"
        )

    ast.fix_missing_locations(new_tree)
    new_code = ast.unparse(new_tree)

    # Validate semantic safety
    is_safe, violations = is_ast_change_safe(new_tree)
    if not is_safe:
        raise AnchoredPatchApplyError(f"Semantic safety violation after patch: {violations}")

    return new_tree, new_code


class AnchoredPatchGenome(EvolabGenome):
    """Genome containing a sequence of structurally anchored AST hunks."""

    def __init__(self, hunks: Sequence[AnchoredHunk] | None = None):
        self.hunks: list[AnchoredHunk] = list(hunks or [])

    def fingerprint(self) -> str:
        serialized = json.dumps([h.serialize() for h in self.hunks], sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def clone(self) -> AnchoredPatchGenome:
        return AnchoredPatchGenome(hunks=copy.deepcopy(self.hunks))

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "AnchoredPatchGenome",
            "fingerprint": self.fingerprint(),
            "hunks": [h.serialize() for h in self.hunks],
        }

    def describe(self) -> dict[str, float | int | str]:
        return {
            "num_hunks": len(self.hunks),
            "files_touched": len({h.file_path for h in self.hunks}),
            "operations": ",".join(sorted({h.operation for h in self.hunks})),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.serialize()

    def apply_to(self, sources: dict[str, str]) -> dict[str, str]:
        """Applies anchored hunks to a dictionary of source files {filename: code}."""
        result = dict(sources)
        for hunk in self.hunks:
            if hunk.file_path not in result:
                raise AnchoredPatchApplyError(f"File not found in sources: {hunk.file_path}")
            _, new_code = apply_anchored_hunk(result[hunk.file_path], hunk)
            result[hunk.file_path] = new_code
        return result

    def to_unified_diff(self, original_sources: dict[str, str]) -> str:
        """Generates unified diff between original sources and applied patched sources."""
        patched_sources = self.apply_to(original_sources)
        diff_lines = []
        for path, orig_code in original_sources.items():
            new_code = patched_sources.get(path, orig_code)
            if orig_code != new_code:
                orig_split = orig_code.splitlines(keepends=True)
                new_split = new_code.splitlines(keepends=True)
                diff = difflib.unified_diff(
                    orig_split, new_split, fromfile=f"a/{path}", tofile=f"b/{path}"
                )
                diff_lines.extend(diff)
        return "".join(diff_lines)

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, AnchoredPatchGenome):
            return 1.0
        ids_a = {h.hunk_id for h in self.hunks}
        ids_b = {h.hunk_id for h in other.hunks}
        union = ids_a | ids_b
        if not union:
            return 0.0
        return 1.0 - (len(ids_a & ids_b) / len(union))

    def mutate(self, rng: random.Random | None = None, **kwargs: Any) -> AnchoredPatchGenome:
        rng = rng or random.Random()
        if not self.hunks:
            return self.clone()
        clone = self.clone()
        # Randomly alter an operator in a hunk
        idx = rng.randrange(len(clone.hunks))
        target_hunk = clone.hunks[idx]
        if target_hunk.operation == "replace_operator" and "new_op" in target_hunk.parameters:
            candidates = [op for op in ["Eq", "NotEq", "Lt", "LtE", "Gt", "GtE"] if op != target_hunk.parameters["new_op"]]
            target_hunk.parameters["new_op"] = rng.choice(candidates)
            target_hunk.hunk_id = ""
            target_hunk.__post_init__()
        return clone

    def crossover(
        self, other: EvolabGenome, rng: random.Random | None = None, **kwargs: Any
    ) -> AnchoredPatchGenome:
        rng = rng or random.Random()
        if not isinstance(other, AnchoredPatchGenome) or not other.hunks:
            return self.clone()
        combined = list({h.hunk_id: h for h in (self.hunks + other.hunks)}.values())
        k = rng.randint(1, len(combined))
        selected = rng.sample(combined, k)
        return AnchoredPatchGenome(hunks=selected)

    def __len__(self) -> int:
        return len(self.hunks)
