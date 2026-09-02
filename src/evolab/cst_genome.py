"""
cst_genome.py — Concrete Syntax Tree (CST) Genome representation using LibCST.

Unlike standard AST (ast.py / ast.unparse()), which discards comments, blank lines,
and fine-grained whitespace formatting, CSTGenome builds upon LibCST to maintain
lossless representation of the target source code.

Key architectural features:
  - 100% comment and formatting preservation across mutations.
  - Semantic distance metric filtering: excludes Pure Formatting Nodes
    (SimpleWhitespace, TrailingWhitespace, EmptyLine, Comment) so that
    layout differences do not distort speciation or genetic distance.
  - Granular CSTTransformers for constants, binary operations, comparisons,
    and boolean conditions.
  - Full compliance with the EvolabGenome contract.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import random
from typing import Any

import libcst as cst

from .genome import EvolabGenome

# Formatting and layout node types to exclude from semantic distance and MAP-Elites descriptors
FORMATTING_NODE_TYPES: frozenset[str] = frozenset({
    "SimpleWhitespace",
    "TrailingWhitespace",
    "EmptyLine",
    "Comment",
    "Newline",
    "ParenthesizedWhitespace",
})


class CSTSemanticNodeCollector(cst.CSTVisitor):
    """Walks a LibCST module and counts semantic vs formatting nodes."""

    def __init__(self) -> None:
        self.semantic_counts: Counter[str] = Counter()
        self.comment_count: int = 0

    def on_visit(self, node: cst.CSTNode) -> bool:
        t = type(node).__name__
        if t == "Comment":
            self.comment_count += 1
        elif t not in FORMATTING_NODE_TYPES:
            self.semantic_counts[t] += 1
        return True


class CSTConstantMutator(cst.CSTTransformer):
    """Mutates literals (booleans, integers) while leaving surrounding comments and whitespace untouched."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.mutated = False

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        if not self.mutated and original_node.value in ("True", "False") and self.rng.random() < 0.6:
            self.mutated = True
            new_val = "False" if original_node.value == "True" else "True"
            return updated_node.with_changes(value=new_val)
        return updated_node

    def leave_Integer(self, original_node: cst.Integer, updated_node: cst.Integer) -> cst.Integer:
        if not self.mutated and self.rng.random() < 0.5:
            try:
                val = int(original_node.value)
                delta = self.rng.choice([-1, 1, 10, -10])
                new_val = max(0, val + delta)
                self.mutated = True
                return updated_node.with_changes(value=str(new_val))
            except ValueError:
                pass
        return updated_node


class CSTComparisonMutator(cst.CSTTransformer):
    """Mutates comparison operators (<, <=, >, >=, ==, !=) preserving node layout."""

    SWAPS: dict[type[cst.BaseComparisonOp], list[type[cst.BaseComparisonOp]]] = {
        cst.Equal: [cst.NotEqual, cst.LessThanEqual, cst.GreaterThanEqual],
        cst.NotEqual: [cst.Equal],
        cst.LessThan: [cst.LessThanEqual, cst.GreaterThan, cst.Equal],
        cst.LessThanEqual: [cst.LessThan, cst.GreaterThanEqual, cst.Equal],
        cst.GreaterThan: [cst.GreaterThanEqual, cst.LessThan, cst.Equal],
        cst.GreaterThanEqual: [cst.GreaterThan, cst.LessThanEqual, cst.Equal],
    }

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.mutated = False

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.Comparison:
        if not self.mutated and updated_node.comparisons:
            target_idx = self.rng.randrange(len(updated_node.comparisons))
            target_comp = updated_node.comparisons[target_idx]
            op_type = type(target_comp.operator)
            if op_type in self.SWAPS and self.rng.random() < 0.7:
                new_op_cls = self.rng.choice(self.SWAPS[op_type])
                new_comp = target_comp.with_changes(operator=new_op_cls())
                new_comparisons = list(updated_node.comparisons)
                new_comparisons[target_idx] = new_comp
                self.mutated = True
                return updated_node.with_changes(comparisons=new_comparisons)
        return updated_node


class CSTBinaryOpMutator(cst.CSTTransformer):
    """Mutates binary arithmetic operators (+, -, *, /) preserving layout."""

    SWAPS: dict[type[cst.BaseBinaryOp], list[type[cst.BaseBinaryOp]]] = {
        cst.Add: [cst.Subtract, cst.Multiply],
        cst.Subtract: [cst.Add, cst.Multiply],
        cst.Multiply: [cst.Add, cst.FloorDivide, cst.Divide],
        cst.FloorDivide: [cst.Multiply, cst.Divide],
        cst.Divide: [cst.Multiply, cst.FloorDivide],
    }

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.mutated = False

    def leave_BinaryOperation(
        self, original_node: cst.BinaryOperation, updated_node: cst.BinaryOperation
    ) -> cst.BinaryOperation:
        if not self.mutated:
            op_type = type(updated_node.operator)
            if op_type in self.SWAPS and self.rng.random() < 0.6:
                new_op_cls = self.rng.choice(self.SWAPS[op_type])
                self.mutated = True
                return updated_node.with_changes(operator=new_op_cls())
        return updated_node


class CSTGenome(EvolabGenome):
    """Lossless Concrete Syntax Tree (LibCST) representation of a Python module.

    Maintains every single comment, newline, and indentation level throughout
    evolutionary search and mutation.
    """

    def __init__(self, source_code: str | cst.Module) -> None:
        if isinstance(source_code, str):
            self._code = source_code
            self._tree = cst.parse_module(source_code)
        elif isinstance(source_code, cst.Module):
            self._tree = source_code
            self._code = source_code.code
        else:
            raise TypeError(f"source_code must be str or cst.Module, got {type(source_code)}")

        self._cached_fingerprint: str | None = None
        self._cached_descriptors: dict[str, float | int | str] | None = None

    @property
    def tree(self) -> cst.Module:
        return self._tree

    @property
    def code(self) -> str:
        return self._code

    def to_code(self) -> str:
        """Returns the full lossless Python source code."""
        return self._code

    def clone(self) -> CSTGenome:
        """Creates an independent deep clone of the CST genome."""
        return CSTGenome(self._code)

    def fingerprint(self) -> str:
        """Computes deterministic SHA-256 fingerprint over the lossless source code."""
        if self._cached_fingerprint is None:
            self._cached_fingerprint = hashlib.sha256(self._code.encode("utf-8")).hexdigest()
        return self._cached_fingerprint

    def describe(self) -> dict[str, float | int | str]:
        """Calculates behavioral descriptors, filtering out layout/formatting artifacts."""
        if self._cached_descriptors is None:
            collector = CSTSemanticNodeCollector()
            self._tree.visit(collector)
            semantic_total = sum(collector.semantic_counts.values())
            self._cached_descriptors = {
                "semantic_node_count": semantic_total,
                "comment_count": collector.comment_count,
                "line_count": len(self._code.splitlines()),
                "char_count": len(self._code),
                "unique_semantic_types": len(collector.semantic_counts),
            }
        return self._cached_descriptors

    def distance_to(self, other: EvolabGenome) -> float:
        """Computes semantic genetic distance against another genome.

        Formatting nodes (newlines, spacing, comments) are completely excluded from distance
        calculation to preserve speciation integrity.
        """
        if self.fingerprint() == other.fingerprint():
            return 0.0

        if not isinstance(other, CSTGenome):
            return 10.0

        col_self = CSTSemanticNodeCollector()
        self._tree.visit(col_self)

        col_other = CSTSemanticNodeCollector()
        other._tree.visit(col_other)

        # Symmetric difference of semantic node counts
        all_keys = set(col_self.semantic_counts.keys()) | set(col_other.semantic_counts.keys())
        diff_sum = sum(abs(col_self.semantic_counts[k] - col_other.semantic_counts[k]) for k in all_keys)
        total_nodes = max(1, sum(col_self.semantic_counts.values()) + sum(col_other.semantic_counts.values()))

        normalized_distance = diff_sum / total_nodes
        return round(float(normalized_distance), 4)

    def serialize(self) -> dict[str, Any]:
        """Serializes the genome into a JSON-compatible dictionary."""
        return {
            "type": "CSTGenome",
            "fingerprint": self.fingerprint(),
            "describe": self.describe(),
            "code": self._code,
        }

    def mutate(self, rng: random.Random | None = None, **kwargs: Any) -> CSTGenome:
        """Applies a randomly selected layout-preserving CST mutation."""
        r = rng or random.Random()
        mutator_classes = [CSTConstantMutator, CSTComparisonMutator, CSTBinaryOpMutator]
        r.shuffle(mutator_classes)

        for mutator_cls in mutator_classes:
            mutator = mutator_cls(rng=r)
            new_tree = self._tree.visit(mutator)
            if mutator.mutated:
                return CSTGenome(new_tree)

        # Fallback: force mutation
        fallback_mutator = CSTConstantMutator(rng=r)
        new_tree = self._tree.visit(fallback_mutator)
        return CSTGenome(new_tree)

    def crossover(self, other: EvolabGenome, rng: random.Random | None = None) -> CSTGenome:
        """Exchanges compatible statements between two CST modules."""
        if not isinstance(other, CSTGenome):
            return self.clone()

        r = rng or random.Random()
        body_self = list(self._tree.body)
        body_other = list(other._tree.body)

        if len(body_self) > 1 and len(body_other) > 1 and r.random() < 0.5:
            idx_self = r.randrange(len(body_self))
            idx_other = r.randrange(len(body_other))
            new_body = list(body_self)
            new_body[idx_self] = body_other[idx_other]
            try:
                new_module = self._tree.with_changes(body=new_body)
                return CSTGenome(new_module)
            except Exception:
                pass

        return self.clone() if r.random() < 0.5 else other.clone()
