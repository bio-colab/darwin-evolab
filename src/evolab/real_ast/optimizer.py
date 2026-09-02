"""Optimization passes for RealASTGenome (Constant Folding, Dead Code Elimination, Branch Simplification)."""
from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .genome import RealASTGenome


class ConstantFolder(ast.NodeTransformer):
    """Folds static constant arithmetic and boolean operations."""

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            lv = node.left.value
            rv = node.right.value
            res = None
            try:
                if isinstance(node.op, ast.Add):
                    res = lv + rv
                elif isinstance(node.op, ast.Sub):
                    res = lv - rv
                elif isinstance(node.op, ast.Mult):
                    res = lv * rv
                elif isinstance(node.op, ast.FloorDiv) and rv != 0:
                    res = lv // rv
                elif isinstance(node.op, ast.Div) and rv != 0:
                    res = lv / rv
                elif isinstance(node.op, ast.Mod) and rv != 0:
                    res = lv % rv
            except Exception:
                res = None

            if res is not None:
                return ast.Constant(value=res)

        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if isinstance(node.operand, ast.Constant):
            ov = node.operand.value
            if isinstance(node.op, ast.Not):
                return ast.Constant(value=not ov)
            elif isinstance(node.op, ast.USub) and isinstance(ov, (int, float)):
                return ast.Constant(value=-ov)
            elif isinstance(node.op, ast.UAdd) and isinstance(ov, (int, float)):
                return ast.Constant(value=+ov)
        return node


class DeadCodeEliminator(ast.NodeTransformer):
    """Eliminates unreachable statements following unconditional return, raise, break, continue."""

    TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)

    def _prune_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        new_body: list[ast.stmt] = []
        for stmt in body:
            new_body.append(stmt)
            if isinstance(stmt, self.TERMINATORS):
                break
        return new_body

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        node.body = self._prune_body(node.body)
        return node

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        node.body = self._prune_body(node.body)
        if node.orelse:
            node.orelse = self._prune_body(node.orelse)
        return node

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        node.body = self._prune_body(node.body)
        return node

    def visit_While(self, node: ast.While):
        self.generic_visit(node)
        node.body = self._prune_body(node.body)
        return node


class BranchSimplifier(ast.NodeTransformer):
    """Simplifies constant conditional branches (if True: ... / if False: ...)."""

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            if bool(node.test.value) is True:
                # Replace if True with its body
                return node.body
            else:
                # Replace if False with its orelse, or empty
                return node.orelse or []
        return node


class ASTOptimizer:
    """Executes a pipeline of semantic optimization passes."""

    def optimize(self, genome: RealASTGenome) -> RealASTGenome:
        tree = copy.deepcopy(genome.tree)

        tree = ConstantFolder().visit(tree)
        tree = DeadCodeEliminator().visit(tree)
        tree = BranchSimplifier().visit(tree)
        ast.fix_missing_locations(tree)

        try:
            code = ast.unparse(tree)
            compile(code, "<optimized>", "exec")
            from .genome import RealASTGenome
            return RealASTGenome(tree=tree, source_code=code, language=genome.language)
        except Exception:
            return genome.clone()
