"""Intelligent, Semantics-Aware Subtree Crossover for RealASTGenome."""
from __future__ import annotations

import ast
import copy
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .genome import RealASTGenome


class IntelligentCrossover:
    """Performs type-compatible, scope-checked subtree crossover between two RealASTGenomes."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def crossover(
        self, parent_a: RealASTGenome, parent_b: RealASTGenome
    ) -> tuple[RealASTGenome, RealASTGenome]:
        compatible_pairs = self._find_compatible_pairs(parent_a, parent_b)
        if not compatible_pairs:
            return parent_a.clone(), parent_b.clone()

        node_a, node_b = self.rng.choice(compatible_pairs)

        tree_a = copy.deepcopy(parent_a.tree)
        tree_b = copy.deepcopy(parent_b.tree)

        new_tree_a = self._swap_node(tree_a, node_a, node_b)
        new_tree_b = self._swap_node(tree_b, node_b, node_a)

        child_a = self._validate(new_tree_a, parent_a) or parent_a.clone()
        child_b = self._validate(new_tree_b, parent_b) or parent_b.clone()

        return child_a, child_b

    def _find_compatible_pairs(
        self, parent_a: RealASTGenome, parent_b: RealASTGenome
    ) -> list[tuple[ast.AST, ast.AST]]:
        pairs: list[tuple[ast.AST, ast.AST]] = []

        nodes_a = [
            n for n in ast.walk(parent_a.tree)
            if isinstance(n, (ast.BinOp, ast.Compare, ast.Constant, ast.Call, ast.Name))
        ]
        nodes_b = [
            n for n in ast.walk(parent_b.tree)
            if isinstance(n, (ast.BinOp, ast.Compare, ast.Constant, ast.Call, ast.Name))
        ]

        for na in nodes_a:
            meta_a = parent_a.metadata_cache.get(id(na))
            if meta_a and meta_a.is_critical:
                continue
            type_a = parent_a.type_info.get_type(na)
            scope_a = meta_a.scope_id if meta_a else "global"
            visible_syms_a = {s.name for s in parent_a.symbol_table.get_all_symbols_in_scope(scope_a)}

            for nb in nodes_b:
                meta_b = parent_b.metadata_cache.get(id(nb))
                if meta_b and meta_b.is_critical:
                    continue
                type_b = parent_b.type_info.get_type(nb)

                # 1. Type compatibility check
                is_type_ok = False
                if type_a != "unknown" and type_b != "unknown":
                    is_type_ok = parent_a.type_info.is_type_compatible(na, nb)
                elif type(na) is type(nb):
                    is_type_ok = True

                if not is_type_ok:
                    continue

                # 2. Scope compatibility check (any variables in nb must exist in scope_a)
                scope_b = meta_b.scope_id if meta_b else "global"
                visible_syms_b = {s.name for s in parent_b.symbol_table.get_all_symbols_in_scope(scope_b)}

                nb_vars = {n.id for n in ast.walk(nb) if isinstance(n, ast.Name)}
                if not nb_vars.issubset(visible_syms_a):
                    continue

                na_vars = {n.id for n in ast.walk(na) if isinstance(n, ast.Name)}
                if not na_vars.issubset(visible_syms_b):
                    continue

                pairs.append((na, nb))

        return pairs

    def _swap_node(self, target_tree: ast.AST, old_node: ast.AST, new_node: ast.AST) -> ast.AST:
        class Replacer(ast.NodeTransformer):
            def __init__(self, old_n: ast.AST, new_n: ast.AST):
                self.old_n = old_n
                self.new_n = new_n
                self.replaced = False

            def generic_visit(self, node: ast.AST):
                if not self.replaced and ast.dump(node) == ast.dump(self.old_n):
                    self.replaced = True
                    return copy.deepcopy(self.new_n)
                return super().generic_visit(node)

        replacer = Replacer(old_node, new_node)
        new_tree = replacer.visit(target_tree)
        ast.fix_missing_locations(new_tree)
        return new_tree

    def _validate(self, tree: ast.AST, parent: RealASTGenome) -> RealASTGenome | None:
        try:
            code = ast.unparse(tree)
            compile(code, "<crossover>", "exec")
            from ..ast_guard import is_ast_change_safe
            safe, _ = is_ast_change_safe(tree)
            if not safe:
                return None
            from .genome import RealASTGenome
            return RealASTGenome(tree=tree, source_code=code, language=parent.language)
        except Exception:
            return None
