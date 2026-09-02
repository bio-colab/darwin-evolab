"""Intelligent, Semantic-Preserving AST Mutator with Graded Criticality and Type Awareness."""
from __future__ import annotations

import ast
import copy
import random
from typing import TYPE_CHECKING, Any

from .types import CriticalityLevel, NodeType

if TYPE_CHECKING:
    from .genome import RealASTGenome


class IntelligentMutator:
    """Applies type-aware, scope-respecting mutations that preserve program semantics."""

    COMPATIBLE_OPS: dict[str, dict[type[ast.operator], list[type[ast.operator]]]] = {
        "int": {
            ast.Add: [ast.Add, ast.Sub],
            ast.Sub: [ast.Sub, ast.Add],
            ast.Mult: [ast.Mult],
            ast.Mod: [ast.Mod],
            ast.FloorDiv: [ast.FloorDiv],
        },
        "str": {
            ast.Add: [ast.Add],  # String concatenation only! Never swap Add on strings
        },
        "list": {
            ast.Add: [ast.Add],  # List concatenation only!
        },
        "float": {
            ast.Add: [ast.Add, ast.Sub],
            ast.Mult: [ast.Mult, ast.Div],
        },
    }

    COMPATIBLE_STRING_METHODS: dict[str, list[str]] = {
        "upper": ["lower", "capitalize"],
        "lower": ["upper", "capitalize"],
        "isupper": ["islower"],
        "islower": ["isupper"],
        "strip": ["lstrip", "rstrip"],
        "lstrip": ["strip", "rstrip"],
        "rstrip": ["strip", "lstrip"],
    }

    def __init__(self, genome: RealASTGenome, rng: random.Random | None = None):
        self.genome = genome
        self.rng = rng or random.Random()

    def mutate(self) -> RealASTGenome:
        """Applies a semantically guided mutation and returns a new valid RealASTGenome."""
        candidates = self._get_mutation_candidates()
        if not candidates:
            return self.genome.clone()

        # Weighted choice based on mutation_probability
        node_ids, weights = zip(*candidates)
        selected_id = self.rng.choices(node_ids, weights=weights, k=1)[0]
        metadata = self.genome.metadata_cache[selected_id]

        if metadata.node_type == NodeType.EXPRESSION:
            return self._mutate_expression(selected_id)
        elif metadata.node_type == NodeType.CONTROL_FLOW:
            return self._mutate_control_flow(selected_id)
        elif metadata.node_type == NodeType.STATEMENT:
            return self._mutate_statement(selected_id)

        return self._mutate_expression(selected_id)

    def _get_mutation_candidates(self) -> list[tuple[int, float]]:
        candidates: list[tuple[int, float]] = []
        for node_id, meta in self.genome.metadata_cache.items():
            # Skip immutable nodes (function headers, return statements)
            if meta.criticality == CriticalityLevel.IMMUTABLE:
                continue
            prob = meta.mutation_probability
            candidates.append((node_id, max(0.01, prob)))
        return candidates

    def _mutate_expression(self, target_node_id: int) -> RealASTGenome:
        tree_copy = copy.deepcopy(self.genome.tree)
        target = self._find_node_by_id(tree_copy, target_node_id)
        if target is None:
            return self.genome.clone()

        node_type = self.genome.type_info.get_type(target)
        scope = self.genome.metadata_cache.get(target_node_id, None)
        scope_id = scope.scope_id if scope else "global"

        new_node: ast.AST | None = None

        if isinstance(target, ast.Constant):
            val = target.value
            if isinstance(val, bool):
                new_node = ast.Constant(value=not val)
            elif isinstance(val, int):
                delta = self.rng.choice([-2, -1, 1, 2])
                new_node = ast.Constant(value=val + delta)
            elif isinstance(val, float):
                factor = self.rng.choice([0.9, 1.1])
                new_node = ast.Constant(value=round(val * factor, 3))
            elif isinstance(val, str) and len(val) > 0:
                new_node = ast.Constant(value=val.swapcase())

        elif isinstance(target, ast.Name):
            # In-scope variable replacement preserving inferred type
            compatible_syms = self.genome.symbol_table.get_typed_symbols_in_scope(scope_id, node_type)
            compatible_syms = [s for s in compatible_syms if s.name != target.id]
            if compatible_syms:
                chosen_sym = self.rng.choice(compatible_syms)
                new_node = ast.Name(id=chosen_sym.name, ctx=copy.deepcopy(target.ctx))

        elif isinstance(target, ast.BinOp):
            # Type-aware BinOp mutation: protect strings and lists from unsupported operators
            ops_for_type = self.COMPATIBLE_OPS.get(node_type, self.COMPATIBLE_OPS.get("int", {}))
            op_t = type(target.op)
            if op_t in ops_for_type:
                choices = ops_for_type[op_t]
                if len(choices) > 1:
                    new_op_cls = self.rng.choice(choices)
                    new_node = ast.BinOp(left=target.left, op=new_op_cls(), right=target.right)

        elif isinstance(target, ast.Compare):
            cmp_swaps = {
                ast.Lt: ast.LtE,
                ast.LtE: ast.Lt,
                ast.Gt: ast.GtE,
                ast.GtE: ast.Gt,
                ast.Eq: ast.NotEq,
                ast.NotEq: ast.Eq,
            }
            if target.ops:
                first_op = type(target.ops[0])
                if first_op in cmp_swaps:
                    new_ops = [cmp_swaps[first_op]()] + list(target.ops[1:])
                    new_node = ast.Compare(left=target.left, ops=new_ops, comparators=target.comparators)

        elif isinstance(target, ast.Call):
            # String method mutation: upper <-> lower, isupper <-> islower
            if isinstance(target.func, ast.Attribute) and target.func.attr in self.COMPATIBLE_STRING_METHODS:
                new_attr = self.rng.choice(self.COMPATIBLE_STRING_METHODS[target.func.attr])
                new_call = copy.deepcopy(target)
                if isinstance(new_call.func, ast.Attribute):
                    new_call.func.attr = new_attr
                    new_node = new_call

        if new_node is not None:
            self._replace_node_in_tree(tree_copy, target, new_node)
            ast.fix_missing_locations(tree_copy)
            res = self._validate_tree(tree_copy)
            if res:
                return res

        return self.genome.clone()

    def _mutate_control_flow(self, target_node_id: int) -> RealASTGenome:
        tree_copy = copy.deepcopy(self.genome.tree)
        target = self._find_node_by_id(tree_copy, target_node_id)
        if target is None:
            return self.genome.clone()

        if isinstance(target, ast.If):
            choice = self.rng.choice(["negate_condition", "swap_branches"])
            if choice == "negate_condition":
                new_test = ast.UnaryOp(op=ast.Not(), operand=target.test)
                new_if = ast.If(test=new_test, body=target.body, orelse=target.orelse)
                self._replace_node_in_tree(tree_copy, target, new_if)
            elif choice == "swap_branches" and target.orelse:
                new_if = ast.If(test=target.test, body=target.orelse, orelse=target.body)
                self._replace_node_in_tree(tree_copy, target, new_if)

            ast.fix_missing_locations(tree_copy)
            res = self._validate_tree(tree_copy)
            if res:
                return res

        return self.genome.clone()

    def _mutate_statement(self, target_node_id: int) -> RealASTGenome:
        tree_copy = copy.deepcopy(self.genome.tree)
        target = self._find_node_by_id(tree_copy, target_node_id)
        if target is None:
            return self.genome.clone()

        class ParentFinder(ast.NodeVisitor):
            def __init__(self, needle: ast.AST):
                self.needle = needle
                self.parent_body: list[ast.stmt] | None = None
                self.idx: int = -1

            def generic_visit(self, node: ast.AST):
                for field_name, value in ast.iter_fields(node):
                    if isinstance(value, list) and all(isinstance(x, ast.stmt) for x in value):
                        for i, stmt in enumerate(value):
                            if stmt is self.needle or ast.dump(stmt) == ast.dump(self.needle):
                                self.parent_body = value
                                self.idx = i
                                return
                super().generic_visit(node)

        finder = ParentFinder(target)
        finder.visit(tree_copy)

        if finder.parent_body is not None and len(finder.parent_body) > 1:
            idx = finder.idx
            swap_idx = idx + 1 if idx < len(finder.parent_body) - 1 else idx - 1
            s1 = finder.parent_body[idx]
            s2 = finder.parent_body[swap_idx]

            # Avoid swapping terminators
            terminators = (ast.Return, ast.Raise, ast.Break, ast.Continue)
            if isinstance(s1, terminators) or isinstance(s2, terminators):
                return self.genome.clone()

            # Data-flow dependency check: preserve def-use dependencies to prevent UnboundLocalError
            defs_s1 = {n.id for n in ast.walk(s1) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            uses_s1 = {n.id for n in ast.walk(s1) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            defs_s2 = {n.id for n in ast.walk(s2) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            uses_s2 = {n.id for n in ast.walk(s2) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

            if (defs_s1 & uses_s2) or (defs_s2 & uses_s1):
                return self.genome.clone()

            finder.parent_body[idx], finder.parent_body[swap_idx] = s2, s1
            ast.fix_missing_locations(tree_copy)
            res = self._validate_tree(tree_copy)
            if res:
                return res

        return self.genome.clone()

    def _validate_tree(self, tree: ast.AST) -> RealASTGenome | None:
        try:
            code = ast.unparse(tree)
            compile(code, "<real_ast_mutated>", "exec")
            from ..ast_guard import is_ast_change_safe
            safe, _ = is_ast_change_safe(tree)
            if not safe:
                return None
            from .genome import RealASTGenome
            return RealASTGenome(tree=tree, source_code=code, language=self.genome.language)
        except Exception:
            return None

    def _find_node_by_id(self, tree: ast.AST, target_id: int) -> ast.AST | None:
        target_meta = self.genome.metadata_cache.get(target_id)
        if not target_meta:
            return None

        for n in ast.walk(tree):
            if id(n) == target_id:
                return n

        orig_nodes = list(ast.walk(self.genome.tree))
        copy_nodes = list(ast.walk(tree))
        for orig_n, copy_n in zip(orig_nodes, copy_nodes):
            if id(orig_n) == target_id:
                return copy_n

        return None

    def _replace_node_in_tree(self, tree: ast.AST, old_node: ast.AST, new_node: ast.AST) -> None:
        class Replacer(ast.NodeTransformer):
            def __init__(self, old_n: ast.AST, new_n: ast.AST):
                self.old_n = old_n
                self.new_n = new_n
                self.done = False

            def generic_visit(self, node: ast.AST):
                if not self.done and (node is self.old_n or ast.dump(node) == ast.dump(self.old_n)):
                    self.done = True
                    return copy.deepcopy(self.new_n)
                return super().generic_visit(node)

        replacer = Replacer(old_node, new_node)
        replacer.visit(tree)


class BalancedMutator:
    """Performs output-preserving mutations for sensitive dependency chains."""

    @staticmethod
    def mutate_preserving_output(
        genome: RealASTGenome, test_cases: list[dict], max_attempts: int = 20, rng: random.Random | None = None
    ) -> RealASTGenome:
        r = rng or random.Random()
        original_outputs: dict[str, Any] = {}
        for test in test_cases:
            try:
                original_outputs[str(test["input"])] = BalancedMutator._execute(genome.to_code(), test["input"])
            except Exception:
                return genome.clone()

        for _ in range(max_attempts):
            mutated = genome.mutate(rng=r)
            all_match = True
            for test in test_cases:
                try:
                    new_output = BalancedMutator._execute(mutated.to_code(), test["input"])
                    if new_output != original_outputs[str(test["input"])]:
                        all_match = False
                        break
                except Exception:
                    all_match = False
                    break
            if all_match and mutated.fingerprint() != genome.fingerprint():
                return mutated

        return genome.clone()

    @staticmethod
    def _execute(code: str, inputs: tuple) -> Any:
        ns: dict[str, Any] = {}
        exec(code, ns)  # nosec B102  # Controlled execution of candidate mutated function AST in local namespace.
        func = next(v for k, v in ns.items() if callable(v) and not k.startswith("_"))
        return func(*inputs)
