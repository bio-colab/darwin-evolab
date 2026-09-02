"""
AST Semantic Guard: Pre-execution semantic safety gate for darwin-evolab.
Validates scope discipline, variable definitions, function arity, and basic type sanity
before submitting mutated code to expensive sandbox runners.
"""
from __future__ import annotations

import ast
import builtins

# Full set of standard Python built-in identifiers
BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))

# Additional standard module level built-ins
RESERVED_RUNTIME_NAMES: frozenset[str] = frozenset({
    "__name__", "__file__", "__doc__", "__package__", "__annotations__", "__spec__"
})


class Scope:
    """Represents a lexical scope tracking defined variables, conditional branch definitions, and inner closures."""

    def __init__(self, name: str = "global", parent: Scope | None = None, is_function: bool = False):
        self.name = name
        self.parent = parent
        self.is_function = is_function
        self.defined: set[str] = set()
        self.conditionally_defined: dict[str, int] = {}  # var_name -> line of definition
        self.nonlocals: set[str] = set()
        self.globals: set[str] = set()

    def add(self, name: str) -> None:
        self.defined.add(name)
        self.conditionally_defined.pop(name, None)

    def add_conditional(self, name: str, lineno: int) -> None:
        if name not in self.defined and name not in BUILTIN_NAMES and name not in RESERVED_RUNTIME_NAMES:
            if self.parent is None or not self.parent.is_defined(name):
                self.conditionally_defined[name] = lineno

    def is_defined(self, name: str) -> bool:
        """Checks if name is defined in this scope or any enclosing lexical scope."""
        if name in self.defined or name in BUILTIN_NAMES or name in RESERVED_RUNTIME_NAMES:
            return True
        if self.parent is not None:
            return self.parent.is_defined(name)
        return False

    def is_conditional(self, name: str) -> bool:
        if name in self.conditionally_defined and name not in self.defined:
            return True
        if self.parent is not None:
            return self.parent.is_conditional(name)
        return False

    def get_conditional_line(self, name: str) -> int:
        if name in self.conditionally_defined and name not in self.defined:
            return self.conditionally_defined[name]
        if self.parent is not None:
            return self.parent.get_conditional_line(name)
        return 0


class FunctionSignature:
    """Tracks known parameter bounds for local functions to enforce arity safety."""

    def __init__(self, name: str, min_args: int, max_args: int, has_vararg: bool = False, has_kwarg: bool = False):
        self.name = name
        self.min_args = min_args
        self.max_args = max_args
        self.has_vararg = has_vararg
        self.has_kwarg = has_kwarg

    @classmethod
    def from_node(cls, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionSignature:
        args = node.args
        total_pos = len(args.posonlyargs) + len(args.args)
        num_defaults = len(args.defaults)
        min_args = total_pos - num_defaults
        max_args = total_pos
        has_vararg = args.vararg is not None
        has_kwarg = args.kwarg is not None
        return cls(node.name, min_args, max_args, has_vararg, has_kwarg)


class ASTSemanticValidator(ast.NodeVisitor):
    """Deep AST semantic checker rejecting ill-formed mutations before execution."""

    def __init__(self):
        self.violations: list[str] = []
        self.global_scope = Scope("global")
        self.current_scope: Scope = self.global_scope
        self.functions: dict[str, FunctionSignature] = {}

    def push_scope(self, name: str, is_function: bool = False) -> Scope:
        new_scope = Scope(name, parent=self.current_scope, is_function=is_function)
        self.current_scope = new_scope
        return new_scope

    def pop_scope(self) -> Scope:
        if self.current_scope.parent is not None:
            self.current_scope = self.current_scope.parent
        return self.current_scope

    # --- Definition Gathering ---

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.current_scope.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*":
                # Star import: allow everything gracefully
                pass
            else:
                name = alias.asname or alias.name
                self.current_scope.add(name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        for name in node.names:
            self.current_scope.globals.add(name)
            self.global_scope.add(name)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        for name in node.names:
            self.current_scope.nonlocals.add(name)

    def visit_Assign(self, node: ast.Assign):
        # Visit value first to catch load of undefined variable in assignment value
        self.visit(node.value)
        for target in node.targets:
            self._register_target(target)

    def visit_AugAssign(self, node: ast.AugAssign):
        # In x += 1, x must already be defined
        self.visit(node.target)
        self.visit(node.value)
        self._register_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr):
        self.visit(node.value)
        self._register_target(node.target)

    def _register_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.current_scope.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._register_target(elt)

    def visit_For(self, node: ast.For):
        self.visit(node.iter)
        self._register_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.visit(node.iter)
        self._register_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._register_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._register_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.type:
            self.visit(node.type)
        if node.name:
            self.current_scope.add(node.name)
        for stmt in node.body:
            self.visit(stmt)

    # --- Functions & Classes ---

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Register function name in current scope
        self.current_scope.add(node.name)
        # Store signature for local arity checking
        sig = FunctionSignature.from_node(node)
        self.functions[node.name] = sig

        # Evaluate decorators in outer scope
        for d in node.decorator_list:
            self.visit(d)
        if node.returns:
            self.visit(node.returns)

        # Enter function inner scope
        self.push_scope(node.name, is_function=True)
        # Add parameter names
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for a in all_args:
            self.current_scope.add(a.arg)
        if node.args.vararg:
            self.current_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.current_scope.add(node.args.kwarg.arg)

        # Process function body
        for stmt in node.body:
            self.visit(stmt)

        self.pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)  # type: ignore

    def visit_ClassDef(self, node: ast.ClassDef):
        self.current_scope.add(node.name)
        for base in node.bases:
            self.visit(base)
        for d in node.decorator_list:
            self.visit(d)

        self.push_scope(node.name, is_function=False)
        for stmt in node.body:
            self.visit(stmt)
        self.pop_scope()

    # --- Comprehensions & Lambdas ---

    def visit_ListComp(self, node: ast.ListComp):
        self._visit_comp(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp):
        self._visit_comp(node.generators, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        self._visit_comp(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp):
        self._visit_comp(node.generators, [node.key, node.value])

    def _visit_comp(self, generators: list[ast.comprehension], elts: list[ast.AST]) -> None:
        self.push_scope("comprehension", is_function=False)
        for gen in generators:
            self.visit(gen.iter)
            self._register_target(gen.target)
            for if_expr in gen.ifs:
                self.visit(if_expr)
        for elt in elts:
            self.visit(elt)
        self.pop_scope()

    def visit_Lambda(self, node: ast.Lambda):
        self.push_scope("lambda", is_function=True)
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for a in all_args:
            self.current_scope.add(a.arg)
        if node.args.vararg:
            self.current_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.current_scope.add(node.args.kwarg.arg)
        self.visit(node.body)
        self.pop_scope()

    # --- Path-Sensitive Branching & Reaching Definitions ---

    def visit_If(self, node: ast.If):
        self.visit(node.test)

        before_defs = set(self.current_scope.defined)

        for stmt in node.body:
            self.visit(stmt)
        body_new_defs = set(self.current_scope.defined) - before_defs

        if node.orelse:
            self.current_scope.defined = set(before_defs)
            for stmt in node.orelse:
                self.visit(stmt)
            else_new_defs = set(self.current_scope.defined) - before_defs

            # Defs in both branches become definitely defined
            both_defs = body_new_defs & else_new_defs
            either_defs = body_new_defs ^ else_new_defs

            self.current_scope.defined = before_defs | both_defs
            for var in either_defs:
                self.current_scope.add_conditional(var, getattr(node, "lineno", 1))
        else:
            self.current_scope.defined = set(before_defs)
            for var in body_new_defs:
                self.current_scope.add_conditional(var, getattr(node, "lineno", 1))

    def visit_Try(self, node: ast.Try):
        before_defs = set(self.current_scope.defined)
        for stmt in node.body:
            self.visit(stmt)
        body_new_defs = set(self.current_scope.defined) - before_defs

        for handler in node.handlers:
            self.visit(handler)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)

        self.current_scope.defined = set(before_defs)
        for var in body_new_defs:
            self.current_scope.add_conditional(var, getattr(node, "lineno", 1))

    # --- Name Resolution & Semantic Checks ---

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, (ast.Load, ast.Del)):
            line = getattr(node, "lineno", 0)
            if self.current_scope.is_conditional(node.id):
                def_line = self.current_scope.get_conditional_line(node.id)
                self.violations.append(
                    f"ConditionalDefinitionError: variable '{node.id}' is defined conditionally in branch (line {def_line}) but loaded without guaranteed initialization on all paths (line {line})"
                )
            elif not self.current_scope.is_defined(node.id):
                self.violations.append(
                    f"UndefinedVariable: '{node.id}' is used before definition in scope '{self.current_scope.name}' (line {line})"
                )
        self.generic_visit(node)

    # --- Arity & Invocation Checks ---

    def visit_Call(self, node: ast.Call):
        line = getattr(node, "lineno", 0)

        # Check 1: Calling non-callable literal
        if isinstance(node.func, ast.Constant):
            self.violations.append(
                f"NonCallableInvocation: attempting to call literal constant '{node.func.value}' at line {line}"
            )
        elif isinstance(node.func, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
            self.violations.append(
                f"NonCallableInvocation: attempting to call collection literal '{type(node.func).__name__}' at line {line}"
            )

        # Check 2: Arity check for known local functions
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            sig = self.functions[node.func.id]
            # Count positional arguments (non-keyword)
            pos_args_count = len(node.args)
            if not sig.has_vararg and pos_args_count > sig.max_args:
                self.violations.append(
                    f"ArityMismatch: function '{sig.name}' accepts at most {sig.max_args} positional args, but got {pos_args_count} at line {line}"
                )
            elif pos_args_count < sig.min_args:
                # Check if missing arguments are provided as keywords
                kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
                total_supplied = pos_args_count + len(kw_names)
                if total_supplied < sig.min_args:
                    self.violations.append(
                        f"ArityMismatch: function '{sig.name}' requires at least {sig.min_args} arguments, but got {total_supplied} at line {line}"
                    )

        self.generic_visit(node)

    # --- Type Sanity Checks on Operations ---

    def visit_BinOp(self, node: ast.BinOp):
        line = getattr(node, "lineno", 0)
        # Check invalid binary operations with None
        if isinstance(node.left, ast.Constant) and node.left.value is None:
            self.violations.append(
                f"InvalidOperation: unsupported operand '{type(node.op).__name__}' on None at line {line}"
            )
        elif isinstance(node.right, ast.Constant) and node.right.value is None:
            self.violations.append(
                f"InvalidOperation: unsupported operand '{type(node.op).__name__}' with None at line {line}"
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        line = getattr(node, "lineno", 0)
        # Subscripting numbers or booleans directly
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float, bool)):
            self.violations.append(
                f"InvalidSubscript: object of type '{type(node.value.value).__name__}' is not subscriptable at line {line}"
            )
        self.generic_visit(node)


class ASTSemanticGuard:
    """Pre-execution safety gate validating that proposed AST mutations are semantically well-formed."""

    def __init__(self):
        pass

    def validate(self, tree_or_code: ast.AST | str) -> tuple[bool, list[str]]:
        """Validates AST semantics. Returns (is_valid, list_of_violations)."""
        if isinstance(tree_or_code, str):
            try:
                tree = ast.parse(tree_or_code)
            except SyntaxError as exc:
                return False, [f"SyntaxError: {exc}"]
        else:
            tree = tree_or_code

        validator = ASTSemanticValidator()
        validator.visit(tree)
        return len(validator.violations) == 0, validator.violations


def is_ast_change_safe(new_tree_or_code: ast.AST | str, old_tree_or_code: ast.AST | str | None = None) -> tuple[bool, list[str]]:
    """Convenience function checking whether an AST mutation is semantically safe."""
    guard = ASTSemanticGuard()
    return guard.validate(new_tree_or_code)
