"""Deep Semantic Analyzer for RealASTGenome with Graded Criticality and Dependency Analysis."""
from __future__ import annotations

import ast

from .scope import SymbolTable
from .types import CriticalityLevel, NodeMetadata, NodeType, Symbol, TypeInfo


class DependencyAnalyzer(ast.NodeVisitor):
    """Analyzes variable data-flow dependency chains across assignments."""

    def __init__(self):
        self.dependencies: dict[str, set[str]] = {}  # var -> set of vars it depends on
        self.dependents: dict[str, set[str]] = {}    # var -> set of vars that depend on it

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                used_vars = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                self.dependencies[var_name] = used_vars
                for dep in used_vars:
                    if dep not in self.dependents:
                        self.dependents[dep] = set()
                    self.dependents[dep].add(var_name)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            used_vars = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            if var_name not in self.dependencies:
                self.dependencies[var_name] = set()
            self.dependencies[var_name].update(used_vars)
            for dep in used_vars:
                if dep not in self.dependents:
                    self.dependents[dep] = set()
                self.dependents[dep].add(var_name)
        self.generic_visit(node)

    def get_criticality_score(self, var_name: str) -> float:
        """Calculates dependency impact: higher score means more downstream dependents rely on this var."""
        if var_name not in self.dependents:
            return 0.0
        direct = len(self.dependents[var_name])
        indirect = 0
        visited = set()
        queue = list(self.dependents.get(var_name, set()))
        while queue:
            v = queue.pop(0)
            if v not in visited:
                visited.add(v)
                indirect += 1
                queue.extend(self.dependents.get(v, set()))
        return direct + (indirect * 0.5)


class DeepAnalyzer(ast.NodeVisitor):
    """Performs deep semantic analysis, graded criticality tagging, and data-flow dependency tracking."""

    def __init__(self):
        self.metadata: dict[int, NodeMetadata] = {}
        self.symbol_table = SymbolTable()
        self.type_info = TypeInfo()
        self.dep_analyzer = DependencyAnalyzer()
        self.current_scope = "global"
        self.current_depth = 0
        self.control_stack: list[str] = []
        self.data_dependencies: list[str] = []

    def analyze(self, tree: ast.AST) -> tuple[dict[int, NodeMetadata], SymbolTable, TypeInfo]:
        self.dep_analyzer.visit(tree)
        self.visit(tree)
        return self.metadata, self.symbol_table, self.type_info

    def visit_FunctionDef(self, node: ast.FunctionDef):
        parent_scope = self.current_scope
        func_scope = f"func:{node.name}"
        self.symbol_table.set_parent_scope(func_scope, parent_scope)

        self.symbol_table.add_symbol(Symbol(
            name=node.name,
            scope=parent_scope,
            symbol_type="function",
            inferred_type="function",
            line_defined=getattr(node, "lineno", 0),
        ))

        for arg in node.args.args:
            inferred_t = "unknown"
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                inferred_t = arg.annotation.id
            self.symbol_table.add_symbol(Symbol(
                name=arg.arg,
                scope=func_scope,
                symbol_type="parameter",
                inferred_type=inferred_t,
                line_defined=getattr(node, "lineno", 0),
            ))

        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.FUNCTION_DEF,
            criticality=CriticalityLevel.IMMUTABLE,
            depth=self.current_depth,
            scope_id=parent_scope,
            is_critical=True,
            complexity_score=2.0,
            mutation_probability=0.02,
        )

        self.current_scope = func_scope
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1
        self.current_scope = parent_scope

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)  # type: ignore

    def visit_ClassDef(self, node: ast.ClassDef):
        parent_scope = self.current_scope
        class_scope = f"class:{node.name}"
        self.symbol_table.set_parent_scope(class_scope, parent_scope)

        self.symbol_table.add_symbol(Symbol(
            name=node.name,
            scope=parent_scope,
            symbol_type="class",
            inferred_type="type",
            line_defined=getattr(node, "lineno", 0),
        ))

        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.DECLARATION,
            criticality=CriticalityLevel.IMMUTABLE,
            depth=self.current_depth,
            scope_id=parent_scope,
            is_critical=True,
            complexity_score=3.0,
            mutation_probability=0.02,
        )

        self.current_scope = class_scope
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1
        self.current_scope = parent_scope

    def visit_Assign(self, node: ast.Assign):
        inferred = self.type_info.infer_type(node.value, lambda n: self.symbol_table.lookup(n, self.current_scope))
        dep_score = 0.0
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.symbol_table.add_symbol(Symbol(
                    name=target.id,
                    scope=self.current_scope,
                    symbol_type="variable",
                    inferred_type=inferred,
                    line_defined=getattr(node, "lineno", 0),
                ))
                score = self.dep_analyzer.get_criticality_score(target.id)
                dep_score = max(dep_score, score)

        # Scale mutation probability inversely to dependency chain length
        mut_prob = max(0.1, 0.45 / (1.0 + dep_score * 0.4))

        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.STATEMENT,
            criticality=CriticalityLevel.SEMANTIC,
            depth=self.current_depth,
            scope_id=self.current_scope,
            data_flow_deps=list(self.data_dependencies),
            control_deps=list(self.control_stack),
            is_critical=False,
            complexity_score=1.2,
            mutation_probability=mut_prob,
            dependency_score=dep_score,
        )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        inferred = "unknown"
        if isinstance(node.annotation, ast.Name):
            inferred = node.annotation.id
        elif node.value:
            inferred = self.type_info.infer_type(node.value, lambda n: self.symbol_table.lookup(n, self.current_scope))

        dep_score = 0.0
        if isinstance(node.target, ast.Name):
            self.symbol_table.add_symbol(Symbol(
                name=node.target.id,
                scope=self.current_scope,
                symbol_type="variable",
                inferred_type=inferred,
                line_defined=getattr(node, "lineno", 0),
            ))
            dep_score = self.dep_analyzer.get_criticality_score(node.target.id)

        mut_prob = max(0.1, 0.45 / (1.0 + dep_score * 0.4))

        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.STATEMENT,
            criticality=CriticalityLevel.SEMANTIC,
            depth=self.current_depth,
            scope_id=self.current_scope,
            control_deps=list(self.control_stack),
            is_critical=False,
            complexity_score=1.2,
            mutation_probability=mut_prob,
            dependency_score=dep_score,
        )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        dep_score = 0.0
        if isinstance(node.target, ast.Name):
            dep_score = self.dep_analyzer.get_criticality_score(node.target.id)

        mut_prob = max(0.15, 0.45 / (1.0 + dep_score * 0.3))

        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.STATEMENT,
            criticality=CriticalityLevel.SEMANTIC,
            depth=self.current_depth,
            scope_id=self.current_scope,
            control_deps=list(self.control_stack),
            is_critical=False,
            complexity_score=1.5,
            mutation_probability=mut_prob,
            dependency_score=dep_score,
        )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        sym = self.symbol_table.lookup(node.id, self.current_scope)
        if sym and hasattr(node, "lineno"):
            sym.line_uses.append(node.lineno)

        inferred_t = sym.inferred_type if sym else "unknown"
        if inferred_t != "unknown":
            self.type_info.record_type(node, inferred_t)

        dep_score = self.dep_analyzer.get_criticality_score(node.id)
        mut_prob = max(0.15, 0.45 / (1.0 + dep_score * 0.3))

        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.EXPRESSION,
            criticality=CriticalityLevel.COSMETIC if dep_score < 1.0 else CriticalityLevel.SEMANTIC,
            depth=self.current_depth,
            scope_id=self.current_scope,
            data_flow_deps=[sym.name] if sym else [],
            control_deps=list(self.control_stack),
            is_critical=False,
            complexity_score=0.5,
            mutation_probability=mut_prob,
            dependency_score=dep_score,
        )

    def visit_Constant(self, node: ast.Constant):
        t = type(node.value).__name__
        self.type_info.record_type(node, t)
        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.EXPRESSION,
            criticality=CriticalityLevel.COSMETIC,
            depth=self.current_depth,
            scope_id=self.current_scope,
            control_deps=list(self.control_stack),
            is_critical=False,
            complexity_score=0.3,
            mutation_probability=0.55,
        )

    def visit_BinOp(self, node: ast.BinOp):
        inferred = self.type_info.infer_type(node, lambda n: self.symbol_table.lookup(n, self.current_scope))
        # String or List Add operations must not be clobbered
        is_concatenation = inferred in ("str", "list")
        mut_prob = 0.05 if is_concatenation else 0.5

        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.EXPRESSION,
            criticality=CriticalityLevel.IMMUTABLE if is_concatenation else CriticalityLevel.SEMANTIC,
            depth=self.current_depth,
            scope_id=self.current_scope,
            control_deps=list(self.control_stack),
            is_critical=is_concatenation,
            complexity_score=1.5,
            mutation_probability=mut_prob,
        )
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.type_info.infer_type(node, lambda n: self.symbol_table.lookup(n, self.current_scope))
        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.EXPRESSION,
            criticality=CriticalityLevel.SEMANTIC,
            depth=self.current_depth,
            scope_id=self.current_scope,
            control_deps=list(self.control_stack),
            is_critical=False,
            complexity_score=1.0,
            mutation_probability=0.45,
        )
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_Compare(self, node: ast.Compare):
        self.type_info.record_type(node, "bool")
        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.EXPRESSION,
            criticality=CriticalityLevel.SEMANTIC,
            depth=self.current_depth,
            scope_id=self.current_scope,
            control_deps=list(self.control_stack),
            is_critical=False,
            complexity_score=1.5,
            mutation_probability=0.45,
        )
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_Call(self, node: ast.Call):
        self.type_info.infer_type(node, lambda n: self.symbol_table.lookup(n, self.current_scope))
        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.EXPRESSION,
            criticality=CriticalityLevel.SEMANTIC,
            depth=self.current_depth,
            scope_id=self.current_scope,
            control_deps=list(self.control_stack),
            is_critical=False,
            complexity_score=1.5,
            mutation_probability=0.4,
        )
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_If(self, node: ast.If):
        cond_str = "if_condition"
        try:
            cond_str = f"if:{ast.unparse(node.test)}"
        except Exception:
            pass

        self.control_stack.append(cond_str)
        # Structural header
        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.CONTROL_FLOW,
            criticality=CriticalityLevel.STRUCTURAL,
            depth=self.current_depth,
            scope_id=self.current_scope,
            is_critical=True,
            complexity_score=2.0,
            mutation_probability=0.05,
        )
        # Mark body children as SEMANTIC with higher mutability
        for child in node.body:
            if id(child) not in self.metadata:
                self.metadata[id(child)] = NodeMetadata(
                    node_type=NodeType.STATEMENT,
                    criticality=CriticalityLevel.SEMANTIC,
                    depth=self.current_depth + 1,
                    scope_id=self.current_scope,
                    mutation_probability=0.4,
                )

        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1
        self.control_stack.pop()

    def visit_For(self, node: ast.For):
        if isinstance(node.target, ast.Name):
            self.symbol_table.add_symbol(Symbol(
                name=node.target.id,
                scope=self.current_scope,
                symbol_type="variable",
                inferred_type="int",
                line_defined=getattr(node, "lineno", 0),
            ))

        self.control_stack.append("for_loop")
        # Structural header
        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.CONTROL_FLOW,
            criticality=CriticalityLevel.STRUCTURAL,
            depth=self.current_depth,
            scope_id=self.current_scope,
            is_critical=True,
            complexity_score=2.5,
            mutation_probability=0.05,
        )
        # Body children are SEMANTIC with higher mutability
        for child in node.body:
            if id(child) not in self.metadata:
                self.metadata[id(child)] = NodeMetadata(
                    node_type=NodeType.STATEMENT,
                    criticality=CriticalityLevel.SEMANTIC,
                    depth=self.current_depth + 1,
                    scope_id=self.current_scope,
                    mutation_probability=0.4,
                )

        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1
        self.control_stack.pop()

    def visit_While(self, node: ast.While):
        self.control_stack.append("while_loop")
        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.CONTROL_FLOW,
            criticality=CriticalityLevel.STRUCTURAL,
            depth=self.current_depth,
            scope_id=self.current_scope,
            is_critical=True,
            complexity_score=3.0,
            mutation_probability=0.05,
        )
        for child in node.body:
            if id(child) not in self.metadata:
                self.metadata[id(child)] = NodeMetadata(
                    node_type=NodeType.STATEMENT,
                    criticality=CriticalityLevel.SEMANTIC,
                    depth=self.current_depth + 1,
                    scope_id=self.current_scope,
                    mutation_probability=0.4,
                )

        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1
        self.control_stack.pop()

    def visit_Return(self, node: ast.Return):
        self.metadata[id(node)] = NodeMetadata(
            node_type=NodeType.RETURN,
            criticality=CriticalityLevel.IMMUTABLE,
            depth=self.current_depth,
            scope_id=self.current_scope,
            control_deps=list(self.control_stack),
            is_critical=True,
            complexity_score=1.5,
            mutation_probability=0.05,
        )
        self.current_depth += 1
        self.generic_visit(node)
        self.current_depth -= 1

    def generic_visit(self, node: ast.AST):
        if id(node) not in self.metadata:
            n_type = NodeType.EXPRESSION if isinstance(node, ast.expr) else (
                NodeType.STATEMENT if isinstance(node, ast.stmt) else NodeType.DECLARATION
            )
            self.metadata[id(node)] = NodeMetadata(
                node_type=n_type,
                criticality=CriticalityLevel.SEMANTIC,
                depth=self.current_depth,
                scope_id=self.current_scope,
                control_deps=list(self.control_stack),
                is_critical=False,
                complexity_score=1.0,
                mutation_probability=0.35,
            )
        super().generic_visit(node)
