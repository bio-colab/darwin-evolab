"""Cross-file AST dependency analysis, call graph tracking, and synchronized multi-module mutations."""
from __future__ import annotations

import ast
import copy
import random
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

from .ast_genome import MultiFileASTGenome


@dataclass
class CallSite:
    """Represents a function or method invocation site."""

    caller_file: str
    line_number: int
    func_name: str
    is_attribute: bool
    caller_obj: str | None = None
    arg_count: int = 0
    keywords: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Represents an import statement in a file."""

    file_path: str
    imported_module: str
    imported_symbol: str | None = None  # None if `import foo`
    alias: str | None = None
    is_from: bool = False


class SymbolCollector(ast.NodeVisitor):
    """Collects top-level function, class, and global definitions from an AST."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.globals: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions[node.name] = node
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.functions[node.name] = node
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes[node.name] = node
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.globals.add(target.id)
        self.generic_visit(node)


class ImportCollector(ast.NodeVisitor):
    """Extracts all module and symbol import declarations from a file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.imports: list[ImportInfo] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    file_path=self.file_path,
                    imported_module=alias.name,
                    imported_symbol=None,
                    alias=alias.asname or alias.name,
                    is_from=False,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    file_path=self.file_path,
                    imported_module=mod,
                    imported_symbol=alias.name,
                    alias=alias.asname or alias.name,
                    is_from=True,
                )
            )


class CallSiteCollector(ast.NodeVisitor):
    """Extracts all function and method call sites with argument metrics."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.call_sites: list[CallSite] = []

    def visit_Call(self, node: ast.Call):
        func = node.func
        kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]
        if isinstance(func, ast.Name):
            self.call_sites.append(
                CallSite(
                    caller_file=self.file_path,
                    line_number=getattr(node, "lineno", 0),
                    func_name=func.id,
                    is_attribute=False,
                    arg_count=len(node.args),
                    keywords=kw_names,
                )
            )
        elif isinstance(func, ast.Attribute):
            caller_name = func.value.id if isinstance(func.value, ast.Name) else None
            self.call_sites.append(
                CallSite(
                    caller_file=self.file_path,
                    line_number=getattr(node, "lineno", 0),
                    func_name=func.attr,
                    is_attribute=True,
                    caller_obj=caller_name,
                    arg_count=len(node.args),
                    keywords=kw_names,
                )
            )
        self.generic_visit(node)


@dataclass
class CrossFileDependencyGraph:
    """Bidirectional dependency and call graph across multiple source modules."""

    file_ast: dict[str, ast.AST] = field(default_factory=dict)
    symbols: dict[str, SymbolCollector] = field(default_factory=dict)
    imports: dict[str, list[ImportInfo]] = field(default_factory=dict)
    call_sites: dict[str, list[CallSite]] = field(default_factory=dict)

    @classmethod
    def build(cls, sources: dict[str, str]) -> CrossFileDependencyGraph:
        graph = cls()
        for path, code in sources.items():
            try:
                tree = ast.parse(code, filename=path)
                graph.file_ast[path] = tree

                sym_col = SymbolCollector(path)
                sym_col.visit(tree)
                graph.symbols[path] = sym_col

                imp_col = ImportCollector(path)
                imp_col.visit(tree)
                graph.imports[path] = imp_col.imports

                call_col = CallSiteCollector(path)
                call_col.visit(tree)
                graph.call_sites[path] = call_col.call_sites
            except SyntaxError:
                continue
        return graph

    def get_module_stem(self, file_path: str) -> str:
        """Returns the module stem name without extension (e.g. 'validator.py' -> 'validator')."""
        return PurePath(file_path).stem

    def find_callers_of(self, target_file: str, func_name: str) -> list[tuple[str, CallSite]]:
        """Finds all call sites across all files that invoke target_file.func_name."""
        target_mod = self.get_module_stem(target_file)
        results: list[tuple[str, CallSite]] = []

        for file_path, calls in self.call_sites.items():
            file_imps = self.imports.get(file_path, [])

            # Check if file imports func_name directly: `from target_mod import func_name`
            direct_import_alias = None
            module_import_alias = None

            for imp in file_imps:
                if imp.is_from and imp.imported_module == target_mod and imp.imported_symbol == func_name:
                    direct_import_alias = imp.alias
                elif not imp.is_from and imp.imported_module == target_mod:
                    module_import_alias = imp.alias

            for call in calls:
                if not call.is_attribute and direct_import_alias and call.func_name == direct_import_alias or call.is_attribute and module_import_alias and call.caller_obj == module_import_alias and call.func_name == func_name:
                    results.append((file_path, call))
                elif file_path == target_file and not call.is_attribute and call.func_name == func_name:
                    # Internal call within the same file
                    results.append((file_path, call))

        return results

    def validate_contract_preservation(
        self,
        target_file: str,
        mutated_code: str,
    ) -> tuple[bool, list[str]]:
        """Validates that a mutated module preserves all cross-file export signatures and required schema keys."""
        violations: list[str] = []
        try:
            new_tree = ast.parse(mutated_code, filename=target_file)
        except SyntaxError as e:
            return False, [f"SyntaxError in {target_file}: {e}"]

        new_symbols = SymbolCollector(target_file)
        new_symbols.visit(new_tree)

        # Inspect all functions previously exported by target_file
        orig_symbols = self.symbols.get(target_file)
        if not orig_symbols:
            return True, []

        for fn_name, orig_fn in orig_symbols.functions.items():
            callers = self.find_callers_of(target_file, fn_name)
            external_callers = [(path, cs) for path, cs in callers if path != target_file]
            if not external_callers:
                continue

            # Check 1: Export Preservation
            if fn_name not in new_symbols.functions:
                violations.append(
                    f"ExportViolation: '{fn_name}' was removed or renamed in {target_file}, "
                    f"breaking {len(external_callers)} external caller(s)"
                )
                continue

            new_fn = new_symbols.functions[fn_name]
            # Check 2: Arity / Parameter Compatibility
            min_args = len(new_fn.args.args) - len(new_fn.args.defaults)

            for caller_file, call in external_callers:
                if call.arg_count < min_args:
                    violations.append(
                        f"SignatureMismatch: '{fn_name}' requires at least {min_args} arguments, "
                        f"but {caller_file}:{call.line_number} supplies {call.arg_count}"
                    )

            # Check 3: Schema / Return Dict Keys
            # Find subscript keys accessed on return value in callers
            caller_accessed_keys: set[str] = set()
            for caller_file, _ in external_callers:
                caller_ast = self.file_ast.get(caller_file)
                if caller_ast:
                    for node in ast.walk(caller_ast):
                        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                            if isinstance(node.slice.value, str):
                                caller_accessed_keys.add(node.slice.value)

            if caller_accessed_keys:
                # Find keys returned in new_fn
                returned_keys: set[str] = set()
                has_dict_return = False
                for node in ast.walk(new_fn):
                    if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                        has_dict_return = True
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                returned_keys.add(k.value)

                if has_dict_return:
                    missing_keys = caller_accessed_keys - returned_keys
                    if missing_keys:
                        violations.append(
                            f"SchemaContractViolation: mutated '{fn_name}' drops dictionary key(s) "
                            f"{sorted(missing_keys)} actively required by downstream callers"
                        )

        return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Cross-File AST Rewriter Transformers
# ---------------------------------------------------------------------------

class FunctionRenamerTransformer(ast.NodeTransformer):
    """Synchronously renames a function definition or invocation node."""

    def __init__(self, old_name: str, new_name: str, target_module: str | None = None, is_definition: bool = False):
        self.old_name = old_name
        self.new_name = new_name
        self.target_module = target_module
        self.is_definition = is_definition
        self.mutated = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.is_definition and node.name == self.old_name:
            node.name = self.new_name
            self.mutated = True
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self.is_definition and node.name == self.old_name:
            node.name = self.new_name
            self.mutated = True
        self.generic_visit(node)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if self.target_module and node.module == self.target_module:
            for alias in node.names:
                if alias.name == self.old_name:
                    alias.name = self.new_name
                    if alias.asname is None:
                        alias.asname = None
                    self.mutated = True
        return node

    def visit_Name(self, node: ast.Name):
        if not self.is_definition and node.id == self.old_name:
            node.id = self.new_name
            self.mutated = True
        return node

    def visit_Attribute(self, node: ast.Attribute):
        if not self.is_definition and node.attr == self.old_name:
            if self.target_module is None or (isinstance(node.value, ast.Name) and node.value.id == self.target_module):
                node.attr = self.new_name
                self.mutated = True
        self.generic_visit(node)
        return node


class ParameterSyncTransformer(ast.NodeTransformer):
    """Adds or modifies a parameter in a function definition or supplies default in call sites."""

    def __init__(
        self,
        func_name: str,
        param_name: str,
        default_val: Any = None,
        is_definition: bool = False,
        module_name: str | None = None,
    ):
        self.func_name = func_name
        self.param_name = param_name
        self.default_val = default_val
        self.is_definition = is_definition
        self.module_name = module_name
        self.mutated = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.is_definition and node.name == self.func_name:
            # Check if parameter already exists
            existing = [arg.arg for arg in node.args.args]
            if self.param_name not in existing:
                node.args.args.append(ast.arg(arg=self.param_name))
                node.args.defaults.append(ast.Constant(value=self.default_val))
                self.mutated = True
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call):
        if not self.is_definition:
            is_target_call = False
            if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
                is_target_call = True
            elif isinstance(node.func, ast.Attribute) and node.func.attr == self.func_name:
                if self.module_name is None or (isinstance(node.func.value, ast.Name) and node.func.value.id == self.module_name):
                    is_target_call = True

            if is_target_call:
                # Add default keyword argument if not present
                kw_args = [kw.arg for kw in node.keywords if kw.arg is not None]
                if self.param_name not in kw_args:
                    node.keywords.append(
                        ast.keyword(arg=self.param_name, value=ast.Constant(value=self.default_val))
                    )
                    self.mutated = True
        self.generic_visit(node)
        return node


class ImportInjectorTransformer(ast.NodeTransformer):
    """Injects a `from provider_module import symbol` at the top of a file if not present."""

    def __init__(self, provider_module: str, symbol_name: str):
        self.provider_module = provider_module
        self.symbol_name = symbol_name
        self.already_present = False

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == self.provider_module:
            for alias in node.names:
                if alias.name == self.symbol_name:
                    self.already_present = True
        return node


# ---------------------------------------------------------------------------
# Cross-File Synchronized Mutator Engine
# ---------------------------------------------------------------------------

class CrossFileMutator:
    """Performs synchronized, atomic AST mutations across multiple files in a project."""

    @staticmethod
    def rename_symbol_synchronized(
        sources: dict[str, str],
        target_file: str,
        old_name: str,
        new_name: str,
    ) -> dict[str, str]:
        """Renames a function in target_file and synchronizes all imports and call sites across all files."""
        graph = CrossFileDependencyGraph.build(sources)
        if target_file not in graph.file_ast:
            return sources

        target_mod = graph.get_module_stem(target_file)
        new_sources: dict[str, str] = {}

        # 1. Update Definition in target_file
        def_tree = copy.deepcopy(graph.file_ast[target_file])
        def_trans = FunctionRenamerTransformer(old_name, new_name, is_definition=True)
        def_tree = def_trans.visit(def_tree)
        ast.fix_missing_locations(def_tree)
        new_sources[target_file] = ast.unparse(def_tree)

        # 2. Update Consumers / Callers across all other files
        for path, tree in graph.file_ast.items():
            if path == target_file:
                continue

            cons_tree = copy.deepcopy(tree)
            cons_trans = FunctionRenamerTransformer(old_name, new_name, target_module=target_mod, is_definition=False)
            cons_tree = cons_trans.visit(cons_tree)
            ast.fix_missing_locations(cons_tree)
            new_sources[path] = ast.unparse(cons_tree)

        return new_sources

    @staticmethod
    def add_parameter_synchronized(
        sources: dict[str, str],
        target_file: str,
        func_name: str,
        param_name: str,
        default_val: Any = None,
    ) -> dict[str, str]:
        """Adds a parameter with default value to target_file.func_name and updates all caller nodes across files."""
        graph = CrossFileDependencyGraph.build(sources)
        if target_file not in graph.file_ast:
            return sources

        target_mod = graph.get_module_stem(target_file)
        new_sources: dict[str, str] = {}

        # 1. Update Function Definition
        def_tree = copy.deepcopy(graph.file_ast[target_file])
        def_trans = ParameterSyncTransformer(
            func_name=func_name,
            param_name=param_name,
            default_val=default_val,
            is_definition=True,
        )
        def_tree = def_trans.visit(def_tree)
        ast.fix_missing_locations(def_tree)
        new_sources[target_file] = ast.unparse(def_tree)

        # 2. Update Call Sites in other files
        for path, tree in graph.file_ast.items():
            if path == target_file:
                continue
            cons_tree = copy.deepcopy(tree)
            cons_trans = ParameterSyncTransformer(
                func_name=func_name,
                param_name=param_name,
                default_val=default_val,
                is_definition=False,
                module_name=target_mod,
            )
            cons_tree = cons_trans.visit(cons_tree)
            ast.fix_missing_locations(cons_tree)
            new_sources[path] = ast.unparse(cons_tree)

        return new_sources

    @staticmethod
    def inject_import(
        sources: dict[str, str],
        consumer_file: str,
        provider_file: str,
        symbol_name: str,
    ) -> dict[str, str]:
        """Injects `from provider_module import symbol_name` into consumer_file."""
        graph = CrossFileDependencyGraph.build(sources)
        if consumer_file not in graph.file_ast or provider_file not in graph.file_ast:
            return sources

        provider_mod = graph.get_module_stem(provider_file)
        cons_tree = copy.deepcopy(graph.file_ast[consumer_file])

        injector = ImportInjectorTransformer(provider_mod, symbol_name)
        injector.visit(cons_tree)

        if not injector.already_present and isinstance(cons_tree, ast.Module):
            new_import = ast.ImportFrom(
                module=provider_mod,
                names=[ast.alias(name=symbol_name, asname=None)],
                level=0,
            )
            cons_tree.body.insert(0, new_import)
            ast.fix_missing_locations(cons_tree)

        new_sources = dict(sources)
        new_sources[consumer_file] = ast.unparse(cons_tree)
        return new_sources

    @classmethod
    def mutate_multi_file_genome(
        cls,
        genome: MultiFileASTGenome,
        rng: random.Random | None = None,
    ) -> MultiFileASTGenome:
        """Applies a synchronized cross-file AST mutation to a MultiFileASTGenome."""
        rng = rng or random.Random()
        sources = genome.to_sources()
        if len(sources) < 2:
            return genome.clone()

        graph = CrossFileDependencyGraph.build(sources)
        file_list = list(sources.keys())
        target_file = rng.choice(file_list)
        sym_col = graph.symbols.get(target_file)

        if not sym_col or not sym_col.functions:
            return genome.clone()

        func_name = rng.choice(list(sym_col.functions.keys()))
        mutation_kind = rng.choice(["rename_sync", "param_sync", "inject_import"])

        try:
            if mutation_kind == "rename_sync":
                new_name = f"{func_name}_opt"
                mut_sources = cls.rename_symbol_synchronized(sources, target_file, func_name, new_name)
            elif mutation_kind == "param_sync":
                param_name = rng.choice(["strict", "verbose", "timeout", "limit", "scale"])
                mut_sources = cls.add_parameter_synchronized(
                    sources, target_file, func_name, param_name, default_val=1
                )
            else:
                other_file = rng.choice([f for f in file_list if f != target_file])
                mut_sources = cls.inject_import(sources, other_file, target_file, func_name)

            # Validate that all files compile cleanly
            for path, code in mut_sources.items():
                compile(code, path, "exec")

            return MultiFileASTGenome.from_sources(mut_sources)
        except Exception:
            return genome.clone()
