"""ASTGenome implementation, tree edit distance metrics, and AST-aware structural mutations."""
from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .genome import EvolabGenome

# Weights for semantic categorization in AST distance
AST_NODE_WEIGHTS: dict[str, float] = {
    # Control flow & Structural declarations (Weight: 5.0)
    "FunctionDef": 5.0, "AsyncFunctionDef": 5.0, "ClassDef": 5.0, "Return": 4.0,
    "If": 5.0, "For": 5.0, "AsyncFor": 5.0, "While": 5.0, "Try": 5.0, "With": 4.0,
    "Raise": 4.0, "Assert": 3.0, "Break": 3.0, "Continue": 3.0,
    # Operators & Semantic Logic (Weight: 3.5)
    "Add": 3.5, "Sub": 3.5, "Mult": 3.5, "Div": 3.5, "FloorDiv": 3.5, "Mod": 3.5,
    "Pow": 3.5, "LShift": 3.5, "RShift": 3.5, "BitOr": 3.5, "BitXor": 3.5, "BitAnd": 3.5,
    "Eq": 3.5, "NotEq": 3.5, "Lt": 3.5, "LtE": 3.5, "Gt": 3.5, "GtE": 3.5,
    "Is": 3.5, "IsNot": 3.5, "In": 3.5, "NotIn": 3.5, "And": 3.5, "Or": 3.5, "Not": 3.5,
    # Expressions & Calls (Weight: 1.5)
    "Call": 2.0, "Assign": 1.5, "AugAssign": 2.0, "Attribute": 1.5, "Subscript": 1.5,
    "ListComp": 2.0, "SetComp": 2.0, "DictComp": 2.0,
    # Identifiers & Leaves (Weight: 0.5)
    "Name": 0.5, "Constant": 0.5, "Store": 0.2, "Load": 0.2, "Del": 0.2,
}


class ASTDepthVisitor(ast.NodeVisitor):
    """Computes total node count, maximum tree depth, and operator frequency."""

    def __init__(self):
        self.count = 0
        self.max_depth = 0
        self.current_depth = 0
        self.node_types = Counter()
        self.func_count = 0
        self.stmt_count = 0

    def generic_visit(self, node: ast.AST):
        self.count += 1
        self.node_types[type(node).__name__] += 1
        if isinstance(node, ast.Call):
            callee = ""
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            if callee:
                self.node_types[f"Call:{callee}"] += 1
        elif isinstance(node, ast.Attribute):
            self.node_types[f"Attr:{node.attr}"] += 1

        if isinstance(node, ast.FunctionDef):
            self.func_count += 1
        if isinstance(node, ast.stmt):
            self.stmt_count += 1

        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        super().generic_visit(node)
        self.current_depth -= 1


class CyclomaticComplexityVisitor(ast.NodeVisitor):
    """Computes McCabe Cyclomatic Complexity (M = 1 + Decision Points)."""

    def __init__(self) -> None:
        self.complexity = 1

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += max(len(node.values) - 1, 1)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.complexity += 1 + len(node.ifs)
        self.generic_visit(node)


def compute_cyclomatic_complexity(tree: ast.AST) -> int:
    visitor = CyclomaticComplexityVisitor()
    visitor.visit(tree)
    return visitor.complexity


def get_ast_metrics(tree: ast.AST) -> dict[str, Any]:
    visitor = ASTDepthVisitor()
    visitor.visit(tree)
    cc = compute_cyclomatic_complexity(tree)
    return {
        "node_count": visitor.count,
        "max_depth": visitor.max_depth,
        "stmt_count": visitor.stmt_count,
        "func_count": visitor.func_count,
        "node_types": visitor.node_types,
        "cyclomatic_complexity": cc,
    }


# Prohibited modules and attributes for anti-sandbox / environment-sensing deterrence
PROHIBITED_INTROSPECTION_MODULES = {
    "inspect",
    "socket",
    "ctypes",
    "subprocess",
    "multiprocessing",
    "pty",
    "termios",
    "resource",
}

PROHIBITED_ATTRIBUTES = {
    "__subclasses__",
    "__globals__",
    "__bases__",
    "__code__",
    "isatty",
    "_getframe",
    "_current_frames",
    "stack",
    "currentframe",
}

PROHIBITED_OS_CALLS = {
    "system",
    "popen",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
}


class ASTIntrospectionSecurityVisitor(ast.NodeVisitor):
    """Detects unauthorized environment sensing, VM evasion, and runtime introspection."""

    def __init__(self):
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod in PROHIBITED_INTROSPECTION_MODULES:
                self.violations.append(
                    f"ProhibitedModuleImport: '{alias.name}' at line {getattr(node, 'lineno', 1)}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod in PROHIBITED_INTROSPECTION_MODULES:
                self.violations.append(
                    f"ProhibitedModuleImportFrom: '{node.module}' at line {getattr(node, 'lineno', 1)}"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in PROHIBITED_ATTRIBUTES:
            self.violations.append(
                f"ProhibitedIntrospectionAttribute: '.{node.attr}' at line {getattr(node, 'lineno', 1)}"
            )
        if node.attr == "environ" and isinstance(node.value, ast.Name) and node.value.id == "os":
            self.violations.append(
                f"ProhibitedEnvironmentProbe: 'os.environ' at line {getattr(node, 'lineno', 1)}"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                if node.func.attr == "getenv":
                    self.violations.append(
                        f"ProhibitedEnvironmentProbe: 'os.getenv' at line {getattr(node, 'lineno', 1)}"
                    )
                elif node.func.attr in PROHIBITED_OS_CALLS:
                    self.violations.append(
                        f"ProhibitedCommandExecution: 'os.{node.func.attr}' at line {getattr(node, 'lineno', 1)}"
                    )
        self.generic_visit(node)


def validate_code_purity(tree_or_code: ast.AST | str) -> tuple[bool, list[str]]:
    """Statically verifies that an AST or code string does not contain introspection/anti-sandbox heuristics."""
    if isinstance(tree_or_code, str):
        try:
            tree = ast.parse(tree_or_code)
        except SyntaxError as e:
            return False, [f"SyntaxError: {e}"]
    else:
        tree = tree_or_code

    visitor = ASTIntrospectionSecurityVisitor()
    visitor.visit(tree)
    return len(visitor.violations) == 0, visitor.violations


def ast_distance(
    a: ASTGenome,
    b: ASTGenome,
    w1: float = 0.5,
    w2: float = 0.3,
    w3: float = 0.2,
) -> float:
    """Computes weighted semantic structural, textual, and complexity distance [0.0, 1.0] between two ASTGenomes."""
    if a.fingerprint() == b.fingerprint():
        return 0.0

    metrics_a = get_ast_metrics(a.tree)
    metrics_b = get_ast_metrics(b.tree)

    types_a = metrics_a["node_types"]
    types_b = metrics_b["node_types"]

    all_keys = set(types_a.keys()) | set(types_b.keys())
    if not all_keys:
        return 0.0

    weighted_intersect = 0.0
    weighted_union = 0.0
    for k in all_keys:
        if k.startswith("Call:"):
            weight = 3.0
        elif k.startswith("Attr:"):
            weight = 1.5
        else:
            weight = AST_NODE_WEIGHTS.get(k, 1.0)
        c_a = types_a.get(k, 0)
        c_b = types_b.get(k, 0)
        weighted_intersect += weight * min(c_a, c_b)
        weighted_union += weight * max(c_a, c_b)

    weighted_jaccard = 1.0 - (weighted_intersect / max(1e-6, weighted_union))

    # Code textual distance
    code_a = a.to_code()
    code_b = b.to_code()
    matcher = difflib.SequenceMatcher(None, code_a, code_b)
    code_dist = 1.0 - matcher.ratio()

    # Structural & Cyclomatic Complexity semantic distance
    cc_a = metrics_a.get("cyclomatic_complexity", 1)
    cc_b = metrics_b.get("cyclomatic_complexity", 1)
    depth_a = metrics_a.get("max_depth", 1)
    depth_b = metrics_b.get("max_depth", 1)
    cc_dist = abs(cc_a - cc_b) / max(cc_a, cc_b, 1)
    depth_dist = abs(depth_a - depth_b) / max(depth_a, depth_b, 1)
    complexity_dist = 0.7 * cc_dist + 0.3 * depth_dist

    total = w1 * weighted_jaccard + w2 * code_dist + w3 * complexity_dist
    return max(0.0, min(1.0, total))


@dataclass
class ASTGenome(EvolabGenome):
    """Genome represented as a Python Abstract Syntax Tree (AST)."""

    tree: ast.AST

    @classmethod
    def from_code(cls, code: str) -> ASTGenome:
        tree = ast.parse(code)
        return cls(tree=tree)

    def to_code(self) -> str:
        return ast.unparse(self.tree)

    def clone(self) -> ASTGenome:
        return ASTGenome(tree=copy.deepcopy(self.tree))

    def fingerprint(self) -> str:
        raw_dump = ast.dump(self.tree, annotate_fields=False)
        return hashlib.sha256(raw_dump.encode("utf-8")).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, ASTGenome):
            raise TypeError(f"Cannot compare ASTGenome with {type(other)}")
        return ast_distance(self, other)

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "ASTGenome",
            "fingerprint": self.fingerprint(),
            "code": self.to_code(),
        }

    def describe(self) -> dict[str, Any]:
        metrics = get_ast_metrics(self.tree)
        return {
            "node_count": metrics["node_count"],
            "max_depth": metrics["max_depth"],
            "stmt_count": metrics["stmt_count"],
            "func_count": metrics["func_count"],
            "cyclomatic_complexity": metrics["cyclomatic_complexity"],
        }

    def validate_purity(self) -> tuple[bool, list[str]]:
        """Statically checks that the genome contains no prohibited introspection or anti-sandbox constructs."""
        return validate_code_purity(self.tree)

    def __len__(self) -> int:
        return get_ast_metrics(self.tree)["node_count"]

    def mutate(self, rng: random.Random | None = None, **kwargs: Any) -> ASTGenome:
        return mutate_ast(self, rng=rng)

    def crossover(self, other: EvolabGenome, rng: random.Random | None = None, **kwargs: Any) -> ASTGenome:
        if not isinstance(other, ASTGenome):
            return self.clone()
        child_a, _ = crossover_ast(self, other, rng=rng)
        return child_a


def create_random_ast_genome(rng: random.Random | None = None) -> ASTGenome:
    """Generates a valid randomized ASTGenome from parameterized syntactic templates."""
    r = rng or random
    templates = [
        "def compute(x, y):\n    return x + y\n",
        "def compute(x, y):\n    return x * y + 1\n",
        "def compute(x, y):\n    if x > y:\n        return x - y\n    return y - x\n",
        "def compute(x, y):\n    total = 0\n    for i in range(x):\n        total += y\n    return total\n",
        "def compute(x, y):\n    res = x\n    while res < y:\n        res += 1\n    return res\n",
    ]
    code = r.choice(templates)
    return ASTGenome.from_code(code)


@dataclass
class MultiFileASTGenome(EvolabGenome):
    """AST Genome representing multiple interconnected Python source modules."""

    files: dict[str, ASTGenome] = field(default_factory=dict)

    @classmethod
    def from_sources(cls, sources: dict[str, str]) -> MultiFileASTGenome:
        files = {
            path: ASTGenome.from_code(code)
            for path, code in sources.items()
        }
        return cls(files=files)

    def to_sources(self) -> dict[str, str]:
        return {path: g.to_code() for path, g in self.files.items()}

    def fingerprint(self) -> str:
        combined = "|".join(f"{path}:{g.fingerprint()}" for path, g in sorted(self.files.items()))
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "MultiFileASTGenome",
            "files": {path: g.serialize() for path, g in self.files.items()},
        }

    def clone(self) -> MultiFileASTGenome:
        return MultiFileASTGenome(files={path: g.clone() for path, g in self.files.items()})

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, MultiFileASTGenome):
            return 1.0
        all_keys = set(self.files.keys()) | set(other.files.keys())
        if not all_keys:
            return 0.0
        total_d = sum(
            self.files[k].distance_to(other.files.get(k, ASTGenome.from_code("")))
            if k in self.files
            else 1.0
            for k in all_keys
        )
        return round(total_d / len(all_keys), 6)

    def describe(self) -> dict[str, Any]:
        total_nodes = sum(g.describe()["node_count"] for g in self.files.values())
        max_depth = max((g.describe()["max_depth"] for g in self.files.values()), default=0)
        total_cc = sum(g.describe().get("cyclomatic_complexity", 1) for g in self.files.values())
        return {
            "files_count": len(self.files),
            "node_count": total_nodes,
            "max_depth": max_depth,
            "cyclomatic_complexity": total_cc,
        }

    def __len__(self) -> int:
        return sum(len(g) for g in self.files.values())




# ---------------------------------------------------------------------------
# AST-Aware Structural Mutations
# ---------------------------------------------------------------------------

class ConstantTransformer(ast.NodeTransformer):
    """Mutates constant literals (numbers, booleans, strings)."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.mutated = False

    def visit_Constant(self, node: ast.Constant):
        if not self.mutated and self.rng.random() < 0.6:
            val = node.value
            if isinstance(val, bool):
                node.value = not val
                self.mutated = True
            elif isinstance(val, int):
                delta = self.rng.choice([-2, -1, 1, 2, 5])
                node.value = val + delta
                self.mutated = True
            elif isinstance(val, float):
                delta = self.rng.choice([-1.0, -0.5, 0.5, 1.0])
                node.value = round(val + delta, 4)
                self.mutated = True
        return node


class BinOpTransformer(ast.NodeTransformer):
    """Swaps binary arithmetic operations."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.mutated = False

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if not self.mutated and self.rng.random() < 0.6:
            swaps = {
                ast.Add: [ast.Sub, ast.Mult],
                ast.Sub: [ast.Add, ast.Mult],
                ast.Mult: [ast.Add, ast.FloorDiv],
                ast.FloorDiv: [ast.Mult, ast.Div],
                ast.Div: [ast.Mult, ast.FloorDiv],
                ast.Mod: [ast.FloorDiv],
            }
            op_cls = type(node.op)
            if op_cls in swaps:
                new_op_cls = self.rng.choice(swaps[op_cls])
                node.op = new_op_cls()
                self.mutated = True
        return node


class CompareTransformer(ast.NodeTransformer):
    """Swaps comparison operators (==, !=, <, <=, >, >=)."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.mutated = False

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if not self.mutated and node.ops and self.rng.random() < 0.6:
            idx = self.rng.randint(0, len(node.ops) - 1)
            op = node.ops[idx]
            swaps = {
                ast.Eq: [ast.NotEq, ast.Lt, ast.Gt],
                ast.NotEq: [ast.Eq],
                ast.Lt: [ast.LtE, ast.Gt, ast.Eq],
                ast.LtE: [ast.Lt, ast.GtE],
                ast.Gt: [ast.GtE, ast.Lt, ast.Eq],
                ast.GtE: [ast.Gt, ast.LtE],
                ast.In: [ast.NotIn],
                ast.NotIn: [ast.In],
                ast.Is: [ast.IsNot],
                ast.IsNot: [ast.Is],
            }
            op_cls = type(op)
            if op_cls in swaps:
                node.ops[idx] = self.rng.choice(swaps[op_cls])()
                self.mutated = True
        return node


class BoolOpTransformer(ast.NodeTransformer):
    """Swaps boolean operators (And <-> Or)."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.mutated = False

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        if not self.mutated and self.rng.random() < 0.6:
            if isinstance(node.op, ast.And):
                node.op = ast.Or()
                self.mutated = True
            elif isinstance(node.op, ast.Or):
                node.op = ast.And()
                self.mutated = True
        return node


class StatementSwapTransformer(ast.NodeTransformer):
    """Swaps order of two adjacent non-branching statements in a body."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.mutated = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        if not self.mutated and len(node.body) >= 2 and self.rng.random() < 0.5:
            # Pick adjacent pair excluding return/break
            candidates = [
                i for i in range(len(node.body) - 1)
                if not isinstance(node.body[i], (ast.Return, ast.Break, ast.Continue))
                and not isinstance(node.body[i + 1], (ast.Return, ast.Break, ast.Continue))
            ]
            if candidates:
                idx = self.rng.choice(candidates)
                node.body[idx], node.body[idx + 1] = node.body[idx + 1], node.body[idx]
                self.mutated = True
        return node


def mutate_ast(
    genome: ASTGenome, rng: random.Random | None = None, max_retries: int = 6
) -> ASTGenome:
    """Applies a safe AST-level mutation and returns a new valid ASTGenome."""
    rng = rng or random.Random()
    transformers = [
        ConstantTransformer,
        BinOpTransformer,
        CompareTransformer,
        BoolOpTransformer,
        StatementSwapTransformer,
    ]

    for _ in range(max_retries):
        tree_copy = copy.deepcopy(genome.tree)
        transformer_cls = rng.choice(transformers)
        transformer = transformer_cls(rng)
        mutated_tree = transformer.visit(tree_copy)
        ast.fix_missing_locations(mutated_tree)

        try:
            code = ast.unparse(mutated_tree)
            compile(code, "<ast_genome>", "exec")
            from .ast_guard import is_ast_change_safe
            is_safe, _ = is_ast_change_safe(mutated_tree)
            if not is_safe:
                continue
            # Successful valid AST mutation
            return ASTGenome(tree=mutated_tree)
        except Exception:
            continue

    return genome.clone()


def crossover_ast(
    parent_a: ASTGenome, parent_b: ASTGenome, rng: random.Random | None = None
) -> tuple[ASTGenome, ASTGenome]:
    """Performs subtree crossover between two compatible ASTGenomes."""
    rng = rng or random.Random()

    def get_expr_nodes(tree: ast.AST) -> list[ast.expr]:
        nodes = []
        for n in ast.walk(tree):
            if isinstance(n, (ast.BinOp, ast.Compare, ast.Constant, ast.Call)):
                nodes.append(n)
        return nodes

    nodes_a = get_expr_nodes(parent_a.tree)
    nodes_b = get_expr_nodes(parent_b.tree)

    if not nodes_a or not nodes_b:
        return parent_a.clone(), parent_b.clone()

    target_a = rng.choice(nodes_a)
    compatible_b = [n for n in nodes_b if type(n) is type(target_a)]
    if not compatible_b:
        return parent_a.clone(), parent_b.clone()
    target_b = rng.choice(compatible_b)

    tree_a = copy.deepcopy(parent_a.tree)
    tree_b = copy.deepcopy(parent_b.tree)

    class NodeReplacer(ast.NodeTransformer):
        def __init__(self, old_node: ast.AST, new_node: ast.AST):
            self.old_node = old_node
            self.new_node = new_node
            self.replaced = False

        def generic_visit(self, node: ast.AST):
            if not self.replaced and ast.dump(node) == ast.dump(self.old_node):
                self.replaced = True
                return copy.deepcopy(self.new_node)
            return super().generic_visit(node)

    replacer_a = NodeReplacer(target_a, target_b)
    replacer_b = NodeReplacer(target_b, target_a)

    new_tree_a = replacer_a.visit(tree_a)
    new_tree_b = replacer_b.visit(tree_b)

    ast.fix_missing_locations(new_tree_a)
    ast.fix_missing_locations(new_tree_b)

    from .ast_guard import is_ast_change_safe

    try:
        compile(ast.unparse(new_tree_a), "<crossover_a>", "exec")
        safe_a, _ = is_ast_change_safe(new_tree_a)
        child_a = ASTGenome(tree=new_tree_a) if safe_a else parent_a.clone()
    except Exception:
        child_a = parent_a.clone()

    try:
        compile(ast.unparse(new_tree_b), "<crossover_b>", "exec")
        safe_b, _ = is_ast_change_safe(new_tree_b)
        child_b = ASTGenome(tree=new_tree_b) if safe_b else parent_b.clone()
    except Exception:
        child_b = parent_b.clone()

    return child_a, child_b

