"""Core types, node metadata, symbols, and type inference abstractions for RealASTGenome."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CriticalityLevel(int, Enum):
    """Graded criticality levels replacing binary is_critical."""
    IMMUTABLE = 0      # Never modify (function headers, return statements)
    STRUCTURAL = 1     # Control structures (loop headers, top if conditions) - very rare (0.05)
    SEMANTIC = 2       # Meaningful logic (operations, body statements) - moderate (0.35)
    COSMETIC = 3       # Low-impact leaves (constants, in-scope names) - higher (0.55)


class NodeType(str, Enum):
    """Categorizes AST nodes to guide intelligent, semantics-preserving mutations."""
    EXPRESSION = "expr"        # Substitutable expressions (BinOp, Constant, Call, Name)
    STATEMENT = "stmt"         # Statements (Assign, Expr, Pass)
    DECLARATION = "decl"       # Declarations (ClassDef, Imports)
    CONTROL_FLOW = "cf"        # Loops, Conditionals (If, For, While, Try)
    FUNCTION_DEF = "func"      # Function definitions (headers, signatures)
    RETURN = "return"          # Return statements (critical for data output)


@dataclass
class NodeMetadata:
    """Rich semantic and structural metadata attached to each AST node."""
    node_type: NodeType
    criticality: CriticalityLevel = CriticalityLevel.SEMANTIC
    depth: int = 0
    scope_id: str = "global"
    data_flow_deps: list[str] = field(default_factory=list)
    control_deps: list[str] = field(default_factory=list)
    is_critical: bool = False
    complexity_score: float = 1.0
    mutation_probability: float = 0.5
    dependency_score: float = 0.0


@dataclass
class Symbol:
    """Represents a named program entity (variable, parameter, function, class)."""
    name: str
    scope: str
    symbol_type: str            # "variable", "parameter", "function", "class", "import"
    inferred_type: str | None = "unknown"
    line_defined: int = 0
    line_uses: list[int] = field(default_factory=list)
    is_mutable: bool = True
    value_range: tuple[float, float] | None = None


# Known signatures for popular standard library and data-science modules
KNOWN_STANDARD_LIBRARY_SIGNATURES: dict[str, str] = {
    # math
    "math.sqrt": "float", "math.sin": "float", "math.cos": "float", "math.tan": "float",
    "math.log": "float", "math.log10": "float", "math.exp": "float", "math.pow": "float",
    "math.floor": "int", "math.ceil": "int", "math.gcd": "int", "math.factorial": "int",
    # random
    "random.random": "float", "random.randint": "int", "random.randrange": "int", "random.uniform": "float",
    # json
    "json.dumps": "str", "json.loads": "dict",
    # re
    "re.findall": "list", "re.sub": "str", "re.split": "list",
    # numpy / data science heuristics
    "np.mean": "float", "np.std": "float", "np.sum": "float", "np.var": "float",
    "np.zeros": "ndarray", "np.ones": "ndarray", "np.array": "ndarray", "np.arange": "ndarray",
    "numpy.mean": "float", "numpy.std": "float", "numpy.sum": "float",
    "pd.DataFrame": "DataFrame", "pd.Series": "Series", "pd.read_csv": "DataFrame",
}


class TypeInfo:
    """Lightweight yet expressive type inference and compatibility engine."""

    COMPATIBLE_PAIRS = {
        ("int", "float"),
        ("float", "int"),
        ("int", "bool"),
        ("bool", "int"),
        ("str", "str"),
        ("list", "list"),
        ("dict", "dict"),
        ("set", "set"),
    }

    def __init__(self):
        self.node_types: dict[int, str] = {}
        self.runtime_types: dict[str, str] = {}

    def record_type(self, node: ast.AST, inferred_type: str) -> None:
        self.node_types[id(node)] = inferred_type

    def record_runtime_type(self, identifier: str, runtime_type: str) -> None:
        """Records concrete runtime type observed from sandbox execution."""
        self.runtime_types[identifier] = runtime_type

    def get_type(self, node: ast.AST) -> str:
        return self.node_types.get(id(node), "unknown")

    def infer_type(self, node: ast.AST, symbol_lookup_fn: Any = None) -> str:
        """Infers the runtime type of an AST node based on literals, symbols, and operators."""
        if isinstance(node, ast.Constant):
            t = type(node.value).__name__
            self.record_type(node, t)
            return t

        if isinstance(node, ast.Name):
            if node.id in self.runtime_types:
                t = self.runtime_types[node.id]
                self.record_type(node, t)
                return t
            if symbol_lookup_fn:
                sym = symbol_lookup_fn(node.id)
                if sym and sym.inferred_type and sym.inferred_type != "unknown":
                    self.record_type(node, sym.inferred_type)
                    return sym.inferred_type
            return "unknown"

        if isinstance(node, ast.BinOp):
            left_t = self.infer_type(node.left, symbol_lookup_fn)
            right_t = self.infer_type(node.right, symbol_lookup_fn)
            inferred = self._infer_binop_type(left_t, right_t, node.op)
            self.record_type(node, inferred)
            return inferred

        if isinstance(node, ast.UnaryOp):
            operand_t = self.infer_type(node.operand, symbol_lookup_fn)
            if isinstance(node.op, ast.Not):
                self.record_type(node, "bool")
                return "bool"
            if isinstance(node.op, (ast.UAdd, ast.USub)):
                self.record_type(node, operand_t if operand_t in ("int", "float") else "unknown")
                return operand_t
            return "unknown"

        if isinstance(node, ast.Compare):
            self.record_type(node, "bool")
            return "bool"

        if isinstance(node, (ast.List, ast.ListComp)):
            self.record_type(node, "list")
            return "list"

        if isinstance(node, (ast.Dict, ast.DictComp)):
            self.record_type(node, "dict")
            return "dict"

        if isinstance(node, (ast.Set, ast.SetComp)):
            self.record_type(node, "set")
            return "set"

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns:
                ret_t = ast.unparse(node.returns).strip()
                self.record_type(node, ret_t)
                return ret_t
            return "function"

        if isinstance(node, ast.Call):
            # 1. Direct function calls
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.runtime_types:
                    t = self.runtime_types[func_name]
                    self.record_type(node, t)
                    return t
                if func_name in ("int", "round", "len", "abs"):
                    self.record_type(node, "int")
                    return "int"
                if func_name in ("float",):
                    self.record_type(node, "float")
                    return "float"
                if func_name in ("str", "repr"):
                    self.record_type(node, "str")
                    return "str"
                if func_name in ("list",):
                    self.record_type(node, "list")
                    return "list"
                if func_name in ("bool",):
                    self.record_type(node, "bool")
                    return "bool"

            # 2. Module / Attribute calls (e.g. math.sqrt, np.mean)
            elif isinstance(node.func, ast.Attribute):
                call_path = ast.unparse(node.func).strip()
                if call_path in self.runtime_types:
                    t = self.runtime_types[call_path]
                    self.record_type(node, t)
                    return t
                if call_path in KNOWN_STANDARD_LIBRARY_SIGNATURES:
                    t = KNOWN_STANDARD_LIBRARY_SIGNATURES[call_path]
                    self.record_type(node, t)
                    return t

            return "unknown"

        return "unknown"

    def _infer_binop_type(self, left: str, right: str, op: ast.operator) -> str:
        if left == "unknown" or right == "unknown":
            return "unknown"

        if isinstance(op, ast.Div):
            return "float"

        if left == right == "int":
            return "int"

        if "float" in (left, right) and all(t in ("int", "float") for t in (left, right)):
            return "float"

        if left == "str" and right == "str" and isinstance(op, ast.Add):
            return "str"

        if left == "list" and right == "list" and isinstance(op, ast.Add):
            return "list"

        if (left == "str" and right == "int" or left == "int" and right == "str") and isinstance(op, ast.Mult):
            return "str"

        return "unknown"

    def is_type_compatible(self, node_a: ast.AST, node_b: ast.AST) -> bool:
        type_a = self.get_type(node_a)
        type_b = self.get_type(node_b)

        if type_a == "unknown" or type_b == "unknown":
            return False

        if type_a == type_b:
            return True

        return (type_a, type_b) in self.COMPATIBLE_PAIRS
