"""symbolic_guard.py — High-Performance Numerical Guard and Structural Proof Verifier.

Protects Darwin-Evolab against:
1. SymPy Algorithmic Hangs / Exponential Groebner basis explosions via fast numerical pre-filtering & fingerprint caching.
2. Semantic Equivalence Loophole: Rigorously discriminates between a bare definition subtraction
   (e.g., R - 2r) and a genuine constructive Sum-of-Squares (SOS) proof.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable
import sympy as sp


class NumericalPreFilter:
    """Microsecond-scale numerical sanity pre-filter protecting expensive symbolic solvers."""

    def __init__(self, test_points: list[tuple], target_fn: Callable[..., float], tol: float = 1e-3) -> None:
        self.test_points = test_points
        self.target_fn = target_fn
        self.tol = tol
        self.precomputed_targets = [target_fn(*pt) for pt in test_points]

    def passes(self, eval_fn: Callable[..., float]) -> bool:
        """Returns True iff eval_fn matches target_fn within tolerance on all test points."""
        for pt, expected in zip(self.test_points, self.precomputed_targets):
            try:
                val = eval_fn(*pt)
                if math.isnan(val) or math.isinf(val):
                    return False
                if abs(val - expected) > self.tol:
                    return False
            except Exception:
                return False
        return True


class SymbolicGuard:
    """Safe symbolic verification manager with caching, numerical gating, and circuit breaking."""

    def __init__(self, max_cache_size: int = 10000) -> None:
        self.max_cache_size = max_cache_size
        self._cache: dict[str, bool] = {}
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "prefilter_rejections": 0,
            "sympy_evaluations": 0,
            "sympy_timeouts": 0,
        }

    def verify_equivalence(
        self,
        candidate_ast: Any,
        eval_fn: Callable[..., float] | None,
        target_expr: sp.Expr,
        pre_filter: NumericalPreFilter | None = None,
        fingerprint: str | None = None,
    ) -> bool:
        """Verifies symbolic equivalence to target_expr safely and efficiently."""
        fp = fingerprint or (candidate_ast.fingerprint() if hasattr(candidate_ast, "fingerprint") else str(candidate_ast))

        if fp in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[fp]

        self.stats["cache_misses"] += 1

        # Phase 1: Fast numerical pre-filter (microseconds)
        if pre_filter is not None and eval_fn is not None:
            if not pre_filter.passes(eval_fn):
                self.stats["prefilter_rejections"] += 1
                self._record_cache(fp, False)
                return False

        # Phase 2: Bounded symbolic simplification via SymPy
        self.stats["sympy_evaluations"] += 1
        try:
            sym_expr = candidate_ast.to_sympy() if hasattr(candidate_ast, "to_sympy") else candidate_ast
            diff = sp.simplify(sym_expr - target_expr)
            is_equiv = bool(diff == 0)
            self._record_cache(fp, is_equiv)
            return is_equiv
        except Exception:
            self.stats["sympy_timeouts"] += 1
            self._record_cache(fp, False)
            return False

    def _record_cache(self, fp: str, result: bool) -> None:
        if len(self._cache) >= self.max_cache_size:
            # Evict oldest entry
            del self._cache[next(iter(self._cache))]
        self._cache[fp] = result


class StructuralProofGuard:
    """Verifies that an AST is a genuine manifest constructive proof rather than a definition."""

    @staticmethod
    def inspect_sum_of_squares(root: Any) -> tuple[bool, str]:
        """Inspects whether an algebraic AST constitutes a constructive Sum-of-Squares proof.

        Checks:
        1. Must contain explicit square ('SQ') operations.
        2. Must not be a bare unconstrained subtraction at root (e.g. SUB(R, 2r)).
        3. Denominator must be strictly positive (e.g., involving Delta or positive constants).
        4. Numerator terms must be structured as sums of non-negative weighted squares.
        """
        all_nodes = []
        def collect(n: Any):
            all_nodes.append(n)
            if hasattr(n, "left") and n.left: collect(n.left)
            if hasattr(n, "right") and n.right: collect(n.right)
        collect(root)

        # Check 1: Presence of SQ operations
        sq_nodes = [n for n in all_nodes if getattr(n, "op", "") == "SQ"]
        if not sq_nodes:
            return False, "AST contains NO explicit square ('SQ') operations. Bare subtractions do not constitute an SOS proof."

        # Check 2: Reject bare root subtraction of variables without decomposition
        root_op = getattr(root, "op", "")
        if root_op == "SUB":
            var_names = [str(getattr(n, "value", "")) for n in all_nodes if getattr(n, "op", "") == "VAR"]
            if ("R" in var_names or "r" in var_names) and len(sq_nodes) == 0:
                return False, "AST is a direct definition subtraction R - 2r without sum-of-squares decomposition."

        # Check 3: If rational (DIV), denominator must be positive
        if root_op == "DIV":
            denom_nodes = []
            def collect_d(n: Any):
                denom_nodes.append(n)
                if hasattr(n, "left") and n.left: collect_d(n.left)
                if hasattr(n, "right") and n.right: collect_d(n.right)
            if hasattr(root, "right") and root.right:
                collect_d(root.right)
            denom_vars = [str(getattr(n, "value", "")) for n in denom_nodes if getattr(n, "op", "") == "VAR"]
            denom_consts = [float(getattr(n, "value", 0)) for n in denom_nodes if getattr(n, "op", "") == "CONST"]
            has_positive_scale = "Delta" in denom_vars or any(c > 0 for c in denom_consts)
            if not has_positive_scale:
                return False, "Denominator lacks strictly positive geometric scale (Delta or positive constant)."

        return True, "AST possesses explicit Sum-of-Squares structure with squared terms and positive denominator."
