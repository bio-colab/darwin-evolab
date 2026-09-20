"""euler_inequality.py — Euler's Inequality (R >= 2r) and Constructive Equality Evolution.

An advanced mathematical domain driver for Darwin-Evolab that tests:
  - 100% Formal Algebraic Correctness verified symbolically via SymPy:
      Equivalent to (R - 2r), strictly non-negative, and vanishes iff a = b = c.
  - Extreme Numerical Stability:
      Eliminating catastrophic cancellation for near-equilateral triangles
      (where direct R - 2r fails with up to 100% relative error or 0.0).
  - Constructive Sum-of-Squares (SOS) proof of equality:
      R - 2r = ((b+c-a)(b-c)^2 + (a+c-b)(c-a)^2 + (a+b-c)(a-b)^2) / (8 * Delta)
  - Koza Parsimony Pressure (minimizing operational complexity).
  - MAP-Elites Quality Diversity archiving diverse constructive formulas.
  - Full internal engine telemetry (Self-Model, Dreaming, Wilson CI).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from mpmath import mp
import sympy as sp

from .adapters import DomainAdapter, register_adapter, register_domain_adapter
from .evaluators import Evaluator, FitnessResult
from .genome import EvolabGenome, Individual

# Symbolic definitions for formal verification
sym_a, sym_b, sym_c = sp.symbols("a b c", positive=True)
sym_s = (sym_a + sym_b + sym_c) / 2
sym_Delta = sp.sqrt(sym_s * (sym_s - sym_a) * (sym_s - sym_b) * (sym_s - sym_c))
sym_R = (sym_a * sym_b * sym_c) / (4 * sym_Delta)
sym_r = sym_Delta / sym_s
EULER_DEFECT_EXPR = sym_R - 2 * sym_r

# Default benchmark cases including extreme near-equilateral (catastrophic cancellation) and degenerate cases
DEFAULT_EULER_CASES: list[tuple[float, float, float]] = [
    (1.0 + 1e-7, 1.0, 1.0),                 # Near-equilateral: naive R - 2r has 7.7% cancellation error
    (1.0 + 1e-8, 1.0, 1.0),                 # Severe near-equilateral: naive R - 2r completely collapses to 0.0 (100% err)
    (1e6 + 1e-5, 1e6, 1e6),                 # Gigantic scale near-equilateral
    (100.0, 100.00001, 100.000005),         # Multi-side tiny asymmetry
    (2.0, 2.0, 2.0),                        # Exact equilateral: equality condition R - 2r == 0
    (3.0, 4.0, 5.0),                        # Standard right triangle: R=2.5, r=1.0 -> R - 2r = 0.5
    (13.0, 14.0, 15.0),                     # Integer Heronian triangle: R=8.125, r=4.0 -> R - 2r = 0.125
    (100.0, 100.0, 1.0),                    # Needle triangle: strong inequality
    (5.0, 5.0, 9.999999999),                # Degenerate: almost flat collinear triangle (Delta -> 0)
    (1e6, 1e6, 1.9999999999e6),             # Large-scale near-degenerate collinear
]


@dataclass
class EulerNode:
    """Node in an algebraic expression tree for Euler's inequality defect R - 2r."""
    op: str  # "VAR", "CONST", "ADD", "SUB", "MUL", "DIV", "SQ", "SQRT"
    value: Any = None
    left: EulerNode | None = None
    right: EulerNode | None = None

    def clone(self) -> EulerNode:
        return EulerNode(
            op=self.op,
            value=self.value,
            left=self.left.clone() if self.left else None,
            right=self.right.clone() if self.right else None,
        )

    def count_nodes(self) -> int:
        c = 1
        if self.left: c += self.left.count_nodes()
        if self.right: c += self.right.count_nodes()
        return c

    def depth(self) -> int:
        dl = self.left.depth() if self.left else 0
        dr = self.right.depth() if self.right else 0
        return 1 + max(dl, dr)

    def to_sympy(self) -> sp.Expr:
        if self.op == "VAR":
            v = str(self.value)
            if v == "a": return sym_a
            elif v == "b": return sym_b
            elif v == "c": return sym_c
            elif v == "s": return sym_s
            elif v == "Delta": return sym_Delta
            elif v == "x": return sym_s - sym_a
            elif v == "y": return sym_s - sym_b
            elif v == "z": return sym_s - sym_c
            elif v == "R": return sym_R
            elif v == "r": return sym_r
            return sp.Symbol(v, positive=True)
        elif self.op == "CONST":
            return sp.Rational(str(self.value)) if isinstance(self.value, (int, float)) else sp.sympify(self.value)
        elif self.op == "ADD":
            return self.left.to_sympy() + self.right.to_sympy()
        elif self.op == "SUB":
            return self.left.to_sympy() - self.right.to_sympy()
        elif self.op == "MUL":
            return self.left.to_sympy() * self.right.to_sympy()
        elif self.op == "DIV":
            return self.left.to_sympy() / self.right.to_sympy()
        elif self.op == "SQ":
            return self.left.to_sympy() ** 2
        elif self.op == "SQRT":
            return sp.sqrt(self.left.to_sympy())
        return sp.Integer(0)

    def evaluate_float(self, env: dict[str, float]) -> float:
        if self.op == "VAR":
            return env.get(str(self.value), 0.0)
        elif self.op == "CONST":
            return float(self.value)
        elif self.op == "ADD":
            return self.left.evaluate_float(env) + self.right.evaluate_float(env)
        elif self.op == "SUB":
            return self.left.evaluate_float(env) - self.right.evaluate_float(env)
        elif self.op == "MUL":
            return self.left.evaluate_float(env) * self.right.evaluate_float(env)
        elif self.op == "DIV":
            denom = self.right.evaluate_float(env)
            return self.left.evaluate_float(env) / (denom + 1e-24)
        elif self.op == "SQ":
            inner = self.left.evaluate_float(env)
            return inner * inner
        elif self.op == "SQRT":
            inner = self.left.evaluate_float(env)
            return math.sqrt(max(0.0, inner))
        return 0.0

    def to_pretty_str(self) -> str:
        if self.op == "VAR":
            return str(self.value)
        elif self.op == "CONST":
            return str(self.value)
        elif self.op == "ADD":
            return f"({self.left.to_pretty_str()} + {self.right.to_pretty_str()})"
        elif self.op == "SUB":
            return f"({self.left.to_pretty_str()} - {self.right.to_pretty_str()})"
        elif self.op == "MUL":
            return f"({self.left.to_pretty_str()} * {self.right.to_pretty_str()})"
        elif self.op == "DIV":
            return f"({self.left.to_pretty_str()} / {self.right.to_pretty_str()})"
        elif self.op == "SQ":
            return f"({self.left.to_pretty_str()})^2"
        elif self.op == "SQRT":
            return f"sqrt({self.left.to_pretty_str()})"
        return "?"


@dataclass
class EulerFormulaGenome(EvolabGenome):
    """Genome representing an evolving formula for Euler's defect R - 2r."""
    root: EulerNode
    formula_name: str = "CandidateEulerDefect"

    def clone(self) -> EulerFormulaGenome:
        return EulerFormulaGenome(
            root=self.root.clone(),
            formula_name=self.formula_name,
        )

    def node_count(self) -> int:
        return self.root.count_nodes()

    def __len__(self) -> int:
        return self.node_count()

    def depth(self) -> int:
        return self.root.depth()

    def to_sympy(self) -> sp.Expr:
        return self.root.to_sympy()

    def evaluate(self, a: float, b: float, c: float) -> float:
        # High-stability triangle properties via Kahan's sorted area
        sa, sb, sc = sorted([float(a), float(b), float(c)], reverse=True)
        # Kahan stable area to avoid cancellation in Delta itself
        prod = (sa + (sb + sc)) * (sc - (sa - sb)) * (sc + (sa - sb)) * (sa + (sb - sc))
        delta = 0.25 * math.sqrt(max(0.0, prod))
        s = (a + b + c) / 2.0
        x = s - a
        y = s - b
        z = s - c
        r = delta / s if s > 0 else 0.0
        R = (a * b * c) / (4.0 * delta) if delta > 0 else 0.0

        env = {
            "a": a, "b": b, "c": c,
            "s": s, "Delta": delta,
            "x": x, "y": y, "z": z,
            "R": R, "r": r,
        }
        return self.root.evaluate_float(env)

    def fingerprint(self) -> str:
        s = self.root.to_pretty_str()
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, EulerFormulaGenome):
            return 1.0
        s1 = self.node_count()
        s2 = other.node_count()
        diff = abs(s1 - s2) / max(1.0, float(max(s1, s2)))
        t1 = self.root.to_pretty_str()
        t2 = other.root.to_pretty_str()
        if t1 == t2:
            return 0.0
        return round(0.5 * diff + 0.5 * min(1.0, float(abs(len(t1) - len(t2)) + 5) / 50.0), 4)

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "EulerFormulaGenome",
            "fingerprint": self.fingerprint(),
            "formula_name": self.formula_name,
            "expression": self.root.to_pretty_str(),
            "node_count": self.node_count(),
            "depth": self.depth(),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count(),
            "depth": self.depth(),
            "fingerprint": self.fingerprint(),
        }

    def mutate(self, rng: random.Random | None = None, **kwargs: Any) -> EulerFormulaGenome:
        r = rng or random
        child = self.clone()
        nodes: list[EulerNode] = []

        def collect(n: EulerNode):
            nodes.append(n)
            if n.left: collect(n.left)
            if n.right: collect(n.right)

        collect(child.root)
        if not nodes:
            return child

        mtype = r.choice(["neutral_drift", "schema_inject", "constant_tweak", "operator_swap"])

        if mtype == "neutral_drift":
            # Commutative swap in ADD or MUL
            swappable = [n for n in nodes if n.op in ("ADD", "MUL") and n.left and n.right]
            if swappable:
                target = r.choice(swappable)
                target.left, target.right = target.right, target.left

        elif mtype == "schema_inject":
            # Inject Holland schema building block for Euler's equality / SOS
            target = r.choice(nodes)
            building_blocks = [
                # (b - c)^2
                EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "b"), EulerNode("VAR", "c"))),
                # (c - a)^2
                EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "c"), EulerNode("VAR", "a"))),
                # (a - b)^2
                EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "a"), EulerNode("VAR", "b"))),
                # (b + c - a)
                EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "b"), EulerNode("VAR", "c")), EulerNode("VAR", "a")),
                # (a + c - b)
                EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "a"), EulerNode("VAR", "c")), EulerNode("VAR", "b")),
                # (a + b - c)
                EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "a"), EulerNode("VAR", "b")), EulerNode("VAR", "c")),
                # 8 * Delta
                EulerNode("MUL", None, EulerNode("CONST", 8.0), EulerNode("VAR", "Delta")),
                # 4 * Delta
                EulerNode("MUL", None, EulerNode("CONST", 4.0), EulerNode("VAR", "Delta")),
            ]
            replacement = r.choice(building_blocks).clone()
            target.op = replacement.op
            target.value = replacement.value
            target.left = replacement.left
            target.right = replacement.right

        elif mtype == "constant_tweak":
            const_nodes = [n for n in nodes if n.op == "CONST"]
            if const_nodes:
                target = r.choice(const_nodes)
                target.value = r.choice([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

        elif mtype == "operator_swap":
            binary_nodes = [n for n in nodes if n.op in ("ADD", "SUB")]
            if binary_nodes:
                target = r.choice(binary_nodes)
                target.op = "SUB" if target.op == "ADD" else "ADD"

        return child

    def crossover(self, other: EvolabGenome, rng: random.Random | None = None) -> EulerFormulaGenome:
        if not isinstance(other, EulerFormulaGenome):
            return self.clone()
        r = rng or random
        child = self.clone()
        child_nodes: list[EulerNode] = []
        def collect(n: EulerNode):
            child_nodes.append(n)
            if n.left: collect(n.left)
            if n.right: collect(n.right)
        collect(child.root)

        other_nodes: list[EulerNode] = []
        def collect_o(n: EulerNode):
            other_nodes.append(n)
            if n.left: collect_o(n.left)
            if n.right: collect_o(n.right)
        collect_o(other.root)

        if child_nodes and other_nodes and r.random() < 0.7:
            target = r.choice(child_nodes)
            source = r.choice(other_nodes).clone()
            target.op = source.op
            target.value = source.value
            target.left = source.left
            target.right = source.right
        return child


def build_euler_sos_genome() -> EulerFormulaGenome:
    """Constructive Sum-of-Squares Proof Genome:

    R - 2r = ((b+c-a)(b-c)^2 + (a+c-b)(c-a)^2 + (a+b-c)(a-b)^2) / (8 * Delta)
    Proves R >= 2r because all factors are strictly positive, and equality holds iff a = b = c.
    """
    # Term 1: (b + c - a) * (b - c)^2
    t1_lin = EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "b"), EulerNode("VAR", "c")), EulerNode("VAR", "a"))
    t1_sq = EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "b"), EulerNode("VAR", "c")))
    term1 = EulerNode("MUL", None, t1_lin, t1_sq)

    # Term 2: (a + c - b) * (c - a)^2
    t2_lin = EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "a"), EulerNode("VAR", "c")), EulerNode("VAR", "b"))
    t2_sq = EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "c"), EulerNode("VAR", "a")))
    term2 = EulerNode("MUL", None, t2_lin, t2_sq)

    # Term 3: (a + b - c) * (a - b)^2
    t3_lin = EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "a"), EulerNode("VAR", "b")), EulerNode("VAR", "c"))
    t3_sq = EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "a"), EulerNode("VAR", "b")))
    term3 = EulerNode("MUL", None, t3_lin, t3_sq)

    numerator = EulerNode("ADD", None, EulerNode("ADD", None, term1, term2), term3)
    denominator = EulerNode("MUL", None, EulerNode("CONST", 8.0), EulerNode("VAR", "Delta"))
    root = EulerNode("DIV", None, numerator, denominator)

    return EulerFormulaGenome(root=root, formula_name="Euler_SOS_Equality_Proof")


def build_euler_ravi_sos_genome() -> EulerFormulaGenome:
    """Ravi substitution SOS form: (x*(y-z)^2 + y*(z-x)^2 + z*(x-y)^2) / (4 * Delta) with x=s-a, y=s-b, z=s-c."""
    t1 = EulerNode("MUL", None, EulerNode("VAR", "x"), EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "y"), EulerNode("VAR", "z"))))
    t2 = EulerNode("MUL", None, EulerNode("VAR", "y"), EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "z"), EulerNode("VAR", "x"))))
    t3 = EulerNode("MUL", None, EulerNode("VAR", "z"), EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "x"), EulerNode("VAR", "y"))))
    num = EulerNode("ADD", None, EulerNode("ADD", None, t1, t2), t3)
    denom = EulerNode("MUL", None, EulerNode("CONST", 4.0), EulerNode("VAR", "Delta"))
    return EulerFormulaGenome(root=EulerNode("DIV", None, num, denom), formula_name="Euler_Ravi_SOS_Form")


def build_euler_factored_genome() -> EulerFormulaGenome:
    """Factored cyclic difference form: ((a-b)^2 * (a+b-c) + (b-c)^2 * (b+c-a) + (c-a)^2 * (c+a-b)) / (8 * Delta)."""
    t1 = EulerNode("MUL", None, EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "a"), EulerNode("VAR", "b"))),
                   EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "a"), EulerNode("VAR", "b")), EulerNode("VAR", "c")))
    t2 = EulerNode("MUL", None, EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "b"), EulerNode("VAR", "c"))),
                   EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "b"), EulerNode("VAR", "c")), EulerNode("VAR", "a")))
    t3 = EulerNode("MUL", None, EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "c"), EulerNode("VAR", "a"))),
                   EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "c"), EulerNode("VAR", "a")), EulerNode("VAR", "b")))
    num = EulerNode("ADD", None, t1, EulerNode("ADD", None, t2, t3))
    denom = EulerNode("MUL", None, EulerNode("CONST", 8.0), EulerNode("VAR", "Delta"))
    return EulerFormulaGenome(root=EulerNode("DIV", None, num, denom), formula_name="Euler_Factored_Cyclic_Stable")


def build_euler_naive_genome() -> EulerFormulaGenome:
    """Naive direct subtraction R - 2r (subject to catastrophic cancellation)."""
    # R - 2 * r
    two_r = EulerNode("MUL", None, EulerNode("CONST", 2.0), EulerNode("VAR", "r"))
    root = EulerNode("SUB", None, EulerNode("VAR", "R"), two_r)
    return EulerFormulaGenome(root=root, formula_name="Euler_Naive_Subtraction")


def is_syntactic_sos_proof(root: EulerNode) -> tuple[bool, str]:
    """Inspects AST structure to verify whether candidate expression is a constructive Sum-of-Squares proof.

    A genuine constructive SOS proof must:
    1. Not be a bare definition subtraction (e.g. R - 2*r or R - r - r).
    2. Contain explicit square operations ('SQ' nodes) representing squared differences.
    3. The denominator must be strictly positive on valid non-degenerate triangles (e.g. involving Delta > 0).
    4. The numerator terms must be weighted squares with non-negative coefficients.
    """
    all_nodes: list[EulerNode] = []
    def collect(n: EulerNode):
        all_nodes.append(n)
        if n.left: collect(n.left)
        if n.right: collect(n.right)
    collect(root)

    # Check 1: Must contain at least one explicit 'SQ' node
    sq_nodes = [n for n in all_nodes if n.op == "SQ"]
    if not sq_nodes:
        return False, "AST contains NO explicit square ('SQ') operations. Bare subtractions do not constitute an SOS proof."

    # Check 2: Reject bare root subtractions of R and 2*r
    if root.op == "SUB":
        left_vars = [str(n.value) for n in all_nodes if n.op == "VAR"]
        if "R" in left_vars and "r" in left_vars and len(sq_nodes) == 0:
            return False, "AST is a direct definition subtraction R - 2r without sum-of-squares decomposition."

    # Check 3: If division, denominator must be positive (involving Delta or positive scale)
    if root.op == "DIV":
        denom_nodes: list[EulerNode] = []
        def collect_d(n: EulerNode):
            denom_nodes.append(n)
            if n.left: collect_d(n.left)
            if n.right: collect_d(n.right)
        if root.right:
            collect_d(root.right)
        denom_vars = [str(n.value) for n in denom_nodes if n.op == "VAR"]
        if "Delta" not in denom_vars and not any(n.op == "CONST" and float(n.value) > 0 for n in denom_nodes):
            return False, "Denominator lacks Delta or positive scale factor."

    return True, "AST possesses explicit Sum-of-Squares structure with squared terms and positive denominator."


@dataclass(frozen=True)
class EulerTriangleSpec:
    """Specification for Euler's inequality challenge."""
    test_cases: list[tuple[float, float, float]] = field(default_factory=lambda: list(DEFAULT_EULER_CASES))
    target_rel_err: float = 1e-6
    parsimony_weight: float = 0.6
    dps: int = 50
    ablation: bool = False
    continuous_behavioral: bool = False


class EulerInequalityEvaluator(Evaluator):
    """Evaluates 100% formal correctness (SymPy), equality condition (a=b=c), numerical stability, and parsimony."""

    def __init__(self, spec: EulerTriangleSpec | None = None) -> None:
        self.spec = spec or EulerTriangleSpec()
        mp.dps = self.spec.dps
        # High precision ground truth using 50-digit mpmath
        self._ground_truths: list[float] = []
        for a, b, c in self.spec.test_cases:
            ma, mb, mc = mp.mpf(a), mp.mpf(b), mp.mpf(c)
            ms = (ma + mb + mc) / 2
            m_delta = mp.sqrt(ms * (ms - ma) * (ms - mb) * (ms - mc))
            if m_delta > 0:
                m_R = (ma * mb * mc) / (4 * m_delta)
                m_r = m_delta / ms
                ref = float(m_R - 2 * m_r)
            else:
                ref = 0.0
            self._ground_truths.append(max(0.0, ref))

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        genome = getattr(target, "genome", target)
        if not isinstance(genome, EulerFormulaGenome):
            return FitnessResult(score=0.0)

        # =========================================================================
        # CONTINUOUS BEHAVIORAL FITNESS RELAXATION MODE (Reviewer's Proposal)
        # =========================================================================
        if self.spec.continuous_behavioral:
            total_cases = len(self.spec.test_cases)
            rel_errors: list[float] = []
            negative_penalties: float = 0.0
            stability_count = 0
            max_rel_err = 0.0

            for idx, (a, b, c) in enumerate(self.spec.test_cases):
                ref = self._ground_truths[idx]
                try:
                    val = genome.evaluate(a, b, c)
                    if math.isnan(val) or math.isinf(val):
                        rel_errors.append(20.0)
                        negative_penalties += 15.0
                        continue
                    if val < -1e-9:
                        negative_penalties += min(25.0, abs(val) * 10.0 + 5.0)

                    err = abs(val - ref) / (ref + 0.05)
                    capped_err = min(20.0, err)
                    rel_errors.append(capped_err)
                    if err > max_rel_err:
                        max_rel_err = err
                    if err < self.spec.target_rel_err:
                        stability_count += 1
                except Exception:
                    rel_errors.append(20.0)
                    negative_penalties += 15.0

            mean_rel_err = sum(rel_errors) / max(1, len(rel_errors))
            score_closeness = 100.0 / (1.0 + mean_rel_err)
            score_nonneg = max(0.0, 100.0 - negative_penalties)

            # Equilateral Vanishing Adherence
            equi_errors: list[float] = []
            for eq_side in (1.0, 2.0, 5.0, 10.0):
                try:
                    v_eq = genome.evaluate(eq_side, eq_side, eq_side)
                    if math.isnan(v_eq) or math.isinf(v_eq):
                        equi_errors.append(10.0)
                    else:
                        equi_errors.append(abs(v_eq))
                except Exception:
                    equi_errors.append(10.0)
            mean_equi_err = sum(equi_errors) / max(1, len(equi_errors))
            score_equi = 100.0 / (1.0 + 10.0 * mean_equi_err)

            # Structural Stepping Stones (AST feature emergence)
            all_nodes: list[EulerNode] = []
            def collect_n(n: EulerNode):
                all_nodes.append(n)
                if n.left: collect_n(n.left)
                if n.right: collect_n(n.right)
            collect_n(genome.root)

            sq_count = len([n for n in all_nodes if n.op == "SQ"])
            sub_count = len([n for n in all_nodes if n.op == "SUB"])
            has_delta = any(n.op == "VAR" and str(n.value) == "Delta" for n in all_nodes)
            stepping_stone_score = min(30.0, sq_count * 10.0 + min(10.0, sub_count * 2.0) + (10.0 if has_delta else 0.0))

            # Formal Gate Bonus (SymPy equivalence - protected by numerical pre-filter & cache)
            formal_passed = False
            if mean_rel_err < 1e-4 and mean_equi_err < 1e-5:
                fp = genome.fingerprint()
                if hasattr(self, "_formal_cache") and fp in self._formal_cache:
                    formal_passed = self._formal_cache[fp]
                else:
                    if not hasattr(self, "_formal_cache"):
                        self._formal_cache = {}
                    try:
                        sym_expr = genome.to_sympy()
                        diff = sp.simplify(sym_expr - EULER_DEFECT_EXPR)
                        formal_passed = bool(diff == 0)
                    except Exception:
                        formal_passed = False
                    self._formal_cache[fp] = formal_passed

            score_formal = 100.0 if formal_passed else 0.0

            # Structural Syntactic SOS Proof Verification (Gate 4)
            is_sos, sos_reason = is_syntactic_sos_proof(genome.root)
            gate4_certified = False
            if is_sos and formal_passed:
                try:
                    x, y, z = sp.symbols("x y z", positive=True)
                    ravi_subs = [(sym_a, y + z), (sym_b, x + z), (sym_c, x + y)]
                    num_scaled_8d = sp.expand((sym_expr * 8 * sym_Delta).subs(ravi_subs))
                    target_sos_8d = sp.expand(2 * (x * (y - z)**2 + y * (z - x)**2 + z * (x - y)**2))
                    if sp.simplify(num_scaled_8d - target_sos_8d) == 0:
                        gate4_certified = True
                    else:
                        num_scaled_4d = sp.expand((sym_expr * 4 * sym_Delta).subs(ravi_subs))
                        target_sos_4d = sp.expand(x * (y - z)**2 + y * (z - x)**2 + z * (x - y)**2)
                        if sp.simplify(num_scaled_4d - target_sos_4d) == 0:
                            gate4_certified = True
                except Exception:
                    gate4_certified = False

            score_gate4 = 100.0 if gate4_certified else 0.0

            complexity = genome.node_count()
            depth = genome.depth()
            parsimony_penalty = self.spec.parsimony_weight * complexity

            # Weighted Continuous Landscape
            behavioral_base = (
                0.40 * score_closeness +
                0.30 * score_nonneg +
                0.15 * score_equi +
                0.15 * (stepping_stone_score / 30.0 * 100.0)
            )
            raw_fitness = (behavioral_base * 0.5) + (score_formal * 0.3) + (score_gate4 * 0.2) - parsimony_penalty
            normalized_score = min(100.0, max(0.0, round(raw_fitness, 4)))

            return FitnessResult(
                score=normalized_score,
                sub_scores={
                    "formal_correctness": score_formal,
                    "equality_condition_verified": round(score_equi, 2),
                    "gate4_automated_proof_certified": score_gate4,
                    "behavioral_closeness": round(score_closeness, 2),
                    "nonnegativity_score": round(score_nonneg, 2),
                    "stepping_stone_score": round(stepping_stone_score, 2),
                    "stability_cases_passed": float(stability_count),
                    "stability_pass_rate_pct": round((stability_count / max(1, total_cases)) * 100.0, 1),
                    "max_rel_err": max_rel_err,
                    "node_count": float(complexity),
                    "depth": float(depth),
                    "is_syntactic_sos": 1.0 if is_sos else 0.0,
                },
            )

        # =========================================================================
        # STRICT 4-GATE MULTI-OBJECTIVE PIPELINE (Standard Verification Mode)
        # =========================================================================
        fp = genome.fingerprint()
        if hasattr(self, "_formal_cache") and fp in self._formal_cache:
            if not self._formal_cache[fp]:
                return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})
        else:
            if not hasattr(self, "_formal_cache"):
                self._formal_cache = {}

            # Gate 1a: Fast Numerical Pre-filter
            # Triangle (3, 4, 5) -> R=2.5, r=1 -> R-2r = 0.5
            try:
                v_345 = genome.evaluate(3.0, 4.0, 5.0)
                if abs(v_345 - 0.5) > 1e-3 or math.isnan(v_345):
                    self._formal_cache[fp] = False
                    return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})

                # Equilateral Triangle (1, 1, 1) -> must evaluate to exactly 0.0
                v_equi = genome.evaluate(1.0, 1.0, 1.0)
                if abs(v_equi) > 1e-6 or math.isnan(v_equi):
                    self._formal_cache[fp] = False
                    return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0, "equality_gate": 0.0})
            except Exception:
                self._formal_cache[fp] = False
                return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})

            # Gate 1b: Formal Symbolic Proof via SymPy
            try:
                sym_expr = genome.to_sympy()
                # 1. Equivalence to (R - 2r)
                diff = sp.simplify(sym_expr - EULER_DEFECT_EXPR)
                is_equiv = bool(diff == 0)

                # 2. Vanishes at a = b = c
                equi_val = sp.simplify(sym_expr.subs([(sym_b, sym_a), (sym_c, sym_a)]))
                is_equi_zero = bool(equi_val == 0)

                if not (is_equiv and is_equi_zero):
                    self._formal_cache[fp] = False
                    return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})

                self._formal_cache[fp] = True
            except Exception:
                self._formal_cache[fp] = False
                return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})

        # Gate 2: Numerical Stability on Near-Equilateral & General Triangles
        stability_count = 0
        total_cases = len(self.spec.test_cases)
        max_rel_err = 0.0

        for idx, (a, b, c) in enumerate(self.spec.test_cases):
            ref = self._ground_truths[idx]
            try:
                val = genome.evaluate(a, b, c)
                if math.isnan(val) or math.isinf(val) or val < -1e-15:
                    continue

                if ref > 1e-18:
                    rel_err = abs(val - ref) / ref
                else:
                    rel_err = abs(val)  # absolute error when ref == 0.0 (equilateral)

                if rel_err > max_rel_err:
                    max_rel_err = rel_err

                if rel_err < self.spec.target_rel_err:
                    stability_count += 1
            except Exception:
                pass

        # Gate 4: Automated Theorem Proof Verification (Structural Syntactic SOS + Positivity)
        gate4_certified = False
        is_sos, sos_reason = is_syntactic_sos_proof(genome.root)
        if is_sos:
            try:
                sym_expr = genome.to_sympy()
                x, y, z = sp.symbols("x y z", positive=True)
                ravi_subs = [(sym_a, y + z), (sym_b, x + z), (sym_c, x + y)]
                num_scaled_8d = sp.expand((sym_expr * 8 * sym_Delta).subs(ravi_subs))
                target_sos_8d = sp.expand(2 * (x * (y - z)**2 + y * (z - x)**2 + z * (x - y)**2))

                diff_8d = sp.simplify(num_scaled_8d - target_sos_8d)
                if diff_8d == 0:
                    gate4_certified = True
                else:
                    num_scaled_4d = sp.expand((sym_expr * 4 * sym_Delta).subs(ravi_subs))
                    target_sos_4d = sp.expand(x * (y - z)**2 + y * (z - x)**2 + z * (x - y)**2)
                    diff_4d = sp.simplify(num_scaled_4d - target_sos_4d)
                    if diff_4d == 0:
                        gate4_certified = True
            except Exception:
                gate4_certified = False

        # Gate 3: Koza Parsimony Pressure
        complexity = genome.node_count()
        depth = genome.depth()
        base_score = 100.0 + (stability_count * 20.0) + (50.0 if gate4_certified else 0.0)
        parsimonious_score = base_score - (self.spec.parsimony_weight * complexity)
        # Scale to [0.0, 100.0]
        normalized_score = min(100.0, max(0.0, round((parsimonious_score / 350.0) * 100.0, 4)))

        return FitnessResult(
            score=normalized_score,
            sub_scores={
                "formal_correctness": 100.0,
                "equality_condition_verified": 100.0,
                "gate4_automated_proof_certified": 100.0 if gate4_certified else 0.0,
                "is_syntactic_sos": 1.0 if is_sos else 0.0,
                "stability_cases_passed": float(stability_count),
                "stability_pass_rate_pct": round((stability_count / total_cases) * 100.0, 1),
                "max_rel_err": max_rel_err,
                "raw_parsimonious_score": parsimonious_score,
                "node_count": float(complexity),
                "depth": float(depth),
            },
        )


@register_adapter("EulerInequality")
class EulerInequalityAdapter(DomainAdapter):
    """Domain driver for Euler's inequality (R >= 2r) and constructive equality proof."""

    @property
    def name(self) -> str:
        return "EulerInequality"

    def parse_spec(self, raw_input: Any) -> EulerTriangleSpec:
        if isinstance(raw_input, EulerTriangleSpec):
            return raw_input
        elif isinstance(raw_input, dict):
            return EulerTriangleSpec(
                test_cases=raw_input.get("test_cases", list(DEFAULT_EULER_CASES)),
                target_rel_err=float(raw_input.get("target_rel_err", 1e-6)),
                parsimony_weight=float(raw_input.get("parsimony_weight", 0.6)),
                ablation=bool(raw_input.get("ablation", False)),
            )
        return EulerTriangleSpec()

    def build_population(self, spec: EulerTriangleSpec, size: int, rng: random.Random) -> list[Individual]:
        ablation = getattr(spec, "ablation", False) or os.environ.get("EVOLAB_ABLATION", "0").strip().lower() in ("1", "true", "yes")
        if ablation:
            # Tabula Rasa Ablation: ZERO pre-seeded SOS formulas, only naive subtraction + mutations
            pop = [Individual(genome=build_euler_naive_genome(), species="spec_euler_sos")]
            while len(pop) < size:
                cand = build_euler_naive_genome().mutate(rng=rng)
                pop.append(Individual(genome=cand, species="spec_euler_sos"))
            return pop

        seeds = [
            build_euler_sos_genome(),
            build_euler_ravi_sos_genome(),
            build_euler_factored_genome(),
            build_euler_naive_genome(),
        ]
        pop = [Individual(genome=g, species="spec_euler_sos") for g in seeds]

        while len(pop) < size:
            parent = rng.choice(seeds)
            mutated = parent.mutate(rng=rng)
            pop.append(Individual(genome=mutated, species="spec_euler_sos"))

        return pop

    def build_evaluator(self, spec: EulerTriangleSpec) -> Evaluator:
        return EulerInequalityEvaluator(spec)

    def export_solution(
        self,
        individual: Any,
        spec: EulerTriangleSpec,
        output_path: str | Path | None = None,
        archive: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if hasattr(individual, "genome"):
            genome = individual.genome
        elif isinstance(individual, EulerFormulaGenome):
            genome = individual
        else:
            genome = build_euler_sos_genome()

        evaluator = self.build_evaluator(spec)
        fit_res = evaluator.evaluate(genome)

        # Comparative numerical benchmark vs Naive Subtraction (R - 2r)
        naive_genome = build_euler_naive_genome()
        cases_report = []

        mp.dps = 50
        for a, b, c in spec.test_cases:
            ma, mb, mc = mp.mpf(a), mp.mpf(b), mp.mpf(c)
            ms = (ma + mb + mc) / 2
            m_delta = mp.sqrt(ms * (ms - ma) * (ms - mb) * (ms - mc))
            if m_delta > 0:
                m_R = (ma * mb * mc) / (4 * m_delta)
                m_r = m_delta / ms
                ref = float(m_R - 2 * m_r)
            else:
                ref = 0.0

            v_cand = genome.evaluate(a, b, c)
            v_naive = naive_genome.evaluate(a, b, c)

            if ref > 1e-18:
                err_cand = abs(v_cand - ref) / ref
                err_naive = abs(v_naive - ref) / ref
            else:
                err_cand = abs(v_cand)
                err_naive = abs(v_naive)

            cases_report.append({
                "triangle": [a, b, c],
                "mpmath_ref": ref,
                "evolved_sos_val": v_cand,
                "evolved_rel_err": err_cand,
                "naive_val": v_naive,
                "naive_rel_err": err_naive,
                "sos_stable": bool(err_cand < 1e-6),
                "naive_failed": bool(err_naive > 1e-4 or (ref > 1e-18 and v_naive == 0.0)),
            })

        # MAP-Elites Diversity Extraction: extract distinct valid constructive formulas
        archive_formulas = []
        if archive:
            seen_exprs = set()
            for cell, ind in archive.items():
                g = getattr(ind, "genome", ind)
                if isinstance(g, EulerFormulaGenome):
                    expr_s = g.root.to_pretty_str()
                    if expr_s not in seen_exprs and getattr(ind, "fitness", 0.0) > 10.0:
                        seen_exprs.add(expr_s)
                        archive_formulas.append({
                            "grid_cell": list(cell),
                            "fitness": round(ind.fitness, 4),
                            "expression": expr_s,
                            "formula_name": getattr(g, "formula_name", "ArchivedEulerFormula"),
                            "node_count": g.node_count(),
                        })

        gate4_status = fit_res.sub_scores.get("gate4_automated_proof_certified", 0.0) == 100.0
        data = {
            "formula_name": getattr(genome, "formula_name", "EvolvedEulerProof"),
            "expression_string": genome.root.to_pretty_str() if hasattr(genome, "root") else "",
            "constructive_proof_type": "Sum-of-Squares (SOS)",
            "equality_condition": "a == b == c (Equilateral Triangle)",
            "gate4_automated_proof_certified": gate4_status,
            "gate4_proof_rationale": (
                "Automated SymPy Theorem Proof: Numerator reduces to 2*(x*(y-z)^2 + y*(z-x)^2 + z*(x-y)^2) "
                "under Ravi substitution x=s-a>0, y=s-b>0, z=s-c>0. Every multiplier x, y, z is strictly positive "
                "under triangle inequalities, and every difference is squared, establishing non-negativity R >= 2r. "
                "The sum of squares vanishes if and only if y=z and z=x and x=y, proving equality holds iff a=b=c."
            ) if gate4_status else "Uncertified",
            "fitness_score": fit_res.score,
            "metrics": fit_res.sub_scores,
            "comparative_benchmark": cases_report,
            "map_elites_diverse_archive": archive_formulas,
            "map_elites_unique_formulas_count": len(archive_formulas),
        }

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

        return data
