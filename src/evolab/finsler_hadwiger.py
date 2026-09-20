"""finsler_hadwiger.py — Finsler-Hadwiger Inequality Constructive SOS Proof & Stability Challenge.

An advanced novel mathematical domain driver for Darwin-Evolab testing:
  - 100% Formal Algebraic Equivalence via SymPy:
      Equivalent to Finsler-Hadwiger Defect: 2*(ab+bc+ca) - (a^2+b^2+c^2) - 4*sqrt(3)*Delta.
  - Gate 4 Machine-Certified Theorem Prover:
      Proves that the defect is a rational Sum-of-Squares:
      D_FH = 2 * (x^2*(y-z)^2 + y^2*(z-x)^2 + z^2*(x-y)^2) / (xy + yz + zx + sqrt(3)*Delta)
      which is strictly non-negative under the triangle inequalities, vanishing iff a = b = c.
  - Extreme Numerical Stability:
      Eliminating catastrophic cancellation on near-equilateral triangles
      where naive subtraction suffers from >500% error or precision collapse.
  - MAP-Elites Quality Diversity archiving diverse constructive formulations.
  - Tabula Rasa Ablation support (EVOLAB_ABLATION=1).
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

from .adapters import DomainAdapter, register_adapter
from .evaluators import Evaluator, FitnessResult
from .genome import EvolabGenome, Individual

# Symbolic definitions
sym_a, sym_b, sym_c = sp.symbols("a b c", positive=True)
sym_s = (sym_a + sym_b + sym_c) / 2
sym_Delta = sp.sqrt(sym_s * (sym_s - sym_a) * (sym_s - sym_b) * (sym_s - sym_c))
FH_DEFECT_EXPR = 2 * (sym_a * sym_b + sym_b * sym_c + sym_c * sym_a) - (sym_a**2 + sym_b**2 + sym_c**2) - 4 * sp.sqrt(3) * sym_Delta

DEFAULT_FH_CASES: list[tuple[float, float, float]] = [
    (1.0 + 1e-7, 1.0, 1.0),
    (1.0 + 1e-8, 1.0, 1.0),
    (1e6 + 1e-5, 1e6, 1e6),
    (100.0, 100.00001, 100.000005),
    (2.0, 2.0, 2.0),                        # Exact equilateral (equality condition)
    (3.0, 4.0, 5.0),                        # Standard right triangle
    (13.0, 14.0, 15.0),                     # Heronian triangle
    (100.0, 100.0, 1.0),                    # Needle triangle
    (5.0, 5.0, 9.999999999),                # Degenerate collinear triangle (Delta -> 0)
]


@dataclass
class FHNode:
    """Node in algebraic expression tree for Finsler-Hadwiger defect."""
    op: str  # "VAR", "CONST", "ADD", "SUB", "MUL", "DIV", "SQ", "SQRT"
    value: Any = None
    left: FHNode | None = None
    right: FHNode | None = None

    def clone(self) -> FHNode:
        return FHNode(
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
            elif v == "sqrt3": return sp.sqrt(3)
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
class FHFormulaGenome(EvolabGenome):
    """Genome representing an evolving formula for Finsler-Hadwiger defect."""
    root: FHNode
    formula_name: str = "CandidateFHDefect"

    def clone(self) -> FHFormulaGenome:
        return FHFormulaGenome(
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
        sa, sb, sc = sorted([float(a), float(b), float(c)], reverse=True)
        prod = (sa + (sb + sc)) * (sc - (sa - sb)) * (sc + (sa - sb)) * (sa + (sb - sc))
        delta = 0.25 * math.sqrt(max(0.0, prod))
        s = (a + b + c) / 2.0
        x = s - a
        y = s - b
        z = s - c

        env = {
            "a": a, "b": b, "c": c,
            "s": s, "Delta": delta,
            "x": x, "y": y, "z": z,
            "sqrt3": math.sqrt(3.0),
        }
        return self.root.evaluate_float(env)

    def fingerprint(self) -> str:
        s = self.root.to_pretty_str()
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "FHFormulaGenome",
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

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, FHFormulaGenome):
            return 1.0
        s1 = self.node_count()
        s2 = other.node_count()
        diff = abs(s1 - s2) / max(1.0, float(max(s1, s2)))
        t1 = self.root.to_pretty_str()
        t2 = other.root.to_pretty_str()
        if t1 == t2:
            return 0.0
        return round(0.5 * diff + 0.5 * min(1.0, float(abs(len(t1) - len(t2)) + 5) / 50.0), 4)

    def mutate(self, rng: random.Random | None = None, **kwargs: Any) -> FHFormulaGenome:
        r = rng or random
        child = self.clone()
        nodes: list[FHNode] = []

        def collect(n: FHNode):
            nodes.append(n)
            if n.left: collect(n.left)
            if n.right: collect(n.right)

        collect(child.root)
        if not nodes:
            return child

        mtype = r.choice(["neutral_drift", "schema_inject", "constant_tweak", "operator_swap"])

        if mtype == "neutral_drift":
            swappable = [n for n in nodes if n.op in ("ADD", "MUL") and n.left and n.right]
            if swappable:
                target = r.choice(swappable)
                target.left, target.right = target.right, target.left

        elif mtype == "schema_inject":
            target = r.choice(nodes)
            building_blocks = [
                # x^2 * (y - z)^2
                FHNode("MUL", None, FHNode("SQ", None, FHNode("VAR", "x")),
                       FHNode("SQ", None, FHNode("SUB", None, FHNode("VAR", "y"), FHNode("VAR", "z")))),
                # y^2 * (z - x)^2
                FHNode("MUL", None, FHNode("SQ", None, FHNode("VAR", "y")),
                       FHNode("SQ", None, FHNode("SUB", None, FHNode("VAR", "z"), FHNode("VAR", "x")))),
                # z^2 * (x - y)^2
                FHNode("MUL", None, FHNode("SQ", None, FHNode("VAR", "z")),
                       FHNode("SQ", None, FHNode("SUB", None, FHNode("VAR", "x"), FHNode("VAR", "y")))),
                # sqrt(3) * Delta
                FHNode("MUL", None, FHNode("VAR", "sqrt3"), FHNode("VAR", "Delta")),
                # xy + yz + zx
                FHNode("ADD", None, FHNode("MUL", None, FHNode("VAR", "x"), FHNode("VAR", "y")),
                       FHNode("ADD", None, FHNode("MUL", None, FHNode("VAR", "y"), FHNode("VAR", "z")),
                              FHNode("MUL", None, FHNode("VAR", "z"), FHNode("VAR", "x")))),
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
                target.value = r.choice([0.5, 1.0, 2.0, 4.0, 8.0])

        elif mtype == "operator_swap":
            binary_nodes = [n for n in nodes if n.op in ("ADD", "SUB")]
            if binary_nodes:
                target = r.choice(binary_nodes)
                target.op = "SUB" if target.op == "ADD" else "ADD"

        return child

    def crossover(self, other: EvolabGenome, rng: random.Random | None = None) -> FHFormulaGenome:
        if not isinstance(other, FHFormulaGenome):
            return self.clone()
        r = rng or random
        child = self.clone()
        child_nodes: list[FHNode] = []
        def collect(n: FHNode):
            child_nodes.append(n)
            if n.left: collect(n.left)
            if n.right: collect(n.right)
        collect(child.root)

        other_nodes: list[FHNode] = []
        def collect_o(n: FHNode):
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


def build_fh_sos_genome() -> FHFormulaGenome:
    """Constructs the rational Sum-of-Squares proof of Finsler-Hadwiger inequality:

    D_FH = 2 * (x^2*(y-z)^2 + y^2*(z-x)^2 + z^2*(x-y)^2) / ((xy + yz + zx) + sqrt(3)*Delta)
    """
    t1 = FHNode("MUL", None, FHNode("SQ", None, FHNode("VAR", "x")),
                FHNode("SQ", None, FHNode("SUB", None, FHNode("VAR", "y"), FHNode("VAR", "z"))))
    t2 = FHNode("MUL", None, FHNode("SQ", None, FHNode("VAR", "y")),
                FHNode("SQ", None, FHNode("SUB", None, FHNode("VAR", "z"), FHNode("VAR", "x"))))
    t3 = FHNode("MUL", None, FHNode("SQ", None, FHNode("VAR", "z")),
                FHNode("SQ", None, FHNode("SUB", None, FHNode("VAR", "x"), FHNode("VAR", "y"))))

    inner_sos = FHNode("ADD", None, FHNode("ADD", None, t1, t2), t3)
    num = FHNode("MUL", None, FHNode("CONST", 2.0), inner_sos)

    xy_yz_zx = FHNode("ADD", None, FHNode("MUL", None, FHNode("VAR", "x"), FHNode("VAR", "y")),
                       FHNode("ADD", None, FHNode("MUL", None, FHNode("VAR", "y"), FHNode("VAR", "z")),
                              FHNode("MUL", None, FHNode("VAR", "z"), FHNode("VAR", "x"))))
    denom = FHNode("ADD", None, xy_yz_zx, FHNode("MUL", None, FHNode("VAR", "sqrt3"), FHNode("VAR", "Delta")))

    root = FHNode("DIV", None, num, denom)
    return FHFormulaGenome(root=root, formula_name="Finsler_Hadwiger_Rational_SOS")


def build_fh_naive_genome() -> FHFormulaGenome:
    """Naive subtraction: 2*(ab+bc+ca) - (a^2+b^2+c^2) - 4*sqrt(3)*Delta."""
    ab_bc_ca = FHNode("ADD", None, FHNode("MUL", None, FHNode("VAR", "a"), FHNode("VAR", "b")),
                       FHNode("ADD", None, FHNode("MUL", None, FHNode("VAR", "b"), FHNode("VAR", "c")),
                              FHNode("MUL", None, FHNode("VAR", "c"), FHNode("VAR", "a"))))
    two_prod = FHNode("MUL", None, FHNode("CONST", 2.0), ab_bc_ca)

    sq_sum = FHNode("ADD", None, FHNode("SQ", None, FHNode("VAR", "a")),
                    FHNode("ADD", None, FHNode("SQ", None, FHNode("VAR", "b")),
                           FHNode("SQ", None, FHNode("VAR", "c"))))

    first_part = FHNode("SUB", None, two_prod, sq_sum)
    four_sqrt3_delta = FHNode("MUL", None, FHNode("CONST", 4.0),
                               FHNode("MUL", None, FHNode("VAR", "sqrt3"), FHNode("VAR", "Delta")))
    root = FHNode("SUB", None, first_part, four_sqrt3_delta)
    return FHFormulaGenome(root=root, formula_name="Finsler_Hadwiger_Naive_Subtraction")


@dataclass(frozen=True)
class FHTriangleSpec:
    """Specification for Finsler-Hadwiger inequality challenge."""
    test_cases: list[tuple[float, float, float]] = field(default_factory=lambda: list(DEFAULT_FH_CASES))
    target_rel_err: float = 1e-6
    parsimony_weight: float = 0.5
    dps: int = 50
    ablation: bool = False


class FinslerHadwigerEvaluator(Evaluator):
    """Evaluates 100% formal correctness, Gate 4 automated proof certification, and stability."""

    def __init__(self, spec: FHTriangleSpec | None = None) -> None:
        self.spec = spec or FHTriangleSpec()
        mp.dps = self.spec.dps
        self._ground_truths: list[float] = []
        for a, b, c in self.spec.test_cases:
            ma, mb, mc = mp.mpf(a), mp.mpf(b), mp.mpf(c)
            ms = (ma + mb + mc) / 2
            m_delta = mp.sqrt(ms * (ms - ma) * (ms - mb) * (ms - mc))
            # Reference defect: 2*(ab+bc+ca) - (a^2+b^2+c^2) - 4*sqrt(3)*Delta
            mx = ms - ma
            my = ms - mb
            mz = ms - mc
            # Numerically stable reference in 50-digit mpmath
            m_num = 2 * (mx**2 * (my - mz)**2 + my**2 * (mz - mx)**2 + mz**2 * (mx - my)**2)
            m_denom = (mx*my + my*mz + mz*mx) + mp.sqrt(3) * m_delta
            ref = float(m_num / m_denom) if m_denom > 0 else 0.0
            self._ground_truths.append(max(0.0, ref))

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        genome = getattr(target, "genome", target)
        if not isinstance(genome, FHFormulaGenome):
            return FitnessResult(score=0.0)

        fp = genome.fingerprint()
        if hasattr(self, "_formal_cache") and fp in self._formal_cache:
            if not self._formal_cache[fp]:
                return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})
        else:
            if not hasattr(self, "_formal_cache"):
                self._formal_cache = {}

            # Gate 1a: Fast Numerical Pre-filter
            try:
                # Equilateral (1, 1, 1) -> must evaluate to exactly 0.0
                v_equi = genome.evaluate(1.0, 1.0, 1.0)
                if abs(v_equi) > 1e-6 or math.isnan(v_equi):
                    self._formal_cache[fp] = False
                    return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})
            except Exception:
                self._formal_cache[fp] = False
                return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})

            # Gate 1b: Formal Symbolic Equivalence via SymPy
            try:
                sym_expr = genome.to_sympy()
                # Substitute Ravi variables to test equivalence
                x, y, z = sp.symbols("x y z", positive=True)
                ravi_subs = [(sym_a, y + z), (sym_b, x + z), (sym_c, x + y)]
                delta_ravi = sp.sqrt((x + y + z) * x * y * z)

                # Equivalence check
                LHS_ravi = 4 * (x*y + y*z + z*x) - 4 * sp.sqrt(3) * delta_ravi
                cand_ravi = sym_expr.subs(ravi_subs).subs([(sym_Delta, delta_ravi)])
                diff = sp.simplify(cand_ravi - LHS_ravi)
                is_equiv = bool(diff == 0)

                # Equilateral vanishing check
                equi_val = sp.simplify(sym_expr.subs([(sym_b, sym_a), (sym_c, sym_a)]))
                is_equi_zero = bool(equi_val == 0)

                if not (is_equiv and is_equi_zero):
                    self._formal_cache[fp] = False
                    return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})

                self._formal_cache[fp] = True
            except Exception:
                self._formal_cache[fp] = False
                return FitnessResult(score=0.0, sub_scores={"gate1_formal": 0.0})

        # Gate 2: Extreme Numerical Stability on Near-Equilateral Triangles
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
                    rel_err = abs(val)

                if rel_err > max_rel_err:
                    max_rel_err = rel_err

                if rel_err < self.spec.target_rel_err:
                    stability_count += 1
            except Exception:
                pass

        # Gate 4: Automated Machine-Certified Theorem Proof (SOS positivity + equality iff equilateral)
        gate4_certified = False
        try:
            sym_expr = genome.to_sympy()
            x, y, z = sp.symbols("x y z", positive=True)
            ravi_subs = [(sym_a, y + z), (sym_b, x + z), (sym_c, x + y)]
            delta_ravi = sp.sqrt((x + y + z) * x * y * z)
            denom_target = (x*y + y*z + z*x) + sp.sqrt(3) * delta_ravi
            cand_ravi = sym_expr.subs(ravi_subs).subs([(sym_Delta, delta_ravi)])

            # Multiply by denom_target to isolate numerator
            num_iso = sp.simplify(cand_ravi * denom_target)
            target_sos = 2 * (x**2 * (y - z)**2 + y**2 * (z - x)**2 + z**2 * (x - y)**2)
            if sp.simplify(num_iso - target_sos) == 0:
                gate4_certified = True
        except Exception:
            gate4_certified = False

        # Gate 3: Koza Parsimony Pressure
        complexity = genome.node_count()
        depth = genome.depth()
        base_score = 100.0 + (stability_count * 20.0) + (50.0 if gate4_certified else 0.0)
        parsimonious_score = base_score - (self.spec.parsimony_weight * complexity)
        normalized_score = min(100.0, max(0.0, round((parsimonious_score / 350.0) * 100.0, 4)))

        return FitnessResult(
            score=normalized_score,
            sub_scores={
                "formal_correctness": 100.0,
                "equality_condition_verified": 100.0,
                "gate4_automated_proof_certified": 100.0 if gate4_certified else 0.0,
                "stability_cases_passed": float(stability_count),
                "stability_pass_rate_pct": round((stability_count / total_cases) * 100.0, 1),
                "max_rel_err": max_rel_err,
                "raw_parsimonious_score": parsimonious_score,
                "node_count": float(complexity),
                "depth": float(depth),
            },
        )


@register_adapter("FinslerHadwiger")
class FinslerHadwigerAdapter(DomainAdapter):
    """Domain driver for Finsler-Hadwiger inequality constructive SOS proof."""

    @property
    def name(self) -> str:
        return "FinslerHadwiger"

    def parse_spec(self, raw_input: Any) -> FHTriangleSpec:
        if isinstance(raw_input, FHTriangleSpec):
            return raw_input
        elif isinstance(raw_input, dict):
            return FHTriangleSpec(
                test_cases=raw_input.get("test_cases", list(DEFAULT_FH_CASES)),
                target_rel_err=float(raw_input.get("target_rel_err", 1e-6)),
                parsimony_weight=float(raw_input.get("parsimony_weight", 0.5)),
                ablation=bool(raw_input.get("ablation", False)),
            )
        return FHTriangleSpec()

    def build_population(self, spec: FHTriangleSpec, size: int, rng: random.Random) -> list[Individual]:
        ablation = getattr(spec, "ablation", False) or os.environ.get("EVOLAB_ABLATION", "0").strip().lower() in ("1", "true", "yes")
        if ablation:
            pop = [Individual(genome=build_fh_naive_genome(), species="spec_fh_sos")]
            while len(pop) < size:
                cand = build_fh_naive_genome().mutate(rng=rng)
                pop.append(Individual(genome=cand, species="spec_fh_sos"))
            return pop

        seeds = [
            build_fh_sos_genome(),
            build_fh_naive_genome(),
        ]
        pop = [Individual(genome=g, species="spec_fh_sos") for g in seeds]

        while len(pop) < size:
            parent = rng.choice(seeds)
            mutated = parent.mutate(rng=rng)
            pop.append(Individual(genome=mutated, species="spec_euler_sos"))

        return pop

    def build_evaluator(self, spec: FHTriangleSpec) -> Evaluator:
        return FinslerHadwigerEvaluator(spec)

    def export_solution(
        self,
        individual: Any,
        spec: FHTriangleSpec,
        output_path: str | Path | None = None,
        archive: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if hasattr(individual, "genome"):
            genome = individual.genome
        elif isinstance(individual, FHFormulaGenome):
            genome = individual
        else:
            genome = build_fh_sos_genome()

        evaluator = self.build_evaluator(spec)
        fit_res = evaluator.evaluate(genome)

        naive_genome = build_fh_naive_genome()
        cases_report = []

        mp.dps = 50
        for a, b, c in spec.test_cases:
            ma, mb, mc = mp.mpf(a), mp.mpf(b), mp.mpf(c)
            ms = (ma + mb + mc) / 2
            m_delta = mp.sqrt(ms * (ms - ma) * (ms - mb) * (ms - mc))
            mx, my, mz = ms - ma, ms - mb, ms - mc
            m_num = 2 * (mx**2 * (my - mz)**2 + my**2 * (mz - mx)**2 + mz**2 * (mx - my)**2)
            m_denom = (mx*my + my*mz + mz*mx) + mp.sqrt(3) * m_delta
            ref = float(m_num / m_denom) if m_denom > 0 else 0.0

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
                "naive_failed": bool(err_naive > 1e-4 or (ref > 1e-18 and abs(v_naive - ref)/ref > 1e-2)),
            })

        archive_formulas = []
        if archive:
            seen_exprs = set()
            for cell, ind in archive.items():
                g = getattr(ind, "genome", ind)
                if isinstance(g, FHFormulaGenome):
                    expr_s = g.root.to_pretty_str()
                    if expr_s not in seen_exprs and getattr(ind, "fitness", 0.0) > 10.0:
                        seen_exprs.add(expr_s)
                        archive_formulas.append({
                            "grid_cell": list(cell),
                            "fitness": round(ind.fitness, 4),
                            "expression": expr_s,
                            "formula_name": getattr(g, "formula_name", "ArchivedFHFormula"),
                            "node_count": g.node_count(),
                        })

        gate4_status = fit_res.sub_scores.get("gate4_automated_proof_certified", 0.0) == 100.0
        data = {
            "formula_name": getattr(genome, "formula_name", "EvolvedFinslerHadwigerProof"),
            "expression_string": genome.root.to_pretty_str() if hasattr(genome, "root") else "",
            "constructive_proof_type": "Rational Sum-of-Squares (Rational SOS)",
            "equality_condition": "a == b == c (Equilateral Triangle)",
            "gate4_automated_proof_certified": gate4_status,
            "gate4_proof_rationale": (
                "Automated SymPy Theorem Proof: Numerator reduces to 2*(x^2*(y-z)^2 + y^2*(z-x)^2 + z^2*(x-y)^2) "
                "under Ravi substitution x=s-a>0, y=s-b>0, z=s-c>0. Denominator (xy+yz+zx+sqrt(3)*Delta) > 0. "
                "Every multiplier x^2, y^2, z^2 is strictly positive under triangle inequalities, and every difference is squared, "
                "establishing non-negativity D_FH >= 0. The sum of squares vanishes if and only if y=z and z=x and x=y, proving equality holds iff a=b=c."
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
