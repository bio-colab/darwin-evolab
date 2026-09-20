"""test_atomic_surgical_upgrades.py — Unit test suite for Darwin-Evolab's 5 atomic surgical upgrades.

Verifies:
1. SymbolicGuard & NumericalPreFilter: Fast rejection & caching.
2. StructuralProofGuard: Rejection of bare definitions (R - 2r) vs SOS certification.
3. CurricularAnnealedEvaluator: Generation-based smooth annealing from behavioral to formal.
4. Clonal Drift Detector: Identification of artificial mean fitness rise and population stagnation.
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

from evolab.instrumentation.symbolic_guard import (
    NumericalPreFilter,
    SymbolicGuard,
    StructuralProofGuard,
)
from evolab.euler_inequality import (
    EulerNode,
    EulerFormulaGenome,
    build_euler_naive_genome,
    build_euler_sos_genome,
    build_euler_ravi_sos_genome,
    EulerInequalityEvaluator,
    EulerTriangleSpec,
)
from evolab.fitness import CurricularAnnealedEvaluator
from evolab.engine_telemetry import calculate_clonal_drift_metrics
from evolab.engine import Individual


class TestAtomicSurgicalUpgrades(unittest.TestCase):

    def test_numerical_prefilter_and_symbolic_guard(self):
        """Verify that NumericalPreFilter filters out invalid trees without calling SymPy."""
        # Target function: f(a, b, c) = (a + b + c) / 2
        pre_filter = NumericalPreFilter(
            test_points=[(3.0, 4.0, 5.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
            target_fn=lambda a, b, c: (a + b + c) / 2.0,
            tol=1e-3,
        )

        # Candidate 1: completely wrong function f(a,b,c) = a * b
        eval_wrong = lambda a, b, c: a * b
        self.assertFalse(pre_filter.passes(eval_wrong))

        # Candidate 2: correct function f(a,b,c) = 0.5 * (a + b + c)
        eval_correct = lambda a, b, c: 0.5 * (a + b + c)
        self.assertTrue(pre_filter.passes(eval_correct))

        # Test SymbolicGuard with caching and prefilter
        guard = SymbolicGuard()
        import sympy as sp
        x, y, z = sp.symbols("x y z")
        target_sym = (x + y + z) / 2

        # Wrap candidate 1 as AST
        class DummyWrongAST:
            def fingerprint(self): return "fp_wrong"
            def to_sympy(self): return x * y

        is_eq = guard.verify_equivalence(
            candidate_ast=DummyWrongAST(),
            eval_fn=eval_wrong,
            target_expr=target_sym,
            pre_filter=pre_filter,
        )
        self.assertFalse(is_eq)
        # Verify prefilter rejected it without sympy
        self.assertEqual(guard.stats["prefilter_rejections"], 1)
        self.assertEqual(guard.stats["sympy_evaluations"], 0)

        # Second call: Cache hit
        is_eq_cached = guard.verify_equivalence(
            candidate_ast=DummyWrongAST(),
            eval_fn=eval_wrong,
            target_expr=target_sym,
            pre_filter=pre_filter,
        )
        self.assertFalse(is_eq_cached)
        self.assertEqual(guard.stats["cache_hits"], 1)

    def test_structural_proof_guard_discrimination(self):
        """Verify that StructuralProofGuard rejects R-2r and certifies genuine SOS."""
        naive = build_euler_naive_genome()
        sos = build_euler_sos_genome()
        ravi = build_euler_ravi_sos_genome()

        # Naive definition R - 2r
        is_naive_sos, reason_naive = StructuralProofGuard.inspect_sum_of_squares(naive.root)
        self.assertFalse(is_naive_sos)
        self.assertIn("NO explicit square", reason_naive)

        # Constructive Sum-of-Squares proof
        is_sos_valid, reason_sos = StructuralProofGuard.inspect_sum_of_squares(sos.root)
        self.assertTrue(is_sos_valid)
        self.assertIn("explicit Sum-of-Squares", reason_sos)

        # Ravi Substitution SOS proof
        is_ravi_valid, reason_ravi = StructuralProofGuard.inspect_sum_of_squares(ravi.root)
        self.assertTrue(is_ravi_valid)

    def test_curricular_annealed_evaluator(self):
        """Verify dynamic generation-based annealing from behavioral to formal."""
        spec_beh = EulerTriangleSpec(continuous_behavioral=True)
        spec_form = EulerTriangleSpec(continuous_behavioral=False)

        eval_beh = EulerInequalityEvaluator(spec_beh)
        eval_form = EulerInequalityEvaluator(spec_form)

        annealed = CurricularAnnealedEvaluator(
            behavioral_evaluator=eval_beh,
            formal_evaluator=eval_form,
            total_generations=40,
            annealing_schedule="linear",
        )

        # Generation 1: formal_weight == 0.0 (Pure behavioral)
        annealed.set_generation(1)
        self.assertAlmostEqual(annealed.formal_weight, 0.0, places=4)

        # Generation 20: mid-way annealing ~ 0.487
        annealed.set_generation(20)
        self.assertAlmostEqual(annealed.formal_weight, 19.0 / 39.0, places=3)

        # Generation 40: formal_weight == 1.0 (Pure formal)
        annealed.set_generation(40)
        self.assertAlmostEqual(annealed.formal_weight, 1.0, places=4)

        # Test evaluation of a random tree at Gen 1 vs Gen 40
        rand_node = EulerNode("ADD", None, EulerNode("VAR", "a"), EulerNode("VAR", "b"))
        rand_genome = EulerFormulaGenome(root=rand_node, formula_name="RandTree")

        # At Gen 1, random tree gets behavioral partial credit > 0.0
        annealed.set_generation(1)
        res_g1 = annealed.evaluate(rand_genome)
        self.assertGreater(res_g1.score, 5.0)

        # At Gen 40, random tree gets strict formal evaluation = 0.0
        annealed.set_generation(40)
        res_g40 = annealed.evaluate(rand_genome)
        self.assertEqual(res_g40.score, 0.0)

    def test_clonal_drift_detector(self):
        """Verify detection of artificial mean fitness rise caused by clonal drift."""
        # Population of 80 individuals: 75 identical clones of Naive + 5 random trees
        naive = build_euler_naive_genome()
        pop = [Individual(genome=naive.clone(), species="spec_euler") for _ in range(75)]
        for _ in range(5):
            rand_node = EulerNode("VAR", "a")
            pop.append(Individual(genome=EulerFormulaGenome(root=rand_node), species="spec_euler"))

        # Case 1: Best fitness is stagnated for 5 generations
        history = [62.0, 62.0, 62.0, 62.0, 62.0]
        metrics = calculate_clonal_drift_metrics(pop, best_fitness_history=history, stagnation_window=5)

        self.assertEqual(metrics["max_clone_count"], 75)
        self.assertGreater(metrics["clonal_ratio"], 0.8)
        self.assertTrue(metrics["is_clonal_drift_stagnation"])

        # Case 2: Diverse population with no stagnation
        diverse_pop = []
        for i in range(50):
            node = EulerNode("CONST", float(i))
            diverse_pop.append(Individual(genome=EulerFormulaGenome(root=node), species="spec_euler"))

        history_improving = [10.0, 20.0, 30.0, 40.0, 50.0]
        metrics_div = calculate_clonal_drift_metrics(diverse_pop, best_fitness_history=history_improving)
        self.assertFalse(metrics_div["is_clonal_drift_stagnation"])
        self.assertEqual(metrics_div["clonal_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
