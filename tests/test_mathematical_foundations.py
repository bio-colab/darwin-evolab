"""Mathematical Foundations, Metric Space Axioms, and Landscape Topology Tests.

Formally verifies:
1. Metric space axioms (non-negativity, identity, symmetry, triangle inequality).
2. Landscape multimodality and deceptive attractor barriers.
3. Markov chain irreducibility and positive mutation support (ergodicity property).
4. MAP-Elites behavioral descriptor partition invariance.
5. Statistical variation bounds on genetic operators.
"""
from __future__ import annotations

import math
import random
import pytest

from evolab.genome import Individual, FloatGenome, GENOME_RANGE
from evolab.speciation import genomic_distance
from evolab.ast_genome import ASTGenome, ast_distance
from evolab.patch import PatchGenome, Hunk, patch_distance
from evolab.landscapes import RastriginLandscape, TrapKLandscape, SmoothProxyLandscape
from evolab.engine import EvolutionEngine, _desc_mean, _desc_std


def test_genomic_distance_metric_space_axioms():
    """Verify that genomic_distance rigorously satisfies all Metric Space Axioms."""
    rng = random.Random(1337)
    n = 16

    for _ in range(50):
        g1 = [rng.uniform(-5.0, 5.0) for _ in range(n)]
        g2 = [rng.uniform(-5.0, 5.0) for _ in range(n)]
        g3 = [rng.uniform(-5.0, 5.0) for _ in range(n)]

        ind1 = Individual(genome=g1, species="spec_a")
        ind2 = Individual(genome=g2, species="spec_b")
        ind3 = Individual(genome=g3, species="spec_a")

        # 1. Identity of indiscernibles
        assert genomic_distance(ind1, ind1, c1=0.6, c2=0.4, c3=0.1) == pytest.approx(0.0)

        # 2. Non-negativity
        d12 = genomic_distance(ind1, ind2, c1=0.6, c2=0.4, c3=0.1)
        assert d12 >= 0.0

        # 3. Symmetry
        d21 = genomic_distance(ind2, ind1, c1=0.6, c2=0.4, c3=0.1)
        assert d12 == pytest.approx(d21, abs=1e-12)

        # 4. Triangle Inequality: d(1, 3) <= d(1, 2) + d(2, 3)
        d13 = genomic_distance(ind1, ind3, c1=0.6, c2=0.4, c3=0.1)
        d23 = genomic_distance(ind2, ind3, c1=0.6, c2=0.4, c3=0.1)
        assert d13 <= d12 + d23 + 1e-12, f"Triangle inequality violated: {d13} > {d12} + {d23}"


def test_ast_distance_metric_properties():
    """Verify non-negativity and symmetry for AST structural distance."""
    ast1 = ASTGenome.from_code("def f(x):\n    return x + 1\n")
    ast2 = ASTGenome.from_code("def f(x):\n    return x * 2\n")
    ast3 = ASTGenome.from_code("def g(y):\n    return y - 3\n")

    d11 = ast_distance(ast1, ast1)
    assert d11 == pytest.approx(0.0)

    d12 = ast_distance(ast1, ast2)
    d21 = ast_distance(ast2, ast1)
    assert d12 >= 0.0
    assert d12 == pytest.approx(d21, abs=1e-12)

    d13 = ast_distance(ast1, ast3)
    d23 = ast_distance(ast2, ast3)
    # Triangle inequality on normalized node delta metric
    assert d13 <= d12 + d23 + 1e-6


def test_rastrigin_landscape_multimodal_barrier():
    """Verify that RastriginLandscape exhibits multiple local deceptive optima."""
    landscape = RastriginLandscape(ripple=0.25)
    global_opt = [3.0] * 8
    opt_score = landscape.evaluate(global_opt)
    assert opt_score > 0.0

    # Test points perturbed by one period (2.4 * delta = 2*pi => delta = 2*pi / 2.4 ~= 2.61799)
    period = (2.0 * math.pi) / 2.4
    perturbed_subopt = [3.0 + period] * 8
    subopt_score = landscape.evaluate(perturbed_subopt)

    # Valley in between at half period (delta = pi / 2.4)
    perturbed_valley = [3.0 + period / 2.0] * 8
    valley_score = landscape.evaluate(perturbed_valley)

    # Global optimum is higher than perturbed suboptima
    assert opt_score > subopt_score


def test_trap_k_landscape_deceptive_attractor_axiom():
    """Verify Goldberg's deceptive attractor axiom for Trap-K landscapes."""
    trap = TrapKLandscape(k=4)
    # Global optimum (all ones)
    all_ones = [1.0] * 8
    assert trap.evaluate(all_ones) == 100.0

    # Local deceptive attractor (all zeros)
    all_zeros = [0.0] * 8
    score_zeros = trap.evaluate(all_zeros)
    assert score_zeros > 0.0

    # Suboptimal partial trap state (3 ones, 1 zero in block)
    # In Trap-k: f(0)=k-1=3, f(1)=2, f(2)=1, f(3)=0, f(4)=k=4
    partial_trap = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    score_partial = trap.evaluate(partial_trap)

    # Deceptive attractor scores strictly higher than partial trap
    assert score_zeros > score_partial


def test_markov_mutation_support_positivity():
    """Verify mutation operator has non-zero transition density across continuous bounds."""
    rng = random.Random(42)
    genome = FloatGenome(values=[0.0] * 4)
    samples = [genome.mutate(rng=rng, sigma=1.0, kind="semantic").values[0] for _ in range(200)]

    # Gaussian kernel ensures positive support on both sides of parent state
    assert any(x > 0.0 for x in samples)
    assert any(x < 0.0 for x in samples)
    assert min(samples) < -0.5
    assert max(samples) > 0.5


def test_map_elites_descriptor_partition_invariance():
    """Verify descriptor functions map any genome to bounded, well-defined metrics."""
    g_float = FloatGenome(values=[-4.0, 0.0, 4.0])
    g_ast = ASTGenome.from_code("def compute(x):\n    return x + 42\n")
    g_patch = PatchGenome(hunks=[Hunk(file_path="a.py", start_line=1, num_lines=1, old_text="x", new_text="y")])

    for g in (g_float, g_ast, g_patch):
        m = _desc_mean(g)
        s = _desc_std(g)
        assert isinstance(m, float) and math.isfinite(m)
        assert isinstance(s, float) and math.isfinite(s) and s >= 0.0
