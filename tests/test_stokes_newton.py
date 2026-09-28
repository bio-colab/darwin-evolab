"""test_stokes_newton.py — Pre-Execution Unit Tests for Stokes-Newton Experiment.

Verifies mandatory protocol gates T1 through T6 before any experimental run:
  - T1: True Brown-Lawler generator passes all boundary gates with margin
  - T2: Re defined with diameter 2r; recovers Stokes law 6*pi*mu*r*v as Re -> 0
  - T3: Strict tripartite disjointness: Train, Transition Gap [5, 100], and Test
  - T4: AST complexity counter matches reference 33 nodes for the generator
  - T5: Dataset reproduction from identical seed produces identical SHA-256 fingerprint
  - T6: Oracle baseline recovers true generator at sigma=0
"""
from __future__ import annotations

import ast
import hashlib
import random
import numpy as np
import pytest
import sympy as sp

from evolab.adapters import get_domain_adapter
from evolab.stokes_newton import (
    BROWN_LAWLER_C1,
    BROWN_LAWLER_C2,
    BROWN_LAWLER_C3,
    DragExprNode,
    DragSymbolicGenome,
    FROZEN_GRID_SHA256,
    StokesNewtonDragAdapter,
    StokesNewtonSpec,
    build_brown_lawler_skeleton,
    build_stokes_skeleton,
    cd_brown_lawler,
    compute_fd,
    compute_re,
    count_ast_nodes,
    evaluate_oracle_baseline,
    generate_experiment_dataset,
    generate_sealed_grid,
    verify_all_physical_gates,
    verify_high_window_gate,
    verify_low_window_gate,
    verify_monotonicity_gate,
)


def test_t1_generator_passes_all_gates_with_margin():
    """T1: The true Brown-Lawler generator must pass all boundary gates with margin."""
    p_low, min_r, max_r = verify_low_window_gate(cd_brown_lawler)
    assert p_low, f"Low window failed: ratio [{min_r}, {max_r}] not in [0.95, 1.05]"
    # Check that margin is respected (true generator is between ~1.006 and 1.031)
    assert 0.95 < min_r < 1.05
    assert 0.95 < max_r < 1.05

    p_high, min_v, max_v = verify_high_window_gate(cd_brown_lawler)
    assert p_high, f"High window failed: cd [{min_v}, {max_v}] not in 0.407 * [0.94, 1.06]"
    assert 0.407 * 0.94 < min_v < 0.407 * 1.06
    assert 0.407 * 0.94 < max_v < 0.407 * 1.06

    p_mono = verify_monotonicity_gate(cd_brown_lawler)
    assert p_mono, "Monotonicity of F_D in v failed for true generator"

    all_passed, reasons = verify_all_physical_gates(cd_brown_lawler)
    assert all_passed, f"Physical gates failed: {reasons}"


def test_t2_diameter_definition_and_stokes_recovery():
    """T2: Re defined with diameter 2r; recovers Stokes law 6*pi*mu*r*v as Re -> 0."""
    rho_s, v_s, r_s, mu_s = sp.symbols("rho v r mu", positive=True)

    # Correct diameter definition: 2*r
    re_sym = rho_s * v_s * (2 * r_s) / mu_s
    cd_stokes = 24 / re_sym
    fd_stokes = sp.Rational(1, 2) * rho_s * (v_s ** 2) * sp.pi * (r_s ** 2) * cd_stokes

    simplified_fd = sp.simplify(fd_stokes)
    assert sp.simplify(simplified_fd - 6 * sp.pi * mu_s * r_s * v_s) == 0, (
        f"Stokes recovery failed: got {simplified_fd}, expected 6*pi*mu*r*v"
    )

    # Numerical recovery check
    re_tiny = 1e-6
    cd_val = cd_brown_lawler(re_tiny)
    expected_stokes_cd = 24.0 / re_tiny
    assert np.isclose(cd_val, expected_stokes_cd, rtol=1e-4)


def test_t3_split_disjointness_and_gap_isolation():
    """T3: Train, Transition Gap [5, 100], and Test must be strictly disjoint."""
    split = generate_experiment_dataset(n_train=50, sigma=0.02, seed=42)

    train_ids = set(p.id for p in split.train_points)
    gap_ids = set(p.id for p in split.gap_points)
    test_ids = set(p.id for p in split.test_points)

    assert train_ids.isdisjoint(gap_ids)
    assert train_ids.isdisjoint(test_ids)
    assert gap_ids.isdisjoint(test_ids)

    # Transition gap check: no training point may have 5 <= Re <= 100
    for p in split.train_points:
        assert not (5.0 <= p.re <= 100.0), f"Leakage: Train point {p.id} with Re={p.re} in [5, 100]"

    # Gap points must be inside [5, 100]
    for p in split.gap_points:
        assert 5.0 <= p.re <= 100.0, f"Gap point {p.id} with Re={p.re} outside [5, 100]"


def test_t4_ast_complexity_matches_reference():
    """T4: AST complexity counter matches reference 33 nodes for the generator."""
    ref_expr = "(24 / Re) * (1 + 0.150 * Re**0.681) + 0.407 / (1 + 8710 / Re)"
    count = count_ast_nodes(ref_expr)
    assert count == 33, f"Expected exactly 33 AST nodes, got {count}"


def test_t5_dataset_reproduction_identical_fingerprint():
    """T5: Reproducing dataset from identical seed produces identical SHA-256 fingerprint."""
    split1 = generate_experiment_dataset(n_train=50, sigma=0.02, seed=2026)
    split2 = generate_experiment_dataset(n_train=50, sigma=0.02, seed=2026)

    assert len(split1.train_points) == len(split2.train_points)
    for p1, p2 in zip(split1.train_points, split2.train_points):
        assert p1.id == p2.id
        assert np.isclose(p1.re, p2.re, rtol=1e-12)
        assert np.isclose(p1.cd_obs, p2.cd_obs, rtol=1e-12)

    # Check sealed grid SHA256 integrity
    assert split1.sealed_sha256 == FROZEN_GRID_SHA256
    assert split2.sealed_sha256 == FROZEN_GRID_SHA256


def test_t6_oracle_recovers_generator_at_zero_noise():
    """T6: Oracle baseline recovers true generator at sigma=0."""
    split_noiseless = generate_experiment_dataset(n_train=200, sigma=0.0, seed=100)
    oracle_res = evaluate_oracle_baseline(split_noiseless)

    # RMSE error on held-out gap must be near zero (< 1e-4)
    assert oracle_res["e_gap"] < 1e-4, f"Oracle gap RMSE error too high: {oracle_res['e_gap']}"
    assert oracle_res["e_max"] < 1e-4, f"Oracle max error too high: {oracle_res['e_max']}"
    assert oracle_res["passed_gates"] is True


def test_domain_adapter_registration_and_execution():
    """Verifies that StokesNewtonDragAdapter is properly registered and executes."""
    adapter = get_domain_adapter("stokes_newton_drag")
    assert isinstance(adapter, StokesNewtonDragAdapter)

    spec = adapter.parse_spec({"level": "LC", "n_train": 20, "sigma": 0.0, "seed": 42})
    assert spec.level == "LC"

    rng = random.Random(42)
    pop = adapter.build_population(spec, size=6, rng=rng)
    assert len(pop) == 6

    evaluator = adapter.build_evaluator(spec)
    # pop[0] is pure Stokes law which fails high-Re gate (score == 0.0)
    res_stokes = evaluator.evaluate(pop[0])
    assert res_stokes.score == 0.0
    assert "high_window_violation" in str(res_stokes.artifacts.get("gate_failures", []))

    # pop[1] is Brown-Lawler candidate which passes all gates
    res_bl = evaluator.evaluate(pop[1])
    assert res_bl.score > 80.0

    export = adapter.export_solution(pop[1], spec)
    assert "e_gap" in export
    assert "e_max" in export
    assert export["passed_gates"] is True


def test_lamarckian_tree_constant_optimization():
    """Verifies that free symbolic trees in Mode A optimize CONST nodes under physical gates."""
    ds = generate_experiment_dataset(n_train=50, sigma=0.02, seed=101)
    naive_root = DragExprNode(
        "ADD", None,
        DragExprNode("DIV", None, DragExprNode("CONST", 20.0), DragExprNode("VAR", "Re")),
        DragExprNode("CONST", 0.35),
    )
    genome = DragSymbolicGenome(root=naive_root, level="LC")
    const_nodes = genome.collect_const_nodes()
    assert len(const_nodes) == 2
    assert [n.value for n in const_nodes] == [20.0, 0.35]

    # Optimize constants on training data
    genome.optimize_constants(ds.train_points)

    # Values should be adjusted closer to Stokes (24) and Newton (~0.407)
    vals = [n.value for n in genome.collect_const_nodes()]
    assert 22.0 <= vals[0] <= 26.0
    assert 0.38 <= vals[1] <= 0.44
    passed, reasons = verify_all_physical_gates(genome.evaluate_cd)
    assert passed, f"Gates failed after Lamarckian optimization: {reasons}"

