"""Tests for high-performance Vectorized & JAX-ready mathematical landscape evaluation."""
import math
import time
import numpy as np
import pytest

from evolab.vectorized import (
    VectorizedLandscapeEvaluator,
    np_sphere,
    np_rastrigin,
    np_rosenbrock,
    np_ackley,
    np_griewank,
    HAS_JAX,
)
from evolab.genome import FloatGenome, Individual
from evolab.adapters import NumericalMathAdapter, NumericalMathSpec


def test_numpy_landscape_functions_minimums():
    # Known global minimum points
    zeros_2d = np.zeros((1, 5))
    ones_2d = np.ones((1, 5))

    # Sphere min is 0 at origin
    assert np_sphere(zeros_2d)[0] == 0.0

    # Rastrigin min is 0 at origin
    assert abs(np_rastrigin(zeros_2d)[0]) < 1e-12

    # Rosenbrock min is 0 at all 1s
    assert abs(np_rosenbrock(ones_2d)[0]) < 1e-12

    # Ackley min is 0 at origin
    assert abs(np_ackley(zeros_2d)[0]) < 1e-12

    # Griewank min is 0 at origin
    assert abs(np_griewank(zeros_2d)[0]) < 1e-12


def test_vectorized_evaluator_single_and_batch():
    evaluator = VectorizedLandscapeEvaluator(landscape="rastrigin")

    candidates = [
        FloatGenome([0.0, 0.0, 0.0]),
        FloatGenome([1.0, 1.0, 1.0]),
        FloatGenome([2.0, -1.0, 0.5]),
    ]
    inds = [Individual(g, species="spec_math") for g in candidates]

    # Evaluate individually
    single_res = [evaluator.evaluate(ind) for ind in inds]

    # Evaluate in batch
    batch_res = evaluator.evaluate_batch(inds)

    assert len(single_res) == len(batch_res) == 3
    for s, b in zip(single_res, batch_res):
        assert math.isclose(s.score, b.score, rel_tol=1e-5)
        assert math.isclose(s.sub_scores["loss"], b.sub_scores["loss"], rel_tol=1e-5)

    # Origin should achieve maximum fitness 100.0
    assert batch_res[0].score == 100.0


def test_vectorized_evaluator_large_batch_performance():
    evaluator = VectorizedLandscapeEvaluator(landscape="sphere")

    n_candidates = 5000
    dim = 10
    rng = np.random.default_rng(42)
    matrix = rng.uniform(-5.0, 5.0, size=(n_candidates, dim))
    pop = [FloatGenome(row.tolist()) for row in matrix]

    t0 = time.perf_counter()
    results = evaluator.evaluate_batch(pop)
    duration = time.perf_counter() - t0

    assert len(results) == n_candidates
    # Vectorized evaluation of 5,000 10-dimensional candidates should complete in < 0.2s
    assert duration < 0.2, f"Expected < 0.2s, took {duration:.4f}s"
    assert results[0].score > 0.0


def test_float_genome_to_numpy_and_to_jax():
    g = FloatGenome([1.5, -2.25, 3.14159])
    arr = g.to_numpy()

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert math.isclose(arr[0], 1.5)
    assert math.isclose(arr[2], 3.14159)

    if HAS_JAX:
        import jax.numpy as jnp
        jarr = g.to_jax()
        assert isinstance(jarr, jnp.ndarray)
    else:
        with pytest.raises(ImportError) as exc_info:
            g.to_jax()
        assert "JAX is not installed" in str(exc_info.value)


def test_numerical_math_adapter_vectorized_integration():
    adapter = NumericalMathAdapter()
    spec = NumericalMathSpec(target_function="rosenbrock", dimensions=4)
    evaluator = adapter.build_vectorized_evaluator(spec)

    assert isinstance(evaluator, VectorizedLandscapeEvaluator)
    assert evaluator.landscape == "rosenbrock"

    rng = np.random.default_rng(123)
    pop = adapter.build_population(spec, size=16, rng=rng)
    assert len(pop) == 16

    results = evaluator.evaluate_batch(pop)
    assert len(results) == 16
    for ind, res in zip(pop, results):
        assert ind.fitness == res.score
        assert res.score > 0.0

