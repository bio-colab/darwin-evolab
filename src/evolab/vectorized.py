"""
vectorized.py — High-performance Vectorized & JAX-ready Continuous Landscape Evaluators.

Supports batch parallel evaluation of continuous numerical genomes (FloatGenome)
across standard non-convex optimization benchmark landscapes (Rastrigin, Rosenbrock,
Ackley, Sphere, Griewank).

Automatically leverages JAX (@jax.jit, jax.vmap) when JAX is installed, with a seamless,
zero-dependency, high-speed NumPy vectorized fallback that guarantees identical mathematical output.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Sequence
import numpy as np

from .evaluators import Evaluator, FitnessResult
from .genome import EvolabGenome, FloatGenome, Individual

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


# ============================================================================
# Pure NumPy Vectorized Implementations (Shape: (N, D) -> (N,))
# ============================================================================

def np_sphere(X: np.ndarray) -> np.ndarray:
    """Sphere function: f(x) = sum(x_i^2). Minimum at 0.0."""
    return np.sum(X ** 2, axis=-1)


def np_rastrigin(X: np.ndarray) -> np.ndarray:
    """Rastrigin function: f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i)). Minimum at 0.0."""
    d = X.shape[-1]
    return 10.0 * d + np.sum(X ** 2 - 10.0 * np.cos(2.0 * np.pi * X), axis=-1)


def np_rosenbrock(X: np.ndarray) -> np.ndarray:
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2). Minimum at 0.0."""
    return np.sum(100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (1.0 - X[..., :-1]) ** 2, axis=-1)


def np_ackley(X: np.ndarray) -> np.ndarray:
    """Ackley function. Highly multimodal with global minimum at 0.0."""
    d = X.shape[-1]
    sum_sq = np.sum(X ** 2, axis=-1)
    sum_cos = np.sum(np.cos(2.0 * np.pi * X), axis=-1)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / max(1, d)))
    term2 = -np.exp(sum_cos / max(1, d))
    return term1 + term2 + 20.0 + np.e


def np_griewank(X: np.ndarray) -> np.ndarray:
    """Griewank function. Minimum at 0.0."""
    d = X.shape[-1]
    sum_sq = np.sum(X ** 2, axis=-1) / 4000.0
    i_factors = np.sqrt(np.arange(1, d + 1, dtype=float))
    prod_cos = np.prod(np.cos(X / i_factors), axis=-1)
    return 1.0 + sum_sq - prod_cos


# ============================================================================
# JAX Vectorized Functions (JIT compiled if available)
# ============================================================================

if HAS_JAX:
    @jax.jit
    def jax_sphere(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x ** 2)

    @jax.jit
    def jax_rastrigin(x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        return 10.0 * d + jnp.sum(x ** 2 - 10.0 * jnp.cos(2.0 * jnp.pi * x))

    @jax.jit
    def jax_rosenbrock(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    @jax.jit
    def jax_ackley(x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        sum_sq = jnp.sum(x ** 2)
        sum_cos = jnp.sum(jnp.cos(2.0 * jnp.pi * x))
        term1 = -20.0 * jnp.exp(-0.2 * jnp.sqrt(sum_sq / jnp.maximum(1, d)))
        term2 = -jnp.exp(sum_cos / jnp.maximum(1, d))
        return term1 + term2 + 20.0 + jnp.e

    @jax.jit
    def jax_griewank(x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        sum_sq = jnp.sum(x ** 2) / 4000.0
        i_factors = jnp.sqrt(jnp.arange(1, d + 1, dtype=float))
        prod_cos = jnp.prod(jnp.cos(x / i_factors))
        return 1.0 + sum_sq - prod_cos

    # vmapped batched versions
    jax_vmap_sphere = jax.vmap(jax_sphere)
    jax_vmap_rastrigin = jax.vmap(jax_rastrigin)
    jax_vmap_rosenbrock = jax.vmap(jax_rosenbrock)
    jax_vmap_ackley = jax.vmap(jax_ackley)
    jax_vmap_griewank = jax.vmap(jax_griewank)


LANDSCAPES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sphere": np_sphere,
    "rastrigin": np_rastrigin,
    "rosenbrock": np_rosenbrock,
    "ackley": np_ackley,
    "griewank": np_griewank,
}


class VectorizedLandscapeEvaluator(Evaluator):
    """
    High-performance evaluator for mathematical optimization landscapes.
    Evaluates individual genomes or entire batches (populations of 10,000+ candidates).
    """

    def __init__(
        self,
        landscape: str = "rastrigin",
        target_score: float = 100.0,
        use_jax: bool = False,
    ) -> None:
        name = landscape.lower()
        if name not in LANDSCAPES:
            raise ValueError(f"Unknown landscape '{landscape}'. Choose from: {list(LANDSCAPES.keys())}")
        self.landscape = name
        self.target_score = target_score
        self.use_jax = bool(use_jax and HAS_JAX)
        self.numpy_fn = LANDSCAPES[name]

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def _to_matrix(self, population: Sequence[Any]) -> tuple[np.ndarray, list[Any]]:
        """Extract float matrix of shape (N, D) from sequence of individuals or genomes."""
        rows: list[list[float]] = []
        clean_pop: list[Any] = []
        for item in population:
            clean_pop.append(item)
            if isinstance(item, Individual):
                g = item.genome
            else:
                g = item

            if hasattr(g, "values"):
                rows.append(list(g.values))
            elif hasattr(g, "genes"):
                rows.append(list(g.genes))
            elif isinstance(g, (list, tuple, np.ndarray)):
                rows.append(list(g))
            else:
                raise TypeError(f"Cannot extract vector coordinates from {type(g)}")

        if not rows:
            return np.empty((0, 0), dtype=float), clean_pop

        d = len(rows[0])
        for r in rows:
            if len(r) != d:
                raise ValueError(f"Inconsistent vector dimensions in batch: expected {d}, got {len(r)}")

        return np.asarray(rows, dtype=float), clean_pop

    def evaluate_batch(
        self,
        population: Sequence[Individual | EvolabGenome | Sequence[float]],
    ) -> list[FitnessResult]:
        """Evaluate an entire batch/population of solutions simultaneously."""
        t0 = time.perf_counter()
        X, raw_items = self._to_matrix(population)
        n = len(X)
        if n == 0:
            return []

        backend = "numpy"
        if self.use_jax and HAS_JAX:
            backend = "jax"
            j_arr = jnp.asarray(X)
            vmap_fn = {
                "sphere": jax_vmap_sphere,
                "rastrigin": jax_vmap_rastrigin,
                "rosenbrock": jax_vmap_rosenbrock,
                "ackley": jax_vmap_ackley,
                "griewank": jax_vmap_griewank,
            }[self.landscape]
            losses = np.asarray(vmap_fn(j_arr))
        else:
            losses = self.numpy_fn(X)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = duration_ms / max(1, n)

        results: list[FitnessResult] = []
        for i in range(n):
            loss_val = float(losses[i])
            # Bounded fitness: 100 / (1 + loss)
            score = round(self.target_score / (1.0 + max(0.0, loss_val)), 4)
            res = FitnessResult(
                score=score,
                sub_scores={"loss": loss_val, "raw_error": abs(loss_val)},
                artifacts={
                    "loss": loss_val,
                    "backend": backend,
                    "landscape": self.landscape,
                    "dimension": int(X.shape[1]),
                },
                evaluation_time_ms=per_item_ms,
            )
            # If target individual passed, update its fitness directly
            if isinstance(raw_items[i], Individual):
                raw_items[i].fitness = score
            results.append(res)

        return results

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> FitnessResult:
        """Single-target evaluation hook conforming to Evaluator interface."""
        res_list = self.evaluate_batch([target])
        return res_list[0]

