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

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

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
# Pure Python Standard-Library Mathematical Implementations (Zero-Dependency)
# ============================================================================

def py_sphere(x: Sequence[float]) -> float:
    """Sphere function: f(x) = sum(x_i^2). Minimum at 0.0."""
    return sum(xi ** 2 for xi in x)


def py_rastrigin(x: Sequence[float]) -> float:
    """Rastrigin function: f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i)). Minimum at 0.0."""
    d = len(x)
    return 10.0 * d + sum(xi ** 2 - 10.0 * math.cos(2.0 * math.pi * xi) for xi in x)


def py_rosenbrock(x: Sequence[float]) -> float:
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2). Minimum at 0.0."""
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1))


def py_ackley(x: Sequence[float]) -> float:
    """Ackley function. Highly multimodal with global minimum at 0.0."""
    d = max(1, len(x))
    sum_sq = sum(xi ** 2 for xi in x)
    sum_cos = sum(math.cos(2.0 * math.pi * xi) for xi in x)
    term1 = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / d))
    term2 = -math.exp(sum_cos / d)
    return term1 + term2 + 20.0 + math.e


def py_griewank(x: Sequence[float]) -> float:
    """Griewank function. Minimum at 0.0."""
    sum_sq = sum(xi ** 2 for xi in x) / 4000.0
    prod_cos = 1.0
    for i, xi in enumerate(x, 1):
        prod_cos *= math.cos(xi / math.sqrt(i))
    return 1.0 + sum_sq - prod_cos


PY_LANDSCAPES: dict[str, Callable[[Sequence[float]], float]] = {
    "sphere": py_sphere,
    "rastrigin": py_rastrigin,
    "rosenbrock": py_rosenbrock,
    "ackley": py_ackley,
    "griewank": py_griewank,
}


# ============================================================================
# NumPy Vectorized Implementations (Shape: (N, D) -> (N,))
# ============================================================================

if HAS_NUMPY:
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

    LANDSCAPES: dict[str, Callable[[Any], Any]] = {
        "sphere": np_sphere,
        "rastrigin": np_rastrigin,
        "rosenbrock": np_rosenbrock,
        "ackley": np_ackley,
        "griewank": np_griewank,
    }
else:
    LANDSCAPES: dict[str, Callable[[Any], Any]] = {  # type: ignore[no-redef]
        k: v for k, v in PY_LANDSCAPES.items()
    }



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

    def _to_matrix(self, population: Sequence[Any]) -> tuple[Any, list[Any]]:
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
                rows.append([float(v) for v in g.values])
            elif hasattr(g, "genes"):
                rows.append([float(v) for v in g.genes])
            elif isinstance(g, (list, tuple)) or (HAS_NUMPY and isinstance(g, np.ndarray)):
                rows.append([float(v) for v in g])
            else:
                raise TypeError(f"Cannot extract vector coordinates from {type(g)}")

        if not rows:
            if HAS_NUMPY:
                return np.empty((0, 0), dtype=float), clean_pop
            return [], clean_pop

        d = len(rows[0])
        for r in rows:
            if len(r) != d:
                raise ValueError(f"Inconsistent vector dimensions in batch: expected {d}, got {len(r)}")

        if HAS_NUMPY:
            return np.asarray(rows, dtype=float), clean_pop
        return rows, clean_pop

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

        backend = "pure_python"
        dim = int(X.shape[1]) if hasattr(X, "shape") else len(X[0])

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
            losses = [float(v) for v in np.asarray(vmap_fn(j_arr))]
        elif HAS_NUMPY:
            backend = "numpy"
            losses = [float(v) for v in self.numpy_fn(X)]
        else:
            backend = "pure_python"
            py_fn = PY_LANDSCAPES[self.landscape]
            losses = [float(py_fn(row)) for row in X]

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
                    "dimension": dim,
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

