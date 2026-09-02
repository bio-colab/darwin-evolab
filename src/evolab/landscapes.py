"""Fitness landscape abstraction (audit A21 recommendation #1).

One interface, many terrains — lets calibration transfer be MEASURED
instead of assumed (charter sections 15-16, audit A16 S3, audit A19 BJ-5).

Every landscape exposes:
    name         stable identifier for reports/config provenance
    stationary   False when the evaluator changes over time
    evaluate(genome) -> float   finite score inside the declared range
    perturb(rng) -> None        advance non-stationary state (optional op)

Determinism contract: given the same construction seed and the same call
sequence, evaluate() reproduces exactly. Non-stationary wrappers change
state ONLY through perturb()/internal evaluation counters — never through
wall-clock or hidden globals.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Protocol, runtime_checkable

GENOME_RANGE = 10.0


@runtime_checkable
class FitnessLandscape(Protocol):
    name: str
    stationary: bool

    def evaluate(self, genome: list[float]) -> float: ...
    def perturb(self, rng: random.Random) -> None: ...


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


class SmoothProxyLandscape:
    """The historical calibration landscape (audits A14-A16 reference it).

    Near-unimodal closeness-to-target proxy. Explicitly labelled a
    mechanism-testing landscape, NOT evidence about rugged search.
    """

    name = "smooth_proxy"
    stationary = True

    def evaluate(self, genome: list[float]) -> float:
        err = sum(abs(g - 3.0) for g in genome) / (len(genome) * 8.0)
        return _clip((1.0 - err) * 100.0)

    def perturb(self, rng: random.Random) -> None:  # pragma: no cover
        return None


class RastriginLandscape:
    """Multimodal stress terrain (introduced in audit A14 as ragged_fitness)."""

    name = "rastrigin"
    stationary = True

    def __init__(self, ripple: float = 0.06) -> None:
        self.ripple = ripple

    def evaluate(self, genome: list[float]) -> float:
        n = len(genome)
        base = sum((g - 3.0) ** 2 / (n * 12.5) for g in genome)
        rp = (
            self.ripple
            * n
            * sum(math.cos(2.4 * (g - 3.0)) for g in genome)
            / n
        )
        return _clip(100.0 - base * 100.0 - abs(rp) * 10.0)

    def perturb(self, rng: random.Random) -> None:  # pragma: no cover
        return None


class TrapKLandscape:
    """Deceptive trap-k over threshold-binarised gene blocks.

    Classic deception (Goldberg): the all-zeros attractor outscores every
    partial solution; only the exact all-ones block reaches the global
    peak. Exposes whether diversity machinery escapes deceptive basins.
    """

    name = "trap_k"
    stationary = True

    def __init__(self, k: int = 4) -> None:
        if k < 2:
            raise ValueError("trap block size k must be >= 2")
        self.k = k

    def evaluate(self, genome: list[float]) -> float:
        k = self.k
        n_blocks = len(genome) // k
        if n_blocks == 0:
            return 0.0
        total = 0.0
        for b in range(n_blocks):
            block = genome[b * k : (b + 1) * k]
            ones = sum(1 for g in block if g >= 0.5)
            if ones == k:
                total += k
            else:
                total += max(0.0, k - 1 - ones)
        return _clip(total * (100.0 / (n_blocks * k)))

    def perturb(self, rng: random.Random) -> None:  # pragma: no cover
        return None


class NoisyWrapper:
    """Wraps any landscape with Gaussian evaluation noise.

    Deterministic given the wrapper's own rng and call order — noise is
    part of the experiment definition, recorded via the wrapper name.
    """

    name = "noisy"

    def __init__(
        self,
        inner,
        sigma: float = 2.0,
        seed: int | None = None,
    ) -> None:
        self.inner = inner
        self.sigma = sigma
        self.name = f"noisy[{inner.name}]"
        self.stationary = getattr(inner, "stationary", True)
        self._rng = random.Random(seed)

    def evaluate(self, genome: list[float]) -> float:
        base = self.inner.evaluate(genome)
        return _clip(base + self._rng.gauss(0.0, self.sigma))

    def perturb(self, rng: random.Random) -> None:
        inner_perturb = getattr(self.inner, "perturb", None)
        if callable(inner_perturb):
            inner_perturb(rng)


class MovingTargetLandscape:
    """Non-stationary: the optimum drifts every `shift_every` evaluations.

    Closes part of audit A20's reachability/shadow findings by making the
    drift rule explicit and reproducible instead of hidden inside a
    closure.
    """

    name = "moving_target"
    stationary = False

    def __init__(
        self,
        genome_size: int = 16,
        shift_every: int = 10,
        step_scale: float = 1.5,
        seed: int | None = None,
    ) -> None:
        self.genome_size = genome_size
        self.shift_every = shift_every
        self.step_scale = step_scale
        self._rng = random.Random(seed)
        self.target = [self._rng.uniform(-2.0, 2.0) for _ in range(genome_size)]
        self._calls = 0

    def evaluate(self, genome: list[float]) -> float:
        self._calls += 1
        if self.shift_every and self._calls % self.shift_every == 0:
            self.perturb(random.Random(self._calls))
        err = sum(abs(g - t) for g, t in zip(genome, self.target)) / (
            len(genome) * GENOME_RANGE
        )
        return _clip((1.0 - err) * 100.0)

    def perturb(self, rng: random.Random) -> None:
        self.target = [
            max(-5.0, min(5.0, t + rng.uniform(-1, 1) * self.step_scale))
            for t in self.target
        ]


def build_landscape(name: str, **kwargs):
    """Registry access by name (report/config provenance friendly)."""
    builders: dict[str, Callable] = {
        "smooth_proxy": SmoothProxyLandscape,
        "rastrigin": RastriginLandscape,
        "trap_k": TrapKLandscape,
        "moving_target": MovingTargetLandscape,
    }
    if name == "noisy":
        raise ValueError(
            "noisy is a wrapper — construct NoisyWrapper(inner=..., sigma=...)"
        )
    if name not in builders:
        raise ValueError(f"unknown landscape {name!r}; known: {sorted(builders)}")
    return builders[name](**kwargs)
