"""
z_protocol.py — Autonomous frontier negotiation and robust invariance engine.

Implements strategic optimization enhancements:
1. Frontier Pioneer Controller (Active Barrier Tunneling & Paradox Navigation)
2. Causal Momentum Vector Tracking (Directional Gradient Inheritance)
3. Internal Robustness Verification (Anti-Fragility & Parasite Resistance)
4. Adaptive Summit Precision Annealing (Zero-Overshoot Exploitation)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .engine import EvolutionEngine
from .genome import FloatGenome, Individual


@dataclass
class ZProtocolConfig:
    """Configuration parameters for Z-Protocol operational mode."""

    enabled: bool = True
    # Frontier Pioneer dynamics
    frontier_pioneer_enabled: bool = True
    tunnel_leap_magnitude: float = 1.2
    stagnation_tunnel_threshold: int = 4

    # Directional Causal Momentum
    causal_momentum_enabled: bool = True
    momentum_beta: float = 0.7
    momentum_weight: float = 0.35

    # Internal Robustness Verification (Anti-Fragility)
    internal_robustness_verification: bool = True
    perturbation_epsilon: float = 0.05

    # Summit Annealing
    adaptive_annealing_enabled: bool = True
    annealing_threshold: float = 65.0


class CausalMomentumTracker:
    """Maintains directional velocity vectors of constructive mutations."""

    def __init__(self, dimension: int = 16, beta: float = 0.7):
        self.dimension = dimension
        self.beta = beta
        self.velocity: list[float] = [0.0] * dimension
        self.total_constructive_events: int = 0

    def update(self, parent_genome: list[float], child_genome: list[float], delta_fitness: float) -> None:
        if delta_fitness <= 0.0 or len(parent_genome) != len(child_genome):
            return
        n = min(len(parent_genome), len(self.velocity))
        self.total_constructive_events += 1
        for i in range(n):
            diff = child_genome[i] - parent_genome[i]
            self.velocity[i] = self.beta * self.velocity[i] + (1.0 - self.beta) * diff

    def get_bias(self, length: int) -> list[float]:
        if len(self.velocity) < length:
            self.velocity.extend([0.0] * (length - len(self.velocity)))
        return self.velocity[:length]

    def magnitude(self) -> float:
        return math.sqrt(sum(v * v for v in self.velocity))


class ZProtocolEngine(EvolutionEngine):
    """Evolutionary Engine with active Z-Protocol architectural enhancements."""

    def __init__(self, *args: Any, z_config: ZProtocolConfig | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.z_config = z_config or ZProtocolConfig()
        self.momentum_tracker = CausalMomentumTracker(dimension=self.genome_size, beta=self.z_config.momentum_beta)
        self.consecutive_stagnant_gens: int = 0
        self._last_peak_fitness: float = -float("inf")
        self.tunnel_leaps_executed: int = 0
        self.robustness_evaluations: int = 0
        self.z_telemetry: list[dict[str, Any]] = []

    def mutate(self, ind: Individual, parent_fitness: float | None = None) -> tuple[Individual, str, float]:
        """Applies mutation governed by Z-Protocol causal momentum and pioneer dynamics."""
        if not self.z_config.enabled:
            return super().mutate(ind, parent_fitness=parent_fitness)

        parent_f = parent_fitness if parent_fitness is not None else getattr(ind, "fitness", 50.0)
        pre_genes = list(getattr(ind.genome, "genes", getattr(ind.genome, "values", [])))

        # 1. Base mutation with adaptive annealing
        ind, kind, l1 = super().mutate(ind, parent_fitness=parent_f)

        # 2. Causal Momentum Vector Steering
        if self.z_config.causal_momentum_enabled and hasattr(ind.genome, "values") and self.momentum_tracker.total_constructive_events > 0:
            bias = self.momentum_tracker.get_bias(len(ind.genome.values))
            weight = self.z_config.momentum_weight
            low, high = getattr(self.config, "clip_bounds", (-5.0, 5.0))
            new_vals = []
            for val, b in zip(ind.genome.values, bias):
                nudged = val + weight * b
                new_vals.append(max(low, min(high, nudged)))
            ind.genome.values = new_vals

        # 3. Frontier Pioneer Tunneling (when trapped in local optima / deceptive walls)
        if self.z_config.frontier_pioneer_enabled and self.consecutive_stagnant_gens >= self.z_config.stagnation_tunnel_threshold:
            if isinstance(ind.genome, FloatGenome) and ind.genome.values:
                self.tunnel_leaps_executed += 1
                leap_mag = self.z_config.tunnel_leap_magnitude
                low, high = getattr(self.config, "clip_bounds", (-5.0, 5.0))
                vals = list(ind.genome.values)
                # Coordinated frontier breakthrough: permutation symmetry or barrier leap
                if len(vals) >= 2 and self.rng.random() < 0.45:
                    i1, i2 = self.rng.sample(range(len(vals)), 2)
                    vals[i1], vals[i2] = vals[i2], vals[i1]
                    kind = "frontier_permutation_tunnel"
                else:
                    direction = self.rng.choice([-1.0, 1.0])
                    vals = [
                        max(low, min(high, val + direction * (leap_mag * self.rng.uniform(1.2, 2.5))))
                        for val in vals
                    ]
                    kind = "frontier_tunnel"
                ind.genome.values = vals

        post_genes = list(getattr(ind.genome, "genes", getattr(ind.genome, "values", [])))
        l1_final = sum(abs(a - b) for a, b in zip(pre_genes, post_genes)) if len(pre_genes) == len(post_genes) else l1
        return ind, kind, round(l1_final, 6)

    def evaluate(self, pop: list[Individual], generation: int | None = None, sharing: bool | None = None) -> None:
        """Evaluates population with internal anti-fragility and robustness verification."""
        super().evaluate(pop, generation=generation or 0, sharing=sharing)

        if not self.z_config.enabled:
            return

        # Track generational stagnation to control pioneer triggering
        current_best = max((ind.fitness for ind in pop), default=-float("inf"))
        if current_best > self._last_peak_fitness + 0.01:
            self._last_peak_fitness = current_best
            self.consecutive_stagnant_gens = 0
        else:
            self.consecutive_stagnant_gens += 1

        # Internal Robustness Verification (Anti-Fragility: Directional Stability)
        if self.z_config.internal_robustness_verification and hasattr(self, "fitness_fn"):
            eps = self.z_config.perturbation_epsilon

            for ind in pop:
                if isinstance(ind.genome, FloatGenome) and ind.genome.values:
                    self.robustness_evaluations += 1
                    orig_vals = list(ind.genome.values)
                    pos_genome = FloatGenome(values=[v + eps for v in orig_vals])
                    neg_genome = FloatGenome(values=[v - eps for v in orig_vals])
                    temp_pos = Individual(genome=pos_genome, species=ind.species)
                    temp_neg = Individual(genome=neg_genome, species=ind.species)
                    try:
                        fit_pos = float(self.fitness_fn(temp_pos))
                        fit_neg = float(self.fitness_fn(temp_neg))
                        if fit_pos > ind.fitness:
                            ind.genome = pos_genome
                            ind.fitness = fit_pos
                            if self.z_config.causal_momentum_enabled:
                                self.momentum_tracker.update(orig_vals, pos_genome.values, fit_pos - ind.fitness)
                        elif fit_neg > ind.fitness:
                            ind.genome = neg_genome
                            ind.fitness = fit_neg
                            if self.z_config.causal_momentum_enabled:
                                self.momentum_tracker.update(orig_vals, neg_genome.values, fit_neg - ind.fitness)
                        else:
                            worst_fit = min(fit_pos, fit_neg)
                            ind.fitness = min(ind.fitness, worst_fit)
                    except Exception:
                        pass

        # Update Causal Momentum from pending events
        if self.z_config.causal_momentum_enabled and hasattr(self, "_pending_causal"):
            for event in self._pending_causal:
                child = event.get("individual")
                pf_mean = event.get("parent_fitness_mean", 0.0)
                if child and hasattr(child, "genome") and hasattr(child.genome, "values"):
                    delta = child.fitness - pf_mean
                    if delta > 0:
                        dummy_parent = [v - 0.05 for v in child.genome.values]
                        self.momentum_tracker.update(dummy_parent, child.genome.values, delta)

        # Log Z-Protocol telemetry snapshot
        if generation is not None:
            self.z_telemetry.append({
                "generation": generation,
                "consecutive_stagnant_gens": self.consecutive_stagnant_gens,
                "tunnel_leaps_executed": self.tunnel_leaps_executed,
                "momentum_magnitude": round(self.momentum_tracker.magnitude(), 4),
                "constructive_events": self.momentum_tracker.total_constructive_events,
                "best_fitness": round(current_best, 4),
            })
