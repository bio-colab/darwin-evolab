"""
comparative_runner.py — Comparative Baseline Benchmarks for Electronic Synthesis.
Evaluates Random Search vs Standard GA vs Darwin-EvoLab Engine with Speciation.
Enhanced with RealCodedGA (SBX + polynomial mutation) — pure numpy, no extra deps.
Mirrors opensource-analog-circuits benchmark/example_ga.py (low-cost, non-breaking).
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import time
from pathlib import Path
from typing import Any

from evolab.config import EngineConfig, SpeciationConfig
from evolab.engine import EvolutionEngine
from evolab.genome import FloatGenome, Individual
from ..evaluators.spice_evaluator import AnalogSizingEvaluator, CircuitConfigEvaluator

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    np = None  # type: ignore


@dataclass(frozen=True)
class BenchmarkRunSummary:
    strategy_name: str
    best_fitness: float
    generations_run: int
    evaluations_count: int
    duration_sec: float
    target_met: bool


class ComparativeSizingBenchmark:
    """Compares optimization algorithms on analog transistor sizing tasks."""

    def __init__(
        self,
        budget_evals: int = 60,
        seed: int = 42,
        evaluator: Any | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.budget_evals = budget_evals
        self.seed = seed
        if evaluator is not None:
            self.evaluator: Any = evaluator
        elif config_path is not None:
            self.evaluator = CircuitConfigEvaluator(config_path)
        else:
            self.evaluator = AnalogSizingEvaluator(
                target_gain_db=40.0,
                target_ugbw_mhz=10.0,
                min_phase_margin_deg=45.0,
                max_power_mw=5.0,
            )

    def _get_dim_and_bounds(self) -> tuple[int, list[float], list[float]]:
        """Return (dim, lb, ub) for current evaluator — supports both toy and config-driven."""
        ev: Any = self.evaluator
        if hasattr(ev, "dim"):
            return int(ev.dim), list(ev.lb), list(ev.ub)  # type: ignore
        if hasattr(ev, "design_vars"):
            names = list(ev.design_vars.keys())  # type: ignore
            lb = [float(ev.design_vars[n][1]) for n in names]  # type: ignore
            ub = [float(ev.design_vars[n][2]) for n in names]  # type: ignore
            return len(names), lb, ub
        return 4, [0.5] * 4, [10.0] * 4

    def run_random_search(self) -> BenchmarkRunSummary:
        """Baseline 1: Pure Monte Carlo Random Search."""
        t0 = time.perf_counter()
        rng = random.Random(self.seed)
        best_fit = -1.0
        dim, lb, ub = self._get_dim_and_bounds()

        for _ in range(self.budget_evals):
            genes = [rng.uniform(lb[i], ub[i]) for i in range(dim)]
            genome = FloatGenome(genes)
            res = self.evaluator.evaluate(genome)
            if res.score > best_fit:
                best_fit = res.score

        dur = time.perf_counter() - t0
        return BenchmarkRunSummary(
            strategy_name="Random Search",
            best_fitness=round(best_fit, 4),
            generations_run=1,
            evaluations_count=self.budget_evals,
            duration_sec=round(dur, 4),
            target_met=(best_fit >= 0.85),
        )

    def run_evolab_engine(self, enable_speciation: bool = True) -> BenchmarkRunSummary:
        """Baseline 2/3: Darwin-EvoLab EvolutionEngine (with or without speciation)."""
        t0 = time.perf_counter()
        pop_size = 10
        generations = max(2, self.budget_evals // pop_size)
        dim, lb, ub = self._get_dim_and_bounds()

        cfg = EngineConfig(
            population_size=pop_size,
            generations=generations,
            genome_size=dim,
            crossover_rate=0.3,
            mutation_rate=0.4,
            seed=self.seed,  # deterministic engine rolls (selection/crossover/mutation) — seed=None drew from system entropy
            speciation=SpeciationConfig(enabled=enable_speciation),
        )

        def eval_fn(ind: Individual) -> float:
            res = self.evaluator.evaluate(ind)
            return res.score

        initial_pop = [
            Individual(
                FloatGenome([random.Random(self.seed + i).uniform(lb[d], ub[d]) for d in range(dim)]),
                species="spec_analog",
            )
            for i in range(pop_size)
        ]

        engine = EvolutionEngine(fitness_fn=eval_fn, config=cfg)
        report = engine.run(generations, initial_population=initial_pop)
        dur = time.perf_counter() - t0

        best_ind = report.get("best_individual", {})
        best_fit = float(best_ind.get("fitness", 0.0))
        total_gens = int(report.get("total_generations", generations))

        name = "Darwin-EvoLab (Speciation)" if enable_speciation else "Standard GA"
        return BenchmarkRunSummary(
            strategy_name=name,
            best_fitness=round(best_fit, 4),
            generations_run=total_gens,
            evaluations_count=pop_size * total_gens,
            duration_sec=round(dur, 4),
            target_met=(best_fit >= 85.0 or best_fit >= 0.85),
        )

    def run_all(self) -> list[BenchmarkRunSummary]:
        """Runs all three baselines and returns comparative results (unchanged for test compat)."""
        return [
            self.run_random_search(),
            self.run_evolab_engine(enable_speciation=False),
            self.run_evolab_engine(enable_speciation=True),
        ]

    # ---- NEW: low-cost SBX GA (non-breaking, optional) ----

    def run_sbx_ga(
        self,
        pop_size: int = 20,
        generations: int = 6,
        cx_rate: float = 0.9,
        mut_rate: float = 0.2,
        eta_c: float = 15.0,
        eta_m: float = 20.0,
        seed: int | None = None,
    ) -> BenchmarkRunSummary:
        """Real-coded GA with SBX + polynomial mutation (pure numpy or fallback).

        Mirrors benchmark/example_ga.py (opensource-analog-circuits). Cheap, deterministic.
        Not used in run_all() to preserve existing test contract.
        Now supports variable dim from config-driven evaluator.
        """
        t0 = time.perf_counter()
        rng_seed = seed if seed is not None else self.seed
        budget = pop_size * generations
        dim, lb, ub = self._get_dim_and_bounds()

        def evaluate_vec(vec: list[float]) -> float:
            genome = FloatGenome(vec)
            return float(self.evaluator.evaluate(genome).score)

        # numpy path (preferred)
        if HAS_NUMPY:
            rng = np.random.default_rng(rng_seed)  # type: ignore
            pop = rng.random((pop_size, dim)) * (np.array(ub) - np.array(lb)) + np.array(lb)  # type: ignore
            # ensure first individual is decent: use evaluator defaults if available
            ev_any: Any = self.evaluator
            if hasattr(ev_any, "defaults"):
                try:
                    defaults = list(ev_any.defaults)  # type: ignore
                    if len(defaults) == dim:
                        pop[0] = np.array(defaults, dtype=float)  # type: ignore
                except Exception:
                    pass
            elif dim == 4:
                pop[0] = np.array([2.5, 0.35, 1.2, 0.35]) + rng.normal(0, 0.1, dim)  # type: ignore
            objs = np.array([evaluate_vec(list(pop[i])) for i in range(pop_size)])  # type: ignore
            best = float(np.max(objs))

            for _ in range(generations - 1):
                # tournament select + SBX + poly mutation
                new_pop = []
                # elitism 1
                best_idx = int(np.argmax(objs))
                new_pop.append(pop[best_idx].copy())  # type: ignore
                while len(new_pop) < pop_size:
                    # tournament k=3
                    def tournament() -> Any:
                        cands = rng.choice(pop_size, size=3, replace=False)  # type: ignore
                        winner = cands[int(np.argmax(objs[cands]))]
                        return pop[winner].copy()  # type: ignore

                    p1, p2 = tournament(), tournament()
                    # SBX
                    c1, c2 = p1.copy(), p2.copy()  # type: ignore
                    if rng.random() < cx_rate:  # type: ignore
                        for i in range(dim):
                            if rng.random() > 0.5 or abs(float(p1[i]) - float(p2[i])) < 1e-14:  # type: ignore
                                continue
                            y1, y2 = (float(p1[i]), float(p2[i])) if float(p1[i]) < float(p2[i]) else (float(p2[i]), float(p1[i]))  # type: ignore
                            beta = 1.0 + (2.0 * min(y1 - lb[i], ub[i] - y2) / (y2 - y1))
                            alpha = 2.0 - beta ** (-(eta_c + 1.0))
                            rand = float(rng.random())  # type: ignore
                            beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                            c1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))  # type: ignore
                            c2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))  # type: ignore
                            c1[i] = float(np.clip(c1[i], lb[i], ub[i]))  # type: ignore
                            c2[i] = float(np.clip(c2[i], lb[i], ub[i]))  # type: ignore
                    # poly mutation
                    for c in (c1, c2):
                        if rng.random() < mut_rate:  # type: ignore
                            for i in range(dim):
                                if rng.random() > 1.0 / dim:  # type: ignore
                                    continue
                                delta1 = (float(c[i]) - lb[i]) / (ub[i] - lb[i])  # type: ignore
                                delta2 = (ub[i] - float(c[i])) / (ub[i] - lb[i])  # type: ignore
                                rand = float(rng.random())  # type: ignore
                                mut_pow = 1.0 / (eta_m + 1.0)
                                if rand <= 0.5:
                                    xy = 1.0 - delta1
                                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                                    delta_q = val ** mut_pow - 1.0
                                else:
                                    xy = 1.0 - delta2
                                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                                    delta_q = 1.0 - val ** mut_pow
                                c[i] = float(np.clip(float(c[i]) + delta_q * (ub[i] - lb[i]), lb[i], ub[i]))  # type: ignore
                        new_pop.append(c.copy())  # type: ignore
                        if len(new_pop) >= pop_size:
                            break
                pop = np.array(new_pop[:pop_size])  # type: ignore
                objs = np.array([evaluate_vec(list(pop[i])) for i in range(pop_size)])  # type: ignore
                best = max(best, float(np.max(objs)))
            dur = time.perf_counter() - t0
            return BenchmarkRunSummary(
                strategy_name="RealCodedGA_SBX",
                best_fitness=round(best, 4),
                generations_run=generations,
                evaluations_count=budget,
                duration_sec=round(dur, 4),
                target_met=(best >= 85.0),
            )

        # pure-python fallback (no numpy)
        py_rng = random.Random(rng_seed)
        pop_py = [[py_rng.uniform(lb[i], ub[i]) for i in range(dim)] for _ in range(pop_size)]
        best_py = max(evaluate_vec(ind) for ind in pop_py)
        for _ in range(generations - 1):
            pop_py = sorted(pop_py, key=evaluate_vec, reverse=True)
            new_py = [pop_py[0]]
            while len(new_py) < pop_size:
                # simple uniform crossover + gaussian mutation fallback
                p1 = py_rng.choice(pop_py)
                p2 = py_rng.choice(pop_py)
                c = [(a + b) / 2.0 for a, b in zip(p1, p2)]
                if py_rng.random() < mut_rate:
                    c = [max(lb[i], min(ub[i], v + py_rng.gauss(0, 0.3))) for i, v in enumerate(c)]
                new_py.append(c)
            pop_py = new_py[:pop_size]
            best_py = max(best_py, max(evaluate_vec(ind) for ind in pop_py))
        dur = time.perf_counter() - t0
        return BenchmarkRunSummary(
            strategy_name="RealCodedGA_SBX",
            best_fitness=round(best_py, 4),
            generations_run=generations,
            evaluations_count=budget,
            duration_sec=round(dur, 4),
            target_met=(best_py >= 85.0),
        )


class RealCodedGA:
    """Standalone SBX GA (mirrors benchmark/example_ga.py) — usable directly."""

    def __init__(self, evaluator: Any = None, seed: int = 42) -> None:
        self.evaluator = evaluator or AnalogSizingEvaluator()
        self.seed = seed

    def optimize(self, budget_evals: int = 60) -> BenchmarkRunSummary:
        bench = ComparativeSizingBenchmark(budget_evals=budget_evals, seed=self.seed)
        bench.evaluator = self.evaluator
        return bench.run_sbx_ga(pop_size=10, generations=max(2, budget_evals // 10), seed=self.seed)
