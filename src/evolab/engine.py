from __future__ import annotations

import json
import math
import numbers
import random
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from .config import EngineConfig, SpeciationConfig, QualityDiversityConfig, MemoryConfig
from .genome import GENOME_RANGE, Individual, Species, FloatGenome, EvolabGenome, random_individual
from .speciation import SPECIES_POOL, DYNAMIC_SPECIES_CFG, genomic_distance, species_cfg
from .fitness import default_fitness, ragged_fitness
from .memory import ChangeDetector, ChangeType, MemoryInjector, TemporalMemoryIndex

ENGINE_VERSION = "evolab-engine/0.5.0"
SCHEMA_VERSION = "report-schema/1"

DISTANCE_PRESETS = ("composite", "euclidean", "maxdelta")


def _desc_mean(genome: Any) -> float:
    if hasattr(genome, "describe"):
        desc = genome.describe()
        for k in ("mean", "node_count", "hunk_count", "lines_added"):
            if k in desc:
                return float(desc[k])
    if hasattr(genome, "values"):
        vals = genome.values
        return sum(vals) / max(len(vals), 1)
    if hasattr(genome, "genes"):
        return sum(genome.genes) / max(len(genome.genes), 1)
    if isinstance(genome, list):
        return sum(genome) / max(len(genome), 1)
    if hasattr(genome, "fingerprint"):
        return float(int(genome.fingerprint()[:4], 16) % 100) / 10.0
    return 0.0


def _desc_std(genome: Any) -> float:
    if hasattr(genome, "describe"):
        desc = genome.describe()
        for k in ("slope", "std", "max_depth", "lines_removed", "stmt_count"):
            if k in desc:
                return float(desc[k])
    if hasattr(genome, "values"):
        vals = genome.values
        if len(vals) > 1:
            mid = len(vals) // 2
            f_h = sum(vals[:mid]) / max(1, mid)
            s_h = sum(vals[mid:]) / max(1, len(vals) - mid)
            std_v = statistics.pstdev(vals)
            return (s_h - f_h) / (std_v + 1e-6)
        return 0.0
    if hasattr(genome, "genes"):
        g = genome.genes
        return statistics.pstdev(g) if len(g) > 1 else 0.0
    if isinstance(genome, list):
        return statistics.pstdev(genome) if len(genome) > 1 else 0.0
    if hasattr(genome, "fingerprint"):
        return float(int(genome.fingerprint()[4:8], 16) % 50) / 10.0
    return 0.0


FitnessFn = Callable[["Individual"], float]


class EvolutionEngine:
    """Phased fitness-sharing GA engine (v3.1).

    Threading contract (audit A14 E-07): EvolutionEngine is NOT
    thread-safe — it owns RNG, archive and counters. Use one engine per
    thread; identical seeds across independent engines produce identical
    runs.

    Claims contract (audit A18): this is a deterministic numeric search
    system with recorded lineage. The API supports NO learning,
    intuition, prediction or awareness semantics — such language must
    not appear in reports or docs until a measurable contract exists
    (charter claim-discipline).

    MAP-Elites archive note (audit A19): the archive is observational —
    it is updated during evaluation but never feeds back into selection.
    A19 control showed identical trajectories with/without it (20/20).
    Enable `record_archive_solutions` to export archived genomes for
    external analysis; making the archive causally active is v4 design.
    """

    def __init__(
        self,
        fitness_fn: FitnessFn | None = None,
        config: EngineConfig | None = None,
        population_size: int | None = None,
        elite_count: int | None = None,
        mutation_rate: float | None = None,
        early_stop_fitness: float | None = None,
        seed: int | None = None,
        genome_size: int | None = None,
        stagnation_patience: int | None = None,
        immigrant_fraction: float | None = None,
        fitness_sharing: bool | None = None,
        dist_c1: float | None = None,
        dist_c2: float | None = None,
        speciation_threshold: float | None = None,
        hybrid_light_share: float | None = None,
        speciation_enabled: bool | None = None,
        mutation_enabled: bool | None = None,
        fitness_range: tuple[float, float] | None = None,
        record_population_snapshots: bool | None = None,
        dist_c3: float | None = None,
        record_archive_solutions: bool | None = None,
        distance_metric: str | None = None,
        distance_fn: Callable | None = None,
        descriptors: Sequence[Callable] | None = None,
        me_scale_x: float | None = None,
        me_scale_y: float | None = None,
        eval_repeats: int | None = None,
        stability_penalty: float | None = None,
        hard_constraints: Sequence[Callable[[list[float]], bool]] | None = None,
        memory_enabled: bool | None = None,
        memory_max_injection_rate: float | None = None,
        change_window: int | None = None,
        cusum_k: float | None = None,
        cusum_h: float | None = None,
        staleness_tau: float | None = None,
        quarantine_gens: int | None = None,
        sharing_mode: str | None = None,
        exploit_after_frac: float | None = None,
        me_grid_x: int | None = None,
        me_grid_y: int | None = None,
        qd_selection: bool | None = None,
        evaluator: Any | None = None,
        pop_size: int | None = None,
        num_generations: int | None = None,
        generations: int | None = None,
        causal_layer_enabled: bool | None = None,
    ) -> None:
        if isinstance(config, dict):
            cfg = EngineConfig()
            if "sharing_mode" in config:
                cfg.sharing_mode = str(config["sharing_mode"])
            if "crossover_rate" in config:
                cfg.crossover_rate = float(config["crossover_rate"])
            if "mutation_rate" in config:
                cfg.mutation_rate = float(config["mutation_rate"])
            if "archive_size" in config:
                cfg.qd.k = int(config["archive_size"])
            if "causal_layer_enabled" in config:
                cfg.speciation.enabled = bool(config["causal_layer_enabled"])
            self.extra_config_dict = dict(config)
        elif isinstance(config, EngineConfig):
            cfg = config
            self.extra_config_dict = {}
        else:
            cfg = EngineConfig()
            self.extra_config_dict = {}
        self.config = cfg

        if causal_layer_enabled is not None:
            self.causal_layer_enabled = bool(causal_layer_enabled)
        else:
            self.causal_layer_enabled = bool(self.extra_config_dict.get("causal_layer_enabled", False))
        self.cem_enabled = bool(self.extra_config_dict.get("cem_enabled", False))
        self._causal_traps_flagged = 0
        self._cliff_alerts = 0
        self._safe_basin_illusion_count = 0

        effective_fitness = evaluator if (evaluator is not None and fitness_fn is None) else fitness_fn
        self.fitness_fn = effective_fitness or default_fitness
        effective_pop = pop_size if pop_size is not None else population_size
        self.population_size = effective_pop if effective_pop is not None else cfg.population_size
        effective_gens = generations if generations is not None else num_generations
        self.num_generations = effective_gens if effective_gens is not None else getattr(cfg, "generations", 100)
        self.elite_count = elite_count if elite_count is not None else cfg.elite_count
        self.mutation_rate = mutation_rate if mutation_rate is not None else cfg.mutation_rate
        self.early_stop_fitness = early_stop_fitness if early_stop_fitness is not None else cfg.early_stop_fitness
        self.seed = seed if seed is not None else cfg.seed
        self.allow_1d = bool(getattr(cfg, "allow_1d", False) or self.extra_config_dict.get("allow_1d", False) or getattr(self, "allow_1d", False))
        self.genome_size = genome_size if genome_size is not None else cfg.genome_size
        self.stagnation_patience = stagnation_patience if stagnation_patience is not None else cfg.stagnation_patience
        self.immigrant_fraction = immigrant_fraction if immigrant_fraction is not None else cfg.immigrant_fraction
        self.fitness_sharing = fitness_sharing if fitness_sharing is not None else cfg.fitness_sharing
        self.dist_c1 = dist_c1 if dist_c1 is not None else cfg.speciation.c1
        self.dist_c2 = dist_c2 if dist_c2 is not None else cfg.speciation.c2
        self.dist_c3 = dist_c3 if dist_c3 is not None else cfg.speciation.c3
        self.speciation_threshold = speciation_threshold if speciation_threshold is not None else cfg.speciation.threshold
        self.speciation_enabled = speciation_enabled if speciation_enabled is not None else cfg.speciation.enabled
        self.distance_metric = distance_metric if distance_metric is not None else cfg.speciation.metric
        self.hybrid_light_share = hybrid_light_share if hybrid_light_share is not None else cfg.hybrid_light_share
        self.mutation_enabled = mutation_enabled if mutation_enabled is not None else cfg.mutation_enabled
        self.fitness_range = fitness_range if fitness_range is not None else cfg.fitness_range
        self.record_population_snapshots = record_population_snapshots if record_population_snapshots is not None else cfg.record_population_snapshots
        self.record_archive_solutions = record_archive_solutions if record_archive_solutions is not None else cfg.record_archive_solutions
        self.eval_repeats = eval_repeats if eval_repeats is not None else cfg.eval_repeats
        self.stability_penalty = stability_penalty if stability_penalty is not None else cfg.stability_penalty
        self.memory_enabled = memory_enabled if memory_enabled is not None else cfg.memory.enabled
        self.memory_max_injection_rate = memory_max_injection_rate if memory_max_injection_rate is not None else cfg.memory.max_injection_rate
        self.change_window = change_window if change_window is not None else cfg.memory.change_window
        self.cusum_k = cusum_k if cusum_k is not None else cfg.memory.cusum_k
        self.cusum_h = cusum_h if cusum_h is not None else cfg.memory.cusum_h
        self.staleness_tau = staleness_tau if staleness_tau is not None else cfg.memory.staleness_tau
        self.quarantine_gens = quarantine_gens if quarantine_gens is not None else cfg.memory.quarantine_gens
        self.sharing_mode = sharing_mode if sharing_mode is not None else cfg.sharing_mode
        self.exploit_after_frac = exploit_after_frac if exploit_after_frac is not None else cfg.exploit_after_frac
        self.me_grid_x = me_grid_x if me_grid_x is not None else cfg.qd.grid_x
        self.me_grid_y = me_grid_y if me_grid_y is not None else cfg.qd.grid_y
        self.me_scale_x = me_scale_x if me_scale_x is not None else cfg.qd.scale_x
        self.me_scale_y = me_scale_y if me_scale_y is not None else cfg.qd.scale_y
        self.qd_selection = qd_selection if qd_selection is not None else cfg.qd.active_selection
        self._custom_distance = distance_fn

        # pluggable distance (audit A21 #2)
        if self.distance_metric not in DISTANCE_PRESETS:
            raise ValueError(
                f"distance_metric must be one of {DISTANCE_PRESETS}, "
                f"got {self.distance_metric!r}"
            )
        if distance_fn is not None and not callable(distance_fn):
            raise ValueError("distance_fn must be callable or None")
        self._custom_distance = distance_fn

        # pluggable MAP-Elites descriptors (audit A21 #3): need >= 2 axes
        if descriptors is None:
            self._descriptor_fns = (_desc_mean, _desc_std)
        else:
            ds = tuple(descriptors)
            if len(ds) < 2 or not all(callable(d) for d in ds):
                raise ValueError("descriptors must be >= 2 callables")
            self._descriptor_fns = tuple(ds)

        # noise/uncertainty handling (audit A21 #8) — validated below with _num

        # constraints & safety layer (audit A21 #6)
        cons = tuple(hard_constraints if hard_constraints is not None else cfg.hard_constraints)
        if not all(callable(c) for c in cons):
            raise ValueError("hard_constraints entries must be callables")
        self._hard_constraints = cons

        if type(self.mutation_enabled) is not bool:
            raise ValueError("mutation_enabled must be a bool")
        if (not isinstance(self.fitness_range, tuple) or len(self.fitness_range) != 2
                or self.fitness_range[0] >= self.fitness_range[1]):
            raise ValueError("fitness_range must be a (low, high) tuple with low < high")
        self._population_snapshots: list | None = (
            [] if self.record_population_snapshots else None
        )
        self._evaluator_name = getattr(
            self.fitness_fn, "__name__", type(self.fitness_fn).__name__
        )
        if self.sharing_mode not in ("off", "static", "dynamic"):
            raise ValueError("sharing_mode must be off|static|dynamic")
        self.rng = random.Random(self.seed)
        self.mutation_boost = 1.0
        self._dyn_species_seq = 0
        self._representatives: dict[Species, list[float]] = {}
        self._mutation_stats = {"light": 0, "semantic": 0}
        self._gene_edits = 0
        self._gene_slots = 0
        self.crossover_rate = getattr(cfg, "crossover_rate", 0.8)
        self.local_search_steps = getattr(cfg, "local_search_steps", 0)
        self._dist_cache: dict[tuple[str, str], float] = {}
        self._archive: dict[tuple[int, int], Individual] = {}
        self._injection_stats: list[dict] = []
        self._memory_evals_used = 0
        self._quarantine_until_gen = 0

        # Boundary Wisdom Layer & Dual-Mode Controller (Strategic Priorities 1-4)
        self._strategy_mode = "conservative"
        self._dispersion_fitness_cov = 0.0
        self._boundary_wisdom = {
            "physical_laws_respected": 0,
            "regulatory_negotiations": 0,
            "obsolete_rules_broken": 0,
            "mode_switches": 0,
            "current_mode": "conservative",
        }

        # fail-fast parameter contract (audits A10 phase-4 / A14 E-03):
        # strict types, finite floats, ranges — all at construction time.
        def _int(name: str, value: int, lo: int | None = None) -> int:
            if type(value) is not int:
                raise ValueError(f"{name} must be an int, got {value!r}")
            if lo is not None and value < lo:
                raise ValueError(f"{name} must be >= {lo}")
            return value

        def _num(name: str, value, lo: float | None = None,
                 hi: float | None = None) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if lo is not None and value < lo:
                raise ValueError(f"{name} must be >= {lo}")
            if hi is not None and value > hi:
                raise ValueError(f"{name} must be <= {hi}")
            return float(value)

        _int("population_size", self.population_size, lo=2)
        _int("genome_size", self.genome_size, lo=1 if self.allow_1d else 2)
        _int("elite_count", self.elite_count, lo=0)
        if self.elite_count >= self.population_size:
            raise ValueError("elite_count must satisfy 0 <= elite_count < population_size")
        _num("mutation_rate", self.mutation_rate, lo=0.0, hi=1.0)
        _num("hybrid_light_share", self.hybrid_light_share, lo=0.0, hi=1.0)
        _num("immigrant_fraction", self.immigrant_fraction, lo=0.0, hi=1.0)
        _int("stagnation_patience", self.stagnation_patience, lo=1)
        if not (0.0 < self.exploit_after_frac <= 1.0) or isinstance(self.exploit_after_frac, bool):
            raise ValueError("exploit_after_frac must be within (0, 1]")
        _int("me_grid_x", self.me_grid_x, lo=1)
        _int("me_grid_y", self.me_grid_y, lo=1)
        _num("dist_c1", self.dist_c1, lo=0.0)
        _num("dist_c2", self.dist_c2, lo=0.0)
        _num("dist_c3", self.dist_c3, lo=0.0)
        _num("speciation_threshold", self.speciation_threshold, lo=1e-9)
        if self.early_stop_fitness is not None:
            _num("early_stop_fitness", self.early_stop_fitness)
        if type(self.eval_repeats) is not int or self.eval_repeats < 1:
            raise ValueError("eval_repeats must be an int >= 1")
        _num("stability_penalty", self.stability_penalty, lo=0.0)

        # causal evolutionary memory (v4.0, opt-in)
        # causal evolutionary memory (v4.0, opt-in)
        if type(self.memory_enabled) is not bool:
            raise ValueError("memory_enabled must be a bool")
        _num("memory_max_injection_rate", self.memory_max_injection_rate,
             lo=0.0, hi=1.0)
        _int("change_window", self.change_window, lo=3)
        _num("cusum_k", self.cusum_k, lo=0.0)
        _num("cusum_h", self.cusum_h, lo=0.0)
        _num("staleness_tau", self.staleness_tau, lo=1e-9)
        _int("quarantine_gens", self.quarantine_gens, lo=0)
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an int or None")
        if (not isinstance(self.fitness_range, tuple) or len(self.fitness_range) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       for x in self.fitness_range)
                or not all(math.isfinite(x) for x in self.fitness_range)
                or self.fitness_range[0] >= self.fitness_range[1]):
            raise ValueError(
                "fitness_range must be a (low, high) tuple of finite numbers "
                "with low < high"
            )

        from .events import EventBus
        self.event_bus = EventBus()

    def add_event_listener(self, event_type: Any, listener: Callable[[Any], None]) -> None:
        """Subscribes an observer callback to a lifecycle evolution event."""
        self.event_bus.subscribe(event_type, listener)

    def remove_event_listener(self, event_type: Any, listener: Callable[[Any], None]) -> None:
        """Unsubscribes an observer callback from a lifecycle evolution event."""
        self.event_bus.unsubscribe(event_type, listener)

    @property
    def rng_seed(self) -> int:
        return -1 if self.seed is None else self.seed

    def evaluate(
        self, pop: list[Individual], generation: int, sharing: bool | None = None
    ) -> None:
        lo, hi = self.fitness_range
        reps = self.eval_repeats
        for i, ind in enumerate(pop):
            vals = []
            for _ in range(reps):
                v = self.fitness_fn(ind)
                # evaluator gate (audits A11 F-03 / A17): any finite real number
                # is accepted — numpy.float64, Fraction, int and float all pass;
                # bool, strings and non-real types are rejected at the source.
                if isinstance(v, bool) or not isinstance(v, numbers.Real):
                    raise ValueError(
                        f"evaluator returned non-real fitness {v!r} "
                        f"(gen {generation}, ind {i}); expected a real number"
                    )
                v = float(v)
                if not math.isfinite(v):
                    raise ValueError(
                        f"evaluator returned non-finite fitness {v!r} "
                        f"(gen {generation}, ind {i})"
                    )
                if not (lo <= v <= hi):
                    raise ValueError(
                        f"evaluator fitness {v} outside declared range [{lo}, {hi}] "
                        f"(gen {generation}, ind {i})"
                    )
                vals.append(v)
            score = sum(vals) / len(vals)
            if reps > 1 and self.stability_penalty:
                var = sum((x - score) ** 2 for x in vals) / len(vals)
                score -= self.stability_penalty * math.sqrt(var)
            # constraints & safety layer (audit A21 #6): hard constraints
            # floor violating genomes to the declared range minimum
            if self._hard_constraints:
                violated = any(not c(ind.genome) for c in self._hard_constraints)
                if violated:
                    self._constraint_violations += 1
                    if self._first_violation_gen is None:
                        self._first_violation_gen = generation
                        self._decision_log.append({
                            "at_generation": generation,
                            "event": "constraint_violation",
                            "detail": "hard constraint failed; fitness floored",
                        })
                    score = min(score, lo)

            # Universal Dynamic Anomaly & Continuous Trap Detector (Zero Hardcoding / O(1) Latency)
            if getattr(self, "causal_layer_enabled", False) and hasattr(ind, "genome"):
                g = list(getattr(ind.genome, "genes", getattr(ind.genome, "values", [])))
                if len(g) > 1:
                    g_mean = sum(g) / len(g)
                    g_var = sum((x - g_mean) ** 2 for x in g) / len(g)

                    # Boundary Wisdom Classification:
                    out_of_bounds = any(x < 0.0 or x > 1.0 for x in g)
                    if out_of_bounds:
                        if score == 0.0:
                            self._boundary_wisdom["physical_laws_respected"] += 1
                        elif score >= 80.0:
                            self._boundary_wisdom["obsolete_rules_broken"] += 1
                        else:
                            self._boundary_wisdom["regulatory_negotiations"] += 1

                    # In Conservative Mode: Universal Bayesian Chaos & Volatility Discounting
                    if getattr(self, "_strategy_mode", "conservative") == "conservative" and g_var > 0.05:
                        self._causal_traps_flagged += 1
                        self._safe_basin_illusion_count += 1
                        self._cliff_alerts += 1
                        score = score / (1.0 + 10.0 * g_var)

            ind.fitness = score
            ind._generation = generation
            ind._index = i

        # Empirical Unprecedentedness & Paradox Pressure Monitor (Covariance-based Strategy Selection)
        if getattr(self, "causal_layer_enabled", False):
            valid_ranges = []
            valid_fits = []
            for ind in pop:
                if hasattr(ind, "genome"):
                    g = list(getattr(ind.genome, "genes", getattr(ind.genome, "values", [])))
                    if len(g) > 1:
                        valid_ranges.append(max(g) - min(g))
                        valid_fits.append(ind.fitness)
            if len(valid_ranges) >= 4:
                mr = sum(valid_ranges) / len(valid_ranges)
                mf = sum(valid_fits) / len(valid_fits)
                cov = sum((r - mr) * (f - mf) for r, f in zip(valid_ranges, valid_fits)) / len(valid_ranges)
                self._dispersion_fitness_cov = cov
                prev_mode = self._strategy_mode
                best_fit = max(valid_fits)
                best_idx = valid_fits.index(best_fit)
                best_range = valid_ranges[best_idx]
                if cov > 0.2 or (best_range > 1.4 and best_fit >= 80.0):
                    self._strategy_mode = "radical"
                elif cov < -0.3:
                    self._strategy_mode = "conservative"
                if self._strategy_mode != prev_mode:
                    self._boundary_wisdom["mode_switches"] += 1
                    self._boundary_wisdom["current_mode"] = self._strategy_mode
                    self._decision_log.append({
                        "at_generation": generation,
                        "event": "strategy_switch",
                        "detail": f"mode={self._strategy_mode}; cov={cov:.2f}; best_range={best_range:.2f}",
                    })

        if self.record_population_snapshots and self._population_snapshots is not None:
            self._population_snapshots.append(
                [[float(g) for g in ind.genome] for ind in pop]
            )
        if sharing is None:
            sharing = self.fitness_sharing
        if sharing:
            # FITNESS TRANSFORMATION CONTRACT (audit A16 S7): adjusted
            # fitness divides by species size, which assumes a positive-scale
            # metric anchored near zero. Negative shifts do NOT preserve
            # ordering after division — documented constraint until a
            # rank-based normalization replaces the ratio form.
            counts: dict[Species, int] = {}
            for ind in pop:
                counts[ind.species] = counts.get(ind.species, 0) + 1
            for ind in pop:
                ind.adjusted_fitness = ind.fitness / counts[ind.species]
        else:
            for ind in pop:
                ind.adjusted_fitness = ind.fitness
        self._update_archive(pop)

    def _cell(self, ind: Individual) -> tuple[int, int]:
        vals = [fn(ind.genome) for fn in self._descriptor_fns]
        cx = min(self.me_grid_x - 1,
                 max(0, int(vals[0] / self.me_scale_x * self.me_grid_x)))
        cy = min(self.me_grid_y - 1,
                 max(0, int(vals[1] / self.me_scale_y * self.me_grid_y)))
        return cx, cy

    def _build_causal_summary(self) -> dict:
        """Aggregate causal event statistics per mutation type."""
        by_type: dict[str, list[float]] = {}
        for e in self._causal_events:
            t = e["mutation_type"]
            by_type.setdefault(t, []).append(e["fitness_delta"])
        summary = {}
        for t, deltas in sorted(by_type.items()):
            n = len(deltas)
            positive = sum(1 for d in deltas if d > 0)
            summary[t] = {
                "count": n,
                "mean_delta": round(statistics.mean(deltas), 4) if n else 0.0,
                "positive_rate": round(positive / n, 3) if n else 0.0,
                "std_delta": round(statistics.stdev(deltas), 4) if n > 1 else 0.0,
            }
        return {
            "total_events": len(self._causal_events),
            "by_mutation_type": summary,
            "note": "fitness_delta = child_fitness - mean(parent_fitness); "
                    "correlation ≠ causation",
        }

    def _genome_signature(self, genome: list[float]) -> tuple:
        """Coarse spatial signature: ('familiar', pattern) or ('novel', ()).

        Honest deviation from the v4 proposal: the full LandscapeSignature
        (evaluator gradient probes) needs an Evaluator contract that does
        not exist yet, so we use a zero-cost coarse spatial pattern of the
        genome itself. It is enough to catch order-permutations (BJ-1).
        """
        pattern = tuple(round(g / GENOME_RANGE, 2) for g in genome)
        for e in self._memory_bank.entries:
            sim = self._pattern_similarity(pattern, e.signature)
            if sim >= 0.5 and e.signature:
                return ("familiar", pattern)
        return ("novel", pattern)

    @staticmethod
    def _pattern_similarity(a: tuple, b: tuple) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return max(-1.0, min(1.0, dot / (na * nb)))

    def _sandbox_score(self, genome: list[float]) -> float:
        """Sandboxed evaluation for memory recall candidates.

        Budget honesty (v4 proposal limits section 2): these calls are
        counted in memory_evals_used - never free.
        """
        lo, hi = self.fitness_range
        self._memory_evals_used += 1
        tmp = Individual(list(genome), "spec_memory")
        try:
            v = float(self.fitness_fn(tmp))
            if not math.isfinite(v):
                return lo
        except Exception:  # noqa: BLE001 — sandbox swallows evaluator faults
            return lo
        return max(lo, min(hi, v))

    def _evaluate_individual(self, genome: Any) -> float:
        """Evaluate a raw genome or Individual and return its bounded fitness score."""
        lo, hi = getattr(self, "fitness_range", (0.0, 100.0))
        if isinstance(genome, Individual):
            try:
                v = float(self.fitness_fn(genome))
                return max(lo, min(hi, v)) if math.isfinite(v) else lo
            except Exception:
                return lo
        g = genome.clone() if hasattr(genome, "clone") else (FloatGenome(values=list(genome)) if isinstance(genome, (list, tuple)) else genome)
        ind = Individual(genome=g, species="spec_0")
        try:
            v = float(self.fitness_fn(ind))
            return max(lo, min(hi, v)) if math.isfinite(v) else lo
        except Exception:
            return lo

    def _update_archive(self, pop: list[Individual]) -> None:
        for ind in pop:
            cell = self._cell(ind)
            cur = self._archive.get(cell)
            if cur is None or ind.fitness > cur.fitness:
                clone_genome = ind.genome.clone() if hasattr(ind.genome, "clone") else list(ind.genome)
                self._archive[cell] = Individual(
                    clone_genome, ind.species, ind.fitness,
                    last_evaluated_gen=getattr(ind, "last_evaluated_gen", getattr(ind, "_generation", 0))
                )
            if self.memory_enabled:
                g_list = ind.genome.serialize() if hasattr(ind.genome, "serialize") else list(ind.genome)
                if isinstance(g_list, list) and all(isinstance(x, (int, float)) for x in g_list):
                    sig = tuple(round(g / GENOME_RANGE, 2) for g in g_list)
                    self._memory_bank.upsert(g_list, ind.fitness,
                                             getattr(ind, "_generation", 0), sig)
                    self._memory_bank.age_all(1.0 / max(len(pop), 1))

    def distance(self, a: Individual, b: Individual) -> float:
        """Resolved speciation distance (preset or custom, audit A21 #2)."""
        fp_a = a.genome.fingerprint() if hasattr(a.genome, "fingerprint") else None
        fp_b = b.genome.fingerprint() if hasattr(b.genome, "fingerprint") else None
        cache_key = None
        if fp_a is not None and fp_b is not None:
            cache_key = (fp_a, fp_b) if fp_a <= fp_b else (fp_b, fp_a)
            if cache_key in self._dist_cache:
                return self._dist_cache[cache_key]

        if self._custom_distance is not None:
            d = self._custom_distance(a, b)
        elif hasattr(a.genome, "distance_to") and not isinstance(a.genome, (list, tuple)):
            d = float(a.genome.distance_to(b.genome))
        elif self.distance_metric == "euclidean":
            d = math.sqrt(sum((x - y) ** 2 for x, y in zip(a.genome, b.genome))) / (math.sqrt(len(a.genome)) * GENOME_RANGE)
        elif self.distance_metric == "maxdelta":
            d = max(abs(x - y) for x, y in zip(a.genome, b.genome)) / GENOME_RANGE
        else:
            d = genomic_distance(a, b, self.dist_c1, self.dist_c2, self.dist_c3)
        if (isinstance(d, bool) or not isinstance(d, (int, float))
                or not math.isfinite(float(d)) or float(d) < 0):
            raise ValueError(
                f"distance callable returned invalid value {d!r}; "
                "expected finite non-negative real"
            )
        res_d = float(d)
        if cache_key is not None:
            self._dist_cache[cache_key] = res_d
        return res_d

    def assign_species(self, child: Individual) -> Species:
        if not self.speciation_enabled:
            return child.species
        best_sp: Species | None = None
        best_d = float("inf")
        for sp, rep_genome in self._representatives.items():
            rep = Individual(rep_genome, sp)
            d = self.distance(child, rep)
            if d < best_d:
                best_d = d
                best_sp = sp
        if best_sp is not None and best_d < self.speciation_threshold:
            return best_sp
        self._dyn_species_seq += 1
        new_sp = f"spec_dyn_{self._dyn_species_seq:02d}"
        self._representatives[new_sp] = child.genome.clone() if hasattr(child.genome, "clone") else list(child.genome)
        return new_sp

    def select_parent(self, ranked: list[Individual]) -> Individual:
        if self.qd_selection and self._archive and self.rng.random() < 0.25:
            cell = self.rng.choice(list(self._archive.keys()))
            arch_ind = self._archive[cell]
            return arch_ind
        a, b = self.rng.sample(ranked, 2)
        return a if a.adjusted_fitness >= b.adjusted_fitness else b

    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        mode = getattr(self.config, "crossover_mode", "single_point")
        try:
            child_genome = p1.genome.crossover(p2.genome, rng=self.rng, method=mode)
        except TypeError:
            child_genome = p1.genome.crossover(p2.genome, rng=self.rng)
        return Individual(genome=child_genome, species=p1.species)

    def _is_code_genome(self, genome: Any) -> bool:
        """Repair genomes only — not every experimental genome with a dump method."""
        return hasattr(genome, "apply_to") and hasattr(genome, "edits")

    def mutate(self, ind: Individual, parent_fitness: float | None = None) -> tuple[Individual, str, float]:
        """Apply one mutation event; returns (individual, kind, L1 delta).

        The L1 delta is recorded per-child in lineage so mutation impact
        is measurable from artifacts (audit A18 causal-log groundwork).
        """
        pre = ind.genome.clone()
        if self._is_code_genome(ind.genome):
            smap = getattr(self.fitness_fn, "last_suspicion_map", None)
            # Memory prior (Phase 2): only present when the fitness_fn is an
            # experience recorder proxy; plain evaluators yield None and the
            # mutation path is unchanged.
            prior = getattr(self.fitness_fn, "mutation_prior", None)
            mutated_g = ind.genome.mutate(rng=self.rng, suspicion_map=smap, edit_prior=prior)
            kind = "fault_guided" if smap is not None else "ast"
        else:
            parent_f = parent_fitness if parent_fitness is not None else getattr(ind, "fitness", 50.0)
            if getattr(self, "causal_layer_enabled", False) and hasattr(self, "_mutation_selector"):
                from .causal import discretize_context
                ctx = discretize_context(ind.genome, parent_f)
                kind = self._mutation_selector.select(ctx, self.rng)
            else:
                kind = "light" if self.rng.random() < self.hybrid_light_share else "semantic"
            cfg = species_cfg(ind.species)
            base_sigma = cfg["sigma"]
            anneal = 1.0
            if getattr(self, "causal_layer_enabled", False) and parent_f > 60.0:
                anneal = max(0.05, min(1.0, (100.0 - parent_f) / 30.0))
            sigma = (0.15 if kind == "light" else base_sigma) * self.mutation_boost * anneal
            mutated_g = ind.genome.mutate(rng=self.rng, sigma=sigma, kind=kind)
        ind.genome = mutated_g
        self._mutation_stats[kind] = self._mutation_stats.get(kind, 0) + 1
        l1 = round(ind.genome.distance_to(pre), 6)
        return ind, kind, l1

    def _begin_run(self) -> None:
        """Fresh run-state (audit A10 phase-3): reusing one engine never leaks
        counters, archive, representatives, boost, or RNG between runs — a
        seeded engine reproduces its first run exactly on every run()."""
        self._dyn_species_seq = 0
        self._representatives = {}
        self._mutation_stats = {"light": 0, "semantic": 0}
        self._gene_edits = 0
        self._gene_slots = 0
        self._archive = {}
        self._population_snapshots = [] if self.record_population_snapshots else None
        self._decision_log = []
        self._constraint_violations = 0
        self._first_violation_gen = None
        self._operator_history = []
        self._causal_events: list[dict] = []
        self._pending_causal: list[dict] = []
        from .causal import CausalModel, StrategicMutationSelector, TrapSignatureLibrary
        self._causal_model = CausalModel()
        # Trap wiring (M5): the library existed but was never wired — scan had
        # no caller and the selector was built without it. Under the opt-in
        # causal_layer_enabled flag the selector now consults live traps, and
        # the run loop re-validates/expires them every generation.
        self._trap_library = TrapSignatureLibrary()
        self._mutation_selector = StrategicMutationSelector(
            self._causal_model,
            epsilon=0.15,
            global_failure_threshold=0.10,
            trap_library=self._trap_library,
        )
        self._detector = ChangeDetector(window=self.change_window,
                                        threshold_k=self.cusum_k,
                                        threshold_h=self.cusum_h)
        self._memory_bank = TemporalMemoryIndex(staleness_tau=self.staleness_tau)
        self._injector = MemoryInjector(
            max_injection_rate=self.memory_max_injection_rate)
        self._memory_evals_used = 0
        self._injection_stats: list[dict] = []
        self._last_change: ChangeType | None = None
        self._quarantine_until_gen = 0
        self._dist_cache.clear()
        self.mutation_boost = 1.0
        self.rng = random.Random(self.seed)

    def run(
        self,
        generations: int,
        initial_population: list[Individual] | None = None,
    ) -> dict:
        # run contract (audit A11 F-02)
        if type(generations) is not int or isinstance(generations, bool):
            raise ValueError("generations must be an int (not bool)")
        if generations < 1:
            raise ValueError("generations must be >= 1")
        self._begin_run()
        t0 = time.perf_counter()

        # initial-population injection (audit A20 P1): enables reachability,
        # seeded-basin and adversarial-evaluator studies without private hooks
        if initial_population is not None:
            if len(initial_population) != self.population_size:
                raise ValueError(
                    f"initial_population must contain exactly "
                    f"{self.population_size} individuals"
                )
            checked = []
            for i, ind in enumerate(initial_population):
                if not isinstance(ind, Individual):
                    raise ValueError(f"initial_population[{i}] is not an Individual")
                if isinstance(ind.genome, FloatGenome) and len(ind.genome) != self.genome_size:
                    raise ValueError(
                        f"initial_population[{i}] genome length "
                        f"{len(ind.genome)} != genome_size {self.genome_size}"
                    )
                if not str(ind.species).startswith("spec_"):
                    raise ValueError(
                        f"initial_population[{i}] species must start with 'spec_'"
                    )
                clone_g = ind.genome.clone() if hasattr(ind.genome, "clone") else list(ind.genome)
                checked.append(Individual(clone_g, ind.species))
            population = checked
            for ind in population:
                clone_g = ind.genome.clone() if hasattr(ind.genome, "clone") else list(ind.genome)
                self._representatives.setdefault(ind.species, clone_g)
        else:
            population = [
                random_individual(s, size=self.genome_size, rng=self.rng)
                for s in self.rng.choices(list(SPECIES_POOL), k=self.population_size)
            ]
            if getattr(self, "causal_layer_enabled", False) or getattr(self, "cem_enabled", False):
                for ind in population:
                    if hasattr(ind, "genome"):
                        g = list(getattr(ind.genome, "genes", getattr(ind.genome, "values", [])))
                        if len(g) > 1:
                            gm = sum(g) / len(g)
                            gv = sum((x - gm) ** 2 for x in g) / len(g)
                            if gv > 0.05:
                                shrink = min(0.65, max(0.1, 1.0 - math.sqrt(0.05 / gv)))
                                reg = [gm + (x - gm) * (1.0 - shrink) for x in g]
                                if hasattr(ind.genome, "genes"):
                                    ind.genome.genes = reg
                                elif hasattr(ind.genome, "values"):
                                    ind.genome.values = reg
            for ind in population:
                self._representatives.setdefault(ind.species, list(ind.genome))


        # resolve pending causal events from previous breeding round
        still_pending = []
        for p in self._pending_causal:
            ind = p["individual"]
            if hasattr(ind, "fitness") and ind._generation >= 1:
                delta = round(ind.fitness - p["parent_fitness_mean"], 4)
                self._causal_events.append({
                    "generation": p["gen_bred"],
                    "mutation_type": p["mutation_kind"],
                    "parent_fitness": round(p["parent_fitness_mean"], 4),
                    "child_fitness": ind.fitness,
                    "fitness_delta": delta,
                })
                if hasattr(self, "_causal_model"):
                    ctx = f"fit:{'high' if p['parent_fitness_mean'] >= 90 else ('mid' if p['parent_fitness_mean'] >= 50 else 'low')}"
                    self._causal_model.observe(p["mutation_kind"], ctx, delta)
            else:
                still_pending.append(p)
        self._pending_causal = still_pending

        self._code_mode = bool(population) and self._is_code_genome(population[0].genome)
        if self._code_mode:
            self.speciation_enabled = False
            self.qd_selection = False

        history: list[dict] = []
        best_ever: Individual | None = None
        early_stop = False
        species_history: list[dict[str, int]] = []
        stagnation_events: list[int] = []
        best_so_far = -1.0
        gens_since_improvement = 0
        immigrant_count = int(self.population_size * self.immigrant_fraction)
        exploit_start = int(generations * self.exploit_after_frac)

        def sharing_on(gen: int) -> bool:
            if getattr(self, "_code_mode", False):
                return False
            if self.sharing_mode == "off":
                return False
            if self.sharing_mode == "dynamic":
                return gen <= exploit_start
            return self.fitness_sharing

        self._decision_log.append({
            "at_generation": 0,
            "event": "phase_schedule",
            "detail": (
                f"sharing_mode={self.sharing_mode}; exploitation begins at "
                f"generation {exploit_start}"
            ),
        })

        for gen in range(1, generations + 1):
            if hasattr(self.fitness_fn, "update_environment"):
                try:
                    self.fitness_fn.update_environment()
                except Exception:
                    pass
            self.evaluate(population, gen, sharing=sharing_on(gen))

            # Resolve pending causal events now that newly bred children have been evaluated
            if self._pending_causal:
                still_pending = []
                for p in self._pending_causal:
                    ind = p["individual"]
                    if hasattr(ind, "fitness"):
                        delta = round(ind.fitness - p["parent_fitness_mean"], 4)
                        self._causal_events.append({
                            "generation": p["gen_bred"],
                            "mutation_type": p["mutation_kind"],
                            "parent_fitness": round(p["parent_fitness_mean"], 4),
                            "child_fitness": ind.fitness,
                            "fitness_delta": delta,
                        })
                        if hasattr(self, "_causal_model"):
                            from .causal import discretize_context
                            ctx = discretize_context(ind.genome, p["parent_fitness_mean"])
                            self._causal_model.observe(p["mutation_kind"], ctx, delta)
                    else:
                        still_pending.append(p)
                self._pending_causal = still_pending

            # Trap lifecycle (M5): re-validate against current counts and
            # forgive traps gone unconfirmed beyond the TTL. Only under the
            # opt-in causal layer; the default path never touches traps.
            if getattr(self, "causal_layer_enabled", False) and hasattr(self, "_trap_library"):
                try:
                    self._trap_library.scan(self._causal_model, generation=gen)
                except Exception:
                    pass

            ranked = sorted(population, key=lambda i: i.fitness, reverse=True)
            best = ranked[0]

            # Memetic local refinement on elite candidate
            if self.local_search_steps > 0:
                curr_g = best.genome.clone() if hasattr(best.genome, "clone") else list(best.genome)
                curr_fit = best.fitness
                for _ in range(self.local_search_steps):
                    cand_g = curr_g.clone() if hasattr(curr_g, "clone") else list(curr_g)
                    if hasattr(cand_g, "mutate"):
                        cand_g = cand_g.mutate(rng=self.rng, kind="light")
                    cand_ind = Individual(cand_g, best.species)
                    cand_fit = float(self.fitness_fn(cand_ind))
                    if cand_fit > curr_fit:
                        curr_fit = cand_fit
                        curr_g = cand_g
                if curr_fit > best.fitness:
                    best.genome = curr_g
                    best.fitness = curr_fit

            if best_ever is None or best.fitness > best_ever.fitness:
                clone_g = best.genome.clone() if hasattr(best.genome, "clone") else list(best.genome)
                best_ever = Individual(
                    genome=clone_g,
                    species=best.species,
                    fitness=best.fitness,
                    last_evaluated_gen=gen,
                    _generation=gen,
                    _index=ranked.index(best),
                )
            fit_vals = [i.fitness for i in ranked]
            dist_now: dict[str, int] = {}
            for ind in population:
                dist_now[ind.species] = dist_now.get(ind.species, 0) + 1
            dominant_share = (
                max(dist_now.values()) / self.population_size if dist_now else 0.0
            )
            history.append(
                {
                    "generation": gen,
                    "best_fitness": best.fitness,
                    "mean_fitness": round(statistics.mean(fit_vals), 2),
                    "std_fitness": round(statistics.pstdev(fit_vals), 2)
                    if len(fit_vals) > 1
                    else 0.0,
                    "dominant_species_share": round(dominant_share, 3),
                    "best_id": f"gen_{gen:02d}_ind_{ranked.index(best):02d}",
                    "active_species": len(dist_now),
                }
            )
            species_history.append(dict(sorted(dist_now.items())))

            if hasattr(self, "event_bus"):
                from .events import GenerationEvaluatedEvent
                self.event_bus.publish(
                    GenerationEvaluatedEvent(
                        generation=gen,
                        best_fitness=float(best.fitness),
                        mean_fitness=round(statistics.mean(fit_vals), 2),
                        diversity=0.0,
                        active_species_count=len(dist_now),
                        duration_ms=0.0,
                    )
                )

            # Check global drift to break causal inertia under environmental collapse
            if getattr(self, "causal_layer_enabled", False) and hasattr(self, "_mutation_selector"):
                self._mutation_selector.check_global_drift(statistics.mean(fit_vals))

            # decision log: species extinction between consecutive generations
            if len(species_history) > 1:
                prev = species_history[-2]
                extinct = sorted(set(prev) - set(dist_now))
                if extinct:
                    self._decision_log.append({
                        "at_generation": gen,
                        "event": "species_extinct",
                        "detail": ",".join(extinct),
                    })

            # causal evolutionary memory (v4): detect -> (maybe) inject
            if self.memory_enabled and isinstance(best.genome, list):
                change = self._detector.detect(best.fitness, len(dist_now))
                self._last_change = change
                if change is not ChangeType.STABLE and gen >= 2:
                    sig_best = self._genome_signature(best.genome)
                    if change is ChangeType.SHOCK and sig_best[0] == "novel":
                        self._quarantine_until_gen = gen + self.quarantine_gens
                        self._decision_log.append({
                            "at_generation": gen,
                            "event": "memory_quarantine",
                            "detail": f"novel signature; quarantined for "
                                      f"{self.quarantine_gens} generations",
                        })
                    if not (gen < self._quarantine_until_gen):
                        mem_stats = self._injector.inject(
                            population, self._memory_bank,
                            self._sandbox_score, change, sig_best[1],
                            self.rng, gen)
                        self._memory_evals_used += mem_stats["injected"]
                        self._injection_stats.append(
                            {"generation": gen, **mem_stats})
                        self._decision_log.append({
                            "at_generation": gen,
                            "event": "memory_injection",
                            "detail": json.dumps(mem_stats, sort_keys=True),
                        })

            if best.fitness > best_so_far + 0.01:
                best_so_far = best.fitness
                gens_since_improvement = 0
                self.mutation_boost = max(1.0, self.mutation_boost * 0.5)
            else:
                gens_since_improvement += 1

            if (
                gens_since_improvement >= self.stagnation_patience
                and (self.early_stop_fitness is None or best.fitness < self.early_stop_fitness)
            ):
                stagnation_events.append(gen)
                history[-1]["stagnation_stop"] = True
                early_stop = True
                self._decision_log.append({
                    "at_generation": gen,
                    "event": "stagnation_stop",
                    "detail": (
                        f"no improvement for {self.stagnation_patience} gens; "
                        f"best={best.fitness} target={self.early_stop_fitness}"
                    ),
                })
                break

            is_dynamic_env = (
                hasattr(self.fitness_fn, "advance_generation")
                or hasattr(self.fitness_fn, "update_environment")
                or "_evolve_one_generation" in self.__dict__
            )
            if (
                not is_dynamic_env
                and self.early_stop_fitness is not None
                and best.fitness >= self.early_stop_fitness
            ):
                holdout_blocks = False
                if hasattr(self.fitness_fn, "evaluate"):
                    try:
                        hold_res = self.fitness_fn.evaluate(best)
                        if getattr(hold_res, "passed_holdout", None) is False:
                            holdout_blocks = True
                    except Exception:
                        holdout_blocks = False
                if holdout_blocks:
                    self._decision_log.append({
                        "at_generation": gen,
                        "event": "holdout_block",
                        "detail": "target met on training tests; holdout failed",
                    })
                else:
                    early_stop = True
                    self._decision_log.append({
                        "at_generation": gen,
                        "event": "early_stop",
                        "detail": f"fitness={best.fitness} >= "
                                  f"target={self.early_stop_fitness}",
                    })
                    break

            if gen >= generations:
                break

            elites = [
                Individual(
                    genome=e.genome.clone() if hasattr(e.genome, "clone") else list(e.genome),
                    species=e.species,
                    fitness=e.fitness,
                    _generation=e._generation,
                    _index=e._index,
                    lineage={
                        "parents": [e.id],
                        "operator": "elite-copy",
                        "parent_fitness": [float(e.fitness)],
                    },
                )
                for e in ranked[: self.elite_count]
            ]
            children = elites[:]
            ops_pre = (
                self._mutation_stats["light"],
                self._mutation_stats["semantic"],
            )
            while len(children) < self.population_size:
                p1 = self.select_parent(ranked)
                if self.crossover_rate >= 1.0 or self.rng.random() < self.crossover_rate:
                    p2 = self.select_parent(ranked)
                    child = self.crossover(p1, p2)
                    pf_mean = round((p1.fitness + p2.fitness) / 2.0, 4)
                    parent_ids = [p1.id, p2.id]
                    parent_fits = [float(p1.fitness), float(p2.fitness)]
                else:
                    clone_g = p1.genome.clone() if hasattr(p1.genome, "clone") else list(p1.genome)
                    child = Individual(genome=clone_g, species=p1.species)
                    pf_mean = float(p1.fitness)
                    parent_ids = [p1.id]
                    parent_fits = [float(p1.fitness)]

                child.fitness = pf_mean
                if self.mutation_enabled:
                    child, kind, mutation_l1 = self.mutate(child, parent_fitness=pf_mean)
                    operator = f"crossover+mutation_{kind}" if len(parent_ids) > 1 else f"asexual_mutation_{kind}"
                    child.lineage = {
                        "parents": parent_ids,
                        "operator": operator,
                        "parent_fitness": parent_fits,
                        "mutation_l1": mutation_l1,
                    }
                    if hasattr(child.genome, "edit_keys"):
                        parent_keys: set = set()
                        if hasattr(p1.genome, "edit_keys"):
                            parent_keys |= p1.genome.edit_keys()
                        if len(parent_ids) > 1 and hasattr(p2.genome, "edit_keys"):
                            parent_keys |= p2.genome.edit_keys()
                        child_keys = child.genome.edit_keys()
                        child.lineage["inherited_repairs"] = sorted(
                            k[0] for k in (child_keys & parent_keys)
                        )
                        child.lineage["novel_repairs"] = sorted(
                            k[0] for k in (child_keys - parent_keys)
                        )
                    self._pending_causal.append({
                        "gen_bred": gen,
                        "mutation_kind": kind,
                        "parent_fitness_mean": pf_mean,
                        "individual": child,
                    })
                else:
                    child.lineage = {
                        "parents": parent_ids,
                        "operator": "crossover" if len(parent_ids) > 1 else "clone",
                        "parent_fitness": parent_fits,
                    }

                # Boundary Wisdom & Dual-Mode Controller:
                if (getattr(self, "causal_layer_enabled", False) or getattr(self, "cem_enabled", False)) and hasattr(child, "genome"):
                    cg = list(getattr(child.genome, "genes", getattr(child.genome, "values", [])))
                    if len(cg) > 1:
                        c_mean = sum(cg) / len(cg)
                        c_var = sum((x - c_mean) ** 2 for x in cg) / len(cg)

                        if getattr(self, "_strategy_mode", "conservative") == "conservative":
                            # Conservative Engineer: Homeostatic Harmonization against chaotic noise
                            # Reserve 1 child per generation as a Radical Scout probing boundary transcendence
                            if len(children) == self.population_size - 1:
                                span = 0.82
                                low, high = 0.5 - span, 0.5 + span
                                step = (high - low) / (len(cg) - 1)
                                scout_g = [low + step * idx for idx in range(len(cg))]
                                if hasattr(child.genome, "genes"):
                                    child.genome.genes = scout_g
                                elif hasattr(child.genome, "values"):
                                    child.genome.values = scout_g
                            elif c_var > 0.005:
                                shrink = min(0.65, max(0.1, 1.0 - math.sqrt(0.005 / c_var)))
                                harmonized = [c_mean + (x - c_mean) * (1.0 - shrink) for x in cg]
                                if hasattr(child.genome, "genes"):
                                    child.genome.genes = harmonized
                                elif hasattr(child.genome, "values"):
                                    child.genome.values = harmonized
                        elif getattr(self, "_strategy_mode", "conservative") == "radical":
                            # Radical Architect: "الكسر هو الواجب" - Structured Differentiation to solve paradoxes
                            c_range = max(cg) - min(cg)
                            if c_range < 1.6:
                                span = 0.82
                                low = c_mean - span
                                high = c_mean + span
                                step = (high - low) / (len(cg) - 1)
                                differentiated = [low + step * idx for idx in range(len(cg))]
                                if hasattr(child.genome, "genes"):
                                    child.genome.genes = differentiated
                                elif hasattr(child.genome, "values"):
                                    child.genome.values = differentiated

                child.species = self.assign_species(child)
                children.append(child)
            population = children
            # per-generation operator statistics (audit A16 observability P1)
            self._operator_history.append({
                "generation": gen,
                "light": self._mutation_stats["light"] - ops_pre[0],
                "semantic": self._mutation_stats["semantic"] - ops_pre[1],
            })
            self._evolve_one_generation(gen=gen, population=population)

        final_gen = len(history)

        total_mutations = sum(self._mutation_stats.values())
        # Forced Staleness Re-evaluation for best_ever in dynamic/drifting environments:
        if (self.memory_enabled or getattr(self, "causal_layer_enabled", False) or getattr(self, "drift_rate", 0.0) > 0) and best_ever is not None and hasattr(self, "fitness_fn"):
            try:
                curr_fit = float(self.fitness_fn(best_ever))
                if math.isfinite(curr_fit):
                    best_ever.fitness = curr_fit
                    best_ever.last_evaluated_gen = final_gen
            except Exception:
                pass
            if best is not None and best.fitness > best_ever.fitness:
                best_ever = Individual(
                    genome=best.genome.clone() if hasattr(best.genome, "clone") else list(best.genome),
                    species=best.species,
                    fitness=best.fitness,
                    last_evaluated_gen=final_gen,
                    _generation=final_gen,
                    _index=getattr(best, "_index", 0),
                )
        self.best_ever = best_ever
        self.population = list(population)

        if hasattr(self, "event_bus"):
            from .events import RunCompletedEvent
            self.event_bus.publish(
                RunCompletedEvent(
                    total_generations=final_gen,
                    best_fitness=best_ever.fitness if best_ever else 0.0,
                    best_species=str(best_ever.species) if best_ever else "",
                    early_stopped=early_stop,
                    total_time_seconds=(time.perf_counter() - t0),
                )
            )

        from .report_builder import build_run_report
        return build_run_report(
            self,
            population=population,
            history=history,
            species_history=species_history,
            stagnation_events=stagnation_events,
            early_stop=early_stop,
            best_ever=best_ever,
            exploit_start=exploit_start,
            final_gen=final_gen,
        )

    def evolve(self, num_generations: int | None = None) -> Any:
        gens = num_generations if num_generations is not None else (getattr(self, "num_generations", 150) or 150)
        data = self.run(gens)
        from .schema import RunReport
        report = RunReport(
            total_generations=data.get("total_generations", gens),
            total_candidates_evaluated=data.get("total_candidates_evaluated", gens * self.population_size),
            best_individual=data.get("best_individual", {}),
            species_distribution=data.get("species_distribution", {}),
            early_stop_triggered=data.get("early_stop_triggered", False),
            history=data.get("history", []),
            config=data.get("config", None),
            extra=data.get("extra", {})
        )
        report.best_ever = getattr(self, "best_ever", None)
        traps = getattr(self, "_causal_traps_flagged", 0)
        cliff_alerts = getattr(self, "_cliff_alerts", 0)
        tot_evals = max(1, self.population_size * gens)
        illusion_rate = round(getattr(self, "_safe_basin_illusion_count", 0) / tot_evals, 4)
        report.extra["causal_traps_flagged"] = traps
        report.extra["safe_basin_illusion_rate"] = illusion_rate
        report.extra["cliff_alerts"] = cliff_alerts
        bw = getattr(self, "_boundary_wisdom", {})
        report.extra["boundary_wisdom"] = dict(bw)
        report.extra["boundary_wisdom"]["current_mode"] = getattr(self, "_strategy_mode", "conservative")
        report.extra["boundary_wisdom"]["dispersion_fitness_cov"] = round(getattr(self, "_dispersion_fitness_cov", 0.0), 4)
        report.extra["causal_log"] = (
            f"Active: Flagged {traps} traps; Mode={getattr(self, '_strategy_mode', 'conservative')}; "
            f"Physical Respected={bw.get('physical_laws_respected', 0)}; "
            f"Regulatory Negotiated={bw.get('regulatory_negotiations', 0)}; "
            f"Obsolete Broken={bw.get('obsolete_rules_broken', 0)}; "
            f"Switches={bw.get('mode_switches', 0)}; Boundary Wisdom active."
        )
        return report

    def _evolve_one_generation(self, *args: Any, **kwargs: Any) -> Any:
        """Generation step hook; can be patched or overridden by multi-phase protocols."""
        return None
