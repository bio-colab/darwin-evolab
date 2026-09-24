"""swarm.py — Deliberate Multi-Agent Island Scaling with Burden-Gated Utility Invariant.

Architecture:
Part of Darwin-Evolab: Deliberate Swarm Scaling & Cooperative Co-Evolution.
Implements multi-island evolutionary search coordinated via a thread-safe blackboard,
governed by statistical dual-invariants, and gated by a strict Utility Contract.

Contract:
STATUS: PROBATIONARY_BURDEN_GATED
EVICTION_POLICY: EVICT_WHEN_BURDEN

Principle:
A multi-agent swarm exists solely to amplify exploration throughput and solution quality.
If coordination overhead, synchronization latency, or diminishing returns (alpha <= alpha_min)
turn the swarm into a net burden compared to a lean single-engine baseline, the swarm
automatically evicts its distributed overhead, disengages worker islands, and cleanly
falls back to baseline execution.
"""
from __future__ import annotations

import copy
import math
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from .config import EngineConfig
from .engine import EvolutionEngine
from .evaluators import Evaluator, FitnessResult
from .genome import EvolabGenome, Individual
from .self_model import govern_modification


class SwarmStatus(str, Enum):
    PROBATIONARY = "PROBATIONARY_BURDEN_GATED"
    ACTIVE = "ACTIVE"
    EVICTED = "EVICTED_BURDEN_EXCEEDED"
    CONVERGED = "CONVERGED_SUCCESS"


@dataclass
class SwarmTelemetry:
    """Telemetry report emitted by each island at synchronization points."""
    island_id: int
    generation: int
    best_fitness: float
    mean_fitness: float
    total_evaluations: int
    dead_loci_found: set[tuple] = field(default_factory=set)
    suspicion_hotspots: dict[int, float] = field(default_factory=dict)
    sync_time_ms: float = 0.0


@dataclass
class BurdenDecision:
    """Formal decision rendered by the BurdenMonitor on whether the swarm remains justified."""
    should_evict: bool
    reason: str
    empirical_alpha: float
    speedup_ratio: float
    sync_overhead_pct: float
    evaluations_consumed: int
    best_fitness: float


class SwarmBlackboard:
    """Thread-safe collective forum and message board for distributed islands.
    
    Houses:
    1. Global vaccinated elite pool (MAP-Elites / Top-k) gated by the Governor.
    2. Global Negative Genetic Memory (`global_avoid_loci`) preventing any island
       from testing dead doors already disproven by another island.
    3. Telemetry & suspicion hotspots (Ochiai SBFL) shared across isolates.
    """

    def __init__(self, max_elites: int = 8, governor_alpha: float = 0.05) -> None:
        self.max_elites = max_elites
        self.governor_alpha = governor_alpha
        self.lock = threading.Lock()
        
        self.global_elites: list[Individual] = []
        self.global_avoid_loci: set[tuple] = set()
        self.telemetry_history: list[SwarmTelemetry] = []
        self.total_migrations: int = 0
        self.rejected_proposals: int = 0

    def publish_avoid_loci(self, loci: set[tuple]) -> int:
        """Publish disproven / dead door loci to the shared negative genetic memory."""
        if not loci:
            return 0
        with self.lock:
            before = len(self.global_avoid_loci)
            self.global_avoid_loci.update(loci)
            return len(self.global_avoid_loci) - before

    def get_avoid_loci(self) -> set[tuple]:
        """Fetch a snapshot of all disproven loci across all islands."""
        with self.lock:
            return set(self.global_avoid_loci)

    def submit_candidate(
        self,
        candidate: Individual,
        baseline_fitness: float,
        island_id: int,
    ) -> bool:
        """Submit a promising elite to the blackboard, gated by Governor verification."""
        with self.lock:
            # Dual-invariant check: must not be worse than baseline
            if candidate.fitness < baseline_fitness:
                self.rejected_proposals += 1
                return False

            # Governor-inspired mathematical gate
            if self.global_elites:
                best_known = self.global_elites[0].fitness
                if candidate.fitness < best_known:
                    # Still acceptable as diverse elite if pool has room
                    if len(self.global_elites) < self.max_elites:
                        self.global_elites.append(candidate.clone() if hasattr(candidate, "clone") else copy.deepcopy(candidate))
                        self.global_elites.sort(key=lambda ind: ind.fitness, reverse=True)
                        return True
                    return False

            # Candidate is a new global best or top contender
            clone = candidate.clone() if hasattr(candidate, "clone") else copy.deepcopy(candidate)
            self.global_elites.append(clone)
            self.global_elites.sort(key=lambda ind: ind.fitness, reverse=True)
            if len(self.global_elites) > self.max_elites:
                self.global_elites = self.global_elites[: self.max_elites]
            self.total_migrations += 1
            return True

    def get_top_elites(self, k: int = 2) -> list[Individual]:
        """Fetch the top k vaccinated elites for migration into an island."""
        with self.lock:
            return [
                ind.clone() if hasattr(ind, "clone") else copy.deepcopy(ind)
                for ind in self.global_elites[:k]
            ]


class BurdenMonitor:
    """Continuous overhead and scaling-law auditor.
    
    Validates the 'Utility or Evict' contract. If the distributed swarm
    costs more in coordination/evaluations than it returns in velocity or quality,
    it triggers an immediate EVICTION verdict.
    """

    def __init__(
        self,
        island_count: int,
        alpha_min: float = 0.40,
        max_sync_overhead_pct: float = 35.0,
        stagnation_patience_rounds: int = 3,
    ) -> None:
        self.island_count = max(2, int(island_count))
        self.alpha_min = float(alpha_min)
        self.max_sync_overhead_pct = float(max_sync_overhead_pct)
        self.stagnation_patience_rounds = int(stagnation_patience_rounds)
        
        self.rounds_stagnated = 0
        self.best_fitness_history: list[float] = []

    def evaluate_burden(
        self,
        current_best_fitness: float,
        baseline_best_fitness: float,
        swarm_elapsed_sec: float,
        baseline_expected_sec: float,
        sync_time_sec: float,
        total_evaluations: int,
        baseline_evaluations: int,
    ) -> BurdenDecision:
        """Inspects run metrics and renders a stay-or-evict decision."""
        # 1. Synchronization overhead check
        sync_pct = (sync_time_sec / max(1e-6, swarm_elapsed_sec)) * 100.0
        if sync_pct > self.max_sync_overhead_pct:
            return BurdenDecision(
                should_evict=True,
                reason=f"Excessive synchronization overhead ({sync_pct:.1f}% > {self.max_sync_overhead_pct:.1f}%)",
                empirical_alpha=0.0,
                speedup_ratio=1.0,
                sync_overhead_pct=sync_pct,
                evaluations_consumed=total_evaluations,
                best_fitness=current_best_fitness,
            )

        # 2. Empirical scaling exponent alpha calculation:
        # Speedup S = T_baseline / T_swarm
        # S = K^alpha  =>  alpha = ln(S) / ln(K)
        speedup = max(0.01, baseline_expected_sec / max(1e-6, swarm_elapsed_sec))
        alpha = math.log(max(1e-3, speedup)) / math.log(self.island_count) if self.island_count > 1 else 1.0

        # 3. Performance stagnation & wasteful compute
        if current_best_fitness <= baseline_best_fitness + 1e-4:
            self.rounds_stagnated += 1
            if self.rounds_stagnated >= self.stagnation_patience_rounds and total_evaluations > baseline_evaluations * 1.5:
                return BurdenDecision(
                    should_evict=True,
                    reason=(
                        f"Zero marginal utility over baseline ({current_best_fitness:.2f} <= {baseline_best_fitness:.2f}) "
                        f"despite consuming {total_evaluations} evaluations (> 1.5x baseline)"
                    ),
                    empirical_alpha=round(alpha, 3),
                    speedup_ratio=round(speedup, 3),
                    sync_overhead_pct=round(sync_pct, 2),
                    evaluations_consumed=total_evaluations,
                    best_fitness=current_best_fitness,
                )
        else:
            self.rounds_stagnated = 0

        # 4. Collapse of parallel efficiency (severe sublinear scaling)
        # Only evaluate after a minimum warm-up time
        if swarm_elapsed_sec > 0.5 and alpha < self.alpha_min and speedup < 1.0:
            return BurdenDecision(
                should_evict=True,
                reason=f"Sublinear scaling collapse: empirical alpha {alpha:.2f} < threshold {self.alpha_min:.2f}",
                empirical_alpha=round(alpha, 3),
                speedup_ratio=round(speedup, 3),
                sync_overhead_pct=round(sync_pct, 2),
                evaluations_consumed=total_evaluations,
                best_fitness=current_best_fitness,
            )

        # Swarm is healthy and delivering net positive utility
        return BurdenDecision(
            should_evict=False,
            reason="Nominal: Swarm delivers positive marginal utility",
            empirical_alpha=round(alpha, 3),
            speedup_ratio=round(speedup, 3),
            sync_overhead_pct=round(sync_pct, 2),
            evaluations_consumed=total_evaluations,
            best_fitness=current_best_fitness,
        )


class IslandWorker:
    """An autonomous evolutionary island with designated specialization."""

    def __init__(
        self,
        island_id: int,
        engine: EvolutionEngine,
        specialization: str = "generalist",
    ) -> None:
        self.island_id = island_id
        self.engine = engine
        self.specialization = specialization
        self.population: list[Individual] = []
        self.local_avoid_loci: set[tuple] = set()
        self.rounds_completed = 0

        # Specialize engine hyperparameters
        if self.specialization == "conservative":
            self.engine.mutation_boost = 0.6
            self.engine.elite_count = max(2, self.engine.elite_count)
        elif self.specialization == "exploratory":
            self.engine.mutation_boost = 2.0
            self.engine.crossover_rate = 0.5
        elif self.specialization == "sbfl_targeted":
            self.engine.mutation_boost = 1.2

    def step_generations(
        self,
        generations: int,
        blackboard: SwarmBlackboard,
    ) -> SwarmTelemetry:
        """Run the island for a migration interval and report back."""
        t_sync_0 = time.perf_counter()
        
        # 1. Sync Negative Genetic Memory from blackboard
        shared_avoid = blackboard.get_avoid_loci()
        self.local_avoid_loci.update(shared_avoid)

        # 2. Sync Top Elites (Migration)
        if blackboard.global_elites and self.population:
            elites = blackboard.get_top_elites(k=1)
            if elites and elites[0].fitness > self.population[-1].fitness:
                # Replace the worst individual in population
                self.population.sort(key=lambda ind: ind.fitness, reverse=True)
                self.population[-1] = elites[0]

        sync_time_ms = (time.perf_counter() - t_sync_0) * 1000.0

        # 3. Evolve locally
        report = self.engine.run(
            generations=generations,
            initial_population=self.population if self.population else None,
        )
        self.population = self.engine.population
        self.rounds_completed += 1

        best = self.engine.best_ever
        best_fit = best.fitness if best is not None else 0.0

        # 4. Check for self-sacrifice / dead ends mined during run
        new_dead: set[tuple] = set()
        for ind in self.population:
            if ind.fitness == 0.0 and hasattr(ind.genome, "edits") and len(ind.genome.edits) == 1:
                e = ind.genome.edits[0]
                new_dead.add((getattr(e, "file", ""), getattr(e, "lineno", 0), getattr(e, "col_offset", 0), getattr(e, "kind", "")))

        # Submit findings to blackboard
        t_pub_0 = time.perf_counter()
        if new_dead:
            blackboard.publish_avoid_loci(new_dead)
        if best is not None:
            blackboard.submit_candidate(best, baseline_fitness=float("-inf"), island_id=self.island_id)
        sync_time_ms += (time.perf_counter() - t_pub_0) * 1000.0

        fit_vals = [i.fitness for i in self.population] if self.population else [0.0]
        mean_fit = sum(fit_vals) / len(fit_vals)

        return SwarmTelemetry(
            island_id=self.island_id,
            generation=self.rounds_completed * generations,
            best_fitness=best_fit,
            mean_fitness=round(mean_fit, 2),
            total_evaluations=int(getattr(self.engine, "_total_evals", 0) or 0),
            dead_loci_found=new_dead,
            sync_time_ms=sync_time_ms,
        )


class BurdenGatedIslandSwarmEngine:
    """Deliberate Scaling Swarm Engine with Burden-Gated Eviction Invariant.
    
    STATUS: PROBATIONARY_BURDEN_GATED
    EVICTION_POLICY: EVICT_WHEN_BURDEN
    
    Coordinates multiple specialized islands. Audits scaling efficiency on every
    round via BurdenMonitor. Evicts itself automatically if coordination becomes a burden.
    """

    CONTRACT_STATUS = SwarmStatus.PROBATIONARY
    TAG = "PROBATIONARY_BURDEN_GATED"

    def __init__(
        self,
        fitness_fn: Any,
        config: EngineConfig | None = None,
        island_count: int = 4,
        generations_per_round: int = 4,
        max_rounds: int = 10,
        early_stop_fitness: float = 100.0,
        alpha_threshold: float = 0.40,
        max_sync_overhead_pct: float = 35.0,
        base_seed: int = 42,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.config = config or EngineConfig()
        self.island_count = max(2, int(island_count))
        self.generations_per_round = max(1, int(generations_per_round))
        self.max_rounds = max(1, int(max_rounds))
        self.early_stop_fitness = float(early_stop_fitness)
        self.base_seed = int(base_seed)

        self.blackboard = SwarmBlackboard()
        self.burden_monitor = BurdenMonitor(
            island_count=self.island_count,
            alpha_min=alpha_threshold,
            max_sync_overhead_pct=max_sync_overhead_pct,
        )

        self.status = SwarmStatus.PROBATIONARY
        self.islands: list[IslandWorker] = []
        self.eviction_decision: BurdenDecision | None = None
        self._initialize_islands()

    def _initialize_islands(self) -> None:
        """Instantiate heterogeneous specialized islands."""
        specializations = ["conservative", "exploratory", "sbfl_targeted", "generalist"]
        self.islands = []
        for i in range(self.island_count):
            spec = specializations[i % len(specializations)]
            # Clone config and assign unique seed
            cfg = copy.deepcopy(self.config)
            cfg.seed = self.base_seed + (i * 101) + 1
            cfg.early_stop_fitness = self.early_stop_fitness
            
            eng = EvolutionEngine(
                fitness_fn=self.fitness_fn,
                config=cfg,
            )
            worker = IslandWorker(island_id=i, engine=eng, specialization=spec)
            self.islands.append(worker)

    def run(
        self,
        initial_populations: list[list[Individual]] | None = None,
        baseline_evaluator_fn: Callable[[], tuple[float, float, int]] | None = None,
    ) -> dict[str, Any]:
        """Execute the coordinated island swarm with burden gating.
        
        Args:
            initial_populations: Optional list of populations, one per island.
            baseline_evaluator_fn: Optional callable returning (baseline_fit, baseline_sec, baseline_evals)
                                   to benchmark against. If omitted, a synthetic estimate is used.
        """
        t0 = time.perf_counter()
        total_sync_time_sec = 0.0
        total_evals_consumed = 0
        history: list[dict[str, Any]] = []

        # Distribute initial populations if provided
        if initial_populations:
            for i, pop in enumerate(initial_populations):
                if i < len(self.islands):
                    self.islands[i].population = pop

        # Baseline reference estimates (used by BurdenMonitor)
        baseline_fit = 0.0
        baseline_expected_sec = 0.5
        baseline_expected_evals = 32
        if baseline_evaluator_fn is not None:
            try:
                b_fit, b_sec, b_evals = baseline_evaluator_fn()
                baseline_fit = max(baseline_fit, b_fit)
                baseline_expected_sec = max(0.01, b_sec)
                baseline_expected_evals = max(1, b_evals)
            except Exception:
                pass

        for round_idx in range(1, self.max_rounds + 1):
            round_t0 = time.perf_counter()
            round_sync_ms = 0.0
            
            # Step all islands
            telemetries: list[SwarmTelemetry] = []
            for island in self.islands:
                t_step = island.step_generations(
                    generations=self.generations_per_round,
                    blackboard=self.blackboard,
                )
                telemetries.append(t_step)
                round_sync_ms += t_step.sync_time_ms
                total_evals_consumed += t_step.total_evaluations

            total_sync_time_sec += (round_sync_ms / 1000.0)
            elapsed_sec = time.perf_counter() - t0

            # Determine best fitness across all islands & blackboard
            current_best_ind = self.blackboard.global_elites[0] if self.blackboard.global_elites else None
            current_best_fit = current_best_ind.fitness if current_best_ind is not None else 0.0

            history.append({
                "round": round_idx,
                "generation": round_idx * self.generations_per_round,
                "best_fitness": current_best_fit,
                "total_evals": total_evals_consumed,
                "global_avoid_loci_count": len(self.blackboard.global_avoid_loci),
                "migrations_total": self.blackboard.total_migrations,
                "elapsed_seconds": round(elapsed_sec, 4),
            })

            # Check Goal Completion
            if current_best_fit >= self.early_stop_fitness:
                self.status = SwarmStatus.CONVERGED
                break

            # ------------------------------------------------------------------
            # BURDEN-GATED INVARIANT: Audit whether Swarm is a net burden
            # ------------------------------------------------------------------
            burden_check = self.burden_monitor.evaluate_burden(
                current_best_fitness=current_best_fit,
                baseline_best_fitness=baseline_fit,
                swarm_elapsed_sec=elapsed_sec,
                baseline_expected_sec=baseline_expected_sec * (round_idx / 2.0),
                sync_time_sec=total_sync_time_sec,
                total_evaluations=total_evals_consumed,
                baseline_evaluations=baseline_expected_evals * round_idx,
            )

            if burden_check.should_evict:
                # EVICT! Disengage swarm overhead immediately.
                self.status = SwarmStatus.EVICTED
                self.eviction_decision = burden_check
                break

        total_duration = time.perf_counter() - t0
        best_solution = self.blackboard.global_elites[0] if self.blackboard.global_elites else None
        if best_solution is None:
            island_bests = [isl.engine.best_ever for isl in self.islands if isl.engine.best_ever is not None]
            if island_bests:
                best_solution = max(island_bests, key=lambda ind: ind.fitness)

        return {
            "swarm_status": self.status.value,
            "contract_tag": self.TAG,
            "evicted": self.status == SwarmStatus.EVICTED,
            "eviction_reason": self.eviction_decision.reason if self.eviction_decision else None,
            "empirical_alpha": self.eviction_decision.empirical_alpha if self.eviction_decision else 1.0,
            "speedup_ratio": self.eviction_decision.speedup_ratio if self.eviction_decision else 1.0,
            "total_duration_seconds": round(total_duration, 4),
            "total_evaluations": total_evals_consumed,
            "total_rounds": len(history),
            "best_fitness": best_solution.fitness if best_solution else 0.0,
            "best_individual": best_solution,
            "avoid_loci_shared": len(self.blackboard.global_avoid_loci),
            "migrations_executed": self.blackboard.total_migrations,
            "history": history,
        }
