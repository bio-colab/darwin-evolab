"""
parallel_budget_executor.py — Persistent Worker Pool with Resource Budget Governance.

Provides high-throughput parallel evaluation across isolated worker processes
with active watchdog timeout enforcement, cross-platform two-phase termination
(cancel event -> process tree kill), and automatic worker recycling.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .events import EventBus, EvaluationTimeoutEvent, WorkerRecycledEvent
from .sandbox import kill_process_tree

logger = logging.getLogger("evolab.parallel")


@dataclass(frozen=True)
class BudgetConfig:
    """Immutable budget configuration, built from DomainSpec."""

    max_eval_seconds: float = 30.0
    grace_period_seconds: float = 1.0
    penalty_floor: float = 0.0
    n_workers: int = 0  # 0 = auto: max(1, os.cpu_count() - 1)

    @classmethod
    def from_spec(cls, spec: Any) -> BudgetConfig:
        """Builds configuration from a DomainSpec or dict without inspecting internals."""
        if isinstance(spec, dict):
            return cls(
                max_eval_seconds=float(spec.get("max_eval_seconds", 30.0)),
                grace_period_seconds=float(spec.get("grace_period_seconds", 1.0)),
                penalty_floor=float(spec.get("fitness_floor", spec.get("penalty_floor", 0.0))),
                n_workers=int(spec.get("n_workers", 0)),
            )
        return cls(
            max_eval_seconds=float(getattr(spec, "max_eval_seconds", 30.0)),
            grace_period_seconds=float(getattr(spec, "grace_period_seconds", 1.0)),
            penalty_floor=float(getattr(spec, "fitness_floor", getattr(spec, "penalty_floor", 0.0))),
            n_workers=int(getattr(spec, "n_workers", 0)),
        )


@dataclass(frozen=True)
class BudgetResult:
    """Normalized result of a budgeted evaluation."""

    fitness_score: float
    termination_reason: str = "completed"  # completed | timeout | worker_crash
    wall_clock_seconds: float = 0.0
    partial_progress: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timed_out(self) -> bool:
        return self.termination_reason == "timeout"

    @property
    def score(self) -> float:
        return self.fitness_score


def _worker_loop(
    worker_id: int,
    eval_fn: Callable[[Any], Any],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    cancel_event: mp.Event,
) -> None:
    """
    Persistent worker loop.
    Workers stay alive for the lifetime of the executor, eliminating process-spawn churn.
    """
    while True:
        try:
            task = task_queue.get()
        except (EOFError, OSError):
            break

        if task is None or task[0] == "shutdown":
            break

        _, genome_id, genome = task

        # Atomically notify executor that this worker has actively begun this genome
        t_start = time.perf_counter()
        result_queue.put(("started", worker_id, genome_id, t_start))

        try:
            if cancel_event.is_set():
                result_queue.put(("cancelled", worker_id, genome_id, 0.0))
                continue

            raw_res = eval_fn(genome)
            t_elapsed = time.perf_counter() - t_start

            if cancel_event.is_set():
                result_queue.put(("cancelled", worker_id, genome_id, t_elapsed))
                continue

            # Extract normalized score and metadata
            if hasattr(raw_res, "score"):
                score = float(raw_res.score)
                meta = getattr(raw_res, "sub_scores", {}) or getattr(raw_res, "metadata", {}) or {}
            elif isinstance(raw_res, (int, float)):
                score = float(raw_res)
                meta = {}
            elif isinstance(raw_res, dict):
                score = float(raw_res.get("score", 0.0))
                meta = raw_res
            else:
                score = float(raw_res)
                meta = {}

            result_queue.put(("done", worker_id, genome_id, (score, meta, t_elapsed)))

        except Exception as exc:
            t_elapsed = time.perf_counter() - t_start
            result_queue.put(("error", worker_id, genome_id, (str(exc), t_elapsed)))


class ParallelBudgetExecutor:
    """
    Production-grade persistent worker pool with active watchdog governance.
    
    Guarantees:
    - Zero spawn-churn (processes live across generations).
    - True synchronization (workers announce task start atomically).
    - Zero function pickling overhead over IPC.
    - Two-phase cross-platform process tree termination.
    - Automatic worker recycling and fault isolation.
    """

    def __init__(
        self,
        eval_fn: Callable[[Any], Any],
        budget_config: BudgetConfig | None = None,
        event_bus: EventBus | None = None,
        kill_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.eval_fn = eval_fn
        self.config = budget_config or BudgetConfig()
        self.event_bus = event_bus
        self._kill_fn = kill_fn or kill_process_tree

        # Determine worker count
        configured_workers = self.config.n_workers
        if configured_workers <= 0:
            cpu_n = os.cpu_count() or 2
            self.n_workers = max(1, cpu_n - 1)
        else:
            self.n_workers = configured_workers

        self._task_queue: mp.Queue = mp.Queue()
        self._result_queue: mp.Queue = mp.Queue()

        self._workers: Dict[int, dict] = {}
        self._next_worker_id = 0
        self._pending_futures: Dict[str, Future] = {}
        self._genome_counter = 0

        self._lock = threading.Lock()
        self._running = True

        # Spawn initial persistent workers
        for _ in range(self.n_workers):
            self._spawn_worker()

        # Start background watchdog monitor
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="evolab-watchdog",
        )
        self._watchdog.start()

    def _spawn_worker(self) -> int:
        with self._lock:
            wid = self._next_worker_id
            self._next_worker_id += 1
            cancel_evt = mp.Event()

            proc = mp.Process(
                target=_worker_loop,
                args=(wid, self.eval_fn, self._task_queue, self._result_queue, cancel_evt),
                daemon=True,
                name=f"evolab-worker-{wid}",
            )
            proc.start()

            self._workers[wid] = {
                "process": proc,
                "cancel_event": cancel_evt,
                "active_genome_id": None,
                "start_time": 0.0,
            }
            logger.debug("Spawned persistent worker %d (PID %s)", wid, proc.pid)
            return wid

    def submit(self, genome: Any) -> Future:
        """Submits a single genome to the worker pool. Returns a concurrent.futures.Future."""
        with self._lock:
            self._genome_counter += 1
            gid = f"g-{self._genome_counter}"
            fut: Future = Future()
            self._pending_futures[gid] = fut

        self._task_queue.put(("eval", gid, genome))
        return fut

    def submit_batch(self, genomes: List[Any]) -> List[Future]:
        """Submits a batch of genomes. Returns an ordered list of Futures."""
        return [self.submit(g) for g in genomes]

    def map(self, genomes: List[Any], timeout: float | None = None) -> List[BudgetResult]:
        """Convenience evaluation: submits a batch and blocks until all results are gathered."""
        futures = self.submit_batch(genomes)
        effective_timeout = timeout if timeout is not None else (self.config.max_eval_seconds * 2 + 5.0)
        return [f.result(timeout=effective_timeout) for f in futures]

    def _watchdog_loop(self) -> None:
        """Active watchdog: drains results and enforces per-task timeouts."""
        while self._running:
            # 1. Drain results queue
            while not self._result_queue.empty():
                try:
                    msg = self._result_queue.get_nowait()
                except Exception:
                    break

                msg_type = msg[0]
                if msg_type == "started":
                    _, wid, gid, t_start = msg
                    with self._lock:
                        if wid in self._workers:
                            self._workers[wid]["active_genome_id"] = gid
                            self._workers[wid]["start_time"] = t_start

                elif msg_type == "done":
                    _, wid, gid, (score, meta, elapsed) = msg
                    self._resolve_future(
                        gid,
                        wid,
                        BudgetResult(
                            fitness_score=score,
                            termination_reason="completed",
                            wall_clock_seconds=elapsed,
                            metadata=meta,
                        ),
                    )

                elif msg_type == "error":
                    _, wid, gid, (err_msg, elapsed) = msg
                    self._resolve_future(
                        gid,
                        wid,
                        BudgetResult(
                            fitness_score=self.config.penalty_floor,
                            termination_reason="worker_crash",
                            wall_clock_seconds=elapsed,
                            metadata={"error": err_msg},
                        ),
                    )

                elif msg_type == "cancelled":
                    _, wid, gid, elapsed = msg
                    self._resolve_future(
                        gid,
                        wid,
                        BudgetResult(
                            fitness_score=self.config.penalty_floor,
                            termination_reason="cancelled",
                            wall_clock_seconds=elapsed,
                        ),
                    )

            # 2. Check for active task timeouts
            now = time.perf_counter()
            with self._lock:
                workers_snapshot = list(self._workers.items())

            for wid, info in workers_snapshot:
                gid = info["active_genome_id"]
                t_start = info["start_time"]
                if gid and t_start > 0:
                    elapsed = now - t_start
                    if elapsed > self.config.max_eval_seconds:
                        self._handle_timeout(wid, gid, elapsed)

            time.sleep(0.02)

    def _resolve_future(self, gid: str, wid: int, result: BudgetResult) -> None:
        with self._lock:
            fut = self._pending_futures.pop(gid, None)
            if fut and not fut.done():
                fut.set_result(result)
            if wid in self._workers:
                self._workers[wid]["active_genome_id"] = None
                self._workers[wid]["start_time"] = 0.0

    def _handle_timeout(self, wid: int, gid: str, elapsed: float) -> None:
        logger.warning(
            "Worker %d exceeded budget on task %s (%.2fs > %.2fs). Initiating two-phase kill...",
            wid,
            gid,
            elapsed,
            self.config.max_eval_seconds,
        )

        with self._lock:
            info = self._workers.pop(wid, None)

        if not info:
            return

        # Phase 1: Graceful cancellation event
        info["cancel_event"].set()
        proc: mp.Process = info["process"]
        proc.join(timeout=self.config.grace_period_seconds)

        # Phase 2: Forced termination
        if proc.is_alive() and proc.pid:
            try:
                self._kill_fn(proc.pid)
            except Exception as exc:
                logger.debug("Error during process kill: %s", exc)
            proc.join(timeout=0.5)

        # Spawn replacement worker to maintain pool capacity
        new_wid = self._spawn_worker()

        # Emit observability events
        if self.event_bus:
            self.event_bus.publish(
                EvaluationTimeoutEvent(
                    genome_id=gid,
                    worker_id=wid,
                    timeout_seconds=self.config.max_eval_seconds,
                    elapsed_seconds=elapsed,
                    termination_phase="phase2_force_kill" if proc.is_alive() else "phase1_cancel",
                    replacement_worker_id=new_wid,
                )
            )
            self.event_bus.publish(
                WorkerRecycledEvent(
                    old_worker_id=wid,
                    new_worker_id=new_wid,
                    reason="timeout",
                )
            )

        # Resolve the pending future with penalty
        with self._lock:
            fut = self._pending_futures.pop(gid, None)
            if fut and not fut.done():
                fut.set_result(
                    BudgetResult(
                        fitness_score=self.config.penalty_floor,
                        termination_reason="timeout",
                        wall_clock_seconds=elapsed,
                        metadata={
                            "timeout_triggered": True,
                            "budget_seconds": self.config.max_eval_seconds,
                        },
                    )
                )

    def shutdown(self, wait: bool = True) -> None:
        """Shuts down all workers and the watchdog thread."""
        self._running = False
        with self._lock:
            worker_count = len(self._workers)

        for _ in range(worker_count * 2):
            try:
                self._task_queue.put(("shutdown", None, None))
            except Exception:
                pass

        if wait:
            with self._lock:
                workers = list(self._workers.values())
            for info in workers:
                proc: mp.Process = info["process"]
                if proc.is_alive():
                    proc.join(timeout=0.5)
                    if proc.is_alive() and proc.pid:
                        try:
                            self._kill_fn(proc.pid)
                        except Exception:
                            pass

    def __enter__(self) -> ParallelBudgetExecutor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown(wait=True)
