"""Unit tests for ParallelBudgetExecutor with Budget Governance."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import pytest

from evolab.events import EventBus, EvaluationTimeoutEvent, WorkerRecycledEvent
from evolab.evaluators import FitnessResult
from evolab.parallel_budget_executor import BudgetConfig, BudgetResult, ParallelBudgetExecutor


def _eval_fast(genome: Any) -> float:
    if isinstance(genome, (list, tuple)):
        return float(sum(genome))
    return 42.0


def _eval_fitness_result(genome: Any) -> FitnessResult:
    val = float(sum(genome)) if isinstance(genome, (list, tuple)) else 50.0
    return FitnessResult(
        score=val,
        sub_scores={"sum": val, "half": val / 2.0},
        evaluation_time_ms=1.5,
    )


def _eval_crash(genome: Any) -> float:
    raise RuntimeError("Intended test evaluator explosion")


def _eval_pid(genome: Any) -> float:
    return float(os.getpid())


def _eval_conditional(genome: Any) -> float:
    if isinstance(genome, dict) and genome.get("hang"):
        time.sleep(100.0)
        return 0.0
    return 88.0


def test_budget_config_construction():
    # Defaults
    cfg = BudgetConfig()
    assert cfg.max_eval_seconds == 30.0
    assert cfg.grace_period_seconds == 1.0
    assert cfg.penalty_floor == 0.0
    assert cfg.n_workers == 0

    # From dict
    d = {"max_eval_seconds": 12.5, "grace_period_seconds": 0.5, "penalty_floor": -5.0, "n_workers": 3}
    cfg_d = BudgetConfig.from_spec(d)
    assert cfg_d.max_eval_seconds == 12.5
    assert cfg_d.grace_period_seconds == 0.5
    assert cfg_d.penalty_floor == -5.0
    assert cfg_d.n_workers == 3

    # From object with attributes
    @dataclass
    class DummySpec:
        max_eval_seconds: float = 8.0
        grace_period_seconds: float = 0.2
        fitness_floor: float = -10.0
        n_workers: int = 2

    cfg_spec = BudgetConfig.from_spec(DummySpec())
    assert cfg_spec.max_eval_seconds == 8.0
    assert cfg_spec.grace_period_seconds == 0.2
    assert cfg_spec.penalty_floor == -10.0
    assert cfg_spec.n_workers == 2


def test_batch_fast_evaluations():
    cfg = BudgetConfig(max_eval_seconds=5.0, n_workers=2)
    with ParallelBudgetExecutor(eval_fn=_eval_fast, budget_config=cfg) as executor:
        items = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        results = executor.map(items, timeout=10.0)

        assert len(results) == 4
        expected_scores = [3.0, 7.0, 11.0, 15.0]
        for res, expected in zip(results, expected_scores):
            assert res.termination_reason == "completed"
            assert res.fitness_score == expected
            assert res.timed_out is False
            assert res.wall_clock_seconds >= 0.0


def test_fitness_result_unwrapping():
    cfg = BudgetConfig(max_eval_seconds=5.0, n_workers=2)
    with ParallelBudgetExecutor(eval_fn=_eval_fitness_result, budget_config=cfg) as executor:
        items = [[10.0, 20.0]]
        results = executor.map(items, timeout=10.0)
        assert len(results) == 1
        res = results[0]
        assert res.fitness_score == 30.0
        assert res.metadata.get("sum") == 30.0
        assert res.metadata.get("half") == 15.0


def test_zero_spawn_churn():
    cfg = BudgetConfig(max_eval_seconds=5.0, n_workers=2)
    with ParallelBudgetExecutor(eval_fn=_eval_pid, budget_config=cfg) as executor:
        initial_pids = {info["process"].pid for info in executor._workers.values()}

        batch1 = executor.map([1, 2, 3, 4], timeout=10.0)
        pids1 = set(int(r.fitness_score) for r in batch1)

        batch2 = executor.map([5, 6, 7, 8], timeout=10.0)
        pids2 = set(int(r.fitness_score) for r in batch2)

        # Worker PIDs in pool must remain strictly identical across multiple batches (zero churn)
        current_pids = {info["process"].pid for info in executor._workers.values()}
        assert initial_pids == current_pids
        assert pids1.issubset(initial_pids)
        assert pids2.issubset(initial_pids)


def test_evaluator_crash_isolation():
    cfg = BudgetConfig(max_eval_seconds=5.0, penalty_floor=-100.0, n_workers=2)
    with ParallelBudgetExecutor(eval_fn=_eval_crash, budget_config=cfg) as executor:
        results = executor.map([1, 2], timeout=10.0)
        assert len(results) == 2
        for res in results:
            assert res.termination_reason == "worker_crash"
            assert res.fitness_score == -100.0
            assert "Intended test evaluator explosion" in res.metadata.get("error", "")


def test_timeout_watchdog_recycles_and_emits_events():
    events_caught: list[Any] = []
    bus = EventBus()
    bus.subscribe(EvaluationTimeoutEvent, lambda e: events_caught.append(e))
    bus.subscribe(WorkerRecycledEvent, lambda e: events_caught.append(e))

    cfg = BudgetConfig(
        max_eval_seconds=0.25,
        grace_period_seconds=0.1,
        penalty_floor=-999.0,
        n_workers=1,
    )

    with ParallelBudgetExecutor(
        eval_fn=_eval_conditional,
        budget_config=cfg,
        event_bus=bus,
    ) as executor:
        # Task 1: Hangs -> watchdog kills & recycles worker
        hang_fut = executor.submit({"hang": True})
        res_hang = hang_fut.result(timeout=5.0)

        assert res_hang.termination_reason == "timeout"
        assert res_hang.timed_out is True
        assert res_hang.fitness_score == -999.0
        assert res_hang.metadata.get("timeout_triggered") is True

        # Verify events were published
        timeout_events = [e for e in events_caught if isinstance(e, EvaluationTimeoutEvent)]
        recycle_events = [e for e in events_caught if isinstance(e, WorkerRecycledEvent)]
        assert len(timeout_events) == 1
        assert len(recycle_events) == 1
        assert timeout_events[0].timeout_seconds == 0.25
        assert recycle_events[0].reason == "timeout"

        # Task 2: Sent to the replacement worker -> should complete normally
        normal_fut = executor.submit({"hang": False})
        res_normal = normal_fut.result(timeout=5.0)

        assert res_normal.termination_reason == "completed"
        assert res_normal.fitness_score == 88.0
        assert res_normal.timed_out is False
