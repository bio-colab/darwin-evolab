"""Unit tests for Cooperative Signal Controller and Graceful Interruption."""
from __future__ import annotations

import time
import pytest
from evolab.signals import EvolutionSignal, SignalController
from evolab.engine import EvolutionEngine
from evolab.config import EngineConfig


def test_signal_controller_pause_resume():
    sc = SignalController(register_os_signals=False)
    assert not sc.is_paused
    assert not sc.stop_requested

    sc.pause()
    assert sc.is_paused

    sc.resume()
    assert not sc.is_paused


def test_signal_controller_request_stop():
    sc = SignalController(register_os_signals=False)
    sc.pause()
    assert sc.is_paused

    sc.request_stop(reason="test_termination")
    assert sc.stop_requested
    assert sc.stop_reason == "test_termination"
    # request_stop unpauses to let loops exit
    assert not sc.is_paused


def test_signal_controller_checkpoint_request():
    sc = SignalController(register_os_signals=False)
    assert not sc.checkpoint_requested
    assert not sc.consume_checkpoint_request()

    sc.request_checkpoint()
    assert sc.checkpoint_requested
    assert sc.consume_checkpoint_request()
    assert not sc.checkpoint_requested


def test_signal_controller_callbacks():
    sc = SignalController(register_os_signals=False)
    received = []

    sc.on_signal(EvolutionSignal.PAUSE, lambda: received.append("paused"))
    sc.on_signal(EvolutionSignal.RESUME, lambda: received.append("resumed"))
    sc.on_signal(EvolutionSignal.CHECKPOINT, lambda: received.append("checkpoint"))

    sc.pause()
    sc.resume()
    sc.request_checkpoint()

    assert received == ["paused", "resumed", "checkpoint"]


def test_engine_graceful_stop_via_signal():
    """Verifies that an engine run can be cooperatively interrupted mid-run without state loss."""
    cfg = EngineConfig(population_size=10, generations=50, seed=42)
    engine = EvolutionEngine(config=cfg)
    sc = SignalController(register_os_signals=False)

    # Attach signal controller to engine and schedule a stop after 3 generations
    gen_counter = 0

    def on_gen(ev):
        nonlocal gen_counter
        gen_counter += 1
        if gen_counter >= 3:
            sc.request_stop("reached_3_generations")

    from evolab.events import GenerationEvaluatedEvent
    engine.add_event_listener(GenerationEvaluatedEvent, on_gen)

    report = engine.run(generations=20, signal_controller=sc)
    assert report["early_stop_triggered"] is True
    assert report["stopped_by_signal"] is True
    assert report["total_generations"] <= 4
    assert engine.best_ever is not None
    assert engine.best_ever.fitness >= 0.0


def test_engine_programmatic_pause_resume():
    """Verifies engine helper methods for pause and resume."""
    cfg = EngineConfig(population_size=8, generations=10, seed=123)
    engine = EvolutionEngine(config=cfg)
    sc = engine.attach_signal_controller()

    assert not sc.is_paused
    engine.pause()
    assert sc.is_paused
    engine.resume()
    assert not sc.is_paused
    engine.request_stop("user_requested")
    assert sc.stop_requested
