"""Tests for Event-Driven Observability and Lifecycle Telemetry in evolab."""
from __future__ import annotations

import pytest
from evolab.events import (
    EventBus,
    EvolutionEvent,
    GenerationEvaluatedEvent,
    RunCompletedEvent,
    MutationRejectedEvent,
)
from evolab.engine import EvolutionEngine
from evolab.config import EngineConfig


def test_event_bus_subscribe_and_publish():
    bus = EventBus()
    events_received = []

    def on_gen(event: GenerationEvaluatedEvent):
        events_received.append(event)

    bus.subscribe(GenerationEvaluatedEvent, on_gen)

    # Publish an event
    ev1 = GenerationEvaluatedEvent(generation=1, best_fitness=42.0)
    bus.publish(ev1)
    assert len(events_received) == 1
    assert events_received[0].generation == 1
    assert events_received[0].best_fitness == 42.0

    # Unsubscribe
    bus.unsubscribe(GenerationEvaluatedEvent, on_gen)
    bus.publish(GenerationEvaluatedEvent(generation=2, best_fitness=50.0))
    assert len(events_received) == 1  # No new event received


def test_event_bus_fault_tolerance():
    """Verifies that an exception in an observer does not crash the dispatcher."""
    bus = EventBus()
    normal_received = []

    def broken_listener(event):
        raise RuntimeError("Crash in dashboard")

    def safe_listener(event):
        normal_received.append(event)

    bus.subscribe(RunCompletedEvent, broken_listener)
    bus.subscribe(RunCompletedEvent, safe_listener)

    # Should not raise
    bus.publish(RunCompletedEvent(total_generations=5, best_fitness=99.0))
    assert len(normal_received) == 1


def test_engine_emits_generation_and_run_events():
    """Verifies that EvolutionEngine automatically dispatches lifecycle events during run()."""
    gen_events = []
    run_events = []

    engine = EvolutionEngine(
        config=EngineConfig(generations=4, population_size=6, genome_size=2, seed=42)
    )

    engine.add_event_listener(GenerationEvaluatedEvent, lambda ev: gen_events.append(ev))
    engine.add_event_listener(RunCompletedEvent, lambda ev: run_events.append(ev))

    # Run for 3 generations
    engine.run(generations=3)

    # Check generation events
    assert len(gen_events) >= 3
    for ev in gen_events:
        assert isinstance(ev, GenerationEvaluatedEvent)
        assert ev.generation >= 1
        assert ev.best_fitness >= 0.0

    # Check run completed event
    assert len(run_events) == 1
    assert isinstance(run_events[0], RunCompletedEvent)
    assert run_events[0].total_generations >= 3
    assert run_events[0].best_fitness >= 0.0
