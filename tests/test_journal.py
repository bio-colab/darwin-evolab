"""Unit tests for Append-Only Structured Event Journal and Query Engine."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from evolab.events import EventBus, GenerationEvaluatedEvent, RunCompletedEvent
from evolab.journal import StructuredJournal, JournalRecord
from evolab.config import EngineConfig
from evolab.engine import EvolutionEngine


def test_structured_journal_recording_and_query(tmp_path: Path):
    jsonl_file = tmp_path / "test_events.jsonl"
    journal = StructuredJournal(log_file=jsonl_file)
    bus = EventBus()
    journal.attach_to_bus(bus)

    # Publish lifecycle events
    bus.publish(GenerationEvaluatedEvent(generation=1, best_fitness=50.0))
    bus.publish(GenerationEvaluatedEvent(generation=2, best_fitness=75.0))
    bus.publish(GenerationEvaluatedEvent(generation=3, best_fitness=90.0))
    bus.publish(RunCompletedEvent(total_generations=3, best_fitness=90.0))

    assert len(journal) == 4

    # Query by event_type
    gen_events = journal.query(event_type="GenerationEvaluatedEvent")
    assert len(gen_events) == 3

    run_events = journal.query(event_type="RunCompletedEvent")
    assert len(run_events) == 1
    assert run_events[0].payload["best_fitness"] == 90.0

    # Query by generation range
    mid_events = journal.query(since_generation=2, until_generation=2)
    assert len(mid_events) == 1
    assert mid_events[0].generation == 2
    assert mid_events[0].payload["best_fitness"] == 75.0

    # Query with limit
    limited = journal.query(event_type="GenerationEvaluatedEvent", limit=2)
    assert len(limited) == 2

    # Counts by type
    counts = journal.count_by_type()
    assert counts["GenerationEvaluatedEvent"] == 3
    assert counts["RunCompletedEvent"] == 1

    # Verify JSONL file on disk
    assert jsonl_file.is_file()
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    first_record = json.loads(lines[0])
    assert first_record["event_type"] == "GenerationEvaluatedEvent"
    assert first_record["generation"] == 1


def test_engine_has_integrated_journal():
    """Verifies that EvolutionEngine automatically maintains a queryable StructuredJournal."""
    cfg = EngineConfig(population_size=6, generations=4, seed=42)
    engine = EvolutionEngine(config=cfg)

    assert hasattr(engine, "journal")
    assert isinstance(engine.journal, StructuredJournal)

    engine.run(generations=3)

    records = engine.journal.query(event_type="GenerationEvaluatedEvent")
    assert len(records) == 3
    assert records[-1].generation == 3
