"""
journal.py — Append-Only Structured Event Journal and Query Engine for darwin-evolab.

Provides a unified, queryable event journal across the evolutionary lifecycle,
subscribing to EventBus events and maintaining in-memory or JSONL audit trails.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .events import (
    EventBus,
    EvolutionEvent,
    GenerationEvaluatedEvent,
    IndividualEvaluatedEvent,
    MutationRejectedEvent,
    RunCompletedEvent,
)

logger = logging.getLogger("evolab.journal")


@dataclass(frozen=True)
class JournalRecord:
    """Immutable single record within the structured journal."""
    timestamp: float
    event_type: str
    generation: int | None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "generation": self.generation,
            "payload": self.payload,
        }


class StructuredJournal:
    """
    Append-only journal that captures, formats, and queries evolutionary lifecycle events.
    """

    def __init__(self, log_file: str | Path | None = None, max_in_memory: int = 10000):
        self._records: list[JournalRecord] = []
        self._max_in_memory = max_in_memory
        self._log_file: Path | None = Path(log_file).resolve() if log_file else None
        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: EvolutionEvent) -> JournalRecord:
        """Transforms an EvolutionEvent into a JournalRecord and appends it."""
        event_type = event.__class__.__name__
        generation = getattr(event, "generation", None)
        timestamp = getattr(event, "timestamp", time.time())

        # Extract payload dict
        raw_dict = asdict(event) if hasattr(event, "__dataclass_fields__") else dict(event.__dict__)
        raw_dict.pop("timestamp", None)
        raw_dict.pop("generation", None)

        record = JournalRecord(
            timestamp=timestamp,
            event_type=event_type,
            generation=generation,
            payload=raw_dict,
        )

        self._records.append(record)
        if len(self._records) > self._max_in_memory:
            self._records.pop(0)

        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")
            except Exception as e:
                logger.debug("Failed to write journal record to %s: %s", self._log_file, e)

        return record

    def attach_to_bus(self, bus: EventBus) -> None:
        """Subscribes this journal to all EvolutionEvent instances on the EventBus."""
        bus.subscribe(EvolutionEvent, self.append)

    def query(
        self,
        event_type: str | None = None,
        since_generation: int | None = None,
        until_generation: int | None = None,
        limit: int | None = None,
    ) -> list[JournalRecord]:
        """
        Queries recorded events matching criteria.
        """
        results: list[JournalRecord] = []
        for rec in self._records:
            if event_type and rec.event_type != event_type:
                continue
            if since_generation is not None:
                if rec.generation is None or rec.generation < since_generation:
                    continue
            if until_generation is not None:
                if rec.generation is None or rec.generation > until_generation:
                    continue
            results.append(rec)
            if limit is not None and len(results) >= limit:
                break
        return results

    def count_by_type(self) -> dict[str, int]:
        """Returns aggregated event counts by type."""
        counts: dict[str, int] = {}
        for rec in self._records:
            counts[rec.event_type] = counts.get(rec.event_type, 0) + 1
        return counts

    def clear(self) -> None:
        """Clears in-memory journal records."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)
