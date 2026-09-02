"""
events.py — Lightweight Event-Driven Observability and Telemetry Bus for darwin-evolab.
Decouples evolutionary engine cycles from dashboards, metrics aggregators, and distributed listeners.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvolutionEvent:
    """Base event emitted during evolutionary lifecycle."""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GenerationEvaluatedEvent(EvolutionEvent):
    """Emitted after all individuals in a generation have been evaluated."""
    generation: int = 0
    best_fitness: float = 0.0
    mean_fitness: float = 0.0
    diversity: float = 0.0
    active_species_count: int = 0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class IndividualEvaluatedEvent(EvolutionEvent):
    """Emitted after evaluating a single individual."""
    species: str = ""
    fitness: float = 0.0
    success: bool = True
    fault_category: str = "normal_success"


@dataclass(frozen=True)
class MutationRejectedEvent(EvolutionEvent):
    """Emitted when a proposed mutation fails semantic validation or triggers security oracles."""
    reason: str = ""
    details: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunCompletedEvent(EvolutionEvent):
    """Emitted upon completion of an evolutionary run."""
    total_generations: int = 0
    best_fitness: float = 0.0
    best_species: str = ""
    early_stopped: bool = False
    total_time_seconds: float = 0.0


class EventBus:
    """Lightweight in-process event dispatcher supporting decoupled observers and telemetry."""

    def __init__(self):
        self._listeners: dict[type[EvolutionEvent], list[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: type[EvolutionEvent], listener: Callable[[Any], None]) -> None:
        """Registers a listener callback for a specific event type."""
        self._listeners.setdefault(event_type, []).append(listener)

    def unsubscribe(self, event_type: type[EvolutionEvent], listener: Callable[[Any], None]) -> None:
        """Removes a registered listener callback."""
        if event_type in self._listeners:
            self._listeners[event_type] = [fn for fn in self._listeners[event_type] if fn != listener]

    def publish(self, event: EvolutionEvent) -> None:
        """Dispatches an event to all registered listeners."""
        for event_cls, listeners in self._listeners.items():
            if isinstance(event, event_cls):
                for listener in listeners:
                    try:
                        listener(event)
                    except Exception:
                        pass
