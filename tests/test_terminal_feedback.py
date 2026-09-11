"""Unit tests for terminal progress observers and live telemetry rendering."""
from __future__ import annotations

import io
from evolab.events import GenerationEvaluatedEvent, RunCompletedEvent
from evolab.ui.terminal import StepProgressObserver, TerminalProgressObserver


def test_terminal_progress_observer_tty():
    stream = io.StringIO()
    # Mock stream to appear as TTY
    stream.isatty = lambda: True

    observer = TerminalProgressObserver(total_generations=10, quiet=False, stream=stream)
    event = GenerationEvaluatedEvent(
        generation=3,
        best_fitness=85.5,
        mean_fitness=72.0,
        diversity=0.45,
        active_species_count=2,
        duration_ms=12.5,
    )
    observer.on_generation_evaluated(event)
    output = stream.getvalue()
    assert "Gen   3/ 10" in output
    assert "Best:" in output or "85.50" in output

    observer.on_run_completed(RunCompletedEvent(total_generations=10, best_fitness=85.5))
    output_final = stream.getvalue()
    assert "Run Complete" in output_final


def test_terminal_progress_observer_non_tty():
    stream = io.StringIO()
    stream.isatty = lambda: False

    observer = TerminalProgressObserver(total_generations=5, quiet=False, stream=stream)
    event = GenerationEvaluatedEvent(generation=1, best_fitness=50.0, mean_fitness=40.0)
    observer.on_generation_evaluated(event)
    assert "Gen   1/  5" in stream.getvalue()


def test_terminal_progress_observer_quiet():
    stream = io.StringIO()
    observer = TerminalProgressObserver(total_generations=10, quiet=True, stream=stream)
    event = GenerationEvaluatedEvent(generation=1, best_fitness=50.0)
    observer.on_generation_evaluated(event)
    observer.on_run_completed()
    assert stream.getvalue() == ""


def test_step_progress_observer():
    stream = io.StringIO()
    stream.isatty = lambda: True
    observer = StepProgressObserver(quiet=False, stream=stream)
    observer.on_step(step=2, score=75.0, evaluations=10, candidate_name="replace_op")
    output = stream.getvalue()
    assert "Step   2" in output
    assert "75.00" in output

    observer.complete(best_score=100.0, total_evals=15)
    output_final = stream.getvalue()
    assert "APR Search Finished" in output_final
    assert "100.00" in output_final
