"""terminal.py — Zero-dependency live terminal progress bars, telemetry rendering, and ANSI formatting.

Features:
- Pure Python standard library implementation with zero external dependencies.
- Beautiful ANSI progress bars with generation counters, fitness metrics, diversity, and throughput.
- Automatic TTY detection: smooth in-place rewrites on interactive terminals, clean periodic logs when piped or in CI.
- Optional dynamic upgrade if `rich` is available in the user environment.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, TextIO

from ..events import GenerationEvaluatedEvent, RunCompletedEvent


# ANSI Color and Styling constants
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"


def supports_color(stream: TextIO) -> bool:
    """Detect whether the given stream supports ANSI color sequences."""
    if os.environ.get("NO_COLOR") or os.environ.get("EVOLAB_NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return getattr(stream, "isatty", lambda: False)()


class TerminalProgressObserver:
    """Listens to EvolutionEngine event_bus to render live terminal progress updates."""

    def __init__(
        self,
        total_generations: int,
        quiet: bool = False,
        stream: TextIO | None = None,
        bar_width: int = 24,
    ) -> None:
        self.total_generations = max(1, total_generations)
        self.quiet = quiet
        self.stream = stream or sys.stderr
        self.bar_width = bar_width
        self.is_tty = getattr(self.stream, "isatty", lambda: False)()
        self.use_color = supports_color(self.stream)
        self.start_time = time.perf_counter()
        self.last_update_time = self.start_time
        self._completed = False
        self._last_line_len = 0

    def on_generation_evaluated(self, event: GenerationEvaluatedEvent) -> None:
        """Handle per-generation evaluated telemetry event."""
        if self.quiet:
            return

        now = time.perf_counter()
        elapsed = max(1e-5, now - self.start_time)
        gen = event.generation
        total = self.total_generations
        ratio = min(1.0, gen / total)
        rate = gen / elapsed

        filled = int(self.bar_width * ratio)
        empty = self.bar_width - filled
        bar = "█" * filled + "░" * empty

        pct = ratio * 100.0
        best_str = f"{event.best_fitness:6.2f}"
        mean_str = f"{event.mean_fitness:6.2f}"
        div_str = f"{event.diversity:4.2f}"
        sp_count = event.active_species_count

        if self.use_color:
            prefix = f"{_BOLD}{_CYAN}[evolab]{_RESET}"
            bar_styled = f"{_GREEN}{bar[:filled]}{_RESET}{_DIM}{bar[filled:]}{_RESET}"
            metrics = (
                f"{_BOLD}Best:{_RESET} {_YELLOW}{best_str}{_RESET} | "
                f"{_DIM}Mean:{_RESET} {mean_str} | "
                f"{_DIM}Div:{_RESET} {div_str} | "
                f"{_DIM}Sp:{_RESET} {sp_count}"
            )
            speed = f"{_DIM}({rate:4.1f} gen/s){_RESET}"
        else:
            prefix = "[evolab]"
            bar_styled = bar
            metrics = f"Best: {best_str} | Mean: {mean_str} | Div: {div_str} | Sp: {sp_count}"
            speed = f"({rate:4.1f} gen/s)"

        line = f"{prefix} Gen {gen:3d}/{total:3d} [{bar_styled}] {pct:5.1f}% | {metrics} {speed}"

        if self.is_tty:
            # Overwrite line with carriage return
            padded = line
            if len(line) < self._last_line_len:
                padded += " " * (self._last_line_len - len(line))
            self._last_line_len = len(line)
            self.stream.write("\r" + padded)
            self.stream.flush()
        else:
            # Non-interactive / CI / piped: print periodically (every 5 gens or last)
            if gen == 1 or gen % 5 == 0 or gen == total:
                self.stream.write(line + "\n")
                self.stream.flush()

    def on_run_completed(self, event: RunCompletedEvent | None = None) -> None:
        """Called when evolution finishes to finalize terminal output."""
        if self.quiet or self._completed:
            return
        self._completed = True
        elapsed = time.perf_counter() - self.start_time
        if self.is_tty:
            self.stream.write("\n")
            if self.use_color:
                summary = (
                    f"{_BOLD}{_GREEN}✔ Run Complete{_RESET} in {_BOLD}{elapsed:.2f}s{_RESET}"
                )
            else:
                summary = f"[evolab] Run Complete in {elapsed:.2f}s"
            self.stream.write(summary + "\n")
            self.stream.flush()

    def attach_to_engine(self, engine: Any) -> None:
        """Convenience helper to attach this observer to an EvolutionEngine instance."""
        if hasattr(engine, "event_bus"):
            engine.event_bus.subscribe(GenerationEvaluatedEvent, self.on_generation_evaluated)
            engine.event_bus.subscribe(RunCompletedEvent, self.on_run_completed)


class StepProgressObserver:
    """Observer for discrete evaluation steps (such as forward-greedy code repair)."""

    def __init__(self, quiet: bool = False, stream: TextIO | None = None) -> None:
        self.quiet = quiet
        self.stream = stream or sys.stderr
        self.is_tty = getattr(self.stream, "isatty", lambda: False)()
        self.use_color = supports_color(self.stream)
        self.start_time = time.perf_counter()
        self._last_line_len = 0
        self._step_spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0

    def on_step(self, step: int, score: float, evaluations: int, candidate_name: str = "") -> None:
        """Report evaluation step update."""
        if self.quiet:
            return

        spin = self._step_spinner[self._spinner_idx % len(self._step_spinner)]
        self._spinner_idx += 1
        elapsed = max(1e-5, time.perf_counter() - self.start_time)
        rate = evaluations / elapsed

        if self.use_color:
            prefix = f"{_BOLD}{_CYAN}[evolab:apr]{_RESET}"
            spin_styled = f"{_YELLOW}{spin}{_RESET}"
            score_styled = f"{_GREEN}{score:6.2f}{_RESET}"
            detail = f"{_DIM}Evals:{_RESET} {evaluations:4d} | {_DIM}Score:{_RESET} {score_styled} | {_DIM}Speed:{_RESET} {rate:4.1f} ev/s"
        else:
            prefix = "[evolab:apr]"
            spin_styled = spin
            detail = f"Evals: {evaluations:4d} | Score: {score:6.2f} | Speed: {rate:4.1f} ev/s"

        cand = f" ({candidate_name})" if candidate_name else ""
        line = f"{prefix} {spin_styled} Step {step:3d} | {detail}{cand}"

        if self.is_tty:
            padded = line
            if len(line) < self._last_line_len:
                padded += " " * (self._last_line_len - len(line))
            self._last_line_len = len(line)
            self.stream.write("\r" + padded)
            self.stream.flush()
        else:
            if step == 1 or step % 10 == 0:
                self.stream.write(line + "\n")
                self.stream.flush()

    def complete(self, best_score: float, total_evals: int) -> None:
        """Mark search as complete."""
        if self.quiet:
            return
        elapsed = time.perf_counter() - self.start_time
        if self.is_tty:
            self.stream.write("\n")
        if self.use_color:
            summary = (
                f"{_BOLD}{_GREEN}✔ APR Search Finished:{_RESET} "
                f"Best Score = {_BOLD}{best_score:6.2f}{_RESET} | "
                f"Evaluations = {total_evals} | Elapsed = {elapsed:.2f}s"
            )
        else:
            summary = (
                f"[evolab:apr] Search Finished: Best Score = {best_score:6.2f} | "
                f"Evaluations = {total_evals} | Elapsed = {elapsed:.2f}s"
            )
        self.stream.write(summary + "\n")
        self.stream.flush()
