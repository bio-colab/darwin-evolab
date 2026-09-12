"""
signals.py — Cooperative Signal Controller and Graceful Interruption for darwin-evolab.

Provides a cross-platform signal abstraction supporting graceful shutdown (SIGINT/SIGTERM),
cooperative pause/resume (SIGSTOP/SIGCONT semantics), and checkpoint snapshots (SIGUSR1 semantics).
"""
from __future__ import annotations

import enum
import logging
import signal
import sys
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("evolab.signals")


class EvolutionSignal(str, enum.Enum):
    """Evolutionary signal types."""
    INTERRUPT = "SIGINT"
    TERMINATE = "SIGTERM"
    PAUSE = "SIGSTOP"
    RESUME = "SIGCONT"
    CHECKPOINT = "SIGUSR1"


class SignalController:
    """
    Cooperative signal controller for evolutionary runs.
    
    Allows long-running evolutionary cycles to intercept termination signals gracefully,
    pause and resume execution cooperatively, and dump state checkpoints on demand.
    """

    def __init__(self, register_os_signals: bool = False):
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._checkpoint_event = threading.Event()
        self._stop_reason: str = ""
        self._registered = False
        self._previous_handlers: dict[int, Any] = {}
        self._callbacks: dict[EvolutionSignal, list[Callable[[], None]]] = {
            sig: [] for sig in EvolutionSignal
        }

        if register_os_signals:
            self.register_handlers()

    @property
    def stop_requested(self) -> bool:
        """True if an interrupt or termination signal has been requested."""
        return self._stop_event.is_set()

    @property
    def stop_reason(self) -> str:
        """Description of why the stop was requested."""
        return self._stop_reason

    @property
    def is_paused(self) -> bool:
        """True if the evolutionary engine is cooperatively paused."""
        return self._pause_event.is_set()

    @property
    def checkpoint_requested(self) -> bool:
        """True if an asynchronous state checkpoint dump has been requested."""
        return self._checkpoint_event.is_set()

    def request_stop(self, reason: str = "signal_interrupt") -> None:
        """Signals the engine to complete the current generation and terminate gracefully."""
        self._stop_reason = reason
        self._stop_event.set()
        # If currently paused, unpause so the engine can exit the loop
        self._pause_event.clear()
        self._dispatch_callbacks(EvolutionSignal.INTERRUPT)

    def pause(self) -> None:
        """Signals the engine to pause between generation boundaries."""
        self._pause_event.set()
        self._dispatch_callbacks(EvolutionSignal.PAUSE)

    def resume(self) -> None:
        """Signals a paused engine to resume its generation loop."""
        self._pause_event.clear()
        self._dispatch_callbacks(EvolutionSignal.RESUME)

    def request_checkpoint(self) -> None:
        """Requests an immediate state checkpoint at the next generation boundary."""
        self._checkpoint_event.set()
        self._dispatch_callbacks(EvolutionSignal.CHECKPOINT)

    def consume_checkpoint_request(self) -> bool:
        """Checks and clears the checkpoint flag atomically."""
        if self._checkpoint_event.is_set():
            self._checkpoint_event.clear()
            return True
        return False

    def wait_if_paused(self, poll_interval: float = 0.05, timeout: float | None = None) -> bool:
        """
        Blocks while paused. Returns True if execution should continue,
        or False if a stop was requested during pause or timeout elapsed.
        """
        start = time.perf_counter()
        while self._pause_event.is_set() and not self._stop_event.is_set():
            if timeout is not None and (time.perf_counter() - start) >= timeout:
                return False
            time.sleep(poll_interval)
        return not self._stop_event.is_set()

    def on_signal(self, sig: EvolutionSignal, callback: Callable[[], None]) -> None:
        """Registers a user callback to execute when a specific signal is dispatched."""
        self._callbacks[sig].append(callback)

    def _dispatch_callbacks(self, sig: EvolutionSignal) -> None:
        for cb in self._callbacks.get(sig, []):
            try:
                cb()
            except Exception as e:
                logger.debug("Error in signal callback for %s: %s", sig, e)

    def register_handlers(self) -> None:
        """Registers OS signal handlers on the main thread."""
        if self._registered:
            return

        # Check if we are running in the main thread
        if threading.current_thread() is not threading.main_thread():
            return

        sigs_to_register = []
        if hasattr(signal, "SIGINT"):
            sigs_to_register.append(signal.SIGINT)
        if hasattr(signal, "SIGTERM"):
            sigs_to_register.append(signal.SIGTERM)
        if hasattr(signal, "SIGUSR1"):
            sigs_to_register.append(signal.SIGUSR1)

        def _handle_os_signal(signum, frame):
            if signum in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
                sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
                sys.stderr.write(f"\n[evolab.signals] Received {sig_name}. Initiating graceful shutdown...\n")
                sys.stderr.flush()
                self.request_stop(reason=f"os_{sig_name.lower()}")
            elif hasattr(signal, "SIGUSR1") and signum == signal.SIGUSR1:
                sys.stderr.write("\n[evolab.signals] Received SIGUSR1. Scheduling checkpoint snapshot...\n")
                sys.stderr.flush()
                self.request_checkpoint()

        for s in sigs_to_register:
            try:
                prev = signal.getsignal(s)
                self._previous_handlers[s] = prev
                signal.signal(s, _handle_os_signal)
            except (ValueError, OSError) as exc:
                logger.debug("Could not register signal %s: %s", s, exc)

        self._registered = True

    def restore_handlers(self) -> None:
        """Restores previous OS signal handlers."""
        if not self._registered:
            return
        if threading.current_thread() is not threading.main_thread():
            return

        for s, prev in self._previous_handlers.items():
            try:
                signal.signal(s, prev)
            except (ValueError, OSError):
                pass
        self._previous_handlers.clear()
        self._registered = False

    def __enter__(self) -> SignalController:
        self.register_handlers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.restore_handlers()
