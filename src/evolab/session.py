"""src/evolab/session.py — Stateful Session Engine with Undo/Redo & Atomic Snapshots.

Maintains interactive state across operations, enabling parameter adjustments,
trial checkpoints, and reversible configuration changes via an undo/redo stack.
Persists state deterministically to .evolab_session.json.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULT_SESSION_FILE = ".evolab_session.json"

DEFAULT_PARAMS: dict[str, Any] = {
    "engine": "GA",
    "population_size": 16,
    "generations": 20,
    "mutation_rate": 0.05,
    "seed": 42,
    "target_fitness": 99.7,
    "landscape": "rastrigin",
    "sharing_mode": "dynamic",
    "stagnation_patience": 15,
    "sandbox": True,
}


class SessionManager:
    """Stateful session controller with transaction history and undo/redo capabilities."""

    def __init__(self, session_path: str | Path | None = None) -> None:
        self.session_path = Path(session_path or DEFAULT_SESSION_FILE)
        self.params: dict[str, Any] = copy.deepcopy(DEFAULT_PARAMS)
        self.history: list[dict[str, Any]] = []
        self._undo_stack: list[dict[str, Any]] = []
        self._redo_stack: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None

    def snapshot(self) -> None:
        """Push current state onto the undo stack and clear redo stack."""
        state = copy.deepcopy(self.params)
        self._undo_stack.append(state)
        self._redo_stack.clear()

    def set_param(self, key: str, value: Any) -> None:
        """Update a session parameter with automatic snapshotting for rollback."""
        self.snapshot()
        self.params[key] = value

    def get_param(self, key: str, default: Any = None) -> Any:
        """Retrieve a session parameter value."""
        return self.params.get(key, default)

    def undo(self) -> bool:
        """Revert parameters to previous state. Returns True if undone, False if stack empty."""
        if not self._undo_stack:
            return False
        current_state = copy.deepcopy(self.params)
        self._redo_stack.append(current_state)
        self.params = self._undo_stack.pop()
        return True

    def redo(self) -> bool:
        """Reapply previously undone parameter state. Returns True if redone, False if empty."""
        if not self._redo_stack:
            return False
        current_state = copy.deepcopy(self.params)
        self._undo_stack.append(current_state)
        self.params = self._redo_stack.pop()
        return True

    def reset(self) -> None:
        """Reset parameters to clean default configuration."""
        self.snapshot()
        self.params = copy.deepcopy(DEFAULT_PARAMS)
        self.last_result = None

    def record_run(self, result: dict[str, Any]) -> None:
        """Record execution outcome in session history."""
        self.last_result = result
        summary = {
            "generations": result.get("total_generations"),
            "best_fitness": result.get("best_individual", {}).get("fitness"),
            "timestamp": result.get("timestamp_utc"),
        }
        self.history.append(summary)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete session state."""
        return {
            "params": copy.deepcopy(self.params),
            "history_count": len(self.history),
            "undo_available": len(self._undo_stack) > 0,
            "redo_available": len(self._redo_stack) > 0,
            "has_last_result": self.last_result is not None,
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Persist session state atomically to JSON file."""
        target = Path(path or self.session_path)
        payload = {
            "params": self.params,
            "history": self.history,
            "undo_stack": self._undo_stack,
            "redo_stack": self._redo_stack,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target.with_suffix(".tmp")
        temp_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp_file.replace(target)
        return target

    def load(self, path: str | Path | None = None) -> bool:
        """Load session state from file if present."""
        target = Path(path or self.session_path)
        if not target.is_file():
            return False
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.params = payload.get("params", copy.deepcopy(DEFAULT_PARAMS))
            self.history = payload.get("history", [])
            self._undo_stack = payload.get("undo_stack", [])
            self._redo_stack = payload.get("redo_stack", [])
            return True
        except Exception:
            return False
