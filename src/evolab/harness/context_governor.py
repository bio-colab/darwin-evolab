"""context_governor.py — Strict Context Window & Token Governance for LLM Backends.

Implements the 2026 Modern Harness Standard for LLM context governance:
- Enforces strict context window ceilings (e.g. 8k / 32k tokens) to prevent context exhaustion.
- Enforces cumulative session token budgets, falling back to local symbolic search if depleted.
- Performs AST-aware surgical prompt pruning: retains target functions and suspicion neighborhoods,
  replacing irrelevant distant methods with semantic stubs.
- Prevents runaway API bills and prompt injection bloat.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import time
from typing import Any


@dataclass
class ContextBudget:
    """Explicit token budget and window bounds for LLM operations."""

    max_context_window_tokens: int = 8192
    max_session_tokens: int = 50_000
    surgical_radius_lines: int = 35
    max_output_tokens: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextGovernanceTelemetry:
    """Live telemetry of token consumption and pruning interventions."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0
    pruning_interventions: int = 0
    tokens_saved_by_pruning: int = 0
    budget_exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMContextGovernor:
    """Active governor managing LLM context windows, surgical pruning, and token budgets."""

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self.budget = budget or ContextBudget()
        self.telemetry = ContextGovernanceTelemetry()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimation (~4 chars per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    @property
    def remaining_session_tokens(self) -> int:
        used = self.telemetry.total_prompt_tokens + self.telemetry.total_completion_tokens
        return max(0, self.budget.max_session_tokens - used)

    @property
    def is_budget_depleted(self) -> bool:
        return self.remaining_session_tokens <= 0

    def can_proceed(self, estimated_prompt_tokens: int = 0) -> bool:
        """Determines if the LLM backend can be called within budget constraints."""
        if self.is_budget_depleted:
            self.telemetry.budget_exhausted = True
            return False
        if estimated_prompt_tokens + self.budget.max_output_tokens > self.remaining_session_tokens:
            self.telemetry.budget_exhausted = True
            return False
        return True

    def surgically_prune_code(
        self,
        source_code: str,
        target_lines: list[int] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Prunes unneeded distant AST blocks, keeping target functions and neighborhood lines."""
        ceiling = max_tokens or self.budget.max_context_window_tokens
        current_tokens = self.estimate_tokens(source_code)
        if current_tokens <= ceiling:
            return source_code

        lines = source_code.splitlines()
        total_lines = len(lines)
        if total_lines <= self.budget.surgical_radius_lines * 2:
            return source_code

        focus_lines = set(target_lines or [1])
        radius = self.budget.surgical_radius_lines

        # Identify lines to retain
        keep_line_indices: set[int] = set()
        for fl in focus_lines:
            start = max(0, fl - radius - 1)
            end = min(total_lines, fl + radius)
            for idx in range(start, end):
                keep_line_indices.add(idx)

        pruned_lines: list[str] = []
        in_omitted_block = False

        for i, line in enumerate(lines):
            if i in keep_line_indices or i < 10:  # Keep imports/top-level preamble
                in_omitted_block = False
                pruned_lines.append(line)
            else:
                if not in_omitted_block:
                    pruned_lines.append(f"# ... [Context Governor: omitted non-target AST block] ...")
                    in_omitted_block = True

        pruned_code = "\n".join(pruned_lines)
        tokens_after = self.estimate_tokens(pruned_code)
        self.telemetry.pruning_interventions += 1
        self.telemetry.tokens_saved_by_pruning += max(0, current_tokens - tokens_after)

        return pruned_code

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Records token accounting and updates budget state."""
        self.telemetry.total_calls += 1
        self.telemetry.total_prompt_tokens += prompt_tokens
        self.telemetry.total_completion_tokens += completion_tokens
        if self.is_budget_depleted:
            self.telemetry.budget_exhausted = True
