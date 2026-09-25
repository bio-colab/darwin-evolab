"""JEV System One Client for Darwin-Evolab.

Strictly private and local to JEV/ — excluded from Git tracking.
Interfaces with TypeSafe AI System One models (model: jev-latest).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

KEY_FILE = Path(__file__).resolve().parent / "API key for test.txt"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL_NAME = "jev-latest"

_WARNED_JEV_KEY = False


class JevClient:
    """Client for TypeSafe System One API with live endpoint & offline reproducible fallback."""

    def __init__(self, api_key: str | None = None, offline_mode: bool = False) -> None:
        if api_key is None:
            api_key = os.environ.get("JEV_API_KEY")
            if not api_key and KEY_FILE.exists():
                try:
                    api_key = KEY_FILE.read_text(encoding="utf-8").strip()
                except Exception:
                    api_key = None
        self.api_key = api_key
        self.offline_mode = offline_mode or (not api_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.total_api_time_seconds = 0.0

        global _WARNED_JEV_KEY
        if not _WARNED_JEV_KEY and not os.environ.get("EVOLAB_QUIET"):
            import sys
            msg = (
                "[WARN] [evolab:jev] JEV_API_KEY not found in environment -- using local offline heuristic routing (System-2 fallback).\n"
                "       For live cloud System-1 routing (76% search space reduction): export JEV_API_KEY=\"<your_key>\""
                if self.offline_mode
                else "[INFO] [evolab:jev] Connected to TypeSafe AI System-One live endpoint (model: jev-latest)."
            )
            try:
                print(msg, file=sys.stderr)
            except Exception:
                pass
            _WARNED_JEV_KEY = True

    def evaluate(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        model: str = MODEL_NAME,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Evaluates state against typed questions and returns structured answers."""
        if self.offline_mode:
            self.total_calls += 1
            self.total_input_tokens += 120
            self.total_output_tokens += 40
            answers = {}
            for q_name, q_val in questions.items():
                crit = q_val.get("criteria", {})
                probs = {}
                failing_info = str(state.get("failing_test", "")).lower() if isinstance(state, dict) else ""
                for k in crit.keys():
                    p = 0.05
                    if k == "int_wrap" and ("int" in failing_info or "string" in failing_info):
                        p = 0.85
                    elif k == "bool_flip" and ("bool" in failing_info or "true" in failing_info or "false" in failing_info or "disjunction" in failing_info):
                        p = 0.85
                    elif k in ("hit_move_to_end", "pop_to_front") and ("cache" in failing_info or "lru" in failing_info or "order" in failing_info):
                        p = 0.85
                    elif k == "string_sep" and ("separator" in failing_info or "&" in failing_info or "delim" in failing_info or "ampersand" in failing_info):
                        p = 0.85
                    elif k in ("boundary_flip", "compare_flip") and ("<" in failing_info or ">" in failing_info or "bound" in failing_info):
                        p = 0.85
                    elif k == "index_flip" and ("index" in failing_info or "subscript" in failing_info):
                        p = 0.85
                    probs[k] = p
                tot = sum(probs.values()) or 1.0
                normalized = {k: round(v / tot, 4) for k, v in probs.items()}
                best_k = max(normalized, key=normalized.get) if normalized else "unknown"
                answers[q_name] = {
                    "value": best_k,
                    "probabilities": normalized,
                }
            return {
                "model": model,
                "answers": answers,
                "usage": {"input_tokens": 120, "output_tokens": 40},
                "offline_mock": True,
            }

        payload = {
            "state": state,
            "model": model,
            "questions": questions,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=body,
        )

        for attempt in range(max_retries):
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    raw_data = resp.read().decode("utf-8")
                    duration = time.perf_counter() - t0
                    self.total_api_time_seconds += duration
                    self.total_calls += 1

                    data = json.loads(raw_data)
                    usage = data.get("usage", {})
                    self.total_input_tokens += usage.get("input_tokens", 0)
                    self.total_output_tokens += usage.get("output_tokens", 0)
                    return data
            except urllib.error.HTTPError as err:
                duration = time.perf_counter() - t0
                if err.code in (429, 529) and attempt < max_retries - 1:
                    time.sleep(2.0 ** attempt)
                    continue
                err_msg = err.read().decode("utf-8") if hasattr(err, "read") else str(err)
                raise RuntimeError(f"Jev API HTTP {err.code}: {err_msg}") from err
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                    continue
                raise RuntimeError(f"Jev API network failure: {exc}") from exc

        raise RuntimeError("Jev API max retries exceeded.")

    def prioritize_operators(
        self,
        code: str,
        failing_test_info: str,
        candidate_kinds: list[str],
    ) -> dict[str, float]:
        """Predicts probability distribution over candidate mutation kinds."""
        if not candidate_kinds:
            return {}
        if len(candidate_kinds) == 1:
            return {candidate_kinds[0]: 1.0}

        # Format criteria descriptions for known kinds
        kind_descriptions = {
            "int_wrap": "Wrap variable or subscript with int(...) type cast",
            "bool_flip": "Invert boolean literal (True <-> False)",
            "index_flip": "Change numeric index (e.g., 0 to 1, or 1 to 0)",
            "string_sep": "Change string separator or delimiter (e.g., ',' to '&')",
            "hit_move_to_end": "Move accessed item to end of cache list",
            "pop_to_front": "Pop from beginning of list (index 0) instead of end",
            "swap_bounds": "Swap boundary values or argument order",
            "compare_flip": "Invert comparison operator (<, <=, >, >=, ==, !=)",
            "boundary_flip": "Adjust strict vs non-strict boundary (< to <=, > to >=)",
            "insert_guard": "Insert defensive guard condition / None check",
            "off_by_one": "Adjust integer arithmetic by +1 or -1",
            "delete_statement": "Remove redundant or erroneous statement",
        }

        criteria = {}
        for k in candidate_kinds:
            criteria[k] = kind_descriptions.get(k, f"Apply {k} mutation")

        state = {
            "source_code": code,
            "failing_test": failing_test_info,
        }
        questions = {
            "best_operator": {
                "type": "choice",
                "instructions": "Which mutation kind is most directly capable of fixing this test failure?",
                "criteria": criteria,
            }
        }

        response = self.evaluate(state=state, questions=questions)
        answer = response.get("answers", {}).get("best_operator", {})
        probs = answer.get("probabilities", {})
        # Normalize and ensure all candidate kinds have an entry
        total = sum(probs.values()) or 1.0
        return {k: probs.get(k, 0.001) / total for k in candidate_kinds}
