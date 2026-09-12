"""ipc_evaluator.py — Language-agnostic External Driver Evaluator via standard IPC JSON-RPC 2.0.

Enables darwin-evolab to drive external evaluators compiled in Rust, C++, Julia, Go,
or standalone scripts over standard subprocess pipes (stdin/stdout).

JSON-RPC 2.0 Protocol:
Request (stdin):
    {"jsonrpc": "2.0", "id": 1, "method": "evaluate", "params": {"candidate": [0.1, 0.2], "context": {}}}

Response (stdout):
    {"jsonrpc": "2.0", "id": 1, "result": {"fitness": 85.3, "passed_holdout": true, "artifacts": {}}}

Shutdown:
    {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}}
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from typing import Any

from .evaluators import Evaluator, FitnessResult
from .genome import Individual


class ExternalProcessEvaluator(Evaluator):
    """Evaluates candidates by communicating with an external process via JSON-RPC 2.0."""

    def __init__(
        self,
        driver_cmd: str | list[str],
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.driver_cmd = driver_cmd
        self.cwd = cwd
        self.timeout = timeout
        self._req_id = 0
        self._proc: subprocess.Popen | None = None
        self._start_process()

    def _start_process(self) -> None:
        """Start the external driver subprocess."""
        if isinstance(self.driver_cmd, str):
            is_windows = sys.platform.startswith("win")
            cmd = self.driver_cmd if is_windows else shlex.split(self.driver_cmd)
            shell = is_windows
        else:
            cmd = self.driver_cmd
            shell = False

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            text=True,
            bufsize=1,  # Line-buffered
            shell=shell,
        )

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def evaluate(
        self, target: Any, context: dict[str, Any] | None = None
    ) -> FitnessResult:
        """Serialize candidate, send JSON-RPC evaluate request, and parse response."""
        if not self.is_alive:
            raise RuntimeError("External driver process is not running or terminated prematurely.")

        cand = self._serialize_candidate(target)
        self._req_id += 1
        req_id = self._req_id

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "evaluate",
            "params": {
                "candidate": cand,
                "context": context or {},
            },
        }

        start_time = time.perf_counter()
        req_line = json.dumps(payload) + "\n"

        try:
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(req_line)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            stderr_out = self._proc.stderr.read() if self._proc and self._proc.stderr else ""
            raise RuntimeError(
                f"Failed writing to external driver stdin: {exc}. Stderr: {stderr_out}"
            ) from exc

        assert self._proc.stdout is not None
        resp_line = self._proc.stdout.readline()
        if not resp_line:
            stderr_out = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(
                f"External driver closed stdout unexpectedly. Process exit code: {self._proc.poll()}. Stderr: {stderr_out}"
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        try:
            resp = json.loads(resp_line.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"External driver returned invalid JSON: {resp_line!r}. Error: {exc}"
            ) from exc

        if "error" in resp:
            err = resp["error"]
            raise RuntimeError(f"External driver returned JSON-RPC error: {err}")

        result_data = resp.get("result", {})
        if isinstance(result_data, (int, float)):
            score = float(result_data)
            sub_scores: dict[str, float] = {}
            passed_holdout = None
            artifacts: dict[str, Any] = {}
        elif isinstance(result_data, dict):
            score = float(result_data.get("fitness", result_data.get("score", 0.0)))
            sub_scores = dict(result_data.get("sub_scores", {}))
            passed_holdout = result_data.get("passed_holdout")
            artifacts = dict(result_data.get("artifacts", {}))
        else:
            score = 0.0
            sub_scores = {}
            passed_holdout = None
            artifacts = {}

        return FitnessResult(
            score=score,
            sub_scores=sub_scores,
            passed_holdout=passed_holdout,
            artifacts=artifacts,
            evaluation_time_ms=elapsed_ms,
        )

    def _serialize_candidate(self, target: Any) -> Any:
        """Extract serializable candidate representation."""
        if isinstance(target, Individual):
            return self._serialize_candidate(target.genome)
        if isinstance(target, (list, tuple)):
            return list(target)
        if hasattr(target, "genes"):
            return list(target.genes)
        if hasattr(target, "code"):
            return target.code
        if hasattr(target, "edits"):
            return [
                e.to_dict() if hasattr(e, "to_dict") else dict(e)
                for e in target.edits
            ]
        if hasattr(target, "to_dict"):
            return target.to_dict()
        return str(target)

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "external_ipc"

    def close(self) -> None:
        """Shutdown the external driver cleanly."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    shutdown_msg = json.dumps({
                        "jsonrpc": "2.0",
                        "id": self._req_id + 1,
                        "method": "shutdown",
                        "params": {},
                    }) + "\n"
                    self._proc.stdin.write(shutdown_msg)
                    self._proc.stdin.flush()
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=1.0)
        self._proc = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> ExternalProcessEvaluator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
