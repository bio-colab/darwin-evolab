"""taxonomy.py — Fault and Crash Classification for darwin-evolab."""
from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Any


class FaultCategory(str, enum.Enum):
    """Fine-grained taxonomy of execution outcomes and fault categories."""

    NORMAL_SUCCESS = "normal_success"
    NORMAL_FAILURE = "normal_failure"
    LOGIC_DEVIATION = "logic_deviation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNEXPECTED_TERMINATION = "unexpected_termination"
    MEMORY_SAFETY_SIGNAL = "memory_safety_signal"
    PRIVILEGE_BOUNDARY_VIOLATION = "privilege_boundary_violation"
    STATE_CORRUPTION = "state_corruption"


# Well-known memory safety and fatal crash signals across platforms
_POSIX_MEMORY_SIGNALS = {
    11: "SIGSEGV (Segmentation Fault)",
    139: "SIGSEGV (Exit 139)",
    6: "SIGABRT (Abnormal Termination)",
    134: "SIGABRT (Exit 134)",
    7: "SIGBUS (Bus Error)",
    135: "SIGBUS (Exit 135)",
    4: "SIGILL (Illegal Instruction)",
    132: "SIGILL (Exit 132)",
}

_WIN32_CRASH_CODES = {
    3221225477: "STATUS_ACCESS_VIOLATION (0xC0000005)",
    -1073741819: "STATUS_ACCESS_VIOLATION (0xC0000005)",
    3221225725: "STATUS_STACK_OVERFLOW (0xC00000FD)",
    -1073741571: "STATUS_STACK_OVERFLOW (0xC00000FD)",
    3221226505: "STATUS_STACK_BUFFER_OVERRUN (0xC0000409)",
    -1073740791: "STATUS_STACK_BUFFER_OVERRUN (0xC0000409)",
}


def classify_fault(
    exit_code: int | None = 0,
    error: str | None = None,
    stderr: str = "",
    timeout_triggered: bool = False,
    memory_limit_triggered: bool = False,
    telemetry_events: Sequence[dict[str, Any]] | None = None,
    details: Sequence[str] | None = None,
) -> FaultCategory:
    """Classifies an execution outcome into a precise FaultCategory."""
    details_str = " ".join(str(d) for d in (details or []))
    err_str = (error or "") + " " + stderr + " " + details_str

    # 1. Resource Exhaustion
    if timeout_triggered or memory_limit_triggered:
        return FaultCategory.RESOURCE_EXHAUSTION
    if any(sig in err_str for sig in ("MemoryError", "RLIMIT_AS", "ResourceExhausted", "TimeoutExpired")):
        return FaultCategory.RESOURCE_EXHAUSTION

    # 2. Memory Safety Signals (POSIX signals & Windows NTSTATUS)
    if exit_code is not None:
        if exit_code in _POSIX_MEMORY_SIGNALS or exit_code in _WIN32_CRASH_CODES:
            return FaultCategory.MEMORY_SAFETY_SIGNAL
        if any(sig in err_str for sig in ("Segmentation fault", "STATUS_ACCESS_VIOLATION", "SIGSEGV", "SIGBUS")):
            return FaultCategory.MEMORY_SAFETY_SIGNAL

    # 3. Privilege & Boundary Violations
    if any(sig in err_str for sig in ("PermissionError", "Network access is blocked", "socket.socket = _blocked", "Operation not permitted", "ENVIRONMENT_PROBE", "SECURITY_VIOLATION")):
        return FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION
    if telemetry_events:
        for ev in telemetry_events:
            cat = str(ev.get("cat", ""))
            if cat.startswith(("socket", "network", "blocked_syscall")):
                return FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION

    # 4. State Corruption
    if any(sig in err_str for sig in ("StateCorruption", "WorkspaceBleed", "ConflictingGlobalState")):
        return FaultCategory.STATE_CORRUPTION

    # 5. Unexpected Termination
    if exit_code is not None and exit_code not in (0, 1):
        return FaultCategory.UNEXPECTED_TERMINATION

    # 6. Logic Deviations (Standard unhandled exceptions)
    standard_exceptions = (
        "ValueError", "TypeError", "IndexError", "KeyError", "AttributeError",
        "ZeroDivisionError", "NameError", "UnboundLocalError", "RecursionError"
    )
    if any(exc in err_str for exc in standard_exceptions):
        return FaultCategory.LOGIC_DEVIATION

    # 7. Normal Failure vs Normal Success
    if error or (exit_code is not None and exit_code != 0):
        return FaultCategory.NORMAL_FAILURE

    return FaultCategory.NORMAL_SUCCESS


