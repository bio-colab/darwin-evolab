"""tests/test_hermetic_sandbox_jail.py — Verification for Hermetic Execution Jail.

Verifies that sandbox isolation protects host boundaries:
1. Filesystem write jail: blocks attempts to write files outside temporary scratch directory.
2. Subprocess jail: blocks subprocess.Popen and subprocess.run.
3. Native jail: blocks ctypes C-extension invocation.
4. Shell jail: blocks os.system and os.popen.
5. Network jail: blocks socket operations.
6. Privilege boundary violation classification.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from evolab.sandbox import SandboxConfig, SandboxRunner
from evolab.instrumentation import FaultCategory


def test_filesystem_jail_blocks_outside_write(tmp_path: Path):
    """Adversarial check: mutated function attempting to write host file is trapped."""
    forbidden_file = tmp_path / "host_pwned.txt"
    adversarial_code = f"""
def victim_func():
    with open(r"{forbidden_file}", "w") as f:
        f.write("compromised")
    return "done"
"""
    runner = SandboxRunner(SandboxConfig(timeout_seconds=3.0))
    res = runner.run_function(
        sources={"victim.py": adversarial_code},
        target_file="victim.py",
        func_name="victim_func",
        args=(),
    )
    assert not forbidden_file.exists(), "Host file was written! Jail failed."
    assert not res.success
    assert res.fault_category == FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION.value
    assert "PermissionError" in (res.error or "") or "SECURITY_VIOLATION" in res.stderr


def test_subprocess_spawning_blocked():
    """Adversarial check: mutated function attempting subprocess execution is trapped."""
    adversarial_code = """
import subprocess

def victim_func():
    subprocess.run(["echo", "escape"])
    return "done"
"""
    runner = SandboxRunner(SandboxConfig(timeout_seconds=3.0))
    res = runner.run_function(
        sources={"victim.py": adversarial_code},
        target_file="victim.py",
        func_name="victim_func",
        args=(),
    )
    assert not res.success
    assert res.fault_category == FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION.value
    assert "BLOCKED_SUBPROCESS" in res.stderr or "PermissionError" in (res.error or "")


def test_ctypes_loading_blocked():
    """Adversarial check: mutated function attempting ctypes loading is trapped."""
    adversarial_code = """
import ctypes

def victim_func():
    ctypes.CDLL("kernel32.dll" if False else "nonexistent.dll")
    return "done"
"""
    runner = SandboxRunner(SandboxConfig(timeout_seconds=3.0))
    res = runner.run_function(
        sources={"victim.py": adversarial_code},
        target_file="victim.py",
        func_name="victim_func",
        args=(),
    )
    assert not res.success
    assert res.fault_category == FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION.value
    assert "BLOCKED_CTYPES" in res.stderr or "PermissionError" in (res.error or "")


def test_shell_execution_blocked():
    """Adversarial check: mutated function attempting os.system is trapped."""
    adversarial_code = """
import os

def victim_func():
    os.system("whoami")
    return "done"
"""
    runner = SandboxRunner(SandboxConfig(timeout_seconds=3.0))
    res = runner.run_function(
        sources={"victim.py": adversarial_code},
        target_file="victim.py",
        func_name="victim_func",
        args=(),
    )
    assert not res.success
    assert res.fault_category == FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION.value
    assert "BLOCKED_SHELL" in res.stderr or "PermissionError" in (res.error or "")


def test_network_access_blocked():
    """Adversarial check: mutated function attempting socket connection is trapped."""
    adversarial_code = """
import socket

def victim_func():
    s = socket.socket()
    return "done"
"""
    runner = SandboxRunner(SandboxConfig(timeout_seconds=3.0, allow_network=False))
    res = runner.run_function(
        sources={"victim.py": adversarial_code},
        target_file="victim.py",
        func_name="victim_func",
        args=(),
    )
    assert not res.success
    assert res.fault_category == FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION.value
    assert "BLOCKED_SOCKET" in res.stderr or "PermissionError" in (res.error or "")


def test_filesystem_jail_permits_scratch_write():
    """Sanity check: writing to a local scratch file in cwd is allowed."""
    legitimate_code = """
def normal_func():
    # Writes to local current working directory (the ephemeral scratch jail)
    with open("temp_output.dat", "w") as f:
        f.write("safe_data")
    with open("temp_output.dat", "r") as f:
        content = f.read()
    return content
"""
    runner = SandboxRunner(SandboxConfig(timeout_seconds=3.0))
    res = runner.run_function(
        sources={"calc.py": legitimate_code},
        target_file="calc.py",
        func_name="normal_func",
        args=(),
    )
    assert res.success
    assert res.return_value == "safe_data"
