"""Sandbox Runner: Process isolation, execution timeout enforcement, and memory safeguards."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def _force_remove_readonly(func: Any, path: str, excinfo: Any) -> None:
    """Error handler for shutil.rmtree that removes write protection on Windows."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _kill_process_tree(pid: int) -> None:
    """Kills a process and all of its spawned children across Windows and Unix platforms."""
    if pid <= 0:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
    else:
        try:
            import signal
            pgid = os.getpgid(pid)
            if pgid == pid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, 9)
            except Exception:
                pass


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryLimit", ctypes.c_size_t),
            ("PeakJobMemoryLimit", ctypes.c_size_t),
        ]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x0200
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x0100
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x0008
    JobObjectExtendedLimitInformation = 9


class WindowsJobGuard:
    """Windows-native Job Object to enforce kernel-level memory caps and automatic child cleanup."""

    def __init__(self, max_memory_mb: int = 256, max_processes: int = 4):
        self.max_memory_mb = max_memory_mb
        self.max_processes = max_processes
        self.job_handle = None
        if sys.platform == "win32":
            self._setup_job()

    def _setup_job(self) -> None:
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return

            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            if self.max_memory_mb > 0:
                flags |= (JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_PROCESS_MEMORY)
                limit_bytes = max(self.max_memory_mb, 128) * 1024 * 1024
                info.JobMemoryLimit = limit_bytes
                info.ProcessMemoryLimit = limit_bytes

            info.BasicLimitInformation.LimitFlags = flags
            info.BasicLimitInformation.ActiveProcessLimit = max(self.max_processes, 1)

            ok = kernel32.SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if ok:
                self.job_handle = handle
            else:
                kernel32.CloseHandle(handle)
        except Exception:
            self.job_handle = None

    def assign_process(self, process_handle: int) -> bool:
        if not self.job_handle or sys.platform != "win32":
            return False
        try:
            return bool(ctypes.windll.kernel32.AssignProcessToJobObject(self.job_handle, int(process_handle)))
        except Exception:
            return False

    def close(self) -> None:
        if self.job_handle and sys.platform == "win32":
            try:
                ctypes.windll.kernel32.CloseHandle(self.job_handle)
            except Exception:
                pass
            self.job_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@dataclass
class SandboxConfig:
    """Configuration for subprocess isolation and safety limits."""

    timeout_seconds: float = 2.0
    max_memory_mb: int = 256
    allow_network: bool = False
    clean_env: bool = True


@dataclass
class ExecutionResult:
    """Result of an isolated code execution inside the sandbox."""

    success: bool
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    timeout_triggered: bool = False
    memory_limit_triggered: bool = False
    duration_ms: float = 0.0
    exit_code: int | None = 0
    fault_category: str = "normal_success"

    def serialize(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "return_value": self.return_value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "timeout_triggered": self.timeout_triggered,
            "memory_limit_triggered": self.memory_limit_triggered,
            "duration_ms": round(self.duration_ms, 2),
            "exit_code": self.exit_code,
            "fault_category": self.fault_category,
        }


# Worker script executed in isolated child subprocess
_WORKER_SCRIPT = r"""
import json
import sys
import types
import traceback

def _apply_sandboxing(allow_network: bool, max_memory_mb: int):
    # Enforce network isolation and log probe telemetry if invoked
    if not allow_network:
        try:
            import socket
            def _blocked(*args, **kwargs):
                sys.stderr.write("[SECURITY_VIOLATION:ENVIRONMENT_PROBE:BLOCKED_SOCKET]\n")
                sys.stderr.flush()
                raise PermissionError("Network access is blocked by Sandbox policy (allow_network=False)")
            socket.socket = _blocked
            socket.create_connection = _blocked
            socket.getaddrinfo = _blocked
            socket.gethostbyname = _blocked
        except Exception:
            pass

    # Trap introspection calls (inspect.stack, sys._getframe)
    try:
        import inspect
        _orig_stack = inspect.stack
        def _trapped_stack(*args, **kwargs):
            sys.stderr.write("[SECURITY_VIOLATION:ENVIRONMENT_PROBE:STACK_INSPECTION]\n")
            sys.stderr.flush()
            return _orig_stack(*args, **kwargs)
        inspect.stack = _trapped_stack
    except Exception:
        pass

    try:
        _orig_getframe = sys._getframe
        def _trapped_getframe(*args, **kwargs):
            sys.stderr.write("[SECURITY_VIOLATION:ENVIRONMENT_PROBE:GETFRAME_INSPECTION]\n")
            sys.stderr.flush()
            return _orig_getframe(*args, **kwargs)
        sys._getframe = _trapped_getframe
    except Exception:
        pass

    # Block direct shell command execution inside worker
    try:
        import os
        def _blocked_shell(*args, **kwargs):
            sys.stderr.write("[SECURITY_VIOLATION:ENVIRONMENT_PROBE:BLOCKED_SHELL]\n")
            sys.stderr.flush()
            raise PermissionError("Direct shell execution (os.system/os.popen) is blocked by Sandbox policy")
        os.system = _blocked_shell
        os.popen = _blocked_shell
    except Exception:
        pass

    # Enforce POSIX memory limits if supported
    if max_memory_mb > 0:
        try:
            import resource
            # On 64-bit Linux, virtual address space (RLIMIT_AS) requires headroom
            # for dynamic linker shared objects and standard allocator mapping.
            limit_mb = max(max_memory_mb, 1024)
            bytes_limit = limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        except Exception:
            pass

def main():
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input)
        
        _apply_sandboxing(
            allow_network=payload.get("allow_network", False),
            max_memory_mb=payload.get("max_memory_mb", 256)
        )
        
        sources = payload.get("sources", {})
        target_file = payload["target_file"]
        func_name = payload["func_name"]
        test_cases = payload.get("test_cases", [])
        holdout_cases = payload.get("holdout_cases", [])
        single_args = payload.get("single_args", None)
        single_kwargs = payload.get("single_kwargs", {})

        # Inject auxiliary modules
        for file_path, code in sources.items():
            if file_path != target_file:
                mod_name = file_path.removesuffix(".py").replace("/", ".").replace("\\", ".")
                compiled = compile(code, file_path, "exec")
                mod_ns = {}
                exec(compiled, mod_ns)
                mod = types.ModuleType(mod_name)
                mod.__dict__.update(mod_ns)
                sys.modules[mod_name] = mod

        # Compile and execute target file
        target_code = sources.get(target_file, "")
        target_compiled = compile(target_code, target_file, "exec")
        target_ns = {}
        exec(target_compiled, target_ns)

        if func_name not in target_ns:
            out = {
                "success": False,
                "error": f"Function {func_name} not found in {target_file}",
                "tests_passed": 0,
                "total_tests": len(test_cases),
                "holdout_passed": False,
            }
            print(json.dumps(out))
            return

        func = target_ns[func_name]

        # Mode 1: Single function call
        if single_args is not None:
            res = func(*single_args, **single_kwargs)
            out = {"success": True, "return_value": res, "error": None}
            print(json.dumps(out))
            return

        # Mode 2: Test cases execution
        def _unpack_sb_case(case):
            if len(case) == 3:
                return list(case[0]), dict(case[1]), case[2]
            if len(case) == 2:
                raw_args, expected = case
                if isinstance(raw_args, (list, tuple)) and len(raw_args) == 2 and isinstance(raw_args[1], dict):
                    return list(raw_args[0]), dict(raw_args[1]), expected
                return list(raw_args), {}, expected
            return list(case), {}, None

        passed = 0
        details = []
        for case in test_cases:
            args, kwargs, expected = _unpack_sb_case(case)
            exp_exc = None
            if isinstance(expected, str) and expected.startswith("raises:"):
                exp_exc = expected.split(":", 1)[1].strip()

            if exp_exc:
                try:
                    res = func(*args, **kwargs)
                    details.append(f"Expected exception {exp_exc}, got return value {res!r}")
                except Exception as e:
                    if type(e).__name__ == exp_exc:
                        passed += 1
                    else:
                        details.append(f"Expected exception {exp_exc}, got {type(e).__name__}: {e}")
            else:
                try:
                    res = func(*args, **kwargs)
                    if res == expected:
                        passed += 1
                    else:
                        details.append(f"Expected {expected}, got {res} for args {args}")
                except Exception as e:
                    details.append(f"Exception {type(e).__name__}: {e}")

        # Holdout cases
        holdout_passed = True
        if holdout_cases:
            h_passed = 0
            for case in holdout_cases:
                args, kwargs, expected = _unpack_sb_case(case)
                exp_exc = None
                if isinstance(expected, str) and expected.startswith("raises:"):
                    exp_exc = expected.split(":", 1)[1].strip()

                try:
                    if exp_exc:
                        try:
                            func(*args, **kwargs)
                            ok = False
                        except Exception as e:
                            ok = (type(e).__name__ == exp_exc)
                    else:
                        ok = (func(*args, **kwargs) == expected)
                    if ok:
                        h_passed += 1
                except Exception:
                    pass
            holdout_passed = (h_passed == len(holdout_cases))

        out = {
            "success": True,
            "tests_passed": passed,
            "total_tests": len(test_cases),
            "holdout_passed": holdout_passed,
            "details": details,
            "error": None,
        }
        print(json.dumps(out))

    except Exception as e:
        out = {
            "success": False,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "tests_passed": 0,
            "total_tests": 0,
            "holdout_passed": False,
        }
        print(json.dumps(out))

if __name__ == "__main__":
    main()
"""


class SandboxRunner:
    """Executes code in isolated child processes with timeout and resource limits."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()

    def run_function(
        self,
        sources: dict[str, str],
        target_file: str,
        func_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Runs a single function call inside the isolated sandbox process."""
        payload = {
            "sources": sources,
            "target_file": target_file,
            "func_name": func_name,
            "single_args": list(args),
            "single_kwargs": kwargs or {},
        }
        return self._execute_worker(payload)

    def run_test_suite(
        self,
        sources: dict[str, str],
        target_file: str,
        func_name: str,
        test_cases: Sequence[tuple[tuple[Any, ...], Any]],
        holdout_cases: Sequence[tuple[tuple[Any, ...], Any]] | None = None,
    ) -> ExecutionResult:
        """Executes a full test suite against a function inside the isolated sandbox process."""
        payload = {
            "sources": sources,
            "target_file": target_file,
            "func_name": func_name,
            "test_cases": list(test_cases),
            "holdout_cases": list(holdout_cases) if holdout_cases is not None else [],
        }
        return self._execute_worker(payload)

    def _execute_worker(self, payload: dict[str, Any]) -> ExecutionResult:
        t0 = time.perf_counter()
        payload["allow_network"] = self.config.allow_network
        payload["max_memory_mb"] = self.config.max_memory_mb
        json_input = json.dumps(payload)

        # Prepare environment
        env = os.environ.copy()
        if self.config.clean_env:
            # Minimal clean environment preserving OS runtime essentials
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "PYTHONUNBUFFERED": "1",
            }
            for k in (
                "SYSTEMROOT",
                "WINDIR",
                "HOME",
                "USER",
                "LOGNAME",
                "TMP",
                "TEMP",
                "TMPDIR",
                "LANG",
                "LC_ALL",
                "PYTHONHASHSEED",
            ):
                if k in os.environ:
                    env[k] = os.environ[k]

        cmd = [sys.executable, "-c", _WORKER_SCRIPT]

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "env": env,
        }
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True

        proc = None
        job_guard = WindowsJobGuard(max_memory_mb=self.config.max_memory_mb, max_processes=4)
        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
            if sys.platform == "win32" and proc and hasattr(proc, "_handle"):
                job_guard.assign_process(proc._handle)

            stdout, stderr = proc.communicate(
                input=json_input, timeout=self.config.timeout_seconds
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0

            from .instrumentation import FaultCategory, classify_fault

            is_mem_limit = (
                "MemoryError" in (stderr or "")
                or "MemoryError" in (stdout or "")
                or proc.returncode in (3221225495, -1073741795, 137)  # Windows 0xC0000017 / SIGKILL
            )

            if proc.returncode != 0:
                err_msg = f"Process exited with non-zero code {proc.returncode}\nStderr: {stderr}"
                fault = classify_fault(exit_code=proc.returncode, error=err_msg, stderr=stderr)
                return ExecutionResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    error=err_msg,
                    timeout_triggered=False,
                    memory_limit_triggered=is_mem_limit,
                    duration_ms=duration_ms,
                    exit_code=proc.returncode,
                    fault_category=FaultCategory.RESOURCE_EXHAUSTION.value if is_mem_limit else fault.value,
                )

            # Parse worker response from stdout
            try:
                data = json.loads(stdout.strip())
                err = data.get("error")
                is_mem = is_mem_limit or bool(err and "MemoryError" in err)
                fault = classify_fault(exit_code=0, error=err, stderr=stderr) if err else classify_fault(exit_code=0)
                return ExecutionResult(
                    success=data.get("success", False),
                    return_value=data.get("return_value") or data,
                    stdout=stdout,
                    stderr=stderr,
                    error=err,
                    timeout_triggered=False,
                    memory_limit_triggered=is_mem,
                    duration_ms=duration_ms,
                    exit_code=0,
                    fault_category=FaultCategory.RESOURCE_EXHAUSTION.value if is_mem else fault.value,
                )
            except json.JSONDecodeError:
                err_msg = f"Malformed worker output: {stdout}"
                fault = classify_fault(exit_code=proc.returncode, error=err_msg, stderr=stderr)
                return ExecutionResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    error=err_msg,
                    timeout_triggered=False,
                    memory_limit_triggered=is_mem_limit,
                    duration_ms=duration_ms,
                    exit_code=proc.returncode,
                    fault_category=FaultCategory.RESOURCE_EXHAUSTION.value if is_mem_limit else fault.value,
                )

        except subprocess.TimeoutExpired:
            if proc:
                try:
                    _kill_process_tree(proc.pid)
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            duration_ms = (time.perf_counter() - t0) * 1000.0
            from .instrumentation import FaultCategory
            return ExecutionResult(
                success=False,
                error=f"TimeoutExpired: Execution exceeded {self.config.timeout_seconds} seconds",
                timeout_triggered=True,
                duration_ms=duration_ms,
                exit_code=-1,
                fault_category=FaultCategory.RESOURCE_EXHAUSTION.value,
            )
        except Exception as ex:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            from .instrumentation import classify_fault
            fault = classify_fault(exit_code=-1, error=str(ex))
            return ExecutionResult(
                success=False,
                error=f"Sandbox Execution Exception: {ex}",
                timeout_triggered=False,
                duration_ms=duration_ms,
                exit_code=-1,
                fault_category=fault.value,
            )
        finally:
            job_guard.close()
