"""Tests for Sandbox Runner, Process Isolation, and Timeout Safeguards."""
from __future__ import annotations

import random
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.sandbox import SandboxConfig, SandboxRunner, ExecutionResult
from evolab.evaluators import SandboxFunctionTestEvaluator
from evolab.patch import PatchGenome, Hunk, mutate_patch
from evolab.genome import Individual


def test_sandbox_clean_function_execution():
    sources = {
        "math_utils.py": "def add(a, b):\n    return a + b\n"
    }
    runner = SandboxRunner(SandboxConfig(timeout_seconds=2.0))
    res = runner.run_function(
        sources=sources,
        target_file="math_utils.py",
        func_name="add",
        args=(10, 25),
    )
    assert res.success is True
    assert res.return_value == 35
    assert res.timeout_triggered is False


def test_sandbox_infinite_loop_timeout_protection():
    """Verifies that an infinite loop (e.g. while True) is killed immediately upon timeout."""
    sources = {
        "infinite_loop.py": (
            "def hang_forever(x):\n"
            "    while True:\n"
            "        x += 1\n"
            "    return x\n"
        )
    }
    # Short timeout for fast test
    runner = SandboxRunner(SandboxConfig(timeout_seconds=0.5))
    res = runner.run_function(
        sources=sources,
        target_file="infinite_loop.py",
        func_name="hang_forever",
        args=(1,),
    )
    assert res.success is False
    assert res.timeout_triggered is True
    assert "TimeoutExpired" in (res.error or "")


def test_sandbox_fatal_exit_isolation():
    """Verifies that sys.exit(1) in mutated code does not terminate parent process."""
    sources = {
        "suicide_code.py": "import sys\ndef crash(x):\n    sys.exit(99)\n"
    }
    runner = SandboxRunner(SandboxConfig(timeout_seconds=2.0))
    res = runner.run_function(
        sources=sources,
        target_file="suicide_code.py",
        func_name="crash",
        args=(5,),
    )
    assert res.success is False
    assert res.timeout_triggered is False
    assert res.exit_code != 0


def test_sandbox_multi_file_import_resolution():
    sources = {
        "helper.py": "def square(n):\n    return n * n\n",
        "main.py": "from helper import square\ndef calc(n):\n    return square(n) + 1\n",
    }
    runner = SandboxRunner()
    res = runner.run_function(
        sources=sources,
        target_file="main.py",
        func_name="calc",
        args=(4,),
    )
    assert res.success is True
    assert res.return_value == 17


def test_sandbox_test_suite_evaluation():
    sources = {
        "solution.py": "def double(x):\n    return x * 2\n"
    }
    test_cases = [
        ((2,), 4),
        ((0,), 0),
        ((-5,), -10),
    ]
    holdout_cases = [
        ((100,), 200),
    ]
    runner = SandboxRunner()
    res = runner.run_test_suite(
        sources=sources,
        target_file="solution.py",
        func_name="double",
        test_cases=test_cases,
        holdout_cases=holdout_cases,
    )
    assert res.success is True
    val = res.return_value
    assert val["tests_passed"] == 3
    assert val["holdout_passed"] is True


def test_sandbox_evaluator_with_patch():
    base_sources = {
        "solver.py": "def add_three(x):\n    return x - 3\n" # Buggy
    }
    test_cases = [((1,), 4), ((5,), 8)]
    evaluator = SandboxFunctionTestEvaluator(
        base_sources=base_sources,
        target_file="solver.py",
        func_name="add_three",
        test_cases=test_cases,
        config=SandboxConfig(timeout_seconds=1.5),
    )

    # Initial buggy patch
    initial_res = evaluator.evaluate(PatchGenome())
    assert initial_res.score == 20.0

    # Repair patch
    repair_patch = PatchGenome(
        hunks=[Hunk("solver.py", 1, 1, "    return x - 3\n", "    return x + 3\n")]
    )
    repair_res = evaluator.evaluate(repair_patch)
    assert repair_res.score == 100.0


def test_live_sandbox_evolution():
    """Live evolutionary repair running entirely within isolated subprocess sandboxes."""
    base_sources = {
        "algo.py": "def fix_sign(n):\n    return n - 5\n" # Goal: return n + 5
    }
    test_cases = [
        ((0,), 5),
        ((10,), 15),
        ((-5,), 0),
    ]
    evaluator = SandboxFunctionTestEvaluator(
        base_sources=base_sources,
        target_file="algo.py",
        func_name="fix_sign",
        test_cases=test_cases,
        config=SandboxConfig(timeout_seconds=1.0),
    )

    rng = random.Random(42)
    pop = [Individual(genome=PatchGenome(), species="spec_sand") for _ in range(8)]

    solved = False
    for _ in range(20):
        for ind in pop:
            res = evaluator.evaluate(ind.genome)
            ind.fitness = res.score
            if ind.fitness >= 100.0:
                solved = True
                break
        if solved:
            break

        pop.sort(key=lambda x: x.fitness, reverse=True)
        elites = [ind.genome.clone() for ind in pop[:2]]
        new_pop = [Individual(genome=g, species="spec_sand") for g in elites]
        while len(new_pop) < 8:
            p = rng.choice(elites)
            child_g = mutate_patch(p, base_sources, rng)
            new_pop.append(Individual(genome=child_g, species="spec_sand"))
        pop = new_pop

    assert solved is True


def test_windows_job_guard_lifecycle():
    from evolab.sandbox import WindowsJobGuard
    guard = WindowsJobGuard(max_memory_mb=128, max_processes=2)
    if sys.platform == "win32":
        assert guard.job_handle is not None
    guard.close()
    assert guard.job_handle is None


@pytest.mark.skipif(sys.platform != "win32", reason="Job Object memory caps are Windows-only")
def test_windows_sandbox_memory_limit_containment():
    """Verifies that allocating memory beyond the configured quota is caught and contained."""
    sources = {
        "mem_bomb.py": (
            "def allocate_massive_ram(n):\n"
            "    # Attempt to allocate 500 MB\n"
            "    data = bytearray(500 * 1024 * 1024)\n"
            "    return len(data)\n"
        )
    }
    # Set tight limit (128 MB)
    runner = SandboxRunner(SandboxConfig(timeout_seconds=3.0, max_memory_mb=128))
    res = runner.run_function(
        sources=sources,
        target_file="mem_bomb.py",
        func_name="allocate_massive_ram",
        args=(1,),
    )
    assert res.success is False
    assert res.memory_limit_triggered is True
    assert res.fault_category == "resource_exhaustion"


def test_sandbox_blocks_shell_execution_attempts():
    """Verifies that running os.system or os.popen inside sandbox triggers security exception."""
    sources = {
        "malicious.py": (
            "import os\n"
            "def run_command(x):\n"
            "    return os.system('echo exploited')\n"
        )
    }
    runner = SandboxRunner(SandboxConfig(timeout_seconds=2.0))
    res = runner.run_function(
        sources=sources,
        target_file="malicious.py",
        func_name="run_command",
        args=(1,),
    )
    assert res.success is False
    assert "PermissionError" in (res.error or "") or "BLOCKED_SHELL" in (res.stderr or "")


def test_ast_purity_blocks_shell_command_execution():
    from evolab.ast_genome import validate_code_purity
    code_system = "import os\ndef f():\n    os.system('dir')\n"
    is_pure, violations = validate_code_purity(code_system)
    assert is_pure is False
    assert any("os.system" in v for v in violations)

    code_popen = "import os\ndef f():\n    os.popen('whoami')\n"
    is_pure2, violations2 = validate_code_purity(code_popen)
    assert is_pure2 is False
    assert any("os.popen" in v for v in violations2)

