"""wizard.py — Interactive Onboarding Wizard for darwin-evolab.

Provides a friendly, step-by-step interactive CLI interface for new users and beginners.
Guides users through selecting tasks, detecting local Python and test files,
configuring hyperparameters with sensible defaults, and launching executions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .terminal import _BOLD, _CYAN, _DIM, _GREEN, _RESET, _YELLOW, supports_color


def _prompt(question: str, default: str = "", use_color: bool = True) -> str:
    """Helper to prompt the user with a default value."""
    if default:
        hint = f"{_DIM}[{default}]{_RESET} " if use_color else f"[{default}] "
    else:
        hint = ""
    prefix = f"{_BOLD}{_CYAN}?{_RESET} " if use_color else "? "
    try:
        ans = input(f"{prefix}{question} {hint}").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nOperation cancelled by user.")
        sys.exit(0)
    return ans if ans else default


def run_wizard() -> int:
    """Run interactive onboarding wizard."""
    use_color = supports_color(sys.stdout)

    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║               DARWIN-EVOLAB INTERACTIVE ONBOARDING WIZARD                ║
║  Universal Evolutionary Optimization across Code, Logic, and Mathematics ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    if use_color:
        print(f"{_BOLD}{_CYAN}{banner}{_RESET}")
    else:
        print(banner)

    print("Welcome to evolab! Let's configure and run an evolutionary experiment.\n")
    print(f"{_BOLD}What would you like to run today?{_RESET}" if use_color else "What would you like to run today?")
    print("  [1] Software Bug Repair (AST-based APR with Pytest)")
    print("  [2] Built-in Code Scenario (Click CLI parser demo)")
    print("  [3] Numerical / Mathematical Optimization (GA Engine)")
    print("  [4] Autonomous Self-Audit & Safety Invariant Check")
    print("  [5] SWE-bench Lite Instance Solver")
    print("  [q] Quit\n")

    choice = _prompt("Enter selection (1-5)", default="1", use_color=use_color)
    if choice in ("q", "quit", "exit"):
        return 0

    from ..cli import main as cli_main

    if choice == "1":
        print("\n--- [Software Bug Repair Mode] ---")
        py_files = sorted([p.name for p in Path(".").glob("*.py") if not p.name.startswith("test_") and p.name != "run.py"])
        default_source = py_files[0] if py_files else "app.py"
        source = _prompt("Target Python source file to repair", default=default_source, use_color=use_color)

        test_files = sorted([p.name for p in Path(".").glob("test_*.py")] + [p.name for p in Path("tests").glob("test_*.py") if Path("tests").exists()])
        default_test = test_files[0] if test_files else "tests/test_app.py"
        pytest_file = _prompt("Pytest file containing test assertions", default=default_test, use_color=use_color)

        apply_patch = _prompt("Apply repair in-place if successful? (y/N)", default="n", use_color=use_color).lower() == "y"
        diff_flag = ["--diff"]
        apply_flag = ["--apply"] if apply_patch else []

        cmd = ["repair", "--source", source, "--pytest", pytest_file] + diff_flag + apply_flag
        print(f"\nLaunching: evolab {' '.join(cmd)}\n")
        return cli_main(cmd)

    elif choice == "2":
        print("\n--- [Built-in Scenario: Click CLI Parser] ---")
        cmd = ["evolve", "--scenario", "click_cli_parser", "--diff"]
        print(f"Launching: evolab {' '.join(cmd)}\n")
        return cli_main(cmd)

    elif choice == "3":
        print("\n--- [Numerical Optimization Mode] ---")
        gens = _prompt("Number of generations", default="30", use_color=use_color)
        pop = _prompt("Population size", default="16", use_color=use_color)
        seed = _prompt("Random seed (or blank for random)", default="42", use_color=use_color)

        cmd = ["optimize", "-g", gens, "-p", pop]
        if seed:
            cmd.extend(["-s", seed])
        print(f"\nLaunching: evolab {' '.join(cmd)}\n")
        return cli_main(cmd)

    elif choice == "4":
        print("\n--- [Autonomous Self-Audit Mode] ---")
        full = _prompt("Run full benchmark suite? (y/N)", default="n", use_color=use_color).lower() == "y"
        cmd = ["audit"]
        if full:
            cmd.append("--full")
        print(f"\nLaunching: evolab {' '.join(cmd)}\n")
        return cli_main(cmd)

    elif choice == "5":
        print("\n--- [SWE-bench Lite Solver Mode] ---")
        swe_fixtures = sorted(Path("fixtures/swe_bench").glob("*.json")) if Path("fixtures/swe_bench").exists() else []
        default_swe = str(swe_fixtures[0]) if swe_fixtures else "fixtures/swe_bench/django__django-11099.json"
        swe_path = _prompt("SWE-bench instance JSON file", default=default_swe, use_color=use_color)
        cmd = ["evolve", "--swe-bench", swe_path]
        print(f"\nLaunching: evolab {' '.join(cmd)}\n")
        return cli_main(cmd)

    else:
        print(f"Unknown choice: {choice}")
        return 1
