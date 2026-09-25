"""evolab.mcp.server — FastMCP Server for Darwin-Evolab.

Exposes Darwin-Evolab's zero-LLM symbolic Automated Program Repair (APR),
silicon logic synthesis (CGP -> Verilog RTL), high-dimensional optimization,
and empirical report inspection to AI coding assistants (Claude Desktop,
Cursor, Windsurf, Copilot, etc.) via the Model Context Protocol (MCP).
"""
from __future__ import annotations

import json
import sys
from typing import Any

from evolab.code_fixtures import SCENARIO_REGISTRY
from evolab.mcp.tools import (
    tool_benchmark_scenario,
    tool_inspect_report,
    tool_optimize_continuous,
    tool_repair_code,
    tool_synthesize_silicon,
)


def create_mcp_server():
    """Create and configure the FastMCP server instance."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required to run the Darwin-Evolab MCP server. "
            "Install it via: pip install 'evolab[mcp]' or pip install mcp"
        ) from exc

    server = FastMCP(
        name="Darwin-Evolab",
        instructions=(
            "Darwin-Evolab MCP Server provides symbolic Automated Program Repair (APR), "
            "Cartesian Genetic Programming (CGP) silicon synthesis to synthesizable Verilog-2001 RTL, "
            "vectorized continuous mathematical optimization up to 500D, and empirical run inspection."
        ),
    )

    @server.tool(
        name="repair_code",
        description=(
            "Automated Program Repair (APR) for Python source code using AST mutations and Ochiai SBFL. "
            "Finds verified bug fixes in milliseconds with 100% test pass guarantee and 0 regressions."
        ),
    )
    def repair_code(
        source_code: str | None = None,
        source_file: str | None = None,
        test_code: str | None = None,
        pytest_file: str | None = None,
        test_cases: list[dict[str, Any]] | None = None,
        target_function: str | None = None,
        max_evals: int = 128,
        apply_fix: bool = False,
    ) -> dict[str, Any]:
        """Repair Python source code against tests or assertions.

        Args:
            source_code: Raw Python source code string containing bug(s).
            source_file: Path to a Python file to repair.
            test_code: Python test code string containing assert statements.
            pytest_file: Path to a pytest file specifying test assertions.
            test_cases: List of test cases, each with 'args' (list) and 'expected' (value).
            target_function: Target function name to repair (auto-detected if omitted).
            max_evals: Maximum number of candidate evaluations (default: 128).
            apply_fix: If True and source_file provided, applies the fix directly to disk with .bak backup.
        """
        return tool_repair_code(
            source_code=source_code,
            source_file=source_file,
            test_code=test_code,
            pytest_file=pytest_file,
            test_cases=test_cases,
            target_function=target_function,
            max_evals=max_evals,
            apply_fix=apply_fix,
        )

    @server.tool(
        name="synthesize_silicon",
        description=(
            "Synthesize discrete digital logic circuits from Boolean expressions into synthesizable "
            "Verilog-2001 RTL with FPGA pinouts and formal truth-table verification (2^k)."
        ),
    )
    def synthesize_silicon(
        boolean_expr: str,
        fpga_target: str = "ice40_up5k",
        population_size: int = 20,
        generations: int = 50,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Synthesize digital logic circuits from Boolean expressions.

        Args:
            boolean_expr: Boolean logic equation (e.g. 'Out = A ^ B' or 'Sum = A ^ B ^ Cin; Cout = (A & B) | (Cin & (A ^ B))').
            fpga_target: Target FPGA preset: 'ice40_up5k', 'ice40_hx1k', 'ecp5_25k', 'artix7_35t'.
            population_size: Evolution population size (default: 20).
            generations: Maximum generations to evolve (default: 50).
            seed: Random seed for deterministic reproducibility (default: 42).
        """
        return tool_synthesize_silicon(
            boolean_expr=boolean_expr,
            fpga_target=fpga_target,
            population_size=population_size,
            generations=generations,
            seed=seed,
        )

    @server.tool(
        name="optimize_continuous",
        description=(
            "Vectorized continuous mathematical function optimization up to 500D with dynamic parsimony pressure. "
            "Supports Sphere, Rastrigin, Ackley, Rosenbrock, and Griewank landscapes."
        ),
    )
    def optimize_continuous(
        landscape: str = "sphere",
        dimensions: int = 5,
        population_size: int = 32,
        generations: int = 50,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Optimize high-dimensional mathematical benchmark landscapes.

        Args:
            landscape: Mathematical landscape: 'sphere', 'rastrigin', 'ackley', 'rosenbrock', 'griewank'.
            dimensions: Problem dimensionality (default: 5, scales up to 500).
            population_size: Evolution population size (default: 32).
            generations: Maximum generations (default: 50).
            seed: Random seed for deterministic reproducibility (default: 42).
        """
        return tool_optimize_continuous(
            landscape=landscape,
            dimensions=dimensions,
            population_size=population_size,
            generations=generations,
            seed=seed,
        )

    @server.tool(
        name="inspect_report",
        description=(
            "Inspect and validate an empirical Darwin-Evolab run_report.json for schema compliance, "
            "fitness progress, diversity metrics, and stagnation plateaus."
        ),
    )
    def inspect_report(
        report_path: str,
    ) -> dict[str, Any]:
        """Inspect and validate an empirical Darwin-Evolab run report JSON.

        Args:
            report_path: Absolute or relative path to run_report.json.
        """
        return tool_inspect_report(report_path=report_path)

    @server.tool(
        name="benchmark_scenario",
        description=(
            "Execute a pre-calibrated benchmark repair scenario ('click_cli_parser', 'requests_http_helper', "
            "'lru_cache_logic', 'multi_file_config') in sub-seconds."
        ),
    )
    def benchmark_scenario(
        scenario_name: str = "click_cli_parser",
        max_evals: int = 128,
    ) -> dict[str, Any]:
        """Execute a pre-calibrated benchmark repair scenario.

        Args:
            scenario_name: Pre-calibrated scenario name.
            max_evals: Evaluation budget (default: 128).
        """
        return tool_benchmark_scenario(
            scenario_name=scenario_name,
            max_evals=max_evals,
        )

    @server.resource(
        "evolab://benchmarks",
        name="available_benchmarks",
        description="List of all built-in pre-calibrated benchmark repair scenarios",
        mime_type="application/json",
    )
    def list_benchmarks() -> str:
        """Return available pre-calibrated benchmark repair scenarios."""
        return json.dumps(list(SCENARIO_REGISTRY.keys()), indent=2)

    return server


# Lazy module-level singleton
_server_instance = None


def get_server():
    """Get or create the singleton FastMCP server instance."""
    global _server_instance
    if _server_instance is None:
        _server_instance = create_mcp_server()
    return _server_instance


def main() -> None:
    """CLI entry point for running the FastMCP stdio server."""
    try:
        app = get_server()
        app.run()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
