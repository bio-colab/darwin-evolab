"""Unit and integration tests for Darwin-Evolab FastMCP server and tools."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp extra required for MCP tests")
pytest.importorskip("pytest_asyncio", reason="pytest-asyncio required for async MCP tests")

from evolab.mcp.server import create_mcp_server
from evolab.mcp.tools import (
    tool_benchmark_scenario,
    tool_optimize_continuous,
    tool_repair_code,
)


@pytest.mark.asyncio
async def test_mcp_server_metadata_and_tools():
    """Verify FastMCP server initializes with all 5 core tools and resources."""
    server = create_mcp_server()
    assert server.name == "Darwin-Evolab"

    tools = await server.list_tools()
    tool_names = {t.name for t in tools}
    expected_tools = {
        "repair_code",
        "synthesize_silicon",
        "optimize_continuous",
        "inspect_report",
        "benchmark_scenario",
    }
    assert expected_tools.issubset(tool_names)

    resources = await server.list_resources()
    resource_uris = {str(r.uri) for r in resources}
    assert "evolab://benchmarks" in resource_uris


@pytest.mark.asyncio
async def test_mcp_repair_code_string():
    """Test APR repair of Python string source using assert statements."""
    broken_code = """def add_positive(a, b):
    if a < 0:
        return 0
    return a - b
"""
    test_code = """assert add_positive(2, 3) == 5
assert add_positive(10, 5) == 15
assert add_positive(-1, 5) == 0
"""
    server = create_mcp_server()
    _, res = await server.call_tool(
        "repair_code",
        {"source_code": broken_code, "test_code": test_code, "max_evals": 32},
    )

    assert res["success"] is True
    assert res["score"] == 100.0
    assert "return a + b" in res["fixed_code"]
    assert "--- a/app.py" in res["diff"]


@pytest.mark.asyncio
async def test_mcp_repair_code_file_apply(tmp_path: Path):
    """Test APR repair of a physical file with in-place patch application."""
    source_file = tmp_path / "calc.py"
    source_file.write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    test_cases = [
        {"args": [2, 3], "expected": 5},
        {"args": [10, 5], "expected": 15},
    ]

    server = create_mcp_server()
    _, res = await server.call_tool(
        "repair_code",
        {
            "source_file": str(source_file),
            "test_cases": test_cases,
            "apply_fix": True,
            "max_evals": 64,
        },
    )

    assert res["success"] is True
    assert res["applied"] is True
    assert "return a + b" in source_file.read_text(encoding="utf-8")
    # Ensure backup was created
    bak_file = tmp_path / "calc.py.bak"
    assert bak_file.is_file()


@pytest.mark.asyncio
async def test_mcp_synthesize_silicon():
    """Test Cartesian Genetic Programming (CGP) silicon logic synthesis."""
    server = create_mcp_server()
    _, res = await server.call_tool(
        "synthesize_silicon",
        {
            "boolean_expr": "Out = A ^ B",
            "fpga_target": "ice40_up5k",
            "population_size": 20,
            "generations": 50,
            "seed": 42,
        },
    )

    assert res["success"] is True
    assert res["fitness_accuracy"] == 100.0
    assert res["inputs_count"] == 2
    assert res["outputs_count"] == 1
    assert "module ParsedLogic" in res["verilog_code"]
    assert "pinout_constraints" in res
    assert res["fpga_target"] == "ice40_up5k"


@pytest.mark.asyncio
async def test_mcp_optimize_continuous():
    """Test continuous mathematical landscape optimization."""
    server = create_mcp_server()
    _, res = await server.call_tool(
        "optimize_continuous",
        {
            "landscape": "sphere",
            "dimensions": 4,
            "population_size": 32,
            "generations": 50,
            "seed": 42,
        },
    )

    assert res["success"] is True
    assert res["landscape"] == "sphere"
    assert res["dimensions"] == 4
    assert res["best_fitness"] >= 95.0
    assert len(res["best_vector"]) == 4
    assert res["bloat_verdict"] == "HEALTHY_PARSIMONY"


@pytest.mark.asyncio
async def test_mcp_inspect_report(tmp_path: Path):
    """Test inspect_report on valid and nonexistent report files."""
    server = create_mcp_server()

    # 1. Nonexistent file
    _, res_missing = await server.call_tool(
        "inspect_report",
        {"report_path": str(tmp_path / "nonexistent.json")},
    )
    assert res_missing["is_valid"] is False
    assert "not found" in res_missing["error"]

    # 2. Valid minimal report conforming to Darwin-Evolab report-schema/1
    report_file = tmp_path / "run_report.json"
    report_data = {
        "total_generations": 5,
        "total_candidates_evaluated": 80,
        "best_individual": {
            "id": "gen_05_ind_02",
            "fitness": 99.7,
            "speedup_vs_baseline": "4.8x",
        },
        "species_distribution": {
            "spec_dynamic_programming": 4,
            "spec_bit_manipulation": 12,
        },
        "early_stop_triggered": True,
    }
    report_file.write_text(json.dumps(report_data), encoding="utf-8")

    _, res_valid = await server.call_tool(
        "inspect_report",
        {"report_path": str(report_file)},
    )
    assert res_valid["is_valid"] is True
    assert res_valid["total_generations"] == 5
    assert len(res_valid["summary"]) > 0


@pytest.mark.asyncio
async def test_mcp_benchmark_scenario():
    """Test benchmark_scenario for built-in scenarios and unknown scenario error handling."""
    server = create_mcp_server()

    # 1. Valid scenario
    _, res = await server.call_tool(
        "benchmark_scenario",
        {"scenario_name": "click_cli_parser", "max_evals": 128},
    )
    assert res["success"] is True
    assert res["score"] == 100.0
    assert "config['debug'] = True" in res["diff"]

    # 2. Unknown scenario
    _, res_err = await server.call_tool(
        "benchmark_scenario",
        {"scenario_name": "unknown_scenario_xyz"},
    )
    assert res_err["success"] is False
    assert "Unknown scenario" in res_err["error"]


def test_direct_tool_functions():
    """Test direct functional APIs without FastMCP transport wrapping."""
    res_bm = tool_benchmark_scenario("click_cli_parser", max_evals=128)
    assert res_bm["success"] is True
    assert res_bm["score"] == 100.0

    res_opt = tool_optimize_continuous("sphere", dimensions=3, generations=30)
    assert res_opt["success"] is True

    res_rep = tool_repair_code(
        source_code="def sub(a, b): return a + b",
        test_code="assert sub(5, 2) == 3",
        max_evals=32,
    )
    assert res_rep["success"] is True
    assert "return a - b" in res_rep["fixed_code"]


def test_cli_mcp_help(capsys):
    """Test evolab mcp --help via CLI parser."""
    from evolab.cli import build_parser

    parser = build_parser()
    subparsers = [a for a in parser._actions if a.dest == "command"][0]
    assert "mcp" in subparsers.choices
