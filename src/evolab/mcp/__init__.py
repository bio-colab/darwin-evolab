"""evolab.mcp — Model Context Protocol (MCP) server & tools for Darwin-Evolab."""
from __future__ import annotations

from evolab.mcp.tools import (
    tool_benchmark_scenario,
    tool_inspect_report,
    tool_optimize_continuous,
    tool_repair_code,
    tool_synthesize_silicon,
)

__all__ = [
    "create_mcp_server",
    "get_server",
    "tool_benchmark_scenario",
    "tool_inspect_report",
    "tool_optimize_continuous",
    "tool_repair_code",
    "tool_synthesize_silicon",
]


def __getattr__(name: str):
    if name in ("create_mcp_server", "get_server"):
        from evolab.mcp.server import create_mcp_server, get_server

        return {"create_mcp_server": create_mcp_server, "get_server": get_server}[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
