# Model Context Protocol (MCP) Integration Guide

Darwin-Evolab provides an official, zero-LLM **Model Context Protocol (MCP)** server built on Anthropic's `FastMCP` architecture. This server transforms Darwin-Evolab into an autonomous symbolic repair, hardware synthesis, and optimization engine that any AI agent (Claude Desktop, Cursor, Windsurf, Copilot, etc.) can seamlessly invoke.

---

## 1. Quick Setup & Configuration

### Prerequisites
Install Darwin-Evolab with MCP support:
```bash
pip install "evolab[mcp]"
# Or if working from local source:
pip install -e ".[mcp]"
```

Verify the server responds:
```bash
evolab mcp --help
```

---

### Integration with Claude Desktop
Add Darwin-Evolab to your `claude_desktop_config.json`:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "darwin-evolab": {
      "command": "evolab-mcp"
    }
  }
}
```

Or using an explicit Python environment:
```json
{
  "mcpServers": {
    "darwin-evolab": {
      "command": "python",
      "args": ["-m", "evolab.mcp.server"]
    }
  }
}
```

---

### Integration with Cursor
Add Darwin-Evolab to your `.cursor/mcp.json` (or Cursor Settings -> Features -> MCP):

```json
{
  "mcpServers": {
    "darwin-evolab": {
      "command": "evolab",
      "args": ["mcp"]
    }
  }
}
```

---

## 2. Available Agent Tools

Darwin-Evolab exposes 5 deterministic, verified symbolic tools:

### `repair_code`
Automated Program Repair (APR) for Python source code using AST mutations and Ochiai spectrum-based fault localization (SBFL). Guarantees **100% test pass and zero regressions** in milliseconds.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `source_code` | string | `null` | Raw Python code string containing bug(s). |
| `source_file` | string | `null` | Path to physical `.py` file to repair. |
| `test_code` | string | `null` | Test string containing Python `assert` statements. |
| `pytest_file` | string | `null` | Path to a `pytest` file containing test assertions. |
| `test_cases` | list | `null` | Structured test cases `[{"args": [...], "expected": ...}]`. |
| `target_function` | string | `null` | Target function name (auto-detected if omitted). |
| `max_evals` | integer | `128` | Maximum candidate search evaluation budget. |
| `apply_fix` | boolean | `false` | Apply verified fix directly in-place with `.bak` backup. |

**Example Agent Call:**
```json
{
  "source_code": "def add_positive(a, b):\n    if a < 0: return 0\n    return a - b\n",
  "test_code": "assert add_positive(2, 3) == 5\nassert add_positive(-1, 5) == 0\n"
}
```

---

### `synthesize_silicon`
Synthesizes discrete digital logic circuits from Boolean expressions into synthesizable Verilog-2001 RTL with FPGA pinouts and $2^k$ exhaustive formal truth-table verification.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `boolean_expr` | string | *required* | Boolean equation (e.g. `'Out = A ^ B'` or `'Sum = A ^ B ^ Cin; Cout = (A & B) \| (Cin & (A ^ B))'`). |
| `fpga_target` | string | `"ice40_up5k"` | Board preset: `ice40_up5k`, `ice40_hx1k`, `ecp5_25k`, `artix7_35t`. |
| `population_size` | integer | `20` | Cartesian Genetic Programming (CGP) population size. |
| `generations` | integer | `50` | Maximum evolutionary generations. |
| `seed` | integer | `42` | Random seed for deterministic reproducibility. |

**Example Agent Call:**
```json
{
  "boolean_expr": "Out = A ^ B",
  "fpga_target": "ice40_up5k"
}
```

---

### `optimize_continuous`
Vectorized continuous mathematical function optimization up to 500D with dynamic parsimony pressure.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `landscape` | string | `"sphere"` | Benchmark landscape: `sphere`, `rastrigin`, `ackley`, `rosenbrock`, `griewank`. |
| `dimensions` | integer | `5` | Dimensionality of search space (scales up to 500D). |
| `population_size` | integer | `32` | Evolution population size. |
| `generations` | integer | `50` | Maximum generations. |
| `seed` | integer | `42` | Random seed for deterministic reproducibility. |

---

### `inspect_report`
Inspects and validates an empirical Darwin-Evolab `run_report.json` for schema compliance (`report-schema/1`), fitness progress, and diversity metrics.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `report_path` | string | *required* | Absolute or relative path to `run_report.json`. |

---

### `benchmark_scenario`
Executes pre-calibrated benchmark repair scenarios in sub-seconds for immediate verification.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `scenario_name` | string | `"click_cli_parser"` | Built-in scenario: `click_cli_parser`, `requests_http_helper`, `lru_cache_logic`, `multi_file_config`. |
| `max_evals` | integer | `128` | Search evaluation budget. |

---

## 3. MCP Resources

Darwin-Evolab exposes live system resources accessible via MCP:
- `evolab://benchmarks`: JSON list of all available built-in benchmark repair scenarios.

---

## 4. Architectural Separation

The MCP server adheres strictly to Darwin-Evolab's architectural principles:
1. **Decoupled Transport**: Functional execution logic resides in `evolab.mcp.tools`, isolated from JSON-RPC and stdio protocol transport.
2. **Deterministic & Zero-LLM**: All repairs and syntheses are executed using AST search, Ochiai SBFL, and CGP truth tables. No external LLM token calls are made.
3. **Safe File Handling**: Any in-place file modifications (`apply_fix=True`) automatically generate `.bak` backups before modifying the target file.
