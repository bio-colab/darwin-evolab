# Darwin-Evolab: Hands-On Developer & User Guide

> **A practical, task-oriented guide to automated program repair, digital circuit synthesis, and evolutionary optimization with `evolab`.**

---

## ⚡ 1. Installation & Verification

### Prerequisites
- **Python**: 3.10, 3.11, or 3.12 (64-bit).
- **Operating System**: Linux, macOS, or Windows.
- **Hardware**: Any consumer laptop or desktop (no GPU or cloud cluster required).

### Fast Installation

```bash
# Clone the repository
git clone https://github.com/bio-colab/darwin-evolab.git
cd darwin-evolab

# Install core kernel & CLI in editable mode (zero external dependencies)
pip install -e .

# Or install with the full scientific suite (symbolic math, CST parser, Z3)
pip install -e ".[full]"
```

### Verify Your Installation
Run the version check to ensure `evolab` is globally accessible on your `PATH`:

```bash
evolab --version
```
Expected output:
```text
evolab 0.6.0
```

> [!TIP]
> **Interactive Onboarding Wizard**:  
> If you are new to evolutionary optimization, run `evolab wizard` for an interactive step-by-step tour through software repair, silicon synthesis, and mathematical optimization!

---

## 🔧 2. Automated Program Repair (APR) Track

Darwin-Evolab repairs localized bugs in Python source code using **AST mutations guided by Spectrum-Based Fault Localization (Ochiai SBFL)** and dual invariants (`FAIL_TO_PASS` = 100%, `PASS_TO_PASS` = 0% regressions).

### 2.1 Run a 1-Second Built-in Benchmark Scenario
Try an instant repair on a real-world defect scenario:

```bash
evolab repair --scenario click_cli_parser
```

Output:
```diff
[evolab:apr] Search Finished: Best Score = 100.00 | Evaluations = 4 | Elapsed = 0.17s
Generations run : 4
Candidates      : 4
Best            : gen_04_ind_00 (fitness=100.0, species=spec_code)
--- a/cli_parser.py
+++ b/cli_parser.py
@@ -2,9 +2,9 @@
     config = {'port': 8000, 'debug': False, 'host': '127.0.0.1'}
     for arg in args:
         if arg == '--debug':
-            config['debug'] = False
+            config['debug'] = True
         elif arg.startswith('--port='):
-            config['port'] = arg.split('=')[1]
+            config['port'] = int(arg.split('=')[1])
         elif arg.startswith('--host='):
-            config['host'] = arg.split('=')[0]
+            config['host'] = arg.split('=')[1]
     return config
```

Available built-in scenarios:
- `click_cli_parser`: Option parsing, type coercion, and delimiter correction.
- `requests_http_helper`: HTTP header encoding and authentication prefixing.
- `lru_cache_logic`: Pointer updating, eviction order, and cache boundaries.
- `multi_file_config`: Cross-file configuration passing and schema defaults.

---

### 2.2 Repair Your Own Code Using Pytest
If you have a buggy module (`app.py`) and a test file (`test_app.py`) where tests currently fail:

```bash
evolab repair --source app.py --pytest test_app.py --diff
```

What happens under the hood:
1. `evolab` executes tests in an isolated sandbox and collects failing assertions.
2. Ochiai SBFL scores code statements by suspicion score.
3. The genetic engine generates, validates, and evaluates AST mutations.
4. A verified patch is displayed as a clean unified diff.

#### Applying the Patch In-Place
To apply the discovered patch directly to your source file (with automatic `.bak` safety backup):
```bash
evolab repair --source app.py --pytest test_app.py --apply
```

#### Exporting a Git Patch File
To generate a standardized `.patch` file compatible with `git apply`:
```bash
evolab repair --source app.py --pytest test_app.py --patch-file fix.patch
```

---

### 2.3 Repair Using JSON Specifications
For headless CI pipelines or automated harnesses, pass a JSON test specification:

```bash
evolab repair --source app.py --tests tests.json --func my_function
```

`tests.json` schema:
```json
[
  {"args": [[100, 15]], "expected": 115},
  {"args": [[200, 30]], "expected": 230}
]
```

---

## ⚙️ 3. Silicon Synthesis & Hardware Track (CGP)

Darwin-Evolab includes a Cartesian Genetic Programming (CGP) engine that synthesizes **discrete digital logic circuits** from Boolean specifications, proves their correctness exhaustively via $2^k$ truth-table verification, and exports synthesizable Verilog-2001 RTL.

### 3.1 Synthesizing Logic from Equations
Synthesize a 1-bit full adder from Boolean logic equations:

```bash
evolab evolve \
  --expr "Sum = A ^ B ^ Cin; Cout = (A & B) | (Cin & (A ^ B))" \
  --verilog-file full_adder.v \
  --ui-file workbench.html
```

Output:
- Exhaustive verification across all $2^3 = 8$ truth-table rows.
- Synthesized Verilog module saved to `full_adder.v`.
- Standalone interactive HTML5 workbench saved to `workbench.html`.

### 3.2 Targeting Physical FPGA Boards
Target specific physical FPGA board presets with automated pin constraints:

```bash
evolab evolve \
  --expr "Out = A & B" \
  --fpga-target ice40_up5k \
  --verilog-file gate.v
```

Supported FPGA targets:
- `ice40_hx1k`: Lattice iCE40-HX1K (`.pcf` constraints)
- `ice40_up5k`: Lattice iCE40-UltraPlus (`.pcf` constraints)
- `ecp5_25k`: Lattice ECP5-25F (`.lpf` constraints)
- `artix7_35t`: Xilinx Artix-7 (`.xdc` constraints)

### 3.3 Launching the Interactive Silicon Workbench
Test circuits interactively in your web browser with live WebUSB hardware flashing:

```bash
evolab serve-workbench workbench.html --port 8080
```
Open `http://localhost:8080` to toggle logic switches, inspect oscilloscope waveforms, and flash connected FPGA boards.

---

## 📈 4. Continuous & High-Dimensional Optimization

Darwin-Evolab provides vectorized continuous optimization engines scaling up to 500 dimensions with active parsimony pressure (preventing stagnation and dimensional bloat).

### 4.1 Running Continuous Benchmark Optimization

```bash
# Optimize 100-dimensional Rastrigin landscape
evolab optimize -g 50 -p 32 -s 42
```

### 4.2 Standalone Fitness Oracle (CLI & Pipes)
Evaluate candidate vectors directly from the command line or via standard input:

```bash
# Evaluate a 2D coordinate vector on the default landscape
echo "[0.0, 0.0]" | evolab eval

# Or pass directly as an argument
evolab eval "[1.2, -0.5]" --landscape ackley --format json
```

---

## 📁 5. Configuration Management (`evolab.toml`)

To avoid repeating command-line flags, configure project defaults:

```bash
# Initialize a starter configuration in the current directory
evolab init
```

Generates `evolab.toml`:
```toml
[evolab]
seed = 42
max_evals = 32
target = 99.7
use_cache = true
diff = true
format = "console"

[evolab.repair]
sandbox = false
prioritize_by_suspicion = true

[evolab.silicon]
fpga_target = "ice40_up5k"
objective = "balanced"
```

Configuration resolution hierarchy (lowest to highest precedence):
1. Built-in defaults
2. `~/.config/evolab/config.toml` (User global config)
3. `./evolab.toml` (Project local config)
4. Environment variables (`EVOLAB_*`)
5. Explicit CLI arguments

---

## ⚡ 6. Dual-System Acceleration (TypeSafe AI JEV)

Darwin-Evolab integrates with **TypeSafe AI JEV System-One models** for sub-second instinctual operator routing, reducing the mutation search space by **76.0%**.

### Activation:
Set the environment variable:
```bash
# Linux / macOS
export JEV_API_KEY="your_typesafe_api_key"

# Windows PowerShell
$env:JEV_API_KEY="your_typesafe_api_key"
```

When `JEV_API_KEY` is present, `evolab` automatically pairs System 1 neural routing with System 2 evolutionary verification. If omitted, `evolab` falls back to its built-in local offline heuristic routing seamlessly.

---

## ❓ 7. Troubleshooting & FAQs

### Q: "Target fitness not reached within evaluation budget"
**Cause**: The bug requires multi-hunk composition or the evaluation budget was exhausted before exploring the correct branch.  
**Fix**:
1. Increase search budget: `evolab repair ... --max-evals 64`.
2. Inspect root-cause fault trace: `evolab repair ... --diagnose`.
3. Check test isolation: ensure your test asserts specifically on the function being repaired.

### Q: "How do I inspect a previous run?"
```bash
evolab inspect run_report.json
```
Prints validation status, candidate diversity, fitness trend, and best genome summary.

### Q: "Can I run tests inside a secure subprocess sandbox?"
Yes! Add the `--sandbox` flag:
```bash
evolab repair --source app.py --pytest test_app.py --sandbox
```
All candidate mutations will execute in isolated subprocesses with timeout and memory enforcement.
