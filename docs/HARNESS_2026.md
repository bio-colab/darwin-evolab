# Darwin-Evolab 2026 Loop & Harness Specification

## 1. Overview & 2026 Industry Alignment

In modern agentic software engineering and automated repair (Claude Code / Codex - Harness Books, 2026), the system architecture bifurcates cleanly into two core subsystems:
1. **The Harness:** Establishes the operating environment, capabilities, sandbox boundaries, baseline evaluation snapshots, and deterministic verification gates.
2. **The Loop:** Orchestrates the iterative optimization heartbeat—sampling candidate variations, scoring fitness, detecting stagnation, mutating strategically, and deciding termination.

Darwin-Evolab implements a production-grade 2026 Harness subsystem (`evolab.harness`) featuring:
- **Declarative Manifests (`loop.json`)**: Explicit `Goal -> Loop -> Target` specifications ensuring reproducibility and deterministic control.
- **Additive Baseline Verification Gates**: Evaluates strictly on candidate-introduced deltas ($\Delta_{\text{errors}} = E_{\text{cand}} \setminus E_{\text{base}} = \emptyset$), permitting pre-existing codebase debt while guaranteeing $0$ regressions ($PASS \to PASS = 100\%$) and AST purity.
- **Trajectory Distillation**: Compiles successful multi-step search trajectories into deterministic, reusable $O(1)$ `workflow.json` execution recipes, reducing subsequent execution latency to $< 1\,\text{ms}$.
- **LLM Context Window Governance**: Implements surgical AST-aware neighborhood pruning and strict session/window token ceilings.

---

## 2. Declarative Loop Manifest (`loop.json`)

The manifest defines the complete lifecycle contract adhering to the schema `https://evolab.dev/schemas/loop.v1.json`:

```json
{
  "$schema": "https://evolab.dev/schemas/loop.v1.json",
  "name": "requests_repair_loop",
  "version": "1.0.0",
  "goal": {
    "goal_type": "repair",
    "description": "Repair HTTP authentication header formatting",
    "target_file": "http_helpers.py",
    "sources": {
      "http_helpers.py": "def format_request(auth_type, token, params):\n..."
    },
    "fitness_target": 99.7
  },
  "loop": {
    "strategy": "greedy_ast_prior",
    "budget": {
      "max_evaluations": 32,
      "timeout_seconds": 10.0
    },
    "first_ascent": true,
    "plateau_breaking": true,
    "taboo_memory": true,
    "random_seed": 42
  },
  "target": {
    "verification_mode": "additive_baseline",
    "enforce_additive_type_check": true,
    "enforce_ast_purity": true,
    "distill_trajectory_on_success": true
  }
}
```

---

## 3. Additive Baseline Verification Gate

Legacy verification gates enforce absolute zero-error invariants, failing valid repairs when the surrounding legacy codebase contains pre-existing type debt or lint warnings. 

The **Additive Baseline Gate** (`AdditiveBaselineGate`) computes:
$$\Delta_{\text{errors}} = E_{\text{cand}} \setminus E_{\text{base}}$$

- **New Type Errors Allowed:** $0$ ($|\Delta_{\text{errors}}| = 0$).
- **Pre-existing Type Errors:** Permitted without blocking the repair.
- **Regressions ($PASS \to FAIL$):** Strictly $0$. Any candidate that breaks a pre-passing test is instantly rejected (`REJECT`).
- **Target Defect Resolution ($FAIL \to PASS$):** Must be $\ge 1$ or candidate fitness must reach $\ge 99.7\%$.
- **AST Purity:** Reject any syntax tree violating safety invariants (arbitrary network egress, eval injection).

---

## 4. Trajectory Distillation to Reusable $O(1)$ Workflows

When a repair succeeds through evolutionary search or meta-policy rollouts, the harness distills the winning trajectory into a deterministic workflow manifest (`workflow.json`):

- **Defect Signatures:** Records AST syntax context (e.g. `dict_access`, `boolean_logic`).
- **Minimal Hunk Actions:** Compiles discrete line substitution or AST replacement steps.
- **Deterministic Replay:** The `WorkflowExecutor` executes the workflow against matching codebases without re-running genetic search, achieving **$O(1)$ search complexity** and **$< 1\,\text{ms}$ execution latency** (saving 100% of search evaluations).

---

## 5. LLM Context Window Governance

For hybrid or LLM-augmented mutation backends, `LLMContextGovernor` enforces:
- **Context Window Ceilings:** Truncates prompt payloads before exceeding model token limits.
- **Surgical AST Pruning:** Preserves a configurable line radius around suspicious fault loci identified by Ochiai SBFL, replacing distant functions with semantic stub comments (`# Context Governor: omitted non-target AST block`).
- **Session Budget Tracking:** Tracks input and output tokens and halts requests when cumulative session limits are reached.

---

## 6. CLI Usage

```bash
# 1. Scaffold a declarative loop manifest
python run.py harness init-manifest --scenario requests_http_helper -o loop.json

# 2. Execute the deterministic harness
python run.py harness run --manifest loop.json --output report.json --save-workflow workflow.json

# 3. Replay distilled workflow in O(1) time
python run.py harness replay --workflow workflow.json --source target.py --apply
```

---

## 7. FastMCP Server Tools

AI coding assistants can leverage Darwin-Evolab's 2026 harness via MCP:
- `execute_harness_loop`: Executes declarative `loop.json` manifests.
- `verify_additive_baseline`: Evaluates code against additive baseline gates.
- `replay_workflow`: Instantly replays distilled workflows with $0$ search evaluations.
