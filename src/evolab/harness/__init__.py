"""evolab.harness — Modern 2026 Loop & Harness Engineering Subsystem.

Conforms to modern 2026 Agent Harness standards (Harness Books / Claude Code):
- Declarative Loop Manifests: Goal -> Loop -> Target (loop.json)
- Additive Baseline Scoped Verification Gates: Zero regressions, delta type-checking
- Evolutionary Trajectory Distillation: Reusable O(1) workflow.json recipes
- LLM Context Window Governance & Surgical AST Pruning
- Deterministic Harness Execution Engine
"""

from .context_governor import (
    ContextBudget,
    ContextGovernanceTelemetry,
    LLMContextGovernor,
)
from .distillation import (
    WorkflowAction,
    WorkflowExecutor,
    WorkflowManifest,
    distill_trajectory_to_workflow,
)
from .manifest import (
    GoalSpec,
    GovernorSpec,
    LoopBudget,
    LoopManifest,
    LoopSpec,
    TargetSpec,
)
from .runner import (
    DeterministicHarness,
    HarnessExecutionReport,
)
from .verify import (
    AdditiveBaselineGate,
    AdditiveVerificationReport,
    BaselineSnapshot,
)

__all__ = [
    # Manifest
    "GoalSpec",
    "LoopBudget",
    "GovernorSpec",
    "LoopSpec",
    "TargetSpec",
    "LoopManifest",
    # Verification
    "BaselineSnapshot",
    "AdditiveVerificationReport",
    "AdditiveBaselineGate",
    # Distillation
    "WorkflowAction",
    "WorkflowManifest",
    "distill_trajectory_to_workflow",
    "WorkflowExecutor",
    # Context Governor
    "ContextBudget",
    "ContextGovernanceTelemetry",
    "LLMContextGovernor",
    # Runner
    "HarnessExecutionReport",
    "DeterministicHarness",
]
