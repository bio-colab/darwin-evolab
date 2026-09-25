"""distillation.py — Evolutionary Trajectory Distillation to Deterministic Workflow.

Implements the 2026 Modern Harness Standard for trajectory compilation:
- Transforms successful evolutionary search trajectories into deterministic, reusable `workflow.json` recipes.
- Enables instant O(1) replay: eliminates search overhead for known defect archetypes.
- Records defect syntax signatures, minimal surgical edits, and post-condition verification gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import difflib
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

from evolab.dream.recursive_rsi import extract_syntax_context
from evolab.evaluators import Evaluator
from evolab.patch import PatchGenome, apply_patch, create_patch_from_diff


@dataclass
class WorkflowAction:
    """Atomic deterministic modification step in a distilled workflow."""

    action_type: str  # "replace_content" | "ast_patch" | "line_substitute"
    target_file: str
    target_pattern: str
    replacement: str
    lineno_hint: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowAction:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkflowManifest:
    """Declarative distilled workflow recipe (workflow.json)."""

    workflow_id: str
    name: str
    target_file: str
    defect_signatures: list[str] = field(default_factory=list)
    actions: list[WorkflowAction] = field(default_factory=list)
    pre_condition_snippet: str | None = None
    expected_patch_diff: str | None = None
    search_evaluations_saved: int = 0
    created_at_utc: str = ""
    replay_complexity: str = "O(1) Deterministic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "target_file": self.target_file,
            "defect_signatures": self.defect_signatures,
            "actions": [a.to_dict() for a in self.actions],
            "pre_condition_snippet": self.pre_condition_snippet,
            "expected_patch_diff": self.expected_patch_diff,
            "search_evaluations_saved": self.search_evaluations_saved,
            "created_at_utc": self.created_at_utc,
            "replay_complexity": self.replay_complexity,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowManifest:
        actions_data = data.get("actions", [])
        actions = [WorkflowAction.from_dict(a) for a in actions_data]
        return cls(
            workflow_id=data.get("workflow_id", ""),
            name=data.get("name", "distilled_workflow"),
            target_file=data.get("target_file", "solution.py"),
            defect_signatures=data.get("defect_signatures", []),
            actions=actions,
            pre_condition_snippet=data.get("pre_condition_snippet"),
            expected_patch_diff=data.get("expected_patch_diff"),
            search_evaluations_saved=data.get("search_evaluations_saved", 0),
            created_at_utc=data.get("created_at_utc", ""),
            replay_complexity=data.get("replay_complexity", "O(1) Deterministic"),
        )

    @classmethod
    def load(cls, path: Path | str) -> WorkflowManifest:
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))



def compute_unified_diff(file_path: str, old_code: str, new_code: str) -> str:
    """Computes standard unified diff between two code strings."""
    return "".join(
        difflib.unified_diff(
            old_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
    )


def distill_trajectory_to_workflow(
    initial_sources: dict[str, str],
    fixed_sources: dict[str, str],
    target_file: str,
    workflow_name: str = "Distilled Repair Workflow",
    search_evaluations_consumed: int = 0,
) -> WorkflowManifest:
    """Compiles a successful search trajectory into a deterministic workflow recipe."""
    orig_code = initial_sources.get(target_file, "")
    fixed_code = fixed_sources.get(target_file, "")

    # Extract syntax signatures
    signatures = sorted(list(extract_syntax_context(orig_code)))

    # Compute minimal diff and patch
    patch = create_patch_from_diff(target_file, orig_code, fixed_code)
    diff_text = compute_unified_diff(target_file, orig_code, fixed_code)

    actions: list[WorkflowAction] = []
    for hunk in patch.hunks:
        actions.append(
            WorkflowAction(
                action_type="line_substitute",
                target_file=target_file,
                target_pattern=hunk.old_text,
                replacement=hunk.new_text,
                lineno_hint=hunk.start_line,
            )
        )

    wf_hash = hashlib.sha256(f"{target_file}:{diff_text}".encode("utf-8")).hexdigest()[:12]
    wf_id = f"wf_{Path(target_file).stem}_{wf_hash}"

    return WorkflowManifest(
        workflow_id=wf_id,
        name=workflow_name,
        target_file=target_file,
        defect_signatures=signatures,
        actions=actions,
        expected_patch_diff=diff_text,
        search_evaluations_saved=search_evaluations_consumed,
        created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


class WorkflowExecutor:
    """Executes a distilled workflow recipe deterministically in O(1) time without search."""

    @staticmethod
    def apply(sources: dict[str, str], workflow: WorkflowManifest) -> dict[str, str]:
        """Applies workflow transformations deterministically to source files."""
        result_sources = dict(sources)
        target_code = result_sources.get(workflow.target_file, "")

        for action in workflow.actions:
            if action.action_type == "line_substitute":
                if action.target_pattern in target_code:
                    target_code = target_code.replace(action.target_pattern, action.replacement, 1)

        result_sources[workflow.target_file] = target_code
        return result_sources

    @staticmethod
    def replay_and_verify(
        sources: dict[str, str],
        workflow: WorkflowManifest,
        evaluator: Evaluator,
    ) -> tuple[bool, float, dict[str, str]]:
        """Replays distilled workflow and verifies solution score with zero genetic search."""
        t0 = time.perf_counter()
        repaired_sources = WorkflowExecutor.apply(sources, workflow)

        # Directly evaluate repaired code
        from evolab.patch import PatchGenome, create_patch_from_diff
        orig_code = sources.get(workflow.target_file, "")
        fixed_code = repaired_sources.get(workflow.target_file, "")
        patch = create_patch_from_diff(workflow.target_file, orig_code, fixed_code)

        res = evaluator.evaluate(patch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        is_success = bool(res.score >= 99.7)
        return is_success, elapsed_ms, repaired_sources
