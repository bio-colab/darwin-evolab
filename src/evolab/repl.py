"""src/evolab/repl.py — Interactive State REPL Kernel for Evolab.

Provides a deterministic interactive command loop and scriptable execution engine
connected directly to the session manager, evaluation oracles, and evidence capsules.
"""
from __future__ import annotations

import ast
import cmd
import json
import shlex
import sys
from pathlib import Path
from typing import Any, TextIO

from .capsule import create_evidence_capsule, verify_evidence_capsule
from .engine import EvolutionEngine
from .session import SessionManager
from .vectorized import VectorizedLandscapeEvaluator


class ReplKernel(cmd.Cmd):
    """Interactive command kernel backed by stateful session engine."""

    prompt = "evolab> "

    def __init__(
        self,
        session: SessionManager | None = None,
        stdout: TextIO | None = None,
        stdin: TextIO | None = None,
        output_json: bool = False,
        silent: bool = False,
    ) -> None:
        super().__init__(stdin=stdin, stdout=stdout)
        self.session = session or SessionManager()
        self.output_json = output_json
        self.silent = silent
        self.last_output: dict[str, Any] = {}

    def _emit(self, data: dict[str, Any], text_msg: str = "") -> None:
        self.last_output = data
        if self.silent:
            return
        if self.output_json:
            self.stdout.write(json.dumps(data, indent=2) + "\n")
            self.stdout.flush()
        else:
            if text_msg:
                self.stdout.write(text_msg + "\n")
            self.stdout.flush()

    def default(self, line: str) -> bool:
        self._emit(
            {"status": "error", "message": f"Unknown command: {line.split()[0]}"},
            f"error: unknown command '{line.split()[0]}'. Type 'help' for available commands.",
        )
        return False

    def emptyline(self) -> bool:
        return False

    def do_help(self, arg: str) -> bool:
        """Display help information on available interactive commands."""
        commands = {
            "status": "Show current session parameters and transaction stack",
            "set": "Set session parameter: set <key> <value>",
            "get": "Get session parameter: get <key>",
            "eval": "Evaluate candidate vector: eval [1.0, 0.0]",
            "run": "Run evolutionary search with current session config: run [generations]",
            "undo": "Rollback last parameter change",
            "redo": "Reapply undone parameter change",
            "reset": "Reset parameters to defaults",
            "save": "Save session to file: save [path]",
            "load": "Load session from file: load [path]",
            "bundle": "Create evidence capsule: bundle <report.json> [output.evobundle]",
            "verify": "Verify evidence capsule: verify <bundle.evobundle> [key]",
            "exit / quit": "Exit interactive session",
        }
        if self.output_json:
            self._emit({"status": "success", "commands": commands})
        else:
            self.stdout.write("\nEvolab Interactive Commands:\n")
            for cmd_name, desc in sorted(commands.items()):
                self.stdout.write(f"  {cmd_name:<14} : {desc}\n")
            self.stdout.write("\n")
            self.stdout.flush()
        return False

    def do_status(self, arg: str) -> bool:
        """Inspect current session state and parameters."""
        data = self.session.to_dict()
        lines = [
            "Current Session Parameters:",
            *(f"  {k:<20}: {v}" for k, v in sorted(data["params"].items())),
            f"\nTransactions: undo={data['undo_available']}, redo={data['redo_available']}, runs={data['history_count']}",
        ]
        self._emit({"status": "success", "session": data}, "\n".join(lines))
        return False

    def do_set(self, arg: str) -> bool:
        """Set a session parameter: set <param> <value>"""
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            self._emit({"status": "error", "message": "Usage: set <param> <value>"}, "error: usage: set <param> <value>")
            return False

        key, raw_val = parts[0], parts[1]
        try:
            val = ast.literal_eval(raw_val)
        except Exception:
            val = raw_val

        self.session.set_param(key, val)
        self._emit(
            {"status": "success", "param": key, "value": val},
            f"Updated {key} = {val}",
        )
        return False

    def do_get(self, arg: str) -> bool:
        """Get a session parameter: get <param>"""
        key = arg.strip()
        if not key:
            self._emit({"status": "error", "message": "Usage: get <param>"}, "error: usage: get <param>")
            return False
        val = self.session.get_param(key)
        self._emit({"status": "success", "param": key, "value": val}, f"{key} = {val}")
        return False

    def do_undo(self, arg: str) -> bool:
        """Rollback last parameter change."""
        success = self.session.undo()
        if success:
            self._emit({"status": "success", "message": "Reverted to previous state", "params": self.session.params}, "Reverted to previous state.")
        else:
            self._emit({"status": "error", "message": "Nothing to undo"}, "warning: nothing to undo.")
        return False

    def do_redo(self, arg: str) -> bool:
        """Reapply previously undone parameter change."""
        success = self.session.redo()
        if success:
            self._emit({"status": "success", "message": "Reapplied parameter state", "params": self.session.params}, "Reapplied undone state.")
        else:
            self._emit({"status": "error", "message": "Nothing to redo"}, "warning: nothing to redo.")
        return False

    def do_reset(self, arg: str) -> bool:
        """Reset session to default parameters."""
        self.session.reset()
        self._emit({"status": "success", "message": "Session reset to defaults"}, "Session reset to defaults.")
        return False

    def do_save(self, arg: str) -> bool:
        """Save session state to file."""
        path = arg.strip() or None
        target = self.session.save(path)
        self._emit({"status": "success", "path": str(target)}, f"Session saved to: {target}")
        return False

    def do_load(self, arg: str) -> bool:
        """Load session state from file."""
        path = arg.strip() or None
        ok = self.session.load(path)
        if ok:
            self._emit({"status": "success", "message": "Session loaded successfully"}, "Session loaded successfully.")
        else:
            self._emit({"status": "error", "message": "Failed to load session file"}, "error: failed to load session file.")
        return False

    def do_eval(self, arg: str) -> bool:
        """Evaluate candidate coordinates via Fitness Oracle: eval [0.0, 0.0]"""
        candidate_str = arg.strip()
        if not candidate_str:
            self._emit({"status": "error", "message": "Usage: eval <candidate>"}, "error: usage: eval <candidate>")
            return False

        try:
            candidate = ast.literal_eval(candidate_str)
        except Exception:
            candidate = candidate_str

        landscape = self.session.get_param("landscape", "rastrigin")
        target = float(self.session.get_param("target_fitness", 100.0))

        try:
            ev = VectorizedLandscapeEvaluator(landscape=landscape, target_score=target)
            res = ev.evaluate(candidate)
            data = {
                "status": "success",
                "score": res.score,
                "sub_scores": res.sub_scores,
                "artifacts": res.artifacts,
            }
            self._emit(data, f"Candidate Fitness: {res.score:.4f} (landscape={landscape})")
        except Exception as exc:
            self._emit({"status": "error", "message": str(exc)}, f"error: evaluation failed: {exc}")
        return False

    def do_run(self, arg: str) -> bool:
        """Execute evolutionary optimization with current session parameters."""
        gens = int(arg.strip()) if arg.strip().isdigit() else int(self.session.get_param("generations", 20))
        pop = int(self.session.get_param("population_size", 16))
        seed = int(self.session.get_param("seed", 42))

        try:
            engine = EvolutionEngine(population_size=pop, seed=seed)
            res = engine.run(generations=gens)
            self.session.record_run(res)
            best_fit = res.get("best_individual", {}).get("fitness", 0.0)
            data = {
                "status": "success",
                "generations": gens,
                "best_fitness": best_fit,
                "report_summary": {
                    "total_generations": res.get("total_generations"),
                    "best_fitness": best_fit,
                },
            }
            self._emit(data, f"Run completed: generations={gens}, best_fitness={best_fit:.4f}")
        except Exception as exc:
            self._emit({"status": "error", "message": str(exc)}, f"error: run failed: {exc}")
        return False

    def do_bundle(self, arg: str) -> bool:
        """Create evidence capsule: bundle <report.json> [output.evobundle]"""
        parts = shlex.split(arg)
        if not parts:
            self._emit({"status": "error", "message": "Usage: bundle <report.json> [output.evobundle]"}, "error: usage: bundle <report.json> [output.evobundle]")
            return False

        rep_path = parts[0]
        out_path = parts[1] if len(parts) > 1 else str(Path(rep_path).with_suffix(".evobundle"))
        try:
            p = create_evidence_capsule(rep_path, out_path)
            self._emit({"status": "success", "bundle": str(p)}, f"Capsule created: {p}")
        except Exception as exc:
            self._emit({"status": "error", "message": str(exc)}, f"error: bundle creation failed: {exc}")
        return False

    def do_verify(self, arg: str) -> bool:
        """Verify evidence capsule: verify <bundle.evobundle> [key]"""
        parts = shlex.split(arg)
        if not parts:
            self._emit({"status": "error", "message": "Usage: verify <bundle.evobundle> [key]"}, "error: usage: verify <bundle.evobundle> [key]")
            return False

        bundle_path = parts[0]
        key = parts[1] if len(parts) > 1 else None
        res = verify_evidence_capsule(bundle_path, secret_key=key)
        self._emit(res, f"Verification: {res.get('status')} - {res.get('message')}")
        return False

    def do_exit(self, arg: str) -> bool:
        """Exit the interactive session."""
        return True

    def do_quit(self, arg: str) -> bool:
        """Exit the interactive session."""
        return True

    def execute_line(self, line: str) -> dict[str, Any]:
        """Execute a single command string and return structured result."""
        line = line.strip()
        if not line:
            return {"status": "empty"}
        self.onecmd(line)
        return self.last_output

    def execute_script(self, script: str) -> list[dict[str, Any]]:
        """Execute multiple semicolon-separated or newline-separated commands."""
        results = []
        # split by semicolon or newlines
        commands = [c.strip() for c in script.replace("\n", ";").split(";") if c.strip()]
        for cmd_str in commands:
            res = self.execute_line(cmd_str)
            results.append(res)
        return results
