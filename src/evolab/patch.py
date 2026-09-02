"""PatchGenome implementation, code hunk data structures, and patch distance/mutations."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .genome import EvolabGenome


@dataclass
class Hunk:
    """A single change segment within a specific source file."""

    file_path: str          # relative file path in target project
    start_line: int         # 0-indexed start line in original file
    num_lines: int          # number of lines replaced/deleted (0 for insertion)
    old_text: str           # original text segment (empty for insertion)
    new_text: str           # replacement text segment (empty for deletion)

    def serialize(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "num_lines": self.num_lines,
            "old_text": self.old_text,
            "new_text": self.new_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hunk:
        return cls(
            file_path=data["file_path"],
            start_line=int(data["start_line"]),
            num_lines=int(data["num_lines"]),
            old_text=str(data.get("old_text", "")),
            new_text=str(data.get("new_text", "")),
        )


class PatchApplyError(Exception):
    """Raised when a patch cannot be applied to target source."""


def apply_patch(
    sources: dict[str, str], patch: PatchGenome
) -> dict[str, str]:
    """Applies patch hunks to a dictionary of source files {file_path: code_str}.

    Hunks per file are sorted in descending start_line order to preserve line
    indexing during multi-hunk applications.
    """
    result = {k: v for k, v in sources.items()}

    # Group hunks by file
    hunks_by_file: dict[str, list[Hunk]] = {}
    for h in patch.hunks:
        hunks_by_file.setdefault(h.file_path, []).append(h)

    for file_path, hunks in hunks_by_file.items():
        if file_path not in result:
            # File creation if old_text is empty and start_line == 0
            file_lines = []
        else:
            file_lines = result[file_path].splitlines(keepends=True)

        # Sort descending by start_line
        sorted_hunks = sorted(hunks, key=lambda h: h.start_line, reverse=True)

        for hunk in sorted_hunks:
            sl = hunk.start_line
            nl = hunk.num_lines
            if sl < 0 or sl > len(file_lines):
                raise PatchApplyError(
                    f"Invalid start_line {sl} for file {file_path} with {len(file_lines)} lines"
                )

            # Format new text into lines
            if hunk.new_text:
                new_lines = hunk.new_text.splitlines(keepends=True)
                # Ensure trailing newline matches surrounding context
                if new_lines and not new_lines[-1].endswith("\n") and (sl < len(file_lines) or file_lines):
                    new_lines[-1] = new_lines[-1] + "\n"
            else:
                new_lines = []

            # Replace lines in file_lines
            file_lines[sl : sl + nl] = new_lines

        result[file_path] = "".join(file_lines)

    return result


def create_patch_from_diff(
    file_path: str, old_code: str, new_code: str
) -> PatchGenome:
    """Creates a PatchGenome with hunks from comparing old_code and new_code."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    hunks: list[Hunk] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_segment = "".join(old_lines[i1:i2])
        new_segment = "".join(new_lines[j1:j2])
        hunks.append(
            Hunk(
                file_path=file_path,
                start_line=i1,
                num_lines=i2 - i1,
                old_text=old_segment,
                new_text=new_segment,
            )
        )
    return PatchGenome(hunks=hunks)


def patch_distance(
    a: PatchGenome, b: PatchGenome, w1: float = 0.4, w2: float = 0.6
) -> float:
    """Computes a normalized distance [0.0, 1.0] between two PatchGenomes.

    w1: file overlap Jaccard distance weight
    w2: hunk content / sequence distance weight
    """
    if a.fingerprint() == b.fingerprint():
        return 0.0

    files_a = set(h.file_path for h in a.hunks)
    files_b = set(h.file_path for h in b.hunks)

    if not files_a and not files_b:
        return 0.0

    union_files = files_a | files_b
    intersect_files = files_a & files_b
    jaccard_dist = 1.0 - (len(intersect_files) / len(union_files))

    # Content similarity across hunks (canonical order guarantees exact mathematical symmetry)
    text_a = "\n".join(f"{h.file_path}:{h.start_line}:{h.new_text}" for h in a.hunks)
    text_b = "\n".join(f"{h.file_path}:{h.start_line}:{h.new_text}" for h in b.hunks)

    if not text_a and not text_b:
        content_dist = 0.0
    elif not text_a or not text_b:
        content_dist = 1.0
    else:
        t1, t2 = (text_a, text_b) if text_a <= text_b else (text_b, text_a)
        matcher = difflib.SequenceMatcher(None, t1, t2)
        content_dist = 1.0 - matcher.ratio()

    total = w1 * jaccard_dist + w2 * content_dist
    return max(0.0, min(1.0, total))


@dataclass
class PatchGenome(EvolabGenome):
    """Genome representing a collection of code modification hunks across files."""

    hunks: list[Hunk] = field(default_factory=list)

    def clone(self) -> PatchGenome:
        return PatchGenome(
            hunks=[
                Hunk(
                    file_path=h.file_path,
                    start_line=h.start_line,
                    num_lines=h.num_lines,
                    old_text=h.old_text,
                    new_text=h.new_text,
                )
                for h in self.hunks
            ]
        )

    def fingerprint(self) -> str:
        sorted_hunks = sorted(
            [h.serialize() for h in self.hunks],
            key=lambda h: (h["file_path"], h["start_line"], h["num_lines"], h["new_text"]),
        )
        raw = json.dumps(sorted_hunks, sort_keys=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, PatchGenome):
            raise TypeError(f"Cannot compute distance to non-PatchGenome: {type(other)}")
        return patch_distance(self, other)

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "PatchGenome",
            "fingerprint": self.fingerprint(),
            "hunks": [h.serialize() for h in self.hunks],
        }

    def describe(self) -> dict[str, Any]:
        files = set(h.file_path for h in self.hunks)
        total_added = sum(
            len(h.new_text.splitlines()) if h.new_text else 0 for h in self.hunks
        )
        total_removed = sum(h.num_lines for h in self.hunks)
        return {
            "hunk_count": len(self.hunks),
            "files_count": len(files),
            "lines_added": total_added,
            "lines_removed": total_removed,
        }

    def __len__(self) -> int:
        return len(self.hunks)

    def apply_to(self, sources: dict[str, str]) -> dict[str, str]:
        return apply_patch(sources, self)

    def mutate(
        self,
        rng: random.Random | None = None,
        sources: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> PatchGenome:
        r = rng or random
        if sources:
            return mutate_patch(self, sources, rng=r)
        if not self.hunks:
            return self.clone()
        clone = self.clone()
        idx = r.randrange(len(clone.hunks))
        h = clone.hunks[idx]
        mutated_text = mutate_constant_in_code(h.new_text, r)
        if mutated_text is not None:
            clone.hunks[idx] = Hunk(
                file_path=h.file_path,
                start_line=h.start_line,
                num_lines=h.num_lines,
                old_text=h.old_text,
                new_text=mutated_text,
            )
        return clone

    def crossover(self, other: EvolabGenome, rng: random.Random | None = None) -> PatchGenome:
        if not isinstance(other, PatchGenome):
            return self.clone()
        r = rng or random
        all_hunks = self.hunks + other.hunks
        if not all_hunks:
            return PatchGenome()
        selected = [h for h in all_hunks if r.random() < 0.5]
        return PatchGenome(hunks=selected if selected else [r.choice(all_hunks)])


# ---------------------------------------------------------------------------
# AST-Aware and Deterministic Code Mutation Engine
# ---------------------------------------------------------------------------

def mutate_constant_in_code(code: str, rng: random.Random) -> str | None:
    """Finds numeric/string constants, booleans, or operators in code and mutates one."""
    # Find boolean constants
    bool_matches = list(re.finditer(r"\b(True|False)\b", code))
    if bool_matches and rng.random() < 0.4:
        m = rng.choice(bool_matches)
        repl = "False" if m.group(0) == "True" else "True"
        return code[: m.start()] + repl + code[m.end() :]

    # Find string constants
    str_matches = list(re.finditer(r"['\"]([^'\"]*)['\"]", code))
    if str_matches and rng.random() < 0.4:
        m = rng.choice(str_matches)
        val = m.group(1)
        if val == ",":
            repl = "'&'"
        elif val == "&":
            repl = "','"
        else:
            repl = f"'{val}'"
        return code[: m.start()] + repl + code[m.end() :]

    # Find numeric constants
    num_matches = list(re.finditer(r"\b\d+(\.\d+)?\b", code))
    if num_matches and rng.random() < 0.6:
        m = rng.choice(num_matches)
        old_val = float(m.group(0)) if "." in m.group(0) else int(m.group(0))
        delta = rng.choice([-1, 1, -2, 2, 0.5, -0.5])
        new_val = type(old_val)(old_val + delta)
        return code[: m.start()] + str(new_val) + code[m.end() :]

    # Find comparison and arithmetic operators
    op_pairs = {
        "==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<",
        "+": "-", "-": "+", "*": "//", "//": "*", "and": "or", "or": "and",
    }
    for op, repl in op_pairs.items():
        pattern = rf"(?<=\s){re.escape(op)}(?=\s)"
        matches = list(re.finditer(pattern, code))
        if matches and rng.random() < 0.5:
            m = rng.choice(matches)
            return code[: m.start()] + repl + code[m.end() :]

    return None


def mutate_patch(
    patch: PatchGenome,
    base_sources: dict[str, str],
    rng: random.Random | None = None,
    max_attempts: int = 8,
) -> PatchGenome:
    """Applies a non-lethal, syntactically valid mutation to a PatchGenome.

    Rejects lethal mutations (SyntaxError upon compilation).
    """
    rng = rng or random.Random()
    if not base_sources:
        return patch.clone()

    file_path = rng.choice(list(base_sources.keys()))
    original_code = base_sources[file_path]
    current_applied = patch.apply_to(base_sources)
    current_code = current_applied.get(file_path, original_code)

    for _ in range(max_attempts):
        mutated_code: str | None = None
        mode = rng.choice(["modify_constant", "insert_statement", "delete_line", "tweak_hunk"])

        lines = current_code.splitlines(keepends=True)
        if not lines:
            continue

        if mode == "modify_constant":
            mutated_code = mutate_constant_in_code(current_code, rng)

        elif mode == "insert_statement" and len(lines) > 0:
            target_idx = rng.randint(0, len(lines))
            # Detect indentation
            prev_indent = ""
            if 0 <= target_idx < len(lines):
                match = re.match(r"^(\s*)", lines[target_idx])
                if match:
                    prev_indent = match.group(1)
            candidate_stmts = [
                f"{prev_indent}# evolab exploration\n",
                f"{prev_indent}pass\n",
            ]
            stmt = rng.choice(candidate_stmts)
            new_lines = list(lines)
            new_lines.insert(target_idx, stmt)
            mutated_code = "".join(new_lines)

        elif mode == "delete_line" and len(lines) > 2:
            del_idx = rng.randint(0, len(lines) - 1)
            # Avoid deleting function signatures or class defs
            if not lines[del_idx].strip().startswith(("def ", "class ", "import ", "from ")):
                new_lines = list(lines)
                new_lines.pop(del_idx)
                mutated_code = "".join(new_lines)

        elif mode == "tweak_hunk" and patch.hunks:
            hunk_copy = list(patch.hunks)
            del_hunk_idx = rng.randint(0, len(hunk_copy) - 1)
            hunk_copy.pop(del_hunk_idx)
            test_patch = PatchGenome(hunks=hunk_copy)
            try:
                test_sources = test_patch.apply_to(base_sources)
                compile(test_sources[file_path], file_path, "exec")
                return test_patch
            except Exception:
                continue

        if mutated_code is not None:
            # Check for syntax errors (lethal mutation gate)
            try:
                ast.parse(mutated_code)
                compile(mutated_code, file_path, "exec")
                # Valid non-lethal code! Build new PatchGenome
                new_patch = create_patch_from_diff(file_path, original_code, mutated_code)
                return new_patch
            except (SyntaxError, IndentationError):
                # Lethal mutation rejected
                continue

    return patch.clone()


def mutate_multi_hunk(
    patch: PatchGenome,
    base_sources: dict[str, str],
    rng: random.Random | None = None,
    num_mutations: int = 2,
) -> PatchGenome:
    """Performs coordinated compound mutations across multiple locations in the code."""
    rng = rng or random.Random()
    current_patch = patch.clone()

    for _ in range(num_mutations):
        mutated = mutate_patch(current_patch, base_sources, rng=rng)
        if mutated.fingerprint() != current_patch.fingerprint():
            current_patch = mutated

    return current_patch


# Re-export AST-Anchored Patch Contract

