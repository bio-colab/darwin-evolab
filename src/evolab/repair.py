"""Composable repair genes: one edit per AST locus.

Identity is (file, lineno, col_offset). Crossover keeps at most one edit
per locus. Default code search is forward-greedy over the catalog.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

from .genome import EvolabGenome, Individual


def _payload(**kwargs: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(kwargs.items()))


# M8: bounded veto re-draws for trap-aware initialization — a catalog whose
# candidates are all memory-flagged dead doors must still terminate, so the
# last draw is accepted after this many vetoes.
_AVOID_MAX_REDRAWS = 3


@dataclass(frozen=True)
class RepairEdit:
    kind: str
    file: str
    lineno: int
    col_offset: int
    payload: tuple[tuple[str, Any], ...] = ()

    def locus(self) -> tuple[str, int, int]:
        return (self.file, self.lineno, self.col_offset)

    def key(self) -> tuple[str, int, int]:
        return self.locus()

    def payload_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "file": self.file,
            "lineno": self.lineno,
            "col_offset": self.col_offset,
            "payload": self.payload_dict(),
        }


_COMPARE_FLIP = {
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

_BOUNDARY_FLIP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
}

_NONE_CMP_FLIP = {
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}

_BINOP_FLIP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
}

_SEP_FLIP = {",": "&", "&": ",", " ": ","}

_PATTERN_REGISTRY: dict[str, dict[str, Any]] = {}


def register_repair_pattern(name: str, apply: Any | None = None):
    """Register a catalog finder (and optional apply handler) by kind name.

    Finder signature: ``(tree: ast.AST, file: str = "") -> list[RepairEdit]``.
    Apply signature: ``(node, edit, tree, parents) -> None``.
    """

    def decorator(fn):
        _PATTERN_REGISTRY[name] = {"find": fn, "apply": apply}
        fn.pattern_name = name
        return fn

    return decorator


def unregister_repair_pattern(name: str) -> None:
    _PATTERN_REGISTRY.pop(name, None)


def list_repair_patterns() -> list[str]:
    return list(_PATTERN_REGISTRY)


def _loc(node: ast.AST) -> tuple[int, int] | None:
    line = getattr(node, "lineno", None)
    col = getattr(node, "col_offset", None)
    if line is None or col is None:
        return None
    return (int(line), int(col))


def make_edit(kind: str, node: ast.AST, file: str = "", **payload: Any) -> RepairEdit | None:
    loc = _loc(node)
    if loc is None:
        return None
    return RepairEdit(
        kind=kind,
        file=file,
        lineno=loc[0],
        col_offset=loc[1],
        payload=_payload(**payload) if payload else (),
    )


def catalog_edits(source: str, file: str = "") -> list[RepairEdit]:
    """Edits from the live pattern registry."""
    tree = ast.parse(source)
    by_locus_kind: dict[tuple[tuple[str, int, int], str], RepairEdit] = {}
    for spec in _PATTERN_REGISTRY.values():
        finder = spec.get("find")
        if finder is None:
            continue
        for edit in finder(tree, file) or []:
            by_locus_kind.setdefault((edit.locus(), edit.kind), edit)
    return list(by_locus_kind.values())


def catalog_sources(sources: dict[str, str]) -> list[RepairEdit]:
    edits: list[RepairEdit] = []
    for path, source in sources.items():
        edits.extend(catalog_edits(source, file=path))
    return edits


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            out[id(child)] = parent
    return out


def _apply_bool_flip(node, edit, tree, parents) -> None:
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        node.value = not node.value


def _apply_string_sep(node, edit, tree, parents) -> None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = _SEP_FLIP.get(node.value, node.value)


def _apply_index_flip(node, edit, tree, parents) -> None:
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in (0, 1):
        node.slice.value = 1 - node.slice.value


def _apply_int_wrap(node, edit, tree, parents) -> None:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript):
        node.value = ast.Call(
            func=ast.Name(id="int", ctx=ast.Load()),
            args=[node.value],
            keywords=[],
        )


def _apply_pop_to_front(node, edit, tree, parents) -> None:
    if isinstance(node, ast.Call) and node.args:
        node.args[0] = ast.Constant(value=0)


def _apply_hit_move_to_end(node, edit, tree, parents) -> None:
    if not isinstance(node, ast.Pass):
        return
    parent = parents.get(id(node))
    if (
        isinstance(parent, ast.If)
        and parent.body
        and parent.body[0] is node
        and isinstance(parent.test, ast.Compare)
        and parent.test.ops
        and isinstance(parent.test.ops[0], ast.In)
    ):
        key_expr = parent.test.left
        seq = parent.test.comparators[0]
        parent.body = [
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=copy.deepcopy(seq), attr="remove", ctx=ast.Load()),
                    args=[copy.deepcopy(key_expr)],
                    keywords=[],
                )
            )
        ]


def _apply_compare_flip(node, edit, tree, parents) -> None:
    if isinstance(node, ast.Compare) and node.ops:
        op_t = type(node.ops[0])
        if op_t in _COMPARE_FLIP:
            node.ops[0] = _COMPARE_FLIP[op_t]()


def _apply_swap_int_args(node, edit, tree, parents) -> None:
    if not isinstance(node, ast.Call):
        return
    payload = edit.payload_dict()
    i, j = int(payload["i"]), int(payload["j"])
    if 0 <= i < len(node.args) and 0 <= j < len(node.args):
        node.args[i], node.args[j] = node.args[j], node.args[i]


def _apply_auth_prefix(node, edit, tree, parents) -> None:
    if isinstance(node, ast.JoinedStr):
        node.values = [
            ast.FormattedValue(value=ast.Name(id="auth_type", ctx=ast.Load()), conversion=-1),
            ast.Constant(value=" "),
            *node.values,
        ]


@register_repair_pattern("bool_flip", apply=_apply_bool_flip)
def find_bool_flip(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            edit = make_edit("bool_flip", node, file)
            if edit:
                out.append(edit)
    return out


@register_repair_pattern("string_sep", apply=_apply_string_sep)
def find_string_sep(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _SEP_FLIP:
            edit = make_edit("string_sep", node, file, from_value=node.value)
            if edit:
                out.append(edit)
    return out


@register_repair_pattern("index_flip", apply=_apply_index_flip)
def find_index_flip(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in (0, 1):
            edit = make_edit("index_flip", node, file)
            if edit:
                out.append(edit)
    return out


@register_repair_pattern("int_wrap", apply=_apply_int_wrap)
def find_int_wrap(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript):
            edit = make_edit("int_wrap", node, file)
            if edit:
                out.append(edit)
    return out


@register_repair_pattern("pop_to_front", apply=_apply_pop_to_front)
def find_pop_to_front(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pop" and node.args:
            edit = make_edit("pop_to_front", node, file)
            if edit:
                out.append(edit)
    return out


@register_repair_pattern("hit_move_to_end", apply=_apply_hit_move_to_end)
def find_hit_move_to_end(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            edit = make_edit("hit_move_to_end", node, file)
            if edit:
                out.append(edit)
    return out


@register_repair_pattern("compare_flip", apply=_apply_compare_flip)
def find_compare_flip(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and node.ops and type(node.ops[0]) in _COMPARE_FLIP:
            edit = make_edit("compare_flip", node, file)
            if edit:
                out.append(edit)
    return out


@register_repair_pattern("swap_int_args", apply=_apply_swap_int_args)
def find_swap_int_args(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        int_args = [
            i for i, a in enumerate(node.args)
            if isinstance(a, ast.Constant) and type(a.value) is int
        ]
        if len(int_args) >= 2:
            edit = make_edit("swap_int_args", node, file, i=int_args[-2], j=int_args[-1])
            if edit:
                out.append(edit)
    return out


@register_repair_pattern("auth_prefix", apply=_apply_auth_prefix)
def find_auth_prefix(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and node.values:
            edit = make_edit("auth_prefix", node, file)
            if edit:
                out.append(edit)
    return out


def _apply_logical_flip(node, edit, tree, parents) -> None:
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            node.op = ast.Or()
        elif isinstance(node.op, ast.Or):
            node.op = ast.And()


@register_repair_pattern("logical_flip", apply=_apply_logical_flip)
def find_logical_flip(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            edit = make_edit("logical_flip", node, file)
            if edit:
                out.append(edit)
    return out


def _apply_boundary_cmp(node, edit, tree, parents) -> None:
    if isinstance(node, ast.Compare) and node.ops:
        op_t = type(node.ops[0])
        if op_t in _BOUNDARY_FLIP:
            node.ops[0] = _BOUNDARY_FLIP[op_t]()


@register_repair_pattern("boundary_cmp", apply=_apply_boundary_cmp)
def find_boundary_cmp(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and node.ops and type(node.ops[0]) in _BOUNDARY_FLIP:
            edit = make_edit("boundary_cmp", node, file)
            if edit:
                out.append(edit)
    return out


def _apply_none_check_flip(node, edit, tree, parents) -> None:
    if isinstance(node, ast.Compare) and node.ops:
        op_t = type(node.ops[0])
        if op_t in _NONE_CMP_FLIP:
            node.ops[0] = _NONE_CMP_FLIP[op_t]()


@register_repair_pattern("none_check_flip", apply=_apply_none_check_flip)
def find_none_check_flip(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and node.ops
            and type(node.ops[0]) in _NONE_CMP_FLIP
            and node.comparators
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value is None
        ):
            edit = make_edit("none_check_flip", node, file)
            if edit:
                out.append(edit)
    return out


def _apply_binop_flip(node, edit, tree, parents) -> None:
    if isinstance(node, ast.BinOp):
        op_t = type(node.op)
        if op_t in _BINOP_FLIP:
            node.op = _BINOP_FLIP[op_t]()


@register_repair_pattern("binop_flip", apply=_apply_binop_flip)
def find_binop_flip(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOP_FLIP:
            edit = make_edit("binop_flip", node, file)
            if edit:
                out.append(edit)
    return out


def _apply_off_by_one(node, edit, tree, parents) -> None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        delta = int(edit.payload_dict().get("delta", 1))
        node.value += delta


@register_repair_pattern("off_by_one_inc", apply=_apply_off_by_one)
def find_off_by_one_inc(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            edit = make_edit("off_by_one_inc", node, file, delta=1)
            if edit:
                out.append(edit)
    return out


@register_repair_pattern("off_by_one_dec", apply=_apply_off_by_one)
def find_off_by_one_dec(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            edit = make_edit("off_by_one_dec", node, file, delta=-1)
            if edit:
                out.append(edit)
    return out


def _apply_unary_not_flip(node, edit, tree, parents) -> None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        parent = parents.get(id(node))
        if parent:
            for field_name, val in ast.iter_fields(parent):
                if val is node:
                    setattr(parent, field_name, node.operand)
                    return
                elif isinstance(val, list):
                    for idx, item in enumerate(val):
                        if item is node:
                            val[idx] = node.operand
                            return


@register_repair_pattern("unary_not_flip", apply=_apply_unary_not_flip)
def find_unary_not_flip(tree: ast.AST, file: str = "") -> list[RepairEdit]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            edit = make_edit("unary_not_flip", node, file)
            if edit:
                out.append(edit)
    return out


def apply_edits(source: str, edits: list[RepairEdit]) -> str:
    if not edits:
        return source
    tree = ast.parse(source)
    parents = _parents(tree)
    wanted = {(e.lineno, e.col_offset): e for e in edits}

    for node in list(ast.walk(tree)):
        loc = _loc(node)
        if loc is None or loc not in wanted:
            continue
        edit = wanted[loc]
        spec = _PATTERN_REGISTRY.get(edit.kind)
        handler = spec.get("apply") if spec else None
        if handler is not None:
            handler(node, edit, tree, parents)

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


@dataclass
class RepairGenome(EvolabGenome):
    sources: dict[str, str] = field(default_factory=dict)
    target_file: str = ""
    edits: list[RepairEdit] = field(default_factory=list)
    source: str = ""
    _applied_cache: dict[str, str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.sources and not self.source:
            self.source = self.sources.get(self.target_file, next(iter(self.sources.values()), ""))
        elif self.source and not self.sources:
            name = self.target_file or "<src>"
            self.sources = {name: self.source}
            self.target_file = name
        self._applied_cache = None

    def edit_keys(self) -> set[tuple]:
        return {e.key() for e in self.edits}

    def apply_to(self, base_sources: dict[str, str] | None = None) -> dict[str, str]:
        if self._applied_cache is not None and base_sources is None:
            return dict(self._applied_cache)
        base = dict(base_sources or self.sources)
        grouped: dict[str, list[RepairEdit]] = {}
        for edit in self.edits:
            grouped.setdefault(edit.file or self.target_file, []).append(edit)
        for path, group in grouped.items():
            if path in base:
                base[path] = apply_edits(base[path], group)
        if base_sources is None:
            self._applied_cache = dict(base)
        return base

    def to_code(self) -> str:
        applied = self.apply_to()
        return applied.get(self.target_file, self.source)

    def clone(self) -> RepairGenome:
        return RepairGenome(
            sources=dict(self.sources),
            target_file=self.target_file,
            edits=list(self.edits),
            source=self.source,
        )

    def fingerprint(self) -> str:
        raw = "|".join(
            f"{e.file}:{e.lineno}:{e.col_offset}:{e.kind}" for e in sorted(self.edits, key=lambda x: x.key())
        )
        return hashlib.sha256((self.source + "#" + raw).encode()).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, RepairGenome):
            return 1.0
        a, b = self.edit_keys(), other.edit_keys()
        union = a | b
        if not union:
            return 0.0
        return round(1.0 - len(a & b) / len(union), 6)

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "RepairGenome",
            "edits": [e.serialize() for e in self.edits],
            "code": self.to_code(),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "node_count": len(self.edits),
            "edit_count": len(self.edits),
            "kinds": ",".join(sorted({e.kind for e in self.edits})),
            "fingerprint": self.fingerprint(),
        }

    def __len__(self) -> int:
        return len(self.edits)

    def mutate(self, rng=None, **kwargs: Any) -> RepairGenome:
        catalog = catalog_sources(self.sources)
        owned = self.edit_keys()
        unused = [e for e in catalog if e.key() not in owned]
        if not unused:
            return self.clone()
        import random as _random
        rng = rng or _random.Random()
        smap = kwargs.get("suspicion_map")
        chosen = unused[rng.randrange(len(unused))]
        candidates = unused
        if smap is not None and hasattr(smap, "get_top_nodes"):
            hot_lines = {n.line_no for n in smap.get_top_nodes(top_k=8, min_score=0.0)}
            preferred = [e for e in unused if e.lineno in hot_lines]
            if preferred:
                candidates = preferred
                chosen = preferred[rng.randrange(len(preferred))]
        # Memory prior (Phase 2): soft per-kind weights from the experience
        # store. Absent / empty / broken -> weights None -> the draws above
        # are the exact original behavior (same RNG stream, same choice).
        # M7: the parent's edit kinds are passed as the sequence PREFIX —
        # the base prior ignores it; ExperienceSequencePrior conditions on
        # it ("after [A, B], which next kind succeeded?"). Unknown-prefix
        # handling lives inside the prior, never here.
        prior = kwargs.get("edit_prior")
        if prior is not None and candidates:
            try:
                weights = prior.kind_weights(
                    [e.kind for e in candidates],
                    prefix=[e.kind for e in self.edits],
                )
            except Exception:
                weights = None
            if weights:
                chosen = rng.choices(
                    candidates,
                    weights=[weights.get(e.kind, 1.0) for e in candidates],
                    k=1,
                )[0]
        # M8 — trap-aware initialization: negative genetic memory. The
        # avoid_loci set holds dead (file, lineno, col_offset, kind) doors
        # mined from single-edit failures (ExperienceStore.avoidance_set).
        # The veto applies to the FINAL choice (after the kind prior) and
        # re-draws uniformly from the SAME candidate pool that produced it
        # — memory never overrides SBFL narrowing, same rule as priors.
        # Bounded: after _AVOID_MAX_REDRAWS vetoes the last draw is kept
        # (a heavily-dead catalog must not deadlock initialization).
        # None / empty set -> the loop below never runs -> exact original
        # behavior, byte-for-byte RNG stream.
        avoid = kwargs.get("avoid_loci")
        if avoid:
            for _ in range(_AVOID_MAX_REDRAWS):
                if (chosen.file, chosen.lineno, chosen.col_offset, chosen.kind) not in avoid:
                    break
                chosen = candidates[rng.randrange(len(candidates))]
        return RepairGenome(
            sources=dict(self.sources),
            target_file=self.target_file,
            edits=self.edits + [chosen],
            source=self.source,
        )

    def crossover(self, other: EvolabGenome, rng=None, **kwargs: Any) -> RepairGenome:
        if not isinstance(other, RepairGenome):
            return self.clone()
        merged: dict[tuple, RepairEdit] = {e.locus(): e for e in self.edits}
        for edit in other.edits:
            merged.setdefault(edit.locus(), edit)
        return RepairGenome(
            sources=dict(self.sources),
            target_file=self.target_file,
            edits=list(merged.values()),
            source=self.source,
        )


def unified_source_diff(original: dict[str, str], repaired: dict[str, str]) -> str:
    import difflib

    chunks: list[str] = []
    for path in original:
        old = original[path].splitlines(keepends=True)
        new = repaired.get(path, original[path]).splitlines(keepends=True)
        if old == new:
            continue
        chunks.extend(
            difflib.unified_diff(old, new, fromfile=f"a/{path}", tofile=f"b/{path}")
        )
    return "".join(chunks)


def _score(evaluator: Any, genome: RepairGenome) -> tuple[float, bool | None]:
    result = evaluator.evaluate(genome)
    return float(result.score), result.passed_holdout


def greedy_repair(
    sources: dict[str, str],
    target_file: str,
    evaluator: Any,
    max_evals: int | None = None,
    prioritize_by_suspicion: bool = True,
) -> tuple[RepairGenome, list[dict[str, Any]], int]:
    """Forward greedy: add a gene only if it raises score and does not fail holdout."""
    catalog = catalog_sources(sources)
    current = RepairGenome(sources=dict(sources), target_file=target_file, edits=[])
    best_score, best_hold = _score(evaluator, current)
    history = [{
        "generation": 1,
        "best_fitness": best_score,
        "mean_fitness": best_score,
        "edits": 0,
    }]
    evaluations = 1
    taken = set(current.edit_keys())

    if prioritize_by_suspicion:
        suspicion_map = getattr(evaluator, "last_suspicion_map", None)
        if suspicion_map is not None and getattr(suspicion_map, "line_scores", None):
            def _sbfl_key(e: RepairEdit):
                score = suspicion_map.line_scores.get(e.lineno, 0.0)
                return (-score, e.file, e.lineno, e.col_offset, e.kind)
            catalog = sorted(catalog, key=_sbfl_key)

    improved = True
    gen = 1
    while improved:
        improved = False
        best_trial = None
        best_trial_score = best_score
        best_trial_hold = best_hold
        best_edit = None
        for edit in catalog:
            if edit.locus() in taken:
                continue
            trial = RepairGenome(
                sources=dict(sources),
                target_file=target_file,
                edits=current.edits + [edit],
            )
            score, hold = _score(evaluator, trial)
            evaluations += 1
            if max_evals is not None and evaluations >= max_evals:
                if best_trial is None:
                    return current, history, evaluations
                break
            if score <= best_score:
                continue
            if hold is False and best_hold is True:
                continue
            if score > best_trial_score or (score == best_trial_score and hold is True and best_trial_hold is not True):
                best_trial = trial
                best_trial_score = score
                best_trial_hold = hold
                best_edit = edit
        if best_trial is None or best_edit is None:
            break
        current = best_trial
        best_score = best_trial_score
        best_hold = best_trial_hold
        taken.add(best_edit.locus())
        gen += 1
        history.append({
            "generation": gen,
            "best_fitness": best_score,
            "mean_fitness": best_score,
            "edits": len(current.edits),
            "added": best_edit.kind,
        })
        improved = True
        if best_score >= 100.0 and best_hold is not False:
            break
    return current, history, evaluations


def greedy_run_report(
    sources: dict[str, str],
    target_file: str,
    evaluator: Any,
    scenario_name: str = "",
    max_evals: int | None = None,
    prioritize_by_suspicion: bool = True,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    genome, history, evaluations = greedy_repair(
        sources,
        target_file,
        evaluator,
        max_evals=max_evals,
        prioritize_by_suspicion=prioritize_by_suspicion,
    )
    score, hold = _score(evaluator, genome)
    evaluations += 1
    final_gen = history[-1]["generation"] if history else 1
    payload = {
        "id": f"gen_{final_gen:02d}_ind_00",
        "fitness": score,
        "species": "spec_code",
        "genome_size": len(genome.edits),
        "code": genome.to_code(),
        "edits": [e.serialize() for e in genome.edits],
    }
    if hold is not None:
        payload["passed_holdout"] = hold
    return {
        "total_generations": final_gen,
        "total_candidates_evaluated": final_gen,
        "best_individual": payload,
        "species_distribution": {"spec_code": 1},
        "early_stop_triggered": score >= 99.7 and hold is not False,
        "history": history,
        "config": {
            "population_size": 1,
            "mutation_rate": 0.0,
            "early_stop_fitness": 99.7,
            "seed": None,
            "search": "greedy_forward",
            "evaluations": evaluations,
            "genome": "code",
            "scenario": scenario_name,
            "evaluator": type(evaluator).__name__,
        },
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine_version": "evolab-engine/0.5.0",
        "schema_version": "report-schema/1",
    }
