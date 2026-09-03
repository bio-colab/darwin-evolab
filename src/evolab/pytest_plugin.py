"""pytest plugin: suggest a repair when a marked or --evolab test fails.

Only literal asserts of the form ``assert func(args) == expected`` are harvested.
The plugin is inert unless ``--evolab`` is passed or a test is marked
``@pytest.mark.evolab(func=..., source=...)``.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("evolab")
    group.addoption(
        "--evolab",
        action="store_true",
        default=False,
        help="after failures, run greedy repair and print a unified diff",
    )
    group.addoption("--evolab-func", default=None, help="target function name")
    group.addoption("--evolab-source", default=None, help="target source file")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "evolab(func=None, source=None): collect literal asserts for evolab repair",
    )
    config._evolab_jobs = {}  # type: ignore[attr-defined]


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def extract_literal_cases(source: str, test_name: str, func_name: str) -> list[Any]:
    tree = ast.parse(source)
    cases: list[Any] = []
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name != test_name:
            continue
        for node in ast.walk(fn):
            # Support: with pytest.raises(Exc): func(...)
            if isinstance(node, ast.With):
                for item in node.items:
                    ctx = item.context_expr
                    if isinstance(ctx, ast.Call) and _call_name(ctx.func) == "raises":
                        exc_name = "Exception"
                        if ctx.args:
                            if isinstance(ctx.args[0], ast.Name):
                                exc_name = ctx.args[0].id
                            elif isinstance(ctx.args[0], ast.Attribute):
                                exc_name = ctx.args[0].attr
                        for body_node in ast.walk(node):
                            if isinstance(body_node, ast.Call) and _call_name(body_node.func) == func_name:
                                try:
                                    args = tuple(ast.literal_eval(a) for a in body_node.args)
                                    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in body_node.keywords if kw.arg is not None}
                                    case_args = (args, kwargs) if kwargs else args
                                    cases.append((case_args, f"raises:{exc_name}"))
                                except Exception:
                                    continue

            if not isinstance(node, ast.Assert):
                continue

            # Support: assert func(...)
            if isinstance(node.test, ast.Call) and _call_name(node.test.func) == func_name:
                try:
                    args = tuple(ast.literal_eval(a) for a in node.test.args)
                    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.test.keywords if kw.arg is not None}
                    case_args = (args, kwargs) if kwargs else args
                    cases.append((case_args, True))
                except Exception:
                    continue
            # Support: assert not func(...)
            elif (
                isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Call)
                and _call_name(node.test.operand.func) == func_name
            ):
                try:
                    args = tuple(ast.literal_eval(a) for a in node.test.operand.args)
                    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.test.operand.keywords if kw.arg is not None}
                    case_args = (args, kwargs) if kwargs else args
                    cases.append((case_args, False))
                except Exception:
                    continue

            # Support: assert func(...) == expected / is ...
            if not isinstance(node.test, ast.Compare):
                continue
            cmp = node.test
            if not cmp.ops or not isinstance(cmp.left, ast.Call):
                continue
            if _call_name(cmp.left.func) != func_name:
                continue

            try:
                args = tuple(ast.literal_eval(a) for a in cmp.left.args)
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in cmp.left.keywords if kw.arg is not None}
                case_args = (args, kwargs) if kwargs else args
                op = cmp.ops[0]
                expected = ast.literal_eval(cmp.comparators[0])
                if isinstance(op, (ast.Eq, ast.Is)):
                    cases.append((case_args, expected))
            except Exception:
                continue

    return cases


def _marker_opts(item: pytest.Item) -> dict[str, Any]:
    marker = item.get_closest_marker("evolab")
    if marker is None:
        return {}
    opts = dict(marker.kwargs)
    if marker.args:
        opts.setdefault("func", marker.args[0])
    return opts


def _resolve_source(item: pytest.Item, func_name: str, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (Path(item.path).parent / path).resolve()
        return path if path.is_file() else None
    mod = getattr(item, "module", None)
    obj = getattr(mod, func_name, None) if mod is not None else None
    if obj is None:
        return None
    src = inspect.getsourcefile(obj)
    return Path(src) if src else None


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    if call.when != "call" or call.excinfo is None:
        return
    config = item.config
    marked = item.get_closest_marker("evolab") is not None
    if not config.getoption("--evolab") and not marked:
        return
    opts = _marker_opts(item)
    func_name = opts.get("func") or config.getoption("--evolab-func")
    if not func_name:
        return
    source_opt = opts.get("source") or config.getoption("--evolab-source")
    src_path = _resolve_source(item, func_name, source_opt)
    if src_path is None:
        return
    try:
        test_src = Path(item.path).read_text(encoding="utf-8")
        test_name = getattr(item, "originalname", None) or item.name.split("[")[0]
        cases = extract_literal_cases(test_src, test_name, func_name)
    except Exception:
        return
    if not cases:
        return
    key = (str(src_path), func_name)
    job = config._evolab_jobs.setdefault(  # type: ignore[attr-defined]
        key,
        {
            "source_path": src_path,
            "func_name": func_name,
            "cases": [],
        },
    )
    job["cases"].extend(cases)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    jobs = getattr(session.config, "_evolab_jobs", {}) or {}
    if not jobs:
        return
    tw = session.config.get_terminal_writer()
    from .code_fixtures import CodeScenario
    from .repair import greedy_repair, unified_source_diff

    for job in jobs.values():
        path: Path = job["source_path"]
        func_name = job["func_name"]
        cases = job["cases"]
        sources = {path.name: path.read_text(encoding="utf-8")}
        scenario = CodeScenario(
            name=f"pytest:{path.name}:{func_name}",
            description="harvested from failing pytest asserts",
            sources=sources,
            target_file=path.name,
            func_name=func_name,
            test_cases=cases,
        )
        tw.line("")
        tw.sep("=", f"evolab repair · {func_name}")
        try:
            genome, history, n_eval = greedy_repair(
                scenario.sources, scenario.target_file, scenario.create_evaluator()
            )
            ev = scenario.create_evaluator().evaluate(genome)
            tw.line(f"evals={n_eval} score={ev.score} holdout={ev.passed_holdout}")
            diff = unified_source_diff(scenario.sources, genome.apply_to())
            tw.line(diff or "(no textual diff)")
            if ev.score >= 100:
                tw.line("evolab: suggested patch reaches training tests")
            else:
                tw.line("evolab: no complete repair in catalog")
        except Exception as exc:
            tw.line(f"evolab: repair skipped ({exc})")
