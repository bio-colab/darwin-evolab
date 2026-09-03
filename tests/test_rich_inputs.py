"""Tests for rich test inputs: kwargs, pytest.raises exceptions, and auto-discovery."""
from __future__ import annotations

import pytest
from pathlib import Path
from evolab.evaluators import FunctionTestEvaluator
from evolab.code_fixtures import load_pytest_scenario
from evolab.pytest_plugin import extract_literal_cases


def test_evaluator_supports_kwargs():
    code = (
        "def greet(name, greeting='Hello'):\n"
        "    return f'{greeting}, {name}!'\n"
    )
    ev = FunctionTestEvaluator(
        base_sources={"mod.py": code},
        target_file="mod.py",
        func_name="greet",
        test_cases=[
            ((("Alice",), {"greeting": "Hi"}), "Hi, Alice!"),
            ((("Bob",), {}), "Hello, Bob!"),
        ],
    )
    res = ev.evaluate(code)
    assert res.score == 100.0
    assert res.artifacts["passed_tests"] == 2


def test_evaluator_supports_expected_exceptions():
    code = (
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        raise ValueError('Cannot divide by zero')\n"
        "    return a / b\n"
    )
    ev = FunctionTestEvaluator(
        base_sources={"mod.py": code},
        target_file="mod.py",
        func_name="divide",
        test_cases=[
            (((10, 2), {}), 5.0),
            (((10, 0), {}), "raises:ValueError"),
        ],
    )
    res = ev.evaluate(code)
    assert res.score == 100.0
    assert res.artifacts["passed_tests"] == 2


def test_extract_literal_cases_with_kwargs_and_raises():
    test_src = (
        "import pytest\n"
        "def test_cases():\n"
        "    assert calculate(10, factor=2) == 20\n"
        "    with pytest.raises(ValueError):\n"
        "        calculate(-1)\n"
        "    assert is_ok(1)\n"
        "    assert not is_ok(-1)\n"
    )

    cases_calc = extract_literal_cases(test_src, "test_cases", "calculate")
    assert len(cases_calc) == 2
    # First case should have kwargs
    args_kw, expected = cases_calc[0]
    assert args_kw == ((10,), {"factor": 2})
    assert expected == 20
    # Second case should be raises:ValueError
    args_2, exp_err = cases_calc[1]
    assert args_2 == (-1,)
    assert exp_err == "raises:ValueError"

    cases_bool = extract_literal_cases(test_src, "test_cases", "is_ok")
    assert len(cases_bool) == 2
    assert cases_bool[0] == ((1,), True)
    assert cases_bool[1] == ((-1,), False)


def test_auto_detect_target_function(tmp_path):
    app_py = tmp_path / "app.py"
    app_py.write_text(
        "def helper(x):\n"
        "    return x\n\n"
        "def target_func(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    test_py = tmp_path / "test_app.py"
    test_py.write_text(
        "def test_app():\n"
        "    assert target_func(1, 2) == 3\n"
        "    assert target_func(5, 5) == 10\n"
        "    assert target_func(0, 0) == 0\n",
        encoding="utf-8",
    )

    # Note: func_name is None, should auto-detect target_func!
    scenario = load_pytest_scenario([app_py], test_py, func_name=None)
    assert scenario.func_name == "target_func"
    assert scenario.target_file == "app.py"
    assert len(scenario.test_cases) == 2
    assert len(scenario.holdout_cases) == 1
