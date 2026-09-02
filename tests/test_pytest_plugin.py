"""Harvest failing literal asserts and suggest a repair."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from evolab.pytest_plugin import extract_literal_cases

ROOT = Path(__file__).resolve().parents[1]


def test_extract_literal_cases():
    src = (
        "def test_port():\n"
        "    assert parse_cli(['8080']) == {'port': 8080}\n"
        "    assert parse_cli(['80']) == {'port': 80}\n"
    )
    cases = extract_literal_cases(src, "test_port", "parse_cli")
    assert cases == [
        ((["8080"],), {"port": 8080}),
        ((["80"],), {"port": 80}),
    ]


def test_plugin_suggests_diff_on_failure(tmp_path):
    app = tmp_path / "app.py"
    app.write_text(
        "def parse_cli(args):\n"
        "    port = args[0]\n"
        "    return {'port': port}\n",
        encoding="utf-8",
    )
    test = tmp_path / "test_app.py"
    test.write_text(
        "from app import parse_cli\n"
        "import pytest\n"
        "\n"
        "@pytest.mark.evolab(func='parse_cli', source='app.py')\n"
        "def test_port():\n"
        "    assert parse_cli(['8080']) == {'port': 8080}\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(tmp_path)}
    # The repo ships src/evolab.egg-info, so importlib.metadata exposes the
    # pytest11 "evolab" entry point as soon as src/ is on sys.path. Combined
    # with the explicit `-p evolab.pytest_plugin` below, autoloading registers
    # the same module twice under two names and crashes pluggy. Disable
    # autoloading: the explicit -p flag alone loads the plugin exactly once.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "-p", "evolab.pytest_plugin",
            "--evolab",
            str(test),
            "-q",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    out = proc.stdout + proc.stderr
    assert "evolab repair" in out
    assert "int(" in out or "suggested patch" in out
