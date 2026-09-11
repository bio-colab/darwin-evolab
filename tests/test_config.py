"""Unit tests for declarative project configuration discovery and loading."""
from __future__ import annotations

import json
from pathlib import Path
from evolab.config import (
    find_project_config_file,
    generate_default_config,
    load_project_config,
)


def test_generate_default_config():
    toml_str = generate_default_config(fmt="toml")
    assert "[project]" in toml_str
    assert "sources" in toml_str

    json_str = generate_default_config(fmt="json")
    data = json.loads(json_str)
    assert "project" in data
    assert "sources" in data["project"]


def test_load_project_config_toml(tmp_path: Path):
    cfg_file = tmp_path / "evolab.toml"
    cfg_file.write_text("""
[project]
sources = ["foo.py"]
pytest = "tests/test_foo.py"

[engine]
engine = "auto"
generations = 42
population = 24
target = 98.5
seed = 123
""", encoding="utf-8")

    loaded = load_project_config(tmp_path)
    assert loaded["source"] == ["foo.py"]
    assert loaded["pytest"] == "tests/test_foo.py"
    assert loaded["generations"] == 42
    assert loaded["population"] == 24
    assert loaded["seed"] == 123


def test_load_project_config_json(tmp_path: Path):
    cfg_file = tmp_path / ".evolab.json"
    cfg_file.write_text(json.dumps({
        "project": {
            "sources": ["bar.py"],
            "pytest": "tests/test_bar.py",
        },
        "engine": {
            "generations": 15,
        }
    }), encoding="utf-8")

    loaded = load_project_config(tmp_path)
    assert loaded["source"] == ["bar.py"]
    assert loaded["generations"] == 15


def test_find_project_config_upwards(tmp_path: Path):
    sub_dir = tmp_path / "subdir" / "nested"
    sub_dir.mkdir(parents=True)
    cfg_file = tmp_path / "evolab.toml"
    cfg_file.write_text("[project]\nsources = ['app.py']\n", encoding="utf-8")

    found = find_project_config_file(sub_dir)
    assert found == cfg_file
