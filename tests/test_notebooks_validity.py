"""Tests verifying that all educational Jupyter notebooks conform to nbformat schema and contain valid Python syntax."""
import ast
import json
from pathlib import Path
import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


def test_notebooks_exist():
    assert NOTEBOOKS_DIR.is_dir()
    expected_nbs = [
        "01_software_repair.ipynb",
        "02_silicon_circuit_synthesis.ipynb",
        "03_cgp_and_discrete_logic.ipynb",
        "04_continuous_optimization_jax.ipynb",
    ]
    for nb_name in expected_nbs:
        nb_path = NOTEBOOKS_DIR / nb_name
        assert nb_path.is_file(), f"Missing notebook: {nb_name}"


@pytest.mark.parametrize("nb_name", [
    "01_software_repair.ipynb",
    "02_silicon_circuit_synthesis.ipynb",
    "03_cgp_and_discrete_logic.ipynb",
    "04_continuous_optimization_jax.ipynb",
])
def test_notebook_json_and_syntax(nb_name):
    nb_path = NOTEBOOKS_DIR / nb_name
    data = json.loads(nb_path.read_text(encoding="utf-8"))

    assert data.get("nbformat") == 4
    assert "cells" in data
    assert len(data["cells"]) > 0

    # Ensure all code cells contain valid Python AST syntax
    for idx, cell in enumerate(data["cells"]):
        if cell.get("cell_type") == "code":
            source_lines = cell.get("source", [])
            source_code = "".join(source_lines)
            try:
                ast.parse(source_code)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {nb_name} cell {idx}: {e}")

