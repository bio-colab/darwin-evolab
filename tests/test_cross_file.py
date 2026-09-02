import ast
import random
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.ast_genome import ASTGenome, MultiFileASTGenome
from evolab.cross_file import (
    CrossFileDependencyGraph,
    CrossFileMutator,
    CallSite,
    ImportInfo,
)


@pytest.fixture
def multi_module_sources() -> dict[str, str]:
    return {
        "math_utils.py": (
            "def add_numbers(a, b):\n"
            "    return a + b\n\n"
            "def multiply_numbers(a, b):\n"
            "    return a * b\n"
        ),
        "calculator.py": (
            "from math_utils import add_numbers\n\n"
            "def compute_total(x, y, z):\n"
            "    sub = add_numbers(x, y)\n"
            "    return add_numbers(sub, z)\n"
        ),
        "app_service.py": (
            "import math_utils\n"
            "from calculator import compute_total\n\n"
            "def run_service(val1, val2):\n"
            "    prod = math_utils.multiply_numbers(val1, val2)\n"
            "    return compute_total(prod, val1, val2)\n"
        ),
    }


def test_cross_file_dependency_graph_building(multi_module_sources):
    graph = CrossFileDependencyGraph.build(multi_module_sources)

    assert len(graph.file_ast) == 3
    assert "math_utils.py" in graph.symbols
    assert "add_numbers" in graph.symbols["math_utils.py"].functions
    assert "multiply_numbers" in graph.symbols["math_utils.py"].functions

    # Check imports in calculator.py
    calc_imports = graph.imports["calculator.py"]
    assert any(imp.imported_module == "math_utils" and imp.imported_symbol == "add_numbers" for imp in calc_imports)

    # Check call sites in calculator.py
    calc_calls = graph.call_sites["calculator.py"]
    assert len(calc_calls) == 2
    assert all(call.func_name == "add_numbers" for call in calc_calls)


def test_find_callers_across_modules(multi_module_sources):
    graph = CrossFileDependencyGraph.build(multi_module_sources)

    callers_add = graph.find_callers_of("math_utils.py", "add_numbers")
    assert len(callers_add) == 2
    assert all(caller_file == "calculator.py" for caller_file, _ in callers_add)

    callers_mult = graph.find_callers_of("math_utils.py", "multiply_numbers")
    assert len(callers_mult) == 1
    assert callers_mult[0][0] == "app_service.py"


def test_rename_symbol_synchronized(multi_module_sources):
    new_sources = CrossFileMutator.rename_symbol_synchronized(
        multi_module_sources,
        target_file="math_utils.py",
        old_name="add_numbers",
        new_name="sum_values",
    )

    # 1. math_utils.py has new definition name
    assert "def sum_values(a, b):" in new_sources["math_utils.py"]
    assert "add_numbers" not in new_sources["math_utils.py"]

    # 2. calculator.py updated import and call sites
    assert "from math_utils import sum_values" in new_sources["calculator.py"]
    assert "sub = sum_values(x, y)" in new_sources["calculator.py"]
    assert "return sum_values(sub, z)" in new_sources["calculator.py"]
    assert "add_numbers" not in new_sources["calculator.py"]

    # 3. All files compile without error
    for path, code in new_sources.items():
        compile(code, path, "exec")


def test_add_parameter_synchronized(multi_module_sources):
    new_sources = CrossFileMutator.add_parameter_synchronized(
        multi_module_sources,
        target_file="math_utils.py",
        func_name="add_numbers",
        param_name="scale",
        default_val=1,
    )

    # 1. math_utils.py definition includes parameter
    assert "def add_numbers(a, b, scale=1):" in new_sources["math_utils.py"]

    # 2. calculator.py calls include default keyword
    assert "scale=1" in new_sources["calculator.py"]

    # 3. Code remains valid and compilable
    for path, code in new_sources.items():
        compile(code, path, "exec")


def test_inject_import_and_helper_usage(multi_module_sources):
    new_sources = CrossFileMutator.inject_import(
        multi_module_sources,
        consumer_file="calculator.py",
        provider_file="math_utils.py",
        symbol_name="multiply_numbers",
    )

    assert "from math_utils import multiply_numbers" in new_sources["calculator.py"]
    for path, code in new_sources.items():
        compile(code, path, "exec")


def test_mutate_multi_file_genome_end_to_end(multi_module_sources):
    genome = MultiFileASTGenome.from_sources(multi_module_sources)
    rng = random.Random(42)

    for _ in range(5):
        mutated_genome = CrossFileMutator.mutate_multi_file_genome(genome, rng)
        assert len(mutated_genome.files) == len(multi_module_sources)
        sources = mutated_genome.to_sources()
        for path, code in sources.items():
            tree = ast.parse(code, filename=path)
            assert isinstance(tree, ast.Module)
