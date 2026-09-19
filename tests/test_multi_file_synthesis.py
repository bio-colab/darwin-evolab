"""Tests for Multi-File Cross-Module Synthesis & Compound Repairs."""

from __future__ import annotations

import ast
from pathlib import Path
import pytest

from evolab import (
    CompoundRepairEdit,
    ContractPreservationValidator,
    CrossFileDependencyGraph,
    CrossFileMutator,
    FunctionTestEvaluator,
    MultiFileCompoundMutator,
    RepairEdit,
    RepairGenome,
    greedy_repair,
    unified_source_diff,
)


def test_compound_repair_edit_properties_and_serialization():
    """Verify CompoundRepairEdit has valid locus, keys, serialization, and sub-edit nesting."""
    sub1 = RepairEdit(kind="bool_flip", file="mod_a.py", lineno=10, col_offset=4)
    sub2 = RepairEdit(kind="boundary_cmp", file="mod_b.py", lineno=25, col_offset=8)

    compound = CompoundRepairEdit(
        kind="cross_file_compound",
        file="mod_a.py",
        lineno=10,
        col_offset=4,
        sub_edits=(sub1, sub2),
        description="Synchronized patch across mod_a and mod_b",
    )

    assert compound.locus() == ("mod_a.py", 10, 4)
    key = compound.key()
    assert key[0] == "compound"
    assert len(key[5]) == 2

    serialized = compound.serialize()
    assert serialized["kind"] == "cross_file_compound"
    assert serialized["file"] == "mod_a.py"
    assert len(serialized["sub_edits"]) == 2
    assert serialized["description"] == "Synchronized patch across mod_a and mod_b"


def test_repair_genome_with_compound_edits():
    """Verify RepairGenome applies CompoundRepairEdit across multiple source files cleanly."""
    sources = {
        "service.py": "def is_valid(x):\n    return x > 10\n",
        "client.py": "from service import is_valid\ndef check(x):\n    return is_valid(x)\n",
    }

    sub1 = RepairEdit(kind="boundary_cmp", file="service.py", lineno=2, col_offset=11)
    compound = CompoundRepairEdit(
        kind="cross_file_sync",
        file="service.py",
        lineno=2,
        col_offset=11,
        sub_edits=(sub1,),
    )

    genome = RepairGenome(sources=sources, target_file="service.py", edits=[compound])
    applied = genome.apply_to()

    assert "x >= 10" in applied["service.py"]
    assert "client.py" in applied
    assert genome.fingerprint() != ""


def test_contract_preservation_validator():
    """Verify ContractPreservationValidator screens invalid mutations before sandbox evaluation."""
    sources = {
        "math_lib.py": "def calculate(a, b):\n    return {'result': a + b}\n",
        "app.py": "from math_lib import calculate\ndef run():\n    res = calculate(1, 2)\n    return res['result']\n",
    }

    validator = ContractPreservationValidator(sources)

    # 1. Valid modification: preserves signature and schema
    valid_mut = {
        "math_lib.py": "def calculate(a, b):\n    return {'result': (a + b) * 1}\n",
        "app.py": sources["app.py"],
    }
    ok, violations = validator.validate_candidate(valid_mut)
    assert ok is True
    assert len(violations) == 0

    # 2. Invalid modification: changes signature/arity breaking caller
    invalid_arity = {
        "math_lib.py": "def calculate(a, b, c):\n    return {'result': a + b + c}\n",
        "app.py": sources["app.py"],
    }
    ok, violations = validator.validate_candidate(invalid_arity)
    assert ok is False
    assert any("SignatureMismatch" in v for v in violations)

    # 3. Invalid modification: drops return dictionary key required by caller
    invalid_schema = {
        "math_lib.py": "def calculate(a, b):\n    return {'sum': a + b}\n",
        "app.py": sources["app.py"],
    }
    ok, violations = validator.validate_candidate(invalid_schema)
    assert ok is False
    assert any("SchemaContractViolation" in v for v in violations)


def test_multi_file_compound_mutator_generation():
    """Verify MultiFileCompoundMutator discovers caller-callee misalignments and generates compound candidates."""
    sources = {
        "provider.py": "def process_data(data):\n    return data.strip()\n",
        "consumer.py": "from provider import process_data\ndef main(d):\n    return process_data(d, timeout=5)\n",
    }

    candidates = MultiFileCompoundMutator.generate_compound_candidates(
        sources=sources,
        target_file="provider.py",
    )

    assert len(candidates) >= 1
    # Check that synchronized addition of 'timeout' parameter was generated
    param_cand = [c for c in candidates if c.kind == "sync_add_parameter"]
    assert len(param_cand) >= 1
    assert "timeout" in param_cand[0].description

    # Test applying the transform
    mutated_sources = param_cand[0].apply_to_sources(sources)
    assert "timeout" in mutated_sources["provider.py"]
    for code in mutated_sources.values():
        compile(code, "<test>", "exec")


def test_end_to_end_multi_file_greedy_repair():
    """Verify greedy_repair solves a multi-module defect via compound multi-file synthesis."""
    # Scenario: consumer calls provider.compute with a keyword argument that provider does not accept yet
    sources = {
        "algo.py": "def compute(val, scale=1):\n    return val * scale\n",
        "workflow.py": "from algo import compute\ndef execute(val, strict=1):\n    return compute(val, strict=strict)\n",
    }

    # In algo.py, compute currently only takes (val, scale=1), but workflow passes (val, strict=strict)
    # The fix requires synchronizing parameter 'strict' into algo.compute
    import sys, types
    class MultiModuleEvaluator:
        def evaluate(self, genome: RepairGenome):
            applied = genome.apply_to()
            try:
                mod_algo = types.ModuleType("algo")
                exec(applied["algo.py"], mod_algo.__dict__)
                sys.modules["algo"] = mod_algo

                ns_work = {}
                exec(applied["workflow.py"], ns_work)

                # Test case: execute(10, strict=2)
                res = ns_work["execute"](10, strict=2)
                from evolab.evaluators import FitnessResult
                return FitnessResult(score=100.0, passed_holdout=True)
            except TypeError as e:
                from evolab.evaluators import FitnessResult
                return FitnessResult(score=20.0, passed_holdout=False)
            except Exception as e:
                from evolab.evaluators import FitnessResult
                return FitnessResult(score=0.0, passed_holdout=False)
            finally:
                sys.modules.pop("algo", None)

    ev = MultiModuleEvaluator()
    genome, history, n_evals = greedy_repair(
        sources=sources,
        target_file="algo.py",
        evaluator=ev,
        max_evals=10,
        enable_multi_file_synthesis=True,
    )

    # Confirm repair succeeded
    assert genome is not None
    res = ev.evaluate(genome)
    assert res.score == 100.0
    assert res.passed_holdout is True
    applied = genome.apply_to()
    assert "strict" in applied["algo.py"]
