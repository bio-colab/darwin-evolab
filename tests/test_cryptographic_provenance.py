"""tests/test_cryptographic_provenance.py — Test cryptographic runtime provenance and Merkle attestation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from evolab.engine import EvolutionEngine
from evolab.provenance import (
    build_runtime_attestation,
    get_libraries_digest,
    get_python_digest,
    get_workspace_merkle_root,
    sha256_file,
    verify_workspace_integrity,
)


def test_sha256_file_basic(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Hello, Evolab!", encoding="utf-8")
    h1 = sha256_file(sample)
    assert len(h1) == 64
    assert int(h1, 16) > 0

    # Non-existent file
    assert sha256_file(tmp_path / "non_existent.txt") == ""


def test_python_digest_valid():
    digest = get_python_digest()
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert int(digest, 16) > 0


def test_libraries_digest():
    digests = get_libraries_digest()
    assert isinstance(digests, dict)
    # If numpy or scipy installed in this env, their hashes should be 64-char hex
    for name, h in digests.items():
        assert len(h) == 64


def test_workspace_merkle_root_deterministic():
    root1 = get_workspace_merkle_root()
    root2 = get_workspace_merkle_root()
    assert isinstance(root1, str)
    assert len(root1) == 64
    assert root1 == root2


def test_merkle_root_sensitive_to_tampering(tmp_path: Path):
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "module_a.py").write_text("def a(): return 1\n", encoding="utf-8")
    (src / "module_b.py").write_text("def b(): return 2\n", encoding="utf-8")

    root1 = get_workspace_merkle_root(src)
    assert len(root1) == 64
    assert verify_workspace_integrity(root1, src) is True

    # Tamper with module_a
    (src / "module_a.py").write_text("def a(): return 999\n", encoding="utf-8")
    root2 = get_workspace_merkle_root(src)
    assert root2 != root1
    assert verify_workspace_integrity(root1, src) is False


def test_runtime_attestation_structure():
    t0 = time.perf_counter()
    attestation = build_runtime_attestation()
    elapsed = time.perf_counter() - t0

    # Should execute swiftly
    assert elapsed < 1.0

    assert attestation["schema_version"] == "1.0.0"
    assert len(attestation["python_sha256"]) == 64
    assert len(attestation["core_merkle_root"]) == 64
    assert isinstance(attestation["libraries_digest"], dict)
    assert "environment" in attestation
    env = attestation["environment"]
    assert "python_version" in env
    assert "os_platform" in env
    assert "machine_arch" in env
    assert "timestamp_utc" in attestation


def test_report_builder_embeds_attestation():
    engine = EvolutionEngine(population_size=4, seed=42)
    rep = engine.run(generations=2)
    assert "attestation" in rep
    assert rep["attestation"]["schema_version"] == "1.0.0"
    assert len(rep["attestation"]["python_sha256"]) == 64
    assert len(rep["attestation"]["core_merkle_root"]) == 64


def test_supervisor_audit_embeds_attestation(tmp_path: Path):
    from evolab.supervisor import SelfAuditSupervisor
    sup = SelfAuditSupervisor(
        candidate_mode="control",
        full_suite=False,
        output_path=str(tmp_path / "audit.json"),
        quiet=True,
    )
    sup.seeds = [1]
    sup._run_code_benchmarks = lambda: {}
    audit = sup.run_audit()
    assert "attestation" in audit
    assert audit["attestation"]["schema_version"] == "1.0.0"
    assert len(audit["attestation"]["python_sha256"]) == 64


