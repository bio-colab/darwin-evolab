"""src/evolab/provenance.py — Deterministic Runtime Provenance & Cryptographic Attestation.

Calculates cryptographic fingerprints of:
1. The running Python interpreter binary.
2. Critical numerical C-extension libraries (NumPy, SciPy, BLAS backends).
3. Workspace Merkle tree covering core engine files to detect untracked modifications.
4. Host platform and hardware telemetry.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent


def sha256_file(path: str | Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    h = hashlib.sha256()
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def get_python_digest() -> str:
    """Compute SHA-256 of the active Python binary executable."""
    return sha256_file(sys.executable)


def get_libraries_digest() -> dict[str, str]:
    """Compute cryptographic hashes of loaded scientific dynamic libraries."""
    digests: dict[str, str] = {}

    # Check numpy
    try:
        import numpy as np
        np_file = getattr(np, "__file__", None)
        if np_file:
            np_dir = Path(np_file).parent
            # Find core C-extension .pyd or .so
            for ext in np_dir.glob("*multiarray_umath*.*"):
                if ext.suffix in (".pyd", ".so", ".dylib"):
                    digests[f"numpy:{ext.name}"] = sha256_file(ext)
            if not digests:
                digests[f"numpy:core"] = sha256_file(np_file)
    except Exception:
        pass

    # Check scipy
    try:
        import scipy as sp
        sp_file = getattr(sp, "__file__", None)
        if sp_file:
            sp_dir = Path(sp_file).parent
            for ext in sp_dir.glob("*_lib*.*"):
                if ext.suffix in (".pyd", ".so", ".dylib"):
                    digests[f"scipy:{ext.name}"] = sha256_file(ext)
            if not any(k.startswith("scipy") for k in digests):
                digests[f"scipy:core"] = sha256_file(sp_file)
    except Exception:
        pass

    return digests


def get_workspace_merkle_root(target_dir: str | Path | None = None) -> str:
    """Compute deterministic Merkle root hash across all Python source files in the core."""
    directory = Path(target_dir or SRC_DIR)
    if not directory.is_dir():
        return ""

    file_hashes: list[str] = []
    # Collect and sort all .py files deterministically
    py_files = sorted(directory.rglob("*.py"), key=lambda p: str(p.relative_to(directory)))

    for file_path in py_files:
        rel_path = str(file_path.relative_to(directory)).replace("\\", "/")
        f_hash = sha256_file(file_path)
        file_hashes.append(f"{rel_path}:{f_hash}")

    combined = "\n".join(file_hashes).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def build_runtime_attestation(target_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate complete cryptographic attestation capsule for the active runtime."""
    return {
        "schema_version": "1.0.0",
        "python_sha256": get_python_digest(),
        "core_merkle_root": get_workspace_merkle_root(target_dir),
        "libraries_digest": get_libraries_digest(),
        "environment": {
            "python_version": platform.python_version(),
            "os_platform": sys.platform,
            "os_release": platform.release(),
            "machine_arch": platform.machine(),
            "logical_cpus": os.cpu_count() or 1,
        },
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def verify_workspace_integrity(expected_merkle_root: str, target_dir: str | Path | None = None) -> bool:
    """Verify whether core source files match expected cryptographic Merkle root."""
    current_root = get_workspace_merkle_root(target_dir)
    return current_root == expected_merkle_root
