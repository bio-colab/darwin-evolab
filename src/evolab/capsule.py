"""src/evolab/capsule.py — Tamper-Evident Evidence Capsules (.evobundle).

Packages run reports, execution telemetry, artifacts, and cryptographic
runtime attestations into portable, tamper-evident evidence capsules (.evobundle).
Provides verification of cryptographic hashes and digital HMAC signatures to
prevent tampering and ensure reproduction integrity.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from .provenance import build_runtime_attestation, sha256_file

CAPSULE_VERSION = "1.0.0"


def _compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compute_hmac(key: bytes, message: bytes) -> str:
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def create_evidence_capsule(
    report_path: str | Path,
    output_bundle_path: str | Path,
    artifacts: list[str | Path] | None = None,
    telemetry_path: str | Path | None = None,
    secret_key: str | bytes | None = None,
) -> Path:
    """Pack a run report, attestation, and artifacts into a tamper-evident .evobundle archive."""
    report_file = Path(report_path)
    if not report_file.is_file():
        raise FileNotFoundError(f"Report file not found: {report_file}")

    bundle_file = Path(output_bundle_path)
    bundle_file.parent.mkdir(parents=True, exist_ok=True)

    report_bytes = report_file.read_bytes()
    attestation = build_runtime_attestation()
    attestation_bytes = json.dumps(attestation, indent=2, sort_keys=True).encode("utf-8")

    entries: dict[str, bytes] = {
        "report.json": report_bytes,
        "attestation.json": attestation_bytes,
    }

    if telemetry_path:
        telem_file = Path(telemetry_path)
        if telem_file.is_file():
            entries["telemetry.jsonl"] = telem_file.read_bytes()

    if artifacts:
        for art in artifacts:
            art_path = Path(art)
            if art_path.is_file():
                entries[f"artifacts/{art_path.name}"] = art_path.read_bytes()

    # Generate manifest of all content files
    manifest: dict[str, Any] = {
        "capsule_version": CAPSULE_VERSION,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": {},
    }

    for name, content in sorted(entries.items()):
        manifest["files"][name] = {
            "sha256": _compute_bytes_sha256(content),
            "size_bytes": len(content),
        }

    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    entries["manifest.json"] = manifest_bytes

    # Generate digital signature
    manifest_sha = _compute_bytes_sha256(manifest_bytes)
    if secret_key:
        key_bytes = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
        sig_val = _compute_hmac(key_bytes, manifest_bytes)
        sig_algo = "hmac-sha256"
    else:
        sig_val = manifest_sha
        sig_algo = "sha256-seal"

    signature_data = {
        "algorithm": sig_algo,
        "manifest_sha256": manifest_sha,
        "signature": sig_val,
        "has_secret_key": bool(secret_key),
    }
    entries["signature.json"] = json.dumps(signature_data, indent=2, sort_keys=True).encode("utf-8")

    # Write ZIP bundle
    with zipfile.ZipFile(bundle_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)

    return bundle_file


def inspect_evidence_capsule(bundle_path: str | Path) -> dict[str, Any]:
    """Inspect contents, metadata, and manifest of an evidence bundle without extracting."""
    bundle_file = Path(bundle_path)
    if not bundle_file.is_file():
        raise FileNotFoundError(f"Bundle file not found: {bundle_file}")

    with zipfile.ZipFile(bundle_file, mode="r") as zf:
        namelist = zf.namelist()
        manifest = {}
        signature = {}
        attestation = {}
        report_summary = {}

        if "manifest.json" in namelist:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if "signature.json" in namelist:
            signature = json.loads(zf.read("signature.json").decode("utf-8"))
        if "attestation.json" in namelist:
            attestation = json.loads(zf.read("attestation.json").decode("utf-8"))
        if "report.json" in namelist:
            rep = json.loads(zf.read("report.json").decode("utf-8"))
            report_summary = {
                "overall_verdict": rep.get("overall_verdict"),
                "total_generations": rep.get("total_generations"),
                "best_individual": rep.get("best_individual", {}),
            }

        return {
            "bundle": str(bundle_file),
            "size_bytes": bundle_file.stat().st_size,
            "capsule_version": manifest.get("capsule_version", "unknown"),
            "created_at_utc": manifest.get("created_at_utc"),
            "files": list(manifest.get("files", {}).keys()),
            "signature": signature,
            "attestation": attestation,
            "report_summary": report_summary,
        }


def verify_evidence_capsule(
    bundle_path: str | Path,
    secret_key: str | bytes | None = None,
) -> dict[str, Any]:
    """Cryptographically verify the integrity and signature of an evidence capsule."""
    bundle_file = Path(bundle_path)
    if not bundle_file.is_file():
        return {
            "valid": False,
            "status": "NOT_FOUND",
            "message": f"Bundle file not found: {bundle_file}",
        }

    try:
        with zipfile.ZipFile(bundle_file, mode="r") as zf:
            namelist = zf.namelist()
            if "manifest.json" not in namelist or "signature.json" not in namelist:
                return {
                    "valid": False,
                    "status": "CORRUPT",
                    "message": "Missing manifest.json or signature.json in capsule archive",
                }

            manifest_bytes = zf.read("manifest.json")
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            signature = json.loads(zf.read("signature.json").decode("utf-8"))

            # 1. Verify signature on manifest
            computed_manifest_sha = _compute_bytes_sha256(manifest_bytes)
            if computed_manifest_sha != signature.get("manifest_sha256"):
                return {
                    "valid": False,
                    "status": "TAMPERED",
                    "message": "Manifest SHA-256 does not match recorded signature manifest hash",
                }

            algo = signature.get("algorithm")
            recorded_sig = signature.get("signature")

            if algo == "hmac-sha256":
                if not secret_key:
                    return {
                        "valid": False,
                        "status": "KEY_REQUIRED",
                        "message": "Capsule was signed with HMAC-SHA256 but no verification key provided",
                    }
                key_bytes = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
                expected_sig = _compute_hmac(key_bytes, manifest_bytes)
                if not hmac.compare_digest(recorded_sig, expected_sig):
                    return {
                        "valid": False,
                        "status": "INVALID_SIGNATURE",
                        "message": "HMAC signature verification failed",
                    }
            elif algo == "sha256-seal":
                if recorded_sig != computed_manifest_sha:
                    return {
                        "valid": False,
                        "status": "TAMPERED",
                        "message": "Digest seal verification failed",
                    }

            # 2. Verify all file hashes in manifest
            files = manifest.get("files", {})
            file_results = {}
            all_files_match = True

            for fname, meta in files.items():
                if fname not in namelist:
                    file_results[fname] = {"status": "MISSING"}
                    all_files_match = False
                    continue

                content = zf.read(fname)
                digest = _compute_bytes_sha256(content)
                expected_digest = meta.get("sha256")
                if digest == expected_digest:
                    file_results[fname] = {"status": "MATCH", "size_bytes": len(content)}
                else:
                    file_results[fname] = {
                        "status": "MISMATCH",
                        "expected": expected_digest,
                        "actual": digest,
                    }
                    all_files_match = False

            if not all_files_match:
                return {
                    "valid": False,
                    "status": "TAMPERED",
                    "message": "One or more file digests in capsule do not match manifest",
                    "files": file_results,
                }

            return {
                "valid": True,
                "status": "VERIFIED",
                "message": "All capsule file digests and signatures cryptographically verified",
                "capsule_version": manifest.get("capsule_version"),
                "created_at_utc": manifest.get("created_at_utc"),
                "algorithm": algo,
                "files": file_results,
            }
    except zipfile.BadZipFile:
        return {
            "valid": False,
            "status": "CORRUPT",
            "message": "Invalid or corrupt zip archive file",
        }
    except Exception as exc:
        return {
            "valid": False,
            "status": "ERROR",
            "message": str(exc),
        }
