"""tests/test_evidence_capsule.py — Test tamper-evident evidence capsules (.evobundle)."""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from evolab.capsule import (
    create_evidence_capsule,
    inspect_evidence_capsule,
    verify_evidence_capsule,
)


@pytest.fixture
def sample_report(tmp_path: Path) -> Path:
    rep = {
        "engine_version": "0.5.0",
        "schema_version": "1.0.0",
        "total_generations": 10,
        "best_individual": {
            "fitness": 99.8,
            "passed_holdout": True,
            "id": "gen_10_ind_01",
        },
        "history": [{"generation": 1, "best_fitness": 50.0}],
    }
    p = tmp_path / "run_report.json"
    p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return p


def test_create_and_inspect_capsule(sample_report: Path, tmp_path: Path):
    bundle_path = tmp_path / "experiment.evobundle"
    artifact = tmp_path / "patch.diff"
    artifact.write_text("--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")

    out_path = create_evidence_capsule(
        report_path=sample_report,
        output_bundle_path=bundle_path,
        artifacts=[artifact],
    )
    assert out_path.is_file()
    assert out_path == bundle_path

    info = inspect_evidence_capsule(bundle_path)
    assert info["capsule_version"] == "1.0.0"
    assert "report.json" in info["files"]
    assert "attestation.json" in info["files"]
    assert "artifacts/patch.diff" in info["files"]
    assert "signature" in info
    assert "manifest_sha256" in info["signature"]
    assert info["report_summary"]["best_individual"]["fitness"] == 99.8


def test_verify_digest_seal_capsule(sample_report: Path, tmp_path: Path):
    bundle_path = tmp_path / "sealed.evobundle"
    create_evidence_capsule(sample_report, bundle_path)

    res = verify_evidence_capsule(bundle_path)
    assert res["valid"] is True
    assert res["status"] == "VERIFIED"
    assert res["algorithm"] == "sha256-seal"
    assert res["files"]["report.json"]["status"] == "MATCH"
    assert res["files"]["attestation.json"]["status"] == "MATCH"


def test_verify_hmac_signed_capsule(sample_report: Path, tmp_path: Path):
    bundle_path = tmp_path / "signed.evobundle"
    secret = "lab-super-secret-key-42"
    create_evidence_capsule(sample_report, bundle_path, secret_key=secret)

    # 1. Verification with correct key
    res_valid = verify_evidence_capsule(bundle_path, secret_key=secret)
    assert res_valid["valid"] is True
    assert res_valid["status"] == "VERIFIED"
    assert res_valid["algorithm"] == "hmac-sha256"

    # 2. Verification without key
    res_no_key = verify_evidence_capsule(bundle_path)
    assert res_no_key["valid"] is False
    assert res_no_key["status"] == "KEY_REQUIRED"

    # 3. Verification with wrong key
    res_wrong_key = verify_evidence_capsule(bundle_path, secret_key="wrong-key")
    assert res_wrong_key["valid"] is False
    assert res_wrong_key["status"] == "INVALID_SIGNATURE"


def test_tamper_detection_modified_content(sample_report: Path, tmp_path: Path):
    bundle_path = tmp_path / "tampered.evobundle"
    create_evidence_capsule(sample_report, bundle_path)

    # Read all files from zip, tamper report.json, and write back
    tampered_bundle = tmp_path / "corrupted.evobundle"
    with zipfile.ZipFile(bundle_path, "r") as z_in:
        names = z_in.namelist()
        data_map = {name: z_in.read(name) for name in names}

    # Tamper with report.json
    rep_obj = json.loads(data_map["report.json"].decode("utf-8"))
    rep_obj["best_individual"]["fitness"] = 100.0  # Fraudulent alteration
    data_map["report.json"] = json.dumps(rep_obj).encode("utf-8")

    with zipfile.ZipFile(tampered_bundle, "w") as z_out:
        for name, data in data_map.items():
            z_out.writestr(name, data)

    res = verify_evidence_capsule(tampered_bundle)
    assert res["valid"] is False
    assert res["status"] == "TAMPERED"
    assert res["files"]["report.json"]["status"] == "MISMATCH"


def test_tamper_detection_modified_manifest(sample_report: Path, tmp_path: Path):
    bundle_path = tmp_path / "tampered_manifest.evobundle"
    create_evidence_capsule(sample_report, bundle_path)

    tampered_bundle = tmp_path / "bad_manifest.evobundle"
    with zipfile.ZipFile(bundle_path, "r") as z_in:
        names = z_in.namelist()
        data_map = {name: z_in.read(name) for name in names}

    # Tamper with manifest.json directly
    mani_obj = json.loads(data_map["manifest.json"].decode("utf-8"))
    mani_obj["files"]["report.json"]["sha256"] = "0" * 64
    data_map["manifest.json"] = json.dumps(mani_obj).encode("utf-8")

    with zipfile.ZipFile(tampered_bundle, "w") as z_out:
        for name, data in data_map.items():
            z_out.writestr(name, data)

    res = verify_evidence_capsule(tampered_bundle)
    assert res["valid"] is False
    assert res["status"] == "TAMPERED"
    assert "Manifest SHA-256 does not match" in res["message"]


def test_cli_bundle_and_verify(sample_report: Path, tmp_path: Path):
    bundle_path = tmp_path / "cli_test.evobundle"

    # 1. evolab bundle --json
    res_b = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "bundle", str(sample_report), "-o", str(bundle_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert res_b.returncode == 0
    b_data = json.loads(res_b.stdout)
    assert b_data["status"] == "success"
    assert Path(b_data["bundle_path"]).is_file()

    # 2. evolab verify --json (success)
    res_v = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "verify", str(bundle_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert res_v.returncode == 0
    v_data = json.loads(res_v.stdout)
    assert v_data["valid"] is True
    assert v_data["status"] == "VERIFIED"

    # 3. evolab verify non-existent bundle
    res_missing = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "verify", str(tmp_path / "non_existent.evobundle"), "--json"],
        capture_output=True,
        text=True,
    )
    assert res_missing.returncode != 0
    m_data = json.loads(res_missing.stdout)
    assert m_data["valid"] is False
    assert m_data["status"] == "NOT_FOUND"
