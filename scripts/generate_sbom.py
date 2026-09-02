#!/usr/bin/env python3
"""Generate CycloneDX 1.5 JSON Software Bill of Materials (SBOM) for evolab."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from evolab import __version__ as EVOLAB_VERSION
except ImportError:
    EVOLAB_VERSION = "0.5.0"

def generate_sbom() -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "evolab-sbom-generator",
                        "version": EVOLAB_VERSION
                    }
                ]
            },
            "component": {
                "type": "library",
                "name": "evolab",
                "version": EVOLAB_VERSION,
                "description": "Evidence-first evolutionary code experimentation workbench",
                "licenses": [{"license": {"id": "MIT"}}],
                "purl": f"pkg:pypi/evolab@{EVOLAB_VERSION}"
            }
        },
        "components": [
            {
                "type": "library",
                "name": "pytest",
                "version": "8.4.2",
                "purl": "pkg:pypi/pytest@8.4.2",
                "scope": "optional"
            },
            {
                "type": "library",
                "name": "pyyaml",
                "version": "6.0.2",
                "purl": "pkg:pypi/pyyaml@6.0.2",
                "scope": "optional"
            }
        ]
    }

if __name__ == "__main__":
    out_file = ROOT / "sbom.json"
    sbom = generate_sbom()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sbom, f, indent=2)
    print(f"[OK] CycloneDX SBOM written to {out_file}")
