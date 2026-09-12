"""Unit tests for Hierarchical Cascading Configuration and Provenance Tracking."""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from evolab.config import (
    load_hierarchical_config,
    get_config_provenance,
)


def test_hierarchical_config_layering(tmp_path: Path, monkeypatch):
    # 1. Setup simulated project dir with evolab.toml
    proj_file = tmp_path / "evolab.toml"
    proj_file.write_text("""
[engine]
population = 32
generations = 50
seed = 777
""", encoding="utf-8")

    # 2. Setup environment variable override
    monkeypatch.setenv("EVOLAB_GENERATIONS", "100")
    monkeypatch.setenv("EVOLAB_MUTATION_RATE", "0.25")

    # 3. Supply CLI overrides
    cli_overrides = {"seed": 999}

    # Load merged configuration
    cfg = load_hierarchical_config(start_dir=tmp_path, cli_overrides=cli_overrides)

    # Assert correct cascading resolution:
    # - engine default for 'diff': False (layer 1)
    assert cfg["diff"] is False
    # - project config for 'population': 32 (layer 3)
    assert cfg["population"] == 32
    # - env var for 'generations': 100 (layer 4 overrides project 50)
    assert cfg["generations"] == 100
    assert cfg["mutation_rate"] == 0.25
    # - cli for 'seed': 999 (layer 5 overrides project 777)
    assert cfg["seed"] == 999

    # Verify provenance tracking
    prov = get_config_provenance()
    assert prov["diff"][1] == "default"
    assert "project" in prov["population"][1]
    assert "env" in prov["generations"][1]
    assert prov["seed"][1] == "cli"
