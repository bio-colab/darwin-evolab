"""Unit tests for Checkpoint Freeze, Serialization, and Deterministic Resumption."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from evolab.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    serialize_individual,
    deserialize_individual,
)
from evolab.config import EngineConfig
from evolab.engine import EvolutionEngine
from evolab.genome import FloatGenome, Individual


def test_individual_serialization_roundtrip():
    ind = Individual(
        genome=FloatGenome(values=[1.23, -4.56, 7.89]),
        species="spec_test",
        fitness=88.5,
        adjusted_fitness=75.0,
        lineage={"parent_a": "gen_00_ind_01"},
        last_evaluated_gen=5,
        _generation=5,
        _index=2,
    )

    rec = serialize_individual(ind)
    assert rec["species"] == "spec_test"
    assert rec["fitness"] == 88.5
    assert rec["genome"]["class_name"] == "FloatGenome"

    restored = deserialize_individual(rec)
    assert restored.species == ind.species
    assert restored.fitness == ind.fitness
    assert restored.genome.values == ind.genome.values
    assert restored.last_evaluated_gen == ind.last_evaluated_gen


def test_checkpoint_save_and_load(tmp_path: Path):
    ckpt_file = tmp_path / "test_ckpt.json"
    pop = [
        Individual(FloatGenome([float(i), float(i * 2)]), species="spec_0", fitness=float(i * 10))
        for i in range(4)
    ]
    best = pop[-1]

    import random
    rng = random.Random(42)
    rng.random()  # advance state

    history = [{"generation": 1, "best_fitness": 30.0}]
    species_history = [{"spec_0": 4}]

    saved_path = save_checkpoint(
        filepath=ckpt_file,
        generation=1,
        total_generations=10,
        population=pop,
        best_ever=best,
        rng=rng,
        history=history,
        species_history=species_history,
        metadata={"note": "unit_test"},
    )

    assert saved_path.is_file()

    loaded = load_checkpoint(ckpt_file)
    assert loaded.schema_version == "2.1.0"
    assert loaded.generation == 1
    assert loaded.total_generations == 10
    assert len(loaded.population) == 4
    assert loaded.best_ever is not None
    assert loaded.best_ever.fitness == 30.0
    assert loaded.metadata["note"] == "unit_test"
    assert loaded.rng_state is not None


def test_engine_deterministic_checkpoint_resumption(tmp_path: Path):
    """
    Verifies that saving a checkpoint at generation 5 and resuming to generation 10
    continues smoothly and preserves all history.
    """
    cfg = EngineConfig(population_size=8, generations=10, seed=99)
    engine_initial = EvolutionEngine(config=cfg)

    # Run for 5 generations and save checkpoint
    ckpt_dir = tmp_path / "checkpoints"
    report_5 = engine_initial.run(generations=5, checkpoint_every=5, checkpoint_dir=ckpt_dir)
    assert report_5["total_generations"] == 5

    ckpt_file = ckpt_dir / "checkpoint_gen_0005.json"
    assert ckpt_file.is_file()

    # Now resume with a new engine instance from the checkpoint to generation 10
    engine_resumed = EvolutionEngine(config=cfg)
    report_10 = engine_resumed.run(generations=10, resume_from=ckpt_file)

    assert report_10["total_generations"] == 10
    assert len(report_10["history"]) == 10
    assert engine_resumed.best_ever is not None
    assert engine_resumed.best_ever.fitness >= report_5["best_individual"]["fitness"]


def test_repair_genome_checkpoint_serialization_roundtrip(tmp_path: Path):
    from evolab.repair import RepairGenome, RepairEdit

    edit = RepairEdit(kind="bool_flip", file="main.py", lineno=10, col_offset=4, payload=(("val", True),))
    genome = RepairGenome(
        sources={"main.py": "def foo():\n    return False\n"},
        target_file="main.py",
        edits=[edit],
    )
    ind = Individual(genome=genome, species="spec_code", fitness=95.0)

    rec = serialize_individual(ind)
    assert rec["genome"]["class_name"] == "RepairGenome"
    assert "target_file" in rec["genome"]["data"]

    restored = deserialize_individual(rec)
    assert isinstance(restored.genome, RepairGenome)
    assert restored.genome.target_file == "main.py"
    assert len(restored.genome.edits) == 1
    assert restored.genome.edits[0].kind == "bool_flip"
    assert restored.genome.edits[0].lineno == 10


def test_corrupt_genome_deserialization_raises_type_error():
    from evolab.checkpoint import _deserialize_genome

    bad_record = {
        "class_name": "NonExistentGenome",
        "module": "evolab.nonexistent",
        "data": {"corrupt": "data"},
    }
    with pytest.raises(TypeError, match="cannot restore genome of type NonExistentGenome"):
        _deserialize_genome(bad_record)


def test_gzip_checkpoint_save_and_transparent_load(tmp_path: Path):
    """Verifies that save_checkpoint with compress=True creates valid .json.gz and load_checkpoint reads it."""
    import random
    from evolab.checkpoint import save_checkpoint, load_checkpoint

    pop = [Individual(FloatGenome(values=[float(i), float(i * 2)]), species=f"spec_{i}", fitness=float(i * 10)) for i in range(10)]
    best = pop[-1]
    rng = random.Random(42)

    uncompressed_path = tmp_path / "plain.json"
    save_checkpoint(
        filepath=uncompressed_path,
        generation=3,
        total_generations=10,
        population=pop,
        best_ever=best,
        rng=rng,
        compress=False,
    )
    assert uncompressed_path.is_file()

    compressed_path = tmp_path / "compressed.json.gz"
    save_checkpoint(
        filepath=compressed_path,
        generation=3,
        total_generations=10,
        population=pop,
        best_ever=best,
        rng=rng,
        compress=True,
    )
    assert compressed_path.is_file()

    # Verify size reduction
    raw_size = uncompressed_path.stat().st_size
    gz_size = compressed_path.stat().st_size
    assert gz_size < raw_size

    # Transparent load of compressed file
    loaded_gz = load_checkpoint(compressed_path)
    assert loaded_gz.generation == 3
    assert len(loaded_gz.population) == 10
    assert loaded_gz.best_ever.fitness == 90.0

    # Transparent load when omitting .gz suffix if file exists as .gz
    loaded_auto = load_checkpoint(tmp_path / "compressed.json")
    assert loaded_auto.generation == 3
    assert len(loaded_auto.population) == 10


def test_engine_gzip_checkpoint_resumption(tmp_path: Path):
    """Verifies that engine saves compressed checkpoints and resumes deterministically."""
    from evolab.engine import EvolutionEngine
    from evolab.config import EngineConfig

    cfg = EngineConfig(population_size=8, generations=6, seed=123)
    engine_initial = EvolutionEngine(config=cfg)

    ckpt_dir = tmp_path / "gz_checkpoints"
    engine_initial.run(
        generations=3,
        checkpoint_every=3,
        checkpoint_dir=ckpt_dir,
        checkpoint_compress=True,
    )

    gz_file = ckpt_dir / "checkpoint_gen_0003.json.gz"
    assert gz_file.is_file()

    engine_resumed = EvolutionEngine(config=cfg)
    report = engine_resumed.run(generations=6, resume_from=gz_file)
    assert report["total_generations"] == 6
    assert len(report["history"]) == 6


def test_checkpoint_schema_version_custom_and_backward_compat(tmp_path: Path):
    """Verifies schema_version customization and backward compatibility default."""
    import random
    ckpt_file = tmp_path / "custom_schema.json"
    pop = [Individual(FloatGenome([1.0, 2.0]), species="spec_0", fitness=10.0)]
    save_checkpoint(
        filepath=ckpt_file,
        generation=1,
        total_generations=5,
        population=pop,
        best_ever=pop[0],
        rng=random.Random(1),
        schema_version="2.5.0",
    )
    loaded = load_checkpoint(ckpt_file)
    assert loaded.schema_version == "2.5.0"

    # Backward compatibility test: legacy JSON without schema_version field
    with open(ckpt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    del data["schema_version"]
    legacy_file = tmp_path / "legacy.json"
    with open(legacy_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    loaded_legacy = load_checkpoint(legacy_file)
    assert loaded_legacy.schema_version == "1.0.0"

    # Default schema version when saved without explicit parameter
    default_file = tmp_path / "default_schema.json"
    save_checkpoint(
        filepath=default_file,
        generation=1,
        total_generations=5,
        population=pop,
        best_ever=pop[0],
        rng=random.Random(1),
    )
    loaded_default = load_checkpoint(default_file)
    assert loaded_default.schema_version == "2.1.0"



