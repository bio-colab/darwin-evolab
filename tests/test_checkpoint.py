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
