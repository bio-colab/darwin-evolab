"""
checkpoint.py — State Checkpointing, Freeze, and Resumption for darwin-evolab.

Provides safe, dependency-free, JSON-based serialization and restoration of
evolutionary engine states (populations, elites, RNG states, telemetry history).
"""
from __future__ import annotations

import copy
import importlib
import json
import logging
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .genome import FloatGenome, Individual

logger = logging.getLogger("evolab.checkpoint")


def _serialize_genome(genome: Any) -> dict[str, Any]:
    """Serializes a genome object to a JSON-encodable dictionary."""
    if hasattr(genome, "serialize") and callable(genome.serialize):
        data = genome.serialize()
    elif isinstance(genome, FloatGenome) or hasattr(genome, "values"):
        data = list(genome.values)
    elif hasattr(genome, "genes"):
        data = list(genome.genes)
    elif isinstance(genome, (list, tuple)):
        data = list(genome)
    else:
        data = str(genome)

    return {
        "class_name": genome.__class__.__name__,
        "module": genome.__class__.__module__,
        "data": data,
    }


def _deserialize_genome(record: dict[str, Any]) -> Any:
    """Reconstructs a genome from serialized data."""
    cls_name = record.get("class_name", "")
    module_name = record.get("module", "")
    data = record.get("data")

    if cls_name == "FloatGenome" or (not module_name and isinstance(data, list)):
        return FloatGenome(values=[float(x) for x in data])

    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        if hasattr(cls, "deserialize") and callable(cls.deserialize):
            return cls.deserialize(data)
        elif hasattr(cls, "from_dict") and callable(cls.from_dict):
            return cls.from_dict(data)
        elif hasattr(cls, "from_json") and callable(cls.from_json):
            return cls.from_json(data)
        else:
            return cls(data)
    except Exception as exc:
        logger.debug("Falling back to FloatGenome or raw data for %s.%s: %s", module_name, cls_name, exc)
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            return FloatGenome(values=[float(x) for x in data])
        return data


def serialize_individual(ind: Individual) -> dict[str, Any]:
    """Serializes an individual into a JSON-compatible dict."""
    return {
        "genome": _serialize_genome(ind.genome),
        "species": str(ind.species),
        "fitness": float(ind.fitness),
        "adjusted_fitness": float(ind.adjusted_fitness),
        "lineage": dict(ind.lineage) if ind.lineage else {},
        "last_evaluated_gen": int(ind.last_evaluated_gen),
        "generation": int(getattr(ind, "_generation", 0)),
        "index": int(getattr(ind, "_index", 0)),
    }


def deserialize_individual(record: dict[str, Any]) -> Individual:
    """Restores an Individual from a serialized record."""
    genome = _deserialize_genome(record["genome"])
    species = record.get("species", "spec_0")
    if not species.startswith("spec_"):
        species = f"spec_{species}"

    ind = Individual(
        genome=genome,
        species=species,
        fitness=float(record.get("fitness", 0.0)),
        adjusted_fitness=float(record.get("adjusted_fitness", 0.0)),
        lineage=record.get("lineage", {}),
        last_evaluated_gen=int(record.get("last_evaluated_gen", 0)),
        _generation=int(record.get("generation", 0)),
        _index=int(record.get("index", 0)),
    )
    return ind


@dataclass
class CheckpointData:
    """Container for complete evolutionary state snapshot."""
    generation: int
    total_generations: int
    population: list[Individual]
    best_ever: Individual | None
    rng_state: Any
    history: list[dict[str, Any]] = field(default_factory=list)
    species_history: list[dict[str, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def save_checkpoint(
    filepath: str | Path,
    generation: int,
    total_generations: int,
    population: list[Individual],
    best_ever: Individual | None,
    rng: random.Random | None,
    history: list[dict[str, Any]] | None = None,
    species_history: list[dict[str, int]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Atomically saves a full evolutionary engine snapshot to disk.
    
    Uses atomic write (temporary file followed by os.replace) to ensure
    no corrupted checkpoints exist if interrupted mid-save.
    """
    target_path = Path(filepath).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    rng_state_raw = rng.getstate() if rng is not None else None
    # Convert rng state tuple (version, state_tuple, gauss) to JSON list
    if rng_state_raw:
        rng_state_json = [
            rng_state_raw[0],
            list(rng_state_raw[1]),
            rng_state_raw[2],
        ]
    else:
        rng_state_json = None

    payload = {
        "version": 1,
        "timestamp": time.time(),
        "generation": int(generation),
        "total_generations": int(total_generations),
        "population": [serialize_individual(ind) for ind in population],
        "best_ever": serialize_individual(best_ever) if best_ever is not None else None,
        "rng_state": rng_state_json,
        "history": history or [],
        "species_history": species_history or [],
        "metadata": metadata or {},
    }

    # Atomic write pattern
    temp_dir = target_path.parent
    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
        json.dump(payload, tf, indent=2)
        temp_name = tf.name

    os.replace(temp_name, str(target_path))
    return target_path


def load_checkpoint(filepath: str | Path) -> CheckpointData:
    """
    Loads and deserializes an evolutionary state snapshot from disk.
    """
    src_path = Path(filepath).resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    generation = int(payload["generation"])
    total_generations = int(payload.get("total_generations", generation))
    population = [deserialize_individual(rec) for rec in payload.get("population", [])]
    best_ever = (
        deserialize_individual(payload["best_ever"])
        if payload.get("best_ever") is not None
        else None
    )

    rng_state_raw = payload.get("rng_state")
    if rng_state_raw and len(rng_state_raw) == 3:
        rng_state = (
            rng_state_raw[0],
            tuple(rng_state_raw[1]),
            rng_state_raw[2],
        )
    else:
        rng_state = None

    history = payload.get("history", [])
    species_history = payload.get("species_history", [])
    metadata = payload.get("metadata", {})
    timestamp = float(payload.get("timestamp", 0.0))

    return CheckpointData(
        generation=generation,
        total_generations=total_generations,
        population=population,
        best_ever=best_ever,
        rng_state=rng_state,
        history=history,
        species_history=species_history,
        metadata=metadata,
        timestamp=timestamp,
    )
