"""Cumulative evaluation archive — genome-agnostic persistent cache + training log.

Problem this solves: every `run.py evolve` invocation starts from zero and
throws its results away when it exits. A genome evaluated today gets
re-evaluated from scratch tomorrow even if nothing about it changed.

This module adds one thing only: a durable, append-only record of every
(genome, scenario) -> (fitness, verification tier) pair ever computed, keyed
by the genome's own `fingerprint()` (already required by the EvolabGenome
contract, so this works for FloatGenome, CircuitNetlistGenome, and any future
genome type with zero changes to the engine or existing evaluators).

Two things it deliberately does NOT do (see LAB_NOTES.md discipline):
  - It does not build a surrogate/EDA model on top of the archive. That is a
    separate, later step once the archive has enough rows to train on.
  - It never lets a cheap verification tier masquerade as a strong one: a
    genome scored only via `analytical_fallback` is not returned as a cache
    hit when the caller asks for `ngspice`-verified data (min_tool_rank).

Usage (minimal, in a scenario builder):

    from experimental.electronics.archive import EvaluationArchive, ArchivedEvaluator

    archive = EvaluationArchive("experimental/electronics/data/archive.db")
    raw_evaluator, pop = _half_adder(population_size, rng)
    evaluator = ArchivedEvaluator(raw_evaluator, archive, scenario="half_adder")
    # ... run the GA with `evaluator` exactly as before ...
    print(archive.stats("half_adder"))
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_TOOL_RANK: dict[str, int] = {
    "ngspice": 2,
    "ngspice_transient": 2,
    # ngspice ran but no metric could be parsed: never treated as verified
    "ngspice_no_metrics": 0,
    "analytical_fallback": 1,
    "none": 0,
}


def _tool_rank(tool_used: str | None, ranks: dict[str, int]) -> int:
    return ranks.get(tool_used or "none", 0)


# --------------------------------------------------------------------------- #
# Genome codecs — reconstruction only needs each genome's own public fields.
# Register a codec once per genome type; everything else in this file is
# genome-agnostic.
# --------------------------------------------------------------------------- #


class GenomeCodec:
    def to_json(self, genome: Any) -> dict:
        raise NotImplementedError

    def from_json(self, data: dict) -> Any:
        raise NotImplementedError


class FloatGenomeCodec(GenomeCodec):
    def to_json(self, genome: Any) -> dict:
        return {"values": list(genome.values)}

    def from_json(self, data: dict) -> Any:
        from evolab.genome import FloatGenome

        return FloatGenome(values=list(data["values"]))


class CircuitNetlistGenomeCodec(GenomeCodec):
    def to_json(self, genome: Any) -> dict:
        return {
            "ic_packages": list(genome.ic_packages),
            "connections": [
                {
                    "src_ic": c.source.ic_index,
                    "src_pin": c.source.pin,
                    "dst_ic": c.destination.ic_index,
                    "dst_pin": c.destination.pin,
                }
                for c in genome.connections
            ],
            "num_inputs": genome.num_inputs,
            "num_outputs": genome.num_outputs,
            "functions_needed": list(genome.functions_needed),
        }

    def from_json(self, data: dict) -> Any:
        from experimental.electronics.models.circuit_netlist import (
            CircuitNetlistGenome,
            Connection,
            PinRef,
        )

        conns = [
            Connection(
                PinRef(c["src_ic"], c["src_pin"]),
                PinRef(c["dst_ic"], c["dst_pin"]),
            )
            for c in data["connections"]
        ]
        return CircuitNetlistGenome(
            data["ic_packages"],
            conns,
            data["num_inputs"],
            data["num_outputs"],
            functions_needed=tuple(data.get("functions_needed", ())),
        )


_CODECS_BY_TYPE: dict[type, GenomeCodec] = {}
_CODECS_BY_NAME: dict[str, GenomeCodec] = {}


def register_codec(genome_cls: type, codec: GenomeCodec) -> None:
    _CODECS_BY_TYPE[genome_cls] = codec
    _CODECS_BY_NAME[genome_cls.__name__] = codec


def get_codec(genome: Any) -> GenomeCodec:
    for cls, codec in _CODECS_BY_TYPE.items():
        if isinstance(genome, cls):
            return codec
    raise KeyError(
        f"No archive codec registered for {type(genome).__name__}. "
        f"Call register_codec(YourGenomeClass, YourCodec()) once at import time."
    )


def get_codec_by_name(name: str) -> GenomeCodec:
    if name not in _CODECS_BY_NAME:
        raise KeyError(f"No archive codec registered for genome type '{name}'")
    return _CODECS_BY_NAME[name]


def _register_builtin_codecs() -> None:
    try:
        from evolab.genome import FloatGenome

        register_codec(FloatGenome, FloatGenomeCodec())
    except ImportError:
        pass
    try:
        from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome

        register_codec(CircuitNetlistGenome, CircuitNetlistGenomeCodec())
    except ImportError:
        pass


_register_builtin_codecs()


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


@dataclass
class ArchivedResult:
    fitness: float
    passed_holdout: bool | None
    tool_used: str | None
    scenario: str
    genome_type: str
    created_at: float
    run_id: str


class EvaluationArchive:
    """Append-only SQLite log of every (genome, scenario) evaluation ever run."""

    def __init__(self, db_path: str | Path, tool_rank: dict[str, int] | None = None) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tool_rank = tool_rank or DEFAULT_TOOL_RANK
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                scenario TEXT NOT NULL,
                genome_type TEXT NOT NULL,
                genome_json TEXT NOT NULL,
                fitness REAL NOT NULL,
                passed_holdout INTEGER,
                tool_used TEXT,
                describe_json TEXT,
                run_id TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fp_scenario ON evaluations(fingerprint, scenario)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenario_fitness ON evaluations(scenario, fitness)"
        )
        self._conn.commit()

    def lookup(self, fingerprint: str, scenario: str) -> ArchivedResult | None:
        """Best (highest verification tier, then highest fitness) prior record."""
        cur = self._conn.execute(
            "SELECT fitness, passed_holdout, tool_used, genome_type, run_id, created_at "
            "FROM evaluations WHERE fingerprint=? AND scenario=?",
            (fingerprint, scenario),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        best = max(rows, key=lambda r: (_tool_rank(r[2], self.tool_rank), r[0]))
        fitness, passed_holdout, tool_used, genome_type, run_id, created_at = best
        return ArchivedResult(
            fitness=fitness,
            passed_holdout=None if passed_holdout is None else bool(passed_holdout),
            tool_used=tool_used,
            scenario=scenario,
            genome_type=genome_type,
            created_at=created_at,
            run_id=run_id,
        )

    def record(
        self,
        genome: Any,
        scenario: str,
        fitness: float,
        *,
        passed_holdout: bool | None = None,
        tool_used: str | None = None,
        run_id: str | None = None,
        codec: GenomeCodec | None = None,
    ) -> None:
        codec = codec or get_codec(genome)
        self._conn.execute(
            "INSERT INTO evaluations "
            "(fingerprint, scenario, genome_type, genome_json, fitness, passed_holdout, "
            " tool_used, describe_json, run_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                genome.fingerprint(),
                scenario,
                type(genome).__name__,
                json.dumps(codec.to_json(genome)),
                float(fitness),
                None if passed_holdout is None else int(passed_holdout),
                tool_used,
                json.dumps(genome.describe()),
                run_id or "unknown",
                time.time(),
            ),
        )
        self._conn.commit()

    def best(self, scenario: str, limit: int = 5) -> list[tuple[Any, ArchivedResult]]:
        """Top-`limit` distinct (by fingerprint) genomes ever recorded, reconstructed."""
        cur = self._conn.execute(
            "SELECT genome_type, genome_json, fitness, passed_holdout, tool_used, run_id, created_at "
            "FROM evaluations WHERE scenario=? ORDER BY fitness DESC LIMIT ?",
            (scenario, max(limit * 5, 25)),
        )
        seen: set[str] = set()
        out: list[tuple[Any, ArchivedResult]] = []
        for genome_type, genome_json, fitness, passed_holdout, tool_used, run_id, created_at in cur.fetchall():
            codec = get_codec_by_name(genome_type)
            genome = codec.from_json(json.loads(genome_json))
            fp = genome.fingerprint()
            if fp in seen:
                continue
            seen.add(fp)
            out.append(
                (
                    genome,
                    ArchivedResult(
                        fitness=fitness,
                        passed_holdout=None if passed_holdout is None else bool(passed_holdout),
                        tool_used=tool_used,
                        scenario=scenario,
                        genome_type=genome_type,
                        created_at=created_at,
                        run_id=run_id,
                    ),
                )
            )
            if len(out) >= limit:
                break
        return out

    def stats(self, scenario: str) -> dict[str, Any]:
        cur = self._conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT fingerprint), MAX(fitness), AVG(fitness) "
            "FROM evaluations WHERE scenario=?",
            (scenario,),
        )
        total, distinct, best_fit, avg_fit = cur.fetchone()
        total = total or 0
        distinct = distinct or 0
        return {
            "scenario": scenario,
            "total_evaluations": total,
            "distinct_genomes": distinct,
            "best_fitness": best_fit,
            "avg_fitness": avg_fit,
            "wasted_reevaluations": max(total - distinct, 0),
        }

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------- #
# Transparent evaluator wrapper — drop-in replacement for any (Individual)->float
# fitness_fn used by EvolutionEngine.
# --------------------------------------------------------------------------- #


class ArchivedEvaluator:
    """Wraps any (Individual)->float fitness_fn with the cumulative archive.

    A genome already evaluated for this scenario at an equal-or-better
    verification tier is returned from the archive instead of re-running the
    real evaluator. Everything else is recorded on the way out.
    """

    def __init__(
        self,
        fitness_fn: Callable[[Any], float],
        archive: EvaluationArchive,
        scenario: str,
        run_id: str | None = None,
        tool_used_fn: Callable[[], str | None] | None = None,
        passed_holdout_fn: Callable[[], bool | None] | None = None,
        min_tool_rank: int = 0,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.archive = archive
        self.scenario = scenario
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.tool_used_fn = tool_used_fn
        self.passed_holdout_fn = passed_holdout_fn
        self.min_tool_rank = min_tool_rank
        self.hits = 0
        self.misses = 0

    def __call__(self, individual: Any) -> float:
        genome = individual.genome
        fp = genome.fingerprint()
        cached = self.archive.lookup(fp, self.scenario)
        if cached is not None and _tool_rank(cached.tool_used, self.archive.tool_rank) >= self.min_tool_rank:
            self.hits += 1
            return cached.fitness

        self.misses += 1
        fitness = self.fitness_fn(individual)
        tool_used = self.tool_used_fn() if self.tool_used_fn else None
        passed_holdout = self.passed_holdout_fn() if self.passed_holdout_fn else None
        self.archive.record(
            genome,
            self.scenario,
            fitness,
            passed_holdout=passed_holdout,
            tool_used=tool_used,
            run_id=self.run_id,
        )
        return fitness


# --------------------------------------------------------------------------- #
# Elite carry-over across runs — the second half of "don't start from zero".
# --------------------------------------------------------------------------- #


def seed_population_from_archive(
    archive: EvaluationArchive,
    scenario: str,
    population_size: int,
    random_genome_fn: Callable[[], Any],
    species: str = "spec_electronics",
    n_elites: int = 1,
) -> list[Any]:
    """Builds an initial population seeded with the best genomes any past run
    ever found for this scenario, padded with fresh random individuals."""
    from evolab.genome import Individual

    elites = archive.best(scenario, limit=n_elites)
    pop = [Individual(genome=g, species=species) for g, _ in elites]
    while len(pop) < population_size:
        pop.append(Individual(genome=random_genome_fn(), species=species))
    return pop


# --------------------------------------------------------------------------- #
# Wiring into the real execution path (scenarios / CLI / probes).
# --------------------------------------------------------------------------- #


def evaluator_spec_fingerprint(evaluator: Any) -> str | None:
    """Stable hash of the evaluator's *spec* (targets, limits, truth table).

    The cache key is (genome fingerprint, scenario). A scenario's evaluator
    spec can change between runs (tighter max_delay, different targets); the
    spec hash is appended to the scenario key so a changed spec can never
    read rows produced under the old spec (the stale-cache hazard).
    Returns None for evaluators that do not expose ``spec_descriptor()``.
    """
    desc = getattr(evaluator, "spec_descriptor", None)
    if not callable(desc):
        return None
    try:
        blob = json.dumps(desc(), sort_keys=True, default=str)
    except Exception:
        return None
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


class EvidenceCapture:
    """Calls an Evaluator and keeps the last FitnessResult for provenance.

    ``Evaluator.__call__`` collapses a FitnessResult to a float, losing the
    tool tier and holdout verdict. Calling ``evaluate()`` directly keeps the
    evidence so the archive can store HOW a number was produced, not just the
    number.
    """

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator
        self.last: Any = None

    def __call__(self, individual: Any) -> float:
        res = self.evaluator.evaluate(individual)
        self.last = res if hasattr(res, "artifacts") else None
        return float(res.score)


class ArchivedEvaluatorProxy:
    """Engine-facing fitness_fn that keeps the full Evaluator surface.

    - ``__call__(ind) -> float``: cache-aware path used by EvolutionEngine.
    - ``evaluate(ind) -> FitnessResult``: RAW evaluation, deliberately bypassing
      the archive — introspection/tests need the real artifacts, not a cached
      number.
    - every other attribute delegates to the wrapped evaluator (defaults, dim,
      names, ...) so existing callers keep working unchanged.
    """

    def __init__(
        self,
        raw: Any,
        archived_fn: ArchivedEvaluator,
        archive: EvaluationArchive,
        scenario_key: str,
    ) -> None:
        self.raw = raw
        self.archived_fn = archived_fn
        self.archive = archive
        self.scenario_key = scenario_key

    def __call__(self, individual: Any) -> float:
        return self.archived_fn(individual)

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> Any:
        if context is not None:
            return self.raw.evaluate(target, context)
        return self.raw.evaluate(target)

    @property
    def deterministic(self) -> bool:
        return getattr(self.raw, "deterministic", True)

    @property
    def cost_estimate(self) -> str:
        return getattr(self.raw, "cost_estimate", "cheap")

    def stats(self) -> dict[str, Any]:
        out = dict(self.archive.stats(self.scenario_key))
        out["hits"] = self.archived_fn.hits
        out["misses"] = self.archived_fn.misses
        out["db"] = str(self.archive.path)
        return out

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.raw, attr)


def attach_archive(
    evaluator: Any,
    scenario_name: str,
    *,
    db_path: str | Path | None = None,
    min_tool_rank: int | None = None,
) -> tuple[Any, str, EvaluationArchive | None]:
    """Wires ``evaluator`` into the cumulative archive. Fail-safe by design:
    on any wiring error the raw evaluator is returned unchanged (the archive
    must never break a run).

    Environment knobs:
      - ``EVOLAB_ARCHIVE=0``        disable (returns the raw evaluator)
      - ``EVOLAB_ARCHIVE_DB``       alternate sqlite path
      - ``EVOLAB_ARCHIVE_MIN_RANK`` minimum verification tier served from cache
                                    (0=any tier, 2=ngspice-verified only)

    Returns ``(wired_evaluator, scenario_key, archive_or_None)``. The scenario
    key embeds the evaluator's spec fingerprint, so a changed spec (targets,
    limits, truth table) never reads rows recorded under the old spec.
    """
    flag = os.environ.get("EVOLAB_ARCHIVE", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return evaluator, scenario_name, None
    try:
        fp = evaluator_spec_fingerprint(evaluator)
        scenario_key = f"{scenario_name}:spec-{fp}" if fp else scenario_name
        path = (
            db_path
            or os.environ.get("EVOLAB_ARCHIVE_DB")
            or Path(__file__).resolve().parent / "data" / "archive.db"
        )
        archive = EvaluationArchive(path)
        if min_tool_rank is None:
            try:
                min_tool_rank = int(os.environ.get("EVOLAB_ARCHIVE_MIN_RANK", "0"))
            except ValueError:
                min_tool_rank = 0
        capture = EvidenceCapture(evaluator)

        def _tool_used() -> str | None:
            res = capture.last
            if res is None:
                return None
            return (res.artifacts or {}).get("tool_used")

        def _passed_holdout() -> bool | None:
            res = capture.last
            return getattr(res, "passed_holdout", None) if res is not None else None

        archived_fn = ArchivedEvaluator(
            capture,
            archive,
            scenario=scenario_key,
            tool_used_fn=_tool_used,
            passed_holdout_fn=_passed_holdout,
            min_tool_rank=min_tool_rank,
        )
        proxy = ArchivedEvaluatorProxy(evaluator, archived_fn, archive, scenario_key)
        return proxy, scenario_key, archive
    except Exception as exc:  # pragma: no cover - defensive
        import sys

        print(f"warning: archive disabled ({exc})", file=sys.stderr)
        return evaluator, scenario_name, None


_DEFAULT_SPECIES = "spec_electronics"


def seed_population_with_elites(
    archive: EvaluationArchive | None,
    scenario_key: str,
    pop: list[Any],
    n_elites: int,
    species: str = _DEFAULT_SPECIES,
) -> list[Any]:
    """Replaces trailing random slots with the best archived genomes.

    Slot 0 (the scenario's deterministic proposal seed) is never displaced,
    and elites whose fingerprint already appears in the population are
    skipped, so seeding never shrinks diversity.
    """
    if archive is None or n_elites <= 0 or len(pop) < 2:
        return pop
    from evolab.genome import Individual

    existing: set[str] = set()
    for ind in pop:
        try:
            existing.add(ind.genome.fingerprint())
        except Exception:
            pass
    idx = len(pop) - 1
    for genome, _res in archive.best(scenario_key, limit=n_elites + len(existing)):
        if idx < 1:
            break
        try:
            fp = genome.fingerprint()
        except Exception:
            continue
        if fp in existing:
            continue
        pop[idx] = Individual(genome=genome, species=species)
        existing.add(fp)
        idx -= 1
    return pop
