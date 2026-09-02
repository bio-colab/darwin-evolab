"""Realistic code benchmark fixtures inspired by popular Python libraries (Click, Requests, LRU, Config)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evaluators import FunctionTestEvaluator


@dataclass
class CodeScenario:
    """A realistic software evolution scenario with source files, test suite, and holdout cases."""

    name: str
    description: str
    sources: dict[str, str]
    target_file: str
    func_name: str
    test_cases: list[tuple[tuple[Any, ...], Any]]
    holdout_cases: list[tuple[tuple[Any, ...], Any]] = field(default_factory=list)
    difficulty: str = "medium"

    def create_evaluator(self) -> FunctionTestEvaluator:
        return FunctionTestEvaluator(
            base_sources=self.sources,
            target_file=self.target_file,
            func_name=self.func_name,
            test_cases=self.test_cases,
            holdout_cases=self.holdout_cases,
        )


def scenario_click_parser() -> CodeScenario:
    """Click-inspired CLI Argument & Option Parser scenario.

    Bugs:
    1. Default debug is True instead of False.
    2. Flag '--debug' sets False instead of True.
    3. Option '--port=X' leaves value as string instead of int conversion.
    4. Option '--host=X' extracts split('=')[0] instead of [1].
    """
    buggy_code = (
        "def parse_cli(args):\n"
        "    config = {'port': 8000, 'debug': False, 'host': '127.0.0.1'}\n"
        "    for arg in args:\n"
        "        if arg == '--debug':\n"
        "            config['debug'] = False\n"
        "        elif arg.startswith('--port='):\n"
        "            config['port'] = arg.split('=')[1]\n"
        "        elif arg.startswith('--host='):\n"
        "            config['host'] = arg.split('=')[0]\n"
        "    return config\n"
    )

    test_cases = [
        ((["--debug"],), {"port": 8000, "debug": True, "host": "127.0.0.1"}),
        ((["--port=9090"],), {"port": 9090, "debug": False, "host": "127.0.0.1"}),
        ((["--host=0.0.0.0", "--debug"],), {"port": 8000, "debug": True, "host": "0.0.0.0"}),  # nosec B104  # Test fixture parameter string for CLI parsing, not a live network bind.
        (([],), {"port": 8000, "debug": False, "host": "127.0.0.1"}),
    ]

    holdout_cases = [
        ((["--port=3000", "--host=localhost"],), {"port": 3000, "debug": False, "host": "localhost"}),
    ]

    return CodeScenario(
        name="click_cli_parser",
        description="Click-inspired command-line option parser and type conversion",
        sources={"cli_parser.py": buggy_code},
        target_file="cli_parser.py",
        func_name="parse_cli",
        test_cases=test_cases,
        holdout_cases=holdout_cases,
        difficulty="medium",
    )


def scenario_requests_auth_url() -> CodeScenario:
    """Requests-inspired HTTP Helper: URL Query Params & Auth Header formatting.

    Bugs:
    1. Header is missing the auth_type prefix formatting (e.g. 'Bearer secret' vs 'secret').
    2. Query string joins parameters with ',' instead of standard '&'.
    """
    buggy_code = (
        "def format_request(auth_type, token, params):\n"
        "    header = f'{token}' if token else ''\n"
        "    items = [f'{k}={v}' for k, v in sorted(params.items())]\n"
        "    query = ','.join(items)\n"
        "    return {'Authorization': header, 'query_string': query}\n"
    )

    test_cases = [
        (("Bearer", "secret_123", {"q": "search", "limit": 10}),
         {"Authorization": "Bearer secret_123", "query_string": "limit=10&q=search"}),
        (("Basic", "dXNlcjpwYXNz", {}),
         {"Authorization": "Basic dXNlcjpwYXNz", "query_string": ""}),
        (("", "", {"tag": "ga"}),
         {"Authorization": "", "query_string": "tag=ga"}),
    ]

    holdout_cases = [
        (("Bearer", "token_xyz", {"page": 2, "sort": "asc"}),
         {"Authorization": "Bearer token_xyz", "query_string": "page=2&sort=asc"}),
    ]

    return CodeScenario(
        name="requests_http_helper",
        description="Requests-inspired HTTP auth headers and query string serialization",
        sources={"http_helpers.py": buggy_code},
        target_file="http_helpers.py",
        func_name="format_request",
        test_cases=test_cases,
        holdout_cases=holdout_cases,
        difficulty="easy",
    )


def scenario_lru_cache_logic() -> CodeScenario:
    """LRU Cache eviction policy and chronological retention scenario.

    Bugs:
    1. When existing key is accessed again, it is NOT moved to most-recently-used position.
    2. When capacity is exceeded, it pops index -1 (MRU) instead of index 0 (LRU).
    """
    buggy_code = (
        "def manage_lru(access_history, capacity):\n"
        "    cache = []\n"
        "    for k in access_history:\n"
        "        if k in cache:\n"
        "            pass\n"
        "        elif len(cache) >= capacity:\n"
        "            cache.pop(-1)\n"
        "        cache.append(k)\n"
        "    return cache\n"
    )

    test_cases = [
        ((["a", "b", "c", "a", "d"], 3), ["b", "c", "a", "d"] if False else ["c", "a", "d"]),
        ((["x", "y", "z"], 2), ["y", "z"]),
        ((["k", "k", "k"], 2), ["k"]),
        ((["1", "2", "3", "2", "4"], 3), ["3", "2", "4"]),
        (([], 3), []),
    ]

    holdout_cases = [
        ((["1", "2", "3", "4", "2", "5"], 3), ["4", "2", "5"]),
    ]

    return CodeScenario(
        name="lru_cache_logic",
        description="LRU Cache eviction policy and retention calculation",
        sources={"lru_store.py": buggy_code},
        target_file="lru_store.py",
        func_name="manage_lru",
        test_cases=test_cases,
        holdout_cases=holdout_cases,
        difficulty="medium",
    )


def scenario_multi_file_config() -> CodeScenario:
    """Multi-file configuration loader and validator scenario.

    Bugs:
    1. Validator bounds are inverted (100, 0 instead of 0, 100).
    2. Alert threshold boolean comparison is inverted (< instead of >=).
    """
    validator_code = (
        "def validate_bounds(val, min_val, max_val):\n"
        "    return min_val <= val <= max_val\n"
    )
    loader_code = (
        "from validator import validate_bounds\n\n"
        "def process_metrics(name, count, threshold):\n"
        "    is_valid = validate_bounds(count, 100, 0)\n"
        "    is_above = count < threshold\n"
        "    return {'name': name, 'valid': is_valid, 'alert': is_above}\n"
    )

    sources = {
        "validator.py": validator_code,
        "config_loader.py": loader_code,
    }

    test_cases = [
        (("cpu", 75, 70), {"name": "cpu", "valid": True, "alert": True}),
        (("ram", 40, 80), {"name": "ram", "valid": True, "alert": False}),
        (("disk", 120, 90), {"name": "disk", "valid": False, "alert": True}),
    ]

    holdout_cases = [
        (("gpu", 50, 50), {"name": "gpu", "valid": True, "alert": True}),
    ]

    return CodeScenario(
        name="multi_file_config",
        description="Multi-file configuration and validation across modules",
        sources=sources,
        target_file="config_loader.py",
        func_name="process_metrics",
        test_cases=test_cases,
        holdout_cases=holdout_cases,
        difficulty="hard",
    )


SCENARIO_REGISTRY = {
    "click_cli_parser": scenario_click_parser,
    "requests_http_helper": scenario_requests_auth_url,
    "lru_cache_logic": scenario_lru_cache_logic,
    "multi_file_config": scenario_multi_file_config,
}


def get_all_scenarios() -> list[CodeScenario]:
    return [fn() for fn in SCENARIO_REGISTRY.values()]


def _as_case(raw: Any) -> tuple[tuple[Any, ...], Any]:
    if isinstance(raw, dict):
        args = raw.get("args", ())
        expected = raw.get("expected")
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        args, expected = raw
    else:
        raise ValueError(f"invalid test case: {raw!r}")
    if not isinstance(args, (list, tuple)):
        args = (args,)
    return (tuple(args), expected)


def load_scenario_file(path: str | Path) -> CodeScenario:
    """Load a CodeScenario from JSON (decoupled from engine code)."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    sources = dict(data.get("sources") or {})
    files = data.get("source_files") or {}
    base = p.parent
    for name, rel in files.items():
        sources[name] = (base / rel).read_text(encoding="utf-8")
    if not sources:
        raise ValueError("scenario file needs sources or source_files")
    target = data.get("target_file") or next(iter(sources))
    func_name = data.get("func_name")
    if not func_name:
        raise ValueError("scenario file needs func_name")
    return CodeScenario(
        name=str(data.get("name") or p.stem),
        description=str(data.get("description") or ""),
        sources=sources,
        target_file=target,
        func_name=str(func_name),
        test_cases=[_as_case(c) for c in data.get("test_cases") or []],
        holdout_cases=[_as_case(c) for c in data.get("holdout_cases") or []],
        difficulty=str(data.get("difficulty") or "custom"),
    )


def load_source_scenario(
    source_paths: list[str | Path],
    tests_path: str | Path,
    func_name: str,
    target_file: str | None = None,
) -> CodeScenario:
    sources: dict[str, str] = {}
    for raw in source_paths:
        fp = Path(raw)
        sources[fp.name] = fp.read_text(encoding="utf-8")
    if not sources:
        raise ValueError("no source files")
    payload = json.loads(Path(tests_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"test_cases": payload}
    target = target_file or next(iter(sources))
    return CodeScenario(
        name="custom",
        description="CLI custom sources",
        sources=sources,
        target_file=target,
        func_name=func_name,
        test_cases=[_as_case(c) for c in payload.get("test_cases") or []],
        holdout_cases=[_as_case(c) for c in payload.get("holdout_cases") or []],
        difficulty="custom",
    )


def make_code_population(
    scenario: CodeScenario,
    size: int,
    rng,
    species: str = "spec_code",
    avoid_loci: set | None = None,
    redraws: int = 0,
    seed_keys: list[tuple[str, int, int, str]] | None = None,
    seed_count: int = 0,
):
    """Build a repair-gene population from the scenario sources.

    M8 — trap-aware initialization: ``avoid_loci`` is a set of dead
    ``(file, lineno, col_offset, kind)`` doors (ExperienceStore.avoidance_set)
    consulted by every initial mutation (negative genetic memory: memory
    decides which genotypes exist at generation 0). ``redraws`` requests
    ``redraws`` extra unconditional mutation draws per individual, keeping
    the last — the mechanism-free perturbation arm of the M8 protocol
    (isolates init re-draw noise from memory-directed avoidance).

    M9 — composition-seeded initialization: ``seed_keys`` is a list of
    remembered ``(file, lineno, col_offset, kind)`` edits mined from a
    successful multi-edit composition (ExperienceStore.composition_seeds);
    ``seed_count`` individuals each receive exactly ONE of those edits
    (round-robin: mutating slot i gets ``seed_keys[i % len(seed_keys)]``).
    k=1 is replay-proof by construction — every single-edit genotype is a
    proven dead door on these benchmarks (committed M8 firecheck), so a
    seeded individual can never pass at generation zero and the search
    keeps real work; full-composition seeding would be the cache/archive
    value class and is deliberately unsupported. Seeded individuals are
    constructed deterministically and consume NO rng draws; the remaining
    individuals follow the exact legacy path. A seed key whose locus is
    absent from the catalog, or whose kind does not match the catalog's
    edit at that locus, is skipped defensively — that slot falls back to a
    legacy mutation draw. ``seed_keys`` None/empty with ``seed_count=0``
    is byte-for-byte the original behavior (same RNG stream, same
    population); ``seed_count > 0`` without keys is a caller error.

    ``avoid_loci=None/empty`` with ``redraws=0`` is byte-for-byte the
    original behavior (same RNG stream, same population).
    """
    from .genome import Individual
    from .repair import RepairGenome, catalog_sources

    if redraws < 0:
        raise ValueError("redraws must be >= 0")
    if seed_count < 0:
        raise ValueError("seed_count must be >= 0")
    if seed_count > 0 and not seed_keys:
        raise ValueError("seed_count > 0 requires non-empty seed_keys")

    seed_slots: list = []
    if seed_keys and seed_count > 0:
        catalog = {e.locus(): e for e in catalog_sources(scenario.sources)}
        for i in range(seed_count):
            key = seed_keys[i % len(seed_keys)]
            if len(key) != 4:
                raise ValueError(
                    f"seed_keys[{i}] must be (file, lineno, col_offset, kind)"
                )
            edit = catalog.get((key[0], key[1], key[2]))
            seed_slots.append(
                edit if (edit is not None and edit.kind == key[3]) else None
            )

    seed = RepairGenome(
        sources=dict(scenario.sources),
        target_file=scenario.target_file,
        edits=[],
    )
    population = [Individual(genome=seed.clone(), species=species)]
    slot = 0
    while len(population) < size:
        planted = seed_slots[slot] if slot < len(seed_slots) else None
        slot += 1
        if planted is not None:
            child = RepairGenome(
                sources=dict(scenario.sources),
                target_file=scenario.target_file,
                edits=[planted],
            )
        else:
            child = seed.clone().mutate(rng=rng, avoid_loci=avoid_loci)
            for _ in range(redraws):
                child = seed.clone().mutate(rng=rng, avoid_loci=avoid_loci)
        population.append(Individual(genome=child, species=species))
    return population
