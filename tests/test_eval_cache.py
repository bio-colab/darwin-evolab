"""Tests for the program-keyed evaluation cache (M3, CK key swap).

The probe (scripts/probe_duplicate_evals.py) measured 88.2% duplicate
evaluations; the cache is the memory that recalls them. The non-negotiable
properties under test: a cache hit is indistinguishable from a fresh
evaluation (identical scores, identical search trajectories, side-effect
state replayed), wrong answers are impossible (different APPLIED TEXT is
never a hit, nondeterministic evaluators are never wrapped, unsupported
genomes bypass), and the accounting is honest.

CK swap (registered protocol scripts/ab_cache_key_swap.py): identity is the
MATERIALIZED APPLIED SOURCES (canonical JSON of RepairGenome.apply_to()),
not the edit recipe. Recipes reaching the same text merge by design (same
program => same result by evaluator determinism); the historical recipe key
treated them as distinct — the measured waste (click 54.8% realized vs
72.8% within-run ceiling).
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab import EngineConfig, EvolutionEngine, EvaluationCache, attach_eval_cache
from evolab import attach_experience_recorder, ExperienceStore
from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population
from evolab.evaluators import FitnessResult, FunctionTestEvaluator
from evolab.experience import EvaluationCache as CacheDirect
from evolab.repair import RepairEdit, RepairGenome


BASE = "def f(x):\n    return x + 1\n"

# fixtures where the catalog edits have REAL text effects:
BASE_WRAP = 'def f(d):\n    d["k"] = 1\n    return d\n'   # int_wrap@2:5 -> d["k"] = int(1)
BASE_BOOL = "def f(b):\n    return True\n"                # bool_flip@2:12 -> return False
BASE_MULTI = 'def f(d, flag):\n    d["k"] = flag\n    return True\n'  # both edits real


def make_genome(payload, lineno=2, col=15, kind="int_wrap", source=BASE):
    return RepairGenome(
        sources={"m.py": source},
        target_file="m.py",
        edits=[RepairEdit(kind=kind, file="m.py", lineno=lineno, col_offset=col, payload=payload)],
    )


class CountingRaw:
    """Minimal deterministic evaluator double that counts invocations."""

    def __init__(self, deterministic: bool = True) -> None:
        self.calls = 0
        self._deterministic = deterministic

    def evaluate(self, target, context=None):
        self.calls += 1
        return FitnessResult(score=50.0 + self.calls, passed_holdout=None, artifacts={})

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    def __call__(self, individual):
        return float(self.evaluate(individual).score)


class Collector:
    """Records every fitness_fn call score (engine-facing transparency probe)."""

    def __init__(self, inner):
        self.inner = inner
        self.scores: list[float] = []

    def __call__(self, individual) -> float:
        s = float(self.inner(individual))
        self.scores.append(s)
        return s

    def evaluate(self, target, context=None):
        res = self.inner.evaluate(target, context) if context is not None else self.inner.evaluate(target)
        self.scores.append(float(res.score))
        return res

    def __getattr__(self, attr):
        return getattr(self.inner, attr)


# ---------------------------------------------------------------------------
# identity & correctness
# ---------------------------------------------------------------------------

def test_same_program_is_one_miss_then_hits():
    raw = CountingRaw()
    cache = EvaluationCache(raw)
    g = make_genome(())
    r1 = cache.evaluate(g)
    r2 = cache.evaluate(g)
    assert raw.calls == 1
    assert cache.misses == 1 and cache.hits == 1
    assert r1.score == r2.score
    # de-aliased rebuild: mutating one result never poisons the cached one
    r2.artifacts["poison"] = True
    r3 = cache.evaluate(g)
    assert "poison" not in r3.artifacts
    assert r3.score == r1.score


def test_payload_change_ignored_by_applier_merges_same_program():
    # int_wrap's payload is NOT injected into the text (the wrap is derived
    # from the tree, repair.py:_apply_int_wrap) — these two recipes produce
    # byte-identical programs, so they merge: one entry, one raw call,
    # identical results. (The old recipe key billed them separately — the
    # measured path-duplication waste this swap removes.)
    raw = CountingRaw()
    cache = EvaluationCache(raw)
    a = make_genome((("i", 1), ("j", 2)))
    b = make_genome((("i", 2), ("j", 1)))
    assert EvaluationCache.program_identity(a) == EvaluationCache.program_identity(b)
    r1 = cache.evaluate(a)
    r2 = cache.evaluate(b)
    assert raw.calls == 1
    assert cache.misses == 1 and cache.hits == 1
    assert r1.score == r2.score


def test_text_change_is_never_a_hit():
    # The never-a-hit invariant, pinned where it lives now: edits producing
    # DIFFERENT applied text never share an entry (int_wrap on a real
    # Assign+Subscript vs bool_flip on a real bool constant).
    raw = CountingRaw()
    cache = EvaluationCache(raw)
    a = make_genome((), kind="int_wrap", source=BASE_WRAP)
    b = make_genome((), kind="bool_flip", source=BASE_BOOL)
    assert EvaluationCache.program_identity(a) != EvaluationCache.program_identity(b)
    cache.evaluate(a)
    cache.evaluate(b)
    assert raw.calls == 2 and cache.misses == 2 and cache.hits == 0


def test_edit_order_independent_recipes_merge():
    # Disjoint-locus edits applied in one pass over the original tree:
    # swapped orderings reach the same materialized text -> same key.
    raw = CountingRaw()
    cache = EvaluationCache(raw)
    e_wrap = RepairEdit(kind="int_wrap", file="m.py", lineno=2, col_offset=5, payload=())
    e_flip = RepairEdit(kind="bool_flip", file="m.py", lineno=3, col_offset=12, payload=())
    g1 = RepairGenome(sources={"m.py": BASE_MULTI}, target_file="m.py", edits=[e_wrap, e_flip])
    g2 = RepairGenome(sources={"m.py": BASE_MULTI}, target_file="m.py", edits=[e_flip, e_wrap])
    assert EvaluationCache.program_identity(g1) == EvaluationCache.program_identity(g2)
    cache.evaluate(g1)
    cache.evaluate(g2)
    assert raw.calls == 1 and cache.hits == 1


def test_non_target_file_edits_distinguished():
    # apply_to() covers ALL files: genomes identical in the target file but
    # differing in another file's applied text get different keys — this is
    # why the key hashes apply_to() and not to_code() (which would
    # under-differentiate multi-file edits).
    raw = CountingRaw()
    cache = EvaluationCache(raw)
    g1 = RepairGenome(sources={"m.py": BASE, "o.py": "x = True\n"}, target_file="m.py", edits=[])
    flip = RepairEdit(kind="bool_flip", file="o.py", lineno=1, col_offset=5, payload=())
    g2 = RepairGenome(sources={"m.py": BASE, "o.py": "x = True\n"}, target_file="m.py", edits=[flip])
    assert EvaluationCache.program_identity(g1) != EvaluationCache.program_identity(g2)
    assert g1.to_code() == g2.to_code()  # to_code() alone would have collided
    cache.evaluate(g1)
    cache.evaluate(g2)
    assert raw.calls == 2 and cache.misses == 2 and cache.hits == 0


def test_unsupported_genome_bypasses_never_lies():
    raw = CountingRaw()
    cache = EvaluationCache(raw)
    res = cache.evaluate(object())  # no identity strategy
    assert raw.calls == 1 and cache.bypasses == 1
    assert res.score == 51.0  # first raw call (CountingRaw scores 50.0 + call#)


def test_broken_to_code_falls_through_to_raw():
    class Boom:
        def to_code(self):
            raise RuntimeError("boom")

    raw = CountingRaw()
    cache = EvaluationCache(raw)
    res = cache.evaluate(Boom())
    assert raw.calls == 1 and cache.bypasses == 1
    assert res.score == 51.0  # raw result passed through untouched


def test_hit_timing_is_honest_zero():
    raw = CountingRaw()
    cache = EvaluationCache(raw)
    g = make_genome(())
    r1 = cache.evaluate(g)
    r2 = cache.evaluate(g)
    assert r1.evaluation_time_ms >= 0.0
    assert r2.evaluation_time_ms == 0.0  # no raw work happened


# ---------------------------------------------------------------------------
# side-effect replay (the stale-suspicion trap)
# ---------------------------------------------------------------------------

class StatefulRaw(CountingRaw):
    cacheable_state_attrs = ("last_suspicion_map",)

    def __init__(self):
        super().__init__()
        self.last_suspicion_map = None


def test_side_effect_state_replayed_on_hit():
    raw = StatefulRaw()
    raw.evaluate = lambda target, context=None: (
        _eval_with_map(raw, target)
    )
    cache = EvaluationCache(raw)
    # distinct APPLIED TEXT (real edits on distinct fixtures) -> distinct
    # entries; identical recipes with no text effect would merge under the
    # materialized identity (same program => same map by determinism).
    a = make_genome((), kind="int_wrap", source=BASE_WRAP)
    b = make_genome((), kind="bool_flip", source=BASE_BOOL)
    cache.evaluate(a)          # sets map for A
    map_a = raw.last_suspicion_map
    cache.evaluate(b)          # sets map for B
    map_b = raw.last_suspicion_map
    assert map_a != map_b
    cache.evaluate(a)          # HIT: state must return to A's map
    assert raw.last_suspicion_map == map_a
    cache.evaluate(b)
    assert raw.last_suspicion_map == map_b


def _eval_with_map(raw, target):
    raw.calls += 1
    raw.last_suspicion_map = {"lines": [raw.calls]}
    return FitnessResult(score=50.0 + raw.calls)


def test_function_test_evaluator_declares_cacheable_state():
    ev = FunctionTestEvaluator(
        base_sources={"m.py": BASE}, target_file="m.py", func_name="f",
        test_cases=[((1,), 2)], holdout_cases=None,
    )
    assert "last_suspicion_map" in ev.cacheable_state_attrs


# ---------------------------------------------------------------------------
# attach guardrails
# ---------------------------------------------------------------------------

def test_attach_refuses_nondeterministic():
    raw = CountingRaw(deterministic=False)
    assert attach_eval_cache(raw) is raw


def test_kill_switch_env(monkeypatch):
    raw = CountingRaw()
    monkeypatch.setenv("EVOLAB_EVAL_CACHE", "0")
    assert attach_eval_cache(raw) is raw
    monkeypatch.delenv("EVOLAB_EVAL_CACHE")
    cache = attach_eval_cache(raw)
    assert isinstance(cache, EvaluationCache)
    cache.enabled = False
    cache.evaluate(make_genome(()))
    cache.evaluate(make_genome(()))
    assert raw.calls == 2  # disabled cache is a pure pass-through


def test_import_surface():
    import evolab

    assert evolab.EvaluationCache is CacheDirect
    assert callable(evolab.attach_eval_cache)


# ---------------------------------------------------------------------------
# trajectory transparency (the contract that matters most)
# ---------------------------------------------------------------------------

def _run_scenario_with(wrapped, scenario, seed, generations=8, population=12):
    engine = EvolutionEngine(
        fitness_fn=wrapped,
        config=EngineConfig(
            generations=generations,
            population_size=population,
            elite_count=1,
            genome_size=2,
            seed=seed,
        ),
    )
    initial = make_code_population(scenario, population, random.Random(seed))
    engine.run(generations=generations, initial_population=initial)
    return engine


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_trajectory_byte_identical_with_and_without_cache(seed):
    scenario = SCENARIO_REGISTRY["requests_http_helper"]()

    plain = Collector(scenario.create_evaluator())
    engine_plain = _run_scenario_with(plain, scenario, seed)

    scenario2 = SCENARIO_REGISTRY["requests_http_helper"]()
    cached = Collector(attach_eval_cache(scenario2.create_evaluator()))
    engine_cached = _run_scenario_with(cached, scenario2, seed)

    assert plain.scores == cached.scores          # identical call-for-call scores
    assert engine_plain.best_ever.fitness == engine_cached.best_ever.fitness
    stats = cached.inner.stats
    assert stats["hits"] > 0                      # the cache actually engaged
    assert stats["misses"] + stats["hits"] == len(cached.scores)


def test_recorder_sees_every_call_with_cache_inside():
    scenario = SCENARIO_REGISTRY["requests_http_helper"]()
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "exp.db"
        run_id = "cache_inside"
        ev = scenario.create_evaluator()
        wired = attach_experience_recorder(
            attach_eval_cache(ev),
            scenario.sources,
            scenario.target_file,
            scenario.func_name,
            db_path=db,
            run_id=run_id,
            prior_enabled=False,
        )
        engine = EvolutionEngine(
            fitness_fn=wired,
            config=EngineConfig(
                generations=4, population_size=8, elite_count=1, genome_size=2, seed=7,
            ),
        )
        initial = make_code_population(scenario, 8, random.Random(7))
        engine.run(generations=4, initial_population=initial)
        store = ExperienceStore(db)
        metrics = store.run_metrics(run_id)
        store.close()
        # recorder observes every engine call; cache hits record eval_ms 0.0
        cache = wired.raw
        assert cache.hits > 0
        assert metrics["evals_total"] == cache.hits + cache.misses
        wired.close()
