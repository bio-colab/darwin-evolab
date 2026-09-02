"""Regression: archive.py wired into the real execution path (scenarios/CLI).

Pins the integration contract:
  - prepare_electronics_run returns a cache-aware evaluator (GA __call__ path)
    that still exposes .evaluate() -> raw FitnessResult (introspection path)
  - scenario key embeds the evaluator spec fingerprint (stale-cache hazard)
  - evidence columns (tool_used / passed_holdout) are recorded
  - EVOLAB_ARCHIVE=0 returns the raw evaluator untouched
  - EVOLAB_ARCHIVE_SEED=n injects archived elites without displacing slot 0
"""
import os
import sqlite3

import pytest

from experimental.electronics.archive import (
    ArchivedEvaluatorProxy,
    EvaluationArchive,
    evaluator_spec_fingerprint,
)
from experimental.electronics.evaluators.digital_evaluator import MultiCorner74xxEvaluator
from experimental.electronics.scenarios import prepare_electronics_run


@pytest.fixture()
def archive_db(tmp_path, monkeypatch):
    db = tmp_path / "archive.db"
    monkeypatch.setenv("EVOLAB_ARCHIVE_DB", str(db))
    return db


def test_prepare_wraps_evaluator_and_caches(archive_db):
    ev, pop, name = prepare_electronics_run("half_adder", 6, seed=0)
    assert name == "half_adder"
    assert isinstance(ev, ArchivedEvaluatorProxy)
    assert ev.scenario_key.startswith("half_adder:spec-")

    # GA path: first call evaluates + records, second call is a cache hit
    s1 = ev(pop[0])
    s2 = ev(pop[0])
    assert s1 == s2
    assert ev.archived_fn.hits == 1 and ev.archived_fn.misses == 1

    # introspection path: evaluate() returns the raw FitnessResult, not a float
    res = ev.evaluate(pop[0])
    assert hasattr(res, "artifacts") and hasattr(res, "passed_holdout")
    assert float(res.score) == s1

    assert archive_db.exists()


def test_evidence_columns_recorded(archive_db):
    ev, pop, _ = prepare_electronics_run("analog_sizing", 3, seed=3)
    ev(pop[0])
    con = sqlite3.connect(str(archive_db))
    row = con.execute(
        "SELECT fitness, passed_holdout, tool_used FROM evaluations"
    ).fetchone()
    con.close()
    assert row is not None
    fitness, passed_holdout, tool_used = row
    assert fitness is not None
    assert passed_holdout is not None          # T2 hazard fixed: not NULL
    assert tool_used in ("ngspice", "analytical_fallback", "ngspice_no_metrics")


def test_spec_fingerprint_defeats_stale_cache(archive_db):
    from experimental.electronics.models.independent_verifier import IndependentDigitalVerifier

    table = [
        ((a, b), IndependentDigitalVerifier.half_adder_reference(a, b))
        for a in (0, 1)
        for b in (0, 1)
    ]
    loose = MultiCorner74xxEvaluator(table, max_delay_ns=150.0, max_quiescent_ua=60.0,
                                     functions_needed=("XOR", "AND"))
    strict = MultiCorner74xxEvaluator(table, max_delay_ns=40.0, max_quiescent_ua=60.0,
                                      functions_needed=("XOR", "AND"))
    fp_loose = evaluator_spec_fingerprint(loose)
    fp_strict = evaluator_spec_fingerprint(strict)
    assert fp_loose != fp_strict  # changed spec -> changed cache key

    # same genome evaluated under both specs must NOT collide in the archive
    ev1, pop, _ = prepare_electronics_run("half_adder", 4, seed=0)
    fp_genome = pop[0].genome.fingerprint()
    ev1(pop[0])
    con = sqlite3.connect(str(archive_db))
    scenarios = {r[0] for r in con.execute("SELECT DISTINCT scenario FROM evaluations")}
    con.close()
    assert len(scenarios) == 1
    key = scenarios.pop()
    # a strict re-spec reuses NO rows recorded under the loose spec
    from experimental.electronics.archive import ArchivedEvaluator

    strict_eval = ArchivedEvaluator(strict, EvaluationArchive(archive_db),
                                    scenario=f"half_adder:spec-{fp_strict}")
    strict_eval(pop[0])
    assert strict_eval.misses == 1 and strict_eval.hits == 0
    # and the loose row is still there under its own key
    con = sqlite3.connect(str(archive_db))
    n_scen = con.execute("SELECT COUNT(DISTINCT scenario) FROM evaluations").fetchone()[0]
    row_count = con.execute("SELECT COUNT(*) FROM evaluations WHERE fingerprint=?",
                            (fp_genome,)).fetchone()[0]
    con.close()
    assert n_scen == 2 and row_count == 2


def test_archive_env_disable(archive_db, monkeypatch):
    monkeypatch.setenv("EVOLAB_ARCHIVE", "0")
    ev, pop, name = prepare_electronics_run("half_adder", 4, seed=0)
    assert isinstance(ev, MultiCorner74xxEvaluator)  # raw evaluator, untouched
    assert not archive_db.exists()                   # nothing written


def test_elite_seeding_injects_archive_best(archive_db, monkeypatch):
    ev, pop, _ = prepare_electronics_run("half_adder", 6, seed=0)
    elite = pop[1]                      # a non-seed genome gets into the archive
    ev(elite)
    ev(pop[0])

    monkeypatch.setenv("EVOLAB_ARCHIVE_SEED", "1")
    ev2, pop2, _ = prepare_electronics_run("half_adder", 6, seed=99)
    fps = [ind.genome.fingerprint() for ind in pop2]
    assert pop2[0].genome.fingerprint() == [i.genome.fingerprint() for i in pop][0]
    assert elite.genome.fingerprint() in fps          # archived elite injected
    assert fps.index(elite.genome.fingerprint()) > 0  # never displaces slot 0
    assert len(pop2) == 6


def test_digital_cache_hit_identical_scores(archive_db):
    # determinism contract: cached value == freshly computed value
    ev, pop, _ = prepare_electronics_run("half_adder", 5, seed=0)
    fresh = ev.evaluate(pop[2]).score
    cached = ev(pop[2])
    again = ev(pop[2])
    assert fresh == cached == again
    assert ev.archived_fn.hits == 1
