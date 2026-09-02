"""M8 — trap-aware initialization: dead-door mining and the genetic veto.

Covers the three M8 contracts:

  1. ``ExperienceStore.avoidance_set`` — dead (kind, locus) doors mined
     from SINGLE-edit failures only; one holdout success rescues a door
     forever; fail-safe ``None`` when memory is absent or broken.
  2. ``RepairGenome.mutate(avoid_loci=...)`` — the bounded veto gate:
     None/empty is byte-identical legacy behavior; vetoes re-draw inside
     the same candidate pool (SBFL narrowing is never overridden); the
     veto applies AFTER the kind prior.
  3. ``make_code_population(avoid_loci=..., redraws=...)`` — the genetic
     injection point: memory decides which genotypes exist at generation
     0; ``redraws`` is the mechanism-free perturbation knob used by the
     registered isolation arm.

Plus the harness decision rule (importlib, same idiom as test_ab_metrics).
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population  # noqa: E402
from evolab.experience import ExperienceStore  # noqa: E402
from evolab.repair import RepairGenome, catalog_sources  # noqa: E402

FP = "m8-unit-fingerprint"


def _row(eval_index, kinds, loci, holdout, outcome, run_id="r1", score=0.0):
    return dict(
        run_id=run_id,
        eval_index=eval_index,
        problem_fingerprint=FP,
        func_name="f",
        target_file="t.py",
        genome_class="RepairGenome",
        edit_kinds=kinds,
        edit_loci=loci,
        n_edits=len(kinds),
        score=score,
        fitness_delta=None,
        is_new_best=0,
        passed_holdout=holdout,
        eval_ms=0.0,
        outcome=outcome,
    )


def _dead_key(tag, line, col=0):
    return ("t.py", line, col, tag)


# ---------- 1. avoidance_set mining ----------

def test_avoidance_set_mines_dead_doors(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    store.record(_row(1, ["compare_flip"], [["t.py", 10, 4]], 0, "failed"))
    store.record(_row(2, ["compare_flip"], [["t.py", 10, 4]], 0, "neutral"))
    store.record(_row(3, ["compare_flip"], [["t.py", 10, 4]], 0, "error"))
    # one success rescues a door forever, regardless of failures
    store.record(_row(4, ["index_flip"], [["t.py", 12, 0]], 0, "failed"))
    store.record(_row(5, ["index_flip"], [["t.py", 12, 0]], 1, "success", score=100.0))
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    dead = store.avoidance_set(FP, min_failures=2)
    store.close()
    assert dead == {_dead_key("compare_flip", 10, col=4)}


def test_avoidance_set_respects_min_failures(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    store.record(_row(1, ["int_wrap"], [["t.py", 7, 0]], 0, "failed"))
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    assert store.avoidance_set(FP, min_failures=2) == set()  # consulted, nothing qualified
    assert store.avoidance_set(FP, min_failures=1) == {_dead_key("int_wrap", 7)}
    store.close()


def test_avoidance_set_ignores_multi_edit_rows(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    # two failing MULTI-edit rows touching (kind, locus) D — per-edit credit
    # is unknowable, so D must never be mined
    for i in range(2):
        store.record(_row(
            i + 1,
            ["compare_flip", "int_wrap"],
            [["t.py", 20, 2], ["t.py", 30, 0]],
            0,
            "failed",
        ))
    # one qualifying single-edit row elsewhere -> a set is returned (not None),
    # proving the store was consulted and multi-edit rows were simply skipped
    store.record(_row(3, ["int_wrap"], [["t.py", 7, 0]], 0, "failed"))
    store.record(_row(4, ["int_wrap"], [["t.py", 7, 0]], 0, "failed"))
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    dead = store.avoidance_set(FP, min_failures=2)
    store.close()
    assert dead == {_dead_key("int_wrap", 7)}
    assert _dead_key("compare_flip", 20, col=2) not in dead
    assert _dead_key("int_wrap", 30) not in dead
    # and with ONLY multi-edit rows for the fingerprint -> None by contract
    store2 = ExperienceStore(tmp_path / "y.db")
    for i in range(2):
        store2.record(_row(
            i + 1,
            ["compare_flip", "int_wrap"],
            [["t.py", 20, 2], ["t.py", 30, 0]],
            0,
            "failed",
            run_id="r2",
        ))
    store2.close()
    store2 = ExperienceStore(tmp_path / "y.db")
    assert store2.avoidance_set(FP, min_failures=2) is None
    store2.close()


def test_avoidance_set_none_when_no_single_edit_data(tmp_path):
    store = ExperienceStore(tmp_path / "empty.db")
    assert store.avoidance_set(FP) is None  # empty store
    store.close()
    store = ExperienceStore(tmp_path / "other.db")
    store.record(_row(1, ["int_wrap"], [["t.py", 7, 0]], 0, "failed"))
    store.close()
    store = ExperienceStore(tmp_path / "other.db")
    assert store.avoidance_set("different-fingerprint") is None  # rows exist, other fp
    store.close()


def test_avoidance_set_broken_store_returns_none(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    store.record(_row(1, ["int_wrap"], [["t.py", 7, 0]], 0, "failed"))
    store.close()
    # closed connection -> sqlite3.ProgrammingError (a sqlite3.Error) -> None
    assert store.avoidance_set(FP) is None


def test_avoidance_set_max_entries_keeps_most_attempted(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    idx = 1
    plan = [("k1", 1, 5), ("k2", 2, 4), ("k3", 3, 3), ("k4", 4, 2), ("k5", 5, 2)]
    for tag, line, n in plan:
        for _ in range(n):
            store.record(_row(idx, [tag], [["t.py", line, 0]], 0, "failed"))
            idx += 1
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    dead = store.avoidance_set(FP, min_failures=2, max_entries=3)
    store.close()
    assert dead == {_dead_key("k1", 1), _dead_key("k2", 2), _dead_key("k3", 3)}


def test_avoidance_set_rejects_nonpositive_min_failures(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    with pytest.raises(ValueError):
        store.avoidance_set(FP, min_failures=0)
    store.close()


# ---------- 2. the genetic veto in mutate ----------

def _scenario():
    return SCENARIO_REGISTRY["requests_http_helper"]()


def _seed(scenario):
    return RepairGenome(
        sources=dict(scenario.sources),
        target_file=scenario.target_file,
        edits=[],
    )


def _key(edit):
    return (edit.file, edit.lineno, edit.col_offset, edit.kind)


def test_mutate_avoid_vetoes_the_drawn_door():
    scenario = _scenario()
    seed = _seed(scenario)
    plain = seed.mutate(rng=random.Random(9))
    door = _key(plain.edits[0])
    avoided = seed.mutate(rng=random.Random(9), avoid_loci={door})
    assert _key(avoided.edits[0]) != door


def test_mutate_avoid_none_and_empty_are_byte_identical():
    scenario = _scenario()
    seed = _seed(scenario)
    legacy = seed.mutate(rng=random.Random(5))
    none_kw = seed.mutate(rng=random.Random(5), avoid_loci=None)
    empty_kw = seed.mutate(rng=random.Random(5), avoid_loci=set())
    assert legacy.to_code() == none_kw.to_code() == empty_kw.to_code()
    assert legacy.fingerprint() == none_kw.fingerprint() == empty_kw.fingerprint()


def test_mutate_avoid_bounded_when_every_door_is_dead():
    scenario = _scenario()
    seed = _seed(scenario)
    catalog = catalog_sources(scenario.sources)
    avoid = {_key(e) for e in catalog}
    g = seed.mutate(rng=random.Random(3), avoid_loci=avoid)
    # bounded: after _AVOID_MAX_REDRAWS vetoes the last draw is accepted —
    # the genome still carries exactly one edit, initialization never deadlocks
    assert len(g.edits) == 1


class _StubSmap:
    def __init__(self, lines):
        self._lines = lines

    def get_top_nodes(self, top_k=8, min_score=0.0):
        import types

        return [types.SimpleNamespace(line_no=ln) for ln in self._lines]


def test_mutate_avoid_respects_sbfl_narrowing():
    scenario = _scenario()
    seed = _seed(scenario)
    catalog = catalog_sources(scenario.sources)
    hot = {e.lineno for e in catalog}
    hot_line = sorted(hot)[0]
    smap = _StubSmap([hot_line])
    preferred = [e for e in catalog if e.lineno == hot_line]
    assert len(preferred) < len(catalog)  # narrowing is real for this scenario
    avoid = {_key(e) for e in preferred}  # forbid every hot-line door
    g = seed.mutate(rng=random.Random(4), suspicion_map=smap, avoid_loci=avoid)
    # the veto re-draws from the SAME narrowed pool; after bounded vetoes the
    # accepted edit must still sit on the SBFL-hot line — memory never
    # overrides site narrowing (same rule as the priors)
    assert g.edits[0].lineno == hot_line


def test_mutate_avoid_applies_after_kind_prior():
    scenario = _scenario()
    seed = _seed(scenario)
    catalog = catalog_sources(scenario.sources)
    target_kind = catalog[0].kind
    assert any(e.kind != target_kind for e in catalog)
    avoid = {_key(e) for e in catalog if e.kind == target_kind}

    class _AllOnKind:
        def kind_weights(self, kinds, prefix=None):
            return {k: (100.0 if k == target_kind else 0.001) for k in kinds}

    steered = seed.mutate(rng=random.Random(11), edit_prior=_AllOnKind())
    assert steered.edits[0].kind == target_kind  # the prior alone steers there
    vetoed = seed.mutate(
        rng=random.Random(11), edit_prior=_AllOnKind(), avoid_loci=avoid
    )
    # the veto gates the FINAL choice: the prior's favorite dead kind is
    # rejected even though the prior demanded it
    assert _key(vetoed.edits[0]) not in avoid


# ---------- 3. the genetic injection point ----------

def test_make_code_population_legacy_contract_unchanged():
    scenario = _scenario()
    legacy = make_code_population(scenario, 12, random.Random(7))
    explicit = make_code_population(
        scenario, 12, random.Random(7), avoid_loci=None, redraws=0
    )
    assert [i.genome.to_code() for i in legacy] == [
        i.genome.to_code() for i in explicit
    ]
    # first individual is the clean seed genome
    assert len(legacy[0].genome.edits) == 0
    assert len(legacy) == 12


def test_make_code_population_redraws_perturbs_init():
    scenario = _scenario()
    base = make_code_population(scenario, 12, random.Random(7))
    redrawn = make_code_population(scenario, 12, random.Random(7), redraws=1)
    assert len(redrawn) == 12
    assert [i.genome.fingerprint() for i in base] != [
        i.genome.fingerprint() for i in redrawn
    ]


def test_make_code_population_avoid_all_doors_still_terminates():
    scenario = _scenario()
    catalog = catalog_sources(scenario.sources)
    avoid = {_key(e) for e in catalog}
    pop = make_code_population(scenario, 12, random.Random(7), avoid_loci=avoid)
    assert len(pop) == 12
    assert all(len(i.genome.edits) == 1 for i in pop[1:])


# ---------- 4. the registered decision rule (harness, importlib) ----------

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "ab_genetic_init_memory.py"
)
_spec = importlib.util.spec_from_file_location("ab_genetic_init_memory", _SCRIPT)
ab8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab8)


def test_harness_fisher_known_tables():
    # a=10, b=0, c=0, d=10 -> perfect separation, tiny p
    assert ab8.fisher_exact_2x2(10, 0, 0, 10) < 0.001
    # identical margins -> p = 1.0
    assert ab8.fisher_exact_2x2(5, 25, 5, 25) == 1.0


def test_harness_r1_classification():
    # promising: p < 0.05 AND surplus in >= 3/4 scenarios
    assert (
        ab8.classify_effect(20, 10, 0.01, [3, 3, 3, -1]) == "promising"
    )
    # harmful: deficit consistent
    assert ab8.classify_effect(10, 20, 0.01, [-3, -3, -3, 1]) == "harmful"
    # significant but sign-inconsistent (2 pos, 1 neg, 1 tie) -> no effect
    assert (
        ab8.classify_effect(20, 10, 0.01, [5, 5, -5, 0])
        == "no_detectable_effect"
    )
    # consistent but not significant -> no detectable effect
    assert (
        ab8.classify_effect(12, 10, 0.5, [1, 1, 0, 0]) == "no_detectable_effect"
    )
