"""M9 — composition-seeded initialization (memory via the genetic channel).

Three contracts, each pinned by tests:
  1. ``ExperienceStore.composition_seeds`` — successful multi-edit
     compositions mined from holdout successes only, deduplicated by their
     (kind, locus) SET, ranked by frequency, ``None`` when there is no
     successful multi-edit row (zero signal) or the store is broken.
  2. ``make_code_population(seed_keys=..., seed_count=...)`` — exactly one
     remembered edit per seeded individual (k=1, replay-proof by
     construction: single-edit genotypes are proven dead doors), built
     deterministically without consuming rng, defensive fallback on
     catalog miss / kind mismatch, byte-identical legacy path when off.
  3. The registered M9 decision rules (harness, importlib) — Fisher exact
     and the R1'''' classification function.
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
from evolab.repair import catalog_sources  # noqa: E402

FP = "m9-unit-fingerprint"


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


def _seed_key(tag, line, col=0):
    return (tag, "t.py", line, col)


# ---------- 1. composition_seeds mining ----------

def test_composition_seeds_mines_multi_edit_successes(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    # a successful 2-edit composition, recorded twice (same winner)
    for i in range(2):
        store.record(
            _row(
                i + 1, ["int_wrap", "index_flip"],
                [["t.py", 7, 12], ["t.py", 9, 29]], 1, "success", score=60.0,
            )
        )
    # a FAILED multi-edit row is never mined (only holdout successes count)
    store.record(
        _row(3, ["bool_flip", "int_wrap"], [["t.py", 2, 0], ["t.py", 3, 0]], 0, "failed")
    )
    # a successful SINGLE-edit row is never mined (n_edits >= 2 contract)
    store.record(_row(4, ["int_wrap"], [["t.py", 5, 0]], 1, "success", score=10.0))
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    winners = store.composition_seeds(FP)
    store.close()
    assert winners is not None and len(winners) == 1
    assert winners[0]["count"] == 2
    assert winners[0]["edits"] == [
        ("int_wrap", "t.py", 7, 12),
        ("index_flip", "t.py", 9, 29),
    ]


def test_composition_seeds_none_when_no_success(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    store.record(_row(1, ["int_wrap"], [["t.py", 7, 12]], 0, "failed"))
    store.record(
        _row(2, ["int_wrap", "index_flip"], [["t.py", 7, 12], ["t.py", 9, 29]], 0, "failed")
    )
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    # rows exist, but nothing succeeded -> zero signal, None (not empty list)
    assert store.composition_seeds(FP) is None
    store.close()


def test_composition_seeds_none_on_empty_store(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    store.record(_row(1, ["int_wrap"], [["t.py", 7, 12]], 0, "failed"))
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    assert store.composition_seeds("different-fingerprint") is None  # other fp
    assert store.composition_seeds(FP) is None  # no rows at all
    store.close()


def test_composition_seeds_broken_store_returns_none(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    store.record(
        _row(1, ["int_wrap", "index_flip"], [["t.py", 1, 0], ["t.py", 2, 0]], 1, "success")
    )
    store.close()
    # closed connection -> sqlite3.ProgrammingError (a sqlite3.Error) -> None
    assert store.composition_seeds(FP) is None


def test_composition_seeds_dedup_and_frequency_ranking(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    # winner A: recorded 3 times (in two different edit ORDERS — same set)
    for i, kinds_loci in enumerate([
        (["a", "b"], [["t.py", 1, 0], ["t.py", 2, 0]]),
        (["a", "b"], [["t.py", 1, 0], ["t.py", 2, 0]]),
        (["b", "a"], [["t.py", 2, 0], ["t.py", 1, 0]]),  # same SET
    ]):
        store.record(_row(i + 1, kinds_loci[0], kinds_loci[1], 1, "success"))
    # winner B: recorded once
    store.record(
        _row(4, ["c", "d"], [["t.py", 3, 0], ["t.py", 4, 0]], 1, "success")
    )
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    winners = store.composition_seeds(FP, max_winners=2)
    store.close()
    assert winners is not None and len(winners) == 2
    assert winners[0]["count"] == 3  # most frequent first
    assert {("a", "t.py", 1, 0), ("b", "t.py", 2, 0)} == set(winners[0]["edits"])
    assert winners[1]["count"] == 1


def test_composition_seeds_max_winners_cap_and_validation(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    store.record(_row(1, ["a", "b"], [["t.py", 1, 0], ["t.py", 2, 0]], 1, "success"))
    store.record(_row(2, ["c", "d"], [["t.py", 3, 0], ["t.py", 4, 0]], 1, "success"))
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    winners = store.composition_seeds(FP, max_winners=1)
    store.close()
    assert winners is not None and len(winners) == 1
    with pytest.raises(ValueError):
        ExperienceStore(tmp_path / "y.db").composition_seeds(FP, max_winners=0)


def test_composition_seeds_skips_malformed_rows(tmp_path):
    store = ExperienceStore(tmp_path / "x.db")
    # kinds/loci length mismatch -> skipped
    store.record(_row(1, ["a", "b"], [["t.py", 1, 0]], 1, "success"))
    # malformed locus -> skipped
    store.record(_row(2, ["a", "b"], [["t.py", 1, 0], "junk"], 1, "success"))
    # well-formed winner survives
    store.record(_row(3, ["a", "b"], [["t.py", 1, 0], ["t.py", 2, 0]], 1, "success"))
    store.close()
    store = ExperienceStore(tmp_path / "x.db")
    winners = store.composition_seeds(FP)
    store.close()
    assert winners is not None and len(winners) == 1
    assert winners[0]["edits"] == [("a", "t.py", 1, 0), ("b", "t.py", 2, 0)]


# ---------- 2. make_code_population seeding ----------

def _scenario():
    return SCENARIO_REGISTRY["requests_http_helper"]()


def _real_winner_keys():
    """A real (kind, locus) pair from this scenario's catalog."""
    cat = sorted(catalog_sources(_scenario().sources), key=lambda e: e.locus())
    return [(e.file, e.lineno, e.col_offset, e.kind) for e in cat[:2]]


def test_seeded_individual_carries_exactly_one_remembered_edit():
    scenario = _scenario()
    keys = _real_winner_keys()
    pop = make_code_population(scenario, 12, random.Random(7), seed_keys=keys, seed_count=6)
    assert len(pop) == 12
    seeded = [ind for ind in pop[1:7]]
    for i, ind in enumerate(seeded):
        edits = ind.genome.edits
        assert len(edits) == 1  # k=1: replay-proof by construction
        assert (edits[0].file, edits[0].lineno, edits[0].col_offset, edits[0].kind) == keys[i % len(keys)]
    # round-robin coverage: both remembered edits were planted
    planted = {(ind.genome.edits[0].kind, ind.genome.edits[0].lineno) for ind in seeded}
    assert planted == {(k[3], k[1]) for k in keys}


def test_seeding_is_deterministic_and_rng_free():
    scenario = _scenario()
    keys = _real_winner_keys()
    p1 = make_code_population(scenario, 12, random.Random(7), seed_keys=keys, seed_count=6)
    p2 = make_code_population(scenario, 12, random.Random(7), seed_keys=keys, seed_count=6)
    assert [i.genome.to_code() for i in p1] == [i.genome.to_code() for i in p2]
    # seeded slots do not depend on the rng at all: a different rng seed
    # changes only the NON-seeded individuals
    p3 = make_code_population(scenario, 12, random.Random(99), seed_keys=keys, seed_count=6)
    assert [i.genome.to_code() for i in p1[1:7]] == [i.genome.to_code() for i in p3[1:7]]


def test_seeding_off_is_byte_identical_legacy():
    scenario = _scenario()
    legacy = make_code_population(scenario, 12, random.Random(7))
    explicit = make_code_population(
        scenario, 12, random.Random(7), seed_keys=None, seed_count=0
    )
    assert [i.genome.to_code() for i in legacy] == [i.genome.to_code() for i in explicit]
    assert len(legacy) == 12


def test_seed_slot_falls_back_on_catalog_miss_or_kind_mismatch():
    scenario = _scenario()
    cat = sorted(catalog_sources(scenario.sources), key=lambda e: e.locus())
    good = (cat[0].file, cat[0].lineno, cat[0].col_offset, cat[0].kind)
    bad_locus = (cat[0].file, 99999, 0, cat[0].kind)          # locus not in catalog
    bad_kind = (cat[1].file, cat[1].lineno, cat[1].col_offset, "no_such_kind")
    pop = make_code_population(
        scenario, 12, random.Random(7),
        seed_keys=[good, bad_locus, bad_kind], seed_count=6,
    )
    # slot 0: planted; slots 1-2: defensive fallback to legacy mutation draws
    seeded0 = pop[1].genome.edits
    assert len(seeded0) == 1
    assert (seeded0[0].file, seeded0[0].lineno, seeded0[0].col_offset, seeded0[0].kind) == good
    for ind in pop[2:4]:
        assert len(ind.genome.edits) == 1  # legacy single-edit mutation


def test_seed_count_zero_with_keys_is_legacy():
    scenario = _scenario()
    keys = _real_winner_keys()
    legacy = make_code_population(scenario, 12, random.Random(7))
    off = make_code_population(scenario, 12, random.Random(7), seed_keys=keys, seed_count=0)
    assert [i.genome.to_code() for i in legacy] == [i.genome.to_code() for i in off]


def test_seed_count_validation():
    scenario = _scenario()
    keys = _real_winner_keys()
    with pytest.raises(ValueError):
        make_code_population(scenario, 12, random.Random(7), seed_keys=None, seed_count=3)
    with pytest.raises(ValueError):
        make_code_population(scenario, 12, random.Random(7), seed_keys=keys, seed_count=-1)
    with pytest.raises(ValueError):
        make_code_population(scenario, 12, random.Random(7), seed_keys=[("t.py", 1)], seed_count=1)


# ---------- 3. the registered decision rules (harness, importlib) ----------

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "ab_composition_seeding.py"
)
_spec = importlib.util.spec_from_file_location("ab_composition_seeding", _SCRIPT)
ab9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab9)


def test_harness_fisher_known_tables():
    # perfect separation -> tiny p; identical margins -> p = 1.0
    assert ab9.fisher_exact_2x2(20, 0, 0, 20) < 0.001
    assert ab9.fisher_exact_2x2(5, 25, 5, 25) == 1.0


def test_harness_r1_classification():
    # promising: p < 0.05 AND surplus consistent in >= 3/4 scenarios
    assert ab9.classify_effect(20, 10, 0.01, [3, 3, 3, -1]) == "promising"
    # harmful: deficit consistent
    assert ab9.classify_effect(10, 20, 0.01, [-3, -3, -3, 1]) == "harmful"
    # significant but sign-inconsistent -> no effect
    assert (
        ab9.classify_effect(20, 10, 0.01, [5, 5, -5, 0]) == "no_detectable_effect"
    )
    # consistent but not significant -> no detectable effect
    assert (
        ab9.classify_effect(12, 10, 0.5, [1, 1, 0, 0]) == "no_detectable_effect"
    )


def test_harness_first_pass_classification():
    winner_loci = {(1, 0), (2, 0)}  # (lineno, col) pairs, as the harness builds them
    full = [("a", 1, 0), ("b", 2, 0)]
    subset = [("a", 1, 0)]
    other = [("c", 3, 0)]
    assert ab9.classify_first_pass(full, winner_loci) == "winner_full"
    assert ab9.classify_first_pass(subset, winner_loci) == "winner_subset"
    assert ab9.classify_first_pass(other, winner_loci) == "other"
    assert ab9.classify_first_pass([], winner_loci) == "other"
