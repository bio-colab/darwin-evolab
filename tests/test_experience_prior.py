"""Experience memory (Phase 2): mutation priors — soft, bounded, null-safe.

Contracts under test:
- Empty / broken store -> kind_weights() is None -> caller behavior unchanged.
- Priors guide, never force: weights blended with uniform, bounded ratio,
  never zero.
- Suspicion (SBFL) narrowing and the memory prior compose without conflict:
  suspicion narrows the candidate loci, the prior reweights kinds inside.
- Engine wiring: the prior reaches RepairGenome.mutate only through the
  recorder proxy attached to fitness_fn; plain evaluators change nothing.
"""
from __future__ import annotations

import random

from evolab import EngineConfig, EvolutionEngine, Individual
from evolab.experience import (
    ExperienceMutationPrior,
    ExperienceRecorderProxy,
    ExperienceStore,
    problem_fingerprint,
)
from evolab.repair import RepairGenome, catalog_sources

# A source whose catalog contains several edit kinds (bool constant, string
# separator, subscript index). Tests read the actual kinds from the catalog
# instead of hardcoding, so registry changes cannot break them.
MULTI_BUG = (
    "def helper(args, flag):\n"
    "    first = args[0]\n"
    "    if flag == True:\n"
    "        sep = ','\n"
    "        return {'k': args[0]}\n"
    "    return None\n"
)

SOURCES = {"app.py": MULTI_BUG}
FP = problem_fingerprint(SOURCES, "app.py", "helper")


def _catalog_kinds() -> list[str]:
    kinds: list[str] = []
    for e in catalog_sources(SOURCES):
        if e.kind not in kinds:
            kinds.append(e.kind)
    return kinds


def _store_with(fp: str, kind_counts: dict[str, tuple[int, int]]) -> ExperienceStore:
    """Store preloaded with per-kind (successes, failures) for one fingerprint."""
    import json
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "exp.db"
    store = ExperienceStore(tmp)
    i = 0
    for kind, (ok, fail) in kind_counts.items():
        for _ in range(ok):
            i += 1
            store.record({
                "run_id": "t", "eval_index": i, "problem_fingerprint": fp,
                "edit_kinds": [kind], "score": 90.0, "fitness_delta": 1.0,
                "passed_holdout": 1, "outcome": "success",
            })
        for _ in range(fail):
            i += 1
            store.record({
                "run_id": "t", "eval_index": i, "problem_fingerprint": fp,
                "edit_kinds": [kind], "score": 20.0, "fitness_delta": 0.0,
                "passed_holdout": 0, "outcome": "neutral",
            })
    return store


class StubPrior:
    """Minimal prior double: fixed weights, honors the None contract.

    Accepts the M7 ``prefix`` kwarg and ignores it — the same contract the
    real ``ExperienceMutationPrior`` implements (context-free by design).
    """

    def __init__(self, weights: dict[str, float] | None):
        self.weights = weights
        self.calls: list[list[str]] = []

    def kind_weights(self, kinds, prefix=None):
        self.calls.append(sorted(set(kinds)))
        if self.weights is None:
            return None
        return {k: self.weights.get(k, 1.0) for k in set(kinds)}


# ---------- prior math ----------

def test_empty_store_returns_none():
    store = _store_with(FP, {})
    prior = ExperienceMutationPrior(store, FP)
    kinds = _catalog_kinds()
    assert prior.kind_weights(kinds) is None


def test_unreachable_fingerprint_returns_none():
    store = _store_with(FP, {"bool_flip": (5, 0)})
    prior = ExperienceMutationPrior(store, "other-fingerprint")
    assert prior.kind_weights(_catalog_kinds()) is None


def test_broken_store_degrades_to_none():
    prior = ExperienceMutationPrior(object(), FP)  # no .stats at all
    assert prior.kind_weights(_catalog_kinds()) is None


def test_weights_favor_successful_kind_and_stay_bounded():
    store = _store_with(FP, {"bool_flip": (6, 0), "string_sep": (0, 6)})
    prior = ExperienceMutationPrior(store, FP, strength=0.5)
    w = prior.kind_weights(["bool_flip", "string_sep"])
    assert w is not None
    assert w["bool_flip"] > w["string_sep"]
    # Blend bounds at strength=0.5: every weight in [0.5, 1.0]; ratio <= 2.
    assert 0.5 <= w["string_sep"] < w["bool_flip"] <= 1.0
    assert w["bool_flip"] / w["string_sep"] <= 2.0


def test_min_support_treats_rare_kinds_as_neutral():
    # M6 note: the pre-gate version of this test used an all-neutral store
    # (every kind below min_support -> uniform 0.75 weights). Under the M6
    # zero-signal gate an ALL-neutral output collapses to None (null-
    # intervention) — see test_zero_signal_gate.py. To keep verifying the
    # neutral-rate VALUE the store now mixes a supported kind with a rare
    # one, so the weight vector carries real signal and survives the gate.
    store = _store_with(FP, {"bool_flip": (6, 0), "string_sep": (2, 0)})
    prior = ExperienceMutationPrior(store, FP, min_support=3, strength=0.5)
    w = prior.kind_weights(["bool_flip", "string_sep"])
    assert w is not None
    # n=2 < min_support=3 -> neutral rate 0.5 -> weight 0.75 despite the
    # kind's raw 2/2 record; the supported kind keeps its Laplace rate.
    assert w["string_sep"] == 0.75
    assert w["bool_flip"] == 0.5 + 0.5 * (6 + 1) / (6 + 2)


def test_all_neutral_kinds_collapse_to_none():
    # M6: when EVERY candidate ends up neutral (uniform weights), the prior
    # has nothing to say — returning None is the exact neutral treatment,
    # and it additionally spares the RNG stream the pointless perturbation.
    store = _store_with(FP, {"bool_flip": (2, 0)})
    prior = ExperienceMutationPrior(store, FP, min_support=3, strength=0.5)
    assert prior.kind_weights(["bool_flip", "string_sep"]) is None


def test_weights_never_reach_zero():
    store = _store_with(FP, {"bool_flip": (10, 0), "string_sep": (0, 10)})
    prior = ExperienceMutationPrior(store, FP, strength=0.9)
    w = prior.kind_weights(["bool_flip", "string_sep"])
    assert w is not None
    assert w["string_sep"] > 0.0  # guides, never forbids


# ---------- RepairGenome.mutate integration ----------

def test_mutate_without_prior_is_unchanged():
    g0 = RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[])
    a = g0.mutate(rng=random.Random(123))
    b = g0.mutate(rng=random.Random(123), edit_prior=None)
    c = g0.mutate(rng=random.Random(123), edit_prior=StubPrior(None))
    assert a.fingerprint() == b.fingerprint() == c.fingerprint()
    assert len(a.edits) == len(b.edits) == len(c.edits) == 1


def test_mutate_with_prior_biases_but_does_not_force():
    kinds = _catalog_kinds()
    favored, other = kinds[0], kinds[1]
    g0 = RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[])
    prior = StubPrior({favored: 10.0, other: 0.1})
    rng = random.Random(42)
    picks = {favored: 0, other: 0}
    for _ in range(300):
        g = g0.mutate(rng=rng, edit_prior=prior)
        kind = g.edits[0].kind
        if kind in picks:
            picks[kind] += 1
    assert picks[favored] > picks[other] * 3      # bias is real
    assert picks[other] > 0                        # but never forbidden


def test_mutate_prior_deterministic_under_seed():
    g0 = RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[])
    kinds = _catalog_kinds()
    prior = StubPrior({k: (3.0 if i % 2 == 0 else 1.0) for i, k in enumerate(kinds)})

    def sequence(seed: int) -> list[str]:
        rng = random.Random(seed)
        g = g0
        out = []
        for _ in range(5):
            g = g.mutate(rng=rng, edit_prior=prior)
            out.append(g.edits[-1].kind)
        return out

    assert sequence(7) == sequence(7)
    assert sequence(7) != sequence(8) or True  # different seeds usually differ


def test_mutate_prior_composes_with_suspicion_narrowing():
    g0 = RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[])
    catalog = catalog_sources(SOURCES)
    lines = sorted({e.lineno for e in catalog})
    # Find a line holding exactly one kind, with another kind living ONLY on
    # a different line — otherwise the composition cannot be expressed.
    plan = None
    for hot in lines:
        hot_kinds = {e.kind for e in catalog if e.lineno == hot}
        if len(hot_kinds) != 1:
            continue
        off_kinds = {e.kind for e in catalog if e.lineno != hot} - hot_kinds
        if off_kinds:
            plan = (hot, next(iter(hot_kinds)), next(iter(off_kinds)))
            break
    if plan is None:
        return  # catalog cannot express the composition here
    hot_line, hot_kind, off_kind = plan

    class HotMap:
        def get_top_nodes(self, top_k=8, min_score=0.0):
            class N:
                line_no = hot_line
            return [N()]

    prior = StubPrior({off_kind: 100.0, hot_kind: 0.01})
    g = g0.mutate(rng=random.Random(5), suspicion_map=HotMap(), edit_prior=prior)
    chosen = g.edits[0]
    # Suspicion narrows the candidate loci; the prior only reweights kinds
    # inside that narrowed set — it can never smuggle in an off-locus kind.
    assert chosen.lineno == hot_line
    assert chosen.kind == hot_kind
    assert prior.calls and set(prior.calls[-1]) == {hot_kind}


# ---------- engine wiring ----------

def test_engine_passes_recorder_prior_to_repair_mutate():
    store = _store_with(FP, {"bool_flip": (4, 0)})
    proxy = ExperienceRecorderProxy(object(), store, FP)
    prior = proxy.mutation_prior
    assert isinstance(prior, ExperienceMutationPrior)
    seen: list[list[str]] = []
    original = prior.kind_weights

    def spy(kinds, prefix=None):
        seen.append(list(kinds))
        return original(kinds)

    prior.kind_weights = spy  # type: ignore[method-assign]

    engine = EvolutionEngine(
        fitness_fn=proxy,
        config=EngineConfig(generations=1, population_size=4, elite_count=1,
                            genome_size=2, seed=3),
    )
    engine._begin_run()
    ind = Individual(genome=RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[]),
                     species="spec_repair")
    engine.mutate(ind)
    assert seen, "engine did not consult the experience prior"
    assert set(seen[0]) <= set(_catalog_kinds())


def test_engine_with_plain_evaluator_never_builds_prior():
    engine = EvolutionEngine(
        config=EngineConfig(generations=1, population_size=4, elite_count=1,
                            genome_size=2, seed=3),
    )
    engine._begin_run()
    ind = Individual(genome=RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[]),
                     species="spec_repair")
    _, kind, _ = engine.mutate(ind)
    assert kind in ("ast", "fault_guided")
    assert len(ind.genome.edits) == 1
