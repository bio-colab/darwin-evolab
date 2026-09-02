"""Sequence memory (M7): prefix-conditioned mutation prior.

Contracts under test:
- ``ExperienceStore.sequence_stats`` decomposes stored ordered ``edit_kinds``
  into prefix→next-kind transitions with the row's outcome credited to every
  transition on the path (∅→k1, k1→k2, ...).
- ``ExperienceSequencePrior`` conditions on the parent's edit-kind prefix;
  falls back to the per-kind marginals on UNSEEN prefixes (never more
  ignorant than the base prior); returns None on empty/broken stores.
- The base ``ExperienceMutationPrior`` accepts the ``prefix`` kwarg and
  ignores it (v2 harness / legacy callers keep working).
- Wiring: the recorder proxy selects the prior class via ``mode``
  (default "kind", unchanged); ``mutate`` passes the prefix; a legacy prior
  without a ``prefix`` parameter degrades safely to the null behavior.
- Bias stays bounded and never forbids a kind (priors guide, never force).
"""
from __future__ import annotations

import random

import pytest

from evolab.experience import (
    ExperienceMutationPrior,
    ExperienceRecorderProxy,
    ExperienceSequencePrior,
    ExperienceStore,
    problem_fingerprint,
)
from evolab.repair import RepairGenome, catalog_sources

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


def _seq_store(rows: list[tuple[list[str], int]]) -> ExperienceStore:
    """Store preloaded with (edit_kinds sequence, passed_holdout) rows."""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "exp.db"
    store = ExperienceStore(tmp)
    for i, (kinds, passed) in enumerate(rows, start=1):
        store.record({
            "run_id": "t", "eval_index": i, "problem_fingerprint": FP,
            "edit_kinds": list(kinds), "score": 90.0 if passed else 20.0,
            "fitness_delta": 1.0 if passed else 0.0,
            "passed_holdout": passed,
            "outcome": "success" if passed else "neutral",
        })
    return store


def _catalog_kinds() -> list[str]:
    kinds: list[str] = []
    for e in catalog_sources(SOURCES):
        if e.kind not in kinds:
            kinds.append(e.kind)
    return kinds


# ---------- store-level decomposition ----------

def test_transition_decomposition_exact():
    store = _seq_store([
        (["a", "b", "c"], 1),
        (["a", "b", "d"], 0),
        (["a"], 1),
    ])
    seq = store.sequence_stats(FP)
    assert seq["total_experiences"] == 3
    t = seq["transitions"]
    # every row walks ∅→a (2 of 3 passed)
    assert t[">a"] == {"n": 3, "holdout_success": 2}
    # both [a,b,*] rows walked a→b (one passed, one failed)
    assert t["a>b"] == {"n": 2, "holdout_success": 1}
    # deep prefixes credited from their own row only
    assert t["a,b>c"] == {"n": 1, "holdout_success": 1}
    assert t["a,b>d"] == {"n": 1, "holdout_success": 0}
    store.close()


def test_transition_key_encoding_is_collision_free():
    # guard for the ",".join(prefix) + ">" + kind encoding: catalog kinds are
    # identifiers and must never contain the delimiters
    for kind in _catalog_kinds():
        assert "," not in kind and ">" not in kind


def test_empty_rows_have_no_transitions():
    store = _seq_store([([], 1), ([], 0)])
    seq = store.sequence_stats(FP)
    assert seq["total_experiences"] == 2
    assert seq["transitions"] == {}
    store.close()


# ---------- prior math ----------

def test_prefix_conditioning_changes_weights():
    # kind "b" succeeds after prefix ["a"] but fails after prefix ["c"];
    # its MARGINAL rate is a useless 0.5 — the whole point of M7.
    # M6 note: queries carry a neutral filler kind ("zzz", never observed)
    # so the emitted weight vectors are non-uniform and survive the
    # zero-signal gate; single-candidate weight vectors are pure
    # perturbation and correctly collapse to None.
    rows = [(["a", "b"], 1)] * 6 + [(["c", "b"], 0)] * 6
    store = _seq_store(rows)
    seq = ExperienceSequencePrior(store, FP, min_support=3, strength=0.5)
    base = ExperienceMutationPrior(store, FP, min_support=3, strength=0.5)

    w_after_a = seq.kind_weights(["b", "zzz"], prefix=["a"])["b"]
    w_after_c = seq.kind_weights(["b", "zzz"], prefix=["c"])["b"]

    # conditioning separates what the marginal cannot
    assert w_after_a == 0.5 + 0.5 * (6 + 1) / (6 + 2)   # 0.9375
    assert w_after_c == 0.5 + 0.5 * (0 + 1) / (6 + 2)   # 0.5625
    # M6: the per-kind marginal here is 0.5 for both candidates — the
    # base prior's output would be uniform, so the gate collapses it to
    # None (null-intervention). That is a STRONGER statement of the very
    # point this test has always made: the marginal is blind here.
    assert base.kind_weights(["b", "zzz"]) is None
    assert w_after_a > 0.75 > w_after_c
    store.close()


def test_first_edit_uses_empty_prefix_transitions():
    rows = [(["a", "b"], 1)] * 6 + [(["c", "b"], 0)] * 6
    store = _seq_store(rows)
    seq = ExperienceSequencePrior(store, FP, min_support=3, strength=0.5)
    # no parent edits yet -> ∅ transitions: a first 6/6, c first 0/6
    w = seq.kind_weights(["a", "c"], prefix=[])
    assert w["a"] == 0.5 + 0.5 * (6 + 1) / (6 + 2)
    assert w["c"] == 0.5 + 0.5 * (0 + 1) / (6 + 2)
    store.close()


def test_unseen_prefix_falls_back_to_marginals():
    rows = [(["a", "b"], 1)] * 6 + [(["c", "b"], 0)] * 6
    store = _seq_store(rows)
    seq = ExperienceSequencePrior(store, FP, min_support=3, strength=0.5)
    base = ExperienceMutationPrior(store, FP, min_support=3, strength=0.5)
    kinds = ["b", "c"]
    assert seq.kind_weights(kinds, prefix=["never", "seen"]) == base.kind_weights(kinds)
    store.close()


def test_empty_store_returns_none():
    store = _seq_store([])
    seq = ExperienceSequencePrior(store, FP)
    assert seq.kind_weights(["a"]) is None
    assert seq.kind_weights(["a"], prefix=["a"]) is None
    store.close()


def test_rows_without_edits_return_none():
    store = _seq_store([([], 1), ([], 0)])
    seq = ExperienceSequencePrior(store, FP)
    assert seq.kind_weights(["a"], prefix=[]) is None
    store.close()


def test_min_support_keeps_weak_transitions_neutral():
    # M6 note: restructured to a strong+weak pair so the weight vector is
    # non-uniform and survives the zero-signal gate; the all-weak case is
    # asserted to collapse to None right after (that is the M6 contract).
    store = _seq_store([(["a", "b"], 1)] * 6 + [(["a", "c"], 1)] * 2)
    seq = ExperienceSequencePrior(store, FP, min_support=3, strength=0.5)
    w = seq.kind_weights(["b", "c"], prefix=["a"])
    assert w is not None
    assert w["b"] == 0.5 + 0.5 * (6 + 1) / (6 + 2)  # supported transition
    assert w["c"] == 0.5 + 0.5 * 0.5  # n=2 < min_support -> forced neutral 0.75
    # all-weak counterpart: every candidate neutral -> uniform -> None (M6)
    weak_store = _seq_store([(["a", "b"], 1)] * 2)
    weak_seq = ExperienceSequencePrior(weak_store, FP, min_support=3, strength=0.5)
    assert weak_seq.kind_weights(["b", "zzz"], prefix=["a"]) is None
    weak_store.close()
    store.close()


def test_bias_bounded_and_never_zero():
    store = _seq_store([(["a", "b"], 1)] * 50 + [(["a", "c"], 0)] * 50)
    seq = ExperienceSequencePrior(store, FP, min_support=3, strength=0.9)
    w = seq.kind_weights(["b", "c"], prefix=["a"])
    # Laplace smoothing keeps even the best transition below 1.0
    assert w["b"] == pytest.approx(0.1 + 0.9 * (50 + 1) / (50 + 2))
    assert w["c"] > 0.0   # a losing transition is down-weighted, never forbidden
    assert w["b"] / w["c"] <= 1.0 / (1.0 - 0.9) + 1e-9
    store.close()


def test_broken_store_fails_safe():
    store = _seq_store([(["a", "b"], 1)])
    seq = ExperienceSequencePrior(store, FP)
    store._conn.close()  # break the store
    assert seq.kind_weights(["b"], prefix=["a"]) is None


def test_summarize_reports_mode_and_transitions():
    store = _seq_store([(["a", "b"], 1)] * 4)
    seq = ExperienceSequencePrior(store, FP)
    info = seq.summarize()
    assert info["mode"] == "sequence"
    assert info["transitions"] == 2  # >a and a>b
    assert info["total_experiences"] == 4
    store.close()


# ---------- wiring ----------

def test_proxy_mode_dispatch():
    store = _seq_store([(["a", "b"], 1)])
    raw = object()
    p_default = ExperienceRecorderProxy(raw, store, FP, prior_enabled=True)
    assert type(p_default.mutation_prior) is ExperienceMutationPrior
    p_kind = ExperienceRecorderProxy(
        raw, store, FP, prior_enabled=True, prior_kwargs={"mode": "kind"}
    )
    assert type(p_kind.mutation_prior) is ExperienceMutationPrior
    p_seq = ExperienceRecorderProxy(
        raw, store, FP, prior_enabled=True,
        prior_kwargs={"mode": "sequence", "strength": 0.9},
    )
    prior = p_seq.mutation_prior
    assert isinstance(prior, ExperienceSequencePrior)
    assert prior.strength == 0.9  # "mode" is consumed, never passed down
    p_bad = ExperienceRecorderProxy(
        raw, store, FP, prior_enabled=True, prior_kwargs={"mode": "typo"}
    )
    with pytest.raises(ValueError):
        p_bad.mutation_prior
    store.close()


class LegacyPrior:
    """A pre-M7 prior object: kind_weights without the prefix parameter."""

    def __init__(self):
        self.calls = 0

    def kind_weights(self, kinds):
        self.calls += 1
        return None


class SpyPrior:
    """Records the prefix it receives; returns None (null behavior)."""

    def __init__(self):
        self.prefixes: list[list[str] | None] = []

    def kind_weights(self, kinds, prefix=None):
        self.prefixes.append(list(prefix) if prefix is not None else None)
        return None


def test_mutate_passes_parent_prefix_to_prior():
    g0 = RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[])
    first = g0.mutate(rng=random.Random(7))          # parent after one edit
    spy = SpyPrior()
    first.mutate(rng=random.Random(7), edit_prior=spy)
    assert spy.prefixes == [[first.edits[0].kind]]   # the parent's recipe


def test_legacy_prior_without_prefix_param_degrades_safely():
    g0 = RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[])
    legacy = LegacyPrior()
    rng = random.Random(123)
    a = g0.mutate(rng=random.Random(123))
    b = g0.mutate(rng=rng, edit_prior=legacy)
    # the prefix kwarg raises TypeError at CALL time (before the legacy body
    # runs) -> caught by mutate's guard -> null behavior, no crash
    assert legacy.calls == 0
    assert b.fingerprint() == a.fingerprint()  # same choice as no prior


def test_null_prior_preserves_original_choice():
    # empty store -> weights None -> the exact pre-memory draw (same seed,
    # same chosen edit) with or without a prior object around
    store = _seq_store([])
    g0 = RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[])
    prior = ExperienceSequencePrior(store, FP)
    a = g0.mutate(rng=random.Random(2024))
    b = g0.mutate(rng=random.Random(2024), edit_prior=prior)
    assert a.fingerprint() == b.fingerprint()
    store.close()


def test_sequence_prior_is_deterministic_given_seed():
    rows = [(["a", "b"], 1)] * 6 + [(["c", "b"], 0)] * 6
    store = _seq_store(rows)
    prior = ExperienceSequencePrior(store, FP, strength=0.9)

    def recipe(seed: int) -> list[str]:
        g0 = RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[])
        rng = random.Random(seed)
        kinds = []
        for _ in range(3):
            g0 = g0.mutate(rng=rng, edit_prior=prior)
            kinds.append(g0.edits[-1].kind)
        return kinds

    assert recipe(99) == recipe(99)
    store.close()
