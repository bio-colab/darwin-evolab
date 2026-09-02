"""Tests for trap forgiveness (M5): TTL expiry, re-validation, reset coupling.

Before M5, traps were created and never updated or removed: last_seen_generation
stayed 0 forever, the 0.1x penalty was a life sentence, and the library was
never wired into the engine (scan had no caller; the selector was built
without a trap_library). These tests pin the new contract: a trap is a
hypothesis that must keep being confirmed by current data, and when the
model forgets (inertia breaker reset), the traps are forgiven with it.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab import (
    CausalModel,
    EngineConfig,
    EvolutionEngine,
    StrategicMutationSelector,
    TrapSignatureLibrary,
)

KEY = "light|cplx:complex|fit:low|status:ok"


def feed_failures(model: CausalModel, key: str, n: int, delta: float = -5.0) -> None:
    for _ in range(n):
        model.observe(*key.split("|", 1), delta)


def feed_successes(model: CausalModel, key: str, n: int, delta: float = 3.0) -> None:
    for _ in range(n):
        model.observe(*key.split("|", 1), delta)


# ---------------------------------------------------------------------------
# creation, re-validation, expiry
# ---------------------------------------------------------------------------

def test_trap_created_and_revalidated_with_generation_clock():
    model = CausalModel()
    feed_failures(model, KEY, 6)
    lib = TrapSignatureLibrary()
    created = lib.scan(model, generation=10)
    assert len(created) == 1 and lib.is_known_trap(KEY)
    assert created[0].last_seen_generation == 10
    # re-confirmed later: clock moves, trap stays
    feed_failures(model, KEY, 2)
    lib.scan(model, generation=15)
    assert lib.is_known_trap(KEY)
    assert lib.signatures[KEY].last_seen_generation == 15


def test_unconfirmed_trap_expires_after_ttl():
    model = CausalModel()
    feed_failures(model, KEY, 6)
    lib = TrapSignatureLibrary(ttl_generations=20)
    lib.scan(model, generation=10)
    assert lib.is_known_trap(KEY)
    # context recovers: successes arrive, fail rate drops below threshold
    feed_successes(model, KEY, 10)
    assert not lib._confirmed(KEY, model._stats[KEY])
    # within TTL the trap lingers (no immediate revocation)
    lib.scan(model, generation=25)
    assert lib.is_known_trap(KEY)
    # beyond TTL without re-confirmation: forgiven
    lib.scan(model, generation=31)   # 31 - 10 = 21 > 20
    assert not lib.is_known_trap(KEY)
    assert lib.expired == 1


def test_reconfirmed_trap_never_expires():
    model = CausalModel()
    feed_failures(model, KEY, 6)
    lib = TrapSignatureLibrary(ttl_generations=5)
    for gen in range(0, 100, 3):
        feed_failures(model, KEY, 1)
        lib.scan(model, generation=gen)
    assert lib.is_known_trap(KEY)


def test_neutral_observations_do_not_dilute_confirmation():
    """Documented semantics: confirmation uses the signed total (s+f) — a
    context with 6 failures and 0 successes is still a 100%-failure context
    no matter how many neutral events arrive. Neutral noise neither keeps a
    diluted trap alive nor dilutes a real one; silence (no re-confirmation)
    and recovered fail rates are what trigger forgiveness."""
    model = CausalModel()
    feed_failures(model, KEY, 6)
    lib = TrapSignatureLibrary(ttl_generations=20)
    lib.scan(model, generation=10)
    for _ in range(40):
        kind, bin_ = KEY.split("|", 1)
        model.observe(kind, bin_, 0.0)
    assert lib._confirmed(KEY, model._stats[KEY])   # 6/6 failures = 100%
    lib.scan(model, generation=40)                   # re-confirmed again
    assert lib.is_known_trap(KEY)
    assert lib.signatures[KEY].last_seen_generation == 40


def test_scan_without_generation_still_backward_compatible():
    model = CausalModel()
    feed_failures(model, KEY, 6)
    lib = TrapSignatureLibrary()
    lib.scan(model)   # positional, no generation — pre-M5 call shape
    assert lib.is_known_trap(KEY)
    lib.scan(model, generation=0)  # 0 - 0 <= ttl → survives
    assert lib.is_known_trap(KEY)


# ---------------------------------------------------------------------------
# reset coupling: the model forgets, the traps are forgiven
# ---------------------------------------------------------------------------

def test_inertia_breaker_resets_traps_with_the_model():
    model = CausalModel()
    feed_failures(model, KEY, 6)
    lib = TrapSignatureLibrary()
    lib.scan(model, generation=3)
    selector = StrategicMutationSelector(model, trap_library=lib)
    assert lib.is_known_trap(KEY)
    selector.reset_ucb_weights()
    assert not model._stats                       # model forgets
    assert not lib.is_known_trap(KEY)             # traps forgiven with it


def test_selector_without_library_still_resets_model():
    model = CausalModel()
    feed_failures(model, KEY, 6)
    selector = StrategicMutationSelector(model)   # no trap_library (pre-M5 shape)
    selector.reset_ucb_weights()
    assert not model._stats


# ---------------------------------------------------------------------------
# engine wiring (the library existed but was never wired before M5)
# ---------------------------------------------------------------------------


def test_engine_wires_trap_library_and_scans_under_causal_flag():
    engine = EvolutionEngine(
        config=EngineConfig(generations=5, population_size=8, genome_size=4, seed=11),
        causal_layer_enabled=True,
    )
    engine.run(generations=5)
    # _begin_run (inside run) wires the library into the selector, and the
    # run loop scanned it every generation without breaking the run
    assert engine._trap_library is engine._mutation_selector.trap_library
    assert isinstance(engine._trap_library.signatures, dict)


def test_default_path_never_touches_traps():
    engine = EvolutionEngine(
        config=EngineConfig(generations=4, population_size=8, genome_size=4, seed=11),
    )
    engine.run(generations=4)
    assert engine._mutation_selector.trap_library is not None   # wired
    assert engine._trap_library.signatures == {}                 # scan gated off


def test_live_trap_actually_changes_selector_decision():
    """With the causal layer on, a live trap must flip the selector's choice.

    Constructed so the UNTRAPPED selector picks 'light' (its neutral-inflated
    success rate plus UCB bonus beats semantic's), while the trapped selector
    picks 'semantic' (the 0.1x penalty crushes light's score). This proves
    the newly wired trap_library is consulted, not decorative.
    """
    ctx = KEY.split("|", 1)[1]

    def build():
        model = CausalModel()
        # light: 5 failures (fail_rate 1.0 -> trappable) + 40 neutrals ->
        # success_rate = 0.5*40/45 = 0.444 (neutrals weigh 0.5)
        feed_failures(model, KEY, 5)
        for _ in range(40):
            kind, bin_ = KEY.split("|", 1)
            model.observe(kind, bin_, 0.0)
        # semantic: 20 successes / 30 failures -> rate 0.4
        healthy = "semantic|" + ctx
        feed_successes(model, healthy, 20)
        feed_failures(model, healthy, 30)
        return model

    rng = None  # scored path is deterministic; epsilon=0.0 excludes exploration
    trapped_selector = StrategicMutationSelector(build(), epsilon=0.0, trap_library=TrapSignatureLibrary())
    trapped_selector.trap_library.scan(trapped_selector.model, generation=1)
    free_selector = StrategicMutationSelector(build(), epsilon=0.0)

    trapped_picks = {trapped_selector.select(ctx, random.Random(s)) for s in range(30)}
    free_picks = {free_selector.select(ctx, random.Random(s)) for s in range(30)}
    assert trapped_picks == {"semantic"}
    assert free_picks == {"light"}
