"""Zero-signal suppression (memory-hygiene M6): the prior's self-gate.

Contracts under test:
- A weight vector within the registered zero-signal envelope (expressed
  bias <= ZERO_SIGNAL_MAX_FRACTION of the maximum bias the blend could
  express at its configured strength) collapses to ``None`` — the caller
  keeps its exact existing behavior, byte-for-byte RNG stream.
- The gate is strength-INVARIANT: the same zero-signal data gates at
  strength 0.9 exactly as at 0.5, because the ratio is normalized by the
  strength-dependent maximum (``1 / (1 - strength)`` — the blend's own
  docstring bound).
- Real signal is never gated: differentially successful kinds keep their
  exact pre-M6 weight values.
- Fail-open: a prior actively FORBIDDING a kind (weight <= 0, reachable
  with alpha=0 + strength=1) is a strong opinion and is never suppressed.
- End-to-end: prior-on with a zero-signal store is byte-identical to
  prior-off, at mutation level and across a full engine run.

Threshold provenance (registered, not a tuning knob): 0.10 corresponds at
the default strength 0.5 to weight ratio <= 1.10:1 — exactly the envelope
v2 measured on the repository's real stores (near-zero bias, zero search
value, full RNG-stream divergence; reports/ab_memory_value_v2.json).
"""
from __future__ import annotations

import random

import pytest

from evolab import EngineConfig, EvolutionEngine
from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population
from evolab.experience import (
    ExperienceMutationPrior,
    ExperienceRecorderProxy,
    ExperienceSequencePrior,
    ExperienceStore,
    attach_experience_recorder,
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


def _store_with(fp: str, kind_counts: dict[str, tuple[int, int]]) -> ExperienceStore:
    """Store preloaded with per-kind (successes, failures) for one fingerprint."""
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


# ---------- the detector, directly ----------

def test_gate_boundary_brackets_the_registered_envelope():
    # At strength 0.5 the expressive range is ratio 2.0; the registered
    # envelope (fraction <= 0.10) therefore maps to ratio <= 1.10. Brackets
    # the boundary from both sides (the exact 1.10 float is fragile).
    prior = ExperienceMutationPrior(object(), FP, strength=0.5)
    inside = prior._near_uniform({"a": 0.824, "b": 0.75})   # ratio ~1.0987
    outside = prior._near_uniform({"a": 0.826, "b": 0.75})  # ratio ~1.1013
    assert inside is True
    assert outside is False


def test_gate_is_strength_invariant_on_real_data():
    # Same zero-signal store (rates 6/11 vs 5/11): gated at BOTH strengths.
    store = _store_with(FP, {"bool_flip": (5, 4), "string_sep": (4, 5)})
    kinds = ["bool_flip", "string_sep"]
    for strength in (0.5, 0.9):
        prior = ExperienceMutationPrior(store, FP, strength=strength)
        assert prior.kind_weights(kinds) is None, f"strength={strength}"
    # Same differential store (rates 9/10 vs 1/10): silent at BOTH.
    sig = _store_with(FP, {"bool_flip": (8, 0), "string_sep": (0, 8)})
    for strength in (0.5, 0.9):
        prior = ExperienceMutationPrior(sig, FP, strength=strength)
        w = prior.kind_weights(kinds)
        assert w is not None, f"strength={strength}"
        assert w["bool_flip"] > w["string_sep"]
        if strength == 0.5:
            assert w == {"bool_flip": 0.95, "string_sep": 0.55}
    store.close()
    sig.close()


def test_gate_silent_data_returns_exact_pre_m6_values():
    # The gate only ever REMOVES interventions; when it stays silent the
    # emitted weights are bit-exact the pre-M6 blended values.
    store = _store_with(FP, {"bool_flip": (6, 0), "string_sep": (0, 6)})
    prior = ExperienceMutationPrior(store, FP, strength=0.5)
    assert prior.kind_weights(["bool_flip", "string_sep"]) == {
        "bool_flip": 0.5 + 0.5 * 7 / 8,
        "string_sep": 0.5 + 0.5 * 1 / 8,
    }
    store.close()


def test_zero_strength_prior_is_null_by_definition():
    # strength=0 blends every rate onto the uniform baseline — the prior
    # defines itself as zero-intervention, gate or no gate.
    store = _store_with(FP, {"bool_flip": (6, 0), "string_sep": (0, 6)})
    prior = ExperienceMutationPrior(store, FP, strength=0.0)
    assert prior.kind_weights(["bool_flip", "string_sep"]) is None
    store.close()


def test_failopen_when_prior_tries_to_forbid():
    # alpha=0 + strength=1 makes a zero-weight possible: the prior is
    # actively forbidding "string_sep" — a strong opinion. The gate must
    # NOT suppress it (fail-open), even though the vector is far from
    # uniform anyway.
    store = _store_with(FP, {"bool_flip": (3, 0), "string_sep": (0, 3)})
    prior = ExperienceMutationPrior(store, FP, strength=1.0, alpha=0.0)
    w = prior.kind_weights(["bool_flip", "string_sep"])
    assert w is not None
    assert w["bool_flip"] == 1.0
    assert w["string_sep"] == 0.0
    store.close()


def test_isolation_knob_reproduces_pre_m6_behavior():
    # zero_signal_gate=False is the A/B instrumentation arm: the exact
    # pre-M6 behavior (near-uniform weights emitted, not suppressed).
    store = _store_with(FP, {"bool_flip": (30, 30), "string_sep": (30, 30)})
    gated = ExperienceMutationPrior(store, FP, strength=0.5)
    ungated = ExperienceMutationPrior(store, FP, strength=0.5, zero_signal_gate=False)
    kinds = ["bool_flip", "string_sep"]
    assert gated.kind_weights(kinds) is None
    w = ungated.kind_weights(kinds)
    assert w is not None
    assert w == {"bool_flip": 0.75, "string_sep": 0.75}
    assert gated.summarize()["zero_signal_gate"] is True
    assert ungated.summarize()["zero_signal_gate"] is False
    store.close()


def test_summarize_reports_gate_constant():
    store = _store_with(FP, {})
    prior = ExperienceMutationPrior(store, FP)
    info = prior.summarize()
    assert info["zero_signal_max_fraction"] == 0.10
    assert ExperienceSequencePrior.ZERO_SIGNAL_MAX_FRACTION == 0.10
    store.close()


# ---------- sequence prior inherits the gate ----------

def test_sequence_gate_collapses_zero_signal_transitions():
    # At prefix ["a"], transitions a>b and a>c have identical rates —
    # conditioning found nothing that separates the candidates -> None.
    rows = [(["a", "b"], 1)] * 5 + [(["a", "b"], 0)] * 5
    rows += [(["a", "c"], 1)] * 5 + [(["a", "c"], 0)] * 5
    store = _store_with(FP, {})
    # reuse the store builder only for a temp path; sequence rows recorded
    # below carry multi-edit kind lists so sequence_stats can decompose them
    from evolab.experience import ExperienceStore  # noqa: F401  (already imported)
    i = 0
    for kinds, passed in rows:
        i += 1
        store.record({
            "run_id": "t", "eval_index": i, "problem_fingerprint": FP,
            "edit_kinds": list(kinds), "score": 90.0 if passed else 20.0,
            "fitness_delta": 1.0 if passed else 0.0,
            "passed_holdout": passed,
            "outcome": "success" if passed else "neutral",
        })
    seq = ExperienceSequencePrior(store, FP, min_support=3, strength=0.5)
    assert seq.kind_weights(["b", "c"], prefix=["a"]) is None
    store.close()


# ---------- end-to-end byte identity ----------

def test_mutate_with_zero_signal_store_is_byte_identical_to_no_prior():
    # Data present for EVERY catalog kind, but identical rates (30/30
    # each) — the store is informationally empty. The gated prior must
    # reproduce the exact no-prior choice for every seed.
    kinds = []
    for e in catalog_sources(SOURCES):
        if e.kind not in kinds:
            kinds.append(e.kind)
    store = _store_with(FP, {k: (30, 30) for k in kinds})
    prior = ExperienceMutationPrior(store, FP, strength=0.5)
    g0 = RepairGenome(sources=dict(SOURCES), target_file="app.py", edits=[])
    for seed in range(1, 21):
        a = g0.mutate(rng=random.Random(seed))
        b = g0.mutate(rng=random.Random(seed), edit_prior=prior)
        assert a.fingerprint() == b.fingerprint(), f"seed={seed}"
        assert len(b.edits) == len(a.edits) == 1
    store.close()


def test_engine_run_with_zero_signal_store_matches_control_history():
    # Full-run identity: same seed, same scenario; the prior arm reads a
    # heavily-seeded zero-signal store. The M6 gate fires at every
    # mutation, so the two trajectories must be indistinguishable.
    scenario = SCENARIO_REGISTRY["click_cli_parser"]()
    fp = problem_fingerprint(scenario.sources, scenario.target_file, scenario.func_name)
    kinds = sorted({e.kind for e in catalog_sources(scenario.sources)})

    def build(prior_enabled: bool, db_path):
        ev = scenario.create_evaluator()
        return attach_experience_recorder(
            ev, scenario.sources, scenario.target_file, scenario.func_name,
            db_path=db_path, run_id=f"m6_{prior_enabled}",
            prior_enabled=prior_enabled,
            prior_kwargs={"cache_ttl": 1e9} if prior_enabled else None,
        )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="m6_gate_") as td:
        tdp = Path(td)
        # Seed BOTH arms' stores identically (recording happens regardless);
        # the prior arm's kind_weights sees (30, 30) per kind -> uniform -> gate.
        for name in ("off", "on"):
            s = ExperienceStore(tdp / f"{name}.db")
            i = 0
            for k in kinds:
                for _ in range(30):
                    i += 1
                    s.record({
                        "run_id": f"seed_{name}", "eval_index": i,
                        "problem_fingerprint": fp,
                        "edit_kinds": [k], "score": 90.0, "fitness_delta": 1.0,
                        "passed_holdout": 1, "outcome": "success",
                    })
                    i += 1
                    s.record({
                        "run_id": f"seed_{name}", "eval_index": i,
                        "problem_fingerprint": fp,
                        "edit_kinds": [k], "score": 20.0, "fitness_delta": 0.0,
                        "passed_holdout": 0, "outcome": "neutral",
                    })
            s.close()

        histories = {}
        for name, enabled in (("off", False), ("on", True)):
            wired = build(enabled, tdp / f"{name}.db")
            engine = EvolutionEngine(
                fitness_fn=wired,
                config=EngineConfig(generations=3, population_size=6,
                                    elite_count=1, genome_size=2, seed=7),
            )
            initial = make_code_population(scenario, 6, random.Random(7))
            report = engine.run(generations=3, initial_population=initial)
            histories[name] = report["history"]

    assert histories["off"] == histories["on"]


def test_gate_never_blocks_real_signal_in_engine_run():
    # Counterweight: a DIFFERENTIAL store must still reach the mutation
    # path — the favored kind wins more draws than it would unlabeled.
    # Guards against the gate overfitting into a silent total no-op.
    scenario = SCENARIO_REGISTRY["click_cli_parser"]()
    fp = problem_fingerprint(scenario.sources, scenario.target_file, scenario.func_name)
    kinds = sorted({e.kind for e in catalog_sources(scenario.sources)})
    favored, other = kinds[0], kinds[-1]
    assert favored != other

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="m6_signal_") as td:
        db = Path(td) / "sig.db"
        store = ExperienceStore(db)
        i = 0
        for k in kinds:
            ok, fail = (40, 0) if k == favored else (0, 40)
            for _ in range(ok):
                i += 1
                store.record({
                    "run_id": "sig", "eval_index": i,
                    "problem_fingerprint": fp,
                    "edit_kinds": [k], "score": 90.0, "fitness_delta": 1.0,
                    "passed_holdout": 1, "outcome": "success",
                })
            for _ in range(fail):
                i += 1
                store.record({
                    "run_id": "sig", "eval_index": i,
                    "problem_fingerprint": fp,
                    "edit_kinds": [k], "score": 20.0, "fitness_delta": 0.0,
                    "passed_holdout": 0, "outcome": "neutral",
                })
        store.close()

        wired = attach_experience_recorder(
            scenario.create_evaluator(), scenario.sources, scenario.target_file,
            scenario.func_name, db_path=db, run_id="m6_signal",
            prior_enabled=True, prior_kwargs={"cache_ttl": 1e9},
        )
        proxy_prior = wired.mutation_prior
        w = proxy_prior.kind_weights([favored, other])
        assert w is not None          # real signal survives the gate
        assert w[favored] > w[other]  # and still points the right way
