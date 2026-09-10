"""Phase 0 (telemetry truth) + Phase 1 (Self-Model store) contracts.

Preregistered gates (written before measurement):
  P0-C1 diversity is measured, not 0.0, and lies in [0, 1].
  P0-C2 gen_duration_ms is measured (>0 for at least one generation).
  P0-C3 total_candidates_evaluated equals raw engine evals (pop × gens).
  P0-C4 pareto_front is labeled heuristic, never NSGA-II.
  P0-C5 immigrant_fraction=0.0 is exact legacy (no injection); >0 injects
        deterministically without breaking elites or RNG reproducibility.
  P1-C1 self tables exist beside experiences; broken/empty store degrades
        to silent no-op and self_summary() stays read-only (never search).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _engine(**kw):
    import sys
    sys.path.insert(0, "src")
    from evolab import EvolutionEngine
    return EvolutionEngine(population_size=8, seed=11, early_stop_fitness=None, **kw)


def test_p0_telemetry_truth():
    e = _engine()
    r = e.run(3)
    assert r["total_candidates_evaluated"] == 3 * 8 == e._total_evals
    divs = [h.get("diversity") for h in r["history"]]
    durs = [h.get("gen_duration_ms") for h in r["history"]]
    assert len(divs) == 3 and all(isinstance(d, float) and 0.0 <= d <= 1.0 for d in divs)
    assert any(d > 0.0 for d in divs), "diversity must be measured, not hardcoded 0.0"
    assert any(t > 0.0 for t in durs), "duration must be measured"
    assert "immigrants_injected" in r["history"][0]


def test_p0_events_carry_truth():
    import sys
    sys.path.insert(0, "src")
    from evolab import EvolutionEngine
    from evolab.events import GenerationEvaluatedEvent
    seen = []
    e = EvolutionEngine(population_size=6, seed=5, early_stop_fitness=None)
    e.add_event_listener(GenerationEvaluatedEvent, seen.append)
    e.run(2)
    assert seen and any(ev.diversity > 0.0 for ev in seen)
    assert any(ev.duration_ms > 0.0 for ev in seen)


def test_p0_pareto_label():
    e = _engine()
    r = e.run(2)
    pf = r["pareto_front"]
    assert pf.get("method") == "heuristic_fitness_vs_compactness"
    assert pf.get("is_nsga2") is False


def test_p0_immigrants_legacy_and_active():
    e0 = _engine(immigrant_fraction=0.0)
    r0 = e0.run(2)
    assert all(h.get("immigrants_injected", 0) == 0 for h in r0["history"])
    e1 = _engine(immigrant_fraction=0.25)
    r1 = e1.run(2)
    assert r1["history"][0].get("immigrants_injected") == 2
    # Deterministic: same seed reproduces the same trajectory prefix.
    e2 = _engine(immigrant_fraction=0.25)
    r2 = e2.run(2)
    assert r1["history"][0]["best_fitness"] == r2["history"][0]["best_fitness"]


def test_p1_self_store_contracts(tmp_path):
    import sys
    sys.path.insert(0, "src")
    from evolab.experience import ExperienceStore, wilson_interval
    db = tmp_path / "self.db"
    s = ExperienceStore(db)
    assert s.self_summary()["recent_runs"] == []
    s.record_self_run({"strategy": "test", "seed": 1, "generations": 2,
                       "population_size": 8, "best_fitness": 80.0,
                       "mean_fitness": 70.0, "evals_total": 16,
                       "early_stopped": False, "stagnation_gens": 0,
                       "diversity_final": 0.2})
    assert len(s.self_summary()["recent_runs"]) == 1
    row = s.record_capability("numeric_ga", True)
    assert row["trials"] == 1 and row["pass_rate"] == 1.0
    row = s.record_capability("numeric_ga", False)
    assert row["trials"] == 2 and row["pass_rate"] == 0.5
    lo, hi = wilson_interval(1, 2)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    # Kill-switch: EVOLAB_SELF=0 silences writes but never breaks reads.
    os.environ["EVOLAB_SELF"] = "0"
    try:
        s.record_self_run({"strategy": "x"})
        assert len(s.self_summary()["recent_runs"]) == 1
    finally:
        os.environ.pop("EVOLAB_SELF", None)
