"""Phase 2 (Meta-Evaluator) + Phase 3 (Self-Critic) contracts.

Preregistered gates (written before measurement, frozen in experience.py):
  P2-C1 collapsed plateau -> premature_convergence True with reason
        plateau_with_collapsed_diversity; healthy run -> False.
  P2-C2 short history (<5 gens) -> insufficient_data, no invented verdict.
  P2-C3 holdout_gap None -> overfit_risk None (honest unknown); gap>30 -> True.
  P2-C4 diagnose_run never raises on malformed input.
  P3-C1 critic keys in [0,1] (generalization None when unmeasurable).
  P3-C2 engine report carries extra.meta_evaluator + extra.self_critic and
        search trajectory is unchanged (same best vs disabled mirror).
  P3-C3 EVOLAB_SELF=0 suppresses the mirror but never breaks run().
"""
from __future__ import annotations

import os


def _hist(best, div=0.2, std=2.0, n=6):
    return [{"generation": i + 1, "best_fitness": b, "mean_fitness": b - 1.0,
             "std_fitness": std, "diversity": div} for i, b in enumerate(best)]


def test_p2_collapse_vs_healthy():
    import sys
    sys.path.insert(0, "src")
    from evolab.experience import diagnose_run
    collapsed = diagnose_run(_hist([80.0] * 6, div=0.01, std=0.1))
    assert collapsed["premature_convergence"] is True
    assert collapsed["stagnation"] is True
    assert "plateau_with_collapsed_diversity" in collapsed["reasons"]
    healthy = diagnose_run(_hist([70.0, 74.0, 78.0, 83.0, 87.0, 91.0], div=0.3, std=3.0))
    assert healthy["premature_convergence"] is False
    assert healthy["stagnation"] is False


def test_p2_short_history_and_overfit():
    import sys
    sys.path.insert(0, "src")
    from evolab.experience import diagnose_run
    short = diagnose_run(_hist([80.0, 81.0], n=2))
    assert short["reasons"] == ["insufficient_data"]
    assert short["overfit_risk"] is None
    assert diagnose_run(_hist([80.0] * 6), holdout_gap=35.0)["overfit_risk"] is True
    assert diagnose_run(_hist([80.0] * 6), holdout_gap=5.0)["overfit_risk"] is False
    assert diagnose_run(None)["reasons"] == ["insufficient_data"]
    assert diagnose_run([{"bogus": 1}] * 6)["premature_convergence"] is False


def test_p3_critic_ranges():
    import sys
    sys.path.insert(0, "src")
    from evolab.experience import self_critic_report
    rep = {"best_individual": {"fitness": 84.0}, "total_candidates_evaluated": 128,
           "history": _hist([70.0, 74.0, 78.0, 80.0, 83.0, 84.0], div=0.25)}
    c = self_critic_report(rep)
    assert c["quality"] == 0.84
    assert 0.0 <= c["efficiency"] <= 1.0 and c["efficiency"] > 0.0
    assert c["generalization"] is None
    assert 0.0 <= c["novelty"] <= 1.0
    assert "metrics only" in c["note"]


def test_p3_engine_mirror_and_invariance():
    import sys
    sys.path.insert(0, "src")
    from evolab import EvolutionEngine
    e = EvolutionEngine(population_size=8, seed=21)
    r = e.run(6)
    assert "meta_evaluator" in r["extra"] and "self_critic" in r["extra"]
    assert set(r["extra"]["meta_evaluator"]) >= {"premature_convergence", "stagnation", "overfit_risk"}
    # Mirror must not change search: same seed without mirror hook == same best.
    os.environ["EVOLAB_SELF"] = "0"
    try:
        e2 = EvolutionEngine(population_size=8, seed=21)
        r2 = e2.run(6)
    finally:
        os.environ.pop("EVOLAB_SELF", None)
    assert r["history"][-1]["best_fitness"] == r2["history"][-1]["best_fitness"]
    assert "meta_evaluator" not in r2.get("extra", {}) and "self_critic" not in r2.get("extra", {})
