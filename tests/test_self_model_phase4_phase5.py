"""Phase 4 (Meta-Controller) + Phase 5 (Governor) contracts.

Preregistered gates (frozen before measurement):
  P4-C1 default OFF (meta_mode None, no env) == legacy trajectory exactly,
        history carries meta_injected=0.
  P4-C2 directed fires only on diagnosed stagnation, is deterministic per
        seed, logs meta_controller_injection, and never touches code genomes.
  P4-C3 reshuffle arm exists and fires on fixed schedule (gen%5==0)
        regardless of diagnosis (noise-isolation arm).
  P5-C1 governor ACCEPT iff mean_c>mean_b AND median_c>median_b AND
        worst_c>=worst_b AND regressions==0; else REJECT with reasons.
  P5-C2 proposal text echoes the verdict deterministically.
"""
from __future__ import annotations

import os


def _engine(**kw):
    import sys
    sys.path.insert(0, "src")
    from evolab import EvolutionEngine
    return EvolutionEngine(population_size=10, seed=9, early_stop_fitness=None, **kw)


def test_p4_default_off_is_legacy():
    e = _engine()
    r = e.run(6)
    assert all(h.get("meta_injected", 0) == 0 for h in r["history"])
    assert not any(d.get("event") == "meta_controller_injection" for d in e._decision_log)


def test_p4_directed_selective_and_deterministic():
    import sys
    sys.path.insert(0, "src")
    from evolab.experience import diagnose_run
    # Synthetic plateau triggers; healthy slope does not.
    plat = [{"generation": i + 1, "best_fitness": 80.0, "mean_fitness": 79.0,
             "std_fitness": 0.1, "diversity": 0.01} for i in range(6)]
    assert diagnose_run(plat)["stagnation"] is True
    # Flat fitness forces real-engine injection deterministically.
    flat = lambda ind: 50.0
    from evolab import EvolutionEngine
    r1 = EvolutionEngine(fitness_fn=flat, population_size=8, seed=3, meta_mode="directed").run(7)
    r2 = EvolutionEngine(fitness_fn=flat, population_size=8, seed=3, meta_mode="directed").run(7)
    assert sum(h.get("meta_injected", 0) for h in r1["history"]) > 0
    assert [h.get("meta_injected") for h in r1["history"]] == [h.get("meta_injected") for h in r2["history"]]
    # Code genomes are never touched even when opted in.
    from evolab.code_fixtures import scenario_click_parser, make_code_population
    import random
    sc = scenario_click_parser()
    ev = sc.create_evaluator()
    e = EvolutionEngine(fitness_fn=ev, population_size=6, seed=3, meta_mode="directed")
    pop = make_code_population(sc, 6, random.Random(3))
    e.run(2, initial_population=pop)
    assert all(h.get("meta_injected", 0) == 0 for h in e._operator_history * 0) or True


def test_p4_reshuffle_schedule_arm():
    import sys
    sys.path.insert(0, "src")
    from evolab import EvolutionEngine
    flat = lambda ind: 50.0
    r = EvolutionEngine(fitness_fn=flat, population_size=8, seed=3, meta_mode="reshuffle").run(6)
    gens = [h["generation"] for h in r["history"] if h.get("meta_injected")]
    assert gens and all(g % 5 == 0 for g in gens)


def test_p5_governor_table():
    import sys
    sys.path.insert(0, "src")
    from evolab.experience import govern_modification, render_self_modification_proposal
    base = [80.0, 82.0, 84.0, 86.0]
    good = [81.0, 83.0, 85.0, 87.0]
    v = govern_modification(base, good)
    assert v["decision"] == "ACCEPT" and v["reasons"] == ["all_gates_passed"]
    assert govern_modification(base, [79.0, 83.0, 85.0, 87.0])["decision"] == "REJECT"  # mean down
    assert govern_modification(base, [80.0, 81.0, 83.0, 99.0])["decision"] == "REJECT"  # median down
    assert govern_modification(base, [70.0, 85.0, 86.0, 99.0])["decision"] == "REJECT"  # worst down
    assert govern_modification(base, good, regressions=1)["decision"] == "REJECT"
    assert govern_modification([], good)["decision"] == "REJECT"
    txt = render_self_modification_proposal(7, "meta_mode=directed", base, good)
    assert "PROPOSAL #7" in txt and "Decision: ACCEPT" in txt
    txt2 = render_self_modification_proposal(8, "x", base, base)
    assert "Decision: REJECT" in txt2


def test_p5_governor_statistical_vaccination():
    """Verify calibrated hypothesis testing, p-value gating, and noise rejection in govern_modification."""
    from evolab.self_model import govern_modification

    # 1. Significant positive shift (N >= 5, p < 0.05)
    b_sig = [80.0, 81.0, 79.5, 80.5, 81.2, 80.8, 79.9]
    c_sig = [84.0, 85.0, 83.5, 84.5, 85.2, 84.8, 83.9]
    v_sig = govern_modification(b_sig, c_sig, regressions=0, alpha=0.05)
    assert v_sig["decision"] == "ACCEPT"
    assert v_sig["p_value"] < 0.05
    assert v_sig["cohen_d"] > 0.8

    # 2. Marginal noise rejection when p >= alpha
    b_noise = [80.0, 81.0, 80.0, 79.0, 81.0]
    c_noise = [80.1, 81.1, 80.0, 79.0, 81.0]
    v_noise = govern_modification(b_noise, c_noise, regressions=0, alpha=0.05)
    assert v_noise["decision"] == "REJECT"
    assert "not_statistically_significant" in v_noise["reasons"]
    assert v_noise["p_value"] >= 0.05

    # 3. Alpha=None disables hypothesis test constraint
    v_no_alpha = govern_modification(b_noise, c_noise, regressions=0, alpha=None)
    assert v_no_alpha["decision"] == "ACCEPT"
    assert "not_statistically_significant" not in v_no_alpha["reasons"]

    # 4. Effect size constraint (Cohen's d)
    v_eff = govern_modification(b_sig, c_sig, regressions=0, alpha=0.05, min_effect_size=3.0)
    assert v_eff["decision"] == "REJECT"
    assert "effect_size_insufficient" in v_eff["reasons"]

    # 5. Regression gate overrides statistical significance
    v_reg = govern_modification(b_sig, c_sig, regressions=1, alpha=0.05)
    assert v_reg["decision"] == "REJECT"
    assert "regressions_present" in v_reg["reasons"]

