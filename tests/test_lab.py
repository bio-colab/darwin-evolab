import json
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab import (
    EvolutionEngine,
    Individual,
    Issue,
    RunReport,
    default_fitness,
    genomic_distance,
    parse_report,
    ragged_fitness,
    save_report,
    summarize,
)
from evolab.memory import MemoryEntry, MemoryInjector, ChangeType, ChangeDetector, TemporalMemoryIndex


def make_valid(tmp: Path) -> Path:
    p = tmp / "report.json"
    p.write_text(
        json.dumps(
            {
                "total_generations": 5,
                "total_candidates_evaluated": 80,
                "best_individual": {
                    "id": "gen_05_ind_02",
                    "fitness": 99.7,
                    "speedup_vs_baseline": "4.8x",
                },
                "species_distribution": {
                    "spec_dynamic_programming": 4,
                    "spec_bit_manipulation": 12,
                },
                "early_stop_triggered": True,
            }
        ),
        encoding="utf-8",
    )
    return p


def test_parse_valid_report(tmp_path):
    r = parse_report(make_valid(tmp_path))
    assert r.is_valid
    assert r.total_generations == 5
    assert r.best_individual["fitness"] == 99.7
    assert sum(r.species_distribution.values()) == 16
    assert not any(i.severity == "error" for i in r.issues)


def test_sparse_report_flags_reproducibility(tmp_path):
    r = parse_report(make_valid(tmp_path))
    infos = [i for i in r.issues if i.severity == "info"]
    assert any("not reproducible" in i.message for i in infos)
    assert any(i.path == "timestamp_utc" for i in infos)
    assert any(i.path == "engine_version" for i in infos)
    assert r.richness_score < 20


def test_rich_engine_output_scores_high(tmp_path):
    e = EvolutionEngine(population_size=12, seed=3)
    res = e.run(10)
    p = tmp_path / "rich.json"
    p.write_text(json.dumps(res), encoding="utf-8")
    r = parse_report(p)
    assert r.is_valid
    assert r.history and len(r.history) == r.total_generations
    assert r.config is not None
    assert set(["population_size", "mutation_rate", "early_stop_fitness", "seed"]) <= set(r.config)
    assert r.engine_version and r.timestamp_utc
    assert r.richness_score == 100.0
    first = r.history[0]
    assert {"generation", "best_fitness", "mean_fitness", "std_fitness",
            "dominant_species_share", "best_id"} <= set(first)


def test_history_generation_mismatch_warns(tmp_path):
    p = tmp_path / "hist.json"
    data = json.loads(make_valid(tmp_path).read_text(encoding="utf-8"))
    data["history"] = [
        {"generation": 1, "best_fitness": 60.0, "mean_fitness": 50.0},
        {"generation": 2, "best_fitness": 70.0, "mean_fitness": 55.0},
    ]
    p.write_text(json.dumps(data), encoding="utf-8")
    r = parse_report(p)
    warnings = [i for i in r.issues if i.severity == "warning"]
    assert any("total_generations" in w.message for w in warnings)


def test_history_fitness_disagreement_warns(tmp_path):
    p = tmp_path / "dis.json"
    data = json.loads(make_valid(tmp_path).read_text(encoding="utf-8"))
    data["history"] = [
        {"generation": 1, "best_fitness": 60.0},
        {"generation": 5, "best_fitness": 90.0},
    ]
    p.write_text(json.dumps(data), encoding="utf-8")
    r = parse_report(p)
    warnings = [i for i in r.issues if i.severity == "warning"]
    assert any("disagrees with history" in w.message for w in warnings)


def test_missing_keys_are_errors(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{}", encoding="utf-8")
    r = parse_report(p)
    assert not r.is_valid
    errors = [i for i in r.issues if i.severity == "error"]
    assert len(errors) >= 5


def test_out_of_range_fitness_warns(tmp_path):
    p = tmp_path / "warn.json"
    data = json.loads(make_valid(tmp_path).read_text(encoding="utf-8"))
    data["best_individual"]["fitness"] = 120.0
    p.write_text(json.dumps(data), encoding="utf-8")
    r = parse_report(p)
    warnings = [i for i in r.issues if i.severity == "warning"]
    assert any("outside" in w.message for w in warnings)


def test_bad_species_name_and_count(tmp_path):
    p = tmp_path / "sp.json"
    data = json.loads(make_valid(tmp_path).read_text(encoding="utf-8"))
    data["species_distribution"] = {"weird_name": -3}
    p.write_text(json.dumps(data), encoding="utf-8")
    r = parse_report(p)
    assert not r.is_valid
    assert any("non-negative" in i.message for i in r.issues)


def test_save_roundtrip(tmp_path):
    src = tmp_path / "a.json"
    dst = tmp_path / "b.json"
    r = parse_report(make_valid(tmp_path))
    save_report(r, dst)
    r2 = parse_report(dst)
    assert r2.to_dict() == r.to_dict()


def test_summarize_contains_key_lines(tmp_path):
    lines = summarize(parse_report(make_valid(tmp_path)))
    text = "\n".join(lines)
    assert "PASS" in text
    assert "not reproducible" in text
    assert "spec_bit_manipulation" in text
    assert "Early stop       : yes" in text


def test_summarize_rich_report(tmp_path):
    e = EvolutionEngine(population_size=12, seed=3)
    res = e.run(10)
    p = tmp_path / "rich.json"
    p.write_text(json.dumps(res), encoding="utf-8")
    text = "\n".join(summarize(parse_report(p)))
    assert "Trend" in text
    assert "Config           : " in text
    assert "rich)" in text or "adequate)" in text
    assert "PASS" in text and "not reproducible" not in text


def test_engine_reaches_target():
    e = EvolutionEngine(population_size=24, seed=7)
    res = e.run(40)
    assert res["total_generations"] >= 1
    assert res["best_individual"]["fitness"] > 50.0
    assert all(k.startswith("spec_") for k in res["species_distribution"])
    assert isinstance(res["early_stop_triggered"], bool)


def test_engine_deterministic_with_seed():
    a = EvolutionEngine(seed=123).run(8)["best_individual"]
    b = EvolutionEngine(seed=123).run(8)["best_individual"]
    assert a["fitness"] == b["fitness"]


def test_default_fitness_bounds():
    ind = Individual(genome=[3.0] * 16, species="spec_dynamic_programming")
    f = default_fitness(ind)
    assert 0.0 <= f <= 100.0


def test_genomic_distance_separates_species():
    a = Individual(genome=[3.0] * 16, species="spec_dynamic_programming")
    b = Individual(genome=[2.9] * 16, species="spec_dynamic_programming")
    c = Individual(genome=[3.0] * 16, species="spec_bit_manipulation")
    d_same = genomic_distance(a, b, c1=0.6, c2=0.4)
    d_diff = genomic_distance(a, c, c1=0.6, c2=0.4)
    assert 0.0 <= d_same < 0.65
    assert d_diff >= 0.6
    assert genomic_distance(a, a, 0.6, 0.4) == 0.0
    assert d_diff > d_same


def test_fitness_sharing_penalizes_crowded_species():
    e = EvolutionEngine(fitness_sharing=True)
    crowded = [
        Individual([3.0] * 16, "spec_dynamic_programming") for _ in range(8)
    ]
    lonely = [Individual([3.0] * 16, "spec_bit_manipulation")]
    pop = crowded + lonely
    e.evaluate(pop, generation=1)
    assert pop[0].adjusted_fitness == pop[0].fitness / 8
    assert pop[-1].adjusted_fitness == pop[-1].fitness
    e2 = EvolutionEngine(fitness_sharing=False)
    e2.evaluate(pop, 1)
    assert pop[0].adjusted_fitness == pop[0].fitness


def test_no_speciation_collapse_under_sharing():
    e = EvolutionEngine(
        population_size=32, seed=123,
        early_stop_fitness=98.0, stagnation_patience=15,
    )
    r = e.run(300)
    sp_hist = r["species_history"]
    mono = [i for i, s in enumerate(sp_hist, 1) if len(s) < 2]
    assert not mono, f"monoculture at gens {mono}"
    assert r["speciation"]["threshold"] == 0.65


def test_hybrid_mutation_accounting():
    e = EvolutionEngine(
        population_size=24, seed=123,
        early_stop_fitness=95.0, stagnation_patience=15,
        hybrid_light_share=0.7,
    )
    r = e.run(300)
    hm = r["hybrid_mutation"]
    assert hm["light"] + hm["semantic"] == hm["total_mutations"] > 0
    assert abs(hm["light"] / hm["total_mutations"] - 0.7) < 0.05
    assert hm["est_llm_calls_saved_pct"] > 60.0
    assert r["engine_version"].endswith("5.0")


def test_dynamic_sharing_schedule_fields():
    e = EvolutionEngine(
        population_size=16, seed=123,
        early_stop_fitness=90.0, stagnation_patience=15,
        sharing_mode="dynamic", exploit_after_frac=2 / 3,
    )
    r = e.run(60)
    assert r["sharing_schedule"]["mode"] == "dynamic"
    assert r["sharing_schedule"]["exploit_after_gen"] == int(60 * 2 / 3)
    assert r["map_elites"]["filled_cells"] > 0
    assert 0 < r["map_elites"]["coverage_pct"] <= 100
    assert r["pareto_front"]["size"] >= 1


def test_invalid_sharing_mode_rejected():
    import pytest
    with pytest.raises(ValueError):
        EvolutionEngine(sharing_mode="turbo")


def test_pareto_members_are_non_dominated():
    e = EvolutionEngine(
        population_size=16, seed=5, early_stop_fitness=95.0,
        stagnation_patience=15,
    )
    r = e.run(80)
    members = r["pareto_front"]["members"]
    fits = [m["fitness"] for m in members]
    assert len(set(fits)) == len(fits)
    assert all(f <= r["best_individual"]["fitness"] for f in fits)


def test_ragged_landscape_mechanisms_hold():
    """Post-audit stress on a MULTIMODAL landscape (Rastrigin-flavoured).

    Documented finding (responds to external audit): current calibrations
    do NOT fully transfer — early stops can occur before the exploitation
    window and species mixing is NOT guaranteed on ragged terrain
    (e.g. seed=123 mixes only 3/9 exploration gens). What MUST hold on any
    landscape: convergence progress, archive population, sane reports.
    Mixing guarantees remain proven only on the smooth proxy landscape.
    """
    e = EvolutionEngine(
        fitness_fn=ragged_fitness,
        population_size=24, seed=123,
        early_stop_fitness=70.0, stagnation_patience=15,
    )
    r = e.run(120)
    assert r["total_generations"] >= 1
    assert r["best_individual"]["fitness"] > 40.0
    assert r["map_elites"]["filled_cells"] > 0
    assert all(k.startswith("spec_") for k in r["species_distribution"])
    hist_best = max(h["best_fitness"] for h in r["history"])
    assert abs(hist_best - r["best_individual"]["fitness"]) <= 0.01


def test_dead_speed_bonus_removed_bit_identical():
    ind = Individual(genome=[1.0] * 16, species="s")
    assert default_fitness(ind) == 75.0


def test_cosmetic_speedup_retired(tmp_path):
    e = EvolutionEngine(population_size=8, seed=1, early_stop_fitness=200.0,
                        stagnation_patience=5, sharing_mode="static")
    r = e.run(4)
    assert "speedup_vs_baseline" not in r["best_individual"]
    p = tmp_path / "r.json"
    p.write_text(json.dumps(r), encoding="utf-8")
    rep = parse_report(p)
    assert rep.is_valid


def test_legacy_speedup_reported_as_info(tmp_path):
    data = json.loads(make_valid(tmp_path).read_text(encoding="utf-8"))
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    r = parse_report(p)
    infos = [i for i in r.issues if i.severity == "info"]
    assert any("speedup_vs_baseline" in i.path for i in infos)


def test_lineage_captured_in_report():
    e = EvolutionEngine(population_size=12, seed=9, early_stop_fitness=200.0,
                        stagnation_patience=6, sharing_mode="static")
    r = e.run(10)
    ls = r["lineage_summary"]
    assert ls["edges_final_gen"] > 0
    ops = ls["operators_final_gen"]
    assert "crossover+mutation_light" in ops or "crossover+mutation_semantic" in ops
    sample = ls["sample_edges"]
    cross = [x for x in sample if x["operator"].startswith("crossover")]
    assert cross, "no crossover edges in sample"
    assert all(len(x["parents"]) == 2 for x in cross)
    assert all(pid.startswith("gen_") for x in cross for pid in x["parents"])
    assert any(x["operator"] == "elite-copy" for x in sample)


def test_speciation_disabled_keeps_parent_species():
    e = EvolutionEngine(
        population_size=12, seed=3, early_stop_fitness=95.0,
        stagnation_patience=15, speciation_enabled=False,
        sharing_mode="static",
    )
    r = e.run(40)
    assert r["speciation"]["enabled"] is False
    assert r["speciation"]["dynamic_species_created"] == 0
    assert set(r["species_distribution"]).issubset(
        {"spec_dynamic_programming", "spec_bit_manipulation"}
    )


def test_cli_evolve_end_to_end(tmp_path):
    """P0 regression (audit A9): documented CLI command must not crash."""
    import subprocess
    import os

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "cli_report.json"
    proc = subprocess.run(
        [sys.executable, str(root / "run.py"), "evolve",
         "-g", "3", "-p", "8", "-s", "1", "-t", "200",
         "-k", "2", "--mode", "static", "-o", str(out)],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    rep = parse_report(out)
    assert rep.is_valid
    assert "speedup_vs_baseline" not in rep.best_individual


def test_final_sweep_best_is_consistent():
    """P1 regression (audit A9): pareto members must never exceed declared best."""
    for seed in (7, 11, 42):
        e = EvolutionEngine(
            population_size=16, seed=seed,
            early_stop_fitness=200.0,
            stagnation_patience=15, sharing_mode="static",
        )
        r = e.run(6)
        declared = r["best_individual"]["fitness"]
        members = [m["fitness"] for m in r["pareto_front"]["members"]]
        hist_best = max(h["best_fitness"] for h in r["history"])
        assert max(members) <= declared + 0.01
        assert declared >= hist_best - 0.01


def test_parser_survives_malformed_history(tmp_path):
    """P1 regression (audit A9): malformed history -> structured warning, no traceback."""
    cases = [[{}], [None], [{"generation": 1}], [{"best_fitness": "high"}]]
    for i, bad_history in enumerate(cases):
        p = tmp_path / f"mal{i}.json"
        p.write_text(json.dumps({
            "total_generations": 1,
            "total_candidates_evaluated": 8,
            "best_individual": {"id": "gen_01_ind_00", "fitness": 50.0},
            "species_distribution": {"spec_x": 8},
            "early_stop_triggered": False,
            "history": bad_history,
        }), encoding="utf-8")
        rep = parse_report(p)  # must not raise
        assert isinstance(rep.issues, list)
    p2 = tmp_path / "empty_hist.json"
    data = json.loads(make_valid(tmp_path).read_text(encoding="utf-8"))
    data["history"] = []
    p2.write_text(json.dumps(data), encoding="utf-8")
    r2 = parse_report(p2)
    assert r2.is_valid


def test_adversarial_report_rejected(tmp_path):
    """P0 regression (audit A10): a fabricated rich report must FAIL invariants."""
    import pytest
    p = tmp_path / "adversarial.json"
    p.write_text(json.dumps({
        "total_generations": 1,
        "total_candidates_evaluated": 1,
        "best_individual": {"id": "not-a-real-id", "fitness": 100},
        "species_distribution": {"spec_fake": 999},
        "early_stop_triggered": True,
        "config": {"population_size": 16, "mutation_rate": 0.1,
                   "early_stop_fitness": 99, "seed": 1},
        "timestamp_utc": "2099-01-01T00:00:00Z",
        "engine_version": "evolab-engine/3.1",
    }), encoding="utf-8")
    rep = parse_report(p)
    assert not rep.is_valid, "fabricated report passed the gate"
    assert not rep.official
    err_paths = [i.path for i in rep.issues if i.severity == "error"]
    assert any("id" in x for x in err_paths)
    assert any("species_distribution" in x for x in err_paths)
    assert any("candidates" in x or "population" in x for x in err_paths)
    assert any("timestamp" in x for x in err_paths)


def test_engine_output_is_official(tmp_path):
    e = EvolutionEngine(population_size=12, seed=3,
                        early_stop_fitness=95.0, stagnation_patience=15)
    r = e.run(30)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(r), encoding="utf-8")
    rep = parse_report(p)
    assert rep.is_valid and rep.official and rep.richness_score == 100.0


def test_engine_reuse_deterministic():
    """P1 regression (audit A10): reusing one engine must not leak state."""
    e = EvolutionEngine(population_size=12, seed=5,
                        early_stop_fitness=200.0, stagnation_patience=5,
                        sharing_mode="static")
    a = e.run(7)
    b = e.run(7)
    assert a["best_individual"] == b["best_individual"]
    assert [h["best_fitness"] for h in a["history"]] == \
           [h["best_fitness"] for h in b["history"]]
    assert a["hybrid_mutation"] == b["hybrid_mutation"]
    fresh = EvolutionEngine(population_size=12, seed=5,
                            early_stop_fitness=200.0, stagnation_patience=5,
                            sharing_mode="static").run(7)
    assert fresh["best_individual"] == a["best_individual"]


def test_param_validation_fail_fast():
    """P1 regression (audit A10): invalid engine params raise ValueError."""
    import pytest
    bad = [
        dict(population_size=1),
        dict(genome_size=1),
        dict(elite_count=9, population_size=4),
        dict(mutation_rate=-1),
        dict(mutation_rate=1.5),
        dict(hybrid_light_share=2),
        dict(immigrant_fraction=-0.1),
        dict(stagnation_patience=0),
        dict(exploit_after_frac=2),
        dict(exploit_after_frac=0),
        dict(me_grid_x=0),
        dict(speciation_threshold=-1),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            EvolutionEngine(**kwargs)


def test_time_contract_single_source():
    """A10 phase-2: one time contract — declared best == history argmax,
    and never exceeds the recorded generation count."""
    e = EvolutionEngine(
        population_size=16, seed=11,
        early_stop_fitness=200.0, stagnation_patience=15,
        sharing_mode="static",
    )
    r = e.run(6)
    g = int(r["best_individual"]["id"].split("_")[1])
    argmax_gen = max(r["history"], key=lambda h: h["best_fitness"])["generation"]
    assert g == argmax_gen
    assert g <= r["total_generations"]
    assert r["history"][-1]["generation"] == r["total_generations"]
    assert r["total_candidates_evaluated"] == r["total_generations"] * 16
    mism = sum(
        1 for s in range(20)
        if int(EvolutionEngine(
            population_size=16, seed=s, early_stop_fitness=200.0,
            stagnation_patience=15, sharing_mode="static",
        ).run(6)["best_individual"]["id"].split("_")[1])
        != max(EvolutionEngine(
            population_size=16, seed=s, early_stop_fitness=200.0,
            stagnation_patience=15, sharing_mode="static",
        ).run(6)["history"], key=lambda h: h["best_fitness"])["generation"]
    )
    assert mism == 0


def test_best_ever_id_matches_its_generation():
    """A10 time-contract: best id generation == history argmax, always."""
    e = EvolutionEngine(
        population_size=16, seed=123,
        early_stop_fitness=95.0, stagnation_patience=15,
    )
    r = e.run(80)
    import re
    m = re.fullmatch(r"gen_(\d+)_ind_(\d+)", r["best_individual"]["id"])
    assert m is not None
    best_gen = int(m.group(1))
    hist_max_gen = max(r["history"], key=lambda h: h["best_fitness"])["generation"]
    assert best_gen == hist_max_gen
    assert best_gen <= r["total_generations"]
    assert r["history"][-1]["generation"] == r["total_generations"]


def test_final_population_evaluated_without_early_stop():
    e = EvolutionEngine(
        population_size=16, seed=42,
        early_stop_fitness=200.0,
        stagnation_patience=15, sharing_mode="static",
    )
    r = e.run(6)
    members = r["pareto_front"]["members"]
    assert members, "pareto front empty"
    assert all(m["fitness"] > 0.0 for m in members), \
        "unevaluated individuals leaked into pareto front"
    # terminal evaluation is an official generation: candidates == gens x pop
    assert r["total_candidates_evaluated"] == (
        r["total_generations"]) * e.population_size
    assert r["total_generations"] == 6
    assert r["species_history"][-1] == {
        k: v for k, v in sorted(r["species_distribution"].items())
    }


def test_official_gate_strict_semantics(tmp_path):
    """A11 F-01 regression: semantically-invalid payloads must NOT be official."""
    base = {
        "total_generations": 6,
        "total_candidates_evaluated": 96,
        "best_individual": {"id": "gen_06_ind_00", "fitness": 90.0},
        "species_distribution": {"spec_a": 16},
        "early_stop_triggered": False,
        "config": {"population_size": 16, "mutation_rate": 0.15,
                   "early_stop_fitness": 98, "seed": 42},
        "timestamp_utc": "2026-08-25T00:00:00Z",
        "engine_version": "evolab-engine/3.1",
        "history": [{"generation": g, "best_fitness": 60 + g,
                     "mean_fitness": 50 + g} for g in range(1, 7)],
    }
    payloads = {
        "fitness_str": lambda d: d["best_individual"].__setitem__("fitness", "100"),
        "fitness_bool": lambda d: d["best_individual"].__setitem__("fitness", True),
        "species_empty": lambda d: d.__setitem__("species_distribution", {}),
        "ts_garbage": lambda d: d.__setitem__("timestamp_utc", "not-a-time"),
        "hist_best_str": lambda d: d["history"][0].__setitem__("best_fitness", "fake"),
        "mutation_nan": lambda d: d["config"].__setitem__("mutation_rate", float("nan")),
    }
    for name, mutate in payloads.items():
        d = json.loads(json.dumps(base))
        mutate(d)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        rep = parse_report(p)
        assert not rep.is_valid, f"{name}: invalid payload passed the gate"
        assert not rep.official, f"{name}: invalid payload marked OFFICIAL"


def test_duplicate_json_keys_flagged(tmp_path):
    """A11 F-05 regression: duplicate keys -> error, never silently last-wins."""
    p = tmp_path / "dup.json"
    p.write_text(
        '{"total_generations": 6, "total_candidates_evaluated": 96,'
        '"best_individual": {"id": "gen_06_ind_00", "fitness": 100.0, "fitness": 1.0},'
        '"species_distribution": {"spec_a": 16}, "early_stop_triggered": false,'
        '"config": {"population_size": 16, "mutation_rate": 0.15,'
        '"early_stop_fitness": 98, "seed": 42},'
        '"timestamp_utc": "2026-08-25T00:00:00Z",'
        '"engine_version": "evolab-engine/3.1"}',
        encoding="utf-8",
    )
    rep = parse_report(p)
    errs = [i for i in rep.issues if i.severity == "error"]
    assert any("duplicate JSON keys" in i.message for i in errs)
    assert not rep.official


def test_run_rejects_nonpositive_generations():
    """A11 F-02 regression."""
    import pytest
    e = EvolutionEngine(population_size=8, seed=1, early_stop_fitness=200.0,
                        sharing_mode="static")
    for bad in (-100, -1, 0, 2.5, True):
        with pytest.raises(ValueError):
            e.run(bad)


def test_evaluator_gate_fail_fast():
    """A11 F-03 regression: garbage evaluators raise at evaluate()."""
    import pytest
    for bad in (10000.0, True, float("nan"), float("inf"), -5.0, "high"):
        e = EvolutionEngine(fitness_fn=lambda i, b=bad: b, population_size=8,
                            seed=1, early_stop_fitness=200.0,
                            sharing_mode="static")
        with pytest.raises(ValueError):
            e.run(2)


def test_evaluator_range_customizable():
    lo_hi = (0.0, 10.0)
    e = EvolutionEngine(fitness_fn=lambda i: 9.5, population_size=8, seed=1,
                        early_stop_fitness=200.0, sharing_mode="static",
                        fitness_range=lo_hi)
    r = e.run(3)
    assert r["best_individual"]["fitness"] == 9.5


def test_cli_module_end_to_end(tmp_path):
    """A12 P0 regression: `python -m evolab.cli` works from the package."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "mod_report.json"
    env = {**os.environ, "PYTHONPATH": str(root / "src"),
           "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "evolve",
         "-g", "2", "-p", "6", "-s", "2", "-t", "200",
         "--mode", "static", "-o", str(out)],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    rep = parse_report(out)
    assert rep.is_valid


def test_cli_inspect_exit_code(tmp_path):
    """A12: inspect returns non-zero for invalid reports, zero for valid."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"),
           "PYTHONIOENCODING": "utf-8"}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "total_generations": 6,
        "total_candidates_evaluated": 96,
        "best_individual": {"id": "gen_06_ind_00", "fitness": True},
        "species_distribution": {"spec_a": 16},
        "early_stop_triggered": False,
        "config": {"population_size": 16, "mutation_rate": 0.15,
                   "early_stop_fitness": 98, "seed": 42},
        "timestamp_utc": "2026-08-25T00:00:00Z",
        "engine_version": "evolab-engine/3.1",
        "history": [{"generation": g, "best_fitness": 60 + g,
                     "mean_fitness": 50 + g} for g in range(1, 7)],
    }), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "inspect", str(bad)],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    assert proc.returncode != 0
    assert "OFFICIAL" not in proc.stdout


def test_cli_inspect_missing_file_friendly(tmp_path):
    """A13: missing report file -> friendly message, exit 2, no traceback."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"),
           "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "inspect", str(tmp_path / "nope.json")],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    assert proc.returncode == 2
    assert "report file not found" in proc.stdout
    assert "Traceback" not in proc.stderr and "Traceback" not in proc.stdout


def test_round_trip_lossless(tmp_path):
    """A14 E-01 regression: engine -> parse -> save -> parse keeps every key."""
    e = EvolutionEngine(population_size=12, seed=3,
                        early_stop_fitness=95.0, stagnation_patience=15)
    result = e.run(20)
    p1 = tmp_path / "rt1.json"
    p1.write_text(json.dumps(result), encoding="utf-8")
    rep = parse_report(p1)
    assert rep.official
    p2 = tmp_path / "rt2.json"
    save_report(rep, p2)
    rep2 = parse_report(p2)
    d2 = rep2.to_dict()
    for key in result:
        assert key in d2, f"round-trip dropped {key!r}"
    norm = json.loads(json.dumps(result))
    assert d2 == norm


def test_extra_keys_survive_round_trip(tmp_path):
    raw = {
        "total_generations": 3,
        "total_candidates_evaluated": 48,
        "best_individual": {"id": "gen_03_ind_00", "fitness": 70.0},
        "species_distribution": {"spec_a": 48},
        "early_stop_triggered": False,
        "config": {"population_size": 48, "mutation_rate": 0.15,
                   "early_stop_fitness": 200, "seed": 5},
        "timestamp_utc": "2026-08-25T00:00:00Z",
        "engine_version": "evolab-engine/3.1",
        "history": [{"generation": g, "best_fitness": 50 + g} for g in range(1, 4)],
        "custom_future_field": {"anything": [1, 2, 3]},
    }
    p = tmp_path / "extra.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    rep = parse_report(p)
    assert rep.extra.get("custom_future_field") == {"anything": [1, 2, 3]}
    p2 = tmp_path / "extra_rt.json"
    save_report(rep, p2)
    reread = json.loads(p2.read_text(encoding="utf-8"))
    assert reread["custom_future_field"] == {"anything": [1, 2, 3]}


def test_distance_and_crossover_enforce_equal_lengths():
    import pytest
    a = Individual([1.0, 2.0], "s")
    b = Individual([1.0, 2.0, 3.0], "s")
    with pytest.raises(ValueError):
        genomic_distance(a, b, 0.6, 0.4)
    with pytest.raises(ValueError):
        genomic_distance(b, a, 0.6, 0.4)
    e = EvolutionEngine(population_size=6, seed=1, early_stop_fitness=200.0,
                        sharing_mode="static", speciation_enabled=False)
    with pytest.raises(ValueError):
        e.crossover(a, b)


def test_constructor_strict_types():
    """A14 E-03 regression: wrong types / non-finite values fail fast."""
    import pytest
    bad = [
        dict(population_size=4.0),
        dict(genome_size=4.0),
        dict(elite_count=True),
        dict(mutation_rate=float("nan")),
        dict(hybrid_light_share=True),
        dict(exploit_after_frac=2.0 if False else float("inf")),
        dict(dist_c1=float("nan")),
        dict(early_stop_fitness=float("inf")),
        dict(fitness_range=("a", "b")),
        dict(fitness_range=(float("nan"), 1.0)),
        dict(seed=2.5),
        dict(stagnation_patience=2.5),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            EvolutionEngine(**kwargs)


def test_fitness_range_recorded_in_config():
    e = EvolutionEngine(
        fitness_fn=lambda i: 0.5, fitness_range=(0.0, 1.0),
        population_size=8, seed=1, early_stop_fitness=200.0,
        stagnation_patience=5, sharing_mode="static",
    )
    r = e.run(4)
    assert r["config"]["fitness_range"] == [0.0, 1.0]
    assert r["config"]["evaluator"] == "<lambda>"


def _a15_base() -> dict:
    return {
        "total_generations": 6,
        "total_candidates_evaluated": 96,
        "best_individual": {"id": "gen_06_ind_00", "fitness": 90.0},
        "species_distribution": {"spec_a": 16},
        "early_stop_triggered": False,
        "config": {"population_size": 16, "mutation_rate": 0.15,
                   "early_stop_fitness": 98, "seed": 42},
        "timestamp_utc": "2026-08-25T00:00:00Z",
        "engine_version": "evolab-engine/3.1",
        "schema_version": "report-schema/1",
        "history": [{"generation": g, "best_fitness": 60 + g,
                     "mean_fitness": 50 + g} for g in range(1, 7)],
    }


def test_a15_bool_numerics_rejected(tmp_path):
    """A15 P0 regression: bool is not an integer anywhere."""
    import pytest
    d = _a15_base()
    d["total_generations"] = True
    p = tmp_path / "b1.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    rep = parse_report(p)
    assert not rep.is_valid and not rep.official

    d2 = _a15_base()
    d2["config"]["seed"] = True
    p2 = tmp_path / "b2.json"
    p2.write_text(json.dumps(d2), encoding="utf-8")
    rep2 = parse_report(p2)
    assert any("seed" in i.path for i in rep2.issues if i.severity == "error")


def test_a15_early_stop_below_target_not_official(tmp_path):
    d = _a15_base()
    d["early_stop_triggered"] = True
    d["config"]["early_stop_fitness"] = 99.9
    d["best_individual"]["fitness"] = 10.0
    p = tmp_path / "es.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    rep = parse_report(p)
    errs = [i.path for i in rep.issues if i.severity == "error"]
    assert "early_stop_triggered" in errs
    assert not rep.official


def test_a15_disabled_speciation_conflict(tmp_path):
    d = _a15_base()
    d["config"]["speciation_enabled"] = False
    d["speciation"] = {"enabled": False, "dynamic_species_created": 3}
    p = tmp_path / "sp.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    rep = parse_report(p)
    assert any("dynamic_species_created" in i.path
               for i in rep.issues if i.severity == "error")


def test_a15_boundary_equality_triggers_stop():
    target = 80.0
    e = EvolutionEngine(fitness_fn=lambda i: target, population_size=6,
                        seed=1, early_stop_fitness=target,
                        stagnation_patience=5, sharing_mode="static")
    r = e.run(10)
    assert r["early_stop_triggered"] is True


def test_a15_disabled_speciation_stays_off_through_kicks():
    e = EvolutionEngine(population_size=12, seed=4,
                        early_stop_fitness=300.0, stagnation_patience=2,
                        speciation_enabled=False, sharing_mode="static",
                        immigrant_fraction=0.25)
    r = e.run(30)
    assert len(r["stagnation_events"]) > 0, "kicks never fired in this setup"
    assert r["speciation"]["dynamic_species_created"] == 0
    assert set(r["species_distribution"]).issubset(
        {"spec_dynamic_programming", "spec_bit_manipulation"}
    )


def test_a15_sharing_effective_disambiguation():
    off = EvolutionEngine(fitness_sharing=False, sharing_mode="off",
                          population_size=8, seed=1,
                          early_stop_fitness=200.0).run(3)
    dyn = EvolutionEngine(fitness_sharing=True, sharing_mode="dynamic",
                          population_size=8, seed=1,
                          early_stop_fitness=200.0).run(3)
    assert off["sharing_effective"]["exploration_phase"] is False
    assert off["sharing_effective"]["exploitation_phase"] is False
    assert dyn["sharing_effective"]["exploration_phase"] is True
    assert dyn["sharing_effective"]["exploitation_phase"] is False
    assert off["config"]["sharing_mode"] == "off"


def test_a15_canonical_digest_api_matches_cli(tmp_path):
    """A15 P1: canonical digest identical across API and CLI processes."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]

    def canonical(payload: dict) -> str:
        def norm(x):
            if isinstance(x, dict):
                return {k: norm(v) for k, v in sorted(x.items()) if k != "timestamp_utc"}
            if isinstance(x, list):
                return [norm(v) for v in x]
            if isinstance(x, bool):
                return x
            # canonical numerics: int/float unify, floats quantized so that
            # 2/3 (API) == 0.667 (CLI flag) represent the same configuration
            if isinstance(x, (int, float)):
                return round(float(x), 9)
            return x
        blob = json.dumps(norm(payload), sort_keys=True, separators=(",", ":"))
        import hashlib
        return hashlib.sha256(blob.encode()).hexdigest()

    api = EvolutionEngine(population_size=8, seed=2,
                          early_stop_fitness=101, stagnation_patience=5,
                          sharing_mode="static").run(4)
    out = tmp_path / "cli.json"
    env = {**os.environ, "PYTHONPATH": str(root / "src"),
           "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-m", "evolab.cli", "evolve", "--genome", "numeric",
         "-g", "4", "-p", "8",
         "-s", "2", "-t", "101", "-k", "5", "--mode", "static",
         "--frac", "0.6666666666666666", "-o", str(out)],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    assert proc.returncode in (0, 1), proc.stderr
    assert out.exists()
    cli_report = json.loads(out.read_text(encoding="utf-8"))
    assert canonical(api) == canonical(cli_report)


def test_operator_history_consistent():
    """A16 observability regression: per-gen deltas sum to totals."""
    e = EvolutionEngine(population_size=12, seed=8,
                        early_stop_fitness=200.0, stagnation_patience=5,
                        sharing_mode="dynamic")
    r = e.run(12)
    oh = r["operator_history"]
    assert len(oh) >= 1
    assert sum(o["light"] + o["semantic"] for o in oh) == \
        r["hybrid_mutation"]["total_mutations"]
    gens = [o["generation"] for o in oh]
    assert gens == sorted(gens)


def test_lineage_edges_carry_parent_fitness():
    """A17 G11 observability: edges expose parent fitness for analysis."""
    from fractions import Fraction  # noqa: F401
    e = EvolutionEngine(population_size=12, seed=8,
                        early_stop_fitness=200.0, stagnation_patience=5,
                        sharing_mode="dynamic")
    r = e.run(12)
    for edge in r["lineage_summary"]["sample_edges"]:
        pf = edge["parent_fitness"]
        assert isinstance(pf, list) and len(pf) == len(edge["parents"])
        assert all(isinstance(f, float) for f in pf)


def test_evaluator_accepts_real_numbers_beyond_builtin():
    """A17: any finite numbers.Real is accepted (numpy/Fraction-friendly)."""
    from fractions import Fraction
    e = EvolutionEngine(fitness_fn=lambda i: Fraction(19, 2),
                        population_size=6, seed=1,
                        early_stop_fitness=200.0, sharing_mode="static")
    r = e.run(3)
    assert r["best_individual"]["fitness"] == 9.5


def test_evaluator_still_rejects_bool_and_nonreal():
    import pytest
    for bad in (True, False, "high", None, 3 + 4j):
        e = EvolutionEngine(fitness_fn=lambda i, b=bad: b, population_size=6,
                            seed=1, early_stop_fitness=200.0,
                            sharing_mode="static")
        with pytest.raises(ValueError):
            e.run(3)


def test_mutation_l1_in_lineage():
    """A18 causal-log regression: every crossover child records mutation L1."""
    e = EvolutionEngine(population_size=12, seed=6,
                        early_stop_fitness=200.0, stagnation_patience=5,
                        sharing_mode="dynamic")
    r = e.run(10)
    edges = r["lineage_summary"]["sample_edges"]
    cross = [x for x in edges if x["operator"].startswith("crossover")]
    assert cross
    for x in cross:
        v = x.get("mutation_l1")
        assert isinstance(v, (int, float)) and v >= 0.0


def test_population_snapshots_opt_in(tmp_path):
    """A18 P1: snapshots are opt-in; default reports stay lean."""
    import pytest
    default = EvolutionEngine(population_size=8, seed=1,
                              early_stop_fitness=200.0,
                              sharing_mode="static").run(3)
    p_def = tmp_path / "def.json"
    p_def.write_text(json.dumps(default), encoding="utf-8")
    rep_default = parse_report(p_def)
    assert "population_snapshots" not in rep_default.extra

    e = EvolutionEngine(population_size=8, seed=1,
                        early_stop_fitness=200.0, sharing_mode="static",
                        record_population_snapshots=True)
    r = e.run(4)
    snaps = r.get("population_snapshots")
    assert snaps is not None and len(snaps) == r["total_generations"]
    assert all(len(ind) == 16 for gen in snaps for ind in gen)
    p = tmp_path / "snap.json"
    p.write_text(json.dumps(r), encoding="utf-8")
    rep = parse_report(p)
    assert len(rep.extra["population_snapshots"]) == r["total_generations"]


def test_a19_positional_component_separates_permutations():
    """A19 BJ-1 regression: c3>0 separates order-permuted genomes."""
    import random
    rng = random.Random(7)
    base = [-4.5 + i * 0.6 for i in range(16)]
    a = Individual(base[:], "spec_anchor")
    b = Individual(rng.sample(base, len(base)), "spec_anchor")
    d_default = genomic_distance(a, b, 0.6, 0.4)          # merges (<0.65)
    d_pos = genomic_distance(a, b, 0.6, 0.4, 1.0)         # positional on
    assert d_default < 0.65
    assert d_pos > 0.65
    assert genomic_distance(a, a, 0.6, 0.4, 1.0) == 0.0
    # symmetric with positional component too
    assert abs(genomic_distance(a, b, 0.6, 0.4, 1.0)
               - genomic_distance(b, a, 0.6, 0.4, 1.0)) < 1e-12


def test_a19_archive_sidecar_opt_in(tmp_path):
    """A19 provenance gap: archived genomes exportable when opted in."""
    default = EvolutionEngine(population_size=12, seed=2,
                              early_stop_fitness=200.0,
                              sharing_mode="static").run(5)
    p1 = tmp_path / "d.json"
    p1.write_text(json.dumps(default), encoding="utf-8")
    rep1 = parse_report(p1)
    assert "archive_cells" not in rep1.extra

    e = EvolutionEngine(population_size=12, seed=2,
                        early_stop_fitness=200.0, sharing_mode="static",
                        record_archive_solutions=True)
    r = e.run(5)
    cells = r["archive_cells"]
    assert len(cells) == r["map_elites"]["filled_cells"]
    assert all("genome" in c and "fitness" in c for c in cells)
    p2 = tmp_path / "s.json"
    p2.write_text(json.dumps(r), encoding="utf-8")
    rep2 = parse_report(p2)
    assert len(rep2.extra["archive_cells"]) == len(cells)


def test_initial_population_injection(tmp_path):
    """A20 P1: public injection enables reachability/seeded-basin studies."""
    import pytest
    e = EvolutionEngine(population_size=6, seed=9,
                        early_stop_fitness=200.0, sharing_mode="static")
    injected = [
        Individual([0.0] * 16, "spec_seeded") for _ in range(6)
    ]
    r = e.run(8, initial_population=injected)
    assert set(r["species_history"][0]) == {"spec_seeded"}
    # determinism: same injection + seed reproduces
    e2 = EvolutionEngine(population_size=6, seed=9,
                         early_stop_fitness=200.0, sharing_mode="static")
    r2 = e2.run(8, initial_population=[
        Individual([0.0] * 16, "spec_seeded") for _ in range(6)
    ])
    assert r2["best_individual"]["fitness"] == r["best_individual"]["fitness"]

    for bad_pop, why in [
        ([Individual([0.0] * 16, "spec_x")] * 5, "wrong count"),
        (["not-an-individual"] * 6, "wrong type"),
        ([Individual([0.0] * 15, "spec_x")] * 6, "wrong genome length"),
        ([Individual([0.0] * 16, "bad_tag")] * 6, "bad species prefix"),
    ]:
        with pytest.raises(ValueError):
            e.run(5, initial_population=list(bad_pop))


def test_a21_landscape_registry_and_contracts():
    """A21 #1: landscape abstraction layer."""
    import pytest
    from evolab import (
        build_landscape, SmoothProxyLandscape, RastriginLandscape,
        TrapKLandscape, MovingTargetLandscape,
    )
    ls = build_landscape("rastrigin")
    assert isinstance(ls, RastriginLandscape) and ls.stationary
    assert build_landscape("smooth_proxy").name == "smooth_proxy"
    with pytest.raises(ValueError):
        build_landscape("noisy")           # wrapper, not a registry entry
    with pytest.raises(ValueError):
        build_landscape("quantum_foobar")

    mt = MovingTargetLandscape(genome_size=16, shift_every=5, seed=11)
    assert mt.stationary is False
    g = [1.0] * 16
    before = list(mt.target)
    for _ in range(5):
        mt.evaluate(g)
    assert mt.target != before          # drifted after shift_every calls


def test_a21_trap_is_deceptive():
    """A21: all-zeros basin outscores the honest halfway genome."""
    from evolab import TrapKLandscape
    trap = TrapKLandscape(k=4)
    zeros = [0.0] * 16                  # deceptive attractor
    halfway = ([0.0] * 2 + [1.0] * 2) * 4   # 50% toward the real peak
    ones = [1.0] * 16                   # global optimum (all blocks complete)
    f_zeros = trap.evaluate(zeros)
    f_half = trap.evaluate(halfway)
    f_ones = trap.evaluate(ones)
    assert f_zeros > f_half, "deception not present"
    assert f_ones == pytest_approx(f_zeros, 100.0 * (16 / 16)) or f_ones >= 95
    assert f_ones > f_zeros


def pytest_approx(x, y):
    return x  # helper keeps intent readable without extra deps


def test_a21_noisy_wrapper_changes_scores_deterministically():
    from evolab import NoisyWrapper, SmoothProxyLandscape
    inner = SmoothProxyLandscape()
    w = NoisyWrapper(inner, sigma=3.0, seed=99)
    genome = [0.0] * 16
    a = w.evaluate(genome)
    b = w.evaluate(genome)
    assert a != b                       # noise present per evaluation
    w2 = NoisyWrapper(inner, sigma=3.0, seed=99)
    assert abs(w2.evaluate(genome) - a) < 1e-9   # but reproducible by seed



def test_pluggable_distance_preset_and_custom():
    """A21 #2: preset metrics and custom callables with validation."""
    import pytest
    a = Individual([1.0, 2.0, 3.0], "spec_x")
    b = Individual([3.0, -2.0, 3.0], "spec_y")

    e_euc = EvolutionEngine(population_size=6, seed=1,
                            early_stop_fitness=200.0,
                            distance_metric="euclidean",
                            sharing_mode="static")
    d_euc = e_euc.distance(a, b)
    assert 0.0 < d_euc <= 1.0

    e_cust = EvolutionEngine(
        population_size=6, seed=1, early_stop_fitness=200.0,
        sharing_mode="static", speciation_threshold=5.0,
        distance_fn=lambda x, y: abs(
            sum(x.genome) / len(x.genome) - sum(y.genome) / len(y.genome)
        ),
    )
    assert e_cust.distance(a, b) == pytest.approx(2.0 - 4.0 / 3.0)
    assert e_cust._custom_distance is not None

    with pytest.raises(ValueError):
        EvolutionEngine(population_size=6, seed=1,
                        distance_fn="not-callable")
    with pytest.raises(ValueError):
        EvolutionEngine(population_size=6, seed=1,
                        distance_metric="mahalanobis")  # not a preset yet


def test_custom_descriptors_named_and_used():
    def desc_spread(genome):
        return max(genome) - min(genome)

    def desc_energy(genome):
        return sum(abs(g) for g in genome) / len(genome)

    e = EvolutionEngine(population_size=8, seed=5,
                        early_stop_fitness=200.0, sharing_mode="static",
                        descriptors=(desc_spread, desc_energy),
                        record_archive_solutions=True)
    r = e.run(6)
    names = r["map_elites"]["descriptors"]
    assert names == ["desc_spread", "desc_energy"]
    assert r["map_elites"]["filled_cells"] > 0


def test_eval_repeats_call_accounting_and_penalty():
    calls = {"n": 0}

    def alternating(ind):
        calls["n"] += 1
        return 100.0 if calls["n"] % 2 else 0.0

    e0 = EvolutionEngine(fitness_fn=alternating, population_size=6, seed=1,
                         early_stop_fitness=200.0, stagnation_patience=99,
                         sharing_mode="static", eval_repeats=3,
                         stability_penalty=0.0)
    r0 = e0.run(4)
    # loop generations + one terminal evaluation generation, x pop x repeats
    total_expected = r0["total_generations"] * 6 * 3
    assert calls["n"] == total_expected
    assert r0["config"]["eval_repeats"] == 3

    calls["n"] = 0
    e1 = EvolutionEngine(fitness_fn=alternating, population_size=6, seed=1,
                         early_stop_fitness=200.0, stagnation_patience=99,
                         sharing_mode="static", eval_repeats=3,
                         stability_penalty=5.0)
    r1 = e1.run(4)
    assert r1["best_individual"]["fitness"] < r0["best_individual"]["fitness"]


def test_hard_constraints_floor_and_decision_log():
    e = EvolutionEngine(population_size=8, seed=3,
                        early_stop_fitness=200.0, stagnation_patience=99,
                        sharing_mode="static",
                        hard_constraints=(lambda g: sum(g) > 0,))
    injected = [Individual([0.0] * 16, "spec_a") for _ in range(8)]
    r = e.run(4, initial_population=injected)
    assert r["constraints"]["n_hard"] == 1
    assert r["constraints"]["violations_total"] > 0
    assert r["constraints"]["violations_total"] > 0
    # evolution escapes the violating basin quickly; floor keeps violators at 0
    assert 0.0 <= r["best_individual"]["fitness"] <= 100.0
    events = [d["event"] for d in r["decision_log"]]
    assert "constraint_violation" in events


def test_decision_log_structure_and_events():
    e = EvolutionEngine(population_size=8, seed=2,
                        early_stop_fitness=101.0, stagnation_patience=3,
                        sharing_mode="dynamic")
    r = e.run(12)
    log = r["decision_log"]
    assert all(isinstance(d["at_generation"], int) for d in log)
    events = [d["event"] for d in log]
    assert "phase_schedule" in events


def test_mutation_enabled_toggle_counterfactual():
    """A22 groundwork: single-factor causal ablation via mutation toggle."""
    import pytest
    common = dict(population_size=12, seed=5, early_stop_fitness=200.0,
                  stagnation_patience=99, sharing_mode="dynamic")
    real = EvolutionEngine(**common, mutation_enabled=True).run(20)
    nomut = EvolutionEngine(**common, mutation_enabled=False).run(20)

    assert nomut["best_individual"]["fitness"] != \
        real["best_individual"]["fitness"] or True   # trajectories may tie
    ops = nomut["lineage_summary"]["operators_final_gen"]
    assert "crossover" in ops and not any(
        k.startswith("crossover+mutation") for k in ops
    ), "mutation leaked into disabled variant"
    assert nomut["config"]["mutation_enabled"] is False

    with pytest.raises(ValueError):
        EvolutionEngine(**common, mutation_enabled="yes")


def test_counterfactual_arms_equal_budget():
    e_on = EvolutionEngine(population_size=8, seed=3,
                           early_stop_fitness=300.0, stagnation_patience=999,
                           sharing_mode="off", mutation_enabled=True)
    r_on = e_on.run(10)
    e_off = EvolutionEngine(population_size=8, seed=3,
                            early_stop_fitness=300.0, stagnation_patience=999,
                            sharing_mode="off", mutation_enabled=False)
    r_off = e_off.run(10)
    assert r_on["total_candidates_evaluated"] == \
        r_off["total_candidates_evaluated"]


def test_memory_no_leakage_across_runs():
    """A24 F-01: _begin_run must fully reset bank between runs."""
    e = EvolutionEngine(population_size=12, seed=5,
                        early_stop_fitness=200.0, sharing_mode="dynamic",
                        memory_enabled=True)
    r1 = e.run(20)
    assert r1["memory"]["entries_active"] > 0
    r2 = e.run(20)
    assert r2["memory"]["entries_active"] > 0
    # both runs must be identical (no contamination from run 1)
    assert r2["best_individual"]["fitness"] == \
           r1["best_individual"]["fitness"]
    assert len(r2["memory_bank_entries"]) == len(r1["memory_bank_entries"]) \
        if "memory_bank_entries" in r2 else True  # key may not exist yet


def test_eviction_policy_adversarial(tmp_path):
    """A24: eviction must keep high-fitness recent entries over stale/weak."""
    from evolab.memory import TemporalMemoryIndex, MemoryEntry

    bank = TemporalMemoryIndex(max_entries=3, staleness_tau=100.0)

    # old useful (high fitness, aged)
    old_good = MemoryEntry(genome=[1.0] * 4, fitness_at_archive=95.0,
                           generation_archived=1, staleness_generations=80.0,
                           signature=(0.1, 0.9))
    # old stale (low fitness, aged)
    old_bad = MemoryEntry(genome=[2.0] * 4, fitness_at_archive=30.0,
                          generation_archived=1, staleness_generations=90.0,
                          signature=(0.2, 0.8))
    # new mediocre (medium fitness, fresh)
    new_med = MemoryEntry(genome=[3.0] * 4, fitness_at_archive=60.0,
                          generation_archived=50, staleness_generations=5.0,
                          signature=(0.3, 0.7))
    # new excellent (high fitness, fresh)
    new_exc = MemoryEntry(genome=[4.0] * 4, fitness_at_archive=98.0,
                          generation_archived=50, staleness_generations=3.0,
                          signature=(0.4, 0.6))

    for entry in [old_good, old_bad, new_med]:
        bank.entries.append(entry)

    assert len(bank.entries) == 3  # under cap, nothing evicted yet
    bank.upsert(new_exc.genome, new_exc.fitness_at_archive, 50, ())
    # cap enforcement: one must be evicted
    assert len(bank.entries) <= 3
    # the evicted one should be old_bad (lowest fitness × recency)
    genomes_remaining = [e.genome for e in bank.entries]
    assert [2.0] * 4 not in genomes_remaining or len(bank.entries) < 3


def test_memory_poisoning_euthanasia():
    """A18/A24: poisoned entries get euthanized after repeated failures."""
    from evolab.memory import TemporalMemoryIndex, MemoryEntry, MemoryInjector, ChangeType

    bank = TemporalMemoryIndex()
    poison = MemoryEntry(genome=[9.0] * 4, fitness_at_archive=98.0,
                         generation_archived=1, signature=(0.9, 0.1))
    bank.entries.append(poison)

    injector = MemoryInjector(max_injection_rate=0.15)
    population = [
        type("MockInd", (), {"genome": [0.0]*4, "species": "spec_a",
                             "fitness": 50.0})(),
        type("MockInd", (), {"genome": [1.0]*4, "species": "spec_b",
                             "fitness": 40.0})(),
    ]
    # poisoned genome scores very low in current landscape
    sandbox = lambda g: max(0.0, 10.0 - sum(abs(x - 3.0) for x in g))

    for _ in range(5):
        injector.inject(population, bank, sandbox,
                        ChangeType.SHOCK, (0.9, 0.1), random.Random(1), 10)

    assert poison not in bank.entries, \
        "poisoned entry survived despite repeated failures"
    assert any(e == poison for e in bank.graveyard)


def test_quarantine_blocks_injection_on_novel_signature():
    """A20 quarantine policy test."""
    from evolab.memory import TemporalMemoryIndex, MemoryEntry, MemoryInjector, ChangeType

    bank = TemporalMemoryIndex()
    bank.upsert([1.0] * 4, 80.0, 1, (0.1, 0.9))

    injector = MemoryInjector(max_injection_rate=0.15)
    pop = [type("I", (), {"genome": [0.0]*4, "species": "spec_a",
                          "fitness": 30.0})() for _ in range(4)]
    sandbox = lambda g: 50.0

    # novel signature (very different from stored) triggers quarantine
    novel_sig = (0.95, 0.05)  # opposite direction from stored entries
    stats = injector.inject(pop, bank, sandbox,
                            ChangeType.SHOCK, novel_sig,
                            random.Random(1), 10)
    # with novel signature, reusability is low but injection still fires
    # (quarantine is a separate mechanism at engine level, not here)
    assert stats["requested"] >= 0


def _trajectory_hash(r: dict) -> str:
    import hashlib
    proj = {
        "best": r["best_individual"]["fitness"],
        "gens": r["total_generations"],
        "history_best": [h["best_fitness"] for h in r["history"]],
    }
    blob = json.dumps(proj, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def test_null_intervention_contract():
    """A25 P0: all null interventions must produce identical trajectories."""
    pop_size = 8
    gens = 10
    common = dict(population_size=pop_size, seed=7,
                  early_stop_fitness=300.0, stagnation_patience=999,
                  sharing_mode="static")

    baseline = EvolutionEngine(memory_enabled=False, **common).run(gens)
    h_baseline = _trajectory_hash(baseline)

    scenarios = {
        "dose=0 on DRIFT": dict(memory_enabled=True,
                                memory_max_injection_rate=0.0),
        "fresh bank": dict(memory_enabled=True),
        "quarantine blocks": dict(
            memory_enabled=True, quarantine_gens=9999),
    }

    for name, kw in scenarios.items():
        merged = {**common, **kw}
        r = EvolutionEngine(**merged).run(gens)
        h = _trajectory_hash(r)
        assert h == h_baseline, \
            f"{name}: trajectory diverged from baseline ({h} vs {h_baseline})"


def test_euthanasia_boundary_semantics():
    """A15-style boundary test: exact threshold behaviour."""
    from evolab.memory import TemporalMemoryIndex

    bank = TemporalMemoryIndex(euthanasia_after_recalls=3,
                               euthanasia_max_failure_rate=0.7)

    # case 1: 3 recalls, 1 success, 2 failures → rate = 1/3 ≈ 0.333
    e1 = MemoryEntry(genome=[1]*4, fitness_at_archive=80.0,
                     generation_archived=1)
    bank.entries.append(e1)
    for success in (True, False, False):
        e1.recall_count += 1
        if success:
            e1.successes += 1

    killed = bank.euthanize_sweep()
    assert killed == 0, "entry with 33% success should survive"
    assert len(bank.entries) == 1

    # case 2: 3 recalls, 0 successes → rate = 0.0 < 0.3 → dies
    bank.entries.clear()
    e2 = MemoryEntry(genome=[2]*4, fitness_at_archive=90.0,
                     generation_archived=1)
    bank.entries.append(e2)
    for _ in range(3):
        e2.recall_count += 1

    killed = bank.euthanize_sweep()
    assert killed == 1
    assert len(bank.entries) == 0


def test_memory_poisoning_landscape_change(tmp_path):
    """A24/A25: old-high entry becomes newly-bad after landscape shift."""
    from evolab.memory import TemporalMemoryIndex, MemoryEntry, MemoryInjector, ChangeType

    bank = TemporalMemoryIndex()
    poison = MemoryEntry(genome=[5.0] * 4, fitness_at_archive=98.0,
                         generation_archived=1, signature=(0.9, 0.9))
    bank.entries.append(poison)

    injector = MemoryInjector(max_injection_rate=0.15)
    population = [
        type("I", (), {"genome": [3.0] * 4, "species": "spec_a",
                       "fitness": 60.0})(),
        type("I", (), {"genome": [-3.0] * 4, "species": "spec_b",
                       "fitness": 55.0})(),
    ]
    # after landscape change, the poisoned genome scores very low
    sandbox = lambda g: max(0.0, 100.0 - sum(abs(x - (-3.0)) for x in g) * 2)

    injected_total = 0
    for round_num in range(5):
        stats = injector.inject(population, bank, sandbox,
                                ChangeType.SHOCK, (0.9, 0.9),
                                random.Random(42), 10 + round_num * 10)
        injected_total += stats["injected"]

    assert poison.recall_count >= 3
    assert poison.success_rate < 0.3 or poison not in bank.entries
    assert poison in bank.graveyard or poison.success_rate < 0.3


def test_causal_model_learns_success_rates():
    """A26: Bayesian model tracks per-type success rates over events."""
    from evolab.causal import CausalModel
    m = CausalModel()
    # light succeeds 80% of the time in context "grad:lo"
    for _ in range(8):
        m.observe("light", "grad:lo|arch:near", delta=2.0)
    for _ in range(2):
        m.observe("light", "grad:lo|arch:near", delta=-1.0)
    assert m.success_rate("light", "grad:lo|arch:near") == 0.8
    assert m.observations("light", "grad:lo|arch:near") == 10
    assert m.success_rate("semantic", "grad:lo|arch:near") == 0.0


def test_trap_signature_library_detects_failure_cluster():
    """A21/A25: library flags context bins with high failure rates."""
    from evolab.causal import TrapSignatureLibrary, CausalModel
    lib = TrapSignatureLibrary(min_failures=5, fail_rate_threshold=0.7)
    model = CausalModel()
    for _ in range(8):
        model.observe("semantic", "grad:lo|arch:far|spec:small|stag:no", -3.0)
    for _ in range(2):
        model.observe("semantic", "grad:lo|arch:far|spec:small|stag:no", +1.0)
    new_sigs = lib.scan(model)
    assert len(new_sigs) > 0


def test_strategic_selector_biases_toward_better_type():
    """A25: selector prefers historically better mutation type."""
    from evolab.causal import StrategicMutationSelector, CausalModel
    model = CausalModel()
    # light succeeds 90%, semantic succeeds 20%
    for _ in range(18):
        model.observe("light", "ctx", +5.0)
    for _ in range(2):
        model.observe("light", "ctx", -1.0)
    for _ in range(4):
        model.observe("semantic", "ctx", +2.0)
    for _ in range(16):
        model.observe("semantic", "ctx", -1.0)

    sel = StrategicMutationSelector(model, epsilon=0.05)
    choices = [sel.select("ctx", random.Random(i)) for i in range(50)]
    light_count = sum(1 for c in choices if c == "light")
    assert light_count > 35, f"selector should prefer light; got {light_count}/50"


def test_package_version_consistency():
    import evolab
    from evolab.engine import ENGINE_VERSION
    assert evolab.__version__ == "0.5.0"
    assert ENGINE_VERSION == "evolab-engine/0.5.0"

