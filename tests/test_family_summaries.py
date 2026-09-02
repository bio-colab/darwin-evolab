"""Phase 3: family summaries, per-function views, run metrics, A/B control knob."""
from __future__ import annotations

import random

import pytest

from evolab import CodeScenario, EngineConfig, EvolutionEngine
from evolab.code_fixtures import make_code_population
from evolab.experience import (
    ExperienceMutationPrior,
    ExperienceRecorderProxy,
    ExperienceStore,
    attach_experience_recorder,
    function_summaries,
    problem_fingerprint,
    render_family_report,
)


def _row(run_id, fp, eval_index, kinds, score, holdout, outcome, func="parse_cli", tgt="app.py"):
    return {
        "run_id": run_id,
        "eval_index": eval_index,
        "problem_fingerprint": fp,
        "func_name": func,
        "target_file": tgt,
        "genome_class": "RepairGenome",
        "edit_kinds": kinds,
        "edit_loci": [],
        "n_edits": len(kinds),
        "score": score,
        "fitness_delta": 0.0,
        "is_new_best": False,
        "passed_holdout": holdout,
        "eval_ms": 0.1,
        "outcome": outcome,
        "created_at": "2026-01-01T00:00:00",
    }


# ---------- run_metrics ----------

def test_run_metrics_first_success(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    store.record(_row("r1", "fp", 1, [], 20.0, None, "baseline"))
    store.record(_row("r1", "fp", 2, ["int_wrap"], 55.0, None, "improvement"))
    store.record(_row("r1", "fp", 3, ["int_wrap"], 100.0, True, "success"))
    store.record(_row("r1", "fp", 4, ["int_wrap"], 100.0, True, "success"))
    m = store.run_metrics("r1")
    assert m["evals_total"] == 4
    assert m["first_success_eval"] == 3
    assert m["first_success_score"] == 100.0
    store.close()


def test_run_metrics_censored_when_never_passing_holdout(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    for i in range(1, 6):
        store.record(_row("r1", "fp", i, [], 20.0, None, "baseline"))
    m = store.run_metrics("r1")
    assert m["evals_total"] == 5
    assert m["first_success_eval"] is None
    assert m["first_success_score"] is None
    store.close()


def test_run_metrics_unknown_run_and_broken_store(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    store.record(_row("r1", "fp", 1, [], 20.0, None, "baseline"))
    assert store.run_metrics("nope")["evals_total"] == 0
    store.close()
    # fail-safe: a broken (closed) store returns the zero shape, never raises
    m = store.run_metrics("r1")
    assert m == {"evals_total": 0, "first_success_eval": None, "first_success_score": None}


# ---------- family_summaries ----------

def test_family_summaries_empty_store(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    assert store.family_summaries() == []
    assert render_family_report(store.family_summaries()) == ""
    store.close()


def test_family_summaries_aggregation(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    fp = "f" * 20
    # 5 experiences for one family: 1 baseline, 2 errors, 1 improvement, 1 success
    store.record(_row("a", fp, 1, [], 20.0, None, "baseline"))
    store.record(_row("a", fp, 2, ["int_wrap"], 0.0, None, "error"))
    store.record(_row("a", fp, 3, ["int_wrap"], 30.0, None, "error"))
    store.record(_row("a", fp, 4, ["int_wrap", "bool_flip"], 60.0, None, "improvement"))
    store.record(_row("b", fp, 1, ["int_wrap", "bool_flip"], 100.0, True, "success"))
    rows = store.family_summaries()
    assert len(rows) == 1
    s = rows[0]
    assert s["problem_fingerprint"] == fp
    assert s["function_key"] == "parse_cli@app.py"
    assert s["n_experiences"] == 5
    assert s["n_runs"] == 2  # two distinct run_ids
    assert s["outcomes"] == {"baseline": 1, "error": 2, "improvement": 1, "success": 1}
    assert s["holdout_successes"] == 1
    assert s["best_holdout_score"] == 100.0
    assert s["best_success_kinds"] == ["int_wrap", "bool_flip"]
    kinds = s["per_edit_kind"]
    assert kinds["int_wrap"]["n"] == 4 and kinds["int_wrap"]["holdout_success"] == 1
    assert kinds["int_wrap"]["errors"] == 2
    assert kinds["int_wrap"]["success_rate"] == 0.25
    assert kinds["bool_flip"]["n"] == 2 and kinds["bool_flip"]["success_rate"] == 0.5
    store.close()


def test_family_summaries_ordering_deterministic(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    fp_a = "a" * 20  # 2 experiences
    fp_b = "b" * 20  # 3 experiences -> first
    for i in range(2):
        store.record(_row("r", fp_a, i + 1, [], 20.0, None, "baseline"))
    for i in range(3):
        store.record(_row("r", fp_b, i + 1, [], 20.0, None, "baseline"))
    first = store.family_summaries()
    again = store.family_summaries()
    assert [r["problem_fingerprint"] for r in first] == [fp_b, fp_a]
    assert first == again  # byte-stable across calls
    store.close()


def test_family_summaries_broken_store(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    store.record(_row("a", "f" * 20, 1, [], 20.0, None, "baseline"))
    store.close()
    assert store.family_summaries() == []


def test_family_summaries_success_kind_tie_broken_by_eval_index(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    fp = "f" * 20
    store.record(_row("a", fp, 1, ["index_flip"], 100.0, True, "success"))
    store.record(_row("a", fp, 2, ["int_wrap"], 100.0, True, "success"))
    s = store.family_summaries()[0]
    assert s["best_success_kinds"] == ["index_flip"]  # equal score -> earlier eval wins
    store.close()


# ---------- function_summaries ----------

def test_function_summaries_merges_bug_variants_of_one_function(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    fp1, fp2 = "1" * 20, "2" * 20  # same function, different bugs
    store.record(_row("r1", fp1, 1, ["int_wrap"], 100.0, True, "success"))
    store.record(_row("r1", fp1, 2, ["int_wrap"], 30.0, None, "error"))
    store.record(_row("r2", fp2, 1, ["int_wrap"], 100.0, True, "success"))
    store.record(_row("r2", fp2, 2, ["compare_flip"], 60.0, None, "improvement"))
    merged = function_summaries(store.family_summaries())
    assert len(merged) == 1
    slot = merged[0]
    assert slot["function_key"] == "parse_cli@app.py"
    assert slot["n_fingerprints"] == 2
    assert slot["n_experiences"] == 4
    assert slot["n_runs"] == 2  # run_ids cannot span fingerprints -> no double count
    assert slot["holdout_successes"] == 2
    iw = slot["per_edit_kind"]["int_wrap"]
    assert iw["n"] == 3 and iw["holdout_success"] == 2
    assert iw["success_rate"] == round(2 / 3, 4)
    assert slot["per_edit_kind"]["compare_flip"]["n"] == 1
    store.close()


def test_function_summaries_sorting(tmp_path):
    rows = [
        {"function_key": "b@f.py", "n_fingerprints": 1, "n_experiences": 2, "n_runs": 1,
         "holdout_successes": 0, "per_edit_kind": {}},
        {"function_key": "a@f.py", "n_fingerprints": 1, "n_experiences": 2, "n_runs": 1,
         "holdout_successes": 0, "per_edit_kind": {}},
        {"function_key": "c@f.py", "n_fingerprints": 1, "n_experiences": 9, "n_runs": 1,
         "holdout_successes": 0, "per_edit_kind": {}},
    ]
    out = function_summaries(rows)
    assert [s["function_key"] for s in out] == ["c@f.py", "a@f.py", "b@f.py"]


# ---------- render_family_report ----------

def test_render_family_report_is_deterministic_and_contentful(tmp_path):
    store = ExperienceStore(tmp_path / "m.db")
    fp = "abc1234567" + "0" * 10
    store.record(_row("a", fp, 1, ["int_wrap"], 100.0, True, "success"))
    text1 = render_family_report(store.family_summaries())
    text2 = render_family_report(store.family_summaries())
    assert text1 == text2
    assert "abc1234567" in text1
    assert "parse_cli@app.py" in text1
    assert "int_wrap:1@1.00" in text1
    store.close()


# ---------- prior_enabled / prior_kwargs (A/B control knob) ----------

def _seeded_store(tmp_path):
    store = ExperienceStore(tmp_path / "p.db")
    fp = "f" * 20
    store.record(_row("r", fp, 1, ["int_wrap"], 100.0, True, "success"))
    store.record(_row("r", fp, 2, ["int_wrap"], 30.0, None, "error"))
    store.record(_row("r", fp, 3, ["int_wrap"], 30.0, None, "error"))
    # M6: a second, differentially-losing kind — single-kind weight
    # vectors are uniform (zero information) and now collapse to None,
    # so the passthrough contract is exercised on a two-kind query.
    store.record(_row("r", fp, 4, ["str_swap"], 30.0, None, "error"))
    store.record(_row("r", fp, 5, ["str_swap"], 30.0, None, "error"))
    store.record(_row("r", fp, 6, ["str_swap"], 30.0, None, "error"))
    return store, fp


def test_recorder_prior_disabled_returns_none(tmp_path):
    store, fp = _seeded_store(tmp_path)
    proxy = ExperienceRecorderProxy(object(), store, fp, prior_enabled=False)
    assert proxy.mutation_prior is None  # engine's getattr -> None -> old behavior
    store.close()


def test_recorder_prior_kwargs_passthrough(tmp_path):
    store, fp = _seeded_store(tmp_path)
    # n=3 >= default min_support=3: int_wrap rate=(1+1)/(3+2)=0.4 -> 0.7;
    # str_swap rate=(0+1)/(3+2)=0.2 -> 0.6. Non-uniform -> survives the
    # M6 zero-signal gate, and the exact blended values are still asserted.
    default = ExperienceRecorderProxy(object(), store, fp)
    assert default.mutation_prior.kind_weights(["int_wrap", "str_swap"]) == {
        "int_wrap": pytest.approx(0.7),
        "str_swap": pytest.approx(0.6),
    }
    # passthrough raises min_support -> BOTH kinds neutral -> uniform
    # weights -> the M6 gate collapses to None. The None-vs-weights
    # difference is itself the observable proof that min_support flowed
    # through prior_kwargs (at min_support=3 this query returns weights).
    custom = ExperienceRecorderProxy(object(), store, fp, prior_kwargs={"min_support": 10})
    assert custom.mutation_prior.kind_weights(["int_wrap", "str_swap"]) is None
    store.close()


def _knob_scenario():
    return CodeScenario(
        name="knob", description="", sources={"app.py": "def f():\n    return 1\n"},
        target_file="app.py", func_name="f",
        test_cases=[((), 1)], holdout_cases=[((), 1)],
    )


def test_attach_prior_defaults_off_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLAB_EXPERIENCE", raising=False)
    monkeypatch.delenv("EVOLAB_EXPERIENCE_PRIOR", raising=False)
    scenario = _knob_scenario()
    wired = attach_experience_recorder(
        scenario.create_evaluator(), scenario.sources, scenario.target_file,
        scenario.func_name, db_path=tmp_path / "x.db",
    )
    assert isinstance(wired, ExperienceRecorderProxy)
    assert wired.prior_enabled is False  # Phase-3 verdict: prior is opt-in
    assert wired.mutation_prior is None


def test_attach_prior_opt_in_via_env(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLAB_EXPERIENCE", raising=False)
    monkeypatch.setenv("EVOLAB_EXPERIENCE_PRIOR", "1")
    scenario = _knob_scenario()
    wired = attach_experience_recorder(
        scenario.create_evaluator(), scenario.sources, scenario.target_file,
        scenario.func_name, db_path=tmp_path / "x.db",
    )
    assert wired.prior_enabled is True
    assert wired.mutation_prior is not None


def test_attach_explicit_flag_overrides_env(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLAB_EXPERIENCE", raising=False)
    monkeypatch.setenv("EVOLAB_EXPERIENCE_PRIOR", "1")
    scenario = _knob_scenario()
    wired = attach_experience_recorder(
        scenario.create_evaluator(), scenario.sources, scenario.target_file,
        scenario.func_name, db_path=tmp_path / "x.db", prior_enabled=False,
    )
    assert wired.prior_enabled is False  # explicit beats environment
    assert wired.mutation_prior is None


# ---------- engine-level: control arm runs unchanged and still records ----------

def test_engine_control_arm_records_without_prior(tmp_path):
    buggy = (
        "def parse_cli(args, verbose):\n"
        "    port = args[0]\n"
        "    if verbose == True:\n"
        "        return {'port': args[0]}\n"
        "    return {'port': port}\n"
    )
    scenario = CodeScenario(
        name="port_int", description="", sources={"app.py": buggy},
        target_file="app.py", func_name="parse_cli",
        test_cases=[((["8080"], False), {"port": 8080})],
        holdout_cases=[((["3000"], False), {"port": 3000})],
    )
    ev = scenario.create_evaluator()
    wired = attach_experience_recorder(
        ev, scenario.sources, scenario.target_file, scenario.func_name,
        db_path=tmp_path / "c.db", run_id="ctrl_1", prior_enabled=False,
    )
    assert getattr(wired, "mutation_prior", None) is None
    engine = EvolutionEngine(
        fitness_fn=wired,
        config=EngineConfig(generations=3, population_size=6, elite_count=1,
                            genome_size=2, seed=7),
    )
    initial = make_code_population(scenario, 6, random.Random(7))
    engine.run(generations=3, initial_population=initial)
    store = ExperienceStore(tmp_path / "c.db")
    m = store.run_metrics("ctrl_1")
    assert m["evals_total"] > 0
    assert len(store.family_summaries()) == 1
    store.close()
