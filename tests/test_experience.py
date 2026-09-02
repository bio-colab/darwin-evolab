"""Experience memory (Phase 1): fingerprint, recording, recall, fail-safety."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from evolab import CodeScenario
from evolab.experience import (
    ExperienceStore,
    ExperienceRecorderProxy,
    attach_experience_recorder,
    problem_fingerprint,
)
from evolab.repair import greedy_repair
from evolab.evaluators import FitnessResult


BUGGY = (
    "def parse_cli(args):\n"
    "    return {'port': args[0]}\n"
)
# Same structure, different formatting/comments — fingerprint must not change.
BUGGY_REFORMATTED = (
    "# a comment that must not matter\n"
    "def parse_cli(args):\n"
    "\n"
    "    return {'port': args[0]}\n"
)
# Structurally different bug: int conversion missing vs present.
FIXED = (
    "def parse_cli(args):\n"
    "    return {'port': int(args[0])}\n"
)


def _scenario(sources: str) -> CodeScenario:
    return CodeScenario(
        name="fp_test",
        description="port should be int",
        sources={"app.py": sources},
        target_file="app.py",
        func_name="parse_cli",
        test_cases=[((["8080"],), {"port": 8080})],
        holdout_cases=[((["3000"],), {"port": 3000})],
    )


# ---------- fingerprint ----------

def test_fingerprint_ignores_formatting_and_comments():
    a = problem_fingerprint({"app.py": BUGGY}, "app.py", "parse_cli")
    b = problem_fingerprint({"app.py": BUGGY_REFORMATTED}, "app.py", "parse_cli")
    assert a == b


def test_fingerprint_changes_with_structure():
    a = problem_fingerprint({"app.py": BUGGY}, "app.py", "parse_cli")
    b = problem_fingerprint({"app.py": FIXED}, "app.py", "parse_cli")
    assert a != b


def test_fingerprint_is_deterministic_and_short():
    fp1 = problem_fingerprint({"app.py": BUGGY}, "app.py", "parse_cli")
    fp2 = problem_fingerprint({"app.py": BUGGY}, "app.py", "parse_cli")
    assert fp1 == fp2 and len(fp1) == 20


# ---------- recording through a real greedy run ----------

def _attach(scenario: CodeScenario, tmp_path: Path):
    ev = scenario.create_evaluator()
    wired = attach_experience_recorder(
        ev,
        scenario.sources,
        scenario.target_file,
        scenario.func_name,
        db_path=tmp_path / "exp.db",
    )
    return wired, tmp_path / "exp.db"


def test_recorder_returns_identical_results(tmp_path):
    scenario = _scenario(BUGGY)
    raw_ev = scenario.create_evaluator()
    wired, _ = _attach(scenario, tmp_path)
    g = scenario.sources  # sources dict is not a genome; use a RepairGenome via greedy first eval
    from evolab.repair import RepairGenome
    genome = RepairGenome(sources=scenario.sources, target_file=scenario.target_file)
    a = raw_ev.evaluate(genome)
    b = wired.evaluate(genome)
    assert a.score == b.score
    assert a.passed_holdout == b.passed_holdout


def test_greedy_run_writes_experiences(tmp_path):
    scenario = _scenario(BUGGY)
    wired, db = _attach(scenario, tmp_path)
    genome, history, n_eval = greedy_repair(
        scenario.sources, scenario.target_file, wired, max_evals=32
    )
    store = ExperienceStore(db)
    fp = problem_fingerprint(scenario.sources, scenario.target_file, scenario.func_name)
    store.conn = store._conn
    rows = store._conn.execute(
        "SELECT COUNT(*), outcome FROM experiences WHERE problem_fingerprint=? GROUP BY outcome",
        (fp,),
    ).fetchall()
    total = sum(r[0] for r in rows)
    assert total == n_eval > 0
    outcomes = {r[1] for r in rows}
    assert outcomes <= {"baseline", "improvement", "success", "neutral", "error"}


def test_outcome_semantics_baseline_improvement(tmp_path):
    store = ExperienceStore(tmp_path / "sem.db")
    proxy = ExperienceRecorderProxy(
        raw=None, store=store, fingerprint="fpX", func_name="f", target_file="t.py"
    )

    class _G:  # genome stub without edits
        pass

    class _R:  # result stub
        def __init__(self, score, holdout=None):
            self.score = score
            self.passed_holdout = holdout
            self.artifacts = {}
            self.evaluation_time_ms = 1.0

    proxy._record(_G(), _R(50.0))
    proxy._record(_G(), _R(70.0))
    proxy._record(_G(), _R(60.0))
    rows = store._conn.execute(
        "SELECT outcome, fitness_delta, is_new_best FROM experiences ORDER BY id"
    ).fetchall()
    assert rows[0][0] == "baseline" and rows[0][1] == 0.0
    assert rows[1][0] == "improvement" and rows[1][2] == 1
    assert rows[2][0] == "neutral" and rows[2][2] == 0


# ---------- recall & stats ----------

def test_recall_buckets_and_stats(tmp_path):
    store = ExperienceStore(tmp_path / "recall.db")
    proxy = ExperienceRecorderProxy(
        raw=None, store=store, fingerprint="fpR", func_name="f", target_file="t.py"
    )

    class _G:
        pass

    class _R:
        def __init__(self, score, holdout):
            self.score = score
            self.passed_holdout = holdout
            self.artifacts = {}
            self.evaluation_time_ms = 1.0

    from evolab.repair import RepairEdit
    e1 = RepairEdit(kind="int_wrap", file="t.py", lineno=1, col_offset=0)
    e2 = RepairEdit(kind="bool_flip", file="t.py", lineno=2, col_offset=0)

    proxy._record(_G(), _R(40.0, None))          # baseline
    g2 = type("G2", (), {"edits": [e1]})()
    proxy._record(g2, _R(100.0, True))           # success with int_wrap
    g3 = type("G3", (), {"edits": [e2]})()
    proxy._record(g3, _R(0.0, False))            # error with bool_flip

    rec = store.recall("fpR", k=5)
    assert len(rec["successful"]) == 1 and rec["successful"][0]["outcome"] == "success"
    assert any(item["outcome"] in ("error",) for item in rec["failed"])

    stats = store.stats("fpR")
    kinds = stats["per_edit_kind"]
    assert kinds["int_wrap"]["success_rate"] == 1.0
    assert kinds["bool_flip"]["success_rate"] == 0.0
    assert stats["total_experiences"] == 3


# ---------- null-intervention & fail-safety ----------

def test_env_disable_returns_raw_evaluator(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLAB_EXPERIENCE", "0")
    scenario = _scenario(BUGGY)
    raw_ev = scenario.create_evaluator()
    wired = attach_experience_recorder(
        raw_ev, scenario.sources, scenario.target_file, scenario.func_name,
        db_path=tmp_path / "nope.db",
    )
    assert wired is raw_ev
    assert not (tmp_path / "nope.db").exists()


def test_broken_db_degrades_without_breaking_evaluation(tmp_path):
    scenario = _scenario(BUGGY)
    raw_ev = scenario.create_evaluator()
    # A directory as sqlite path → connect fails at attach → raw evaluator returned.
    wired = attach_experience_recorder(
        raw_ev, scenario.sources, scenario.target_file, scenario.func_name,
        db_path=tmp_path,  # directory, not a file
    )
    from evolab.repair import RepairGenome
    genome = RepairGenome(sources=scenario.sources, target_file=scenario.target_file)
    a = raw_ev.evaluate(genome)
    if isinstance(wired, ExperienceRecorderProxy):
        b = wired.evaluate(genome)
        assert b.score == a.score
        assert wired.store.healthy is False
    else:
        assert wired is raw_ev  # attach failed safe


def test_json_columns_round_trip(tmp_path):
    scenario = _scenario(BUGGY)
    wired, db = _attach(scenario, tmp_path)
    greedy_repair(scenario.sources, scenario.target_file, wired, max_evals=16)
    conn = sqlite3.connect(str(db))
    kinds, loci = conn.execute(
        "SELECT edit_kinds, edit_loci FROM experiences WHERE n_edits > 0 LIMIT 1"
    ).fetchone()
    parsed_kinds, parsed_loci = json.loads(kinds), json.loads(loci)
    assert isinstance(parsed_kinds, list) and all(isinstance(k, str) for k in parsed_kinds)
    assert all(isinstance(l, list) and len(l) == 3 for l in parsed_loci)
