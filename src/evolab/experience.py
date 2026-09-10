"""experience.py — Cross-run episodic experience memory for the code track.

Phase 1 of the experience-memory plan: an append-only SQLite store of
per-evaluation experiences, written by a transparent evaluator proxy, keyed by
a deterministic structure-only problem fingerprint. Retrieval is exact-match
and cheap (stdlib sqlite3 + json). No embeddings, no LLM, no genome changes.

Design contracts (non-negotiable, mirroring the repo's audit culture):

- Observation only. The proxy never modifies scores, holdout flags, or
  artifacts. Memory is a search prior, never a fitness term.
- Null-intervention: ``EVOLAB_EXPERIENCE=0`` returns the raw evaluator, and a
  broken store degrades to a silent no-op recorder — the store must never
  break a run.
- Facts, not answers. Rows store what happened (edits, score, holdout,
  outcome), not solutions to replay.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    eval_index INTEGER NOT NULL,
    problem_fingerprint TEXT NOT NULL,
    func_name TEXT NOT NULL DEFAULT '',
    target_file TEXT NOT NULL DEFAULT '',
    genome_class TEXT NOT NULL DEFAULT '',
    edit_kinds TEXT NOT NULL DEFAULT '[]',
    edit_loci TEXT NOT NULL DEFAULT '[]',
    n_edits INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL,
    fitness_delta REAL,
    is_new_best INTEGER NOT NULL DEFAULT 0,
    passed_holdout INTEGER,
    eval_ms REAL NOT NULL DEFAULT 0.0,
    outcome TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_fp ON experiences(problem_fingerprint);
CREATE INDEX IF NOT EXISTS idx_exp_fp_outcome ON experiences(problem_fingerprint, outcome);
"""

_OUTCOMES = ("baseline", "improvement", "success", "neutral", "error")

# Phase 1 (Self-Model, observation only): run-level self facts live in the
# SAME store file but in SEPARATE tables — never mixed with per-evaluation
# experience rows. Disabled via EVOLAB_SELF=0; broken tables degrade to
# silent no-op (same fail-safe contract as experiences).
_SELF_SCHEMA = """
CREATE TABLE IF NOT EXISTS self_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL DEFAULT '',
    seed INTEGER,
    generations INTEGER NOT NULL DEFAULT 0,
    population_size INTEGER NOT NULL DEFAULT 0,
    best_fitness REAL NOT NULL DEFAULT 0.0,
    mean_fitness REAL,
    evals_total INTEGER NOT NULL DEFAULT 0,
    early_stopped INTEGER,
    stagnation_gens INTEGER NOT NULL DEFAULT 0,
    diversity_final REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_self_runs_strategy ON self_runs(strategy);
CREATE TABLE IF NOT EXISTS self_capabilities (
    domain TEXT PRIMARY KEY,
    trials INTEGER NOT NULL DEFAULT 0,
    passes INTEGER NOT NULL DEFAULT 0,
    pass_rate REAL NOT NULL DEFAULT 0.0,
    ci_low REAL NOT NULL DEFAULT 0.0,
    ci_high REAL NOT NULL DEFAULT 1.0,
    updated_at TEXT NOT NULL
);
"""



from .priors import ExperienceMutationPrior, ExperienceSequencePrior

def problem_fingerprint(
    sources: dict[str, str],
    target_file: str,
    func_name: str = "",
) -> str:
    """Deterministic, formatting-independent fingerprint of the problem.

    Built from the structure-only AST dump (``include_attributes=False``) of
    the target file plus the function name — so whitespace, comments and line
    shifts do not change it, while any structural change (a different bug) does.
    Two authors writing the same buggy function with different formatting
    share a fingerprint; that is the point.
    """
    source = sources.get(target_file, next(iter(sources.values()), "")) if sources else ""
    try:
        dump = ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError:
        dump = hashlib.sha256(source.encode()).hexdigest()
    raw = f"{func_name}#{target_file}#{len(sources)}#{dump}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


class ExperienceStore:
    """Append-only SQLite store of evaluation experiences. Fail-safe: every
    write is best-effort; a broken store reports ``healthy == False`` and the
    run continues untouched."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        try:
            self._conn.executescript(_SELF_SCHEMA)
        except sqlite3.Error:
            pass
        self._conn.commit()
        self.healthy = True

    def record(self, row: dict[str, Any]) -> None:
        if not self.healthy:
            return
        try:
            self._conn.execute(
                """INSERT INTO experiences (
                       run_id, eval_index, problem_fingerprint, func_name,
                       target_file, genome_class, edit_kinds, edit_loci,
                       n_edits, score, fitness_delta, is_new_best,
                       passed_holdout, eval_ms, outcome, created_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["run_id"],
                    row["eval_index"],
                    row["problem_fingerprint"],
                    row.get("func_name", ""),
                    row.get("target_file", ""),
                    row.get("genome_class", ""),
                    json.dumps(row.get("edit_kinds", [])),
                    json.dumps(row.get("edit_loci", [])),
                    row.get("n_edits", 0),
                    float(row["score"]),
                    row.get("fitness_delta"),
                    int(row.get("is_new_best", 0)),
                    (None if row.get("passed_holdout") is None else int(row["passed_holdout"])),
                    float(row.get("eval_ms", 0.0)),
                    row["outcome"],
                    row.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            self._conn.commit()
        except sqlite3.Error:
            self.healthy = False

    def recall(self, fingerprint: str, k: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Exact-match retrieval: successful vs failed experiences for a problem.

        successful = holdout passed; failed = error outcome or holdout failed.
        Ordered by score (best evidence first). Cheap by design — Phase 2 adds
        slot generalization, not an LLM.
        """
        try:
            cur = self._conn.execute(
                """SELECT eval_index, edit_kinds, edit_loci, n_edits, score,
                          fitness_delta, is_new_best, passed_holdout, outcome
                   FROM experiences WHERE problem_fingerprint = ?
                   ORDER BY (passed_holdout IS NULL), passed_holdout DESC,
                            score DESC LIMIT ?""",
                (fingerprint, int(k)),
            )
            rows = cur.fetchall()
        except sqlite3.Error:
            return {"successful": [], "failed": []}
        out: dict[str, list[dict[str, Any]]] = {"successful": [], "failed": []}
        for r in rows:
            item = {
                "eval_index": r[0],
                "edit_kinds": json.loads(r[1]),
                "edit_loci": json.loads(r[2]),
                "n_edits": r[3],
                "score": r[4],
                "fitness_delta": r[5],
                "is_new_best": r[6],
                "passed_holdout": bool(r[7]) if r[7] is not None else None,
                "outcome": r[8],
            }
            bucket = "successful" if r[7] == 1 else "failed"
            out[bucket].append(item)
        return out

    def stats(self, fingerprint: str) -> dict[str, Any]:
        """Aggregate per-edit-kind success rates for one problem fingerprint.

        This is the primitive a memory-guided mutation policy consumes in
        Phase 2 (priors, never fitness).
        """
        try:
            cur = self._conn.execute(
                """SELECT edit_kinds, passed_holdout, outcome FROM experiences
                   WHERE problem_fingerprint = ?""",
                (fingerprint,),
            )
            rows = cur.fetchall()
        except sqlite3.Error:
            rows = []
        per_kind: dict[str, dict[str, float]] = {}
        for kinds_json, holdout, outcome in rows:
            passed = 1 if holdout == 1 else 0
            errored = 1 if outcome == "error" else 0
            for kind in json.loads(kinds_json) or ["<no_edits>"]:
                slot = per_kind.setdefault(kind, {"n": 0, "holdout_success": 0, "errors": 0})
                slot["n"] += 1
                slot["holdout_success"] += passed
                slot["errors"] += errored
        for slot in per_kind.values():
            slot["success_rate"] = round(slot["holdout_success"] / slot["n"], 4) if slot["n"] else 0.0
        return {
            "total_experiences": len(rows),
            "per_edit_kind": per_kind,
        }

    def sequence_stats(self, fingerprint: str) -> dict[str, Any]:
        """Prefix→next-kind transition counts for one problem fingerprint (M7).

        Every stored experience holds the genome's full ordered ``edit_kinds``
        list — the recipe order in which the program was assembled (one edit
        appended per mutation along the lineage). A stored sequence
        ``[k1..kn]`` is decomposed into the Markov path

            (∅)→k1, (k1)→k2, ..., (k1..kn-1)→kn

        and the row's final outcome (holdout pass) is credited to EVERY
        transition on that path — the standard sequence-model credit rule:
        a transition is "on a successful path" if the program it helped
        assemble passed holdout, and on a failed path otherwise.

        The transition key is ``"<prefix_joined_by_commas>>kind"`` (empty
        prefix for first edits). Edit-kind names contain no commas or ``>``
        (catalog kinds are identifiers), so the encoding is collision-free;
        this is asserted in the M7 tests. Consumers:
        ``ExperienceSequencePrior`` — the per-kind marginals in
        ``stats()`` stay the fallback for unseen prefixes.
        """
        try:
            cur = self._conn.execute(
                """SELECT edit_kinds, passed_holdout FROM experiences
                   WHERE problem_fingerprint = ?""",
                (fingerprint,),
            )
            rows = cur.fetchall()
        except sqlite3.Error:
            rows = []
        transitions: dict[str, dict[str, int]] = {}
        for kinds_json, holdout in rows:
            kinds = json.loads(kinds_json) or []
            passed = 1 if holdout == 1 else 0
            for i, kind in enumerate(kinds):
                prefix = ",".join(kinds[:i])
                key = f"{prefix}>{kind}"
                slot = transitions.setdefault(key, {"n": 0, "holdout_success": 0})
                slot["n"] += 1
                slot["holdout_success"] += passed
        return {
            "total_experiences": len(rows),
            "transitions": transitions,
        }

    def avoidance_set(
        self,
        fingerprint: str,
        min_failures: int = 2,
        max_entries: int = 256,
    ) -> set[tuple[str, int, int, str]] | None:
        """Dead-door mining for trap-aware initialization (M8).

        The genetic channel carries memory as *which genotypes exist*:
        a single-edit genotype that memory watched fail repeatedly — and
        never once succeed — is a dead door. Seeding the initial population
        elsewhere concentrates the search's real budget on live doors.

        Definition (registered in the M8 A/B protocol before measurement):
        a key ``(file, lineno, col_offset, kind)`` is avoided iff it has
        ≥ ``min_failures`` SINGLE-edit experiences for this fingerprint and
        ZERO holdout successes among them. Only single-edit rows qualify:
        for multi-edit rows the per-edit credit is unknowable, so they are
        never mined (conservative). A key that succeeded even once is
        never avoided, regardless of failures — one proof beats any count
        of failures.

        Returns ``None`` when the store has no single-edit data for the
        fingerprint (or on any sqlite error) — the caller must treat None
        as "no memory", i.e. exact legacy behavior. An empty set means the
        store was consulted and nothing qualified (behaviorally identical
        to None downstream, epistemically different).

        Payloads are not persisted (schema records skeletons by design),
        so the door granularity is ``(kind, locus)``: every payload variant
        at a dead (kind, locus) is avoided with it.
        """
        if min_failures < 1:
            raise ValueError("min_failures must be >= 1")
        try:
            cur = self._conn.execute(
                """SELECT edit_kinds, edit_loci, passed_holdout
                   FROM experiences
                   WHERE problem_fingerprint = ? AND n_edits = 1""",
                (fingerprint,),
            )
            rows = cur.fetchall()
        except sqlite3.Error:
            return None
        if not rows:
            return None
        attempts: dict[tuple[str, int, int, str], dict[str, int]] = {}
        for kinds_json, loci_json, holdout in rows:
            kinds = json.loads(kinds_json) or []
            loci = json.loads(loci_json) or []
            if len(kinds) != 1 or len(loci) != 1:
                continue  # defensive: n_edits=1 rows only, well-formed
            kind = kinds[0]
            locus = loci[0]
            if not isinstance(locus, list) or len(locus) != 3:
                continue
            key = (str(locus[0]), int(locus[1]), int(locus[2]), str(kind))
            slot = attempts.setdefault(key, {"n": 0, "success": 0})
            slot["n"] += 1
            if holdout == 1:
                slot["success"] += 1
        dead = [
            (key, slot["n"])
            for key, slot in attempts.items()
            if slot["success"] == 0 and slot["n"] >= min_failures
        ]
        dead.sort(key=lambda kv: (-kv[1], kv[0]))
        return {key for key, _ in dead[:max(0, int(max_entries))]}

    def composition_seeds(
        self,
        fingerprint: str,
        max_winners: int = 1,
    ) -> list[dict[str, Any]] | None:
        """Successful multi-edit composition mining (M9 — composition-seeded
        initialization).

        The positive genetic channel: a multi-edit genotype that memory
        watched PASS holdout is a remembered composition. Seeding parts of
        it into the initial population concentrates the search's budget
        near remembered-successful structure while leaving the search real
        work. By mining contract this method only returns compositions of
        ``n_edits >= 2``, and the caller (``make_code_population``) plants
        exactly ONE edit per seeded individual — a single-edit genotype,
        which the committed M8 firecheck proved is always a dead door on
        these benchmarks. Seeded individuals therefore cannot pass at
        generation zero: the mechanism is replay-proof by construction
        (full-composition seeding would be the cache/archive value class,
        outside the hypothesis space by the repo's registered position).

        Definition (registered in the M9 A/B protocol before measurement):
        a winner is a distinct SET of (kind, locus) pairs from rows with
        ``passed_holdout = 1`` AND ``n_edits >= 2`` for this fingerprint;
        winners are ranked by occurrence count (desc, then key order) and
        the top ``max_winners`` are returned with their first-occurrence
        edit order preserved.

        Returns ``None`` when the store has no successful multi-edit row
        for the fingerprint (zero signal — the caller must treat None as
        "no memory", i.e. exact legacy initialization) or on any sqlite
        error. Malformed rows are skipped defensively.
        """
        if max_winners < 1:
            raise ValueError("max_winners must be >= 1")
        try:
            cur = self._conn.execute(
                """SELECT edit_kinds, edit_loci FROM experiences
                   WHERE problem_fingerprint = ? AND passed_holdout = 1
                     AND n_edits >= 2
                   ORDER BY id""",
                (fingerprint,),
            )
            rows = cur.fetchall()
        except sqlite3.Error:
            return None
        if not rows:
            return None
        winners: dict[frozenset, dict[str, Any]] = {}
        order: list[frozenset] = []
        for kinds_json, loci_json in rows:
            try:
                kinds = json.loads(kinds_json) or []
                loci = json.loads(loci_json) or []
            except (TypeError, ValueError):
                continue
            if len(kinds) != len(loci) or not kinds:
                continue
            edits: list[tuple[str, str, int, int]] = []
            well_formed = True
            for kind, locus in zip(kinds, loci):
                if (
                    not isinstance(kind, str)
                    or not isinstance(locus, list)
                    or len(locus) != 3
                ):
                    well_formed = False
                    break
                edits.append(
                    (str(kind), str(locus[0]), int(locus[1]), int(locus[2]))
                )
            if not well_formed:
                continue
            key = frozenset(edits)
            slot = winners.get(key)
            if slot is None:
                winners[key] = {"count": 1, "edits": edits}
                order.append(key)
            else:
                slot["count"] += 1
        if not winners:
            return None
        ranked = sorted(order, key=lambda k: (-winners[k]["count"], sorted(k)))
        return [winners[k] for k in ranked[: max(1, int(max_winners))]]

    def run_metrics(self, run_id: str) -> dict[str, Any]:
        """Evaluation-level metrics for one recorded run (Phase 3 A/B).

        ``first_success_eval`` is the 1-based eval_index of the first
        holdout-passing evaluation, or ``None`` when the run never passed
        holdout (a censored run — consumers count it at ``evals_total``).
        """
        try:
            row = self._conn.execute(
                """SELECT COUNT(*), MAX(eval_index) FROM experiences
                   WHERE run_id = ?""",
                (run_id,),
            ).fetchone()
            succ = self._conn.execute(
                """SELECT MIN(eval_index), MAX(score) FROM experiences
                   WHERE run_id = ? AND passed_holdout = 1""",
                (run_id,),
            ).fetchone()
        except sqlite3.Error:
            return {
                "evals_total": 0,
                "first_success_eval": None,
                "first_success_score": None,
            }
        return {
            "evals_total": int(row[1]) if row and row[1] is not None else 0,
            "first_success_eval": (
                int(succ[0]) if succ and succ[0] is not None else None
            ),
            "first_success_score": (
                float(succ[1]) if succ and succ[1] is not None else None
            ),
        }

    def family_summaries(self) -> list[dict[str, Any]]:
        """One deterministic summary per problem family (Phase 3).

        A family is every experience recorded for the same structural
        problem fingerprint — same bug, any formatting, any run. Facts only:
        counts, kind-level success rates and the kind-multiset of the best
        holdout-passing evaluation. Never a replayable patch (the
        "facts, not answers" contract). Rows are ordered by
        ``(-n_experiences, fingerprint)`` so the output is reproducible.
        """
        try:
            cur = self._conn.execute(
                """SELECT problem_fingerprint, func_name, target_file,
                          COUNT(*), COUNT(DISTINCT run_id),
                          MIN(created_at), MAX(created_at),
                          SUM(passed_holdout = 1),
                          MAX(CASE WHEN passed_holdout = 1 THEN score END)
                   FROM experiences
                   GROUP BY problem_fingerprint
                   ORDER BY COUNT(*) DESC, problem_fingerprint ASC"""
            )
            rows = cur.fetchall()
            outcome_rows = self._conn.execute(
                """SELECT problem_fingerprint, outcome, COUNT(*)
                   FROM experiences GROUP BY problem_fingerprint, outcome"""
            ).fetchall()
        except sqlite3.Error:
            return []
        outcomes_by_fp: dict[str, dict[str, int]] = {}
        for fp, outcome, n in outcome_rows:
            outcomes_by_fp.setdefault(fp, {})[outcome] = int(n)
        out: list[dict[str, Any]] = []
        for (fp, func, tgt, n, runs, first, last, ok, best) in rows:
            best_kinds: list[str] | None = None
            try:
                brow = self._conn.execute(
                    """SELECT edit_kinds FROM experiences
                       WHERE problem_fingerprint = ? AND passed_holdout = 1
                       ORDER BY score DESC, eval_index ASC LIMIT 1""",
                    (fp,),
                ).fetchone()
                if brow is not None:
                    best_kinds = json.loads(brow[0]) or None
            except sqlite3.Error:
                best_kinds = None
            out.append(
                {
                    "problem_fingerprint": fp,
                    "function_key": f"{func}@{tgt}",
                    "n_experiences": int(n),
                    "n_runs": int(runs),
                    "first_seen": first,
                    "last_seen": last,
                    "outcomes": outcomes_by_fp.get(fp, {}),
                    "holdout_successes": int(ok or 0),
                    "best_holdout_score": (float(best) if best is not None else None),
                    "best_success_kinds": best_kinds,
                    "per_edit_kind": self.stats(fp).get("per_edit_kind", {}),
                }
            )
        return out

    def close(self) -> None:
        try:
            self._conn.close()
        except (sqlite3.Error, AttributeError):
            pass

    # ---- Phase 1: Self-Model (run-level self facts, observation only) ----

    def record_self_run(self, row: dict[str, Any]) -> None:
        """Append one run-level self fact. Best-effort; never raises."""
        if not self.healthy or not _self_enabled():
            return
        try:
            self._conn.execute(
                """INSERT INTO self_runs (
                       run_id, strategy, seed, generations, population_size,
                       best_fitness, mean_fitness, evals_total, early_stopped,
                       stagnation_gens, diversity_final, created_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(row.get("run_id", uuid.uuid4().hex[:12])),
                    str(row.get("strategy", "")),
                    row.get("seed"),
                    int(row.get("generations", 0)),
                    int(row.get("population_size", 0)),
                    float(row.get("best_fitness", 0.0)),
                    (None if row.get("mean_fitness") is None else float(row["mean_fitness"])),
                    int(row.get("evals_total", 0)),
                    (None if row.get("early_stopped") is None else int(bool(row["early_stopped"]))),
                    int(row.get("stagnation_gens", 0)),
                    (None if row.get("diversity_final") is None else float(row["diversity_final"])),
                    row.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            self._conn.commit()
        except (sqlite3.Error, ValueError, TypeError):
            pass

    def fetch_self_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Latest self-run facts, newest first. Empty on any error."""
        try:
            cur = self._conn.execute(
                """SELECT run_id, strategy, seed, generations, population_size,
                          best_fitness, mean_fitness, evals_total, early_stopped,
                          stagnation_gens, diversity_final, created_at
                   FROM self_runs ORDER BY id DESC LIMIT ?""",
                (max(int(limit), 1),),
            )
            rows = cur.fetchall()
        except (sqlite3.Error, ValueError):
            return []
        out = []
        for r in rows:
            out.append({
                "run_id": r[0], "strategy": r[1], "seed": r[2],
                "generations": r[3], "population_size": r[4],
                "best_fitness": r[5], "mean_fitness": r[6],
                "evals_total": r[7], "early_stopped": r[8],
                "stagnation_gens": r[9], "diversity_final": r[10],
                "created_at": r[11],
            })
        return out

    def record_capability(self, domain: str, passed: bool) -> dict[str, Any] | None:
        """Update one domain capability trial. Returns the calibrated row."""
        if not self.healthy or not _self_enabled():
            return None
        try:
            cur = self._conn.execute(
                "SELECT trials, passes FROM self_capabilities WHERE domain = ?",
                (str(domain),),
            ).fetchone()
            trials, passes = (cur if cur else (0, 0))
            trials, passes = int(trials) + 1, int(passes) + (1 if passed else 0)
            rate = round(passes / trials, 4)
            lo, hi = wilson_interval(passes, trials)
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._conn.execute(
                """INSERT INTO self_capabilities
                   (domain, trials, passes, pass_rate, ci_low, ci_high, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(domain) DO UPDATE SET
                     trials=excluded.trials, passes=excluded.passes,
                     pass_rate=excluded.pass_rate, ci_low=excluded.ci_low,
                     ci_high=excluded.ci_high, updated_at=excluded.updated_at""",
                (str(domain), trials, passes, rate, lo, hi, now),
            )
            self._conn.commit()
            return {"domain": str(domain), "trials": trials, "passes": passes,
                    "pass_rate": rate, "ci_low": lo, "ci_high": hi}
        except (sqlite3.Error, ValueError, TypeError):
            return None

    def self_summary(self) -> dict[str, Any]:
        """Read-only Self-Model snapshot: capabilities + recent runs."""
        try:
            caps = self._conn.execute(
                "SELECT domain, trials, passes, pass_rate, ci_low, ci_high "
                "FROM self_capabilities ORDER BY domain ASC"
            ).fetchall()
        except sqlite3.Error:
            caps = []
        return {
            "capabilities": [
                {"domain": c[0], "trials": c[1], "passes": c[2],
                 "pass_rate": c[3], "ci_low": c[4], "ci_high": c[5]}
                for c in caps
            ],
            "recent_runs": self.fetch_self_runs(limit=20),
            "note": "observation only - no search decision reads this in Phase 1",
        }

    def __enter__(self) -> ExperienceStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def function_summaries(
    family_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge per-fingerprint family summaries into per-function views.

    The same function with different bug variants shares a ``function_key``
    (``func@file``) but has different fingerprints. Merging is additive over
    exact counts; run_ids cannot overlap fingerprints (one recorder binds one
    fingerprint per run), so summed ``n_runs`` stays double-count-free.
    Deterministic: sorted by ``(-n_experiences, function_key)``.
    """
    merged: dict[str, dict[str, Any]] = {}
    for s in family_rows:
        key = s["function_key"]
        slot = merged.setdefault(
            key,
            {
                "function_key": key,
                "n_fingerprints": 0,
                "n_experiences": 0,
                "n_runs": 0,
                "holdout_successes": 0,
                "per_edit_kind": {},
            },
        )
        slot["n_fingerprints"] += 1
        slot["n_experiences"] += s["n_experiences"]
        slot["n_runs"] += s["n_runs"]
        slot["holdout_successes"] += s["holdout_successes"]
        for kind, st in s["per_edit_kind"].items():
            agg = slot["per_edit_kind"].setdefault(
                kind, {"n": 0, "holdout_success": 0, "errors": 0}
            )
            agg["n"] += st["n"]
            agg["holdout_success"] += st["holdout_success"]
            agg["errors"] += st["errors"]
    for slot in merged.values():
        for agg in slot["per_edit_kind"].values():
            agg["success_rate"] = (
                round(agg["holdout_success"] / agg["n"], 4) if agg["n"] else 0.0
            )
    return sorted(merged.values(), key=lambda s: (-s["n_experiences"], s["function_key"]))


def render_family_report(family_rows: list[dict[str, Any]]) -> str:
    """Deterministic plain-text rendering of ``family_summaries`` output.

    Suitable for reports and README evidence blocks; identical input always
    renders byte-identical output (no timestamps, no wall-clock, no dict
    ordering surprises — kinds are sorted).
    """
    if not family_rows:
        return ""
    lines = [
        "fp          function                     exps  runs   ok  per-kind (n@success_rate)",
        "-" * 96,
    ]
    for s in family_rows:
        kinds = ", ".join(
            f"{k}:{v['n']}@{v['success_rate']:.2f}"
            for k, v in sorted(s["per_edit_kind"].items())
        ) or "-"
        lines.append(
            f"{s['problem_fingerprint'][:10]:10s} {s['function_key'][:26]:26s}"
            f" {s['n_experiences']:>5d} {s['n_runs']:>4d} {s['holdout_successes']:>4d}  {kinds}"
        )
    return "\n".join(lines)




class ExperienceRecorderProxy:
    """Engine-facing evaluator wrapper that records every evaluation.

    Mirrors the electronics ``ArchivedEvaluatorProxy`` contract: full
    Evaluator surface, ``__call__`` used by EvolutionEngine, ``evaluate``
    returning the untouched ``FitnessResult``, all other attributes delegated
    to the wrapped evaluator. Recording happens AFTER the result exists and
    can never alter it.
    """

    def __init__(
        self,
        raw: Any,
        store: ExperienceStore,
        fingerprint: str,
        func_name: str = "",
        target_file: str = "",
        run_id: str | None = None,
        *,
        prior_enabled: bool = True,
        prior_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.raw = raw
        self.store = store
        self.fingerprint = fingerprint
        self.func_name = func_name
        self.target_file = target_file
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.prior_enabled = bool(prior_enabled)
        self.prior_kwargs = dict(prior_kwargs or {})
        self._eval_index = 0
        self._best_score: float | None = None
        self._prior: ExperienceMutationPrior | None = None

    @property
    def mutation_prior(self) -> ExperienceMutationPrior | None:
        """Phase 2 hook: the engine reads this off its ``fitness_fn`` (only
        present when the recorder is attached) and hands it to genome
        mutation as a soft prior. Disable the recorder (``EVOLAB_EXPERIENCE=0``)
        and this object simply never reaches the engine. A/B control arms set
        ``prior_enabled=False`` so recording continues but the prior reads as
        ``None`` — the exact pre-memory behavior with identical
        instrumentation on both arms.

        M7: ``prior_kwargs`` may carry ``mode`` — ``"kind"`` (default, the
        context-free ``ExperienceMutationPrior``) or ``"sequence"`` (the
        prefix-conditioned ``ExperienceSequencePrior``). The default is
        unchanged by M7: an explicit ``mode`` key is required to select the
        sequence prior."""
        if not self.prior_enabled:
            return None
        if self._prior is None:
            kwargs = dict(self.prior_kwargs)
            mode = str(kwargs.pop("mode", "kind"))
            if mode == "sequence":
                self._prior = ExperienceSequencePrior(
                    self.store, self.fingerprint, **kwargs
                )
            elif mode == "kind":
                self._prior = ExperienceMutationPrior(
                    self.store, self.fingerprint, **kwargs
                )
            else:
                raise ValueError(
                    f"unknown prior mode {mode!r} (expected 'kind' or 'sequence')"
                )
        return self._prior

    def _classify(self, score: float, holdout: Any, error: bool) -> tuple[str, float, bool]:
        if self._best_score is None:
            return ("error" if error else "baseline"), 0.0, False
        delta = score - self._best_score
        if error:
            return "error", delta, False
        if holdout is True:
            return "success", delta, delta > 0
        if delta > 1e-9:
            return "improvement", delta, True
        return "neutral", delta, False

    def _record(self, genome: Any, res: Any) -> None:
        self._eval_index += 1
        score = float(getattr(res, "score", 0.0))
        holdout = getattr(res, "passed_holdout", None)
        artifacts = getattr(res, "artifacts", {}) or {}
        error = score <= 0.0 or bool(artifacts.get("error"))
        outcome, delta, new_best = self._classify(score, holdout, error)
        if self._best_score is None or score > self._best_score:
            self._best_score = score
        edits = getattr(genome, "edits", None)
        if edits is not None:
            kinds = [e.kind for e in edits]
            loci = [[e.file, e.lineno, e.col_offset] for e in edits]
            genome_class = type(genome).__name__
        else:
            kinds, loci, genome_class = [], [], type(genome).__name__
        self.store.record(
            {
                "run_id": self.run_id,
                "eval_index": self._eval_index,
                "problem_fingerprint": self.fingerprint,
                "func_name": self.func_name,
                "target_file": self.target_file,
                "genome_class": genome_class,
                "edit_kinds": kinds,
                "edit_loci": loci,
                "n_edits": len(kinds),
                "score": score,
                "fitness_delta": delta,
                "is_new_best": new_best,
                "passed_holdout": holdout,
                "eval_ms": float(getattr(res, "evaluation_time_ms", 0.0)),
                "outcome": outcome,
            }
        )

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> Any:
        res = self.raw.evaluate(target, context) if context is not None else self.raw.evaluate(target)
        try:
            genome = target.genome if hasattr(target, "genome") else target
            self._record(genome, res)
        except Exception:
            pass
        return res

    def __call__(self, individual: Any) -> float:
        return float(self.evaluate(individual).score)

    @property
    def deterministic(self) -> bool:
        return getattr(self.raw, "deterministic", True)

    @property
    def cost_estimate(self) -> str:
        return getattr(self.raw, "cost_estimate", "cheap")

    def close(self) -> None:
        if hasattr(self.store, "close"):
            self.store.close()
        if hasattr(self.raw, "close"):
            self.raw.close()

    def __enter__(self) -> ExperienceRecorderProxy:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.raw, attr)


def attach_experience_recorder(
    evaluator: Any,
    sources: dict[str, str],
    target_file: str,
    func_name: str = "",
    *,
    db_path: str | Path | None = None,
    run_id: str | None = None,
    prior_enabled: bool | None = None,
    prior_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Wires ``evaluator`` into the cross-run experience store. Fail-safe by
    design: on any wiring error the raw evaluator is returned unchanged.

    Environment knobs (mirroring ``experimental/electronics/archive.py``):
      - ``EVOLAB_EXPERIENCE=0``        disable (returns the raw evaluator)
      - ``EVOLAB_EXPERIENCE_DB``       alternate sqlite path
      - ``EVOLAB_EXPERIENCE_PRIOR=1``  opt in to the mutation prior

    Phase-3 verdict (reports/ab_memory_value.json): the pre-registered A/B
    measured search_efficiency_gain = -14.6% pooled over 40 paired runs —
    the prior did NOT prove its value, so by its own decision rule it is
    NOT default-on. Recording and family summaries (the facts layer) stay
    on: they are pure observation. Pass ``prior_enabled=True/False`` to
    override the environment explicitly (the A/B harness does exactly that);
    ``None`` (default) defers to ``EVOLAB_EXPERIENCE_PRIOR`` (default off).

    ``prior_kwargs`` passes construction options to ``ExperienceMutationPrior``
    (e.g. a frozen ``cache_ttl`` snapshot).

    Default path: ``./data/experience.db`` (created lazily relative to cwd).
    """
    flag = os.environ.get("EVOLAB_EXPERIENCE", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return evaluator
    if prior_enabled is None:
        prior_flag = os.environ.get("EVOLAB_EXPERIENCE_PRIOR", "0").strip().lower()
        prior_enabled = prior_flag in ("1", "true", "yes", "on")
    try:
        fp = problem_fingerprint(sources, target_file, func_name)
        path = (
            db_path
            or os.environ.get("EVOLAB_EXPERIENCE_DB")
            or Path.cwd() / "data" / "experience.db"
        )
        store = ExperienceStore(path)
        return ExperienceRecorderProxy(
            evaluator,
            store,
            fp,
            func_name=func_name,
            target_file=target_file,
            run_id=run_id,
            prior_enabled=prior_enabled,
            prior_kwargs=prior_kwargs,
        )
    except Exception:
        return evaluator




# ---------------------------------------------------------------------------
# Re-exports for 100% backward compatibility
# ---------------------------------------------------------------------------
from .eval_cache import EvaluationCache, attach_eval_cache
from .priors import ExperienceMutationPrior, ExperienceSequencePrior
from .self_model import (
    META_CONTROLLER_FRACTION,
    META_CONTROLLER_MODES,
    META_DIVERSITY_LOW,
    META_OVERFIT_GAP,
    META_PLATEAU_EPS,
    META_PLATEAU_GENS,
    META_STD_LOW,
    _meta_enabled,
    _meta_resolve_mode,
    _self_enabled,
    attach_self_assessment,
    diagnose_run,
    govern_modification,
    meta_control_step,
    record_engine_self_run,
    render_self_modification_proposal,
    self_critic_report,
    wilson_interval,
)

__all__ = [
    # Core Experience Store
    "ExperienceStore",
    "ExperienceRecorderProxy",
    "attach_experience_recorder",
    "problem_fingerprint",
    "function_summaries",
    "render_family_report",
    # Evaluation Cache
    "EvaluationCache",
    "attach_eval_cache",
    # Priors
    "ExperienceMutationPrior",
    "ExperienceSequencePrior",
    # Self Model
    "wilson_interval",
    "_self_enabled",
    "diagnose_run",
    "self_critic_report",
    "attach_self_assessment",
    "_meta_enabled",
    "_meta_resolve_mode",
    "meta_control_step",
    "govern_modification",
    "render_self_modification_proposal",
    "record_engine_self_run",
    "META_DIVERSITY_LOW",
    "META_PLATEAU_GENS",
    "META_PLATEAU_EPS",
    "META_STD_LOW",
    "META_OVERFIT_GAP",
    "META_CONTROLLER_FRACTION",
    "META_CONTROLLER_MODES",
]
