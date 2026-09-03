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


class ExperienceMutationPrior:
    """Memory-guided mutation prior (Phase 2): turns per-edit-kind success
    rates from one problem fingerprint into soft sampling weights.

    Contracts (mirroring the Phase 1 rules):
    - Priors guide, never force: each weight is blended with the uniform
      baseline (``1 - strength + strength * rate``), so the maximum bias
      ratio is bounded at ``1 / (1 - strength)`` and no kind's weight can
      reach zero.
    - No usable data -> ``kind_weights`` returns ``None`` and the caller
      keeps its existing behavior untouched (null-intervention).
    - Zero-signal data -> ``None`` as well (memory-hygiene M6): when the
      stored rates fail to differentiate the candidates, the weight vector
      the prior would emit is nearly uniform — reweighting with it changes
      nothing about the choice distribution yet still perturbs the RNG
      stream (the exact mechanism behind v2's near-zero-bias/no-benefit
      readings and v1's scary artifact). The prior therefore suppresses
      itself: bias within ``ZERO_SIGNAL_MAX_FRACTION`` of the maximum bias
      it could express at its configured strength reads as "nothing to
      say" and collapses to the null-intervention. See ``_near_uniform``.
    - Read-only observation: consumes ``ExperienceStore.stats(fp)`` and
      never writes, never touches fitness, holdout or selection decisions.
    - Kinds below ``min_support`` observations are treated as neutral
      (rate 0.5) until enough evidence accumulates — and if EVERY
      candidate ends up neutral the output is uniform, which the same M6
      gate collapses to ``None``.
    """

    #: M6 — a prior whose expressed bias is at most this fraction of the
    #: maximum bias it could express at its strength is treated as
    #: zero-signal and returns ``None``. 0.10 is registered from the v2
    #: instrument's measured zero-signal envelope: across all four repair
    #: scenarios the real stores produced weight ratios <= 1.10:1 at the
    #: default strength 0.5, i.e. <= 10% of the maximal blend bias, with
    #: zero measured search value (reports/ab_memory_value_v2.json). The
    #: ratio is normalized by the strength-dependent maximum so the gate is
    #: strength-invariant. Changing this constant is a design decision that
    #: requires its own registered measurement — not a tuning knob.
    ZERO_SIGNAL_MAX_FRACTION: float = 0.10

    def __init__(
        self,
        store: ExperienceStore,
        fingerprint: str,
        *,
        min_support: int = 3,
        strength: float = 0.5,
        alpha: float = 1.0,
        cache_ttl: float = 1.0,
        zero_signal_gate: bool = True,
    ) -> None:
        self.store = store
        self.fingerprint = fingerprint
        self.min_support = int(min_support)
        self.strength = min(1.0, max(0.0, float(strength)))
        self.alpha = float(alpha)
        self.cache_ttl = float(cache_ttl)
        # M6 isolation knob (mirrors the A/B isolation-arm pattern): False
        # reproduces the exact pre-M6 behavior (near-uniform weights are
        # emitted instead of collapsing to None). Instrumentation only —
        # the default is armed.
        self.zero_signal_gate = bool(zero_signal_gate)
        self._cache: dict[str, Any] | None = None
        self._cache_ts = 0.0

    def _stats(self) -> dict[str, Any]:
        import time as _time
        now = _time.monotonic()
        if self._cache is None or (now - self._cache_ts) >= self.cache_ttl:
            try:
                self._cache = self.store.stats(self.fingerprint)
            except Exception:
                self._cache = {"total_experiences": 0, "per_edit_kind": {}}
            self._cache_ts = now
        return self._cache

    def _near_uniform(self, weights: dict[str, float]) -> bool:
        """M6 zero-signal detector: is this weight vector within noise of
        uniform, relative to what this prior COULD have expressed?

        The blend ``w = (1 - strength) + strength * rate`` bounds the
        expressible weight ratio at ``1 / (1 - strength)`` (this class's own
        docstring contract). We measure how much of that expressive range
        the actual weights use: ``(ratio - 1) / (max_ratio - 1)``. At the
        default strength 0.5 the gate fires at ratio <= 1.10 — exactly the
        envelope v2 measured on real stores that yielded zero search value
        but full RNG-stream divergence. Normalizing by the strength-dependent
        maximum makes the threshold strength-invariant: the same zero-signal
        data gates at strength 0.9 just as it does at 0.5.

        Fail-open by design: non-positive weights (reachable only with
        ``alpha=0`` + ``strength=1``, i.e. a prior actively forbidding a
        kind) are a strong opinion and are never suppressed.
        """
        if self.strength <= 0.0:
            return True  # a zero-strength blend IS the uniform baseline
        vals = list(weights.values())
        if not vals or any(v <= 0.0 for v in vals):
            return False  # forbidding is information — never gate it
        ratio = max(vals) / min(vals)
        max_ratio = 1.0 / (1.0 - self.strength)
        fraction = (ratio - 1.0) / (max_ratio - 1.0)
        return fraction <= self.ZERO_SIGNAL_MAX_FRACTION

    def kind_weights(
        self, kinds: list[str], prefix: list[str] | None = None
    ) -> dict[str, float] | None:
        """Soft sampling weights for the candidate edit kinds.

        ``prefix`` is part of the M7 sequence contract: the caller (``Repair
        Genome.mutate``) passes the edit kinds already applied to the parent
        genome. The per-kind prior is context-free by design and IGNORES the
        prefix — ``ExperienceSequencePrior`` (subclass below) is the
        consumer. The parameter exists on the base class so the mutation
        path can call both priors uniformly.

        Returns ``None`` when the store holds no experiences for this
        fingerprint (the caller must then keep its existing behavior) OR
        when the data carries no differential signal (M6 gate: near-uniform
        weights would only perturb the RNG stream, never the choice).
        Weights use a Laplace-smoothed holdout success rate; error outcomes
        need no separate term because an errored evaluation never passes
        holdout and therefore already lowers the rate.
        """
        stats = self._stats()
        per = stats.get("per_edit_kind") or {}
        if not stats.get("total_experiences") or not per:
            return None
        weights: dict[str, float] = {}
        for kind in set(kinds):
            slot = per.get(kind)
            if slot and int(slot.get("n", 0)) >= self.min_support:
                rate = (slot["holdout_success"] + self.alpha) / (slot["n"] + 2 * self.alpha)
            else:
                rate = 0.5  # neutral until the kind has enough observations
            rate = min(1.0, max(0.0, rate))
            weights[kind] = (1.0 - self.strength) + self.strength * rate
        if self.zero_signal_gate and self._near_uniform(weights):
            return None
        return weights

    def summarize(self) -> dict[str, Any]:
        """Honest introspection for reports: current prior state, no claims."""
        stats = self._stats()
        return {
            "fingerprint": self.fingerprint,
            "total_experiences": stats.get("total_experiences", 0),
            "min_support": self.min_support,
            "strength": self.strength,
            "zero_signal_max_fraction": self.ZERO_SIGNAL_MAX_FRACTION,
            "zero_signal_gate": self.zero_signal_gate,
        }


class ExperienceSequencePrior(ExperienceMutationPrior):
    """Sequence-aware mutation prior (memory-hygiene M7).

    Why: the per-kind prior rates each edit kind in isolation, but a program
    is assembled as an ORDERED recipe — the lru_cache A/B evidence (v1) and
    the family summaries both show successes that are combinations of kinds
    (``edit_kinds`` ordered lists are stored yet nothing consumed them as
    sequences). This prior conditions on the parent's current edit-kind
    PREFIX: "after [A, B], which next kind sat on successful paths?"

    Math (identical contracts to the base class, applied per TRANSITION
    instead of per kind):
    - ``kind_weights(kinds, prefix)`` looks up transitions
      ``prefix→kind`` from ``ExperienceStore.sequence_stats``; the rate is
      Laplace-smoothed ``(s + alpha) / (n + 2 * alpha)`` and blended with
      the uniform baseline ``(1 - strength) + strength * rate`` — bounded
      bias, never zero, ``min_support`` keeps rare transitions neutral.
    - EMPTY parent prefix (first edit of a program) reads the ``∅→kind``
      transitions — the per-kind FIRST-EDIT rates.
    - UNSEEN prefix (no stored sequence ever started this way): graceful
      degradation to the base per-kind marginals — the mechanism is a
      strict superset of ``ExperienceMutationPrior``, never more ignorant
      than it.
    - Empty / broken store -> ``None`` -> the caller keeps its exact
      existing behavior (null-intervention, same as the base contract).
    - Zero-signal transitions -> ``None`` too (M6 gate, inherited via the
      shared ``_near_uniform`` check): transitions whose rates fail to
      separate the candidates collapse to the null-intervention instead of
      emitting near-uniform weights that only shuffle the RNG stream.
    - Read-only: consumes ``sequence_stats``; never writes, never touches
      fitness, holdout or selection decisions.

    Selection at A/B time (M7): arms must compare against BOTH the base
    prior (head-to-head: does conditioning add value over marginals?) and
    control (absolute value), under a pre-registered protocol on the
    hardened v2 instrument. No default activation without proven gain —
    the standing memory-hygiene gate.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._trans_cache: dict[str, Any] | None = None
        self._trans_ts = 0.0

    def _transitions(self) -> dict[str, Any]:
        """Cached sequence_stats fetch — SEPARATE from the base ``_stats``.

        The base ``_stats`` keeps serving the per-kind marginal shape that
        ``ExperienceMutationPrior.kind_weights`` consumes, so the unseen-
        prefix fallback (``super().kind_weights``) reads true marginals,
        not the transition table.
        """
        import time as _time
        now = _time.monotonic()
        if self._trans_cache is None or (now - self._trans_ts) >= self.cache_ttl:
            try:
                self._trans_cache = self.store.sequence_stats(self.fingerprint)
            except Exception:
                self._trans_cache = {"total_experiences": 0, "transitions": {}}
            self._trans_ts = now
        return self._trans_cache

    @staticmethod
    def _prefix_key(prefix: list[str] | None) -> str:
        return ",".join(prefix or [])

    def kind_weights(
        self, kinds: list[str], prefix: list[str] | None = None
    ) -> dict[str, float] | None:
        stats = self._transitions()
        trans = stats.get("transitions") or {}
        if not stats.get("total_experiences") or not trans:
            return None
        pkey = self._prefix_key(prefix)
        # total support behind this prefix, over ALL next-kinds (not just
        # the candidates) — the unseen-prefix detector
        support = sum(s["n"] for k, s in trans.items() if k.rsplit(">", 1)[0] == pkey)
        if support == 0:
            if pkey:
                # unseen prefix: degrade to the per-kind marginals — never
                # more ignorant than the base prior
                return super().kind_weights(kinds)
            # no first-edit transitions at all (all stored rows had no
            # edits) -> nothing usable
            return None
        weights: dict[str, float] = {}
        for kind in set(kinds):
            slot = trans.get(f"{pkey}>{kind}")
            if slot and int(slot["n"]) >= self.min_support:
                rate = (slot["holdout_success"] + self.alpha) / (slot["n"] + 2 * self.alpha)
            else:
                rate = 0.5  # neutral until the transition has enough evidence
            rate = min(1.0, max(0.0, rate))
            weights[kind] = (1.0 - self.strength) + self.strength * rate
        if self._near_uniform(weights):
            return None
        return weights

    def summarize(self) -> dict[str, Any]:
        info = super().summarize()
        info["mode"] = "sequence"
        info["transitions"] = len((self._transitions() or {}).get("transitions") or {})
        return info


class EvaluationCache:
    """Program-keyed memoization of raw evaluator results (M3).

    The duplicate-evaluation probe (reports/duplicate_evals_probe.json)
    measured 88.2% of evaluator calls re-testing programs already evaluated
    within the same run — the engine re-evaluates the whole population every
    generation and has no recall of what it already computed. This cache is
    the memory that recall: identical programs get the identical result back
    without invoking the raw evaluator again.

    Contracts (mirroring the module's non-negotiables):

    - Behavior-transparency: a cache hit must be indistinguishable from a
      miss to every consumer. Scores, sub_scores, passed_holdout and
      artifacts are rebuilt field-for-field (only ``evaluation_time_ms``
      drops to 0.0 — the honest signal that no raw work happened). RNG
      streams are never touched, so search trajectories stay byte-identical.
    - Side-effect replay: evaluators may mutate observable state during
      ``evaluate`` (e.g. ``FunctionTestEvaluator.last_suspicion_map``, which
      the SBFL-narrowed mutator later reads). A naive cache would leave a
      STALE map after a hit and silently change search behavior. Evaluators
      therefore declare ``cacheable_state_attrs``; the cache snapshots those
      attributes on a miss and restores them on the hit. Default: no
      attributes, no replay.
    - Correctness before hit-rate: the cache key is the genome's full
      evaluation-relevant state — the materialized applied sources for
      edit-genomes (canonical JSON of ``apply_to()``; CK swap — see
      ``program_identity``), materialized sources for source-genomes.
      ``None`` (unsupported genome) means bypass: raw call, never a wrong
      answer.
    - Deterministic evaluators only: ``attach_eval_cache`` refuses to wrap
      an evaluator that does not declare ``deterministic == True``.
    - Fail-safe: any cache error falls through to the raw evaluator; the
      cache never breaks a run. ``EVOLAB_EVAL_CACHE=0`` disables.
    """

    def __init__(self, raw: Any, *, max_entries: int = 4096) -> None:
        self.raw = raw
        self.max_entries = int(max_entries)
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.enabled = os.environ.get("EVOLAB_EVAL_CACHE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._cache: dict[str, tuple[Any, dict[str, Any]]] = {}

    # -- program identity -------------------------------------------------

    @staticmethod
    def program_identity(genome: Any) -> str | None:
        """Hash of the genome's full evaluation-relevant state.

        Edit-genomes (``edits`` + ``apply_to``): canonical JSON of the
        APPLIED SOURCES (``apply_to()`` — the full materialized state the
        evaluator actually tests, all files). CK swap (registered protocol
        scripts/ab_cache_key_swap.py, rules C1-C4): the historical key was
        the ordered edit recipe (kinds+loci+payloads) + base hash — a
        safe-false identity that never merges different recipes producing
        the SAME program (measured cost: click 438 recipe-keys -> 77
        distinct programs, realized savings 54.8% vs 72.8% within-run
        ceiling). Recipes reaching the same materialized state now merge
        (same program => same result by evaluator determinism — enforced
        by ``attach_eval_cache``); different materialized states never do.
        Source-genomes (``to_sources``): canonical JSON of all files.
        Plain-string genomes: the string itself. ``None`` = unsupported.
        """
        try:
            edits = getattr(genome, "edits", None)
            if edits is not None and hasattr(genome, "apply_to"):
                applied = genome.apply_to()
                blob = json.dumps(
                    {k: applied[k] for k in sorted(applied)}, sort_keys=True
                )
                return hashlib.sha256(blob.encode()).hexdigest()
            if hasattr(genome, "to_sources"):
                srcs = genome.to_sources()
                blob = json.dumps({k: srcs[k] for k in sorted(srcs)}, sort_keys=True)
                return hashlib.sha256(blob.encode()).hexdigest()
            if hasattr(genome, "to_code"):
                return hashlib.sha256(genome.to_code().encode()).hexdigest()
            if isinstance(genome, str):
                return hashlib.sha256(genome.encode()).hexdigest()
        except Exception:
            return None
        return None

    # -- side-effect state -------------------------------------------------

    def _state_attrs(self) -> tuple[str, ...]:
        return tuple(getattr(self.raw, "cacheable_state_attrs", ()) or ())

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            attr: copy.deepcopy(getattr(self.raw, attr))
            for attr in self._state_attrs()
            if hasattr(self.raw, attr)
        }

    def _restore_state(self, state: dict[str, Any]) -> None:
        for attr, value in state.items():
            try:
                setattr(self.raw, attr, copy.deepcopy(value))
            except Exception:
                pass

    # -- result rebuild ----------------------------------------------------

    @staticmethod
    def _fresh(result: Any) -> Any:
        """Field-for-field copy of a cached result, de-aliased, honest timing."""
        try:
            from .evaluators import FitnessResult

            if isinstance(result, FitnessResult):
                return FitnessResult(
                    score=result.score,
                    sub_scores=dict(result.sub_scores),
                    passed_holdout=result.passed_holdout,
                    artifacts=copy.deepcopy(result.artifacts),
                    evaluation_time_ms=0.0,
                )
        except Exception:
            pass
        return result

    # -- core ---------------------------------------------------------------

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> Any:
        if not self.enabled:
            return (
                self.raw.evaluate(target, context)
                if context is not None
                else self.raw.evaluate(target)
            )
        try:
            genome = target.genome if hasattr(target, "genome") else target
            key = self.program_identity(genome)
        except Exception:
            key = None
        if key is None:
            self.bypasses += 1
            return (
                self.raw.evaluate(target, context)
                if context is not None
                else self.raw.evaluate(target)
            )
        if key in self._cache:
            self.hits += 1
            result, state = self._cache[key]
            self._restore_state(state)
            return self._fresh(result)
        self.misses += 1
        res = (
            self.raw.evaluate(target, context)
            if context is not None
            else self.raw.evaluate(target)
        )
        try:
            if len(self._cache) >= self.max_entries:
                oldest = next(iter(self._cache))
                self._cache.pop(oldest, None)
            self._cache[key] = (res, self._snapshot_state())
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

    @property
    def stats(self) -> dict[str, int]:
        """Honest accounting: engine calls vs raw invocations actually saved."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "raw_evals_saved": self.hits,
            "size": len(self._cache),
            "enabled": int(self.enabled),
        }

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.raw, attr)


def attach_eval_cache(
    evaluator: Any,
    *,
    max_entries: int = 4096,
) -> Any:
    """Wires ``evaluator`` behind a program-keyed evaluation cache.

    Opt-in wiring (never auto-applied): the caller decides per evaluator.
    Guarded attach:
      - ``EVOLAB_EVAL_CACHE=0`` disables (returns the evaluator unchanged)
      - evaluators that do not declare ``deterministic == True`` are never
        wrapped (caching nondeterminism would return wrong answers)
      - any wiring error returns the raw evaluator unchanged

    Compose with the experience recorder so the engine sees
    ``recorder(cache(raw))`` — the recorder still observes every call (its
    eval_index semantics are untouched); only raw invocations drop:

        wired = attach_experience_recorder(attach_eval_cache(ev), ...)
    """
    flag = os.environ.get("EVOLAB_EVAL_CACHE", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return evaluator
    if not getattr(evaluator, "deterministic", False):
        return evaluator
    try:
        return EvaluationCache(evaluator, max_entries=max_entries)
    except Exception:
        return evaluator


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
