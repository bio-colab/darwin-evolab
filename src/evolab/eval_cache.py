"""eval_cache.py — Program-keyed memoization of raw evaluator results (M3).

Extracted from experience.py for modular decomposition.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

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
    if isinstance(evaluator, EvaluationCache):
        return evaluator
    if not getattr(evaluator, "deterministic", False):
        return evaluator
    try:
        return EvaluationCache(evaluator, max_entries=max_entries)
    except Exception:
        return evaluator


__all__ = ["EvaluationCache", "attach_eval_cache"]
