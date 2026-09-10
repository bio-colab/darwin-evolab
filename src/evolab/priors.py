"""priors.py — Memory-guided mutation and sequence priors (M6, M7).

Extracted from experience.py for modular decomposition.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .experience import ExperienceStore

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



__all__ = ["ExperienceMutationPrior", "ExperienceSequencePrior"]
