"""Causal inference layer (audit A25 / SEE proposal): minimal measurable slice.

Tracks per-event causal outcomes, aggregates context-conditioned success
rates, and uses those rates to bias future mutation-type selection.

Honest scope: operates on the data the engine already records
(mutation_l1, parent_fitness, operator kind). Does NOT compute evaluator
gradient probes or AST-level analysis — those require the v5 Evaluator
contract.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Layer 1: CausalEvent — atomic record of one breeding outcome
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CausalEvent:
    """One crossover+mutation outcome with enough context to learn from."""
    generation: int
    mutation_type: str          # "light" | "semantic"
    species_id: str
    parent_fitness_mean: float
    child_fitness: float
    fitness_delta: float        # child_fitness - parent_fitness_mean
    mutation_l1: float
    species_size: int           # how many same-species peers existed
    archive_nearest: float      # L1 distance to closest archived genome
    phase: str                  # "exploration" | "exploitation"

    @property
    def succeeded(self) -> bool:
        return self.fitness_delta > 0


# ---------------------------------------------------------------------------
# Layer 2: Bayesian model — online Beta-Bernoulli per (type, context_bin)
# ---------------------------------------------------------------------------

class CausalModel:
    """Online Bayesian success-rate tracker per mutation type per context with bounded capacity.

    Supports tri-state outcome classification:
      - Success: delta > +epsilon
      - Failure: delta < -epsilon
      - Neutral / Inert: |delta| <= epsilon (e.g. dead-code, formatting, comments)

    Forgetting policy (memory-hygiene M4): forgetting here is event-triggered
    and wholesale, never gradual. The inertia breaker
    (``StrategicMutationSelector.reset_ucb_weights``) clears the counters
    under environmental shift — and forgives trap signatures with them —
    while per-context staleness is handled by the trap library's own TTL.

    A gradual ``apply_decay`` (uniform multiplicative count shrinkage)
    existed here once and was removed deliberately: multiplying ``s``, ``f``
    and ``n`` by any common factor is a no-op on ``success_rate`` (the ratio
    is scale-invariant), so it could not mitigate the historical poisoning
    its docstring claimed — a poisoned 1.0 rate survives decay exactly —
    while silently corrupting the integer count contract (floats), startling
    trap confirmations below threshold with zero new data, and pushing the
    selector under its min-observations gate into a coin flip. Any future
    decay-like mechanism must be a preregistered, measured design (additive
    shrinkage toward uniform, as the experience prior uses — not
    multiplicative count decay).
    """

    def __init__(self, max_keys: int = 5000, epsilon: float = 1e-7) -> None:
        self.max_keys = max_keys
        self.epsilon = epsilon
        self.has_neutral_class = True
        self._stats: dict[str, dict[str, int]] = {}

    def observe(self, mtype: str, context_bin: str, delta: float) -> None:
        key = f"{mtype}|{context_bin}"
        if key not in self._stats:
            # Prune least observed key if max capacity is exceeded
            if len(self._stats) >= self.max_keys:
                min_key = min(self._stats.keys(), key=lambda k: self._stats[k]["s"] + self._stats[k]["f"] + self._stats[k].get("n", 0))
                self._stats.pop(min_key, None)
            self._stats[key] = {"s": 0, "f": 0, "n": 0}

        if delta > self.epsilon:
            self._stats[key]["s"] += 1
        elif delta < -self.epsilon:
            self._stats[key]["f"] += 1
        else:
            # Neutral / causally inert mutation
            self._stats[key]["n"] = self._stats[key].get("n", 0) + 1

    def success_rate(self, mtype: str, context_bin: str, neutral_weight: float = 0.5) -> float:
        stats = self._stats.get(f"{mtype}|{context_bin}", {})
        s = stats.get("s", 0)
        f = stats.get("f", 0)
        n = stats.get("n", 0)
        total = s + f + n
        if total == 0:
            return 0.0
        return (s + neutral_weight * n) / total

    def neutral_rate(self, mtype: str, context_bin: str) -> float:
        stats = self._stats.get(f"{mtype}|{context_bin}", {})
        n = stats.get("n", 0)
        total = stats.get("s", 0) + stats.get("f", 0) + n
        return n / max(1, total)

    def observations(self, mtype: str, context_bin: str) -> int:
        d = self._stats.get(f"{mtype}|{context_bin}", {})
        return d.get("s", 0) + d.get("f", 0) + d.get("n", 0)

    def summary(self) -> dict:
        out = {}
        for key, counts in sorted(self._stats.items()):
            total = counts["s"] + counts["f"] + counts.get("n", 0)
            if total == 0:
                continue
            out[key] = {
                "success_rate": round((counts["s"] + 0.5 * counts.get("n", 0)) / total, 3),
                "total": total,
                "neutral": counts.get("n", 0),
            }
        return out


# ---------------------------------------------------------------------------
# Layer 3: Trap Signature Library — detects repeated failure patterns
# ---------------------------------------------------------------------------

@dataclass
class TrapSignature:
    context_bin: str
    failure_count: int
    mean_negative_delta: float
    confidence: float
    last_seen_generation: int


class TrapSignatureLibrary:
    """Learns 'this context keeps producing failures' from accumulated events.

    A trap signature is created when >= min_failures negative-delta events
    cluster in the same context bin. It is validated when the failure rate
    exceeds 70% across all observations in that bin.

    Forgiveness contract (memory-hygiene M5): a trap is a HYPOTHESIS about a
    context, not a life sentence. Every ``scan`` re-validates surviving
    traps against the model's CURRENT counts; a trap that is no longer
    confirmed (fail rate recovered below the threshold, or the context
    stopped producing failures) is not revoked immediately — it is simply
    not re-confirmed, and expires once it has gone
    ``ttl_generations`` generations without re-confirmation. ``reset``
    clears all signatures (the inertia breaker uses it: when the model
    forgets its statistics under environmental drift, traps describing the
    forgotten world are forgiven with it).

    NOTE: before M5, traps were created and never updated or removed —
    ``last_seen_generation`` stayed 0 forever and a 0.1x penalty applied
    permanently even if the context later improved. The library was also
    never wired into the engine (scan had no caller, and the selector was
    built without a trap_library) — it existed only in direct tests. Both
    defects are fixed: the class forgives, and the engine wires it (under
    the opt-in ``causal_layer_enabled`` flag).
    """

    def __init__(
        self,
        min_failures: int = 5,
        fail_rate_threshold: float = 0.7,
        ttl_generations: int = 20,
    ):
        self.min_failures = min_failures
        self.fail_rate_threshold = fail_rate_threshold
        self.ttl_generations = max(1, int(ttl_generations))
        self.signatures: dict[str, TrapSignature] = {}
        self.expired: int = 0

    def _confirmed(self, key: str, counts: dict) -> bool:
        total = counts["s"] + counts["f"]
        return (
            counts["f"] >= self.min_failures
            and counts["f"] / max(total, 1) >= self.fail_rate_threshold
        )

    def scan(self, model: CausalModel, generation: int = 0) -> list[TrapSignature]:
        """Scan model stats: create, re-validate and expire trap signatures.

        ``generation`` anchors the forgiveness clock: a trap survives only
        while it is re-confirmed by current data at least once per
        ``ttl_generations``. Deterministic in (model stats, generation).
        """
        new_or_updated = []
        # 1) re-validate or create, from CURRENT model stats (signed total
        #    s+f as the confirmation base — neutral observations must not
        #    keep a stale trap alive)
        for key, counts in model._stats.items():
            if not self._confirmed(key, counts):
                continue
            fails = counts["f"]
            if key in self.signatures:
                sig = self.signatures[key]
                sig.failure_count = fails
                sig.confidence = min(1.0, fails / 20)
                sig.last_seen_generation = generation
                new_or_updated.append(sig)
            else:
                sig = TrapSignature(
                    context_bin=key,
                    failure_count=fails,
                    mean_negative_delta=0.0,   # populated externally
                    confidence=min(1.0, fails / 20),
                    last_seen_generation=generation,
                )
                self.signatures[key] = sig
                new_or_updated.append(sig)
        # 2) forgive traps gone unconfirmed for longer than the TTL
        for key in [k for k, s in self.signatures.items()
                    if generation - s.last_seen_generation > self.ttl_generations]:
            del self.signatures[key]
            self.expired += 1
        return new_or_updated

    def reset(self) -> None:
        """Forgive every signature (used by the causal inertia breaker)."""
        self.signatures.clear()

    def is_known_trap(self, context_bin: str) -> bool:
        return context_bin in self.signatures


# ---------------------------------------------------------------------------
# Layer 4: Strategic Mutation Selector — biases kind choice using evidence
# ---------------------------------------------------------------------------

class StrategicMutationSelector:
    """Replaces the raw coin-flip with an evidence-informed decision.

    For each breeding event, queries CausalModel for historical success
    rates of light vs semantic mutations in the current context bin.
    Biases the choice toward the historically better-performing type,
    while maintaining exploration via epsilon-greedy fallback.
    """

    def __init__(
        self,
        model: CausalModel,
        epsilon: float = 0.15,
        global_failure_threshold: float = 0.10,
        trap_library: TrapSignatureLibrary | None = None,
    ) -> None:
        self.model = model
        self.epsilon = epsilon  # explore even when evidence favours one type
        self.global_failure_threshold = global_failure_threshold
        self.trap_library = trap_library
        self._fitness_peak = 0.0
        self._inertia_resets = 0

    def check_global_drift(self, current_mean_fitness: float) -> bool:
        """Reset UCB statistics if mean fitness drops significantly from peak (Causal Inertia Breaker)."""
        if current_mean_fitness > self._fitness_peak:
            self._fitness_peak = current_mean_fitness
            return False

        if self._fitness_peak > 0:
            rel_drop = (self._fitness_peak - current_mean_fitness) / self._fitness_peak
            if rel_drop >= self.global_failure_threshold:
                self.reset_ucb_weights()
                self._fitness_peak = current_mean_fitness
                self._inertia_resets += 1
                return True
        return False

    def reset_ucb_weights(self) -> None:
        """Reset CausalModel statistics to break causal inertia under environmental shift.

        Forgiveness coupling (M5): when the model forgets its statistics,
        trap signatures describing the forgotten world are forgiven with it
        — otherwise stale traps would keep penalizing contexts the model
        itself no longer remembers."""
        if hasattr(self.model, "_stats"):
            self.model._stats.clear()
        if self.trap_library is not None:
            self.trap_library.reset()

    def select(self, context_bin: str, rng: random.Random) -> str:
        """Return 'light' or 'semantic' with epsilon-greedy exploration and proactive trap penalty."""
        # epsilon-greedy: explore randomly epsilon fraction of the time
        if rng.random() < self.epsilon:
            return "light" if rng.random() < 0.5 else "semantic"

        light_rate = self.model.success_rate("light", context_bin)
        sem_rate = self.model.success_rate("semantic", context_bin)
        light_n = self.model.observations("light", context_bin)
        sem_n = self.model.observations("semantic", context_bin)

        if light_n < 5 and sem_n < 5:
            return "light" if rng.random() < 0.5 else "semantic"

        light_score = light_rate + math.sqrt(
            2 * math.log(max(light_n + sem_n, 1)) / max(light_n, 1))
        sem_score = sem_rate + math.sqrt(
            2 * math.log(max(light_n + sem_n, 1)) / max(sem_n, 1))

        # Proactive Trap Signature Immunity: penalize known traps
        if self.trap_library:
            if self.trap_library.is_known_trap(f"light|{context_bin}") or self.trap_library.is_known_trap(context_bin):
                light_score *= 0.1
            if self.trap_library.is_known_trap(f"semantic|{context_bin}") or self.trap_library.is_known_trap(context_bin):
                sem_score *= 0.1

        return "light" if light_score >= sem_score else "semantic"


# ---------------------------------------------------------------------------
# Aggregator: converts raw events into context-binned summaries
# ---------------------------------------------------------------------------

def discretize_code_context(
    genome: Any, parent_fitness: float, last_error: str | None = None
) -> str:
    """Discretizes code context (ASTGenome / PatchGenome) for causal learning."""
    if hasattr(genome, "describe"):
        desc = genome.describe()
        complexity = (
            "complex"
            if isinstance(desc, dict) and (desc.get("max_depth", 1) > 4 or desc.get("hunk_count", 0) > 2)
            else "simple"
        )
    else:
        complexity = "simple"

    fit_tier = "low" if parent_fitness < 40 else ("mid" if parent_fitness < 90 else "high")
    err_flag = "err" if last_error else "ok"
    return f"cplx:{complexity}|fit:{fit_tier}|status:{err_flag}"


def discretize_context(
    genome: Any,
    parent_fitness: float,
    archive_nearest: float = 1.0,
    species_size: int = 10,
    last_error: str | None = None,
) -> str:
    """Produce a discrete context key for the causal model across numerical and code genomes."""
    if isinstance(genome, (list, tuple)) or hasattr(genome, "genes") or hasattr(genome, "values"):
        g_vals = list(genome.genes if hasattr(genome, "genes") else (genome.values if hasattr(genome, "values") else genome))
        mean_g = sum(g_vals) / len(g_vals) if g_vals else 0.0
        spread = max(abs(g - mean_g) for g in g_vals) if g_vals else 0.0
        grad_lo = spread < 2.5
        arch_near = archive_nearest < 0.4
        spec_small = species_size < 5
        stagnant = parent_fitness < 50
        return "|".join([
            f"grad:{'lo' if grad_lo else 'hi'}",
            f"arch:{'near' if arch_near else 'far'}",
            f"spec:{'small' if spec_small else 'large'}",
            f"stag:{'yes' if stagnant else 'no'}",
        ])
    return discretize_code_context(genome, parent_fitness, last_error=last_error)

