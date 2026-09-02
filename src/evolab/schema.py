from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL = {
    "total_generations": int,
    "total_candidates_evaluated": int,
    "best_individual": dict,
    "species_distribution": dict,
    "early_stop_triggered": bool,
}

REQUIRED_BEST_KEYS = {"id", "fitness"}

LEGACY_OPTIONAL_BEST_KEYS = {"speedup_vs_baseline"}

RICHNESS_KEYS = {
    "history",
    "config",
    "timestamp_utc",
}

REQUIRED_CONFIG_KEYS = {
    "population_size",
    "mutation_rate",
    "early_stop_fitness",
    "seed",
}

# top-level keys modeled explicitly by RunReport; anything else in the raw
# JSON is captured verbatim in RunReport.extra so save/round-trips never
# drop evidence (audit A14 E-01)
MODELED_RAW_KEYS = {
    "total_generations",
    "total_candidates_evaluated",
    "best_individual",
    "species_distribution",
    "early_stop_triggered",
    "history",
    "config",
    "timestamp_utc",
    "engine_version",
}

SCHEMA_VERSION = "report-schema/1"


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.path}: {self.message}"


@dataclass
class RunReport:
    total_generations: int
    total_candidates_evaluated: int
    best_individual: dict
    species_distribution: dict[str, int]
    early_stop_triggered: bool
    history: list[dict] = field(default_factory=list)
    config: dict | None = None
    timestamp_utc: str | None = None
    engine_version: str | None = None
    source_path: str | None = None
    issues: list[Issue] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    best_ever: Any = None

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def official(self) -> bool:
        """Audit A10 phase-1: three-tier gate.

        parseable  -> JSON decoded into a RunReport
        valid      -> no error-severity issues (structural + invariants)
        official   -> valid AND richness >= 90% (fit for the official record)

        Richness measures metadata PRESENCE only — an official report can
        still be scientifically weak (charter section 8).
        """
        return self.is_valid and self.richness_score >= 90.0

    @property
    def richness_score(self) -> float:
        total = len(RICHNESS_KEYS) + len(REQUIRED_CONFIG_KEYS) + 3
        present = sum(1 for k in RICHNESS_KEYS if getattr(self, k, None))
        if self.config:
            present += sum(1 for k in REQUIRED_CONFIG_KEYS if k in self.config)
        if self.engine_version:
            present += 1
        if self.history and len(self.history) >= 1:
            present += 1
        if self.total_candidates_evaluated > 0 and self.total_generations > 0:
            present += 1
        return round(present / total * 100.0, 1)

    def to_dict(self) -> dict:
        out: dict = {
            "total_generations": self.total_generations,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "best_individual": self.best_individual,
            "species_distribution": self.species_distribution,
            "early_stop_triggered": self.early_stop_triggered,
        }
        if self.history:
            out["history"] = self.history
        if self.config is not None:
            out["config"] = self.config
        if self.timestamp_utc:
            out["timestamp_utc"] = self.timestamp_utc
        if self.engine_version:
            out["engine_version"] = self.engine_version
        # lossless round-trip (audit A14 E-01): unmodeled fields survive
        out.update(self.extra)
        return out


def _reject_dup_keys(pairs: list) -> dict:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON keys detected")
    return dict(pairs)


def parse_report(path: str | Path) -> RunReport:
    path = Path(path)
    issues: list[Issue] = []
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text, object_pairs_hook=_reject_dup_keys)
    except ValueError:
        raw = None
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc

    if raw is None:
        # duplicate-key variant: re-parse leniently so the report is still
        # parseable, but mark it invalid (audit A11 F-05)
        issues.append(
            Issue(
                "error",
                "json",
                "duplicate JSON keys detected — human and machine readers may "
                "disagree on values; report cannot be official",
            )
        )
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("top-level JSON value must be an object")

    for key, expected in REQUIRED_TOP_LEVEL.items():
        if key not in raw:
            issues.append(Issue("error", key, "missing required key"))
        elif not isinstance(raw[key], expected):
            issues.append(
                Issue(
                    "error",
                    key,
                    f"expected {expected.__name__}, got {type(raw[key]).__name__}",
                )
            )

    best = raw.get("best_individual", {})
    if isinstance(best, dict):
        for key in REQUIRED_BEST_KEYS - set(best):
            issues.append(Issue("error", f"best_individual.{key}", "missing key"))
        if "speedup_vs_baseline" in best:
            issues.append(
                Issue(
                    "info",
                    "best_individual.speedup_vs_baseline",
                    "legacy cosmetic metric (1 + fitness/25) — retired, treat as "
                    "fitness-derived label not measured speedup",
                )
            )
        fitness = best.get("fitness")
        if isinstance(fitness, (int, float)) and not 0 <= fitness <= 100:
            issues.append(
                Issue("warning", "best_individual.fitness", f"{fitness} outside [0, 100]")
            )
    else:
        best = {}

    dist = raw.get("species_distribution", {})
    if isinstance(dist, dict):
        for name, count in dist.items():
            if not isinstance(name, str) or not name.startswith("spec_"):
                issues.append(
                    Issue("warning", f"species_distribution.{name}", "unusual species name")
                )
            if not isinstance(count, int) or count < 0:
                issues.append(
                    Issue("error", f"species_distribution.{name}", "count must be a non-negative int")
                )
    else:
        dist = {}

    generations = raw.get("total_generations", 0)
    candidates = raw.get("total_candidates_evaluated", 0)
    if isinstance(generations, int) and isinstance(candidates, int) and generations > 0:
        per_gen = candidates / generations
        if per_gen < 1:
            issues.append(
                Issue(
                    "warning",
                    "total_candidates_evaluated",
                    f"fewer than 1 candidate per generation ({per_gen:.2f})",
                )
            )
    if raw.get("early_stop_triggered") is True and isinstance(generations, int) and generations <= 1:
        issues.append(
            Issue("warning", "early_stop_triggered", "early stop with <= 1 generation is suspicious")
        )

    history = raw.get("history")
    if history is not None and not isinstance(history, list):
        issues.append(Issue("warning", "history", "present but not a list - ignored"))
        history = None
    if isinstance(history, list) and history:
        gens_in_history = [h.get("generation") for h in history if isinstance(h, dict)]
        if gens_in_history != sorted(gens_in_history):
            issues.append(
                Issue("warning", "history", "generations not strictly increasing")
            )
        last_gen = gens_in_history[-1] if gens_in_history else None
        if (
            isinstance(generations, int)
            and isinstance(last_gen, int)
            and last_gen != generations
        ):
            issues.append(
                Issue(
                    "warning",
                    "history",
                    f"last history generation ({last_gen}) != total_generations ({generations})",
                )
            )

    config = raw.get("config")
    if config is None:
        missing_cfg = ", ".join(sorted(REQUIRED_CONFIG_KEYS))
        issues.append(
            Issue(
                "info",
                "config",
                f"missing run parameters - report is not reproducible (need: {missing_cfg})",
            )
        )
    elif not isinstance(config, dict):
        issues.append(Issue("warning", "config", "present but not an object - ignored"))
        config = None
    else:
        for k in REQUIRED_CONFIG_KEYS - set(config):
            issues.append(Issue("warning", f"config.{k}", "missing parameter"))

    if not raw.get("timestamp_utc"):
        issues.append(Issue("info", "timestamp_utc", "missing run time - cannot order runs"))

    if not raw.get("engine_version"):
        issues.append(Issue("info", "engine_version", "missing engine version"))

    fitness_val = best.get("fitness") if isinstance(best, dict) else None
    valid_hist = (
        [
            h
            for h in (history or [])
            if isinstance(h, dict)
            and isinstance(h.get("best_fitness"), (int, float))
        ]
        if isinstance(history, list)
        else []
    )
    if history is not None and isinstance(history, list) and len(valid_hist) != len(history):
        issues.append(
            Issue(
                "warning",
                "history",
                f"{len(history) - len(valid_hist)} malformed entr(ies) ignored "
                "(missing/non-numeric best_fitness)",
            )
        )
    if (
        isinstance(fitness_val, (int, float))
        and valid_hist
    ):
        hist_best = max(h["best_fitness"] for h in valid_hist)
        if abs(hist_best - float(fitness_val)) > 0.01:
            issues.append(
                Issue(
                    "warning",
                    "best_individual.fitness",
                    f"disagrees with history best ({fitness_val} vs {hist_best})",
                )
            )

    # ---- semantic invariants (audits A10 phase-1 / A15) ----
    def inv_err(path: str, message: str) -> None:
        issues.append(Issue("error", path, message))

    def _real_int(v) -> bool:
        return type(v) is int

    def _strict_num(v) -> bool:
        return type(v) in (int, float) and math.isfinite(v)

    # booleans are never legitimate counts/indices (audit A15 P0)
    for k in ("total_generations", "total_candidates_evaluated"):
        v = raw.get(k)
        if v is not None and not _real_int(v):
            inv_err(k, f"must be a real int (got {v!r}); bool is not an integer")

    # config contract: validate every field that affects reproducibility
    # whenever it is present (audit A15: forged config metadata must fail)
    cfg_items = config.items() if isinstance(config, dict) else []
    for ck, cv in cfg_items:
        if ck == "population_size" and not (_real_int(cv) and cv >= 1):
            inv_err(f"config.{ck}", f"must be a real int >= 1 (got {cv!r})")
        elif ck == "seed" and cv is not None and not _real_int(cv):
            inv_err(f"config.{ck}", f"must be a real int or null (got {cv!r})")
        elif ck == "elite_count" and not (_real_int(cv) and cv >= 0):
            inv_err(f"config.{ck}", f"must be a real int >= 0 (got {cv!r})")
        elif ck == "mutation_rate" and not (
            _strict_num(cv) and 0.0 <= float(cv) <= 1.0
        ):
            inv_err(f"config.{ck}", f"must be finite numeric within [0, 1] (got {cv!r})")
        elif ck == "early_stop_fitness" and cv is not None and not _strict_num(cv):
            inv_err(f"config.{ck}", f"must be finite numeric (got {cv!r})")
        elif ck == "sharing_mode" and cv not in ("off", "static", "dynamic"):
            inv_err(f"config.{ck}", f"unknown mode {cv!r}")
        elif ck == "exploit_after_frac" and not (
            _strict_num(cv) and 0.0 < float(cv) <= 1.0
        ):
            inv_err(f"config.{ck}", f"must be within (0, 1] (got {cv!r})")
        elif ck == "fitness_range" and (
            not isinstance(cv, list)
            or len(cv) != 2
            or any(not _strict_num(x) for x in cv)
            or float(cv[0]) >= float(cv[1])
        ):
            inv_err(f"config.{ck}", f"must be [low, high] with low < high (got {cv!r})")

    bid = best.get("id") if isinstance(best, dict) else None
    if isinstance(bid, str) and bid and not re.fullmatch(r"gen_\d+_ind_\d+", bid):
        inv_err("best_individual.id", f"malformed id {bid!r} (expected gen_NN_ind_NN)")

    fv = best.get("fitness") if isinstance(best, dict) else None
    if type(fv) not in (int, float):
        inv_err("best_individual.fitness", f"non-numeric fitness {fv!r}")
    elif isinstance(fv, float) and not math.isfinite(fv):
        inv_err("best_individual.fitness", "non-finite value")

    gens_v = raw.get("total_generations")
    if isinstance(gens_v, int):
        if gens_v < 1:
            inv_err("total_generations", "must be >= 1 for an official record")
        m_id = re.match(r"gen_(\d+)_", bid) if isinstance(bid, str) else None
        if m_id and int(m_id.group(1)) > gens_v:
            inv_err(
                "time_contract",
                f"best id generation {int(m_id.group(1))} exceeds "
                f"total_generations {gens_v}",
            )

    cands_v = raw.get("total_candidates_evaluated")
    pop_v = config.get("population_size") if isinstance(config, dict) else None
    if isinstance(config, dict):
        mr = config.get("mutation_rate")
        if type(mr) not in (int, float) or (
            isinstance(mr, float) and not math.isfinite(mr)
        ) or not (0.0 <= float(mr) <= 1.0):
            inv_err("config.mutation_rate", f"invalid mutation_rate {mr!r}")
        for k in ("early_stop_fitness", "population_size", "seed"):
            v_ = config.get(k)
            if isinstance(v_, float) and not math.isfinite(v_):
                inv_err(f"config.{k}", "non-finite value")
    if dist and isinstance(pop_v, int):
        ssum = sum(v for v in dist.values() if isinstance(v, int))
        if ssum != pop_v:
            inv_err(
                "species_distribution",
                f"specimen sum {ssum} != config.population_size {pop_v}",
            )
    if (
        isinstance(cands_v, int)
        and isinstance(gens_v, int)
        and gens_v >= 1
        and isinstance(pop_v, int)
        and cands_v != gens_v * pop_v
    ):
        inv_err(
            "total_candidates_evaluated",
            f"{cands_v} != generations x population ({gens_v} x {pop_v})",
        )

    hist_raw = raw.get("history")
    if (
        isinstance(hist_raw, list)
        and hist_raw
        and isinstance(gens_v, int)
        and len(hist_raw) != gens_v
    ):
        inv_err(
            "history",
            f"covers {len(hist_raw)} generation(s) != total_generations {gens_v}",
        )
    if isinstance(hist_raw, list) and hist_raw:
        bad_entries = sum(
            1 for h in hist_raw
            if not (
                isinstance(h, dict) and type(h.get("generation")) is int
                and type(h.get("best_fitness")) in (int, float)
                and math.isfinite(h.get("best_fitness"))
            )
        )
        if bad_entries:
            inv_err(
                "history",
                f"{bad_entries} malformed entr(ies): each entry needs integer "
                "'generation' and finite numeric 'best_fitness'",
            )

    if isinstance(dist, dict) and not dist and isinstance(pop_v, int) and pop_v > 0:
        inv_err("species_distribution", "empty despite non-empty population")

    ts = raw.get("timestamp_utc")
    if isinstance(ts, str) and ts:
        try:
            tdt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            if tdt > datetime.now(timezone.utc) + timedelta(days=1):
                inv_err("timestamp_utc", "future timestamp")
        except ValueError:
            inv_err("timestamp_utc", f"unparseable format {ts!r} (expect ISO Zulu)")

    extra = {k: v for k, v in raw.items() if k not in MODELED_RAW_KEYS}

    # ---- cross-report semantic invariants (audit A15 P1) ----
    # Strict (error) for new-format reports carrying schema_version; legacy
    # artifacts keep parsing as warnings per charter section 28.
    strict = raw.get("schema_version") == SCHEMA_VERSION

    def cross_err(path: str, message: str) -> None:
        if strict:
            inv_err(path, message)
        else:
            issues.append(Issue("warning", path, f"[legacy] {message}"))

    es = raw.get("early_stop_triggered")
    target = config.get("early_stop_fitness") if isinstance(config, dict) else None
    last = hist_raw[-1] if isinstance(hist_raw, list) and hist_raw else None
    stopped_for_stagnation = isinstance(last, dict) and last.get("stagnation_stop") is True
    if (
        es is True
        and isinstance(target, (int, float))
        and isinstance(fitness_val, (int, float))
        and float(fitness_val) < float(target) - 1e-9
        and not stopped_for_stagnation
    ):
        cross_err(
            "early_stop_triggered",
            f"claims early stop but best fitness {fitness_val} is below "
            f"declared target {target}",
        )

    spec = raw.get("speciation")
    cfg_spec_off = isinstance(config, dict) and config.get("speciation_enabled") is False
    spec_says_off = isinstance(spec, dict) and spec.get("enabled") is False
    dyn = spec.get("dynamic_species_created") if isinstance(spec, dict) else None
    if (cfg_spec_off or spec_says_off) and _real_int(dyn) and dyn > 0:
        cross_err(
            "speciation.dynamic_species_created",
            f"speciation disabled but {dyn} dynamic species recorded",
        )

    return RunReport(
        total_generations=raw.get("total_generations", 0),
        total_candidates_evaluated=raw.get("total_candidates_evaluated", 0),
        best_individual=best,
        species_distribution=dist,
        early_stop_triggered=bool(raw.get("early_stop_triggered", False)),
        history=history if isinstance(history, list) else [],
        config=config if isinstance(config, dict) else None,
        timestamp_utc=raw.get("timestamp_utc"),
        engine_version=raw.get("engine_version"),
        source_path=str(path),
        issues=issues,
        extra=extra,
    )


def save_report(report: RunReport, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
