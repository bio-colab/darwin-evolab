"""self_model.py — Self-Model, diagnostic evaluation, and meta-control.

Extracted from experience.py for modular decomposition.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

def wilson_interval(passes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a pass rate — the Self-Model confidence.

    Raw ``passes/trials`` without uncertainty is false consciousness; the
    interval is the honest part of the capability claim.
    """
    if trials <= 0:
        return (0.0, 1.0)
    p = max(0.0, min(float(passes) / trials, 1.0))
    denom = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    spread = z * ((p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) ** 0.5)
    return (
        round(max(0.0, (center - spread) / denom), 4),
        round(min(1.0, (center + spread) / denom), 4),
    )


def _self_enabled() -> bool:
    try:
        flag = os.environ.get("EVOLAB_SELF", "1").strip().lower()
    except Exception:
        return True
    return flag not in ("0", "false", "no", "off")


# Phase 2/3 preregistered thresholds (frozen before measurement — tuning
# them after looking at results is forbidden by the repo's preregistration
# discipline).
META_DIVERSITY_LOW = 0.05
META_PLATEAU_GENS = 5
META_PLATEAU_EPS = 0.01
META_STD_LOW = 0.5
META_OVERFIT_GAP = 30.0


def diagnose_run(
    history: list[dict[str, Any]] | None,
    holdout_gap: float | None = None,
    suspicion_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Phase 2 (+ Atom 2): read-only self-diagnosis over one run's telemetry.

    Asks "what happened to me?" — never "what should I do?". Pure function:
    no DB, no RNG, no search side effects. Never raises; insufficient data
    yields explicit unknown flags instead of invented verdicts.

    Atom 2 splits the mirror in two honest senses (same verdict keys as
    before, so all Phase 2 callers keep working):
      inner (interoception) — diversity, spread, plateau, energy burn rate.
      outer (exteroception) — holdout gap and fault-localization state.
    Rule: premature_convergence is an INNER verdict (plateau + collapsed
    diversity); overfit_risk is an OUTER verdict (gap only) and stays None
    when unmeasured — inner collapse must never imply outer overfit.
    """
    thresholds = {
        "diversity_low": META_DIVERSITY_LOW,
        "plateau_gens": META_PLATEAU_GENS,
        "plateau_eps": META_PLATEAU_EPS,
        "std_low": META_STD_LOW,
        "overfit_gap": META_OVERFIT_GAP,
    }
    try:
        hist = list(history or [])
        if len(hist) < META_PLATEAU_GENS:
            return {
                "premature_convergence": False,
                "stagnation": False,
                "overfit_risk": None if holdout_gap is None else bool(holdout_gap > META_OVERFIT_GAP),
                "reasons": ["insufficient_data"],
                "thresholds": thresholds,
                "inner": {"plateau": None, "diversity": None, "diversity_low": None,
                          "unique_programs": None, "unique_programs_ratio": None,
                          "std": None, "energy_burn_per_eval": None},
                "outer": {"holdout_gap": holdout_gap, "overfit_risk": None if holdout_gap is None else bool(holdout_gap > META_OVERFIT_GAP),
                          "bad_localization": None},
            }
        tail = hist[-META_PLATEAU_GENS:]
        bests = [float(h.get("best_fitness", 0.0) or 0.0) for h in tail]
        plateau = (max(bests) - min(bests)) <= META_PLATEAU_EPS
        last = tail[-1]
        try:
            div = float(last.get("diversity", 1.0))
        except (TypeError, ValueError):
            div = 1.0
        try:
            std = float(last.get("std_fitness", 1.0))
        except (TypeError, ValueError):
            std = 1.0
        # Atom 2 inner: energy burn = fitness points gained per eval over tail.
        try:
            spent = sum(float(h.get("energy_spent", 0.0) or 0.0) for h in tail)
            burn = round((bests[-1] - bests[0]) / spent, 4) if spent > 0 else None
        except (TypeError, ValueError):
            burn = None
        # Atom 2 outer: localization state is caller-supplied or unknown.
        try:
            bad_loc = None
            if isinstance(suspicion_state, dict) and suspicion_state:
                empty = suspicion_state.get("empty")
                misleading = suspicion_state.get("misleading")
                if empty is True:
                    bad_loc = "empty"
                elif misleading is True:
                    bad_loc = "misleading"
                else:
                    bad_loc = "ok"
        except Exception:
            bad_loc = None
        premature = bool(plateau and div < META_DIVERSITY_LOW and std < META_STD_LOW)
        reasons = []
        if premature:
            reasons.append("plateau_with_collapsed_diversity")
        elif plateau:
            reasons.append("plateau_without_collapse")
        if div < META_DIVERSITY_LOW:
            reasons.append("diversity_low")
        if not reasons:
            reasons.append("nominal")
        overfit = None if holdout_gap is None else bool(float(holdout_gap) > META_OVERFIT_GAP)
        return {
            "premature_convergence": premature,
            "stagnation": bool(plateau),
            "overfit_risk": overfit,
            "reasons": reasons,
            "thresholds": thresholds,
            "inner": {"plateau": bool(plateau), "diversity": round(div, 4),
                      "diversity_low": bool(div < META_DIVERSITY_LOW),
                      "unique_programs": last.get("unique_programs"),
                      "unique_programs_ratio": last.get("unique_programs_ratio"),
                      "std": round(std, 4), "energy_burn_per_eval": burn},
            "outer": {"holdout_gap": holdout_gap, "overfit_risk": overfit,
                      "bad_localization": bad_loc},
        }
    except Exception:
        return {
            "premature_convergence": False,
            "stagnation": False,
            "overfit_risk": None,
            "reasons": ["diagnosis_error"],
            "thresholds": thresholds,
            "inner": {"plateau": None, "diversity": None, "diversity_low": None,
                      "unique_programs": None, "unique_programs_ratio": None,
                      "std": None, "energy_burn_per_eval": None},
            "outer": {"holdout_gap": holdout_gap, "overfit_risk": None,
                      "bad_localization": None},
        }


def self_critic_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Phase 3: read-only self-criticism metrics for one run report.

    quality      = best_fitness / 100 (solution goodness).
    efficiency   = fitness points gained per evaluation (best-first)/evals.
    generalization = None when the run has no train/holdout gap to measure
                     (honest unknown, not 0.0); otherwise gap-based flag.
    novelty      = mean measured diversity across generations.
    Metrics only — no caller may branch search on them in Phase 3.
    """
    try:
        rep = dict(report or {})
        hist = rep.get("history", []) or []
        best = rep.get("best_individual", {}) or {}
        try:
            best_f = float(best.get("fitness", 0.0) or 0.0)
        except (TypeError, ValueError):
            best_f = 0.0
        quality = round(max(0.0, min(best_f / 100.0, 1.0)), 4)
        try:
            first_f = float(hist[0].get("best_fitness", best_f) or best_f) if hist else best_f
        except (TypeError, ValueError, IndexError):
            first_f = best_f
        try:
            evals = int(rep.get("total_candidates_evaluated", 0) or 0)
        except (TypeError, ValueError):
            evals = 0
        efficiency = round(max(0.0, min((best_f - first_f) / max(evals, 1), 1.0)), 4) if hist else 0.0
        divs = []
        for h in hist:
            try:
                divs.append(float(h.get("diversity", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        novelty = round(sum(divs) / len(divs), 4) if divs else 0.0
        return {
            "quality": quality,
            "efficiency": efficiency,
            "generalization": None,
            "novelty": novelty,
            "note": "metrics only - no search decision reads this in Phase 3",
        }
    except Exception:
        return {
            "quality": 0.0, "efficiency": 0.0, "generalization": None,
            "novelty": 0.0, "note": "critic_error",
        }


def attach_self_assessment(engine: Any, report: dict[str, Any]) -> None:
    """Phase 2+3: attach read-only mirror entries to a finished report.

    Mutates only ``report["extra"]`` after the evolutionary loop — search
    trajectory is already fixed. Silent no-op when ``EVOLAB_SELF=0`` or on
    any error.
    """
    if not _self_enabled():
        return
    try:
        if not isinstance(report, dict):
            return
        hist = report.get("history", [])
        report.setdefault("extra", {})["meta_evaluator"] = diagnose_run(hist)
        report.setdefault("extra", {})["self_critic"] = self_critic_report(report)
    except Exception:
        pass


# Phase 4 preregistered constants (frozen before measurement).
META_CONTROLLER_FRACTION = 0.15
META_CONTROLLER_MODES = ("directed", "reshuffle")


def _meta_enabled(mode: Any = None) -> bool:
    """Phase 4 kill-switch: explicit opt-in only, default OFF.

    Enabled iff ``mode`` is a known controller mode OR ``EVOLAB_META=1``.
    ``None``/unknown without the env flag → exact legacy behavior.
    """
    try:
        if isinstance(mode, str) and mode.strip().lower() in META_CONTROLLER_MODES:
            return True
        flag = os.environ.get("EVOLAB_META", "0").strip().lower()
        return flag in ("1", "true", "yes", "on")
    except Exception:
        return False


def _meta_resolve_mode(mode: Any) -> str:
    try:
        m = str(mode or "").strip().lower()
        if m in META_CONTROLLER_MODES:
            return m
    except Exception:
        pass
    try:
        flag = os.environ.get("EVOLAB_META", "0").strip().lower()
        if flag in ("directed", "reshuffle"):
            return flag
    except Exception:
        pass
    return "directed"


def meta_control_step(
    engine: Any,
    children: list[Any],
    history: list[dict[str, Any]] | None,
    gen: int,
    mode: str = "directed",
) -> int:
    """Phase 4: single-effector reactive immigrant injection.

    The ONLY search intervention allowed to the meta-controller: replace up
    to ``k = max(1, pop*FRACTION)`` trailing (non-elite) children with fresh
    random individuals. ``directed`` fires only on diagnosed stagnation /
    premature convergence; ``reshuffle`` fires on a fixed schedule
    (``gen % PLATEAU_GENS == 0``) to isolate injection noise from trigger
    value (the M7/M8 lesson). Numeric genomes only; code track untouched.
    Deterministic via ``engine.rng``. Never raises — 0 means "did nothing".
    """
    try:
        if bool(getattr(engine, "_code_mode", False)):
            return 0
        pop_size = int(getattr(engine, "population_size", len(children)) or len(children))
        if pop_size < 2 or not children:
            return 0
        fire = False
        reason = ""
        if mode == "reshuffle":
            fire = (int(gen) % max(int(META_PLATEAU_GENS), 1) == 0)
            reason = "reshuffle_schedule"
        else:
            diag = diagnose_run(list(history or []))
            if diag.get("stagnation") or diag.get("premature_convergence"):
                fire = True
                reason = "+".join(diag.get("reasons", ["stagnation"]))
        if not fire:
            return 0
        k = max(1, int(round(pop_size * META_CONTROLLER_FRACTION)))
        elite = int(getattr(engine, "elite_count", 0) or 0)
        room = max(len(children) - elite, 0)
        k = max(0, min(k, room))
        if k <= 0:
            return 0
        from .genome import random_individual as _random_individual
        from .speciation import SPECIES_POOL as _POOL
        pool = list(_POOL) if isinstance(_POOL, dict) else list(_POOL)
        injected = 0
        for i in range(k):
            try:
                sp = engine.rng.choice(pool) if pool else "spec_0"
                children[-(1 + i)] = _random_individual(
                    sp, size=int(getattr(engine, "genome_size", 16)), rng=engine.rng)
                injected += 1
            except Exception:
                break
        if injected:
            try:
                engine._decision_log.append({
                    "at_generation": int(gen),
                    "event": "meta_controller_injection",
                    "detail": f"mode={mode};injected={injected};reason={reason}",
                })
            except Exception:
                pass
        return injected
    except Exception:
        return 0


def _compute_governor_statistics(b: list[float], c: list[float]) -> tuple[float, float]:
    """Computes one-tailed hypothesis test p-value and Cohen's d for candidate vs baseline."""
    import math
    import statistics as _st

    if len(b) == len(c) and len(b) > 1:
        diffs = [ci - bi for bi, ci in zip(b, c)]
        mean_diff = _st.mean(diffs)
        try:
            std_diff = _st.stdev(diffs)
        except Exception:
            std_diff = 0.0

        if std_diff <= 1e-9:
            # Deterministic/uniform change across all paired seeds
            p_val = 0.0 if mean_diff > 0 else (1.0 if mean_diff < 0 else 0.5)
            cohen_d = 1.0 if mean_diff > 0 else (-1.0 if mean_diff < 0 else 0.0)
            return p_val, cohen_d

        n = len(diffs)
        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se
        cohen_d = round(mean_diff / std_diff, 4)

        try:
            import scipy.stats as _stats
            p_val = float(_stats.t.sf(t_stat, df=n - 1))
        except Exception:
            # Standard normal survival approximation fallback
            p_val = 0.5 * math.erfc(t_stat / math.sqrt(2.0))
        return round(max(0.0, min(1.0, p_val)), 6), cohen_d
    elif len(b) > 1 and len(c) > 1:
        # Unpaired two-sample Welch t-test
        mean_b, mean_c = _st.mean(b), _st.mean(c)
        var_b = _st.variance(b) if len(b) > 1 else 0.0
        var_c = _st.variance(c) if len(c) > 1 else 0.0
        pooled_s = math.sqrt((var_b + var_c) / 2.0) if (var_b + var_c) > 0 else 1e-6
        cohen_d = round((mean_c - mean_b) / pooled_s, 4)
        se = math.sqrt(var_b / len(b) + var_c / len(c))
        if se <= 1e-9:
            p_val = 0.0 if mean_c > mean_b else 1.0
            return p_val, cohen_d
        t_stat = (mean_c - mean_b) / se
        try:
            import scipy.stats as _stats
            df_denom = (var_b / len(b)) ** 2 / (len(b) - 1) + (var_c / len(c)) ** 2 / (len(c) - 1)
            df = ((var_b / len(b) + var_c / len(c)) ** 2) / max(1e-9, df_denom)
            p_val = float(_stats.t.sf(t_stat, df=df))
        except Exception:
            p_val = 0.5 * math.erfc(t_stat / math.sqrt(2.0))
        return round(max(0.0, min(1.0, p_val)), 6), cohen_d
    else:
        mean_b = b[0] if b else 0.0
        mean_c = c[0] if c else 0.0
        p_val = 0.0 if mean_c > mean_b else 1.0
        cohen_d = 1.0 if mean_c > mean_b else 0.0
        return p_val, cohen_d


def govern_modification(
    baseline: list[float] | None,
    candidate: list[float] | None,
    regressions: int = 0,
    alpha: float | None = None,
    min_effect_size: float | None = None,
) -> dict[str, Any]:
    """Phase 5: mathematically calibrated evolutionary self-governance decision.

    ACCEPT iff ALL hold:
      1. mean_c > mean_b (positive central tendency shift)
      2. median_c > median_b (median non-dominated)
      3. worst_c >= worst_b (worst-case performance bounded)
      4. regressions == 0 (zero holdout / test regressions)
      5. p_value < alpha (statistically significant against null hypothesis, N >= 5)
      6. cohen_d >= min_effect_size (if requested)
    Anything else → REJECT with explicit failure reasons. Never raises.
    """
    import statistics as _st
    try:
        b = [float(x) for x in (baseline or [])]
        c = [float(x) for x in (candidate or [])]
        if not b or not c:
            return {"decision": "REJECT", "reasons": ["insufficient_evidence"],
                    "mean_b": None, "mean_c": None, "median_b": None,
                    "median_c": None, "worst_b": None, "worst_c": None,
                    "regressions": int(regressions or 0),
                    "p_value": None, "cohen_d": None}
        mean_b, mean_c = round(_st.mean(b), 4), round(_st.mean(c), 4)
        median_b, median_c = round(_st.median(b), 4), round(_st.median(c), 4)
        worst_b, worst_c = round(min(b), 4), round(min(c), 4)
        reasons = []
        if not (mean_c > mean_b):
            reasons.append("mean_not_improved")
        if not (median_c > median_b):
            reasons.append("median_not_improved")
        if not (worst_c >= worst_b):
            reasons.append("worst_regressed")
        if int(regressions or 0) != 0:
            reasons.append("regressions_present")

        p_val, cohen_d = _compute_governor_statistics(b, c)
        if alpha is not None and len(b) >= 5 and p_val >= alpha:
            reasons.append("not_statistically_significant")
        if min_effect_size is not None and cohen_d < min_effect_size:
            reasons.append("effect_size_insufficient")

        decision = "ACCEPT" if not reasons else "REJECT"
        return {"decision": decision, "reasons": reasons or ["all_gates_passed"],
                "mean_b": mean_b, "mean_c": mean_c, "median_b": median_b,
                "median_c": median_c, "worst_b": worst_b, "worst_c": worst_c,
                "regressions": int(regressions or 0),
                "delta_mean": round(mean_c - mean_b, 4),
                "delta_median": round(median_c - median_b, 4),
                "p_value": p_val,
                "cohen_d": cohen_d}
    except Exception:
        return {"decision": "REJECT", "reasons": ["governor_error"],
                "mean_b": None, "mean_c": None, "median_b": None,
                "median_c": None, "worst_b": None, "worst_c": None,
                "regressions": int(regressions or 0),
                "p_value": None, "cohen_d": None}



def render_self_modification_proposal(
    proposal_id: Any,
    changed: str,
    baseline: list[float] | None,
    candidate: list[float] | None,
    regressions: int = 0,
) -> str:
    """Phase 5: deterministic SELF-MODIFICATION PROPOSAL text block."""
    v = govern_modification(baseline, candidate, regressions)
    lines = [
        f"SELF-MODIFICATION PROPOSAL #{proposal_id}",
        "",
        f"Changed: {changed}",
        f"Baseline: mean={v['mean_b']} median={v['median_b']} worst={v['worst_b']}",
        f"Candidate: mean={v['mean_c']} median={v['median_c']} worst={v['worst_c']}",
        f"Delta: mean={v.get('delta_mean')} median={v.get('delta_median')}",
        f"Regressions: {v['regressions']}",
        f"Reasons: {', '.join(v['reasons'])}",
        "",
        f"Decision: {v['decision']}",
    ]
    return "\n".join(lines)



def record_engine_self_run(engine: Any, report: dict[str, Any]) -> None:
    """Phase 1: silent post-run self fact. Never raises, never affects search.

    Strategy string is the engine's honest configuration (sharing/speciation/
    causal/memory/qd), NOT a capability claim. Capability calibration
    (pass/fail per domain) is recorded separately via record_capability by
    the caller that owns the domain verdict (e.g. holdout gate).
    """
    if not _self_enabled():
        return
    try:
        path = os.environ.get("EVOLAB_SELF_DB") or Path.cwd() / "data" / "self_model.db"
        from .experience import ExperienceStore
        store = ExperienceStore(path)
        try:
            hist = report.get("history", []) if isinstance(report, dict) else []
            best = report.get("best_individual", {}) if isinstance(report, dict) else {}
            mean_fit = hist[-1].get("mean_fitness") if hist else None
            div_fin = hist[-1].get("diversity") if hist else None
            stag = sum(1 for h in hist if h.get("stagnation_stop"))
            store.record_self_run({
                "run_id": uuid.uuid4().hex[:12],
                "strategy": (
                    f"sharing={getattr(engine, 'sharing_mode', '?')};"
                    f"spec={bool(getattr(engine, 'speciation_enabled', False))};"
                    f"causal={bool(getattr(engine, 'causal_layer_enabled', False))};"
                    f"memory={bool(getattr(engine, 'memory_enabled', False))};"
                    f"qd={bool(getattr(engine, 'qd_selection', False))};"
                    f"imm_frac={getattr(engine, 'immigrant_fraction', 0.0)};"
                    f"meta={getattr(engine, 'meta_mode', None)}"
                ),
                "seed": getattr(engine, "seed", None),
                "generations": report.get("total_generations", 0) if isinstance(report, dict) else 0,
                "population_size": getattr(engine, "population_size", 0),
                "best_fitness": float(best.get("fitness", 0.0) or 0.0),
                "mean_fitness": mean_fit,
                "evals_total": int(report.get("total_candidates_evaluated", 0) or 0) if isinstance(report, dict) else 0,
                "early_stopped": bool(report.get("early_stop_triggered", False)) if isinstance(report, dict) else None,
                "stagnation_gens": stag,
                "diversity_final": div_fin,
            })
        finally:
            try:
                store.close()
            except Exception:
                pass
    except Exception:
        pass


__all__ = [
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
