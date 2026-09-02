from __future__ import annotations

from .schema import RunReport


def _trend(history: list[dict]) -> tuple[float, int]:
    best_vals = [
        h["best_fitness"]
        for h in history
        if isinstance(h.get("best_fitness"), (int, float))
    ]
    if len(best_vals) < 2:
        return 0.0, 0
    total_gain = best_vals[-1] - best_vals[0]
    plateau = 0
    threshold = (max(best_vals) - min(best_vals)) * 0.01 if best_vals else 0.0
    for i in range(len(best_vals) - 1, 0, -1):
        if abs(best_vals[i] - best_vals[i - 1]) <= max(threshold, 0.01):
            plateau += 1
        else:
            break
    return total_gain, plateau


def _convergence_gen(history: list[dict], target: float) -> int | None:
    for h in history:
        f = h.get("best_fitness")
        if isinstance(f, (int, float)) and f >= target:
            return h.get("generation")
    return None


def summarize(report: RunReport) -> list[str]:
    lines: list[str] = []
    lines.append(f"Source           : {report.source_path}")
    if report.timestamp_utc:
        lines.append(f"Run time (UTC)   : {report.timestamp_utc}")
    if report.engine_version:
        lines.append(f"Engine           : {report.engine_version}")
    lines.append(f"Generations      : {report.total_generations}")
    lines.append(f"Candidates       : {report.total_candidates_evaluated}")
    if report.total_generations > 0:
        per_gen = report.total_candidates_evaluated / report.total_generations
        lines.append(f"Candidates/gen   : {per_gen:.2f}")

    if report.config:
        cfg = report.config
        parts = [f"{k}={cfg[k]}" for k in sorted(cfg)]
        lines.append(f"Config           : {' '.join(parts)}")
    else:
        lines.append("Config           : <not recorded>")

    best = report.best_individual
    fitness = best.get("fitness")
    species_tag = f", species={best['species']}" if "species" in best else ""
    lines.append(f"Best individual  : {best.get('id')} (fitness={fitness}{species_tag})")
    if best.get("passed_holdout") is True:
        lines.append("Holdout          : passed")
    elif best.get("passed_holdout") is False:
        lines.append("Holdout          : failed")
    top = best.get("suspicion_top")
    if isinstance(top, list) and top:
        preview = ", ".join(
            f"{item.get('node')}@{item.get('line')}" for item in top[:3] if isinstance(item, dict)
        )
        if preview:
            lines.append(f"Suspicion        : {preview}")
    if "speedup_vs_baseline" in best:
        lines.append(
            f"Speedup          : {best.get('speedup_vs_baseline')} (LEGACY cosmetic label, retired)"
        )
    lines.append(f"Early stop       : {'yes' if report.early_stop_triggered else 'no'}")

    history = [
        h
        for h in (report.history or [])
        if isinstance(h, dict) and isinstance(h.get("best_fitness"), (int, float))
    ]
    if history:
        first, last = history[0], history[-1]
        gain = last["best_fitness"] - first["best_fitness"]
        lines.append(
            f"Trend            : {first['best_fitness']} -> {last['best_fitness']} "
            f"(+{gain:.2f})"
        )
        _, plateau = _trend(history)
        if plateau > 0:
            lines.append(f"Plateau          : {plateau} generation(s) without improvement")
        restarts = [
            h["generation"]
            for h in history
            if h.get("stagnation_restart")
        ]
        if restarts:
            lines.append(
                f"Stagnation kicks : {len(restarts)} at gens "
                f"{restarts[:5]}{'...' if len(restarts) > 5 else ''}"
            )
        target = None
        if isinstance(report.config, dict) and report.config.get("early_stop_fitness") is not None:
            try:
                target = float(report.config["early_stop_fitness"])
            except (TypeError, ValueError):
                target = None
        conv = _convergence_gen(history, target) if target is not None else None
        if conv is not None:
            lines.append(f"Target reached   : generation {conv}")
        elif target is not None and isinstance(fitness, (int, float)):
            lines.append(f"Target missed    : best={fitness} target={target}")

        std_last = last.get("std_fitness")
        if isinstance(std_last, (int, float)):
            state = (
                "healthy exploration"
                if std_last > 1.0
                else ("narrowing" if std_last > 0.3 else "collapsed diversity")
            )
            lines.append(f"Diversity (std)  : {std_last:.2f} ({state})")
        share_last = last.get("dominant_species_share")
        if isinstance(share_last, (int, float)):
            lines.append(f"Dominant share   : {share_last * 100:.1f}% of population")

    dist = report.species_distribution or {}
    total_specimens = sum(dist.values())
    lines.append(f"Species          : {len(dist)} kinds / {total_specimens} specimens")
    for name, count in sorted(dist.items(), key=lambda kv: kv[1], reverse=True):
        share = (count / total_specimens * 100) if total_specimens else 0.0
        bar = "#" * round(share / 5)
        lines.append(f"  {name:<32}{count:>4}  {share:5.1f}%  {bar}")

    if dist and total_specimens:
        dominant_count = max(dist.values())
        share = dominant_count / total_specimens * 100
        verdict = (
            "converged (dominant species >= 60%)" if share >= 60 else "diverse"
        )
        lines.append(f"Diversity        : {verdict}")

    if isinstance(fitness, (int, float)):
        gap = 100 - float(fitness)
        lines.append(f"Fitness gap      : {gap:.2f} to perfect score")

    score = report.richness_score
    grade = "rich" if score >= 90 else ("adequate" if score >= 50 else "poor")
    errors = sum(1 for i in report.issues if i.severity == "error")
    warnings = sum(1 for i in report.issues if i.severity == "warning")
    infos = sum(1 for i in report.issues if i.severity == "info")
    lines.append(
        f"Issues           : {errors} error(s), {warnings} warning(s), {infos} info"
    )
    for issue in report.issues:
        lines.append(f"  - {issue}")
    lines.append(f"Richness         : {score}% ({grade})")
    verdict_txt = "PASS" if report.is_valid else "FAIL"
    if report.is_valid and score < 50:
        verdict_txt += " (but not reproducible)"
    target = None
    if isinstance(report.config, dict) and report.config.get("early_stop_fitness") is not None:
        try:
            target = float(report.config["early_stop_fitness"])
        except (TypeError, ValueError):
            target = None
    if isinstance(fitness, (int, float)) and target is not None:
        verdict_txt += " | TARGET_HIT" if float(fitness) >= target else " | TARGET_MISSED"
    if report.early_stop_triggered:
        verdict_txt += " | STOPPED_EARLY"
    lines.append(f"Verdict          : {verdict_txt}")
    return lines
