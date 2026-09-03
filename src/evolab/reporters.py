"""Production-grade output reporters: Terminal diagnostics, GitHub Markdown, and Git Patch formatters."""
from __future__ import annotations

import datetime
import difflib
from pathlib import Path
from typing import Any


def format_terminal_diagnostics(
    result: dict[str, Any],
    scenario: Any = None,
    baseline_fitness: float | None = None,
    baseline_failures: list[str] | None = None,
) -> str:
    """Format an informative, developer-first terminal diagnostics summary."""
    bi = result.get("best_individual") or {}
    final_score = float(bi.get("fitness", 0.0))
    passed_holdout = bi.get("passed_holdout")

    is_electronics = (
        result.get("config", {}).get("genome") == "electronics"
        or bi.get("species") == "spec_electronics"
    )
    if is_electronics:
        lines = [
            "",
            "======================== ELECTRONICS SYNTHESIS REPORT ========================",
            f"Scenario Name   : {result.get('config', {}).get('scenario', 'electronics')}",
            f"Search Engine   : {result.get('config', {}).get('search', 'GA').upper()}",
            f"Final Fitness   : {final_score:6.2f}%",
        ]
        artifacts = bi.get("artifacts") or {}
        if "functional_accuracy" in artifacts:
            lines.append(f"Logic Accuracy  : {artifacts['functional_accuracy'] * 100.0:5.1f}%")
        if "nominal_delay_ns" in artifacts:
            lines.append(f"Nominal Delay   : {artifacts['nominal_delay_ns']} ns")
        if "quiescent_icc_ua" in artifacts:
            lines.append(f"Quiescent Power : {artifacts['quiescent_icc_ua']} uA")
        if "ic_count" in artifacts:
            lines.append(f"Components Used : {artifacts['ic_count']} ICs | Wires: {artifacts.get('wire_count', 'n/a')}")
        lines.append("--------------------------------------------------------------------------------")
        status_str = "SUCCESS" if final_score >= 80.0 else "OPTIMIZING"
        lines.append(f"Outcome Status  : {status_str}")
        lines.append(f"Total Evaluated : {result.get('total_candidates_evaluated', 'n/a')} circuits")
        lines.append(f"Generations Run : {result.get('total_generations', 'n/a')}")
        lines.append("================================================================================")
        return "\n".join(lines)

    lines = [
        "",
        "============================== REPAIR DIAGNOSTICS ==============================",
    ]

    if scenario is not None:
        lines.append(f"Target File     : {getattr(scenario, 'target_file', 'n/a')}")
        lines.append(f"Target Function : {getattr(scenario, 'func_name', 'n/a')}")

    if baseline_fitness is not None:
        lines.append(f"Baseline Score  : {baseline_fitness:6.2f}%")
        if baseline_failures:
            lines.append("Baseline Faults :")
            for f in baseline_failures[:3]:
                lines.append(f"  - [FAIL] {f}")
            if len(baseline_failures) > 3:
                lines.append(f"  ... and {len(baseline_failures) - 3} more failure(s)")

    lines.append("--------------------------------------------------------------------------------")
    status_str = "SUCCESS" if final_score >= 99.7 and passed_holdout is not False else "INCOMPLETE"
    lines.append(f"Final Status    : {status_str} (Fitness: {final_score:6.2f}%)")

    holdout_str = "PASSED (100% generalization, zero overfitting)" if passed_holdout is True else (
        "FAILED (overfitted to test suite)" if passed_holdout is False else "N/A"
    )
    lines.append(f"Holdout Gate    : {holdout_str}")
    lines.append(f"Total Evaluated : {result.get('total_candidates_evaluated', 'n/a')} candidates")
    lines.append(f"Generations Run : {result.get('total_generations', len(result.get('history', [])))}")
    lines.append("================================================================================")
    return "\n".join(lines)


def format_markdown_summary(
    result: dict[str, Any],
    scenario: Any = None,
    diff_text: str = "",
) -> str:
    """Format a rich Markdown summary report ready for GitHub Actions / Pull Request comments."""
    bi = result.get("best_individual") or {}
    fitness = float(bi.get("fitness", 0.0))
    passed_holdout = bi.get("passed_holdout")

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    is_electronics = (
        result.get("config", {}).get("genome") == "electronics"
        or bi.get("species") == "spec_electronics"
    )
    if is_electronics:
        scenario_name = result.get("config", {}).get("scenario", "Circuit Synthesis")
        status_badge = "✅ **PASSED**" if fitness >= 80.0 else "⚠️ **INCOMPLETE**"
        md_lines = [
            f"## ⚡ Darwin-Evolab Circuit Synthesis Report: `{scenario_name}`",
            "",
            f"> Generated automatically at `{now_iso}`",
            "",
            "| Metric | Value / Status |",
            "| :--- | :--- |",
            f"| **Domain** | Hardware & Circuit Synthesis |",
            f"| **Scenario** | `{scenario_name}` |",
            f"| **Synthesis Outcome** | {status_badge} ({fitness:.2f}% score) |",
            f"| **Search Budget** | {result.get('total_candidates_evaluated', 0)} candidates across {result.get('total_generations', 1)} generations |",
            "",
        ]
        artifacts = bi.get("artifacts") or {}
        if artifacts:
            md_lines.extend([
                "### 🔬 Circuit Measured Characteristics",
                "",
                "| Metric / Pin | Value |",
                "| :--- | :--- |",
            ])
            for k, v in sorted(artifacts.items()):
                if isinstance(v, (int, float, str, list)):
                    md_lines.append(f"| **{k}** | `{v}` |")
            md_lines.append("")

        if diff_text and diff_text.strip() != "(no textual diff)":
            md_lines.extend([
                "### 📐 Synthesized Topology / Netlist Connections",
                "",
                "```spice",
                diff_text.strip(),
                "```",
                "",
            ])
        return "\n".join(md_lines) + "\n"

    status_badge = "✅ **PASSED**" if fitness >= 99.7 and passed_holdout is not False else "⚠️ **INCOMPLETE**"
    holdout_badge = "🛡️ **PASSED**" if passed_holdout is True else ("❌ **FAILED**" if passed_holdout is False else "➖ **NONE**")

    target_file = getattr(scenario, "target_file", "source file") if scenario else "source"
    func_name = getattr(scenario, "func_name", "target function") if scenario else "function"

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    md_lines = [
        "## 🧬 Darwin-Evolab Automated Repair Report",
        "",
        f"> Generated automatically at `{now_iso}`",
        "",
        "| Metric | Value / Status |",
        "| :--- | :--- |",
        f"| **Target Scope** | `{target_file}` :: `{func_name}()` |",
        f"| **Repair Outcome** | {status_badge} ({fitness:.1f}% fitness) |",
        f"| **Holdout Verification** | {holdout_badge} (Overfitting Protection) |",
        f"| **Search Budget** | {result.get('total_candidates_evaluated', 0)} candidates across {result.get('total_generations', 1)} generations |",
        "",
    ]

    history = result.get("history") or []
    if history:
        md_lines.extend([
            "### 📈 Search Trajectory",
            "",
            "| Gen | Fitness | Edits Applied | Strategy |",
            "| :---: | :---: | :---: | :--- |",
        ])
        for h in history:
            added = h.get("added", "baseline")
            md_lines.append(f"| {h.get('generation', 1)} | {h.get('best_fitness', 0.0):.1f}% | {h.get('edits', 0)} | `{added}` |")
        md_lines.append("")

    if diff_text and diff_text.strip() != "(no textual diff)":
        md_lines.extend([
            "### 📝 Proposed Patch",
            "",
            "```diff",
            diff_text.strip(),
            "```",
            "",
            "> 💡 Apply locally using: `git apply fix.patch`",
        ])
    else:
        md_lines.append("> ℹ️ No changes were required or generated.")

    return "\n".join(md_lines) + "\n"


def format_git_patch(
    scenario: Any,
    repaired_sources: dict[str, str],
    commit_msg: str = "fix: automated repair by darwin-evolab",
) -> str:
    """Generate a standard unified diff patch compatible with `git apply`."""
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    header = [
        f"From: darwin-evolab <repair@evolab.local>",
        f"Date: {now_str}",
        f"Subject: [PATCH] {commit_msg}",
        "",
        "---",
    ]

    diff_chunks = []
    total_files = 0
    orig_sources = getattr(scenario, "sources", {}) if scenario else {}

    for path, orig_code in orig_sources.items():
        repaired_code = repaired_sources.get(path, orig_code)
        old_lines = orig_code.splitlines(keepends=True)
        new_lines = repaired_code.splitlines(keepends=True)
        if old_lines == new_lines:
            continue
        total_files += 1
        chunk = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        ))
        if chunk:
            diff_chunks.extend(chunk)

    if not diff_chunks:
        return ""

    stat_line = f" {total_files} file(s) changed\n"
    header.append(stat_line)
    return "\n".join(header) + "\n" + "".join(diff_chunks)


def apply_in_place(
    scenario: Any,
    repaired_sources: dict[str, str],
    create_backup: bool = True,
    file_mapping: dict[str, Path | str] | None = None,
) -> list[str]:
    """Write repaired sources directly to disk in-place, optionally creating .bak backups."""
    modified_paths: list[str] = []
    orig_sources = getattr(scenario, "sources", {}) if scenario else {}

    for path, orig_code in orig_sources.items():
        new_code = repaired_sources.get(path)
        if new_code is None or new_code == orig_code:
            continue

        target_p = None
        if file_mapping and path in file_mapping:
            target_p = Path(file_mapping[path])
        elif hasattr(scenario, "source_paths") and path in getattr(scenario, "source_paths", {}):
            target_p = Path(scenario.source_paths[path])
        elif Path(path).exists():
            target_p = Path(path)

        if target_p and target_p.exists():
            if create_backup:
                bak_path = target_p.with_suffix(target_p.suffix + ".bak")
                bak_path.write_text(target_p.read_text(encoding="utf-8"), encoding="utf-8")
            target_p.write_text(new_code, encoding="utf-8")
            modified_paths.append(str(target_p))

    return modified_paths
