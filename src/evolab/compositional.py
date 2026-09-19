"""compositional.py — Multi-Hunk & Compositional Expressivity Repair Engine.

Part of Darwin-Evolab Pillar 1: Autonomous Self-Evolution.
Solves the Compositional Depth problem (e.g. COMPOSITIONAL_DEPTH_2) where bugs
require multi-node or multi-line coordinated edits that cannot be resolved by
single-point mutation ascent alone.
"""

from __future__ import annotations

import ast
import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from evolab.repair import (
    RepairEdit,
    RepairGenome,
    _score,
    catalog_sources,
)


@dataclass(frozen=True)
class SuspicionCluster:
    """A spatial or causal cluster of candidate edits on a target file."""
    file: str
    start_line: int
    end_line: int
    edits: tuple[RepairEdit, ...]
    total_suspicion: float = 1.0

    @property
    def span(self) -> int:
        return max(1, self.end_line - self.start_line + 1)


def cluster_edits_by_proximity(
    edits: Sequence[RepairEdit],
    max_line_distance: int = 5,
    suspicion_map: Any | None = None,
) -> list[SuspicionCluster]:
    """Group edits into locality clusters to avoid combinatorial $O(N^2)$ explosion."""
    if not edits:
        return []

    # Sort edits by file and line number
    sorted_edits = sorted(edits, key=lambda e: (e.file, e.lineno, e.col_offset))
    clusters: list[list[RepairEdit]] = []
    current_cluster: list[RepairEdit] = [sorted_edits[0]]

    for edit in sorted_edits[1:]:
        last_edit = current_cluster[-1]
        if edit.file == last_edit.file and abs(edit.lineno - last_edit.lineno) <= max_line_distance:
            current_cluster.append(edit)
        else:
            clusters.append(current_cluster)
            current_cluster = [edit]
    if current_cluster:
        clusters.append(current_cluster)

    result: list[SuspicionCluster] = []
    for c in clusters:
        f = c[0].file
        s_line = min(e.lineno for e in c)
        e_line = max(e.lineno for e in c)
        # Compute cluster suspicion weight if map available
        weight = 1.0
        if suspicion_map is not None:
            line_scores = getattr(suspicion_map, "line_scores", {})
            weight = sum(line_scores.get(ln, 0.1) for ln in range(s_line, e_line + 1))
        result.append(
            SuspicionCluster(
                file=f,
                start_line=s_line,
                end_line=e_line,
                edits=tuple(c),
                total_suspicion=round(weight, 4),
            )
        )

    return sorted(result, key=lambda sc: -sc.total_suspicion)


def generate_dual_hunk_candidates(
    catalog: Sequence[RepairEdit],
    clusters: Sequence[SuspicionCluster] | None = None,
    max_pairs: int = 64,
    lead_edits: Sequence[RepairEdit] | None = None,
) -> list[tuple[RepairEdit, RepairEdit]]:
    """Generate intelligent dual-hunk candidate pairs $(e_1, e_2)$.

    Prioritizes:
      1. Pairs anchored on proven lead edits that raised partial fitness.
      2. Pairs co-located within high-suspicion clusters.
      3. Cross-cluster complementary pairs (e.g. guard + boundary comparison).
    """
    pairs: list[tuple[RepairEdit, RepairEdit]] = []
    seen_pairs: set[tuple[tuple, tuple]] = set()

    def _add_pair(e1: RepairEdit, e2: RepairEdit) -> None:
        if e1.key() == e2.key():
            return
        p_key = tuple(sorted([e1.key(), e2.key()]))
        if p_key not in seen_pairs:
            seen_pairs.add(p_key)
            pairs.append((e1, e2))

    # Strategy 1: Lead edit anchoring (highest priority)
    if lead_edits:
        for lead in lead_edits:
            for cand in catalog:
                if len(pairs) >= max_pairs:
                    return pairs
                _add_pair(lead, cand)

    # Strategy 2: Intra-cluster pairing
    if clusters:
        for cl in clusters:
            cl_edits = cl.edits
            for i in range(len(cl_edits)):
                for j in range(i + 1, len(cl_edits)):
                    if len(pairs) >= max_pairs:
                        return pairs
                    _add_pair(cl_edits[i], cl_edits[j])

    # Strategy 3: Global catalog complementary pairing fallback
    for i in range(min(len(catalog), 16)):
        for j in range(i + 1, min(len(catalog), 16)):
            if len(pairs) >= max_pairs:
                return pairs
            _add_pair(catalog[i], catalog[j])

    return pairs


def compositional_repair(
    sources: dict[str, str],
    target_file: str,
    evaluator: Any,
    max_evals: int = 64,
    beam_width: int = 3,
    lookahead_depth: int = 2,
    prioritize_by_suspicion: bool = True,
    candidate_ranker: Callable[[list[RepairEdit]], list[RepairEdit]] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[RepairGenome, list[dict[str, Any]], int]:
    """Execute Multi-Hunk & Compositional AST repair over hard bug plateaus.

    Seamlessly transitions from single-point gradient ascent to dual-hunk lookahead
    beam search when single edits hit a sub-optimal ceiling.
    """
    catalog = catalog_sources(sources)

    if prioritize_by_suspicion:
        suspicion_map = getattr(evaluator, "last_suspicion_map", None)
        if suspicion_map is not None and getattr(suspicion_map, "line_scores", None):
            def _sbfl_key(e: RepairEdit):
                score = suspicion_map.line_scores.get(e.lineno, 0.0)
                return (-score, e.file, e.lineno, e.col_offset, e.kind)
            catalog = sorted(catalog, key=_sbfl_key)

    if candidate_ranker is not None:
        try:
            ranked = candidate_ranker(list(catalog))
            if isinstance(ranked, list) and ranked:
                catalog = ranked
        except Exception:
            pass

    current = RepairGenome(sources=dict(sources), target_file=target_file, edits=[])
    best_score, best_hold = _score(evaluator, current)
    evaluations = 1
    history: list[dict[str, Any]] = [
        {"iteration": 0, "edits": [], "score": best_score, "holdout": best_hold, "stage": "baseline"}
    ]

    if best_score >= 99.7 and best_hold is not False:
        return current, history, evaluations

    # -------------------------------------------------------------------------
    # Phase 1: Single-Hunk Fast Gradient Scan
    # -------------------------------------------------------------------------
    partial_successes: list[tuple[float, RepairEdit]] = []

    for edit in catalog:
        if evaluations >= max_evals:
            break
        trial = current.clone()
        trial.edits.append(edit)
        s, h = _score(evaluator, trial)
        evaluations += 1

        # Direct single-hunk resolution!
        if s >= 99.7 and h is not False:
            current = trial
            best_score, best_hold = s, h
            history.append({
                "iteration": len(current.edits),
                "edits": [e.to_dict() for e in current.edits],
                "score": best_score,
                "holdout": best_hold,
                "stage": "single_hunk_direct",
            })
            if on_step:
                on_step(history[-1])
            return current, history, evaluations

        # Track candidate edits that raised score or broke baseline tie
        if s > best_score and h is not False:
            partial_successes.append((s, edit))

    # Sort partials by descending fitness
    partial_successes.sort(key=lambda x: -x[0])

    if partial_successes:
        # Commit top partial improvement
        top_score, top_edit = partial_successes[0]
        current.edits.append(top_edit)
        best_score = top_score
        history.append({
            "iteration": 1,
            "edits": [e.to_dict() for e in current.edits],
            "score": best_score,
            "holdout": True,
            "stage": "single_hunk_partial",
        })
        if on_step:
            on_step(history[-1])

    # -------------------------------------------------------------------------
    # Phase 2: Multi-Hunk / Dual-Hunk Compositional Lookahead
    # -------------------------------------------------------------------------
    if best_score < 99.7 and lookahead_depth >= 2 and evaluations < max_evals:
        # Extract clusters based on target file and suspicion map
        susp_map = getattr(evaluator, "last_suspicion_map", None)
        clusters = cluster_edits_by_proximity(catalog, max_line_distance=6, suspicion_map=susp_map)

        # Generate dual pairs anchored on partial successes or top clusters
        lead_pool = [e for _, e in partial_successes[:beam_width]] if partial_successes else [catalog[0]] if catalog else []
        remaining_budget = max_evals - evaluations
        dual_pairs = generate_dual_hunk_candidates(
            catalog=catalog,
            clusters=clusters,
            max_pairs=min(remaining_budget, 48),
            lead_edits=lead_pool,
        )

        best_dual_genome: RepairGenome | None = None
        best_dual_score = best_score
        best_dual_hold = best_hold

        for e1, e2 in dual_pairs:
            if evaluations >= max_evals:
                break

            dual_trial = RepairGenome(sources=dict(sources), target_file=target_file, edits=[e1, e2])
            s, h = _score(evaluator, dual_trial)
            evaluations += 1

            # Check for breakthrough
            if s > best_dual_score and h is not False:
                best_dual_score = s
                best_dual_hold = h
                best_dual_genome = dual_trial

                if s >= 99.7:
                    # Instant resolution via dual-hunk composition!
                    current = dual_trial
                    best_score = s
                    best_hold = h
                    history.append({
                        "iteration": 2,
                        "edits": [e.to_dict() for e in current.edits],
                        "score": best_score,
                        "holdout": best_hold,
                        "stage": "dual_hunk_breakthrough",
                    })
                    if on_step:
                        on_step(history[-1])
                    return current, history, evaluations

        if best_dual_genome is not None and best_dual_score > best_score:
            current = best_dual_genome
            best_score = best_dual_score
            best_hold = best_dual_hold
            history.append({
                "iteration": len(current.edits),
                "edits": [e.to_dict() for e in current.edits],
                "score": best_score,
                "holdout": best_hold,
                "stage": "dual_hunk_ascent",
            })
            if on_step:
                on_step(history[-1])

    return current, history, evaluations
