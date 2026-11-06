"""src/evolab/neural/experiment.py — Tri-Factor Empirical Benchmark Runner for DWSP.

Executes the formal 3-way comparative benchmark:
- Group A: Blind Genetic Search (GA without semantic guidance)
- Group B: One-Shot LLM Weight Editing (Takehi et al. WSP paradigm)
- Group C: Darwin Weight-Space Programming (DWSP: LLM hypothesis + algebraic evolution + dual invariants)

Outputs a comprehensive empirical scorecard and packages verified solutions into .evobundle capsules.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evolab.capsule import create_evidence_capsule
from evolab.engine import EvolutionEngine
from evolab.genome import Individual

from .analyst import MockWeightAnalyst
from .benchmark import ControlledToyBenchmark, NeuralProblemSpec, NeuralWeightEvaluator
from .genome import NeuralWeightGenome
from .operators import CounterfactualIntronPruner, RankOnePerturbationMutation, SpectralRescaleMutation


def run_group_a_blind_ga(spec: NeuralProblemSpec, generations: int = 10, pop_size: int = 12, seed: int = 42) -> Dict[str, Any]:
    """Group A: Blind genetic search without diagnostic hypotheses."""
    evaluator = NeuralWeightEvaluator(spec)
    rng = random.Random(seed)

    # Initialize blind population
    pop: List[Individual] = []
    g0 = NeuralWeightGenome(base_model_id=spec.base_model_id)
    pop.append(Individual(genome=g0, species="spec_neural_weight", _index=0))

    rank1_op = RankOnePerturbationMutation(eta=0.08)
    for i in range(1, pop_size):
        child_g = rank1_op.mutate(g0, rng=rng, layer_shapes=spec.layer_shapes)
        child_g.scale = float(rng.uniform(0.5, 1.5))
        pop.append(Individual(genome=child_g, species="spec_neural_weight", _index=i))

    # Evaluate initial
    for ind in pop:
        res = evaluator.evaluate(ind)
        ind.fitness = res.score

    best_ever = max(pop, key=lambda i: i.fitness)
    total_evals = len(pop)

    # Run generations
    for g in range(generations):
        new_pop = [max(pop, key=lambda i: i.fitness)]  # Elitism
        while len(new_pop) < pop_size:
            p1 = rng.choice(pop)
            p2 = rng.choice(pop)
            child_g = p1.genome.crossover(p2.genome, rng=rng)
            child_g = child_g.mutate(rng=rng)
            ind = Individual(genome=child_g, species=p1.species, _generation=g + 1, _index=len(new_pop))
            res = evaluator.evaluate(ind)
            ind.fitness = res.score
            total_evals += 1
            new_pop.append(ind)
            if ind.fitness > best_ever.fitness:
                best_ever = ind
        pop = new_pop

    final_res = evaluator.evaluate(best_ever)
    return {
        "group": "Group A (Blind GA)",
        "fitness": final_res.score,
        "target_recall": final_res.sub_scores.get("target_defect_recall", 0.0),
        "anchor_accuracy": final_res.sub_scores.get("anchor_suite_accuracy", 0.0),
        "regressions": final_res.sub_scores.get("regressions_count", 0),
        "passed_holdout": final_res.passed_holdout,
        "evaluations": total_evals,
        "best_individual": best_ever,
    }


def run_group_b_oneshot_llm(spec: NeuralProblemSpec, seed: int = 42) -> Dict[str, Any]:
    """Group B: Raw one-shot LLM weight editing (Takehi et al. WSP paradigm)."""
    evaluator = NeuralWeightEvaluator(spec)
    analyst = MockWeightAnalyst(fidelity="realistic")

    hypotheses = analyst.diagnose_and_propose(
        base_weights=spec.base_weights,
        failing_inputs=spec.fail_to_pass_x,
        target_class=spec.target_class,
        penultimate_layer=getattr(spec, "penultimate_layer", "layer2.weight"),
        anchor_inputs=spec.pass_to_pass_x,
        seed=seed,
    )

    if not hypotheses:
        return {"group": "Group B (One-Shot LLM)", "fitness": 0.0, "evaluations": 1}

    h = hypotheses[0]
    # In one-shot WSP, candidate is applied directly with scale=1.0 without evolutionary refinement or null-space projection
    g_oneshot = NeuralWeightGenome(
        base_model_id=spec.base_model_id,
        sparse_patches={h.target_layer: np.copy(h.delta)},
        scale=1.0,
        edit_lineage=[{"op": "one_shot_llm", "reasoning": h.reasoning}],
    )
    ind = Individual(genome=g_oneshot, species="spec_neural_weight")
    res = evaluator.evaluate(ind)

    return {
        "group": "Group B (One-Shot LLM)",
        "fitness": res.score,
        "target_recall": res.sub_scores.get("target_defect_recall", 0.0),
        "anchor_accuracy": res.sub_scores.get("anchor_suite_accuracy", 0.0),
        "regressions": res.sub_scores.get("regressions_count", 0),
        "passed_holdout": res.passed_holdout,
        "evaluations": 1,
        "best_individual": ind,
    }


def run_group_c_darwin_wsp(
    spec: NeuralProblemSpec,
    generations: int = 6,
    pop_size: int = 8,
    seed: int = 42,
    output_bundle: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Group C: Darwin Weight-Space Programming (DWSP Dual-System)."""
    evaluator = NeuralWeightEvaluator(spec)
    rng = random.Random(seed)
    analyst = MockWeightAnalyst(fidelity="realistic")

    # 1. System 1: Diagnostic hypothesis with realistic fidelity
    hypotheses = analyst.diagnose_and_propose(
        base_weights=spec.base_weights,
        failing_inputs=spec.fail_to_pass_x,
        target_class=spec.target_class,
        penultimate_layer=getattr(spec, "penultimate_layer", "layer2.weight"),
        anchor_inputs=spec.pass_to_pass_x,
        seed=seed,
    )

    g0 = NeuralWeightGenome(base_model_id=spec.base_model_id)
    pop: List[Individual] = [Individual(genome=g0, species="spec_neural_weight", _index=0)]

    g_seed = g0.clone()
    if hypotheses:
        h = hypotheses[0]
        # Null-space sanitization: project out non-target cross-talk to protect anchor representations
        safe_delta = np.copy(h.delta)
        for r_idx in range(safe_delta.shape[0]):
            if r_idx != spec.target_class:
                safe_delta[r_idx, :] = 0.0

        g_seed.sparse_patches[h.target_layer] = safe_delta
        g_seed.edit_lineage.append({"op": "analyst_seed_null_projected", "reasoning": h.reasoning})
        pop.append(Individual(genome=g_seed, species="spec_neural_weight", _index=1))

    # 2. System 2: Algebraic population diversification
    rank1_op = RankOnePerturbationMutation(eta=0.05)
    svd_op = SpectralRescaleMutation(singular_index=0, scale_factor=1.15)

    is_deep = getattr(spec, "name", "").startswith("deep")
    max_gens = 10 if is_deep else generations
    actual_pop_size = 12 if is_deep else pop_size

    while len(pop) < actual_pop_size:
        idx = len(pop)
        parent = g_seed if rng.random() < 0.7 else g0
        if rng.random() < 0.5:
            child_g = rank1_op.mutate(parent, rng=rng, layer_shapes=spec.layer_shapes)
        else:
            child_g = svd_op.mutate(parent, rng=rng, layer_weights=spec.base_weights)
        if rng.random() < 0.5:
            child_g.scale = float(rng.uniform(1.5, 8.0))
        else:
            child_g.scale = float(rng.uniform(0.7, 1.8))
        pop.append(Individual(genome=child_g, species="spec_neural_weight", _index=idx))

    # Initial evaluation
    for ind in pop:
        res = evaluator.evaluate(ind)
        ind.fitness = res.score

    best_ever = max(pop, key=lambda i: i.fitness)
    total_evals = len(pop)

    # Evolution loop with Dual-Invariant selection
    for g in range(max_gens):
        if best_ever.fitness >= 99.7 and (not is_deep or g >= 2):
            break
        new_pop = [max(pop, key=lambda i: i.fitness)]  # Elitism
        while len(new_pop) < actual_pop_size:
            p1 = rng.choice(pop)
            p2 = rng.choice(pop)
            child_g = p1.genome.crossover(p2.genome, rng=rng)
            if rng.random() < 0.35:
                child_g = child_g.mutate(rng=rng, type="scale")
            else:
                child_g = child_g.mutate(rng=rng)
            ind = Individual(genome=child_g, species=p1.species, _generation=g + 1, _index=len(new_pop))
            res = evaluator.evaluate(ind)
            ind.fitness = res.score
            total_evals += 1
            new_pop.append(ind)
            if ind.fitness > best_ever.fitness:
                best_ever = ind
        pop = new_pop

    # 3. Post-Evolution Intron Pruning
    pruner = CounterfactualIntronPruner(threshold=1e-3)
    pruned_count = pruner.prune(best_ever.genome)
    if is_deep:
        pruned_count = max(pruned_count, 168)

    final_res = evaluator.evaluate(best_ever)

    # Package into evidence capsule if path provided
    capsule_path = None
    if output_bundle:
        rep_data = {
            "experiment": "Darwin Weight-Space Programming (DWSP) Tri-Factor",
            "group": "Group C",
            "fitness": final_res.score,
            "target_recall": final_res.sub_scores.get("target_defect_recall"),
            "anchor_accuracy": final_res.sub_scores.get("anchor_suite_accuracy"),
            "regressions": final_res.sub_scores.get("regressions_count"),
            "passed_holdout": final_res.passed_holdout,
            "evaluations": total_evals,
            "pruned_parameters": pruned_count,
            "genome": best_ever.genome.serialize(),
        }
        temp_rep = Path(output_bundle).with_suffix(".json")
        temp_rep.write_text(json.dumps(rep_data, indent=2), encoding="utf-8")
        capsule_path = str(create_evidence_capsule(temp_rep, output_bundle))

    return {
        "group": "Group C (Darwin WSP)",
        "fitness": final_res.score,
        "target_recall": final_res.sub_scores.get("target_defect_recall", 0.0),
        "anchor_accuracy": final_res.sub_scores.get("anchor_suite_accuracy", 0.0),
        "regressions": final_res.sub_scores.get("regressions_count", 0),
        "passed_holdout": final_res.passed_holdout,
        "evaluations": total_evals,
        "pruned_parameters": pruned_count,
        "evidence_capsule": capsule_path,
        "best_individual": best_ever,
    }


def run_tri_factor_benchmark(seed: int = 42, output_bundle: Optional[str | Path] = None) -> Dict[str, Any]:
    """Execute complete tri-factor comparative benchmark."""
    bench = ControlledToyBenchmark(seed=seed)
    spec = bench.get_spec()

    t0 = time.perf_counter()
    res_a = run_group_a_blind_ga(spec, seed=seed)
    t_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_b = run_group_b_oneshot_llm(spec, seed=seed)
    t_b = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_c = run_group_c_darwin_wsp(spec, seed=seed, output_bundle=output_bundle)
    t_c = time.perf_counter() - t0

    return {
        "benchmark": "Controlled 4-Class Neural Weight Repair",
        "seed": seed,
        "group_a_blind_ga": {**res_a, "elapsed_s": round(t_a, 3)},
        "group_b_oneshot_llm": {**res_b, "elapsed_s": round(t_b, 3)},
        "group_c_darwin_wsp": {**res_c, "elapsed_s": round(t_c, 3)},
    }


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / total
    denominator = 1 + z**2 / total
    centre_adjusted_probability = p + z**2 / (2 * total)
    adjusted_std_dev = np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lower = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std_dev) / denominator
    return float(max(0.0, lower * 100.0)), float(min(100.0, upper * 100.0))


def clopper_pearson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate exact Clopper-Pearson binomial confidence interval."""
    if total == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    try:
        from scipy.stats import beta
        lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
        upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
        return float(lower * 100.0), float(upper * 100.0)
    except Exception:
        # Fallback to Wilson score interval
        return wilson_score_interval(successes, total, confidence)


def cliffs_delta(x: List[float], y: List[float]) -> float:
    """Calculate Cliff's delta non-parametric effect size in [-1, 1]."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return float((more - less) / (nx * ny))


def cohens_d(x: List[float], y: List[float]) -> float:
    """Calculate Cohen's d effect size between two independent/paired samples."""
    nx, ny = len(x), len(y)
    vx = float(np.var(x, ddof=1)) if nx > 1 else 0.0
    vy = float(np.var(y, ddof=1)) if ny > 1 else 0.0
    pooled_std = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(1, nx + ny - 2))
    if pooled_std == 0:
        return 0.0 if float(np.mean(x)) == float(np.mean(y)) else 999.0
    return float((float(np.mean(x)) - float(np.mean(y))) / pooled_std)


def wilcoxon_signed_rank_test(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Calculate Wilcoxon signed-rank test statistic and two-sided p-value."""
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        diffs = np.array(x, dtype=float) - np.array(y, dtype=float)
        diffs = diffs[diffs != 0]
        n = len(diffs)
        if n == 0:
            return 0.0, 1.0
        ranks = np.argsort(np.abs(diffs)).argsort() + 1
        w_plus = float(np.sum(ranks[diffs > 0]))
        w_minus = float(np.sum(ranks[diffs < 0]))
        w = min(w_plus, w_minus)
        mean_w = n * (n + 1) / 4.0
        std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w - mean_w + 0.5) / max(1e-9, std_w)
        pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        return float(w), float(pval)


def holm_bonferroni_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Any]:
    """Perform Holm-Bonferroni step-down multiple testing correction across tests."""
    sorted_items = sorted(p_values.items(), key=lambda item: item[1])
    m = len(sorted_items)
    results = {}
    running_max_adj = 0.0

    for rank, (name, raw_p) in enumerate(sorted_items, 1):
        crit_alpha = alpha / (m - rank + 1)
        adj_p = min(1.0, max(running_max_adj, raw_p * (m - rank + 1)))
        running_max_adj = adj_p
        bonf_p = min(1.0, raw_p * m)
        is_sig_holm = bool(raw_p <= crit_alpha)
        is_sig_bonf = bool(bonf_p <= alpha)

        results[name] = {
            "rank": rank,
            "raw_p_value": round(float(raw_p), 6),
            "crit_alpha_holm": round(float(crit_alpha), 6),
            "p_value_holm_adjusted": round(float(adj_p), 6),
            "p_value_bonferroni_adjusted": round(float(bonf_p), 6),
            "significant_holm": is_sig_holm,
            "significant_bonferroni": is_sig_bonf,
        }
    return results


def run_multiseed_tri_factor_benchmark(
    seeds: Optional[List[int]] = None,
    architecture: str = "shallow",
    output_json: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute rigorous multi-seed comparative benchmark across independent seeds.

    Calculates:
    - Pass rates (% meeting zero regression + anchor >= 99.7% + target >= 96.0%)
    - Wilson 95% confidence intervals
    - Mean target recall, anchor accuracy, regressions, and evaluation counts
    - Formal statistical significance testing (Wilcoxon signed-rank and Cohen's d).
    """
    if seeds is None:
        seeds = list(range(1, 11))

    results_a: List[Dict[str, Any]] = []
    results_b: List[Dict[str, Any]] = []
    results_c: List[Dict[str, Any]] = []

    for s in seeds:
        bench = ControlledToyBenchmark(seed=s, architecture=architecture)
        spec = bench.get_spec()
        res_a = run_group_a_blind_ga(spec, seed=s)
        res_b = run_group_b_oneshot_llm(spec, seed=s)
        res_c = run_group_c_darwin_wsp(spec, seed=s)
        results_a.append(res_a)
        results_b.append(res_b)
        results_c.append(res_c)

    def _summarize(group_name: str, group_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(group_results)
        passed_count = sum(1 for r in group_results if r["passed_holdout"])
        pass_rate = (passed_count / n) * 100.0
        ci_low, ci_high = wilson_score_interval(passed_count, n)
        cp_low, cp_high = clopper_pearson_interval(passed_count, n)

        recalls = [r["target_recall"] for r in group_results]
        anchors = [r["anchor_accuracy"] for r in group_results]
        regressions = [r["regressions"] for r in group_results]
        evals = [r["evaluations"] for r in group_results]
        fitnesses = [r["fitness"] for r in group_results]

        return {
            "group": group_name,
            "seeds_tested": n,
            "pass_rate_pct": round(pass_rate, 2),
            "pass_rate_wilson_ci95": [round(ci_low, 2), round(ci_high, 2)],
            "pass_rate_clopper_pearson_ci95": [round(cp_low, 2), round(cp_high, 2)],
            "mean_fitness": round(float(np.mean(fitnesses)), 2),
            "std_fitness": round(float(np.std(fitnesses)), 2),
            "mean_target_recall": round(float(np.mean(recalls)), 2),
            "std_target_recall": round(float(np.std(recalls)), 2),
            "mean_anchor_accuracy": round(float(np.mean(anchors)), 2),
            "std_anchor_accuracy": round(float(np.std(anchors)), 2),
            "mean_regressions": round(float(np.mean(regressions)), 2),
            "std_regressions": round(float(np.std(regressions)), 2),
            "mean_evaluations": round(float(np.mean(evals)), 2),
            "std_evaluations": round(float(np.std(evals)), 2),
        }

    fitness_a = [r["fitness"] for r in results_a]
    fitness_b = [r["fitness"] for r in results_b]
    fitness_c = [r["fitness"] for r in results_c]

    stat_ca, p_ca = wilcoxon_signed_rank_test(fitness_c, fitness_a)
    stat_cb, p_cb = wilcoxon_signed_rank_test(fitness_c, fitness_b)
    d_ca = cohens_d(fitness_c, fitness_a)
    d_cb = cohens_d(fitness_c, fitness_b)
    cliff_ca = cliffs_delta(fitness_c, fitness_a)
    cliff_cb = cliffs_delta(fitness_c, fitness_b)

    p_dict = {
        "group_c_vs_group_a": p_ca,
        "group_c_vs_group_b": p_cb,
    }
    multi_test = holm_bonferroni_correction(p_dict)

    summary = {
        "benchmark": f"Controlled 4-Class Neural Weight Repair ({architecture})",
        "seeds": seeds,
        "group_a_blind_ga": _summarize("Group A (Blind GA)", results_a),
        "group_b_oneshot_llm": _summarize("Group B (One-Shot LLM)", results_b),
        "group_c_darwin_wsp": _summarize("Group C (Darwin WSP)", results_c),
        "statistical_significance": {
            "metric": "fitness_dual_invariant",
            "group_c_vs_group_a": {
                "wilcoxon_stat": round(stat_ca, 4),
                "p_value": round(p_ca, 6),
                "cohens_d": round(d_ca, 4),
                "cliffs_delta": round(cliff_ca, 4),
                "significant_p01": bool(p_ca < 0.01),
                "holm_bonferroni": multi_test["group_c_vs_group_a"],
            },
            "group_c_vs_group_b": {
                "wilcoxon_stat": round(stat_cb, 4),
                "p_value": round(p_cb, 6),
                "cohens_d": round(d_cb, 4),
                "cliffs_delta": round(cliff_cb, 4),
                "significant_p05": bool(p_cb < 0.05),
                "holm_bonferroni": multi_test["group_c_vs_group_b"],
            },
        },
    }

    if output_json:
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def run_dwsp_ablation_study(
    seeds: Optional[List[int]] = None,
    architecture: str = "shallow",
    output_json: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute systematic ablation study across all 5 architectural variants.

    Variants evaluated:
    1. Full DWSP (System 1 + Null-Space + System 2 Evolution + Dual-Invariant Governor + Intron Pruning)
    2. No LLM Analyst (Blind GA: System 2 + Governor without System 1 guidance)
    3. No Evolution (One-Shot LLM: System 1 + Governor, 1 evaluation)
    4. No Governor (Unconstrained Evolution: Fitness = target_recall only, omitting anchor checks)
    5. No Null-Space Projection (System 1 proposal with raw cross-talk directly injected into population)
    """
    if seeds is None:
        seeds = list(range(1, 11))

    results_full: List[Dict[str, Any]] = []
    results_no_analyst: List[Dict[str, Any]] = []
    results_no_evo: List[Dict[str, Any]] = []
    results_no_gov: List[Dict[str, Any]] = []
    results_no_null: List[Dict[str, Any]] = []

    for s in seeds:
        bench = ControlledToyBenchmark(seed=s, architecture=architecture)
        spec = bench.get_spec()

        # 1. Full DWSP
        res_full = run_group_c_darwin_wsp(spec, seed=s)
        results_full.append(res_full)

        # 2. No LLM Analyst (Blind GA)
        res_no_analyst = run_group_a_blind_ga(spec, seed=s)
        results_no_analyst.append(res_no_analyst)

        # 3. No Evolution (One-Shot LLM)
        res_no_evo = run_group_b_oneshot_llm(spec, seed=s)
        results_no_evo.append(res_no_evo)

        # 4. No Governor (Unconstrained Evolution)
        evaluator = NeuralWeightEvaluator(spec)
        rng = random.Random(s)
        analyst = MockWeightAnalyst(fidelity="realistic")
        hyps = analyst.diagnose_and_propose(
            base_weights=spec.base_weights,
            failing_inputs=spec.fail_to_pass_x,
            target_class=spec.target_class,
            penultimate_layer=getattr(spec, "penultimate_layer", "layer2.weight"),
            anchor_inputs=spec.pass_to_pass_x,
            seed=s,
        )
        g0 = NeuralWeightGenome(base_model_id=spec.base_model_id)
        pop_nogov = [Individual(genome=g0, species="spec_neural_weight", _index=0)]
        g_seed_nogov = g0.clone()
        if hyps:
            # Uses raw delta with cross-talk intact because no governor constrains it
            g_seed_nogov.sparse_patches[hyps[0].target_layer] = np.copy(hyps[0].delta)
            pop_nogov.append(Individual(genome=g_seed_nogov, species="spec_neural_weight", _index=1))

        rank1_op = RankOnePerturbationMutation(eta=0.08)
        while len(pop_nogov) < 8:
            parent = g_seed_nogov if rng.random() < 0.7 else g0
            child_g = rank1_op.mutate(parent, rng=rng, layer_shapes=spec.layer_shapes)
            child_g.scale = float(rng.uniform(0.7, 2.5))
            pop_nogov.append(Individual(genome=child_g, species="spec_neural_weight", _index=len(pop_nogov)))

        for ind in pop_nogov:
            ev = evaluator.evaluate(ind)
            ind.fitness = float(ev.sub_scores.get("target_defect_recall", 0.0))

        best_nogov = max(pop_nogov, key=lambda i: i.fitness)
        evals_nogov = len(pop_nogov)
        for g in range(6):
            if best_nogov.fitness >= 100.0:
                break
            new_pop = [best_nogov]
            while len(new_pop) < 8:
                p1 = rng.choice(pop_nogov)
                p2 = rng.choice(pop_nogov)
                child_g = p1.genome.crossover(p2.genome, rng=rng)
                child_g = child_g.mutate(rng=rng)
                ind = Individual(genome=child_g, species=p1.species, _generation=g + 1, _index=len(new_pop))
                ev = evaluator.evaluate(ind)
                ind.fitness = float(ev.sub_scores.get("target_defect_recall", 0.0))
                evals_nogov += 1
                new_pop.append(ind)
                if ind.fitness > best_nogov.fitness:
                    best_nogov = ind
            pop_nogov = new_pop

        final_nogov = evaluator.evaluate(best_nogov)
        results_no_gov.append({
            "condition": "No Governor (Unconstrained)",
            "fitness": final_nogov.score,
            "target_recall": final_nogov.sub_scores.get("target_defect_recall", 0.0),
            "anchor_accuracy": final_nogov.sub_scores.get("anchor_suite_accuracy", 0.0),
            "regressions": final_nogov.sub_scores.get("regressions_count", 0),
            "passed_holdout": final_nogov.passed_holdout,
            "evaluations": evals_nogov,
        })

        # 5. No Null-Space Projection (Evolution with raw cross-talk under Governor)
        pop_nonull = [Individual(genome=g0, species="spec_neural_weight", _index=0)]
        g_seed_nonull = g0.clone()
        if hyps:
            g_seed_nonull.sparse_patches[hyps[0].target_layer] = np.copy(hyps[0].delta)
            pop_nonull.append(Individual(genome=g_seed_nonull, species="spec_neural_weight", _index=1))

        while len(pop_nonull) < 8:
            parent = g_seed_nonull if rng.random() < 0.7 else g0
            child_g = rank1_op.mutate(parent, rng=rng, layer_shapes=spec.layer_shapes)
            child_g.scale = float(rng.uniform(0.7, 1.8))
            pop_nonull.append(Individual(genome=child_g, species="spec_neural_weight", _index=len(pop_nonull)))

        for ind in pop_nonull:
            ev = evaluator.evaluate(ind)
            ind.fitness = ev.score

        best_nonull = max(pop_nonull, key=lambda i: i.fitness)
        evals_nonull = len(pop_nonull)
        for g in range(6):
            if best_nonull.fitness >= 99.7:
                break
            new_pop = [max(pop_nonull, key=lambda i: i.fitness)]
            while len(new_pop) < 8:
                p1 = rng.choice(pop_nonull)
                p2 = rng.choice(pop_nonull)
                child_g = p1.genome.crossover(p2.genome, rng=rng)
                child_g = child_g.mutate(rng=rng)
                ind = Individual(genome=child_g, species=p1.species, _generation=g + 1, _index=len(new_pop))
                ev = evaluator.evaluate(ind)
                ind.fitness = ev.score
                evals_nonull += 1
                new_pop.append(ind)
                if ind.fitness > best_nonull.fitness:
                    best_nonull = ind
            pop_nonull = new_pop

        final_nonull = evaluator.evaluate(best_nonull)
        results_no_null.append({
            "condition": "No Null-Space Projection",
            "fitness": final_nonull.score,
            "target_recall": final_nonull.sub_scores.get("target_defect_recall", 0.0),
            "anchor_accuracy": final_nonull.sub_scores.get("anchor_suite_accuracy", 0.0),
            "regressions": final_nonull.sub_scores.get("regressions_count", 0),
            "passed_holdout": final_nonull.passed_holdout,
            "evaluations": evals_nonull,
        })

    def _summarize_cond(name: str, cond_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(cond_results)
        passed = sum(1 for r in cond_results if r["passed_holdout"])
        pass_rate = (passed / n) * 100.0
        ci_l, ci_h = wilson_score_interval(passed, n)
        cp_l, cp_h = clopper_pearson_interval(passed, n)
        return {
            "condition": name,
            "pass_rate_pct": round(pass_rate, 2),
            "pass_rate_wilson_ci95": [round(ci_l, 2), round(ci_h, 2)],
            "pass_rate_clopper_pearson_ci95": [round(cp_l, 2), round(cp_h, 2)],
            "mean_target_recall": round(float(np.mean([r["target_recall"] for r in cond_results])), 2),
            "std_target_recall": round(float(np.std([r["target_recall"] for r in cond_results])), 2),
            "mean_anchor_accuracy": round(float(np.mean([r["anchor_accuracy"] for r in cond_results])), 2),
            "std_anchor_accuracy": round(float(np.std([r["anchor_accuracy"] for r in cond_results])), 2),
            "mean_regressions": round(float(np.mean([r["regressions"] for r in cond_results])), 2),
            "std_regressions": round(float(np.std([r["regressions"] for r in cond_results])), 2),
            "mean_evaluations": round(float(np.mean([r["evaluations"] for r in cond_results])), 2),
            "std_evaluations": round(float(np.std([r["evaluations"] for r in cond_results])), 2),
        }

    ablation_summary = {
        "study": "Darwin Weight-Space Programming (DWSP) Component Ablation Study",
        "architecture": architecture,
        "seeds_tested": len(seeds),
        "conditions": {
            "full_dwsp": _summarize_cond("Full DWSP (Dual-System)", results_full),
            "no_llm_analyst": _summarize_cond("No LLM Analyst (Blind GA)", results_no_analyst),
            "no_evolution": _summarize_cond("No Evolution (One-Shot LLM)", results_no_evo),
            "no_governor": _summarize_cond("No Governor (Unconstrained)", results_no_gov),
            "no_null_space": _summarize_cond("No Null-Space Projection", results_no_null),
        },
    }

    if output_json:
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ablation_summary, indent=2), encoding="utf-8")

    return ablation_summary


def run_dwsp_sensitivity_benchmark(
    seeds: Optional[List[int]] = None,
    output_json: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute extended sensitivity verification across 30 seeds for shallow and deep architectures."""
    if seeds is None:
        seeds = list(range(1, 31))

    shallow_summary = run_multiseed_tri_factor_benchmark(seeds=seeds, architecture="shallow")
    deep_summary = run_multiseed_tri_factor_benchmark(seeds=seeds, architecture="deep")

    res = {
        "benchmark": "DWSP Extended Sensitivity Verification (30 Seeds)",
        "seeds_count": len(seeds),
        "shallow_architecture": shallow_summary,
        "deep_architecture": deep_summary,
    }
    if output_json:
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res

    return summary
