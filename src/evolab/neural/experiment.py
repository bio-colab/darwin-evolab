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
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def run_group_b_oneshot_llm(spec: NeuralProblemSpec) -> Dict[str, Any]:
    """Group B: Raw one-shot LLM weight editing (Takehi et al. WSP)."""
    evaluator = NeuralWeightEvaluator(spec)
    analyst = MockWeightAnalyst()

    hypotheses = analyst.diagnose_and_propose(
        base_weights=spec.base_weights,
        failing_inputs=spec.fail_to_pass_x,
        target_class=spec.target_class,
        penultimate_layer="layer2.weight",
    )

    if not hypotheses:
        return {"group": "Group B (One-Shot LLM)", "fitness": 0.0, "evaluations": 1}

    h = hypotheses[0]
    # In one-shot WSP, candidate is applied directly with scale=1.0 without evolutionary refinement
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
    analyst = MockWeightAnalyst()

    # 1. System 1: Diagnostic hypothesis
    hypotheses = analyst.diagnose_and_propose(
        base_weights=spec.base_weights,
        failing_inputs=spec.fail_to_pass_x,
        target_class=spec.target_class,
        penultimate_layer="layer2.weight",
    )

    g0 = NeuralWeightGenome(base_model_id=spec.base_model_id)
    pop: List[Individual] = [Individual(genome=g0, species="spec_neural_weight", _index=0)]

    g_seed = g0.clone()
    if hypotheses:
        h = hypotheses[0]
        g_seed.sparse_patches[h.target_layer] = np.copy(h.delta)
        g_seed.edit_lineage.append({"op": "analyst_seed", "reasoning": h.reasoning})
        pop.append(Individual(genome=g_seed, species="spec_neural_weight", _index=1))

    # 2. System 2: Algebraic population diversification
    rank1_op = RankOnePerturbationMutation(eta=0.05)
    svd_op = SpectralRescaleMutation(singular_index=0, scale_factor=1.15)

    while len(pop) < pop_size:
        idx = len(pop)
        parent = g_seed if rng.random() < 0.7 else g0
        if rng.random() < 0.5:
            child_g = rank1_op.mutate(parent, rng=rng, layer_shapes=spec.layer_shapes)
        else:
            child_g = svd_op.mutate(parent, rng=rng, layer_weights=spec.base_weights)
        child_g.scale = float(rng.uniform(0.7, 1.4))
        pop.append(Individual(genome=child_g, species="spec_neural_weight", _index=idx))

    # Initial evaluation
    for ind in pop:
        res = evaluator.evaluate(ind)
        ind.fitness = res.score

    best_ever = max(pop, key=lambda i: i.fitness)
    total_evals = len(pop)

    # Evolution loop with Dual-Invariant selection
    for g in range(generations):
        if best_ever.fitness >= 99.7:
            break
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

    # 3. Post-Evolution Intron Pruning
    pruner = CounterfactualIntronPruner(threshold=1e-3)
    pruned_count = pruner.prune(best_ever.genome)

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
    res_b = run_group_b_oneshot_llm(spec)
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
