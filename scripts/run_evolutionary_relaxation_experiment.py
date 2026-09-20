"""run_evolutionary_relaxation_experiment.py — Scientific Comparative Benchmark:
Binary Fitness Cliff vs. Continuous Behavioral Fitness Relaxation.

Implements the reviewer's evolutionary proposal:
1. Structural Syntactic SOS Verification (Gate 4 Fix): Prevents bare definitions like (R - 2r)
   from obtaining automated proof certification (must be 0.0).
2. Continuous Behavioral Fitness Relaxation: Smooth multi-objective gradient providing
   partial credit for numerical closeness, non-negativity, equilateral vanishing,
   and structural stepping stones.
3. Comparative Population Dynamics: Quantifies true evolutionary search versus
   clonal drift / neutral replication under both regimes.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

from evolab.euler_inequality import (
    EulerFormulaGenome,
    EulerInequalityEvaluator,
    EulerNode,
    EulerTriangleSpec,
    build_euler_naive_genome,
    build_euler_sos_genome,
    build_euler_ravi_sos_genome,
    is_syntactic_sos_proof,
)
from evolab.engine import Individual


def generate_pure_random_node(depth: int, max_depth: int, rng: random.Random) -> EulerNode:
    """Generates a strictly primitive random subtree with NO pre-packaged schemata."""
    terminals = [
        ("VAR", "a"), ("VAR", "b"), ("VAR", "c"), ("VAR", "Delta"),
        ("CONST", 0.5), ("CONST", 1.0), ("CONST", 2.0), ("CONST", 4.0), ("CONST", 8.0),
    ]
    binary_ops = ["ADD", "SUB", "MUL", "DIV"]
    unary_ops = ["SQ"]

    if depth >= max_depth or (depth > 1 and rng.random() < 0.35):
        t_op, t_val = rng.choice(terminals)
        return EulerNode(t_op, t_val)

    op_type = "unary" if rng.random() < 0.25 else "binary"
    if op_type == "binary":
        op = rng.choice(binary_ops)
        left = generate_pure_random_node(depth + 1, max_depth, rng)
        right = generate_pure_random_node(depth + 1, max_depth, rng)
        return EulerNode(op, None, left, right)
    else:
        op = rng.choice(unary_ops)
        left = generate_pure_random_node(depth + 1, max_depth, rng)
        return EulerNode(op, None, left, None)


def pure_primitive_mutate(genome: EulerFormulaGenome, rng: random.Random) -> EulerFormulaGenome:
    """Mutation using primitive operations and random subtree generation."""
    child = genome.clone()
    nodes: list[EulerNode] = []

    def collect(n: EulerNode):
        nodes.append(n)
        if n.left: collect(n.left)
        if n.right: collect(n.right)

    collect(child.root)
    if not nodes:
        return child

    mtype = rng.choice(["subtree_mutate", "constant_tweak", "operator_swap", "square_wrap", "neutral_drift"])

    if mtype == "subtree_mutate":
        target = rng.choice(nodes)
        rand_sub = generate_pure_random_node(depth=0, max_depth=rng.randint(1, 3), rng=rng)
        target.op = rand_sub.op
        target.value = rand_sub.value
        target.left = rand_sub.left
        target.right = rand_sub.right

    elif mtype == "constant_tweak":
        consts = [n for n in nodes if n.op == "CONST"]
        if consts:
            t = rng.choice(consts)
            t.value = rng.choice([0.5, 1.0, 2.0, 4.0, 8.0])

    elif mtype == "operator_swap":
        bins = [n for n in nodes if n.op in ("ADD", "SUB", "MUL", "DIV")]
        if bins:
            t = rng.choice(bins)
            t.op = rng.choice(["ADD", "SUB", "MUL", "DIV"])

    elif mtype == "square_wrap":
        target = rng.choice(nodes)
        cloned = target.clone()
        target.op = "SQ"
        target.value = None
        target.left = cloned
        target.right = None

    elif mtype == "neutral_drift":
        swappable = [n for n in nodes if n.op in ("ADD", "MUL") and n.left and n.right]
        if swappable:
            t = rng.choice(swappable)
            t.left, t.right = t.right, t.left

    return child


def run_condition(
    condition_name: str,
    continuous_behavioral: bool,
    pop_size: int = 80,
    generations: int = 40,
    seed: int = 42,
) -> dict:
    """Runs a controlled evolutionary trial under either strict or continuous behavioral regime."""
    rng = random.Random(seed)
    spec = EulerTriangleSpec(continuous_behavioral=continuous_behavioral)
    evaluator = EulerInequalityEvaluator(spec)

    # Initial population: 1 naive subtraction + (pop_size - 1) pure random trees
    pop: list[Individual] = [
        Individual(genome=build_euler_naive_genome(), species="spec_euler")
    ]
    for _ in range(pop_size - 1):
        r_node = generate_pure_random_node(depth=0, max_depth=rng.randint(2, 4), rng=rng)
        g = EulerFormulaGenome(root=r_node, formula_name="PurePrimitiveRandom")
        pop.append(Individual(genome=g, species="spec_euler"))

    history = []
    best_overall = None
    best_score = -1.0

    for gen in range(1, generations + 1):
        # Evaluation
        for ind in pop:
            res = evaluator.evaluate(ind.genome)
            ind.fitness = res.score
            ind.metadata = res.sub_scores
            if res.score > best_score:
                best_score = res.score
                best_overall = ind.genome.clone()

        sorted_pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
        fitnesses = [ind.fitness for ind in pop]
        mean_fit = sum(fitnesses) / len(fitnesses)

        # Count clonal copies of naive genotype
        naive_clones = sum(1 for ind in pop if ind.genome.root.to_pretty_str() in ("(R - (2.0 * r))", "(R - (r * 2.0))"))

        # Measure distinct expressions
        unique_exprs = len(set(ind.genome.root.to_pretty_str() for ind in pop))

        # Structural traits
        sos_count = sum(1 for ind in pop if ind.metadata.get("is_syntactic_sos", 0.0) == 1.0)
        nonneg_rate = sum(1 for ind in pop if ind.metadata.get("nonnegativity_score", 0.0) >= 80.0)

        if gen == 1 or gen % 10 == 0 or gen == generations:
            print(f"    Gen {gen:2d}/{generations}: best={sorted_pop[0].fitness:.2f}, mean={mean_fit:.2f}, clones={naive_clones}, unique={unique_exprs}", flush=True)

        history.append({
            "gen": gen,
            "best_fitness": round(sorted_pop[0].fitness, 4),
            "mean_fitness": round(mean_fit, 4),
            "naive_clones": naive_clones,
            "unique_expressions": unique_exprs,
            "syntactic_sos_individuals": sos_count,
            "high_nonnegative_individuals": nonneg_rate,
            "best_expr": sorted_pop[0].genome.root.to_pretty_str(),
            "best_metadata": sorted_pop[0].metadata,
        })

        # Elitism
        elites = [Individual(genome=ind.genome.clone(), species=ind.species) for ind in sorted_pop[:3]]
        children = list(elites)

        # Tournament selection and reproduction
        while len(children) < pop_size:
            p1 = max(rng.sample(pop, 3), key=lambda x: x.fitness)
            p2 = max(rng.sample(pop, 3), key=lambda x: x.fitness)

            if rng.random() < 0.65:
                child_genome = p1.genome.crossover(p2.genome, rng=rng)
            else:
                child_genome = p1.genome.clone()

            child_genome = pure_primitive_mutate(child_genome, rng=rng)
            children.append(Individual(genome=child_genome, species="spec_euler"))

        pop = children

    final_eval = evaluator.evaluate(best_overall)
    sample_indices = [0, 4, 9, 19, 29, len(history) - 1]
    history_sample = [history[i] for i in sample_indices if i < len(history)]

    return {
        "condition": condition_name,
        "continuous_behavioral": continuous_behavioral,
        "best_fitness": final_eval.score,
        "best_expression": best_overall.root.to_pretty_str(),
        "final_metrics": final_eval.sub_scores,
        "history_sample": history_sample,
    }


def verify_gate4_forensics() -> dict:
    """Verifies that Gate 4 properly discriminates between definitions and proofs."""
    evaluator_strict = EulerInequalityEvaluator(EulerTriangleSpec(continuous_behavioral=False))

    naive = build_euler_naive_genome()
    sos = build_euler_sos_genome()
    ravi = build_euler_ravi_sos_genome()

    res_naive = evaluator_strict.evaluate(naive)
    res_sos = evaluator_strict.evaluate(sos)
    res_ravi = evaluator_strict.evaluate(ravi)

    is_naive_sos, reason_naive = is_syntactic_sos_proof(naive.root)
    is_sos_sos, reason_sos = is_syntactic_sos_proof(sos.root)

    return {
        "naive_definition_R_minus_2r": {
            "expression": naive.root.to_pretty_str(),
            "overall_fitness": res_naive.score,
            "formal_correctness": res_naive.sub_scores["formal_correctness"],
            "gate4_automated_proof_certified": res_naive.sub_scores["gate4_automated_proof_certified"],
            "is_syntactic_sos": res_naive.sub_scores["is_syntactic_sos"],
            "syntactic_check_reason": reason_naive,
            "verdict": "REJECTED_AS_PROOF" if res_naive.sub_scores["gate4_automated_proof_certified"] == 0.0 else "INCORRECTLY_CERTIFIED",
        },
        "constructive_sos_proof": {
            "expression": sos.root.to_pretty_str(),
            "overall_fitness": res_sos.score,
            "formal_correctness": res_sos.sub_scores["formal_correctness"],
            "gate4_automated_proof_certified": res_sos.sub_scores["gate4_automated_proof_certified"],
            "is_syntactic_sos": res_sos.sub_scores["is_syntactic_sos"],
            "syntactic_check_reason": reason_sos,
            "verdict": "CERTIFIED_CONSTRUCTIVE_PROOF" if res_sos.sub_scores["gate4_automated_proof_certified"] == 100.0 else "FAILED",
        },
        "ravi_substitution_sos_proof": {
            "expression": ravi.root.to_pretty_str(),
            "overall_fitness": res_ravi.score,
            "gate4_automated_proof_certified": res_ravi.sub_scores["gate4_automated_proof_certified"],
            "is_syntactic_sos": res_ravi.sub_scores["is_syntactic_sos"],
            "verdict": "CERTIFIED_CONSTRUCTIVE_PROOF" if res_ravi.sub_scores["gate4_automated_proof_certified"] == 100.0 else "FAILED",
        },
    }


def main():
    print("=" * 80)
    print("EVOLAB SCIENTIFIC BENCHMARK: CONTINUOUS BEHAVIORAL FITNESS & STRUCTURAL SOS")
    print("=" * 80)

    # 1. Gate 4 Forensic Verification
    print("\n[1/3] Running Gate 4 Forensic Discrimination Verification...")
    gate4_audit = verify_gate4_forensics()
    print("  -> Naive (R - 2r) Gate 4 Certified:", gate4_audit["naive_definition_R_minus_2r"]["gate4_automated_proof_certified"])
    print("  -> Constructive SOS Gate 4 Certified:", gate4_audit["constructive_sos_proof"]["gate4_automated_proof_certified"])

    # 2. Condition A: Strict Binary Cliff (Baseline)
    print("\n[2/3] Running Condition A (Strict Binary Cliff - Baseline)...")
    res_strict = run_condition(
        condition_name="Condition A: Strict Binary Cliff (Lethal Gate 1)",
        continuous_behavioral=False,
        pop_size=80,
        generations=40,
        seed=42,
    )
    print(f"  -> Gen 1: mean={res_strict['history_sample'][0]['mean_fitness']}, clones={res_strict['history_sample'][0]['naive_clones']}")
    print(f"  -> Final: mean={res_strict['history_sample'][-1]['mean_fitness']}, clones={res_strict['history_sample'][-1]['naive_clones']}")

    # 3. Condition B: Continuous Behavioral Fitness Relaxation
    print("\n[3/3] Running Condition B (Continuous Behavioral Fitness Relaxation)...")
    res_continuous = run_condition(
        condition_name="Condition B: Continuous Behavioral Fitness Relaxation (Reviewer's Proposal)",
        continuous_behavioral=True,
        pop_size=80,
        generations=40,
        seed=42,
    )
    print(f"  -> Gen 1: mean={res_continuous['history_sample'][0]['mean_fitness']}, unique_exprs={res_continuous['history_sample'][0]['unique_expressions']}")
    print(f"  -> Final: mean={res_continuous['history_sample'][-1]['mean_fitness']}, unique_exprs={res_continuous['history_sample'][-1]['unique_expressions']}, sos_indivs={res_continuous['history_sample'][-1]['syntactic_sos_individuals']}")

    # Save to reports
    report_path = Path("reports/continuous_behavioral_relaxation_experiment.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    full_report = {
        "benchmark": "Evolutionary Relaxation & Structural SOS Audit",
        "gate4_forensic_verification": gate4_audit,
        "experiment_strict_binary_cliff": res_strict,
        "experiment_continuous_behavioral_relaxation": res_continuous,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[SUCCESS] Full scientific experiment report saved to {report_path}")


if __name__ == "__main__":
    main()
