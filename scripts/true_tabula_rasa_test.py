"""true_tabula_rasa_test.py — Scientific Ablation Experiment: Pure Primitive Koza GP vs Schema-Assisted GP.

Investigates:
  1. Experiment A (True Tabula Rasa): ZERO pre-packaged schemata. Only primitive operators
     (+, -, *, /, ^2, sqrt) and raw terminals (a, b, c, Delta) with random subtree mutation.
  2. Experiment B (Ablated Denominator): Schemata without ready-made '8*Delta' or '4*Delta',
     forcing the GA to synthesize the denominator (8 * Delta) from primitive CONST(8) and VAR(Delta).
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
    DEFAULT_EULER_CASES,
)
from evolab.engine import EvolutionEngine, Individual


def generate_pure_random_node(depth: int, max_depth: int, rng: random.Random) -> EulerNode:
    """Generates a strictly primitive random subtree with NO pre-engineered schemata."""
    terminals = [
        ("VAR", "a"), ("VAR", "b"), ("VAR", "c"), ("VAR", "Delta"),
        ("CONST", 0.5), ("CONST", 1.0), ("CONST", 2.0), ("CONST", 4.0), ("CONST", 8.0),
    ]
    binary_ops = ["ADD", "SUB", "MUL", "DIV"]
    unary_ops = ["SQ", "SQRT"]

    if depth >= max_depth or (depth > 1 and rng.random() < 0.4):
        t_op, t_val = rng.choice(terminals)
        return EulerNode(t_op, t_val)

    # Choose operator
    op_type = rng.choice(["binary", "unary"])
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
    """Mutation using ONLY primitive operations and random subtree generation (Koza 1992 style)."""
    child = genome.clone()
    nodes: list[EulerNode] = []

    def collect(n: EulerNode):
        nodes.append(n)
        if n.left: collect(n.left)
        if n.right: collect(n.right)

    collect(child.root)
    if not nodes:
        return child

    mtype = rng.choice(["subtree_mutate", "constant_tweak", "operator_swap", "neutral_drift"])

    if mtype == "subtree_mutate":
        target = rng.choice(nodes)
        # Generate random primitive subtree of depth 1 to 3
        rand_sub = generate_pure_random_node(depth=0, max_depth=rng.randint(1, 3), rng=rng)
        target.op = rand_sub.op
        target.value = rand_sub.value
        target.left = rand_sub.left
        target.right = rand_sub.right

    elif mtype == "constant_tweak":
        consts = [n for n in nodes if n.op == "CONST"]
        if consts:
            t = rng.choice(consts)
            t.value = rng.choice([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    elif mtype == "operator_swap":
        bins = [n for n in nodes if n.op in ("ADD", "SUB", "MUL", "DIV")]
        if bins:
            t = rng.choice(bins)
            t.op = rng.choice(["ADD", "SUB", "MUL", "DIV"])

    elif mtype == "neutral_drift":
        swappable = [n for n in nodes if n.op in ("ADD", "MUL") and n.left and n.right]
        if swappable:
            t = rng.choice(swappable)
            t.left, t.right = t.right, t.left

    return child


def run_true_tabula_rasa_experiment(pop_size: int = 120, generations: int = 60, seed: int = 42) -> dict:
    """Runs True Tabula Rasa GP: Zero Schemata, Pure Primitives Only."""
    rng = random.Random(seed)
    spec = EulerTriangleSpec()
    evaluator = EulerInequalityEvaluator(spec)

    # Initial population: 1 naive subtraction + 119 pure random primitive trees
    pop: list[Individual] = [
        Individual(genome=build_euler_naive_genome(), species="spec_euler_sos")
    ]
    for _ in range(pop_size - 1):
        rand_root = generate_pure_random_node(depth=0, max_depth=rng.randint(2, 4), rng=rng)
        g = EulerFormulaGenome(root=rand_root, formula_name="PurePrimitiveRandom")
        pop.append(Individual(genome=g, species="spec_euler_sos"))

    history = []
    best_overall = None
    best_score = -1.0

    for gen in range(1, generations + 1):
        # Evaluate
        for ind in pop:
            res = evaluator.evaluate(ind.genome)
            ind.fitness = res.score
            ind.metadata = res.sub_scores
            if res.score > best_score:
                best_score = res.score
                best_overall = ind.genome.clone()

        # Selection and reproduction (Tournament selection + crossover + pure primitive mutation)
        sorted_pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
        elites = [Individual(genome=ind.genome.clone(), species=ind.species) for ind in sorted_pop[:4]]
        children = list(elites)

        while len(children) < pop_size:
            # Tournament selection
            p1 = max(rng.sample(pop, 3), key=lambda x: x.fitness)
            p2 = max(rng.sample(pop, 3), key=lambda x: x.fitness)

            # Crossover
            if rng.random() < 0.7:
                child_genome = p1.genome.crossover(p2.genome, rng=rng)
            else:
                child_genome = p1.genome.clone()

            # Pure primitive mutation (NO SCHEMATA)
            child_genome = pure_primitive_mutate(child_genome, rng=rng)
            children.append(Individual(genome=child_genome, species="spec_euler_sos"))

        history.append({
            "gen": gen,
            "best_fitness": round(sorted_pop[0].fitness, 4),
            "mean_fitness": round(sum(ind.fitness for ind in pop) / len(pop), 4),
            "best_expr": sorted_pop[0].genome.root.to_pretty_str(),
        })
        pop = children

    final_eval = evaluator.evaluate(best_overall)
    return {
        "experiment": "True Tabula Rasa (Pure Primitive GP, Zero Schemata)",
        "best_fitness": final_eval.score,
        "best_expression": best_overall.root.to_pretty_str(),
        "metrics": final_eval.sub_scores,
        "history_sample": [history[0], history[4], history[9], history[19], history[39], history[-1]],
    }


def run_ablated_denominator_experiment(pop_size: int = 120, generations: int = 60, seed: int = 42) -> dict:
    """Runs Experiment B: Schemata provided for differences, but 8*Delta is REMOVED."""
    rng = random.Random(seed)
    spec = EulerTriangleSpec()
    evaluator = EulerInequalityEvaluator(spec)

    # Building blocks WITHOUT 8*Delta or 4*Delta
    blocks = [
        EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "b"), EulerNode("VAR", "c"))),
        EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "c"), EulerNode("VAR", "a"))),
        EulerNode("SQ", None, EulerNode("SUB", None, EulerNode("VAR", "a"), EulerNode("VAR", "b"))),
        EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "b"), EulerNode("VAR", "c")), EulerNode("VAR", "a")),
        EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "a"), EulerNode("VAR", "c")), EulerNode("VAR", "b")),
        EulerNode("SUB", None, EulerNode("ADD", None, EulerNode("VAR", "a"), EulerNode("VAR", "b")), EulerNode("VAR", "c")),
        # Terminals for denominator: CONST(8), CONST(4), VAR(Delta) must be assembled via MUL!
        EulerNode("VAR", "Delta"),
        EulerNode("CONST", 8.0),
    ]

    def mutate_no_denominator_block(genome: EulerFormulaGenome, r: random.Random) -> EulerFormulaGenome:
        child = genome.clone()
        nodes: list[EulerNode] = []
        def collect(n: EulerNode):
            nodes.append(n)
            if n.left: collect(n.left)
            if n.right: collect(n.right)
        collect(child.root)
        if not nodes: return child

        mtype = r.choice(["inject_block", "operator_swap", "constant_tweak", "neutral_drift"])
        if mtype == "inject_block":
            t = r.choice(nodes)
            rep = r.choice(blocks).clone()
            t.op, t.value, t.left, t.right = rep.op, rep.value, rep.left, rep.right
        elif mtype == "operator_swap":
            bins = [n for n in nodes if n.op in ("ADD", "SUB", "MUL", "DIV")]
            if bins:
                t = r.choice(bins)
                t.op = r.choice(["ADD", "SUB", "MUL", "DIV"])
        elif mtype == "constant_tweak":
            consts = [n for n in nodes if n.op == "CONST"]
            if consts:
                t = r.choice(consts)
                t.value = r.choice([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
        elif mtype == "neutral_drift":
            swaps = [n for n in nodes if n.op in ("ADD", "MUL") and n.left and n.right]
            if swaps:
                t = r.choice(swaps)
                t.left, t.right = t.right, t.left
        return child

    # Initial pop from naive subtraction
    pop = [Individual(genome=build_euler_naive_genome(), species="spec_euler_sos")]
    for _ in range(pop_size - 1):
        cand = build_euler_naive_genome().mutate(rng=rng)
        pop.append(Individual(genome=cand, species="spec_euler_sos"))

    best_overall = None
    best_score = -1.0

    for gen in range(1, generations + 1):
        for ind in pop:
            res = evaluator.evaluate(ind.genome)
            ind.fitness = res.score
            ind.metadata = res.sub_scores
            if res.score > best_score:
                best_score = res.score
                best_overall = ind.genome.clone()

        sorted_pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
        elites = [Individual(genome=ind.genome.clone(), species=ind.species) for ind in sorted_pop[:4]]
        children = list(elites)
        while len(children) < pop_size:
            p1 = max(rng.sample(pop, 3), key=lambda x: x.fitness)
            p2 = max(rng.sample(pop, 3), key=lambda x: x.fitness)
            child_g = p1.genome.crossover(p2.genome, rng=rng) if rng.random() < 0.7 else p1.genome.clone()
            child_g = mutate_no_denominator_block(child_g, r=rng)
            children.append(Individual(genome=child_g, species="spec_euler_sos"))
        pop = children

    final_eval = evaluator.evaluate(best_overall)
    return {
        "experiment": "Experiment B (Ablated Denominator: 8*Delta NOT in blocks)",
        "best_fitness": final_eval.score,
        "best_expression": best_overall.root.to_pretty_str(),
        "metrics": final_eval.sub_scores,
    }


if __name__ == "__main__":
    print("Running True Tabula Rasa GP Experiment (Pure Primitives, Zero Schemata)...")
    res_a = run_true_tabula_rasa_experiment(pop_size=120, generations=60, seed=42)
    print(f"Exp A Done! Best Fitness: {res_a['best_fitness']}")
    print(f"Exp A Expression: {res_a['best_expression']}")

    print("\nRunning Experiment B (Ablated Denominator: No 8*Delta block)...")
    res_b = run_ablated_denominator_experiment(pop_size=120, generations=60, seed=42)
    print(f"Exp B Done! Best Fitness: {res_b['best_fitness']}")
    print(f"Exp B Expression: {res_b['best_expression']}")

    out_data = {"experiment_A_true_tabula_rasa": res_a, "experiment_B_ablated_denominator": res_b}
    Path("reports/true_tabula_rasa_experiment.json").write_text(json.dumps(out_data, indent=2), encoding="utf-8")

