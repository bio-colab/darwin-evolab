import json
from pathlib import Path

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()]
    }

out_dir = Path("notebooks")
out_dir.mkdir(exist_ok=True)

# 1. Software Repair Notebook
nb1_cells = [
    md_cell("# 01. Automated Program Repair & SWE-bench Lite with Darwin-Evolab\n\nLearn how Darwin-Evolab localizes bugs with Ochiai Spectrum-Based Fault Localization (SBFL), applies targeted AST mutations, and solves real-world SWE-bench Lite issues with zero regressions."),
    code_cell("""import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

import evolab
print(f"Loaded Darwin-Evolab version: {evolab.__version__}")"""),
    md_cell("## 1. Quick Program Repair using Built-in Scenario\nWe evaluate a buggy click CLI parser function with Ochiai SBFL and repair it automatically."),
    code_cell("""from evolab.adapters import get_domain_adapter

adapter = get_domain_adapter("software_repair")
spec = adapter.parse_spec("click_cli_parser")

print(f"Target File: {spec.target_file}")
print(f"Target Function: {spec.func_name}")
print(f"Total Test Cases: {len(spec.tests)}")"""),
    md_cell("## 2. Ingesting and Solving a SWE-bench Lite Issue\nDarwin-Evolab includes a dedicated SWE-bench Lite adapter enforcing dual invariants: 100% FAIL_TO_PASS passed and 0% regression on PASS_TO_PASS."),
    code_cell("""from evolab.swe_bench import SWEBenchAdapter

swe_adapter = SWEBenchAdapter()
fixture_path = repo_root / "src" / "evolab" / "fixtures" / "swe_bench" / "sympy__sympy_13480.json"

instance = swe_adapter.parse_spec(str(fixture_path))
print(f"Instance ID: {instance.instance_id}")
print(f"Repository:  {instance.repo}")
print(f"Problem:     {instance.problem_statement}")"""),
    md_cell("## 3. Running Targeted Evolutionary Repair"),
    code_cell("""resolution = swe_adapter.solve_instance(instance, max_evals=16)

print(f"Resolved:            {resolution.resolved}")
print(f"FAIL_TO_PASS Passed: {resolution.fail_to_pass_passed}")
print(f"PASS_TO_PASS Clean:  {resolution.pass_to_pass_clean}")
print(f"Evaluations Used:    {resolution.evaluations_used}")
print(f"Execution Time:      {resolution.execution_time_seconds:.4f}s")
print("\\n--- Generated Git Patch ---\\n" + resolution.generated_patch)""")
]
(out_dir / "01_software_repair.ipynb").write_text(json.dumps(make_nb(nb1_cells), indent=2), encoding="utf-8")

# 2. Silicon Circuit Synthesis Notebook
nb2_cells = [
    md_cell("# 02. Silicon & Electronic Circuit Synthesis with Darwin-Evolab\n\nExplore Darwin-Evolab's physical hardware track: synthesize logic circuits from Boolean equations, simulate transistor physics with SPICE, and export vector schematics."),
    code_cell("""import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from evolab.adapters import get_domain_adapter
adapter = get_domain_adapter("electronics")
print("Electronics domain driver loaded successfully.")"""),
    md_cell("## 1. Specify Target Circuit via Boolean Equation\nSynthesize a full adder from Boolean logic equations:"),
    code_cell("""spec = adapter.parse_spec("Sum = A ^ B ^ Cin; Cout = (A & B) | (Cin & (A ^ B))")
print(f"Circuit Inputs: {spec.num_inputs}")
print(f"Circuit Outputs: {spec.num_outputs}")
print(f"Truth Table entries: {len(spec.truth_table)}")"""),
    md_cell("## 2. Evolve Circuit Population"),
    code_cell("""import random
rng = random.Random(42)
population = adapter.build_population(spec, size=16, rng=rng)
evaluator = adapter.build_evaluator(spec)

best_candidate = population[0]
best_fitness = evaluator.evaluate(best_candidate).score
print(f"Initial candidate fitness: {best_fitness:.2f}%")"""),
    md_cell("## 3. Export Synthesized Verilog RTL"),
    code_cell("""solution = adapter.export_solution(best_candidate, spec)
print("Synthesized Verilog Header:")
for line in solution.get("verilog_code", "").splitlines()[:12]:
    print(line)""")
]
(out_dir / "02_silicon_circuit_synthesis.ipynb").write_text(json.dumps(make_nb(nb2_cells), indent=2), encoding="utf-8")

# 3. CGP and Discrete Logic Notebook
nb3_cells = [
    md_cell("# 03. Cartesian Genetic Programming (CGP) & NSGA-II Pareto Optimization\n\nDiscover how Darwin-Evolab uses Cartesian Genetic Programming (CGP) to optimize digital arithmetic logic units (ALUs) across 4 competing physical objectives: Correctness, Dynamic Power, Delay, and Area."),
    code_cell("""import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from evolab.pareto import NSGA2Engine, build_silicon_multiobjective_evaluator
from evolab.cgp_logic import create_random_cgp_genome
from evolab.genome import Individual
import random
print("CGP and NSGA-II modules loaded successfully.")"""),
    md_cell("## 1. Setup 4-Objective Silicon Evaluation\nWe evaluate a 1-bit Half-Adder on 4 objectives simultaneously."),
    code_cell("""truth_table = [
    ((0, 0), (0, 0)),
    ((0, 1), (1, 0)),
    ((1, 0), (1, 0)),
    ((1, 1), (0, 1)),
]
objectives, eval_vector_fn = build_silicon_multiobjective_evaluator(truth_table)
for obj in objectives:
    print(f"Objective: {obj.name:15s} | Direction: {obj.direction:8s} | Weight: {obj.weight}")"""),
    md_cell("## 2. Run NSGA-II Non-Dominated Sorting"),
    code_cell("""rng = random.Random(42)
pop = [Individual(create_random_cgp_genome(2, 2, 8, rng=rng), species="spec_logic") for _ in range(12)]

engine = NSGA2Engine(
    objectives=objectives,
    evaluate_vector_fn=eval_vector_fn,
    population_size=12,
    generations=5,
    seed=42,
)
result = engine.run(initial_population=pop, generations=5)
print(f"Total Non-Dominated Pareto Solutions: {len(result['pareto_front'])}")
for i, sol in enumerate(result['pareto_front'][:3]):
    print(f"Solution {i+1}: Objective Vector = {sol['objectives']}")""")
]
(out_dir / "03_cgp_and_discrete_logic.ipynb").write_text(json.dumps(make_nb(nb3_cells), indent=2), encoding="utf-8")

# 4. Continuous Math & Vectorized/JAX Notebook
nb4_cells = [
    md_cell("# 04. High-Speed Vectorized & JAX Continuous Landscape Optimization\n\nEvaluate 10,000+ candidate solutions simultaneously across non-convex optimization benchmarks (Rastrigin, Rosenbrock, Ackley, Sphere) with hardware-accelerated SIMD/GPU vectorization."),
    code_cell("""import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import time
from evolab.vectorized import VectorizedLandscapeEvaluator, HAS_JAX
from evolab.genome import FloatGenome
print(f"JAX Hardware Acceleration Available: {HAS_JAX}")"""),
    md_cell("## 1. Parallel Batch Evaluation of 10,000 Candidates\nWe evaluate 10,000 8-dimensional solutions across the highly multimodal Rastrigin landscape."),
    code_cell("""n_candidates = 10000
dim = 8
rng = np.random.default_rng(42)
X = rng.uniform(-5.12, 5.12, size=(n_candidates, dim))
population = [FloatGenome(row.tolist()) for row in X]

evaluator = VectorizedLandscapeEvaluator(landscape="rastrigin", use_jax=True)
t0 = time.perf_counter()
results = evaluator.evaluate_batch(population)
elapsed = time.perf_counter() - t0

print(f"Evaluated {n_candidates} solutions in {elapsed * 1000.0:.2f} ms")
print(f"Throughput: {n_candidates / elapsed:.0f} evaluations/sec")
print(f"Backend used: {results[0].artifacts['backend']}")
print(f"Best candidate fitness in batch: {max(r.score for r in results):.4f}%")""")
]
(out_dir / "04_continuous_optimization_jax.ipynb").write_text(json.dumps(make_nb(nb4_cells), indent=2), encoding="utf-8")

print("Generated 4 notebooks successfully!")

