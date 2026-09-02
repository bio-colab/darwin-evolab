"""CLI for evolab.

Usage:
    python run.py evolve --engine greedy --scenario click_cli_parser
    python run.py evolve --engine ga --genome numeric -g 50 -s 123
    python run.py evolve --source app.py --tests tests.json --func parse_cli
    python run.py inspect run_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import EvolutionEngine, parse_report, summarize
from .code_fixtures import (
    SCENARIO_REGISTRY,
    load_scenario_file,
    load_source_scenario,
)


def cmd_inspect(args) -> int:
    try:
        report = parse_report(args.file)
    except FileNotFoundError:
        print(f"error: report file not found: {args.file}")
        print("hint: pass a report path, e.g. evolab inspect run_report.json")
        return 2
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print("\n".join(summarize(report)))
    if not report.is_valid:
        return 2
    best = report.best_individual or {}
    hit = (
        isinstance(best.get("fitness"), (int, float))
        and best.get("passed_holdout") is not False
        and float(best["fitness"]) >= 99.7
    )
    return 0 if hit else 1


def _build_engine(args, fitness_fn=None, genome_size=None) -> EvolutionEngine:
    return EvolutionEngine(
        population_size=args.population,
        early_stop_fitness=args.target,
        stagnation_patience=args.patience,
        sharing_mode=args.mode,
        exploit_after_frac=args.frac,
        seed=args.seed,
        fitness_fn=fitness_fn,
        genome_size=genome_size,
    )


def _resolve_engine(args) -> str:
    if args.engine != "auto":
        return args.engine
    if args.genome in ("numeric", "electronics"):
        return "ga"
    return "greedy"


def _load_code_scenario(args):
    if args.scenario_file:
        return load_scenario_file(args.scenario_file), True
    if args.source:
        if not args.tests or not args.func:
            print("error: --source requires --tests and --func")
            return None, True
        return load_source_scenario(args.source, args.tests, args.func, args.target_file), True
    if args.scenario not in SCENARIO_REGISTRY:
        names = ", ".join(sorted(SCENARIO_REGISTRY))
        print(f"error: unknown scenario {args.scenario!r}")
        print(f"hint: choose one of: {names}")
        return None, True
    return SCENARIO_REGISTRY[args.scenario](), False


def _hit(result: dict, target: float) -> bool:
    bi = result.get("best_individual") or {}
    fit = bi.get("fitness")
    if not isinstance(fit, (int, float)):
        return False
    if bi.get("passed_holdout") is False:
        return False
    cfg = result.get("config") or {}
    cap = target
    if cfg.get("genome") == "code" or cfg.get("search") == "greedy_forward":
        cap = min(float(target), 100.0)
    elif cfg.get("genome") == "electronics":
        cap = min(float(target), 80.0)
    return float(fit) >= float(cap)


def cmd_evolve(args) -> int:
    engine_kind = _resolve_engine(args)
    scenario = None
    external = False
    result: dict

    if args.genome == "electronics":
        import sys
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from experimental.electronics.bridge import (
                list_electronics_scenarios,
                prepare_electronics_run,
            )
        except ImportError as exc:
            print(f"error: electronics track not available ({exc})")
            return 2
        name = args.scenario if args.scenario in list_electronics_scenarios() else "half_adder"
        if args.scenario not in list_electronics_scenarios() and args.scenario != "click_cli_parser":
            print(f"error: unknown electronics scenario {args.scenario!r}")
            print("hint: " + ", ".join(list_electronics_scenarios()))
            return 2
        evaluator, pop, name = prepare_electronics_run(name, args.population, args.seed)
        print(
            f"Engine: GA | genome=electronics | scenario={name} "
            f"| pop={args.population} gens={args.generations} seed={args.seed}"
        )
        # Thread the scenario's true genome size into the engine so the
        # printed config is truthful instead of the numeric default (16):
        # FloatGenome scenarios report their gene count, structured netlist
        # genomes report their own len() (= connection count). The engine's
        # length guard only applies to FloatGenome, and netlist GA operators
        # are topology-aware, so for netlists this is descriptive
        # bookkeeping — never a numeric constraint.
        genome_size = len(pop[0].genome)
        engine = _build_engine(args, fitness_fn=evaluator, genome_size=genome_size)
        result = engine.run(args.generations, initial_population=pop)
        result.setdefault("config", {})
        result["config"]["genome"] = "electronics"
        result["config"]["scenario"] = name
        result["config"]["search"] = "ga"
        if hasattr(evaluator, "stats") and hasattr(evaluator, "scenario_key"):
            try:
                st = evaluator.stats()
                print(
                    f"Archive: scenario={st['scenario']} hits={st['hits']} misses={st['misses']} "
                    f"| rows={st['total_evaluations']} distinct_genomes={st['distinct_genomes']} "
                    f"| db={st['db']}"
                )
            except Exception:
                pass
    elif engine_kind == "ga" or args.genome == "numeric":
        if args.genome != "numeric" and engine_kind == "ga":
            loaded, external = _load_code_scenario(args)
            if loaded is None:
                return 2
            scenario = loaded
            from .code_fixtures import make_code_population
            import random
            evaluator = scenario.create_evaluator()
            engine = _build_engine(args, fitness_fn=evaluator)
            pop = make_code_population(scenario, args.population, random.Random(args.seed))
            print(
                f"Engine: GA | genome=code | pop={args.population} "
                f"gens={args.generations} seed={args.seed} sandbox=False"
            )
            result = engine.run(args.generations, initial_population=pop)
            result.setdefault("config", {})
            result["config"]["genome"] = "code"
            result["config"]["scenario"] = scenario.name
            result["config"]["search"] = "ga"
            best = getattr(engine, "best_ever", None)
            if best is not None and hasattr(best.genome, "to_code"):
                result["best_individual"]["code"] = best.genome.to_code()
        else:
            print(
                f"Engine: GA | genome=numeric | pop={args.population} "
                f"gens={args.generations} seed={args.seed}"
            )
            engine = _build_engine(args)
            result = engine.run(args.generations)
    else:
        loaded, external = _load_code_scenario(args)
        if loaded is None:
            return 2
        scenario = loaded
        use_sandbox = args.sandbox or (external and not args.no_sandbox)
        if external and not use_sandbox:
            print("warning: evaluating external code without --sandbox")
        if use_sandbox:
            from .evaluators import SandboxFunctionTestEvaluator
            evaluator = SandboxFunctionTestEvaluator(
                base_sources=scenario.sources,
                target_file=scenario.target_file,
                func_name=scenario.func_name,
                test_cases=scenario.test_cases,
                holdout_cases=scenario.holdout_cases,
            )
        else:
            evaluator = scenario.create_evaluator()
        from .repair import catalog_sources, greedy_run_report
        catalog_n = len(catalog_sources(scenario.sources))
        print(
            f"Engine: Greedy | Search Budget: Catalog Size (N={catalog_n}) "
            f"| max_evals={args.max_evals} | Sandbox: {use_sandbox}"
        )
        result = greedy_run_report(
            scenario.sources,
            scenario.target_file,
            evaluator,
            scenario_name=scenario.name,
            max_evals=args.max_evals,
        )

    out_path = Path(args.output)
    if out_path.parent and str(out_path.parent):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Generations run : {result['total_generations']}")
    print(f"Candidates      : {result['total_candidates_evaluated']}")
    bi = result["best_individual"]
    print(
        f"Best            : {bi['id']} (fitness={bi['fitness']}, "
        f"species={bi.get('species', 'n/a')})"
    )
    if bi.get("code"):
        print("Best code:")
        print(bi["code"])
    if args.diff and scenario is not None:
        from .repair import RepairGenome, unified_source_diff
        repaired = dict(scenario.sources)
        if bi.get("edits") is not None:
            genome = RepairGenome(
                sources=dict(scenario.sources),
                target_file=scenario.target_file,
                edits=[],
            )
            from .repair import RepairEdit
            genome.edits = [
                RepairEdit(
                    kind=e["kind"],
                    file=e.get("file", scenario.target_file),
                    lineno=int(e["lineno"]),
                    col_offset=int(e["col_offset"]),
                    payload=tuple(sorted((e.get("payload") or {}).items())),
                )
                for e in bi["edits"]
            ]
            repaired = genome.apply_to(scenario.sources)
        elif bi.get("code"):
            repaired = dict(scenario.sources)
            repaired[scenario.target_file] = bi["code"]
        diff_text = unified_source_diff(scenario.sources, repaired)
        print(diff_text or "(no textual diff)")
        if args.diff_file:
            Path(args.diff_file).write_text(diff_text, encoding="utf-8")
    print(f"Early stop      : {'yes' if result['early_stop_triggered'] else 'no'}")
    print("History:")
    for h in result["history"]:
        bar = "#" * int(h["best_fitness"] // 4)
        print(
            f"  gen {h['generation']:>3}  best={h['best_fitness']:6.2f}  "
            f"mean={h['mean_fitness']:6.2f}  {bar}"
        )
    print(f"Saved to        : {out_path}")

    report = parse_report(out_path)
    print("\n".join(summarize(report)))
    if not report.is_valid:
        return 2
    return 0 if _hit(result, args.target) else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="evolab", description="Evolutionary experimentation lab"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="validate and analyze a report file")
    p_inspect.add_argument("file", nargs="?", default=None, help="report JSON path")
    p_inspect.set_defaults(func=cmd_inspect)

    p_evo = sub.add_parser("evolve", help="run a new evolution experiment")
    p_evo.add_argument("--engine", choices=["auto", "greedy", "ga"], default="auto",
                       help="auto: greedy for code, ga for numeric")
    p_evo.add_argument("--max-evals", type=int, default=None,
                       help="greedy evaluation budget")
    p_evo.add_argument("-g", "--generations", type=int, default=30)
    p_evo.add_argument("-p", "--population", type=int, default=16)
    p_evo.add_argument("-t", "--target", type=float, default=99.7,
                       help="early-stop / success fitness target")
    p_evo.add_argument("-s", "--seed", type=int, default=None)
    p_evo.add_argument("-k", "--patience", type=int, default=15)
    p_evo.add_argument("--mode", choices=["off", "static", "dynamic"],
                       default="dynamic", help="GA fitness-sharing schedule")
    p_evo.add_argument("--frac", type=float, default=0.667)
    p_evo.add_argument("--genome", choices=["code", "numeric", "electronics"], default="code")
    p_evo.add_argument("--scenario", default="click_cli_parser")
    p_evo.add_argument("--scenario-file", default=None, help="JSON CodeScenario")
    p_evo.add_argument("--source", action="append", default=[],
                       help="source file (repeatable)")
    p_evo.add_argument("--tests", default=None, help="JSON test cases")
    p_evo.add_argument("--func", default=None, help="target function name")
    p_evo.add_argument("--target-file", default=None)
    p_evo.add_argument("--sandbox", action="store_true")
    p_evo.add_argument("--no-sandbox", action="store_true")
    p_evo.add_argument("--diff", action="store_true", help="print unified diff")
    p_evo.add_argument("--diff-file", default=None, help="write unified diff")
    p_evo.add_argument("-o", "--output", default="run_report.json")
    p_evo.set_defaults(func=cmd_evolve)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if getattr(args, "command", None) == "inspect" and args.file is None:
        print("error: report file not found: <missing path>")
        print("hint: pass a report path, e.g. evolab inspect run_report.json")
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
