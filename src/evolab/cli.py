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
import sys
from pathlib import Path

from . import EvolutionEngine, parse_report, summarize
from .code_fixtures import (
    SCENARIO_REGISTRY,
    load_pytest_scenario,
    load_scenario_file,
    load_source_scenario,
)


def cmd_inspect(args) -> int:
    import sys
    target = getattr(args, "file", None)
    if target is None or target == "":
        if not sys.stdin.isatty():
            target = "-"
        else:
            print("error: report file not found: <missing path>")
            print("hint: pass a report path or '-' for stdin, e.g. evolab inspect run_report.json")
            return 2
    try:
        report = parse_report(target)
    except FileNotFoundError:
        print(f"error: report file not found: {target}")
        print("hint: pass a report path or '-' for stdin, e.g. evolab inspect run_report.json")
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


def _attach_telemetry_stream(engine: EvolutionEngine, path_str: str) -> Any:
    import os
    import time
    p = Path(path_str)
    if hasattr(os, "mkfifo") and not p.exists():
        try:
            os.mkfifo(str(p))
        except OSError:
            pass
    try:
        stream = open(p, "a", encoding="utf-8", buffering=1)
    except Exception as exc:
        print(f"warning: could not open telemetry stream {path_str}: {exc}", file=sys.stderr)
        return None

    def on_gen(event: Any) -> None:
        try:
            data = {
                "event": "generation",
                "gen": getattr(event, "generation", 0),
                "best_fitness": round(getattr(event, "best_fitness", 0.0), 4),
                "mean_fitness": round(getattr(event, "mean_fitness", 0.0), 4),
                "diversity": round(getattr(event, "diversity", 0.0), 4),
                "duration_ms": round(getattr(event, "duration_ms", 0.0), 2),
                "timestamp": getattr(event, "timestamp", time.time()),
            }
            stream.write(json.dumps(data) + "\n")
            stream.flush()
        except Exception:
            pass

    def on_complete(event: Any) -> None:
        try:
            data = {
                "event": "completed",
                "total_generations": getattr(event, "total_generations", 0),
                "best_fitness": round(getattr(event, "best_fitness", 0.0), 4),
                "early_stopped": getattr(event, "early_stopped", False),
                "total_time_seconds": round(getattr(event, "total_time_seconds", 0.0), 4),
                "timestamp": getattr(event, "timestamp", time.time()),
            }
            stream.write(json.dumps(data) + "\n")
            stream.flush()
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    from .events import GenerationEvaluatedEvent, RunCompletedEvent
    engine.event_bus.subscribe(GenerationEvaluatedEvent, on_gen)
    engine.event_bus.subscribe(RunCompletedEvent, on_complete)
    return stream


def _build_engine(args, fitness_fn=None, genome_size=None) -> EvolutionEngine:
    if getattr(args, "external_driver", None) and fitness_fn is None:
        from .ipc_evaluator import ExternalProcessEvaluator
        fitness_fn = ExternalProcessEvaluator(args.external_driver)
    if fitness_fn is not None and getattr(fitness_fn, "deterministic", False):
        from .eval_cache import attach_eval_cache
        fitness_fn = attach_eval_cache(fitness_fn, max_entries=8192)
    engine = EvolutionEngine(
        population_size=args.population,
        early_stop_fitness=args.target,
        stagnation_patience=args.patience,
        sharing_mode=args.mode,
        exploit_after_frac=args.frac,
        seed=args.seed,
        fitness_fn=fitness_fn,
        genome_size=genome_size,
    )
    from .ui.terminal import TerminalProgressObserver
    quiet = getattr(args, "quiet", False)
    gens = getattr(args, "generations", 30)
    observer = TerminalProgressObserver(total_generations=gens, quiet=quiet)
    observer.attach_to_engine(engine)

    telemetry_path = getattr(args, "telemetry_stream", None) or getattr(args, "telemetry_fifo", None)
    if telemetry_path:
        _attach_telemetry_stream(engine, telemetry_path)

    return engine


def _resolve_engine(args) -> str:
    if args.engine != "auto":
        return args.engine
    if args.genome in ("numeric", "electronics"):
        return "ga"
    return "greedy"


def _load_code_scenario(args):
    import sys
    if args.scenario_file:
        return load_scenario_file(args.scenario_file), True
    source = list(args.source or [])
    if not source and not sys.stdin.isatty() and (getattr(args, "pytest", None) or getattr(args, "tests", None)):
        source = ["-"]
    if source:
        target_func = args.func if isinstance(args.func, str) else None
        if getattr(args, "pytest", None):
            try:
                return load_pytest_scenario(source, args.pytest, target_func, args.target_file), True
            except Exception as e:
                print(f"error loading pytest scenario: {e}", file=sys.stderr)
                return None, True
        if not target_func:
            print("error: --source requires --func (or use --pytest for auto-detection)", file=sys.stderr)
            return None, True
        if not args.tests:
            print("error: --source requires either --tests or --pytest", file=sys.stderr)
            return None, True
        return load_source_scenario(source, args.tests, target_func, args.target_file), True
    if args.scenario not in SCENARIO_REGISTRY:
        names = ", ".join(sorted(SCENARIO_REGISTRY))
        print(f"error: unknown scenario {args.scenario!r}", file=sys.stderr)
        print(f"hint: choose one of: {names}", file=sys.stderr)
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
    quiet = getattr(args, "quiet", False)
    engine_kind = _resolve_engine(args)
    scenario = None
    external = False
    baseline_res = None
    result: dict

    is_electronics = (
        args.genome == "electronics"
        or bool(getattr(args, "netlist", None))
        or bool(getattr(args, "spec", None))
        or bool(getattr(args, "expr", None))
        or bool(getattr(args, "verilog_in", None))
        or bool(getattr(args, "waveform", None))
    )

    if getattr(args, "swe_bench", None):
        from .swe_bench import SWEBenchAdapter
        adapter = SWEBenchAdapter()
        spec = adapter.parse_spec(args.swe_bench)
        print(f"SWE-bench Issue    : {spec.instance_id} ({spec.repo})", file=sys.stderr)
        print(f"Problem Summary    : {spec.problem_statement[:80]}...", file=sys.stderr)
        print(f"Target File        : {spec.target_file}", file=sys.stderr)
        print(f"FAIL_TO_PASS Tests : {len(spec.fail_to_pass_tests)}", file=sys.stderr)
        print(f"PASS_TO_PASS Tests : {len(spec.pass_to_pass_tests)}", file=sys.stderr)
        
        resolution = adapter.solve_instance(spec, max_evals=args.max_evals or 32)
        print(f"\n[SWE-bench Resolution Verdict]", file=sys.stderr)
        print(f"Resolved           : {'YES (100% Green)' if resolution.resolved else 'NO'}", file=sys.stderr)
        print(f"FAIL_TO_PASS       : {'PASS' if resolution.fail_to_pass_passed else 'FAIL'}", file=sys.stderr)
        print(f"PASS_TO_PASS       : {'CLEAN (Zero Regressions)' if resolution.pass_to_pass_clean else 'BROKEN'}", file=sys.stderr)
        print(f"Evaluations Used   : {resolution.evaluations_used}", file=sys.stderr)
        print(f"Execution Time     : {resolution.execution_time_seconds}s", file=sys.stderr)
        
        if getattr(args, "patch_file", None):
            Path(args.patch_file).write_text(resolution.generated_patch, encoding="utf-8")
            print(f"Git Patch saved    : {args.patch_file}", file=sys.stderr)

        out_path = Path(args.output)
        out_path.write_text(json.dumps({
            "instance_id": resolution.instance_id,
            "resolved": resolution.resolved,
            "evaluations": resolution.evaluations_used,
            "execution_time_seconds": resolution.execution_time_seconds,
            "patch": resolution.generated_patch,
        }, indent=2) + "\n", encoding="utf-8")
        return 0 if resolution.resolved else 1
    if is_electronics:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from experimental.electronics.bridge import (
                list_electronics_scenarios,
                prepare_electronics_run,
                prepare_custom_electronics_run,
            )
        except ImportError as exc:
            print(f"error: electronics track not available ({exc})", file=sys.stderr)
            return 2

        has_custom_input = any((
            getattr(args, "spec", None),
            getattr(args, "netlist", None),
            getattr(args, "expr", None),
            getattr(args, "verilog_in", None),
            getattr(args, "waveform", None),
        ))
        effective_seed = 42 if args.seed is None else args.seed
        args.seed = effective_seed

        if has_custom_input:
            evaluator, pop, name = prepare_custom_electronics_run(
                spec_path=getattr(args, "spec", None),
                netlist_path=getattr(args, "netlist", None),
                expr=getattr(args, "expr", None),
                verilog_in=getattr(args, "verilog_in", None),
                waveform_path=getattr(args, "waveform", None),
                objective=getattr(args, "objective", None),
                population_size=args.population,
                seed=effective_seed,
            )
        else:
            name = args.scenario if args.scenario in list_electronics_scenarios() else "half_adder"
            if args.scenario not in list_electronics_scenarios() and args.scenario != "click_cli_parser":
                print(f"error: unknown electronics scenario {args.scenario!r}", file=sys.stderr)
                print("hint: " + ", ".join(list_electronics_scenarios()), file=sys.stderr)
                return 2
            evaluator, pop, name = prepare_electronics_run(name, args.population, effective_seed)

        tool_name = "cgp_digital"
        if hasattr(evaluator, "oracle"):
            tool_name = getattr(evaluator.oracle, "tool_name", "ngspice")
        elif hasattr(evaluator, "simulator"):
            tool_name = getattr(getattr(evaluator, "simulator", None), "name", "ngspice")
        elif "analog" in name.lower() or "waveform" in name.lower() or "filter" in name.lower():
            try:
                from experimental.electronics.models.ngspice_bridge import has_ngspice
                tool_name = "ngspice" if has_ngspice() else "analytical_proxy"
            except Exception:
                tool_name = "analytical_proxy"
        elif "boolean" in name.lower() or "verilog" in name.lower() or "adder" in name.lower():
            tool_name = "cgp_logic"

        print(
            f"Engine: GA | genome=electronics | scenario={name} | tool={tool_name} "
            f"| pop={args.population} gens={args.generations} seed={effective_seed}",
            file=sys.stderr,
        )
        # Thread the scenario's true genome size into the engine so the
        # printed config is truthful instead of the numeric default (16):
        # FloatGenome scenarios report their gene count, structured netlist
        # genomes report their own len() (= connection count). The engine's
        # length guard only applies to FloatGenome, and netlist GA operators
        # are topology-aware, so for netlists this is descriptive
        # bookkeeping — never a numeric constraint.
        genome_size = len(pop[0].genome)
        if engine_kind == "nsga2":
            from .pareto import NSGA2Engine, build_silicon_multiobjective_evaluator
            from .genome import Individual
            import random

            expr_target = getattr(args, "expr", None) or "Sum = A ^ B; Cout = A & B"
            from experimental.electronics.inputs.boolean_expr import parse_boolean_spec
            from .cgp_logic import create_random_cgp_genome

            b_spec = parse_boolean_spec(expr_target)
            objs, eval_vec = build_silicon_multiobjective_evaluator(b_spec.truth_table)
            rng = random.Random(args.seed or 42)
            pop = [
                Individual(
                    create_random_cgp_genome(b_spec.num_inputs, b_spec.num_outputs, max(12, b_spec.num_inputs * 4), rng=rng),
                    species="spec_logic",
                )
                for _ in range(args.population)
            ]
            engine = NSGA2Engine(
                objectives=objs,
                evaluate_vector_fn=eval_vec,
                population_size=args.population,
                generations=args.generations,
                seed=args.seed,
            )
            nsga_res = engine.run(initial_population=pop, generations=args.generations)
            print(f"Engine: NSGA-II | Multi-Objective Pareto Frontier Discovered", file=sys.stderr)
            print(f"Pareto Front Size: {len(nsga_res['front_0'])} non-dominated solutions", file=sys.stderr)
            for i, sol in enumerate(nsga_res['front_0'][:5]):
                print(f"  Pareto #{i+1}: {sol['scores']}", file=sys.stderr)
            if getattr(args, "pareto_export", None):
                engine.export_pareto_front(args.pareto_export)
                print(f"Pareto Front saved: {args.pareto_export}", file=sys.stderr)
            return 0

        from .signals import SignalController
        engine = _build_engine(args, fitness_fn=evaluator, genome_size=genome_size)
        with SignalController(register_os_signals=True) as sc:
            result = engine.run(
                args.generations,
                initial_population=pop,
                resume_from=getattr(args, "resume", None),
                checkpoint_every=getattr(args, "checkpoint_every", None),
                checkpoint_dir=getattr(args, "checkpoint_dir", None),
                signal_controller=sc,
            )
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
                    f"| db={st['db']}",
                    file=sys.stderr,
                )
            except Exception:
                pass
    elif engine_kind == "ga" or args.genome == "numeric":
        from .signals import SignalController
        if args.genome != "numeric" and engine_kind == "ga":
            loaded, external = _load_code_scenario(args)
            if loaded is None:
                return 2
            scenario = loaded
            from .code_fixtures import make_code_population
            from .eval_cache import attach_eval_cache
            import random
            evaluator = attach_eval_cache(scenario.create_evaluator())
            engine = _build_engine(args, fitness_fn=evaluator)
            pop = make_code_population(scenario, args.population, random.Random(args.seed))
            if not quiet:
                print(
                    f"Engine: GA | genome=code | pop={args.population} "
                    f"gens={args.generations} seed={args.seed} sandbox=False",
                    file=sys.stderr,
                )
            with SignalController(register_os_signals=True) as sc:
                result = engine.run(
                    args.generations,
                    initial_population=pop,
                    resume_from=getattr(args, "resume", None),
                    checkpoint_every=getattr(args, "checkpoint_every", None),
                    checkpoint_dir=getattr(args, "checkpoint_dir", None),
                    signal_controller=sc,
                )
            result.setdefault("config", {})
            result["config"]["genome"] = "code"
            result["config"]["scenario"] = scenario.name
            result["config"]["search"] = "ga"
            best = getattr(engine, "best_ever", None)
            if best is not None and hasattr(best.genome, "to_code"):
                result["best_individual"]["code"] = best.genome.to_code()
        else:
            if not quiet:
                print(
                    f"Engine: GA | genome=numeric | pop={args.population} "
                    f"gens={args.generations} seed={args.seed}",
                    file=sys.stderr,
                )
            engine = _build_engine(args)
            with SignalController(register_os_signals=True) as sc:
                result = engine.run(
                    args.generations,
                    resume_from=getattr(args, "resume", None),
                    checkpoint_every=getattr(args, "checkpoint_every", None),
                    checkpoint_dir=getattr(args, "checkpoint_dir", None),
                    signal_controller=sc,
                )
    else:
        loaded, external = _load_code_scenario(args)
        if loaded is None:
            return 2
        scenario = loaded
        use_sandbox = args.sandbox or (external and not args.no_sandbox)
        if external and not use_sandbox:
            print("warning: evaluating external code without --sandbox", file=sys.stderr)
        from .eval_cache import attach_eval_cache
        if use_sandbox:
            from .evaluators import SandboxFunctionTestEvaluator
            evaluator = attach_eval_cache(
                SandboxFunctionTestEvaluator(
                    base_sources=scenario.sources,
                    target_file=scenario.target_file,
                    func_name=scenario.func_name,
                    test_cases=scenario.test_cases,
                    holdout_cases=scenario.holdout_cases,
                )
            )
        else:
            evaluator = attach_eval_cache(scenario.create_evaluator())
        from .repair import catalog_sources, greedy_run_report
        catalog_n = len(catalog_sources(scenario.sources))
        if not quiet:
            print(
                f"Engine: Greedy | Search Budget: Catalog Size (N={catalog_n}) "
                f"| max_evals={args.max_evals} | Sandbox: {use_sandbox}",
                file=sys.stderr,
            )
        baseline_res = None
        try:
            from .repair import RepairGenome
            base_genome = RepairGenome(sources=dict(scenario.sources), target_file=scenario.target_file, edits=[])
            baseline_res = evaluator.evaluate(base_genome)
        except Exception:
            pass

        from .ui.terminal import StepProgressObserver
        step_observer = StepProgressObserver(quiet=quiet)

        telemetry_path = getattr(args, "telemetry_stream", None) or getattr(args, "telemetry_fifo", None)
        greedy_telemetry_file = None
        if telemetry_path:
            import os
            tp = Path(telemetry_path)
            if hasattr(os, "mkfifo") and not tp.exists():
                try:
                    os.mkfifo(str(tp))
                except OSError:
                    pass
            try:
                greedy_telemetry_file = open(tp, "a", encoding="utf-8", buffering=1)
            except Exception:
                pass

        def _step_callback(step: int, score: float, evals: int, name: str = "") -> None:
            step_observer.on_step(step, score, evals, name)
            if greedy_telemetry_file:
                try:
                    import time
                    greedy_telemetry_file.write(json.dumps({
                        "event": "step",
                        "step": step,
                        "score": round(float(score), 4),
                        "evaluations": int(evals),
                        "candidate": name,
                        "timestamp": time.time(),
                    }) + "\n")
                    greedy_telemetry_file.flush()
                except Exception:
                    pass

        from .strategies import get_search_strategy
        strategy = get_search_strategy(
            "greedy",
            sources=scenario.sources,
            target_file=scenario.target_file,
            scenario_name=scenario.name,
            max_evals=args.max_evals,
            on_step=_step_callback,
        )
        result = strategy.search(evaluator)
        step_observer.complete(
            best_score=float((result.get("best_individual") or {}).get("fitness", 0.0)),
            total_evals=int(result.get("total_candidates_evaluated", 0)),
        )
        if greedy_telemetry_file:
            try:
                import time
                greedy_telemetry_file.write(json.dumps({
                    "event": "completed",
                    "total_evaluations": int(result.get("total_candidates_evaluated", 0)),
                    "best_score": float((result.get("best_individual") or {}).get("fitness", 0.0)),
                    "timestamp": time.time(),
                }) + "\n")
                greedy_telemetry_file.flush()
                greedy_telemetry_file.close()
            except Exception:
                pass

    if getattr(args, "llm", None) and not _hit(result, args.target):
        bi = result.get("best_individual") or {}
        current_fit = float(bi.get("fitness", 0.0))
        print(
            f"\n[Hybrid LLM] Evolutionary search stagnated at {current_fit:.2f}%. "
            f"Invoking {args.llm} stagnation breaker...",
            file=sys.stderr,
        )
        try:
            from .llm_mutator import LLMConfig, LLMSemanticMutator
            llm_model = args.llm_model or ("qwen/qwen3.8-27b" if args.llm == "groq" else "mock-model")
            cfg = LLMConfig(provider=args.llm, model_name=llm_model)
            mutator = LLMSemanticMutator(config=cfg)

            if is_electronics and "evaluator" in locals() and "engine" in locals() and engine is not None:
                best_g = getattr(engine, "best_ever", None)
                if best_g and hasattr(best_g.genome, "connections"):
                    from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
                    conns_str = "\n".join(
                        f"  wire {c.source.ic_index}:{c.source.pin} -> {c.destination.ic_index}:{c.destination.pin}"
                        for c in best_g.genome.connections
                    )
                    truth_str = getattr(evaluator, "truth_table", "Target logic")
                    c_data, resp = mutator.mutate_circuit_netlist(
                        current_topology=conns_str,
                        truth_table_specs=str(truth_str),
                        current_fitness=current_fit,
                        available_parts=list(getattr(best_g.genome, "ic_packages", [])),
                    )
                    if resp.success and c_data:
                        new_conns = [
                            Connection(
                                PinRef(c["src_ic"], c["src_pin"]),
                                PinRef(c["dst_ic"], c["dst_pin"]),
                            )
                            for c in c_data.get("connections", [])
                        ]
                        cand_circuit = CircuitNetlistGenome(
                            ic_packages=c_data.get("ic_packages", best_g.genome.ic_packages),
                            connections=new_conns,
                            num_inputs=best_g.genome.num_inputs,
                            num_outputs=best_g.genome.num_outputs,
                            functions_needed=best_g.genome.functions_needed,
                        )
                        fit_res = evaluator.evaluate(cand_circuit)
                        if fit_res.score > current_fit:
                            print(
                                f"[Hybrid LLM] Circuit stagnation broken! Fitness improved from {current_fit:.2f}% to {fit_res.score:.2f}%.",
                                file=sys.stderr,
                            )
                            bi["fitness"] = fit_res.score
                            bi["passed_holdout"] = fit_res.passed_holdout
                            result["best_individual"] = bi
                            result["total_candidates_evaluated"] = result.get("total_candidates_evaluated", 0) + 1
                            result.setdefault("history", []).append({
                                "generation": len(result.get("history", [])) + 1,
                                "best_fitness": fit_res.score,
                                "mean_fitness": fit_res.score,
                                "added": f"llm_circuit_{args.llm}",
                            })
                            engine.best_ever = Individual(genome=cand_circuit, fitness=fit_res.score, species="spec_electronics")
                        else:
                            print(
                                f"[Hybrid LLM] Circuit candidate rejected: score={fit_res.score:.2f}%. Safety preserved.",
                                file=sys.stderr,
                            )

            elif scenario is not None:
                src = bi.get("code") or scenario.sources.get(scenario.target_file, "")
                mutated_code, resp = mutator.mutate_code(src, current_fitness=current_fit)
                if resp.success and mutated_code:
                    from .repair import RepairGenome
                    cand_genome = RepairGenome(
                        sources=dict(scenario.sources),
                        target_file=scenario.target_file,
                        source=mutated_code,
                    )
                    fit_res = evaluator.evaluate(cand_genome)
                    if fit_res.score > current_fit and fit_res.passed_holdout is not False:
                        print(
                            f"[Hybrid LLM] Stagnation broken! Fitness improved from {current_fit:.2f}% to {fit_res.score:.2f}%.",
                            file=sys.stderr,
                        )
                        bi["fitness"] = fit_res.score
                        bi["code"] = mutated_code
                        bi["passed_holdout"] = fit_res.passed_holdout
                        result["best_individual"] = bi
                        result["total_candidates_evaluated"] = result.get("total_candidates_evaluated", 0) + 1
                        result["history"].append({
                            "generation": len(result.get("history", [])) + 1,
                            "best_fitness": fit_res.score,
                            "mean_fitness": fit_res.score,
                            "edits": len(bi.get("edits", [])) + 1,
                            "added": f"llm_{args.llm}",
                        })
                    else:
                        print(
                            f"[Hybrid LLM] Candidate rejected: score={fit_res.score:.2f}%, holdout={fit_res.passed_holdout}. Safety preserved.",
                            file=sys.stderr,
                        )
        except Exception as err:
            print(f"[Hybrid LLM] Stagnation breaker error: {err}", file=sys.stderr)

    if is_electronics and "engine" in locals() and engine is not None:
        best_g = getattr(engine, "best_ever", None)
        if best_g and hasattr(best_g.genome, "get_active_nodes"):
            from evolab.cgp_logic import estimate_fpga_resources
            fpga_target = getattr(args, "fpga_target", "ice40_hx1k")
            fpga_rep = estimate_fpga_resources(best_g.genome, fpga_target)
            result["fpga_resources"] = {
                "target": fpga_rep.target_preset,
                "board": fpga_rep.board_name,
                "vendor": fpga_rep.vendor,
                "estimated_luts": fpga_rep.estimated_luts,
                "total_luts": fpga_rep.total_luts,
                "lut_utilization_pct": fpga_rep.lut_utilization_pct,
                "pins_used": fpga_rep.total_pins_used,
                "fmax_mhz": fpga_rep.estimated_fmax_mhz,
                "fits": fpga_rep.fits_on_target,
            }

    out_path = Path(args.output)
    if out_path.parent and str(out_path.parent):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    bi = result.get("best_individual") or {}
    repaired = dict(scenario.sources) if scenario else {}
    diff_text = ""
    if scenario is not None:
        from .repair import RepairGenome, unified_source_diff
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
    elif is_electronics and "engine" in locals() and engine is not None:
        best_g = getattr(engine, "best_ever", None)
        if best_g and hasattr(best_g.genome, "connections"):
            conns_str = "\n".join(
                f"  wire {c.source.ic_index}:{c.source.pin} -> {c.destination.ic_index}:{c.destination.pin}"
                for c in best_g.genome.connections
            )
            ics = getattr(best_g.genome, "ic_packages", getattr(best_g.genome, "ics", []))
            ics_str = ", ".join(ics)
        elif best_g and hasattr(best_g.genome, "to_verilog"):
            diff_text = best_g.genome.to_verilog()
        elif best_g and hasattr(best_g.genome, "to_spice_netlist"):
            diff_text = best_g.genome.to_spice_netlist()
        elif best_g and hasattr(best_g.genome, "genes"):
            diff_text = "* Sized Circuit Parameters:\n" + "\n".join(
                f"  param[{i}] = {v:.6f}" for i, v in enumerate(best_g.genome.genes)
            )

    from .reporters import (
        format_terminal_diagnostics,
        format_markdown_summary,
        format_git_patch,
        apply_in_place,
    )

    if getattr(args, "patch_file", None):
        if scenario is not None:
            patch_str = format_git_patch(scenario, repaired)
        else:
            patch_str = f"=== SYNTHESIZED NETLIST TOPOLOGY ===\n{diff_text}\n"
        Path(args.patch_file).write_text(patch_str, encoding="utf-8")
        if not quiet:
            print(f"Patch saved     : {args.patch_file}", file=sys.stderr)

    if getattr(args, "summary_file", None):
        summary_str = format_markdown_summary(result, scenario, diff_text)
        Path(args.summary_file).write_text(summary_str, encoding="utf-8")
        if not quiet:
            print(f"Markdown Summary: {args.summary_file}", file=sys.stderr)

    if getattr(args, "schematic_file", None) and is_electronics and "engine" in locals() and engine is not None:
        best_g = getattr(engine, "best_ever", None)
        if best_g and (hasattr(best_g.genome, "circuit") or hasattr(best_g.genome, "connections") or hasattr(best_g.genome, "get_active_nodes")):
            from experimental.electronics.instruments.schematic import save_circuit_svg
            save_circuit_svg(best_g.genome, args.schematic_file)
            if not quiet:
                print(f"Schematic saved : {args.schematic_file}", file=sys.stderr)

    if getattr(args, "verilog_file", None) and is_electronics and "engine" in locals() and engine is not None:
        best_g = getattr(engine, "best_ever", None)
        if best_g and hasattr(best_g.genome, "to_verilog"):
            v_code = best_g.genome.to_verilog(module_name="synthesized_circuit")
            Path(args.verilog_file).write_text(v_code, encoding="utf-8")
            if not quiet:
                print(f"Verilog saved   : {args.verilog_file}", file=sys.stderr)

    if getattr(args, "ui_file", None) and is_electronics and "engine" in locals() and engine is not None:
        best_g = getattr(engine, "best_ever", None)
        if best_g:
            from experimental.electronics.ui.workbench_generator import save_workbench_html
            fpga_target = getattr(args, "fpga_target", "ice40_hx1k")
            meta = {
                "scenario": result.get("config", {}).get("scenario", "synthesized_logic"),
                "fitness": (result.get("best_individual") or {}).get("fitness", 100.0),
                "generations": result.get("total_generations", args.generations),
                "candidates": result.get("total_candidates_evaluated", 0),
                "fpga_target": fpga_target,
            }
            save_workbench_html(best_g.genome, args.ui_file, metadata=meta)
            if not quiet:
                print(f"Workbench UI saved: {args.ui_file}", file=sys.stderr)

            if hasattr(best_g.genome, "get_active_nodes"):
                from evolab.cgp_logic import estimate_fpga_resources
                fpga_rep = estimate_fpga_resources(best_g.genome, fpga_target)
                result["fpga_resources"] = {
                    "target": fpga_rep.target_preset,
                    "board": fpga_rep.board_name,
                    "vendor": fpga_rep.vendor,
                    "estimated_luts": fpga_rep.estimated_luts,
                    "total_luts": fpga_rep.total_luts,
                    "lut_utilization_pct": fpga_rep.lut_utilization_pct,
                    "pins_used": fpga_rep.total_pins_used,
                    "fmax_mhz": fpga_rep.estimated_fmax_mhz,
                    "fits": fpga_rep.fits_on_target,
                }

    if getattr(args, "apply", False) and scenario is not None and _hit(result, args.target):
        file_mapping = {Path(raw).name: Path(raw) for raw in (args.source or [])}
        applied = apply_in_place(scenario, repaired, create_backup=True, file_mapping=file_mapping)
        if applied and not quiet:
            print(f"[In-Place Apply] Successfully patched: {', '.join(applied)} (backup saved with .bak)", file=sys.stderr)

    out_format = getattr(args, "format", "console")
    if out_format == "markdown":
        print(format_markdown_summary(result, scenario, diff_text))
    elif out_format == "patch":
        print(format_git_patch(scenario, repaired))
    elif out_format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Generations run : {result['total_generations']}")
        print(f"Candidates      : {result['total_candidates_evaluated']}")
        print(
            f"Best            : {bi['id']} (fitness={bi['fitness']}, "
            f"species={bi.get('species', 'n/a')})"
        )
        if args.diff and diff_text:
            print(diff_text)
        elif bi.get("code") and getattr(args, "verbose", False):
            print("Best code:")
            print(bi["code"])
        if args.diff_file and diff_text:
            Path(args.diff_file).write_text(diff_text, encoding="utf-8")

        base_fit = float(baseline_res.score) if baseline_res else None
        base_fails = (baseline_res.artifacts or {}).get("failures", []) if baseline_res else []
        if getattr(args, "diagnose", False) or getattr(args, "verbose", False):
            print(format_terminal_diagnostics(result, scenario, base_fit, base_fails))
            print(f"Saved to        : {out_path}")
            report = parse_report(out_path)
            print("\n".join(summarize(report)))
        else:
            print(f"Saved to        : {out_path} (use --diagnose for detailed fault analysis & report summary)")

    if "engine" in locals() and engine is not None:
        fn = getattr(engine, "fitness_fn", None)
        if hasattr(fn, "close"):
            try:
                fn.close()
            except Exception:
                pass

    report = parse_report(out_path)
    if not report.is_valid:
        return 2
    return 0 if _hit(result, args.target) else 1


def cmd_serve_workbench(args) -> int:
    import http.server
    import socketserver
    import webbrowser

    target_file = Path(args.file)
    if not target_file.is_file():
        print(f"error: workbench file not found: {target_file}")
        return 2

    port = args.port
    host = args.host
    serve_dir = target_file.parent.resolve()
    rel_name = target_file.name

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(serve_dir), **kw)

    url = f"http://{host}:{port}/{rel_name}"
    print(f"[Silicon Workbench Server] Serving on {url}")
    print(f"[WebUSB Secure Context] WebUSB API is enabled on http://localhost:{port}")
    if not getattr(args, "no_browser", False):
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        with socketserver.TCPServer((host, port), Handler) as httpd:
            print("[Silicon Workbench Server] Press Ctrl+C to terminate server.")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Silicon Workbench Server] Stopped.")
    return 0


def _ensure_evolve_defaults(args) -> None:
    """Ensure all attributes accessed by cmd_evolve are present even when called from subcommands."""
    defaults = {
        "engine": "auto",
        "genome": "code",
        "scenario": "click_cli_parser",
        "scenario_file": None,
        "source": [],
        "tests": None,
        "pytest": None,
        "func": None,
        "target_file": None,
        "generations": 30,
        "population": 16,
        "target": 99.7,
        "seed": 42,
        "patience": 15,
        "mode": "dynamic",
        "frac": 0.667,
        "max_evals": None,
        "sandbox": False,
        "no_sandbox": False,
        "diff": False,
        "diff_file": None,
        "apply": False,
        "format": "console",
        "patch_file": None,
        "summary_file": None,
        "output": "run_report.json",
        "llm": None,
        "llm_model": None,
        "quiet": False,
        "diagnose": False,
        "verbose": False,
        "telemetry_stream": None,
        "telemetry_fifo": None,
        "external_driver": None,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)


def cmd_repair(args) -> int:
    """Streamlined subcommand for Automated Program Repair (APR)."""
    args.genome = "code"
    args.engine = "greedy"
    if not hasattr(args, "diff") or args.diff is None:
        args.diff = True
    _ensure_evolve_defaults(args)
    return cmd_evolve(args)


def cmd_optimize(args) -> int:
    """Streamlined subcommand for Numerical and Algorithmic Optimization."""
    args.genome = "numeric"
    args.engine = "ga"
    _ensure_evolve_defaults(args)
    return cmd_evolve(args)


def cmd_audit(args) -> int:
    """Run autonomous self-audit and governance evaluation."""
    from .supervisor import SelfAuditSupervisor
    supervisor = SelfAuditSupervisor(
        candidate_mode=getattr(args, "candidate", "directed"),
        full_suite=getattr(args, "full", False),
        output_path=getattr(args, "output", "reports/self_audit.json"),
        quiet=getattr(args, "quiet", False),
    )
    report = supervisor.run_audit()
    return 0 if report.get("overall_verdict") == "PASS" else 1


def cmd_wizard(args) -> int:
    """Interactive onboarding wizard."""
    from .ui.wizard import run_wizard
    return run_wizard()


def cmd_init(args) -> int:
    """Initialize starter configuration file in current project."""
    from .config import generate_default_config
    fmt = getattr(args, "format", "toml")
    target = Path("evolab.toml" if fmt == "toml" else ".evolab.json")
    if target.exists() and not getattr(args, "force", False):
        print(f"error: {target} already exists. Use --force to overwrite.", file=sys.stderr)
        return 2
    template = generate_default_config(fmt=fmt)
    target.write_text(template, encoding="utf-8")
    print(f"Initialized configuration file: {target}", file=sys.stderr)
    return 0


def cmd_eval(args) -> int:
    """Standalone Fitness Oracle evaluating candidate representation from CLI argument or stdin."""
    import sys
    raw_input = getattr(args, "candidate", None)
    if raw_input is None or raw_input == "-":
        if not sys.stdin.isatty():
            raw_input = sys.stdin.read().strip()
        else:
            print("error: eval requires a candidate vector/JSON argument or '-' via stdin", file=sys.stderr)
            print("hint: echo '[0.0, 0.0]' | evolab eval", file=sys.stderr)
            return 2

    if not raw_input:
        print("error: empty candidate provided to eval", file=sys.stderr)
        return 2

    vector = None
    try:
        parsed = json.loads(raw_input)
        if isinstance(parsed, list):
            vector = [float(x) for x in parsed]
        elif isinstance(parsed, dict) and "values" in parsed:
            vector = [float(x) for x in parsed["values"]]
        elif isinstance(parsed, dict) and "genes" in parsed:
            vector = [float(x) for x in parsed["genes"]]
    except Exception:
        pass

    if vector is None:
        try:
            cleaned = raw_input.replace(",", " ").split()
            vector = [float(x) for x in cleaned]
        except Exception:
            pass

    if vector is None:
        print(f"error: could not parse candidate vector coordinates from: {raw_input[:60]!r}", file=sys.stderr)
        return 2

    landscape = getattr(args, "landscape", "rastrigin")
    target = getattr(args, "target", 100.0)
    from .vectorized import VectorizedLandscapeEvaluator
    try:
        evaluator = VectorizedLandscapeEvaluator(landscape=landscape, target_score=target)
        result = evaluator.evaluate(vector)
    except Exception as exc:
        print(f"error during evaluation: {exc}", file=sys.stderr)
        return 2

    out_fmt = getattr(args, "format", "score")
    if out_fmt == "json":
        print(json.dumps({
            "score": result.score,
            "sub_scores": result.sub_scores,
            "artifacts": result.artifacts,
            "evaluation_time_ms": result.evaluation_time_ms,
        }, indent=2))
    else:
        print(f"{result.score:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    ap = argparse.ArgumentParser(
        prog="evolab", description="Universal Evolutionary Optimization & Synthesis Kernel"
    )
    ap.add_argument("--version", action="version", version=f"evolab {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    # Subcommand: eval (Fitness Oracle)
    p_eval = sub.add_parser("eval", help="evaluate candidate representation or vector as a Fitness Oracle")
    p_eval.add_argument("candidate", nargs="?", default=None, help="candidate coordinates e.g. '[0.0, 0.0]' or '-' for stdin")
    p_eval.add_argument("--landscape", choices=["rastrigin", "sphere", "rosenbrock", "ackley", "griewank"],
                        default="rastrigin", help="mathematical benchmark landscape (default: rastrigin)")
    p_eval.add_argument("-t", "--target", type=float, default=100.0, help="target score (default: 100.0)")
    p_eval.add_argument("--format", choices=["score", "json"], default="score", help="output format (score or json)")
    p_eval.set_defaults(func=cmd_eval)

    # Subcommand: inspect
    p_inspect = sub.add_parser("inspect", help="validate and analyze a report file")
    p_inspect.add_argument("file", nargs="?", default=None, help="report JSON path")
    p_inspect.set_defaults(func=cmd_inspect)

    # Subcommand: serve-workbench
    p_serve = sub.add_parser("serve-workbench", help="launch local HTTP server for interactive Silicon Workbench with WebUSB enabled")
    p_serve.add_argument("file", help="path to HTML workbench file")
    p_serve.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    p_serve.add_argument("--no-browser", action="store_true", help="do not auto-open browser")
    p_serve.set_defaults(func=cmd_serve_workbench)

    # Subcommand: wizard
    p_wiz = sub.add_parser("wizard", help="launch interactive step-by-step onboarding wizard")
    p_wiz.set_defaults(func=cmd_wizard)

    # Subcommand: init
    p_init = sub.add_parser("init", help="initialize starter evolab.toml or .evolab.json in current directory")
    p_init.add_argument("--format", choices=["toml", "json"], default="toml", help="configuration file format (default: toml)")
    p_init.add_argument("--force", action="store_true", help="overwrite existing configuration file")
    p_init.set_defaults(func=cmd_init)

    # Subcommand: audit
    p_audit = sub.add_parser("audit", help="run autonomous self-audit and governance verification")
    p_audit.add_argument("--full", action="store_true", help="run full 30-seed benchmark suite")
    p_audit.add_argument("--candidate", default="directed", help="candidate mutation strategy mode (default: directed)")
    p_audit.add_argument("-o", "--output", default="reports/self_audit.json", help="audit report JSON path")
    p_audit.add_argument("--quiet", action="store_true", help="suppress interactive terminal scorecard")
    p_audit.set_defaults(func=cmd_audit)

    # Subcommand: repair (focused APR)
    p_rep = sub.add_parser("repair", help="automated program repair for Python source code")
    p_rep.add_argument("--source", action="append", default=[], help="source file to repair (repeatable)")
    p_rep.add_argument("--pytest", default=None, help="path to pytest file containing assertions")
    p_rep.add_argument("--tests", default=None, help="JSON test cases")
    p_rep.add_argument("--func", default=None, help="target function name")
    p_rep.add_argument("--target-file", default=None, help="target file within multi-file project")
    p_rep.add_argument("--scenario", default="click_cli_parser", help="built-in code scenario (default: click_cli_parser)")
    p_rep.add_argument("--scenario-file", default=None, help="JSON CodeScenario file")
    p_rep.add_argument("--max-evals", type=int, default=None, help="search evaluation budget")
    p_rep.add_argument("-t", "--target", type=float, default=99.7, help="early-stop fitness target")
    p_rep.add_argument("--diff", action="store_true", default=True, help="print unified diff (default: true)")
    p_rep.add_argument("--no-diff", dest="diff", action="store_false", help="do not print diff")
    p_rep.add_argument("--diff-file", default=None, help="write unified diff to file")
    p_rep.add_argument("--apply", action="store_true", help="apply repair in-place (creates .bak backup)")
    p_rep.add_argument("--patch-file", default=None, help="save git-apply patch to file")
    p_rep.add_argument("--format", choices=["console", "markdown", "patch", "json"], default="console")
    p_rep.add_argument("-o", "--output", default="run_report.json")
    p_rep.add_argument("--diagnose", action="store_true", help="display detailed failure diagnostics & report summary")
    p_rep.add_argument("-v", "--verbose", action="store_true", help="enable verbose diagnostic output")
    p_rep.add_argument("--sandbox", action="store_true")
    p_rep.add_argument("--no-sandbox", action="store_true")
    p_rep.add_argument("--llm", choices=["groq", "gemini", "openai", "mock"], default=None)
    p_rep.add_argument("--llm-model", default=None)
    p_rep.add_argument("--quiet", action="store_true", help="suppress live progress updates")
    p_rep.add_argument("--telemetry-stream", default=None, help="stream live generation/step telemetry JSONL to file")
    p_rep.add_argument("--telemetry-fifo", default=None, help="stream live generation/step telemetry JSONL to named pipe/FIFO")
    p_rep.add_argument("-s", "--seed", type=int, default=42, help="random seed (default: 42)")
    p_rep.add_argument("--external-driver", default=None, help="executable path for external JSON-RPC 2.0 evaluation driver")
    p_rep.set_defaults(func=cmd_repair)

    # Subcommand: optimize (focused numeric optimization)
    p_opt = sub.add_parser("optimize", help="numerical and algorithm optimization")
    p_opt.add_argument("-g", "--generations", type=int, default=30, help="number of generations (default: 30)")
    p_opt.add_argument("-p", "--population", type=int, default=16, help="population size (default: 16)")
    p_opt.add_argument("-t", "--target", type=float, default=99.7, help="early-stop fitness target")
    p_opt.add_argument("-s", "--seed", type=int, default=42, help="random seed (default: 42)")
    p_opt.add_argument("-k", "--patience", type=int, default=15, help="stagnation patience")
    p_opt.add_argument("--mode", choices=["off", "static", "dynamic"], default="dynamic", help="fitness sharing mode")
    p_opt.add_argument("--frac", type=float, default=0.667, help="exploitation start fraction")
    p_opt.add_argument("-o", "--output", default="run_report.json")
    p_opt.add_argument("--format", choices=["console", "markdown", "patch", "json"], default="console")
    p_opt.add_argument("--checkpoint-every", type=int, default=None, help="save state checkpoint every N generations")
    p_opt.add_argument("--checkpoint-dir", default=None, help="directory to store checkpoints (default: checkpoints/)")
    p_opt.add_argument("--resume", default=None, help="resume execution from a checkpoint JSON file")
    p_opt.add_argument("--diagnose", action="store_true", help="display detailed fitness sharing diagnostics & report summary")
    p_opt.add_argument("-v", "--verbose", action="store_true", help="enable verbose diagnostic output")
    p_opt.add_argument("--quiet", action="store_true", help="suppress live progress updates")
    p_opt.add_argument("--telemetry-stream", default=None, help="stream live generation telemetry JSONL to file")
    p_opt.add_argument("--telemetry-fifo", default=None, help="stream live generation telemetry JSONL to named pipe/FIFO")
    p_opt.add_argument("--external-driver", default=None, help="executable path for external JSON-RPC 2.0 evaluation driver")
    p_opt.set_defaults(func=cmd_optimize)

    # Subcommand: evolve (original full command with all 35+ flags organized into structured groups)
    p_evo = sub.add_parser("evolve", help="run a full evolution experiment with complete parameter control")

    g_input = p_evo.add_argument_group("Target Specification & Scenario Inputs")
    g_input.add_argument("--scenario", default="click_cli_parser", help="built-in benchmark scenario")
    g_input.add_argument("--scenario-file", default=None, help="JSON CodeScenario")
    g_input.add_argument("--source", action="append", default=[], help="source file (repeatable)")
    g_input.add_argument("--pytest", default=None, help="path to pytest file containing assertions")
    g_input.add_argument("--tests", default=None, help="JSON test cases")
    g_input.add_argument("--func", default=None, help="target function name")
    g_input.add_argument("--target-file", default=None)
    g_input.add_argument("--swe-bench", default=None, help="path to official SWE-bench Lite instance JSON file")

    g_budget = p_evo.add_argument_group("Evolution Budget & Hyperparameters")
    g_budget.add_argument("--engine", choices=["auto", "greedy", "ga", "nsga2"], default="auto",
                          help="search engine: auto, greedy (code), ga, nsga2 (Pareto)")
    g_budget.add_argument("--genome", choices=["code", "numeric", "electronics"], default="code")
    g_budget.add_argument("-g", "--generations", type=int, default=30)
    g_budget.add_argument("-p", "--population", type=int, default=16)
    g_budget.add_argument("-t", "--target", type=float, default=99.7,
                          help="early-stop / success fitness target")
    g_budget.add_argument("-s", "--seed", type=int, default=42, help="random seed (default: 42)")
    g_budget.add_argument("-k", "--patience", type=int, default=15)
    g_budget.add_argument("--max-evals", type=int, default=None, help="greedy evaluation budget")
    g_budget.add_argument("--mode", choices=["off", "static", "dynamic"],
                          default="dynamic", help="GA fitness-sharing schedule")
    g_budget.add_argument("--frac", type=float, default=0.667)
    g_budget.add_argument("--checkpoint-every", type=int, default=None, help="periodically save state checkpoint every N generations")
    g_budget.add_argument("--checkpoint-dir", default=None, help="directory to store state checkpoints (default: checkpoints/)")
    g_budget.add_argument("--resume", default=None, help="resume execution from a checkpoint JSON file")
    g_budget.add_argument("--telemetry-stream", default=None, help="stream live generation telemetry JSONL to file")
    g_budget.add_argument("--telemetry-fifo", default=None, help="stream live generation telemetry JSONL to named pipe/FIFO")
    g_budget.add_argument("--external-driver", default=None, help="executable path for external JSON-RPC 2.0 evaluation driver")

    g_diag = p_evo.add_argument_group("Diagnostics, Output & Reporting")
    g_diag.add_argument("--diff", action="store_true", help="print unified diff")
    g_diag.add_argument("--diff-file", default=None, help="write unified diff")
    g_diag.add_argument("--diagnose", action="store_true", help="display detailed failure diagnostics & report summary")
    g_diag.add_argument("-v", "--verbose", action="store_true", help="enable verbose diagnostic output")
    g_diag.add_argument("--quiet", action="store_true", help="suppress live terminal progress")
    g_diag.add_argument("--format", choices=["console", "markdown", "patch", "json"], default="console",
                        help="primary stdout format")
    g_diag.add_argument("-o", "--output", default="run_report.json")
    g_diag.add_argument("--apply", action="store_true", help="apply successful repair in-place to source file (creates .bak)")
    g_diag.add_argument("--patch-file", "--patch-out", default=None, help="write git-apply compatible patch to file")
    g_diag.add_argument("--summary-file", default=None, help="write GitHub Markdown summary to file")
    g_diag.add_argument("--pareto-export", default=None, help="write non-dominated Pareto front JSON to file")
    g_diag.add_argument("--sandbox", action="store_true")
    g_diag.add_argument("--no-sandbox", action="store_true")
    g_diag.add_argument("--llm", choices=["groq", "gemini", "openai", "mock"], default=None,
                        help="LLM provider for stagnation-breaking mutation")
    g_diag.add_argument("--llm-model", default=None, help="LLM model name")

    g_silicon = p_evo.add_argument_group("Silicon & Domain Synthesis (Hardware Track)")
    g_silicon.add_argument("--netlist", default=None, help="path to custom SPICE netlist file (.cir)")
    g_silicon.add_argument("--spec", default=None, help="path to custom circuit specification (.json)")
    g_silicon.add_argument("--expr", default=None, help="Boolean logic equation string to synthesize")
    g_silicon.add_argument("--verilog-in", default=None, help="path to synthesizable Verilog RTL module file (.v)")
    g_silicon.add_argument("--verilog-file", default=None, help="write synthesized digital circuit Verilog netlist to file")
    g_silicon.add_argument("--waveform", default=None, help="path to target oscilloscope waveform CSV file")
    g_silicon.add_argument("--fpga-target", choices=["ice40_hx1k", "ice40_up5k", "ecp5_25k", "artix7_35t"],
                           default="ice40_hx1k", help="target FPGA board preset")
    g_silicon.add_argument("--objective", choices=["power", "speed", "area", "balanced"], default="balanced",
                           help="multi-objective optimization priority for circuit synthesis")
    g_silicon.add_argument("--schematic-file", default=None, help="write synthesized circuit SVG schematic to file")
    g_silicon.add_argument("--ui-file", default=None, help="write interactive HTML5 Silicon Workbench dashboard to file")

    p_evo.set_defaults(func=cmd_evolve)
    return ap


def _run_cli(argv: list[str] | None = None) -> int:
    import sys
    # If invoked with no arguments in an interactive terminal, offer to launch onboarding wizard
    if argv is None and len(sys.argv) <= 1:
        if sys.stdin.isatty() and sys.stdout.isatty():
            from .ui.wizard import run_wizard
            return run_wizard()

    ap = build_parser()

    # Discover and apply cascading hierarchical configuration defaults (defaults < user < project < env < cli)
    try:
        from .config import load_hierarchical_config
        cfg = load_hierarchical_config()
        if cfg:
            valid_dest = {action.dest for action in ap._actions}
            if ap._subparsers and hasattr(ap._subparsers, "_actions"):
                for sub_act in ap._subparsers._actions:
                    if hasattr(sub_act, "choices") and isinstance(sub_act.choices, dict):
                        for subp in sub_act.choices.values():
                            valid_dest.update({a.dest for a in subp._actions})
            valid_cfg = {k: v for k, v in cfg.items() if k in valid_dest}
            if valid_cfg:
                ap.set_defaults(**valid_cfg)
    except Exception:
        pass

    args = ap.parse_args(argv)
    _ensure_evolve_defaults(args)
    if getattr(args, "command", None) == "inspect" and getattr(args, "file", None) is None:
        if not sys.stdin.isatty():
            args.file = "-"
        else:
            print("error: report file not found: <missing path>", file=sys.stderr)
            print("hint: pass a report path or '-' for stdin, e.g. evolab inspect run_report.json", file=sys.stderr)
            return 2
    cmd = getattr(args, "command", None)
    if cmd == "inspect":
        return cmd_inspect(args)
    if cmd == "eval":
        return cmd_eval(args)
    if cmd == "serve-workbench":
        return cmd_serve_workbench(args)
    if cmd == "evolve":
        return cmd_evolve(args)
    if cmd == "repair":
        return cmd_repair(args)
    if cmd == "optimize":
        return cmd_optimize(args)
    if cmd == "audit":
        return cmd_audit(args)
    if cmd == "wizard":
        return cmd_wizard(args)
    if cmd == "init":
        return cmd_init(args)
    return 2


def main(argv: list[str] | None = None) -> int:
    import os
    import sys
    try:
        return _run_cli(argv)
    except BrokenPipeError:
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except Exception:
            pass
        return 141


if __name__ == "__main__":
    raise SystemExit(main())
