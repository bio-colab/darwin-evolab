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
    load_pytest_scenario,
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
        target_func = args.func if isinstance(args.func, str) else None
        if getattr(args, "pytest", None):
            try:
                return load_pytest_scenario(args.source, args.pytest, target_func, args.target_file), True
            except Exception as e:
                print(f"error loading pytest scenario: {e}")
                return None, True
        if not target_func:
            print("error: --source requires --func (or use --pytest for auto-detection)")
            return None, True
        if not args.tests:
            print("error: --source requires either --tests or --pytest")
            return None, True
        return load_source_scenario(args.source, args.tests, target_func, args.target_file), True
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
        print(f"SWE-bench Issue    : {spec.instance_id} ({spec.repo})")
        print(f"Problem Summary    : {spec.problem_statement[:80]}...")
        print(f"Target File        : {spec.target_file}")
        print(f"FAIL_TO_PASS Tests : {len(spec.fail_to_pass_tests)}")
        print(f"PASS_TO_PASS Tests : {len(spec.pass_to_pass_tests)}")
        
        resolution = adapter.solve_instance(spec, max_evals=args.max_evals or 32)
        print(f"\n[SWE-bench Resolution Verdict]")
        print(f"Resolved           : {'YES (100% Green)' if resolution.resolved else 'NO'}")
        print(f"FAIL_TO_PASS       : {'PASS' if resolution.fail_to_pass_passed else 'FAIL'}")
        print(f"PASS_TO_PASS       : {'CLEAN (Zero Regressions)' if resolution.pass_to_pass_clean else 'BROKEN'}")
        print(f"Evaluations Used   : {resolution.evaluations_used}")
        print(f"Execution Time     : {resolution.execution_time_seconds}s")
        
        if getattr(args, "patch_file", None):
            Path(args.patch_file).write_text(resolution.generated_patch, encoding="utf-8")
            print(f"Git Patch saved    : {args.patch_file}")

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
        import sys
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
            print(f"error: electronics track not available ({exc})")
            return 2

        has_custom_input = any((
            getattr(args, "spec", None),
            getattr(args, "netlist", None),
            getattr(args, "expr", None),
            getattr(args, "verilog_in", None),
            getattr(args, "waveform", None),
        ))
        if has_custom_input:
            evaluator, pop, name = prepare_custom_electronics_run(
                spec_path=getattr(args, "spec", None),
                netlist_path=getattr(args, "netlist", None),
                expr=getattr(args, "expr", None),
                verilog_in=getattr(args, "verilog_in", None),
                waveform_path=getattr(args, "waveform", None),
                objective=getattr(args, "objective", None),
                population_size=args.population,
                seed=args.seed,
            )
        else:
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
            print(f"Engine: NSGA-II | Multi-Objective Pareto Frontier Discovered")
            print(f"Pareto Front Size: {len(nsga_res['front_0'])} non-dominated solutions")
            for i, sol in enumerate(nsga_res['front_0'][:5]):
                print(f"  Pareto #{i+1}: {sol['scores']}")
            if getattr(args, "pareto_export", None):
                engine.export_pareto_front(args.pareto_export)
                print(f"Pareto Front saved: {args.pareto_export}")
            return 0

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
        baseline_res = None
        try:
            from .repair import RepairGenome
            base_genome = RepairGenome(sources=dict(scenario.sources), target_file=scenario.target_file, edits=[])
            baseline_res = evaluator.evaluate(base_genome)
        except Exception:
            pass

        result = greedy_run_report(
            scenario.sources,
            scenario.target_file,
            evaluator,
            scenario_name=scenario.name,
            max_evals=args.max_evals,
        )

    if getattr(args, "llm", None) and not _hit(result, args.target):
        bi = result.get("best_individual") or {}
        current_fit = float(bi.get("fitness", 0.0))
        print(
            f"\n[Hybrid LLM] Evolutionary search stagnated at {current_fit:.2f}%. "
            f"Invoking {args.llm} stagnation breaker..."
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
                                f"[Hybrid LLM] Circuit stagnation broken! Fitness improved from {current_fit:.2f}% to {fit_res.score:.2f}%."
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
                                f"[Hybrid LLM] Circuit candidate rejected: score={fit_res.score:.2f}%. Safety preserved."
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
                            f"[Hybrid LLM] Stagnation broken! Fitness improved from {current_fit:.2f}% to {fit_res.score:.2f}%."
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
                            f"[Hybrid LLM] Candidate rejected: score={fit_res.score:.2f}%, holdout={fit_res.passed_holdout}. Safety preserved."
                        )
        except Exception as err:
            print(f"[Hybrid LLM] Stagnation breaker error: {err}")

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
        print(f"Patch saved     : {args.patch_file}")

    if getattr(args, "summary_file", None):
        summary_str = format_markdown_summary(result, scenario, diff_text)
        Path(args.summary_file).write_text(summary_str, encoding="utf-8")
        print(f"Markdown Summary: {args.summary_file}")

    if getattr(args, "schematic_file", None) and is_electronics and "engine" in locals() and engine is not None:
        best_g = getattr(engine, "best_ever", None)
        if best_g and (hasattr(best_g.genome, "circuit") or hasattr(best_g.genome, "connections") or hasattr(best_g.genome, "get_active_nodes")):
            from experimental.electronics.instruments.schematic import save_circuit_svg
            save_circuit_svg(best_g.genome, args.schematic_file)
            print(f"Schematic saved : {args.schematic_file}")

    if getattr(args, "verilog_file", None) and is_electronics and "engine" in locals() and engine is not None:
        best_g = getattr(engine, "best_ever", None)
        if best_g and hasattr(best_g.genome, "to_verilog"):
            v_code = best_g.genome.to_verilog(module_name="synthesized_circuit")
            Path(args.verilog_file).write_text(v_code, encoding="utf-8")
            print(f"Verilog saved   : {args.verilog_file}")

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
            print(f"Workbench UI saved: {args.ui_file}")

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
        if applied:
            print(f"[In-Place Apply] Successfully patched: {', '.join(applied)} (backup saved with .bak)")

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
        if bi.get("code"):
            print("Best code:")
            print(bi["code"])
        if args.diff and diff_text:
            print(diff_text)
        if args.diff_file and diff_text:
            Path(args.diff_file).write_text(diff_text, encoding="utf-8")

        base_fit = float(baseline_res.score) if baseline_res else None
        base_fails = (baseline_res.artifacts or {}).get("failures", []) if baseline_res else []
        print(format_terminal_diagnostics(result, scenario, base_fit, base_fails))
        print(f"Saved to        : {out_path}")
        report = parse_report(out_path)
        print("\n".join(summarize(report)))

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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="evolab", description="Evolutionary experimentation lab"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="validate and analyze a report file")
    p_inspect.add_argument("file", nargs="?", default=None, help="report JSON path")
    p_inspect.set_defaults(func=cmd_inspect)

    p_serve = sub.add_parser("serve-workbench", help="launch local HTTP server for interactive Silicon Workbench with WebUSB enabled")
    p_serve.add_argument("file", help="path to HTML workbench file")
    p_serve.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    p_serve.add_argument("--no-browser", action="store_true", help="do not auto-open browser")
    p_serve.set_defaults(func=cmd_serve_workbench)

    p_evo = sub.add_parser("evolve", help="run a new evolution experiment")
    p_evo.add_argument("--engine", choices=["auto", "greedy", "ga", "nsga2"], default="auto",
                       help="search engine: auto, greedy (code), ga, nsga2 (Pareto)")
    p_evo.add_argument("--swe-bench", default=None, help="path to official SWE-bench Lite instance JSON file")
    p_evo.add_argument("--pareto-export", default=None, help="write non-dominated Pareto front JSON to file")
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
    p_evo.add_argument("--pytest", default=None, help="path to pytest file containing assertions")
    p_evo.add_argument("--llm", choices=["groq", "gemini", "openai", "mock"], default=None,
                       help="LLM provider for stagnation-breaking mutation")
    p_evo.add_argument("--llm-model", default=None, help="LLM model name")
    p_evo.add_argument("--func", default=None, help="target function name")
    p_evo.add_argument("--target-file", default=None)
    p_evo.add_argument("--sandbox", action="store_true")
    p_evo.add_argument("--no-sandbox", action="store_true")
    p_evo.add_argument("--diff", action="store_true", help="print unified diff")
    p_evo.add_argument("--diff-file", default=None, help="write unified diff")
    p_evo.add_argument("--netlist", default=None, help="path to custom SPICE netlist file (.cir)")
    p_evo.add_argument("--spec", default=None, help="path to custom circuit specification (.json)")
    p_evo.add_argument("--expr", default=None, help="Boolean logic equation string to synthesize (e.g. 'S = A ^ B; C = A & B')")
    p_evo.add_argument("--verilog-in", default=None, help="path to synthesizable Verilog RTL module file (.v)")
    p_evo.add_argument("--waveform", default=None, help="path to target oscilloscope waveform CSV file (time, voltage)")
    p_evo.add_argument("--fpga-target", choices=["ice40_hx1k", "ice40_up5k", "ecp5_25k", "artix7_35t"],
                       default="ice40_hx1k", help="target FPGA board preset for synthesis and resource estimation")
    p_evo.add_argument("--objective", choices=["power", "speed", "area", "balanced"], default="balanced",
                       help="multi-objective optimization priority for circuit synthesis")
    p_evo.add_argument("--format", choices=["console", "markdown", "patch", "json"], default="console",
                       help="primary stdout format")
    p_evo.add_argument("--patch-file", "--patch-out", default=None, help="write git-apply compatible patch to file")
    p_evo.add_argument("--summary-file", default=None, help="write GitHub Markdown summary to file")
    p_evo.add_argument("--schematic-file", default=None, help="write synthesized circuit SVG schematic to file")
    p_evo.add_argument("--verilog-file", default=None, help="write synthesized digital circuit Verilog netlist to file")
    p_evo.add_argument("--ui-file", default=None, help="write interactive HTML5 Silicon Workbench dashboard to file")
    p_evo.add_argument("--apply", action="store_true", help="apply successful repair in-place to source file (creates .bak)")
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
    if getattr(args, "command", None) == "inspect":
        return cmd_inspect(args)
    if getattr(args, "command", None) == "serve-workbench":
        return cmd_serve_workbench(args)
    if getattr(args, "command", None) == "evolve":
        return cmd_evolve(args)
    if hasattr(args, "func") and callable(args.func):
        return args.func(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
