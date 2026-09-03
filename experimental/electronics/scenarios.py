"""Single runnable-scenario registry for the electronics track.

Digital search and analog circuits enter here. Datasheet archives stay
under components/ and are not scenarios.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from evolab.genome import FloatGenome, Individual

from .archive import attach_archive, seed_population_with_elites

ROOT = Path(__file__).resolve().parent
CIRCUITS = ROOT / "circuits"

# name -> circuits/<name>/config.json
CIRCUIT_SCENARIOS = (
    "bjt_ce_amp",
    "chargepump",
    "comparator_simple",
    "ptm180nm_opamp",
    "timer_555_astable",
)


def list_electronics_scenarios() -> list[str]:
    return [
        "half_adder",
        "full_adder",
        "cgp_adder",
        "cgp_alu",
        "cgp_comparator",
        "analog_sizing",
        "analog_filter_synthesis",
        *CIRCUIT_SCENARIOS,
    ]


def _half_adder(population_size: int, rng: random.Random) -> tuple[Any, list[Individual]]:
    from experimental.electronics.evaluators.digital_evaluator import MultiCorner74xxEvaluator
    from experimental.electronics.models.circuit_netlist import (
        CircuitNetlistGenome,
        Connection,
        PinRef,
    )
    from experimental.electronics.models.independent_verifier import IndependentDigitalVerifier

    table = []
    for a in (0, 1):
        for b in (0, 1):
            table.append(((a, b), IndependentDigitalVerifier.half_adder_reference(a, b)))
    needed = ("XOR", "AND")
    evaluator = MultiCorner74xxEvaluator(
        truth_table=table, max_delay_ns=150.0, max_quiescent_ua=60.0, functions_needed=needed
    )
    parts = ["74HC00", "74HC02", "74HC04", "74HC08", "74HC32", "74HC86"]

    def random_genome() -> CircuitNetlistGenome:
        ics = [rng.choice(parts), rng.choice(parts)]
        conns = []
        for _ in range(rng.randint(3, 7)):
            src_ic = rng.choice([-1, 0, 1])
            dst_ic = rng.choice([0, 1, -1])
            src_pin = rng.randint(0, 1) if src_ic == -1 else rng.randint(1, 3)
            dst_pin = 100 + rng.randint(0, 1) if dst_ic == -1 else rng.randint(1, 3)
            conns.append(Connection(PinRef(src_ic, src_pin), PinRef(dst_ic, dst_pin)))
        return CircuitNetlistGenome(ics, conns, 2, 2, functions_needed=needed)

    from experimental.electronics.proposal import seed_for_controller

    seed = seed_for_controller(needed, 2, 2)
    pop = [Individual(genome=seed, species="spec_electronics")]
    while len(pop) < population_size:
        pop.append(Individual(genome=random_genome(), species="spec_electronics"))
    return evaluator, pop


def _full_adder(population_size: int, rng: random.Random) -> tuple[Any, list[Individual]]:
    from experimental.electronics.evaluators.digital_evaluator import MultiCorner74xxEvaluator
    from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
    from experimental.electronics.models.independent_verifier import IndependentDigitalVerifier

    table = [
        ((a, b, c), IndependentDigitalVerifier.full_adder_reference(a, b, c))
        for a in (0, 1) for b in (0, 1) for c in (0, 1)
    ]
    needed = ("XOR", "AND", "OR")
    evaluator = MultiCorner74xxEvaluator(
        truth_table=table, max_delay_ns=180.0, max_quiescent_ua=80.0, functions_needed=needed
    )
    parts = ["74HC00", "74HC08", "74HC32", "74HC86"]

    def random_genome() -> CircuitNetlistGenome:
        ics = [rng.choice(parts) for _ in range(3)]
        conns = []
        for _ in range(rng.randint(4, 8)):
            src_ic = rng.choice([-1, 0, 1, 2])
            dst_ic = rng.choice([0, 1, 2, -1])
            src_pin = rng.randint(0, 2) if src_ic == -1 else rng.randint(1, 3)
            dst_pin = 100 + rng.randint(0, 1) if dst_ic == -1 else rng.randint(1, 3)
            conns.append(Connection(PinRef(src_ic, src_pin), PinRef(dst_ic, dst_pin)))
        return CircuitNetlistGenome(ics, conns, 3, 2, functions_needed=needed)

    from experimental.electronics.proposal import seed_for_controller

    seed = seed_for_controller(needed, 3, 2)
    pop = [Individual(genome=seed, species="spec_electronics")]
    while len(pop) < population_size:
        pop.append(Individual(genome=random_genome(), species="spec_electronics"))
    return evaluator, pop


def _analog_sizing(population_size: int, rng: random.Random) -> tuple[Any, list[Individual]]:
    from experimental.electronics.evaluators.spice_evaluator import AnalogSizingEvaluator

    evaluator = AnalogSizingEvaluator()
    pop = [
        Individual(
            genome=FloatGenome([rng.uniform(0.3, 3.0) for _ in range(4)]),
            species="spec_electronics",
        )
        for _ in range(population_size)
    ]
    return evaluator, pop


def _circuit_config(name: str, population_size: int, rng: random.Random) -> tuple[Any, list[Individual]]:
    from experimental.electronics.evaluators.spice_evaluator import CircuitConfigEvaluator

    cfg = CIRCUITS / name / "config.json"
    evaluator = CircuitConfigEvaluator(cfg)
    base = list(evaluator.defaults)
    pop = []
    for _ in range(population_size):
        genes = []
        for v in base:
            span = abs(v) * 0.25 if v else 0.1
            genes.append(v + rng.uniform(-span, span))
        pop.append(Individual(genome=FloatGenome(genes), species="spec_electronics"))
    return evaluator, pop


def _analog_filter_synthesis(population_size: int, rng: random.Random) -> tuple[Any, list[Individual]]:
    from experimental.electronics.evaluators.analog_topology_evaluator import AnalogFilterTopologyEvaluator
    from experimental.electronics.models.analog_topology import (
        AnalogComponent,
        AnalogComponentKind,
        AnalogTopologyGenome,
    )

    evaluator = AnalogFilterTopologyEvaluator(target_cutoff_hz=1000.0)

    # Initial seed: simple low-pass RC filter (R1 in->out, C1 out->0)
    seed_components = [
        AnalogComponent(kind=AnalogComponentKind.RESISTOR, name="R1", nodes=("in", "out"), value=10000.0),
        AnalogComponent(kind=AnalogComponentKind.CAPACITOR, name="C1", nodes=("out", "0"), value=1.59e-8),
    ]
    seed_genome = AnalogTopologyGenome(seed_components)

    pop = [Individual(genome=seed_genome, species="spec_electronics")]
    while len(pop) < population_size:
        g = seed_genome.clone()
        for _ in range(rng.randint(1, 3)):
            g = g.mutate(rng)
        pop.append(Individual(genome=g, species="spec_electronics"))

    return evaluator, pop


def _cgp_adder(population_size: int, rng: random.Random) -> tuple[Any, list[Individual]]:
    from evolab.cgp_logic import (
        ALUEvaluator,
        FULL_ADDER_TRUTH_TABLE,
        create_random_cgp_genome,
    )
    evaluator = ALUEvaluator(FULL_ADDER_TRUTH_TABLE, target_name="CGP_FullAdder")
    pop = [
        Individual(
            genome=create_random_cgp_genome(num_inputs=3, num_outputs=2, num_nodes=12, rng=rng),
            species="spec_electronics",
        )
        for _ in range(population_size)
    ]
    return evaluator, pop


def _cgp_alu(population_size: int, rng: random.Random) -> tuple[Any, list[Individual]]:
    from evolab.cgp_logic import (
        LowPowerALUEvaluator,
        FULL_ADDER_TRUTH_TABLE,
        create_random_cgp_genome,
    )
    evaluator = LowPowerALUEvaluator(FULL_ADDER_TRUTH_TABLE, target_name="CGP_LowPowerALU")
    pop = [
        Individual(
            genome=create_random_cgp_genome(num_inputs=3, num_outputs=2, num_nodes=15, rng=rng),
            species="spec_electronics",
        )
        for _ in range(population_size)
    ]
    return evaluator, pop


def _cgp_comparator(population_size: int, rng: random.Random) -> tuple[Any, list[Individual]]:
    from evolab.cgp_logic import (
        ALUEvaluator,
        COMPARATOR_TRUTH_TABLE,
        create_random_cgp_genome,
    )
    evaluator = ALUEvaluator(COMPARATOR_TRUTH_TABLE, target_name="CGP_Comparator")
    pop = [
        Individual(
            genome=create_random_cgp_genome(num_inputs=4, num_outputs=3, num_nodes=16, rng=rng),
            species="spec_electronics",
        )
        for _ in range(population_size)
    ]
    return evaluator, pop


def prepare_electronics_run(
    name: str,
    population_size: int,
    seed: int | None,
) -> tuple[Any, list[Individual], str]:
    """Builds (evaluator, initial population) for a scenario.

    The returned evaluator is wired into the cumulative evaluation archive
    (experimental/electronics/data/archive.db): repeated invocations of the
    same scenario + evaluator spec serve prior results from the archive
    instead of re-evaluating identical genomes. ``EVOLAB_ARCHIVE=0`` returns
    the raw evaluator; ``EVOLAB_ARCHIVE_SEED=n`` additionally injects the n
    best archived genomes into trailing population slots (slot 0 stays the
    deterministic proposal seed). ``.evaluate()`` on the returned object is
    always a RAW evaluation with full artifacts — the cache only serves the
    GA's ``__call__`` path.
    """
    rng = random.Random(seed)
    if name == "half_adder":
        ev, pop = _half_adder(population_size, rng)
    elif name == "full_adder":
        ev, pop = _full_adder(population_size, rng)
    elif name == "cgp_adder":
        ev, pop = _cgp_adder(population_size, rng)
    elif name == "cgp_alu":
        ev, pop = _cgp_alu(population_size, rng)
    elif name == "cgp_comparator":
        ev, pop = _cgp_comparator(population_size, rng)
    elif name == "analog_sizing":
        ev, pop = _analog_sizing(population_size, rng)
    elif name == "analog_filter_synthesis":
        ev, pop = _analog_filter_synthesis(population_size, rng)
    elif name in CIRCUIT_SCENARIOS:
        ev, pop = _circuit_config(name, population_size, rng)
    else:
        raise ValueError(f"unknown electronics scenario {name!r}")

    wired, scenario_key, archive_obj = attach_archive(ev, name)
    if archive_obj is not None:
        try:
            n_elites = int(os.environ.get("EVOLAB_ARCHIVE_SEED", "0"))
        except ValueError:
            n_elites = 0
        pop = seed_population_with_elites(archive_obj, scenario_key, pop, n_elites)
    return wired, pop, name


def prepare_custom_electronics_run(
    spec_path: str | Path | None = None,
    netlist_path: str | Path | None = None,
    expr: str | None = None,
    verilog_in: str | Path | None = None,
    waveform_path: str | Path | None = None,
    objective: str | None = None,
    population_size: int = 16,
    seed: int | None = None,
) -> tuple[Any, list[Individual], str]:
    """Builds (evaluator, initial population, scenario_name) from user-supplied custom inputs.

    Supports:
    - Direct Boolean logic equations via `expr` or JSON `"expressions"`.
    - Synthesizable Verilog RTL via `verilog_in` or JSON `"verilog_in"`.
    - Oscilloscope waveform CSV matching via `waveform_path` or JSON `"waveform_csv"`.
    - Analog filter/amplifier specs via JSON `"kind": "filter"|"amplifier"`.
    - Legacy truth_table / design_vars specs.
    """
    import json
    import math
    from .inputs.objective_matrix import ObjectiveMatrix

    rng = random.Random(seed)

    if not any((spec_path, netlist_path, expr, verilog_in, waveform_path)):
        raise ValueError("custom electronics run requires --spec, --netlist, --expr, --verilog-in, or --waveform")

    spec_data: dict[str, Any] = {}
    spec_file = Path(spec_path) if spec_path else None
    if spec_file and spec_file.is_file():
        try:
            spec_data = json.loads(spec_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    obj_matrix = ObjectiveMatrix.from_preset(objective or spec_data.get("objective", "balanced"))

    # Mode 1: Boolean Logic Expressions (CLI --expr or spec file)
    boolean_src = expr or spec_data.get("expressions") or spec_data.get("expr")
    if boolean_src:
        from .inputs.boolean_expr import parse_boolean_spec
        from evolab.cgp_logic import LowPowerALUEvaluator, create_random_cgp_genome

        b_spec = parse_boolean_spec(boolean_src)
        evaluator = LowPowerALUEvaluator(
            truth_table=b_spec.truth_table,
            target_name="SynthesizedBooleanLogic",
            area_weight=obj_matrix.area_weight,
            delay_weight=obj_matrix.delay_weight,
            power_weight=obj_matrix.power_weight,
        )
        pop = [
            Individual(
                genome=create_random_cgp_genome(
                    num_inputs=b_spec.num_inputs,
                    num_outputs=b_spec.num_outputs,
                    num_nodes=max(10, b_spec.num_inputs * 4),
                    rng=rng,
                ),
                species="spec_electronics",
            )
            for _ in range(population_size)
        ]
        scenario_name = "custom_boolean_expr"
        return evaluator, pop, scenario_name

    # Mode 2: Synthesizable Verilog RTL (CLI --verilog-in or spec file)
    v_src = verilog_in or spec_data.get("verilog_in") or spec_data.get("verilog_code")
    if v_src:
        from .inputs.verilog_reader import parse_verilog_spec
        from evolab.cgp_logic import LowPowerALUEvaluator, create_random_cgp_genome

        v_spec = parse_verilog_spec(v_src)
        evaluator = LowPowerALUEvaluator(
            truth_table=v_spec.parse_result.truth_table,
            target_name=v_spec.module_name,
            area_weight=obj_matrix.area_weight,
            delay_weight=obj_matrix.delay_weight,
            power_weight=obj_matrix.power_weight,
        )
        pop = [
            Individual(
                genome=create_random_cgp_genome(
                    num_inputs=v_spec.parse_result.num_inputs,
                    num_outputs=v_spec.parse_result.num_outputs,
                    num_nodes=max(12, v_spec.parse_result.num_inputs * 4),
                    rng=rng,
                ),
                species="spec_electronics",
            )
            for _ in range(population_size)
        ]
        scenario_name = f"verilog_{v_spec.module_name}"
        return evaluator, pop, scenario_name

    # Mode 3: Target Oscilloscope Waveform CSV (CLI --waveform or spec file)
    wf_src = waveform_path or spec_data.get("waveform_csv")
    if wf_src:
        from .inputs.analog_spec import WaveformTraceSpec
        from .models.analog_topology import AnalogComponent, AnalogComponentKind, AnalogTopologyGenome
        from evolab.evaluators import Evaluator, FitnessResult

        wf_spec = WaveformTraceSpec.from_csv(wf_src)

        class WaveformMatchEvaluator(Evaluator):
            @property
            def deterministic(self) -> bool:
                return True

            def evaluate(self, ind: Individual) -> FitnessResult:
                g = ind.genome
                if not isinstance(g, AnalogTopologyGenome):
                    return FitnessResult(score=0.0)
                r_sum = sum(c.value for c in g.components if c.kind == AnalogComponentKind.RESISTOR)
                c_sum = sum(c.value for c in g.components if c.kind == AnalogComponentKind.CAPACITOR)
                if r_sum <= 0 or c_sum <= 0:
                    return FitnessResult(score=10.0, passed_holdout=False, artifacts={"tool_used": "analytical_fallback"})
                tau = r_sum * c_sum
                sim_v = [5.0 * (1.0 - math.exp(-t / max(tau, 1e-12))) for t in wf_spec.times]
                score = wf_spec.evaluate_mse(wf_spec.times, sim_v)
                return FitnessResult(score=score, passed_holdout=False, artifacts={"tool_used": "analytical_fallback", "tau": tau})

        seed_comps = [
            AnalogComponent(kind=AnalogComponentKind.RESISTOR, name="R1", nodes=("in", "out"), value=10000.0),
            AnalogComponent(kind=AnalogComponentKind.CAPACITOR, name="C1", nodes=("out", "0"), value=1e-7),
        ]
        seed_g = AnalogTopologyGenome(seed_comps)
        pop = [Individual(genome=seed_g, species="spec_electronics")]
        while len(pop) < population_size:
            g_clone = seed_g.clone()
            for _ in range(rng.randint(1, 3)):
                g_clone = g_clone.mutate(rng)
            pop.append(Individual(genome=g_clone, species="spec_electronics"))
        return WaveformMatchEvaluator(), pop, "waveform_matching"

    # Mode 4: Analog Filter Specification
    if spec_data and any(k in spec_data for k in ("cutoff_hz", "filter_type")):
        from .inputs.analog_spec import parse_analog_spec
        from .evaluators.analog_topology_evaluator import AnalogFilterTopologyEvaluator
        from .models.analog_topology import AnalogComponent, AnalogComponentKind, AnalogTopologyGenome

        f_spec = parse_analog_spec(spec_data)
        evaluator = AnalogFilterTopologyEvaluator(target_cutoff_hz=getattr(f_spec, "cutoff_hz", 1000.0))
        seed_comps = [
            AnalogComponent(kind=AnalogComponentKind.RESISTOR, name="R1", nodes=("in", "out"), value=10000.0),
            AnalogComponent(kind=AnalogComponentKind.CAPACITOR, name="C1", nodes=("out", "0"), value=1.59e-8),
        ]
        seed_g = AnalogTopologyGenome(seed_comps)
        pop = [Individual(genome=seed_g, species="spec_electronics")]
        while len(pop) < population_size:
            g_clone = seed_g.clone()
            for _ in range(rng.randint(1, 3)):
                g_clone = g_clone.mutate(rng)
            pop.append(Individual(genome=g_clone, species="spec_electronics"))
        return evaluator, pop, "custom_analog_filter"

    # Mode 5: Legacy Digital truth_table
    if "truth_table" in spec_data:
        from experimental.electronics.evaluators.digital_evaluator import MultiCorner74xxEvaluator
        from experimental.electronics.models.circuit_netlist import CircuitNetlistGenome, Connection, PinRef
        from experimental.electronics.proposal import seed_for_controller

        raw_table = spec_data["truth_table"]
        table = [
            (
                tuple(entry[0]),
                tuple(entry[1]) if isinstance(entry[1], (list, tuple)) else (entry[1],),
            )
            for entry in raw_table
        ]
        num_inputs = int(spec_data.get("inputs", len(table[0][0])))
        num_outputs = int(spec_data.get("outputs", len(table[0][1])))
        needed = tuple(spec_data.get("functions_needed", ("XOR", "AND", "OR")))
        max_delay = float(spec_data.get("max_delay_ns", 150.0))
        max_ua = float(spec_data.get("max_quiescent_ua", 60.0))

        evaluator = MultiCorner74xxEvaluator(
            truth_table=table,
            max_delay_ns=max_delay,
            max_quiescent_ua=max_ua,
            functions_needed=needed,
        )

        parts = spec_data.get("parts", ["74HC00", "74HC02", "74HC04", "74HC08", "74HC32", "74HC86"])
        num_ics = int(spec_data.get("num_ics", 2))

        def random_digital_genome() -> CircuitNetlistGenome:
            ics = [rng.choice(parts) for _ in range(num_ics)]
            conns = []
            for _ in range(rng.randint(3, 7)):
                src_ic = rng.choice([-1] + list(range(num_ics)))
                dst_ic = rng.choice(list(range(num_ics)) + [-1])
                src_pin = rng.randint(0, num_inputs - 1) if src_ic == -1 else rng.randint(1, 3)
                dst_pin = 100 + rng.randint(0, num_outputs - 1) if dst_ic == -1 else rng.randint(1, 3)
                conns.append(Connection(PinRef(src_ic, src_pin), PinRef(dst_ic, dst_pin)))
            return CircuitNetlistGenome(ics, conns, num_inputs, num_outputs, functions_needed=needed)

        pop = []
        try:
            seed_g = seed_for_controller(needed, num_inputs, num_outputs)
            pop.append(Individual(genome=seed_g, species="spec_electronics"))
        except Exception:
            pass

        while len(pop) < population_size:
            pop.append(Individual(genome=random_digital_genome(), species="spec_electronics"))

        scenario_name = spec_data.get("name", spec_file.stem if spec_file else "custom_digital")
        return evaluator, pop, scenario_name

    # Mode 6: Legacy Analog design_vars or specs
    from experimental.electronics.evaluators.spice_evaluator import CircuitConfigEvaluator

    if spec_file and ("design_vars" in spec_data or "specs" in spec_data):
        evaluator = CircuitConfigEvaluator(spec_file)
        base = list(evaluator.defaults)
        pop = []
        for _ in range(population_size):
            genes = []
            for v in base:
                span = abs(v) * 0.25 if v else 0.1
                genes.append(v + rng.uniform(-span, span))
            pop.append(Individual(genome=FloatGenome(genes), species="spec_electronics"))
        scenario_name = spec_data.get("name", spec_file.stem)
        return evaluator, pop, scenario_name

    raise ValueError(f"Unrecognized electronics spec format in {spec_path or 'inputs'}. Expected 'truth_table', 'expressions', 'verilog_in', or 'design_vars'.")
