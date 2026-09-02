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
    return ["half_adder", "full_adder", "analog_sizing", *CIRCUIT_SCENARIOS]


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
    elif name == "analog_sizing":
        ev, pop = _analog_sizing(population_size, rng)
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
