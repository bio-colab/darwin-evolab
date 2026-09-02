#!/usr/bin/env python3
"""M8 post-run DIAGNOSTIC (descriptive only — no decision role).

The registered M8 verdict (no detectable effect in any arm) is already
drawn and stays untouched by this script. The question here is the same
integrity question the M7 diagnostic answered for the sequence prior,
and that the v1 review answered for the kind prior ("the prior was
actually active — the measurement is real, not a dormant mechanism"):

    did the avoidance veto actually FIRE inside the student runs, or was
    the mechanism starved — dead doors mined but never drawn?

The veto can only change a population when a DRAWN initial edit lands on
a mined dead door. With 3-7 dead doors out of a catalog of dozens, the
per-draw hit rate is small, so "no detectable effect" could in principle
mean "the mechanism barely engaged". This script measures engagement
directly:

  1. Rebuilds the exact teacher snapshots (3 runs, seeds 9901-9903,
     prior disabled — identical to the A/B).
  2. Mines avoidance_set(min_failures=f) for f in {2, 3}.
  3. Rebuilds the 30 student initial populations per arm (same seeds as
     the A/B) and counts, per scenario:
       - firing_seeds: seeds where the avoid population differs from the
         control population (any divergence == at least one veto fired;
         with no veto the RNG stream is identical, so populations are
         byte-identical);
       - control_dead_doors: dead-door genotypes DRAWN AND KEPT in the
         control populations (what avoidance wanted to remove);
       - avoid_dead_doors: dead-door genotypes still present in the
         avoid populations (bounded vetoes accept the last draw, and
         re-draws can land on other dead doors).

Descriptive numbers only: the registered pooled verdict is a function of
the committed report, not of this script.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab import EngineConfig, EvolutionEngine  # noqa: E402
from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population  # noqa: E402
from evolab.experience import (  # noqa: E402
    ExperienceStore,
    attach_experience_recorder,
    problem_fingerprint,
)
from evolab.repair import catalog_sources  # noqa: E402

TEACHER_SEED_START = 9901
TEACHERS = 3
STUDENT_SEEDS = list(range(1, 31))
POP = 12
GENS = 8


def clean_env() -> None:
    os.environ.pop("EVOLAB_EXPERIENCE", None)
    os.environ.pop("EVOLAB_EXPERIENCE_DB", None)


def build_teacher(scenario, workdir: Path) -> Path:
    name = scenario.name
    db = workdir / f"teacher_{name}.db"
    for i in range(TEACHERS):
        ev = scenario.create_evaluator()
        wired = attach_experience_recorder(
            ev, scenario.sources, scenario.target_file, scenario.func_name,
            db_path=db, run_id=f"teacher_{name}_t{i}",
            prior_enabled=False,
        )
        engine = EvolutionEngine(
            fitness_fn=wired,
            config=EngineConfig(
                generations=GENS, population_size=POP,
                elite_count=1, genome_size=2,
                seed=TEACHER_SEED_START + i,
            ),
        )
        initial = make_code_population(scenario, POP, random.Random(TEACHER_SEED_START + i))
        engine.run(generations=GENS, initial_population=initial)
    return db


def edit_key(edit):
    return (edit.file, edit.lineno, edit.col_offset, edit.kind)


def count_doors(population, avoid):
    return sum(
        1
        for ind in population
        for e in ind.genome.edits
        if edit_key(e) in avoid
    )


def main() -> int:
    clean_env()
    with tempfile.TemporaryDirectory(prefix="evolab_m8diag_") as td:
        for name in SCENARIO_REGISTRY:
            scenario = SCENARIO_REGISTRY[name]()
            fp = problem_fingerprint(
                scenario.sources, scenario.target_file, scenario.func_name
            )
            catalog = catalog_sources(scenario.sources)
            db = build_teacher(scenario, Path(td))
            store = ExperienceStore(db)
            rows = []
            for f in (2, 3):
                avoid = store.avoidance_set(fp, min_failures=f)
                if avoid is None:
                    rows.append({"min_failures": f, "avoid_set": None})
                    continue
                firing = 0
                ctrl_doors = []
                avd_doors = []
                for s in STUDENT_SEEDS:
                    ctrl = make_code_population(scenario, POP, random.Random(s))
                    avd = make_code_population(
                        scenario, POP, random.Random(s), avoid_loci=avoid
                    )
                    diverged = [i.genome.fingerprint() for i in ctrl] != [
                        i.genome.fingerprint() for i in avd
                    ]
                    firing += 1 if diverged else 0
                    ctrl_doors.append(count_doors(ctrl, avoid))
                    avd_doors.append(count_doors(avd, avoid))
                rows.append({
                    "min_failures": f,
                    "avoid_set_size": len(avoid),
                    "firing_seeds": firing,
                    "of_seeds": len(STUDENT_SEEDS),
                    "mean_dead_doors_in_control_pop": round(
                        statistics.mean(ctrl_doors), 2
                    ),
                    "mean_dead_doors_in_avoid_pop": round(
                        statistics.mean(avd_doors), 2
                    ),
                    "seeds_with_zero_doors_in_control": sum(
                        1 for x in ctrl_doors if x == 0
                    ),
                })
            store.close()
            print(f"=== {name} (catalog={len(catalog)} candidates) ===")
            for r in rows:
                print(f"  f={r['min_failures']}: {json.dumps(r)}")

    print("\n(descriptive only — the registered M8 verdict is unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
