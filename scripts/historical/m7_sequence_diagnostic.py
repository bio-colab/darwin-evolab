#!/usr/bin/env python3
"""M7 post-run DIAGNOSTIC (descriptive only — no decision role).

The registered M7 verdict (R4') is already drawn and stays untouched by
this script. The question here is descriptive: did the sequence prior have
any room to act at all? It can only reweight a mutation when the parent's
prefix exists in the frozen teacher snapshot with support >= min_support
(3). Deep prefixes need deep successful/failed sequences — and the A/B
budget (8 generations, population 12) caps how deep student lineages go.

Rebuilds the same teacher snapshots (3 runs per scenario, seeds 9901-9903,
prior disabled) and reports the transition table by prefix depth.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab import EngineConfig, EvolutionEngine  # noqa: E402
from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population  # noqa: E402
from evolab.experience import (  # noqa: E402
    ExperienceStore,
    attach_experience_recorder,
    problem_fingerprint,
)

TEACHER_SEED_START = 9901
TEACHERS = 3
MIN_SUPPORT = 3


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
            config=EngineConfig(generations=8, population_size=12,
                                elite_count=1, genome_size=2, seed=TEACHER_SEED_START + i),
        )
        initial = make_code_population(scenario, 12, __import__("random").Random(TEACHER_SEED_START + i))
        engine.run(generations=8, initial_population=initial)
    return db


def main() -> int:
    clean_env()
    out: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="evolab_m7diag_") as td:
        for name in SCENARIO_REGISTRY:
            scenario = SCENARIO_REGISTRY[name]()
            db = build_teacher(scenario, Path(td))
            store = ExperienceStore(db)
            fp = problem_fingerprint(scenario.sources, scenario.target_file, scenario.func_name)
            rows = store._conn.execute(
                "SELECT edit_kinds FROM experiences WHERE problem_fingerprint = ?",
                (fp,),
            ).fetchall()
            seqs = [json.loads(r[0]) or [] for r in rows]
            seq_stats = store.sequence_stats(fp)
            store.close()

            depth_total = Counter(len(s) for s in seqs)
            by_depth: dict[int, dict[str, int]] = {}
            for key, slot in (seq_stats.get("transitions") or {}).items():
                prefix, kind = key.rsplit(">", 1)
                depth = (len(prefix.split(",")) if prefix else 0)
                d = by_depth.setdefault(depth, {"transitions": 0, "supported": 0, "obs": 0})
                d["transitions"] += 1
                d["obs"] += slot["n"]
                if slot["n"] >= MIN_SUPPORT:
                    d["supported"] += 1
            out[name] = {
                "experiences": len(seqs),
                "sequence_length_histogram": {str(k): v for k, v in sorted(depth_total.items())},
                "transitions_by_prefix_depth": {
                    str(d): v for d, v in sorted(by_depth.items())
                },
            }
            t = by_depth.get(1, {"transitions": 0, "supported": 0})
            print(f"{name:24s} exps={len(seqs):4d}  depth-hist={dict(sorted(depth_total.items()))}")
            for d, v in sorted(by_depth.items()):
                print(f"    depth {d}: {v['supported']:3d}/{v['transitions']:3d} transitions with support>={MIN_SUPPORT}"
                      f"  ({v['obs']} obs)")

    print("\n(descriptive only — the registered R4' verdict is unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
